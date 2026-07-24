"""Sentiment scoring service — primary Gemini (LLM) scorer with local ensemble fallback.

Primary model: Gemini zero-shot (selected as winner in multi-model experiment,
2026-05-28: 95.7% macro-F1 vs all local models on 50-ticket LLM-labeled sample).

Falls back to local ensemble when:
  - SENTIMENT_BACKEND env var is set to "ensemble" or "cardiffnlp"
  - Gemini client is unavailable (e.g. no credentials)

Offline fallback (SENTIMENT_BACKEND=ensemble):
  cardiffnlp/twitter-roberta-base-sentiment-latest + j-hartmann/emotion-english-distilroberta-base
  soft-vote ensemble (Optuna LOO-CV optimized weights, 2026-05-28).

  Experiment results (analysis/ensemble_exhaustive.py, N=50, LOO-CV, 63 combos tested):
    cardiffnlp alone:            69.1%  macro-F1,  221ms/ticket
    cardiffnlp + distilbert:     73.6%  macro-F1,  253ms/ticket  [previous fallback]
    cardiffnlp + emotion:        83.99% macro-F1,  250ms/ticket  [current fallback]
    Gemini zero-shot:            95.7%  macro-F1, 2646ms/ticket
  Ensemble weights: cardiffnlp=0.9051, emotion=0.0949 (Optuna-optimized).
  +14.9pp over previous fallback; CPU-safe; ~1.6GB combined RAM.

Emotion label mapping (j-hartmann 7-class → 3-class):
  NEGATIVE: anger + disgust + fear + sadness
  POSITIVE: joy
  NEUTRAL:  neutral + surprise (residual)

The NEGATIVE label score is returned as ``intensity`` so the M5 router can
decide whether to escalate high-frustration tickets to human review.

Usage:
    from plugins.feedback.sentiment import score_ticket, score_tickets

    result = score_ticket("I've been waiting three days and nothing works!")
    # {'label': 'NEGATIVE', 'score': 0.94, 'intensity': 0.94}

    batch = score_tickets(["Thanks!", "This is broken again."])
"""

import logging
import os
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FINETUNED_MODEL_DIR = _REPO_ROOT / "models" / "sentiment_finetuned"

# ── Backend selection ────────────────────────────────────────────────────────
# "gemini"     → Gemini zero-shot (default, winner model, 95.7% F1)
# "ensemble"   → cardiffnlp + emotion soft-vote (83.99% F1, ~250ms, offline-safe)
# "cardiffnlp" → cardiffnlp alone (69.1% F1, 221ms, lightest)
SENTIMENT_BACKEND = os.getenv("SENTIMENT_BACKEND", "gemini").lower()

SENTIMENT_MODEL = os.getenv(
    "SENTIMENT_MODEL",
    "cardiffnlp/twitter-roberta-base-sentiment-latest",
)

# ── Ensemble weights (Optuna LOO-CV optimized, 2026-05-28, N=50) ─────────────
# cardiffnlp+emotion achieved 83.99% macro-F1 in exhaustive 63-combo search
_ENSEMBLE_W_CARDIFF = 0.9051
_ENSEMBLE_W_EMOTION = 0.0949

_EMPTY_RESULT: dict = {"label": "NEUTRAL", "score": 0.0, "intensity": 0.0}

# ── Gemini zero-shot prompt ─────────────────────────────────────────────────
_GEMINI_PROMPT = """\
You are a sentiment labeler for an IT helpdesk ticket system. Your job is to classify the customer's emotional tone.

Ticket text: {text}

Classify the customer's sentiment as exactly one of: NEGATIVE, NEUTRAL, POSITIVE

NEGATIVE = customer sounds frustrated, angry, urgent, or distressed (e.g. "this is broken", "I've lost a day", "nothing works", "I can't access")
NEUTRAL = normal request with no emotional charge (e.g. "please add me to this group", "need access to X", "how do I configure Y")
POSITIVE = grateful or happy (e.g. "thank you", "this worked great")

Respond with EXACTLY two lines, nothing else:
LABEL: <NEGATIVE|NEUTRAL|POSITIVE>
CONFIDENCE: <0.0-1.0>

Where CONFIDENCE is your certainty in the NEGATIVE classification (0.0 = clearly not distressed, 1.0 = strongly distressed). Use the full range — mild frustration might be 0.4, strong distress 0.9.\
"""

_VALID_LABELS = {"NEGATIVE", "NEUTRAL", "POSITIVE"}

# ── Label → confidence fallback ─────────────────────────────────────────────
# Used only when the two-line structured response cannot be parsed.
_LABEL_CONFIDENCE_FALLBACK = {
    "NEGATIVE": 0.90,
    "NEUTRAL": 0.00,
    "POSITIVE": 0.00,
}

# ── Singletons ──────────────────────────────────────────────────────────────
_pipeline = None
_emotion_pipeline = None
_gemini_client = None
_gemini_model: str | None = None


# ── Gemini backend ───────────────────────────────────────────────────────────

def _get_gemini():
    """Return (client, model) for Gemini, loading on first call."""
    global _gemini_client, _gemini_model
    if _gemini_client is None:
        try:
            from core.genai import get_genai_client
            from config import GEMINI_MODEL

            _gemini_client = get_genai_client()
            _gemini_model = GEMINI_MODEL
            log.info("Sentiment: using Gemini model %s", _gemini_model)
        except Exception as exc:
            log.warning("Sentiment: Gemini unavailable (%s), will fall back to ensemble", exc)
            _gemini_client = None
    return _gemini_client, _gemini_model


def _parse_gemini_response(raw: str) -> tuple[str, float | None]:
    """Parse the two-line structured Gemini response into (label, confidence).

    Returns (label, confidence) where confidence is None if parsing fails.
    Expected format:
        LABEL: NEGATIVE
        CONFIDENCE: 0.75
    """
    label: str = "NEUTRAL"
    confidence: float | None = None

    for line in raw.splitlines():
        line = line.strip().upper()
        if line.startswith("LABEL:"):
            candidate = line.split(":", 1)[1].strip().rstrip(".,;:")
            if candidate in _VALID_LABELS:
                label = candidate
        elif line.startswith("CONFIDENCE:"):
            try:
                confidence = float(line.split(":", 1)[1].strip())
                confidence = max(0.0, min(1.0, confidence))
            except ValueError:
                confidence = None

    return label, confidence


def _score_gemini(text: str) -> dict:
    """Score a single ticket via Gemini zero-shot prompt.

    Requests a structured two-line response (LABEL + CONFIDENCE) so intensity
    is a continuous 0.0–1.0 value rather than a fixed constant. Falls back to
    single-token parsing + ``_LABEL_CONFIDENCE_FALLBACK`` when the structured
    format cannot be parsed.

    Args:
        text: Raw ticket text (summary + description).

    Returns:
        dict with label, score (model confidence in its label), and intensity
        (continuous NEGATIVE confidence used for gate thresholding).
    """
    client, model = _get_gemini()
    if client is None:
        return dict(_EMPTY_RESULT)

    prompt = _GEMINI_PROMPT.format(text=text[:2000])
    try:
        response = client.models.generate_content(model=model, contents=prompt)
        raw = (response.text or "").strip()
    except Exception as exc:
        log.warning("Gemini sentiment inference failed: %s", exc)
        return dict(_EMPTY_RESULT)

    label, confidence = _parse_gemini_response(raw.upper())

    if confidence is None:
        # Structured parse failed — fall back to single-token extraction
        first_token = raw.upper().split()[0].rstrip(".,;:") if raw else "NEUTRAL"
        if first_token not in _VALID_LABELS:
            log.warning("Gemini returned unparseable response %r, defaulting to NEUTRAL", raw[:80])
            first_token = "NEUTRAL"
        label = first_token
        confidence = _LABEL_CONFIDENCE_FALLBACK.get(label, 0.0)

    intensity = confidence if label == "NEGATIVE" else 0.0

    return {
        "label": label,
        "score": round(confidence, 4),
        "intensity": round(intensity, 4),
    }


# ── cardiffnlp backend ───────────────────────────────────────────────────────

def _resolve_sentiment_model_path() -> str:
    """Prefer fine-tuned checkpoint when models/sentiment_finetuned/ exists."""
    if FINETUNED_MODEL_DIR.is_dir() and any(FINETUNED_MODEL_DIR.iterdir()):
        return str(FINETUNED_MODEL_DIR)
    return SENTIMENT_MODEL


def get_sentiment_model():
    """Return the shared cardiffnlp sentiment pipeline, loading it on first call (lazy singleton).

    Loads from models/sentiment_finetuned/ when that directory exists (ANTSE-326 deployment),
    otherwise cardiffnlp/twitter-roberta-base-sentiment-latest (or SENTIMENT_MODEL override).

    Returns:
        A transformers sentiment-analysis pipeline instance.
    """
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline as hf_pipeline

        model_path = _resolve_sentiment_model_path()
        _pipeline = hf_pipeline(
            "sentiment-analysis",
            model=model_path,
            top_k=None,  # return scores for all labels
            truncation=True,  # truncate inputs longer than model's max sequence length
            max_length=512,   # cardiffnlp/twitter-roberta-base uses 512 token limit
        )
        log.info("Loaded sentiment model: %s", model_path)
    return _pipeline


def _score_cardiffnlp(text: str) -> dict:
    """Score a single ticket via cardiffnlp transformer pipeline.

    Args:
        text: Raw ticket text (summary + description).

    Returns:
        dict with label, score, intensity.
    """
    if not text or not text.strip():
        return dict(_EMPTY_RESULT)

    pipe = get_sentiment_model()
    truncated = text.strip()[:1500]

    try:
        raw_results: list[list[dict]] = pipe([truncated])
        scores_list: list[dict] = raw_results[0]
    except Exception as exc:
        log.warning("cardiffnlp inference failed: %s", exc)
        return dict(_EMPTY_RESULT)

    label_scores = {item["label"].upper(): item["score"] for item in scores_list}
    intensity = label_scores.get("NEGATIVE", 0.0)

    dominant = max(scores_list, key=lambda x: x["score"])
    dominant_label = dominant["label"].upper()
    dominant_score = dominant["score"]

    return {
        "label": dominant_label,
        "score": round(dominant_score, 4),
        "intensity": round(intensity, 4),
    }


# ── Ensemble backend (cardiffnlp + emotion/hartmann soft vote) ───────────────

def _get_emotion_pipeline():
    """Return the shared emotion/hartmann pipeline, loading it on first call.

    Returns:
        A transformers text-classification pipeline for j-hartmann/emotion-english-distilroberta-base.
    """
    global _emotion_pipeline
    if _emotion_pipeline is None:
        from transformers import pipeline as hf_pipeline

        _emotion_pipeline = hf_pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,
            truncation=True,  # truncate inputs longer than model's max sequence length
            max_length=512,   # distilroberta-base uses 512 token limit
        )
        log.info("Loaded emotion/hartmann pipeline for ensemble")
    return _emotion_pipeline


def _cardiffnlp_probs(text: str) -> np.ndarray:
    """Return [NEG, NEU, POS] probability vector from cardiffnlp.

    Args:
        text: Truncated ticket text.

    Returns:
        numpy array of shape (3,) summing to 1.
    """
    pipe = get_sentiment_model()
    raw: list[list[dict]] = pipe([text])
    scores_list: list[dict] = raw[0]
    score_dict = {item["label"].upper(): item["score"] for item in scores_list}
    neg = score_dict.get("NEGATIVE", score_dict.get("LABEL_0", 0.0))
    neu = score_dict.get("NEUTRAL", score_dict.get("LABEL_1", 0.0))
    pos = score_dict.get("POSITIVE", score_dict.get("LABEL_2", 0.0))
    vec = np.array([neg, neu, pos], dtype=float)
    total = vec.sum()
    return vec / total if total > 1e-9 else np.array([1 / 3, 1 / 3, 1 / 3])


def _emotion_probs(text: str) -> np.ndarray:
    """Return [NEG, NEU, POS] probability vector from emotion/hartmann (7-class → 3-class).

    Mapping:
        NEGATIVE = anger + disgust + fear + sadness
        POSITIVE = joy
        NEUTRAL  = neutral + surprise (residual)

    Args:
        text: Ticket text (will be automatically truncated to 512 tokens by pipeline).

    Returns:
        numpy array of shape (3,) summing to 1.
    """
    pipe = _get_emotion_pipeline()
    raw: list[dict] = pipe(text)[0]  # pipeline now handles truncation automatically
    score_dict = {item["label"].lower(): item["score"] for item in raw}
    neg = (score_dict.get("anger", 0.0) + score_dict.get("disgust", 0.0)
           + score_dict.get("fear", 0.0) + score_dict.get("sadness", 0.0))
    pos = score_dict.get("joy", 0.0)
    neu = max(0.0, 1.0 - neg - pos)
    vec = np.array([neg, neu, pos], dtype=float)
    total = vec.sum()
    return vec / total if total > 1e-9 else np.array([1 / 3, 1 / 3, 1 / 3])


def _score_ensemble(text: str) -> dict:
    """Score using a soft-vote ensemble of cardiffnlp + emotion/hartmann.

    Weights are Optuna LOO-CV optimized (2026-05-28, N=50, 63 combinations tested):
      cardiffnlp  = 0.9051
      emotion     = 0.0949
    Achieves 83.99% macro-F1 vs cardiffnlp alone at 69.1% (+14.9pp).
    Gap to Gemini: -11.7pp. Use when Gemini is unavailable.

    Args:
        text: Raw ticket text (summary + description).

    Returns:
        dict with label, score, intensity.
    """
    truncated = text.strip()[:1500]
    try:
        c_probs = _cardiffnlp_probs(truncated)
        e_probs = _emotion_probs(truncated)
    except Exception as exc:
        log.warning("Ensemble inference failed: %s — falling back to cardiffnlp", exc)
        return _score_cardiffnlp(text)

    blended = _ENSEMBLE_W_CARDIFF * c_probs + _ENSEMBLE_W_EMOTION * e_probs
    idx = int(np.argmax(blended))
    labels = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
    label = labels[idx]
    score = float(blended[idx])
    intensity = float(blended[0])  # NEG probability

    return {
        "label": label,
        "score": round(score, 4),
        "intensity": round(intensity, 4),
    }


# ── Public API ───────────────────────────────────────────────────────────────

def score_ticket(text: str) -> dict:
    """Score a single ticket text for sentiment and frustration intensity.

    Uses Gemini zero-shot by default (SENTIMENT_BACKEND=gemini).
    Select backend via SENTIMENT_BACKEND env var:
      "gemini"     — Gemini zero-shot (default, 95.7% macro-F1, ~2,646ms)
      "ensemble"   — cardiffnlp + emotion soft-vote (83.99% macro-F1, ~250ms, offline-safe)
      "cardiffnlp" — cardiffnlp alone (69.1% macro-F1, 221ms, lightest)

    When Gemini is the backend and the call fails, automatically falls back
    to the ensemble.

    The NEGATIVE label confidence is used as the ``intensity`` field so
    callers can threshold on it (e.g. intensity > 0.6 → escalate).

    Note:
        When the Gemini backend is active, intensity is binary (high or low only) because
        Gemini returns a single NEGATIVE/NEUTRAL/POSITIVE label. The "medium" bucket is
        only reachable when the ensemble fallback (cardiffnlp) is active. This inconsistency
        is accepted — routing rules that check for "medium" intensity will behave differently
        depending on which backend is in use.

    Args:
        text: Raw ticket text (summary + description).  Empty or None input
              returns a safe zero-intensity result without model inference.

    Returns:
        dict with keys:
            label (str):       Dominant sentiment label — NEGATIVE, NEUTRAL, or POSITIVE.
            score (float):     Confidence score for the dominant label (0–1).
            intensity (float): NEGATIVE label score, 0–1.  Higher = more frustrated.
    """
    if not text or not text.strip():
        return dict(_EMPTY_RESULT)

    if SENTIMENT_BACKEND == "cardiffnlp":
        return _score_cardiffnlp(text)

    if SENTIMENT_BACKEND == "ensemble":
        return _score_ensemble(text)

    # Default: Gemini with ensemble as fallback
    result = _score_gemini(text)
    if result["label"] == "NEUTRAL" and result["score"] == 0.0:
        log.info("Gemini failed, falling back to ensemble")
        return _score_ensemble(text)
    return result


def score_tickets(texts: list[str]) -> list[dict]:
    """Score a batch of ticket texts for sentiment and frustration intensity.

    Processes texts individually to handle per-item truncation and guard
    against single failures poisoning the whole batch.

    Args:
        texts: List of raw ticket strings.

    Returns:
        List of dicts in the same order as ``texts``, each matching the
        schema returned by :func:`score_ticket`.
    """
    return [score_ticket(t) for t in texts]
