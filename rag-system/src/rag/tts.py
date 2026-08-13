"""
tts.py — text-to-speech synthesis for the Answerer's ≤15s spoken summary
(Final deliverable).

Fragment-based synthesis: the Answerer produces one complete `speech` string
(≤ ~55 words / ≤15s per `prompts/answerer_critic.md`) and we synthesize it in
a single shot to a finished audio file, then hand back its path — as opposed
to a streaming approach that would emit audio chunk-by-chunk as tokens arrive.
Fragment-based is simpler, works with every provider (including fully offline
`pyttsx3`), and is the right fit here since the Critic must approve the whole
payload before anything is spoken (see `prompts/answerer_critic.md`) — there
is no partial text to stream ahead of that gate.

Provider dispatch mirrors `web-search-mcp/web_search.py`'s graceful
degradation: a requested provider that is missing its API key/package never
raises, it logs a warning to stderr and falls back to the fully offline
`pyttsx3` default so this pipeline (and the notebook that exercises it) runs
with **zero** setup.

Budget enforcement is defensive-only: the Answerer/Critic pair is supposed to
already guarantee the `speech` text fits the ≤15s budget
(`prompts/answerer_critic.md`), but `speak()` re-checks with `fits_budget()`
before synthesizing anyway and truncates rather than crashing if a caller
hands it something too long.
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path
from typing import Optional

from .config import REPO_ROOT, get_config

DEFAULT_WPM = 150.0
DEFAULT_MAX_SECONDS = 15.0

AUDIO_DIR = REPO_ROOT / "audio"


def estimate_speech_seconds(text: str, wpm: float = DEFAULT_WPM) -> float:
    """Estimate how long `text` takes to speak aloud, from word count alone.

    No network call, no audio synthesis — a cheap word-count/wpm estimate
    used to enforce the ≤15s spoken-answer budget before (and defensively
    inside) synthesis.

    Args:
        text: the candidate spoken text.
        wpm: assumed speaking rate in words per minute (default 150, a
            typical conversational-assistant pace).

    Returns:
        estimated duration in seconds (float).
    """
    words = len(text.split())
    if wpm <= 0:
        return 0.0
    return words / wpm * 60.0


def fits_budget(text: str, max_seconds: float = DEFAULT_MAX_SECONDS) -> bool:
    """Check whether `text` fits within the spoken-answer time budget.

    Args:
        text: the candidate spoken text.
        max_seconds: budget in seconds (default 15.0, per
            `prompts/answerer_critic.md`'s ≤15s spoken-answer rule).

    Returns:
        True if `estimate_speech_seconds(text) <= max_seconds`.
    """
    return estimate_speech_seconds(text) <= max_seconds


def _truncate_to_budget(text: str, max_seconds: float = DEFAULT_MAX_SECONDS,
                         wpm: float = DEFAULT_WPM) -> str:
    """Truncate `text` to the last full sentence that still fits the budget."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    kept: list = []
    for sentence in sentences:
        candidate = " ".join(kept + [sentence])
        if estimate_speech_seconds(candidate, wpm) <= max_seconds:
            kept.append(sentence)
        else:
            break
    if kept:
        return " ".join(kept)
    # Even the first sentence alone exceeds the budget — fall back to a hard
    # word-count truncation of it rather than returning nothing to speak.
    first = sentences[0] if sentences else text
    words = first.split()
    max_words = max(1, int(max_seconds / 60.0 * wpm))
    return " ".join(words[:max_words])


def _deterministic_id(text: str) -> str:
    """Short, deterministic (content-hash) id used for the default filename."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _default_out_path(text: str) -> Path:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    return AUDIO_DIR / f"summary_{_deterministic_id(text)}.wav"


def speak(text: str, out_path: Optional[str | Path] = None,
          provider: Optional[str] = None, voice: Optional[str] = None) -> Path:
    """Synthesize `text` to a finished audio file (fragment-based) and return its path.

    Defensively enforces the ≤15s budget (`fits_budget()`): if `text` is too
    long, it is truncated to the last full sentence that fits and a warning
    is printed — this function never hard-crashes on an over-length payload,
    it degrades.

    Provider is chosen from `provider`, then the `TTS_PROVIDER` env var, then
    `rag.config.Config.tts_provider` (default `"pyttsx3"`, fully offline). If
    `provider="openai"` or `"elevenlabs"` is requested but the required API
    key or package is missing, a warning is logged to stderr and synthesis
    falls back to `_speak_pyttsx3()` rather than raising.

    Args:
        text: the spoken text to synthesize (typically the Answerer's
            `speech` field from `prompts/answerer_critic.md`'s payload).
        out_path: destination file path. Defaults to
            `rag-system/audio/summary_<content-hash>.wav` (deterministic, no
            timestamp, so repeat calls with the same text overwrite in place).
        provider: `"pyttsx3"` | `"openai"` | `"elevenlabs"`. Falls back to the
            `TTS_PROVIDER` env var, then `Config.tts_provider`.
        voice: optional provider-specific voice name/id. Falls back to
            `Config.tts_voice` when not given.

    Returns:
        Path to the written audio file.
    """
    cfg = get_config()

    if not fits_budget(text):
        estimated = estimate_speech_seconds(text)
        truncated = _truncate_to_budget(text)
        print(
            f"tts.speak: text is ~{estimated:.1f}s, over the "
            f"{DEFAULT_MAX_SECONDS:.0f}s budget; truncating to the last full "
            f"sentence that fits (~{estimate_speech_seconds(truncated):.1f}s).",
            file=sys.stderr,
        )
        text = truncated

    provider = (provider or os.environ.get("TTS_PROVIDER") or cfg.tts_provider or "pyttsx3").lower()
    voice = voice or cfg.tts_voice or None

    out_path = Path(out_path) if out_path is not None else _default_out_path(text)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            print(
                "tts.speak: provider='openai' requested but OPENAI_API_KEY is "
                "not set; falling back to offline pyttsx3.",
                file=sys.stderr,
            )
            return _speak_pyttsx3(text, out_path, voice)
        try:
            return _speak_openai(text, out_path, voice, cfg)
        except ImportError as e:
            print(f"tts.speak: {e}; falling back to offline pyttsx3.", file=sys.stderr)
            return _speak_pyttsx3(text, out_path, voice)

    if provider == "elevenlabs":
        if not os.environ.get("ELEVENLABS_API_KEY"):
            print(
                "tts.speak: provider='elevenlabs' requested but "
                "ELEVENLABS_API_KEY is not set; falling back to offline pyttsx3.",
                file=sys.stderr,
            )
            return _speak_pyttsx3(text, out_path, voice)
        try:
            return _speak_elevenlabs(text, out_path, voice)
        except ImportError as e:
            print(f"tts.speak: {e}; falling back to offline pyttsx3.", file=sys.stderr)
            return _speak_pyttsx3(text, out_path, voice)

    return _speak_pyttsx3(text, out_path, voice)


def _speak_openai(text: str, out_path: Path, voice: Optional[str], cfg) -> Path:
    """Synthesize via the OpenAI TTS API (`client.audio.speech.create`). Needs `OPENAI_API_KEY`."""
    from openai import OpenAI  # local import: optional dependency

    client = OpenAI()
    response = client.audio.speech.create(
        model=cfg.tts_model,
        voice=voice or "alloy",
        input=text,
    )
    # `response` is a binary response object across recent SDK versions;
    # `.content` (raw bytes) is the most version-stable way to get the audio.
    out_path.write_bytes(response.content)
    return out_path


def _speak_elevenlabs(text: str, out_path: Path, voice: Optional[str]) -> Path:
    """Synthesize via the ElevenLabs API. Needs `ELEVENLABS_API_KEY` and the `elevenlabs` package."""
    try:
        from elevenlabs import generate, save
    except ImportError as e:
        raise ImportError(
            "tts.speak: provider='elevenlabs' requires the 'elevenlabs' "
            "package (pip install elevenlabs)"
        ) from e

    audio = generate(
        text=text,
        voice=voice or "Rachel",
        api_key=os.environ.get("ELEVENLABS_API_KEY"),
    )
    save(audio, str(out_path))
    return out_path


def _speak_pyttsx3(text: str, out_path: Path, voice: Optional[str]) -> Path:
    """Synthesize fully offline via `pyttsx3` — no API key, this is the zero-setup default."""
    import pyttsx3  # local import: keeps module import light for callers that don't need it

    engine = pyttsx3.init()
    if voice:
        for v in engine.getProperty("voices"):
            if voice.lower() in (v.id or "").lower() or voice.lower() in (v.name or "").lower():
                engine.setProperty("voice", v.id)
                break
    engine.save_to_file(text, str(out_path))
    engine.runAndWait()
    return out_path


if __name__ == "__main__":
    path = speak("This is a quick offline pyttsx3 smoke test of tts.speak.")
    print("wrote", path)
