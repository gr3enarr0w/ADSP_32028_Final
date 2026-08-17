"""
asr.py — speech-to-text for the voice-in half of the pipeline (Final
deliverable), ported out of `notebooks/02_whisper_asr.ipynb` into a callable
module.

`docs/UI_WIREFRAME.md` region [1] flagged this exact gap: the Streamlit mic
control "needs a Streamlit audio-input component (or JS recorder) that saves a
file and calls `notebooks/02_whisper_asr.ipynb`'s `transcribe()` logic (needs
porting from notebook into a callable `src/rag/asr.py` module — it currently
only exists as notebook cells)." This module is that port.

The return contract is **unchanged** from the notebook, because
`docs/UI_WIREFRAME.md` region [2] already specifies the UI reads
`output["text"]` and (optionally) `output["segments"][*]["words"]`:

    {"text": str, "language": str, "language_probability": float,
     "segments": [{"start", "end", "text", "words": [...]}],
     "source_file": str, "engine": str}

`engine` is the one added key — it names what actually produced the
transcript, so the UI/step-log never presents a fallback transcript as if a
real model had run.

Fragment-based, not streaming: `transcribe()` takes a *finished* audio file,
per the assignment spec ("record → send file to ASR") and matching how
`rag.tts.speak()` does fragment-based synthesis on the way back out.

Provider degradation follows the same convention as `rag.tts` and
`web-search-mcp/web_search.py` — a missing optional dependency logs to stderr
rather than exploding — with one deliberate difference: ASR does **not**
silently substitute canned text. Falling back to the reference transcripts in
`REFERENCE_TRANSCRIPTS` requires opting in (`allow_fallback=True` or
`ASR_ALLOW_FALLBACK=true`), and the result is stamped
`engine="fallback-reference"`. A demo that quietly faked its own transcript
would invalidate the whole voice-to-voice claim.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

from .config import REPO_ROOT, get_config

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".webm", ".flac", ".ogg"}

AUDIO_DIR = REPO_ROOT / "audio"

# Ground-truth transcripts for the ten prerecorded demo clips in `audio/`.
# Lifted verbatim from `notebooks/02_whisper_asr.ipynb` §7, where they are the
# references the notebook's WER evaluation scores against — so the offline
# fallback path and the notebook's accuracy numbers stay consistent.
REFERENCE_TRANSCRIPTS = {
    "query01.wav": "Recommend an eco-friendly stainless-steel cleaner under fifteen dollars.",
    "query02.wav": "Compare Weiman with Therapy Clean.",
    "query03.wav": "Show me Seventh Generation products.",
    "query04.wav": "Find a plant-based kitchen cleaner.",
    "query05.wav": "Show products under twelve ninety-nine.",
    "query06.wav": "Compare Method and Mrs. Meyer's.",
    "query07.wav": "Find fragrance-free cleaner.",
    "query08.wav": "Highest rated stainless steel cleaner.",
    "query09.wav": "Best cleaner for granite countertops.",
    "query10.wav": "Show products with four-point-five stars or higher.",
}


class ASRUnavailableError(RuntimeError):
    """Raised when no ASR engine can run and fallback was not opted into."""


# ---------------------------------------------------------------------------
# model loading
# ---------------------------------------------------------------------------

_MODEL = None
_MODEL_KEY: Optional[tuple] = None


def _resolve_device(cfg) -> tuple[str, str]:
    """Pick (device, compute_type), honoring explicit config over autodetect.

    Mirrors `notebooks/02_whisper_asr.ipynb` §2: CUDA + float16 when a GPU is
    present, CPU + int8 otherwise. Torch is optional here — it is a
    `faster-whisper` transitive concern, not something this module should
    hard-require just to answer "is there a GPU".
    """
    device = cfg.asr_device
    if device == "auto":
        try:
            import torch  # noqa: PLC0415 - optional, autodetect only

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            device = "cpu"

    compute_type = cfg.asr_compute_type
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"
    return device, compute_type


def get_model(refresh: bool = False):
    """Load (and cache) the faster-whisper model.

    Cached process-wide the same way `rag.config.get_config()` and
    `rag.retrieval.get_retriever()` cache theirs — model load is by far the
    slowest step in a voice turn (seconds), and Streamlit re-runs the whole
    script on every widget interaction, so re-loading per turn would dominate
    the demo's latency.

    Returns:
        a `faster_whisper.WhisperModel`.

    Raises:
        ASRUnavailableError: if `faster-whisper` is not installed.
    """
    global _MODEL, _MODEL_KEY
    cfg = get_config()
    device, compute_type = _resolve_device(cfg)
    key = (cfg.asr_model, device, compute_type)

    if _MODEL is not None and _MODEL_KEY == key and not refresh:
        return _MODEL

    try:
        from faster_whisper import WhisperModel  # noqa: PLC0415 - optional dep
    except ImportError as e:
        raise ASRUnavailableError(
            "faster-whisper is not installed. Install it with "
            "`pip install faster-whisper` (it is listed in requirements-rag.txt), "
            "or set ASR_ALLOW_FALLBACK=true to use the prerecorded reference "
            "transcripts for the demo clips in audio/."
        ) from e

    print(
        f"rag.asr: loading faster-whisper model={cfg.asr_model} "
        f"device={device} compute_type={compute_type}",
        file=sys.stderr,
    )
    _MODEL = WhisperModel(cfg.asr_model, device=device, compute_type=compute_type)
    _MODEL_KEY = key
    return _MODEL


# ---------------------------------------------------------------------------
# transcription
# ---------------------------------------------------------------------------

def _fallback_transcript(audio_path: Path) -> Optional[str]:
    """Reference transcript for one of the ten prerecorded demo clips, if any."""
    return REFERENCE_TRANSCRIPTS.get(audio_path.name)


def _allow_fallback(explicit: Optional[bool]) -> bool:
    if explicit is not None:
        return explicit
    return os.environ.get("ASR_ALLOW_FALLBACK", "false").strip().lower() in {
        "1", "true", "yes", "on",
    }


def transcribe(audio_file: str | Path,
               allow_fallback: Optional[bool] = None) -> dict[str, Any]:
    """Transcribe one short English product-query audio file.

    Args:
        audio_file: path to a finished audio file (`SUPPORTED_EXTENSIONS`).
        allow_fallback: if True, and no ASR engine is available, return the
            known reference transcript for one of the ten prerecorded demo
            clips instead of raising. Defaults to the `ASR_ALLOW_FALLBACK`
            environment variable (default false).

    Returns:
        the notebook's transcript dict, plus an `engine` key naming what
        produced it (`"faster-whisper:<model>"` or `"fallback-reference"`).

    Raises:
        FileNotFoundError: if `audio_file` does not exist.
        ASRUnavailableError: if no engine is available and fallback is either
            not permitted or has no reference transcript for this file.
    """
    audio_path = Path(audio_file)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    cfg = get_config()
    try:
        model = get_model()
    except ASRUnavailableError:
        text = _fallback_transcript(audio_path) if _allow_fallback(allow_fallback) else None
        if text is None:
            raise
        print(
            f"rag.asr: no ASR engine available; using the reference transcript "
            f"for {audio_path.name} (engine=fallback-reference).",
            file=sys.stderr,
        )
        return {
            "text": text,
            "language": "en",
            "language_probability": 1.0,
            "segments": [],
            "source_file": str(audio_path),
            "engine": "fallback-reference",
        }

    segments, info = model.transcribe(
        str(audio_path),
        language="en",
        beam_size=cfg.asr_beam_size,
        vad_filter=True,
        word_timestamps=True,
        condition_on_previous_text=False,
    )

    transcript_parts: list[str] = []
    segment_results: list[dict[str, Any]] = []

    # Iterating over `segments` is what actually performs the transcription —
    # faster-whisper returns a lazy generator, so nothing runs until here.
    for segment in segments:
        segment_text = segment.text.strip()
        if segment_text:
            transcript_parts.append(segment_text)

        words = [
            {
                "word": word.word.strip(),
                "start": word.start,
                "end": word.end,
                "confidence": word.probability,
            }
            for word in (segment.words or [])
        ]
        segment_results.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment_text,
            "words": words,
        })

    return {
        "text": " ".join(transcript_parts).strip(),
        "language": info.language,
        "language_probability": info.language_probability,
        "segments": segment_results,
        "source_file": str(audio_path),
        "engine": f"faster-whisper:{cfg.asr_model}",
    }


def sample_clips() -> list[Path]:
    """The prerecorded demo clips in `audio/`, sorted, for the UI's picker."""
    if not AUDIO_DIR.is_dir():
        return []
    return sorted(
        p for p in AUDIO_DIR.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTENSIONS and p.name.startswith("query")
    )


if __name__ == "__main__":
    import json

    clips = sample_clips()
    if not clips:
        print("No clips found in audio/", file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps(transcribe(clips[0]), indent=2)[:2000])
