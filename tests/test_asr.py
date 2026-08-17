"""Tests for rag.asr — the notebook-to-module port of the Whisper ASR step.

Deliberately does NOT download or run a real Whisper model: the suite has to
stay offline and fast (`tests/conftest.py` forces the hash embedder for the
same reason). What is worth testing without a model is the contract around
the engine — device resolution, the opt-in-only fallback, and the fact that
a fallback transcript is always labelled as one.
"""
import pytest

from rag import asr
from rag.config import get_config


def test_reference_transcripts_cover_every_demo_clip():
    """Every audio/queryNN.wav must have a reference transcript.

    The clips and the references are the fixture `scripts/run_end_to_end.py
    --source audio` runs on, and `notebooks/02_whisper_asr.ipynb` scores WER
    against — a clip with no reference silently drops out of both.
    """
    clips = asr.sample_clips()
    assert clips, "no demo clips found in audio/"
    missing = [c.name for c in clips if c.name not in asr.REFERENCE_TRANSCRIPTS]
    assert not missing, f"clips with no reference transcript: {missing}"


def test_resolve_device_honors_explicit_config(monkeypatch):
    monkeypatch.setenv("ASR_DEVICE", "cpu")
    monkeypatch.setenv("ASR_COMPUTE_TYPE", "int8")
    cfg = get_config(refresh=True)
    assert asr._resolve_device(cfg) == ("cpu", "int8")


def test_resolve_device_auto_picks_a_valid_pair(monkeypatch):
    monkeypatch.setenv("ASR_DEVICE", "auto")
    monkeypatch.setenv("ASR_COMPUTE_TYPE", "auto")
    cfg = get_config(refresh=True)
    device, compute_type = asr._resolve_device(cfg)
    assert (device, compute_type) in {("cpu", "int8"), ("cuda", "float16")}


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        asr.transcribe("audio/definitely_not_a_real_clip.wav")


def _force_no_engine(monkeypatch):
    def _boom(*_a, **_kw):
        raise asr.ASRUnavailableError("faster-whisper not installed (simulated)")
    monkeypatch.setattr(asr, "get_model", _boom)


def test_no_engine_without_fallback_raises(monkeypatch):
    """A demo must never silently fake its own transcript."""
    _force_no_engine(monkeypatch)
    clip = asr.sample_clips()[0]
    with pytest.raises(asr.ASRUnavailableError):
        asr.transcribe(clip, allow_fallback=False)


def test_no_engine_with_fallback_is_labelled(monkeypatch):
    _force_no_engine(monkeypatch)
    clip = asr.sample_clips()[0]
    out = asr.transcribe(clip, allow_fallback=True)
    assert out["text"] == asr.REFERENCE_TRANSCRIPTS[clip.name]
    assert out["engine"] == "fallback-reference"
    # The UI reads these keys (docs/UI_WIREFRAME.md regions [1]-[2]).
    assert {"text", "language", "language_probability", "segments",
            "source_file"} <= out.keys()


def test_fallback_env_var_opt_in(monkeypatch):
    _force_no_engine(monkeypatch)
    clip = asr.sample_clips()[0]

    monkeypatch.setenv("ASR_ALLOW_FALLBACK", "false")
    with pytest.raises(asr.ASRUnavailableError):
        asr.transcribe(clip)

    monkeypatch.setenv("ASR_ALLOW_FALLBACK", "true")
    assert asr.transcribe(clip)["engine"] == "fallback-reference"


def test_fallback_refuses_unknown_clip(monkeypatch, tmp_path):
    """Fallback only covers the ten known clips — not arbitrary user audio."""
    _force_no_engine(monkeypatch)
    unknown = tmp_path / "user_recording.wav"
    unknown.write_bytes(b"RIFF....WAVEfmt ")
    with pytest.raises(asr.ASRUnavailableError):
        asr.transcribe(unknown, allow_fallback=True)
