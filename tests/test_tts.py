"""Offline smoke tests for rag.tts — no network, no real API calls.

Follows the convention in tests/test_pipeline.py / tests/test_reconcile.py
(plain pytest, PYTHONPATH=src via conftest.py). `speak(provider="pyttsx3")`
is fully offline (no key), so it is exercised for real here rather than
mocked — it is skipped gracefully (not failed) if the local pyttsx3 install
can't find a system TTS engine, since that is an environment issue rather
than a bug in this module.
"""
import importlib.util

import pytest

from rag.tts import estimate_speech_seconds, fits_budget, speak

PYTTSX3_AVAILABLE = importlib.util.find_spec("pyttsx3") is not None


def _pyttsx3_engine_works() -> bool:
    """Best-effort check that pyttsx3 can actually init a system TTS engine."""
    if not PYTTSX3_AVAILABLE:
        return False
    try:
        import pyttsx3

        pyttsx3.init()
        return True
    except Exception:
        return False


PYTTSX3_WORKS = _pyttsx3_engine_works()


# ---- estimate_speech_seconds / fits_budget --------------------------------

def test_estimate_speech_seconds_known_word_count():
    # 150 words at the default 150 wpm should take exactly 60s.
    text = " ".join(["word"] * 150)
    assert estimate_speech_seconds(text) == pytest.approx(60.0)


def test_estimate_speech_seconds_scales_with_wpm():
    text = " ".join(["word"] * 100)
    assert estimate_speech_seconds(text, wpm=100.0) == pytest.approx(60.0)
    assert estimate_speech_seconds(text, wpm=200.0) == pytest.approx(30.0)


def test_estimate_speech_seconds_empty_text():
    assert estimate_speech_seconds("") == 0.0


def test_fits_budget_true_for_short_text():
    # ~55 words at 150 wpm is ~22s... use a genuinely short answer instead.
    speech = ("My top pick is the GreenGleam Steel-Safe Eco cleaner — plant-based, "
              "4.6 stars, about $12.49 for 16 ounces.")
    assert fits_budget(speech, max_seconds=15.0)


def test_fits_budget_false_for_long_text():
    long_text = " ".join(["word"] * 200)  # 80s at 150 wpm
    assert not fits_budget(long_text, max_seconds=15.0)


# ---- speak() (pyttsx3, offline) -------------------------------------------

@pytest.mark.skipif(not PYTTSX3_WORKS, reason="pyttsx3 not installed or no system TTS engine available")
def test_speak_pyttsx3_writes_real_audio_file(tmp_path):
    out_path = tmp_path / "summary_test.wav"
    result = speak("This is a short test of offline speech synthesis.",
                    out_path=out_path, provider="pyttsx3")
    assert result == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0


@pytest.mark.skipif(not PYTTSX3_WORKS, reason="pyttsx3 not installed or no system TTS engine available")
def test_speak_default_out_path_is_deterministic(monkeypatch, tmp_path):
    import rag.tts as tts_mod

    monkeypatch.setattr(tts_mod, "AUDIO_DIR", tmp_path)
    text = "Deterministic default output path test."
    p1 = speak(text, provider="pyttsx3")
    p2 = speak(text, provider="pyttsx3")
    assert p1 == p2
    assert p1.exists()
    assert p1.name.startswith("summary_")
    assert p1.suffix == ".wav"


@pytest.mark.skipif(not PYTTSX3_WORKS, reason="pyttsx3 not installed or no system TTS engine available")
def test_speak_truncates_over_budget_text_instead_of_crashing(tmp_path, capsys):
    long_text = " ".join(["word"] * 200) + "."  # well over 15s
    out_path = tmp_path / "summary_long.wav"
    result = speak(long_text, out_path=out_path, provider="pyttsx3")
    assert result.exists()
    captured = capsys.readouterr()
    assert "budget" in captured.err.lower()


def test_speak_unknown_provider_falls_back_without_key(tmp_path, capsys):
    # provider='openai' with no OPENAI_API_KEY set must degrade to pyttsx3,
    # not raise. This test doesn't require a real audio engine to pass its
    # assertion about the warning; it only checks the file if pyttsx3 works.
    import os

    if "OPENAI_API_KEY" in os.environ:
        pytest.skip("OPENAI_API_KEY is set in this environment; can't test the no-key fallback path")
    if not PYTTSX3_WORKS:
        pytest.skip("pyttsx3 not installed or no system TTS engine available")

    out_path = tmp_path / "summary_fallback.wav"
    result = speak("Short grounded summary.", out_path=out_path, provider="openai")
    assert result.exists()
    captured = capsys.readouterr()
    assert "falling back to offline pyttsx3" in captured.err
