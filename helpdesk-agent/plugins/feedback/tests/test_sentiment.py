"""Unit tests for plugins/feedback/sentiment.py — ANTSE-322."""

import importlib
import sys
import types
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pipeline_mock(label_scores: list[dict]):
    """Return a callable that mimics transformers pipeline output.

    Args:
        label_scores: e.g. [{'label': 'NEGATIVE', 'score': 0.9}, ...]
    """
    pipe = MagicMock(return_value=[label_scores])
    return pipe


def _reset_singleton():
    """Clear the module-level _pipeline singleton between tests."""
    import plugins.feedback.sentiment as mod
    mod._pipeline = None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestScoreTicketReturnShape:
    """score_ticket always returns a dict with the expected keys."""

    def test_score_ticket_returns_expected_keys(self):
        """Result dict must contain label, score, and intensity."""
        mock_scores = [
            {"label": "NEGATIVE", "score": 0.7},
            {"label": "NEUTRAL", "score": 0.2},
            {"label": "POSITIVE", "score": 0.1},
        ]
        with patch("plugins.feedback.sentiment.get_sentiment_model") as mock_get:
            mock_get.return_value = _make_pipeline_mock(mock_scores)
            from plugins.feedback.sentiment import score_ticket
            result = score_ticket("Something went wrong again.")

        assert isinstance(result, dict), "Result should be a dict"
        assert "label" in result, "Missing key: label"
        assert "score" in result, "Missing key: score"
        assert "intensity" in result, "Missing key: intensity"

    def test_score_ticket_value_types(self):
        """label is str, score and intensity are floats in [0, 1]."""
        mock_scores = [
            {"label": "NEGATIVE", "score": 0.85},
            {"label": "NEUTRAL", "score": 0.1},
            {"label": "POSITIVE", "score": 0.05},
        ]
        with patch("plugins.feedback.sentiment.get_sentiment_model") as mock_get:
            mock_get.return_value = _make_pipeline_mock(mock_scores)
            from plugins.feedback.sentiment import score_ticket
            result = score_ticket("This is completely broken.")

        assert isinstance(result["label"], str)
        assert isinstance(result["score"], float)
        assert isinstance(result["intensity"], float)
        assert 0.0 <= result["score"] <= 1.0
        assert 0.0 <= result["intensity"] <= 1.0


class TestFrustrationScoring:
    """Validate that clearly negative/positive texts score as expected."""

    def test_score_ticket_negative_text(self):
        """Angry, frustrated text should produce high intensity (>= 0.6)."""
        mock_scores = [
            {"label": "NEGATIVE", "score": 0.95},
            {"label": "NEUTRAL", "score": 0.03},
            {"label": "POSITIVE", "score": 0.02},
        ]
        with patch("plugins.feedback.sentiment.get_sentiment_model") as mock_get:
            mock_get.return_value = _make_pipeline_mock(mock_scores)
            from plugins.feedback.sentiment import score_ticket
            result = score_ticket("I'm furious, nothing works and nobody is helping me!")

        assert result["label"] == "NEGATIVE", f"Expected NEGATIVE, got {result['label']}"
        assert result["intensity"] >= 0.6, (
            f"Expected intensity >= 0.6 for frustrated text, got {result['intensity']}"
        )

    def test_score_ticket_positive_text(self):
        """Grateful, positive text should produce low intensity (< 0.3)."""
        mock_scores = [
            {"label": "POSITIVE", "score": 0.92},
            {"label": "NEUTRAL", "score": 0.06},
            {"label": "NEGATIVE", "score": 0.02},
        ]
        with patch("plugins.feedback.sentiment.get_sentiment_model") as mock_get:
            mock_get.return_value = _make_pipeline_mock(mock_scores)
            from plugins.feedback.sentiment import score_ticket
            result = score_ticket("Thank you for your help! Everything is working perfectly now.")

        assert result["label"] == "POSITIVE", f"Expected POSITIVE, got {result['label']}"
        assert result["intensity"] < 0.3, (
            f"Expected intensity < 0.3 for positive text, got {result['intensity']}"
        )


class TestEdgeCases:
    """Empty and None inputs should not crash."""

    def test_score_ticket_empty_string(self):
        """Empty string returns safe zero-intensity result without model call."""
        from plugins.feedback.sentiment import score_ticket
        with patch("plugins.feedback.sentiment.get_sentiment_model") as mock_get:
            result = score_ticket("")
            mock_get.assert_not_called()

        assert result["intensity"] == 0.0
        assert result["label"] == "NEUTRAL"

    def test_score_ticket_none_input(self):
        """None input returns safe zero-intensity result without model call."""
        from plugins.feedback.sentiment import score_ticket
        with patch("plugins.feedback.sentiment.get_sentiment_model") as mock_get:
            result = score_ticket(None)
            mock_get.assert_not_called()

        assert result["intensity"] == 0.0

    def test_score_ticket_whitespace_only(self):
        """Whitespace-only input returns safe zero-intensity result."""
        from plugins.feedback.sentiment import score_ticket
        with patch("plugins.feedback.sentiment.get_sentiment_model") as mock_get:
            result = score_ticket("   \n\t  ")
            mock_get.assert_not_called()

        assert result["intensity"] == 0.0


class TestSingleton:
    """get_sentiment_model() returns the same instance across calls."""

    def test_singleton_returns_same_instance(self):
        """Multiple calls to get_sentiment_model should return the identical object.

        Patches the transformers import *inside* get_sentiment_model by
        replacing the ``pipeline`` function that the module calls via its
        local import statement.  We do this by injecting a fake module into
        sys.modules before resetting the singleton so the lazy import picks
        up the mock.
        """
        import plugins.feedback.sentiment as mod

        fake_pipe = MagicMock(name="FakePipeline")
        fake_transformers = MagicMock()
        fake_transformers.pipeline = MagicMock(return_value=fake_pipe)

        mod._pipeline = None  # reset singleton

        with patch.dict("sys.modules", {"transformers": fake_transformers}):
            first = mod.get_sentiment_model()
            second = mod.get_sentiment_model()

        mod._pipeline = None  # clean up after test
        assert first is second, "Singleton should return the same pipeline instance"
        # transformers.pipeline should have been called exactly once
        assert fake_transformers.pipeline.call_count == 1

    def test_model_loads_lazily(self):
        """Importing the module must NOT trigger model loading."""
        # Evict any cached version so we get a clean import
        for key in list(sys.modules.keys()):
            if "plugins.feedback.sentiment" in key:
                del sys.modules[key]

        fake_transformers = MagicMock()

        with patch.dict("sys.modules", {"transformers": fake_transformers}):
            import plugins.feedback.sentiment as mod  # noqa: F401
            # _pipeline must still be None — model not yet loaded
            assert mod._pipeline is None
            # transformers.pipeline must NOT have been called at import time
            fake_transformers.pipeline.assert_not_called()

        # Evict the module so other tests start clean
        for key in list(sys.modules.keys()):
            if "plugins.feedback.sentiment" in key:
                del sys.modules[key]


class TestBatchScoring:
    """score_tickets processes each item independently."""

    def test_score_tickets_returns_list(self):
        """Batch function returns one result per input."""
        mock_scores = [
            {"label": "NEUTRAL", "score": 0.7},
            {"label": "NEGATIVE", "score": 0.2},
            {"label": "POSITIVE", "score": 0.1},
        ]
        texts = ["First ticket.", "Second ticket.", "Third ticket."]
        with patch("plugins.feedback.sentiment.get_sentiment_model") as mock_get:
            mock_get.return_value = _make_pipeline_mock(mock_scores)
            from plugins.feedback.sentiment import score_tickets
            results = score_tickets(texts)

        assert len(results) == len(texts)
        for r in results:
            assert "label" in r and "score" in r and "intensity" in r

    def test_score_tickets_empty_list(self):
        """Empty list input returns empty list without calling the model."""
        from plugins.feedback.sentiment import score_tickets
        with patch("plugins.feedback.sentiment.get_sentiment_model") as mock_get:
            results = score_tickets([])
            mock_get.assert_not_called()

        assert results == []
