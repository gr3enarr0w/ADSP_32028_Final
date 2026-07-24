"""Integration test for sentiment pipeline ordering (ANTSE-323)."""

from unittest.mock import patch

import pytest

from core.pipeline import _phase_analysis, _phase_feedback
from db import get_db_conn, init_db


@pytest.fixture(autouse=True)
def use_temp_db(tmp_path):
    """Use a temporary SQLite DB for full phase flow."""
    db_path = str(tmp_path / "test.db")
    with patch("db.DB_PATH", db_path):
        init_db()
        yield


def _seed_unclassified_ticket(ticket_key: str = "INT-1") -> None:
    with get_db_conn() as conn:
        conn.execute(
            """
            INSERT INTO tickets (ticket_key, summary, description, status, created_at)
            VALUES (?, ?, ?, ?, datetime('now'))
            """,
            (ticket_key, "Cannot access workspace", "I still cannot access the workspace.", "New"),
        )


@patch("analysis.classifier.classify_ticket")
@patch("plugins.feedback.pipeline.score_ticket")
@patch("analysis.router._get_known_categories", return_value={"Access"})
@patch("analysis.router.get_plugin_config", return_value={"sentiment_escalation_threshold": "high"})
def test_pipeline_ordering_classify_then_sentiment_then_router(
    _mock_router_cfg,
    _mock_known,
    mock_score_ticket,
    mock_classify_ticket,
):
    """One flow proves analysis->feedback->router ordering on stored DB values."""
    from analysis.router import route_ticket

    mock_classify_ticket.return_value = {
        "category": "Access",
        "issue_type": "Access request",
        "question_type": "troubleshooting",
        "keywords": ["access"],
        "has_resolution": False,
        "resolution_summary": "",
        "confidence": 0.91,
        "classifier": "gemini",
    }
    mock_score_ticket.return_value = {"label": "NEGATIVE", "score": 0.9, "intensity": 0.9}

    _seed_unclassified_ticket("INT-1")

    # Step 1: analysis phase writes classification and dispatches "classified".
    # Feedback plugin reacts immediately and persists sentiment.
    _phase_analysis()

    with get_db_conn() as conn:
        cls = conn.execute(
            """
            SELECT category, issue_type, confidence, sentiment_score, sentiment_intensity
            FROM ticket_classifications
            WHERE ticket_key = 'INT-1'
            """
        ).fetchone()
    assert cls is not None, "Classification row missing — _save_classification may have failed"
    assert cls["category"] == "Access"
    assert cls["sentiment_score"] == 0.9, (
        "sentiment_score not written — feedback plugin may not have received the 'classified' event. "
        "Check that discover_plugins() can import all plugins without error (missing optional deps "
        "in CI can cause the event to be silently dropped, yielding a false green with NULL score)."
    )
    assert cls["sentiment_intensity"] == "high"

    # Step 2: feedback phase backfill is idempotent (no duplicate rewrites required).
    _phase_feedback()

    with get_db_conn() as conn:
        after = conn.execute(
            """
            SELECT sentiment_score, sentiment_intensity
            FROM ticket_classifications
            WHERE ticket_key = 'INT-1'
            """
        ).fetchone()
    assert after["sentiment_score"] == 0.9
    assert after["sentiment_intensity"] == "high"

    # Step 3: router consumes stored sentiment and escalates.
    result = route_ticket("INT-1")
    assert result.route == "human_review"
    assert "sentiment_intensity=high" in result.reason
