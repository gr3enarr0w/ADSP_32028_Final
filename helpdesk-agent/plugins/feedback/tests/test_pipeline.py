"""Tests for plugins/feedback/pipeline.py — ANTSE-323."""

from unittest.mock import patch

import pytest

from db import get_db_conn, init_db


@pytest.fixture(autouse=True)
def use_memory_db(tmp_path):
    db_path = str(tmp_path / "test.db")
    with patch("db.DB_PATH", db_path):
        init_db()
        yield


def _seed_ticket(ticket_key: str, description: str = "Need access to the repo", reporter_id: str = "cust-1"):
    with get_db_conn() as conn:
        conn.execute(
            """
            INSERT INTO tickets (ticket_key, summary, description, reporter_id)
            VALUES (?, ?, ?, ?)
            """,
            (ticket_key, "Access request", description, reporter_id),
        )


def _seed_classification(ticket_key: str, category: str = "Access"):
    with get_db_conn() as conn:
        conn.execute(
            """
            INSERT INTO ticket_classifications
                (ticket_key, category, issue_type, confidence, classified_at)
            VALUES (?, ?, ?, ?, datetime('now'))
            """,
            (ticket_key, category, "Permission request", 0.85),
        )


class TestScoreToIntensity:
    def test_high_at_threshold(self):
        from plugins.feedback.pipeline import score_to_intensity

        assert score_to_intensity(0.6, 0.6) == "high"

    def test_medium_between_half_and_threshold(self):
        from plugins.feedback.pipeline import score_to_intensity

        assert score_to_intensity(0.35, 0.6) == "medium"

    def test_low_below_medium(self):
        from plugins.feedback.pipeline import score_to_intensity

        assert score_to_intensity(0.1, 0.6) == "low"


class TestBuildSentimentText:
    def test_concatenates_description_and_latest_customer_comment(self):
        from plugins.feedback.pipeline import build_sentiment_text

        _seed_ticket("T-1", "Original description text")
        with get_db_conn() as conn:
            conn.execute(
                """
                INSERT INTO ticket_comments
                    (comment_id, ticket_key, author_id, body, is_public, created_at)
                VALUES ('1', 'T-1', 'agent-1', 'Agent reply', 1, '2026-01-01T10:00:00'),
                       ('2', 'T-1', 'cust-1', 'Customer follow-up', 1, '2026-01-02T10:00:00')
                """,
            )

        text = build_sentiment_text("T-1")
        assert "Original description text" in text
        assert "Customer follow-up" in text
        assert "Agent reply" not in text


class TestScoreClassification:
    @patch("plugins.feedback.pipeline.score_ticket")
    def test_writes_score_and_intensity(self, mock_score):
        from plugins.feedback.pipeline import score_classification

        mock_score.return_value = {"label": "NEGATIVE", "score": 0.9, "intensity": 0.82}
        _seed_ticket("T-2")
        _seed_classification("T-2")

        assert score_classification("T-2") is True

        with get_db_conn() as conn:
            row = conn.execute(
                "SELECT sentiment_score, sentiment_intensity FROM ticket_classifications WHERE ticket_key = 'T-2'"
            ).fetchone()
        assert row["sentiment_score"] == 0.82
        assert row["sentiment_intensity"] == "high"

    @patch("plugins.feedback.pipeline.score_ticket")
    def test_skips_already_scored(self, mock_score):
        from plugins.feedback.pipeline import score_classification

        _seed_ticket("T-3")
        _seed_classification("T-3")
        with get_db_conn() as conn:
            conn.execute(
                "UPDATE ticket_classifications SET sentiment_intensity = 'low' WHERE ticket_key = 'T-3'"
            )

        assert score_classification("T-3") is False
        mock_score.assert_not_called()

    def test_skips_without_classification(self):
        from plugins.feedback.pipeline import score_classification

        _seed_ticket("T-4")
        assert score_classification("T-4") is False
