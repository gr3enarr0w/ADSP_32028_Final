"""Unit tests for plugins/feedback/csat.py — ANTSE-321."""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from db import get_db_conn, init_db


@pytest.fixture(autouse=True)
def use_memory_db(tmp_path):
    """Use a temporary DB for each test."""
    db_path = str(tmp_path / "test.db")
    with patch("db.DB_PATH", db_path):
        init_db()
        yield db_path


def _insert_resolved_ticket(conn, ticket_key: str, days_ago: int = 1):
    resolved_at = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    conn.execute(
        """
        INSERT INTO tickets (ticket_key, summary, status, resolution, resolved_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (ticket_key, "Test ticket", "Done", "Fixed", resolved_at),
    )
    conn.commit()


class TestParseFeedbackPayload:
    def test_parses_rating_and_comment_object(self):
        from plugins.feedback.csat import parse_feedback_payload

        rating, comment, submitted_at = parse_feedback_payload({
            "rating": 5,
            "comment": {"body": "Great support"},
            "createdDate": {"iso8601": "2026-01-15T10:00:00+0000"},
        })
        assert rating == 5
        assert comment == "Great support"
        assert submitted_at == "2026-01-15T10:00:00+0000"

    def test_parses_string_comment(self):
        from plugins.feedback.csat import parse_feedback_payload

        rating, comment, submitted_at = parse_feedback_payload({
            "score": 3,
            "comment": "OK",
            "submittedDate": "2026-01-15T10:00:00+0000",
        })
        assert rating == 3
        assert comment == "OK"
        assert submitted_at == "2026-01-15T10:00:00+0000"


class TestFetchTicketFeedback:
    def test_404_returns_none(self):
        from plugins.feedback.csat import fetch_ticket_feedback

        mock_resp = MagicMock(status_code=404, text="Not found")
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        with patch("plugins.feedback.csat.get_cloud_auth", return_value={"Authorization": "Bearer x"}):
            with patch("plugins.feedback.csat.get_cloud_base_url", return_value="https://api.example.com"):
                result = fetch_ticket_feedback("TEST-1", session=mock_session)

        assert result is None

    def test_403_returns_none(self):
        from plugins.feedback.csat import fetch_ticket_feedback

        mock_resp = MagicMock(status_code=403, text="Forbidden")
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        with patch("plugins.feedback.csat.get_cloud_auth", return_value={"Authorization": "Bearer x"}):
            with patch("plugins.feedback.csat.get_cloud_base_url", return_value="https://api.example.com"):
                result = fetch_ticket_feedback("TEST-1", session=mock_session)

        assert result is None

    def test_200_returns_json(self):
        from plugins.feedback.csat import fetch_ticket_feedback

        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {"rating": 4, "comment": {"body": "Good"}}
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        with patch("plugins.feedback.csat.get_cloud_auth", return_value={"Authorization": "Bearer x"}):
            with patch("plugins.feedback.csat.get_cloud_base_url", return_value="https://api.example.com"):
                result = fetch_ticket_feedback("TEST-1", session=mock_session)

        assert result == {"rating": 4, "comment": {"body": "Good"}}


class TestIngestCsat:
    def test_ingests_feedback_for_resolved_tickets(self):
        from plugins.feedback.csat import ingest_csat

        # No ticket_classification row needed — CSAT ingestion reads only the tickets table.
        with get_db_conn() as conn:
            _insert_resolved_ticket(conn, "TEST-100")

        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {
            "rating": 5,
            "comment": {"body": "Excellent"},
            "createdDate": {"iso8601": "2026-01-20T12:00:00+0000"},
        }
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        with patch("plugins.feedback.csat.get_cloud_auth", return_value={"Authorization": "Bearer x"}):
            with patch("plugins.feedback.csat.get_cloud_base_url", return_value="https://api.example.com"):
                with patch("plugins.feedback.csat.requests.Session", return_value=mock_session):
                    with get_db_conn() as conn:
                        stats = ingest_csat(conn)

        assert stats["checked"] == 1
        assert stats["found"] == 1
        assert stats["errors"] == 0

        with get_db_conn() as conn:
            row = conn.execute(
                "SELECT csat_score, csat_comment FROM ticket_csat WHERE ticket_key = ?",
                ("TEST-100",),
            ).fetchone()
        assert row["csat_score"] == 5
        assert row["csat_comment"] == "Excellent"

    def test_skips_404_silently(self):
        from plugins.feedback.csat import ingest_csat

        with get_db_conn() as conn:
            _insert_resolved_ticket(conn, "TEST-404")

        mock_resp = MagicMock(status_code=404, text="Not found")
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        with patch("plugins.feedback.csat.get_cloud_auth", return_value={"Authorization": "Bearer x"}):
            with patch("plugins.feedback.csat.get_cloud_base_url", return_value="https://api.example.com"):
                with patch("plugins.feedback.csat.requests.Session", return_value=mock_session):
                    with get_db_conn() as conn:
                        stats = ingest_csat(conn)

        assert stats["checked"] == 1
        assert stats["found"] == 0
        assert stats["errors"] == 0

        with get_db_conn() as conn:
            count = conn.execute("SELECT COUNT(*) FROM ticket_csat").fetchone()[0]
        assert count == 0

    def test_upserts_on_conflict(self):
        from plugins.feedback.csat import ingest_csat, upsert_csat

        with get_db_conn() as conn:
            _insert_resolved_ticket(conn, "TEST-200")
            upsert_csat(conn, "TEST-200", 3, "Old comment", "2026-01-01T00:00:00+0000")
            conn.commit()

        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {
            "rating": 5,
            "comment": {"body": "Updated"},
            "createdDate": {"iso8601": "2026-01-20T12:00:00+0000"},
        }
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        with patch("plugins.feedback.csat.get_cloud_auth", return_value={"Authorization": "Bearer x"}):
            with patch("plugins.feedback.csat.get_cloud_base_url", return_value="https://api.example.com"):
                with patch("plugins.feedback.csat.requests.Session", return_value=mock_session):
                    with get_db_conn() as conn:
                        stats = ingest_csat(conn)

        assert stats["found"] == 1

        with get_db_conn() as conn:
            row = conn.execute(
                "SELECT csat_score, csat_comment FROM ticket_csat WHERE ticket_key = ?",
                ("TEST-200",),
            ).fetchone()
        assert row["csat_score"] == 5
        assert row["csat_comment"] == "Updated"

    def test_excludes_tickets_outside_lookback(self):
        from plugins.feedback.csat import get_recently_resolved_ticket_keys

        with get_db_conn() as conn:
            _insert_resolved_ticket(conn, "TEST-OLD", days_ago=30)
            _insert_resolved_ticket(conn, "TEST-NEW", days_ago=2)
            keys = get_recently_resolved_ticket_keys(conn)

        assert "TEST-NEW" in keys
        assert "TEST-OLD" not in keys



class TestFeedbackPluginSchedule:
    def test_on_schedule_runs_once_per_day(self):
        from plugins.feedback import plugin

        with patch("plugins.feedback.ingest_csat", return_value={"checked": 0, "found": 0, "errors": 0}) as mock_ingest:
            with patch(
                "plugins.feedback.correlator.compute_category_correlations",
                return_value=[],
            ):
                with patch("plugins.feedback.score_unscored_classifications", return_value=(0, 0)):
                    plugin.on_schedule()
                    plugin.on_schedule()

        assert mock_ingest.call_count == 1

    def test_uses_oauth_not_hardcoded_credentials(self):
        from plugins.feedback.csat import fetch_ticket_feedback

        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {"rating": 1}
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        with patch("plugins.feedback.csat.get_cloud_auth", return_value={"Authorization": "Bearer token"}) as mock_auth:
            with patch("plugins.feedback.csat.get_cloud_base_url", return_value="https://api.atlassian.com/ex/jira/cloud-id"):
                fetch_ticket_feedback("TEST-1", session=mock_session)

        mock_auth.assert_called_once_with("jsm")
        call_kwargs = mock_session.get.call_args
        assert call_kwargs.kwargs["headers"]["Authorization"] == "Bearer token"

    def test_sends_experimental_api_header(self):
        from plugins.feedback.csat import fetch_ticket_feedback

        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {"rating": 3}
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        with patch("plugins.feedback.csat.get_cloud_auth", return_value={"Authorization": "Bearer x"}):
            with patch("plugins.feedback.csat.get_cloud_base_url", return_value="https://api.example.com"):
                fetch_ticket_feedback("TEST-1", session=mock_session)

        headers_sent = mock_session.get.call_args.kwargs["headers"]
        assert headers_sent["X-ExperimentalApi"] == "opt-in"
