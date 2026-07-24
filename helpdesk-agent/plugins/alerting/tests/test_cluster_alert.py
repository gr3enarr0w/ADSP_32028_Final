"""Tests for plugins/alerting/cluster_alert.py and opsgenie.py — ANTSE-319."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from db import get_db_conn, init_db
from plugins.alerting.clustering import DeltaResult


@pytest.fixture(autouse=True)
def use_memory_db(tmp_path):
    db_path = str(tmp_path / "test.db")
    with patch("db.DB_PATH", db_path):
        init_db()
        yield


def _insert_ticket(conn, ticket_key: str, summary: str):
    conn.execute(
        """
        INSERT INTO tickets (ticket_key, summary, description, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (ticket_key, summary, "desc", datetime.now(timezone.utc).isoformat()),
    )


class TestFilterAlertDeltas:
    def test_new_cluster_qualifies(self):
        from plugins.alerting.cluster_alert import filter_alert_deltas

        deltas = [
            DeltaResult(
                cluster_id=0,
                label="access",
                size=5,
                is_new=True,
                growth_rate=1.0,
                ticket_keys=["A-1"],
            )
        ]
        assert len(filter_alert_deltas(deltas, 0.3)) == 1

    def test_growing_cluster_qualifies(self):
        from plugins.alerting.cluster_alert import filter_alert_deltas

        deltas = [
            DeltaResult(
                cluster_id=1,
                label="bots",
                size=8,
                is_new=False,
                growth_rate=0.5,
                ticket_keys=["B-1"],
            )
        ]
        assert len(filter_alert_deltas(deltas, 0.3)) == 1

    def test_stable_cluster_excluded(self):
        from plugins.alerting.cluster_alert import filter_alert_deltas

        deltas = [
            DeltaResult(
                cluster_id=2,
                label="stable",
                size=4,
                is_new=False,
                growth_rate=0.1,
                ticket_keys=["C-1"],
            )
        ]
        assert filter_alert_deltas(deltas, 0.3) == []


class TestLookupSummaries:
    def test_fetches_summaries_in_key_order(self):
        from plugins.alerting.cluster_alert import lookup_summaries

        with get_db_conn() as conn:
            _insert_ticket(conn, "T-2", "Second issue")
            _insert_ticket(conn, "T-1", "First issue")
            summaries = lookup_summaries(conn, ["T-1", "T-2"])

        assert len(summaries) == 2
        assert "T-1" in summaries[0]
        assert "T-2" in summaries[1]


class TestBuildClusterAlert:
    def test_includes_topic_velocity_and_keys(self):
        from plugins.alerting.cluster_alert import build_cluster_alert

        delta = DeltaResult(
            cluster_id=3,
            label="access / login",
            size=6,
            is_new=False,
            growth_rate=0.75,
            ticket_keys=["K-1", "K-2"],
        )
        message, description, alias, details = build_cluster_alert(
            delta,
            "OAuth Login Failures",
            ["- K-1: Cannot log in"],
            run_date="2026-06-03",
        )
        assert "OAuth Login Failures" in message
        assert "6 tickets" in message
        assert "+75% growth" in message
        assert "K-1" in description
        assert alias == "ai-helpdesk-cluster-3-2026-06-03"
        assert details["growth_rate"] == "0.75"

    def test_truncates_long_message(self):
        from plugins.alerting.cluster_alert import build_cluster_alert

        delta = DeltaResult(
            cluster_id=4,
            label="long",
            size=10,
            is_new=True,
            growth_rate=1.0,
            ticket_keys=["L-1"],
        )
        long_topic = "A" * 150
        message, _, _, _ = build_cluster_alert(
            delta,
            long_topic,
            [],
            run_date="2026-06-03",
        )
        assert len(message) <= 130
        assert message.endswith("...")


class TestProcessClusterAlerts:
    def test_sends_for_qualifying_delta(self):
        from plugins.alerting.cluster_alert import process_cluster_alerts

        deltas = [
            DeltaResult(
                cluster_id=0,
                label="access",
                size=3,
                is_new=True,
                growth_rate=1.0,
                ticket_keys=["CL-1"],
            ),
            DeltaResult(
                cluster_id=1,
                label="stable",
                size=2,
                is_new=False,
                growth_rate=0.0,
                ticket_keys=["CL-2"],
            ),
        ]
        cfg = {
            "cluster_alerts_enabled": True,
            "opsgenie_enabled": True,
            "opsgenie_priority": "P3",
        }

        with get_db_conn() as conn:
            _insert_ticket(conn, "CL-1", "Login broken for SSO users")
            with patch("plugins.alerting.opsgenie.OPSGENIE_API_KEY", "test-key"):
                with patch(
                    "plugins.alerting.cluster_alert.enrich_topic_label",
                    return_value="SSO Login Failures",
                ):
                    with patch(
                        "plugins.alerting.cluster_alert.OpsGenieClient.create_alert",
                        return_value={"requestId": "req-1"},
                    ) as mock_alert:
                        sent = process_cluster_alerts(
                            conn,
                            deltas,
                            velocity_threshold=0.3,
                            run_date="2026-06-03",
                            cfg=cfg,
                        )

        assert sent == 1
        mock_alert.assert_called_once()
        call_kwargs = mock_alert.call_args.kwargs
        assert "SSO Login Failures" in call_kwargs["message"]
        assert call_kwargs["alias"] == "ai-helpdesk-cluster-0-2026-06-03"

    def test_skips_when_opsgenie_disabled(self):
        from plugins.alerting.cluster_alert import process_cluster_alerts

        deltas = [
            DeltaResult(
                cluster_id=0,
                label="access",
                size=3,
                is_new=True,
                growth_rate=1.0,
                ticket_keys=["CL-1"],
            )
        ]
        with get_db_conn() as conn:
            with patch(
                "plugins.alerting.cluster_alert.OpsGenieClient.create_alert",
            ) as mock_alert:
                sent = process_cluster_alerts(
                    conn,
                    deltas,
                    velocity_threshold=0.3,
                    run_date="2026-06-03",
                    cfg={"opsgenie_enabled": False},
                )
        assert sent == 0
        mock_alert.assert_not_called()


class TestOpsGenieClient:
    def test_create_alert_posts_to_api(self):
        from plugins.alerting.opsgenie import OPSGENIE_ALERTS_URL, OpsGenieClient

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"requestId": "abc"}
        mock_resp.raise_for_status = MagicMock()

        with patch("plugins.alerting.opsgenie.requests.post", return_value=mock_resp) as mock_post:
            client = OpsGenieClient("secret-key")
            result = client.create_alert(
                message="Test alert",
                description="Details here",
                alias="test-alias",
            )

        assert result == {"requestId": "abc"}
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == OPSGENIE_ALERTS_URL
        assert kwargs["headers"]["Authorization"] == "GenieKey secret-key"
        assert kwargs["json"]["message"] == "Test alert"

    def test_create_alert_noop_without_key(self):
        from plugins.alerting.opsgenie import OpsGenieClient

        with patch("plugins.alerting.opsgenie.requests.post") as mock_post:
            client = OpsGenieClient("")
            assert client.create_alert(message="x", description="y") is None
        mock_post.assert_not_called()
