"""Tests for plugins/alerting/anomaly_alert.py and opsgenie.py — ANTSE-315."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from plugins.alerting.anomaly import AnomalyResult

FIXED_NOW = datetime(2026, 6, 3, 14, 30, tzinfo=timezone.utc)


def _make_result(**kwargs) -> AnomalyResult:
    defaults = {
        "is_anomaly": True,
        "zscore": 7.5,
        "segment": "tue_business",
        "ticket_count": 42,
        "rolling_mean": 10.0,
        "rolling_std": 2.5,
        "category_breakdown": {
            "Access": 15,
            "Login": 10,
            "Bot": 5,
            "Other": 2,
        },
    }
    defaults.update(kwargs)
    return AnomalyResult(**defaults)


class TestFireAnomalyAlert:
    @patch("plugins.alerting.anomaly_alert.opsgenie.post_alert")
    @patch("plugins.alerting.anomaly_alert.datetime")
    def test_fires_alert_with_correct_payload(self, mock_dt, mock_post):
        mock_dt.now.return_value = FIXED_NOW
        mock_post.return_value = True

        from plugins.alerting.anomaly_alert import fire_anomaly_alert

        assert fire_anomaly_alert(_make_result()) is True

        mock_post.assert_called_once()
        kwargs = mock_post.call_args.kwargs
        assert kwargs["message"] == "Volume anomaly: 42 tickets in tue_business (z=7.5)"
        assert kwargs["alias"] == "anomaly-tue_business-20260603T14"
        assert kwargs["priority"] == "P2"
        assert kwargs["tags"] == ["alerting", "anomaly", "volume", "ai-helpdesk"]
        assert "Segment: tue_business" in kwargs["description"]
        assert "Ticket count: 42" in kwargs["description"]
        assert "Z-score: 7.5" in kwargs["description"]
        assert "Rolling mean: 10.0000" in kwargs["description"]
        assert "Rolling std: 2.5000" in kwargs["description"]
        assert "Access: 15" in kwargs["description"]
        assert "Login: 10" in kwargs["description"]
        assert "Bot: 5" in kwargs["description"]
        assert "Other: 2" not in kwargs["description"]

    @patch("plugins.alerting.anomaly_alert.opsgenie.post_alert")
    @patch("plugins.alerting.anomaly_alert.datetime")
    def test_alias_hourly_dedup_same_segment(self, mock_dt, mock_post):
        mock_dt.now.return_value = FIXED_NOW
        mock_post.return_value = True

        from plugins.alerting.anomaly_alert import fire_anomaly_alert

        fire_anomaly_alert(_make_result(ticket_count=42))
        fire_anomaly_alert(_make_result(ticket_count=50))

        aliases = [call.kwargs["alias"] for call in mock_post.call_args_list]
        assert aliases == [
            "anomaly-tue_business-20260603T14",
            "anomaly-tue_business-20260603T14",
        ]

    @patch("plugins.alerting.anomaly_alert.opsgenie.post_alert")
    @patch("plugins.alerting.anomaly_alert.datetime")
    def test_alias_differs_by_segment(self, mock_dt, mock_post):
        mock_dt.now.return_value = FIXED_NOW
        mock_post.return_value = True

        from plugins.alerting.anomaly_alert import fire_anomaly_alert

        fire_anomaly_alert(_make_result(segment="tue_business"))
        fire_anomaly_alert(_make_result(segment="wed_business"))

        aliases = [call.kwargs["alias"] for call in mock_post.call_args_list]
        assert aliases[0] == "anomaly-tue_business-20260603T14"
        assert aliases[1] == "anomaly-wed_business-20260603T14"

    @patch("plugins.alerting.anomaly_alert.opsgenie.post_alert")
    @patch("plugins.alerting.anomaly_alert.datetime")
    def test_alias_differs_by_hour(self, mock_dt, mock_post):
        mock_post.return_value = True

        from plugins.alerting.anomaly_alert import fire_anomaly_alert

        mock_dt.now.return_value = datetime(2026, 6, 3, 14, 0, tzinfo=timezone.utc)
        fire_anomaly_alert(_make_result())
        mock_dt.now.return_value = datetime(2026, 6, 3, 15, 0, tzinfo=timezone.utc)
        fire_anomaly_alert(_make_result())

        aliases = [call.kwargs["alias"] for call in mock_post.call_args_list]
        assert aliases[0] == "anomaly-tue_business-20260603T14"
        assert aliases[1] == "anomaly-tue_business-20260603T15"

    @pytest.mark.parametrize(
        ("zscore", "priority"),
        [
            (11.0, "P1"),
            (10.0, "P2"),
            (6.0, "P2"),
            (5.0, "P3"),
            (3.0, "P3"),
        ],
    )
    @patch("plugins.alerting.anomaly_alert.opsgenie.post_alert")
    @patch("plugins.alerting.anomaly_alert.datetime")
    def test_priority_tiers(self, mock_dt, mock_post, zscore, priority):
        mock_dt.now.return_value = FIXED_NOW
        mock_post.return_value = True

        from plugins.alerting.anomaly_alert import fire_anomaly_alert

        fire_anomaly_alert(_make_result(zscore=zscore))
        assert mock_post.call_args.kwargs["priority"] == priority


class TestOpsGenieClient:
    @patch("plugins.alerting.opsgenie.requests.post")
    @patch("plugins.alerting.opsgenie.OPSGENIE_API_KEY", "test-key")
    def test_post_alert_success(self, mock_post):
        mock_post.return_value = MagicMock(status_code=202)

        from plugins.alerting.opsgenie import post_alert

        assert post_alert(
            message="Volume anomaly",
            alias="anomaly-tue_business-20260603T14",
            description="details",
            priority="P2",
            tags=["alerting"],
        ) is True

        call_kwargs = mock_post.call_args.kwargs
        assert call_kwargs["json"] == {
            "message": "Volume anomaly",
            "priority": "P2",
            "alias": "anomaly-tue_business-20260603T14",
            "description": "details",
            "tags": ["alerting"],
            "source": "ai-helpdesk-alerting",
        }
        assert call_kwargs["headers"]["Authorization"] == "GenieKey test-key"

    @patch("plugins.alerting.opsgenie.requests.post")
    @patch("plugins.alerting.opsgenie.OPSGENIE_API_KEY", "")
    def test_post_alert_skips_without_api_key(self, mock_post):
        from plugins.alerting.opsgenie import post_alert

        assert post_alert(message="test") is False
        mock_post.assert_not_called()

    @patch("plugins.alerting.opsgenie.requests.post")
    @patch("plugins.alerting.opsgenie.OPSGENIE_API_KEY", "test-key")
    def test_post_alert_handles_http_error(self, mock_post):
        import requests as req_lib
        mock_response = MagicMock(status_code=429, text="rate limited")
        mock_response.raise_for_status.side_effect = req_lib.HTTPError("429")
        mock_post.return_value = mock_response

        from plugins.alerting.opsgenie import post_alert

        assert post_alert(message="test") is False

    @patch("plugins.alerting.opsgenie.requests.post")
    @patch("plugins.alerting.opsgenie.OPSGENIE_API_KEY", "test-key")
    def test_post_alert_handles_request_exception(self, mock_post):
        import requests

        mock_post.side_effect = requests.RequestException("timeout")

        from plugins.alerting.opsgenie import post_alert

        assert post_alert(message="test") is False


class TestAlertingPluginWiring:
    def test_on_schedule_fires_alert_when_anomaly(self):
        result = _make_result(is_anomaly=True)
        mock_detector = MagicMock()
        mock_detector.score.return_value = result

        with (
            patch("plugins.alerting.get_plugin_config", return_value={"anomaly_alerts_enabled": True}),
            patch("plugins.alerting.OPSGENIE_API_KEY", "test-key"),
            patch("plugins.alerting.get_last_run_date", return_value=datetime.now().date()),
            patch("plugins.alerting.ControlChartDetector", return_value=mock_detector),
            patch("plugins.alerting.TopicClusterer"),
            patch("plugins.alerting.fire_anomaly_alert", return_value=True) as mock_fire,
            patch("plugins.alerting.get_db_conn") as mock_conn,
        ):
            mock_conn.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_conn.return_value.__exit__ = MagicMock(return_value=False)

            from plugins.alerting import AlertingPlugin

            AlertingPlugin().on_schedule()

        mock_fire.assert_called_once_with(result)

    def test_on_schedule_skips_alert_when_disabled(self):
        mock_detector = MagicMock()
        mock_detector.score.return_value = _make_result(is_anomaly=True)

        with (
            patch("plugins.alerting.get_plugin_config", return_value={"anomaly_alerts_enabled": False}),
            patch("plugins.alerting.OPSGENIE_API_KEY", "test-key"),
            patch("plugins.alerting.get_last_run_date", return_value=datetime.now().date()),
            patch("plugins.alerting.ControlChartDetector", return_value=mock_detector),
            patch("plugins.alerting.TopicClusterer"),
            patch("plugins.alerting.fire_anomaly_alert") as mock_fire,
            patch("plugins.alerting.get_db_conn") as mock_conn,
        ):
            mock_conn.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_conn.return_value.__exit__ = MagicMock(return_value=False)

            from plugins.alerting import AlertingPlugin

            AlertingPlugin().on_schedule()

        mock_fire.assert_not_called()

    def test_on_schedule_skips_alert_without_api_key(self):
        mock_detector = MagicMock()
        mock_detector.score.return_value = _make_result(is_anomaly=True)

        with (
            patch("plugins.alerting.get_plugin_config", return_value={"anomaly_alerts_enabled": True}),
            patch("plugins.alerting.OPSGENIE_API_KEY", ""),
            patch("plugins.alerting.get_last_run_date", return_value=datetime.now().date()),
            patch("plugins.alerting.ControlChartDetector", return_value=mock_detector),
            patch("plugins.alerting.TopicClusterer"),
            patch("plugins.alerting.fire_anomaly_alert") as mock_fire,
            patch("plugins.alerting.get_db_conn") as mock_conn,
        ):
            mock_conn.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_conn.return_value.__exit__ = MagicMock(return_value=False)

            from plugins.alerting import AlertingPlugin

            AlertingPlugin().on_schedule()

        mock_fire.assert_not_called()
