import logging
from unittest.mock import patch

import pytest

import config as config_mod
from db import init_db, get_db_conn


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr("db.DB_PATH", str(tmp_path / "test.db"))
    init_db()
    yield


def _insert_ticket(ticket_key: str, confidence: float | None = None) -> None:
    with get_db_conn() as conn:
        conn.execute(
            "INSERT INTO tickets (ticket_key, summary, description, status) VALUES (?, ?, ?, ?)",
            (ticket_key, "Summary", "This description is long enough to pass the content gate.", "Open"),
        )
        if confidence is not None:
            conn.execute(
                """
                INSERT INTO ticket_classifications (ticket_key, category, issue_type, confidence)
                VALUES (?, ?, ?, ?)
                """,
                (ticket_key, "Access", "Access request", confidence),
            )


def test__confidence_gate_skips_when_below_threshold(caplog):
    from plugins.responder.gates import _confidence_gate

    _insert_ticket("TEST-1", confidence=0.42)

    with caplog.at_level(logging.INFO):
        assert _confidence_gate("TEST-1", {"confidence_threshold": 0.6}) is False

    assert "TEST-1" in caplog.text
    assert "0.42" in caplog.text
    assert "0.60" in caplog.text


def test__confidence_gate_passes_at_threshold():
    from plugins.responder.gates import _confidence_gate

    _insert_ticket("TEST-5", confidence=0.6)

    assert _confidence_gate("TEST-5", {"confidence_threshold": 0.6}) is True


def test__confidence_gate_passes_above_threshold():
    from plugins.responder.gates import _confidence_gate

    _insert_ticket("TEST-6", confidence=0.91)

    assert _confidence_gate("TEST-6", {"confidence_threshold": 0.6}) is True


def test__confidence_gate_disabled_at_zero():
    from plugins.responder.gates import _confidence_gate

    _insert_ticket("TEST-2", confidence=0.01)

    assert _confidence_gate("TEST-2", {"confidence_threshold": 0.0}) is True


def test_get_confidence_threshold_reads_pipeline_yaml(tmp_path, monkeypatch):
    from core.pipeline import load_pipeline_config
    from plugins.responder.gates import _get_confidence_threshold

    pipeline_path = tmp_path / "pipeline.yaml"
    pipeline_path.write_text(
        "plugins:\n  responder:\n    enabled: true\n    confidence_threshold: 0.55\n"
    )
    monkeypatch.setenv("PIPELINE_CONFIG", str(pipeline_path))
    load_pipeline_config(pipeline_path)

    assert _get_confidence_threshold() == 0.55


def test__confidence_gate_fails_open_without_classification():
    from plugins.responder.gates import _confidence_gate

    _insert_ticket("TEST-3")

    assert _confidence_gate("TEST-3", {"confidence_threshold": 0.6}) is True


def test_handle_new_ticket_stops_before_router(monkeypatch):
    from plugins.responder import handle_new_ticket

    _insert_ticket("TEST-4", confidence=0.21)

    monkeypatch.setattr("plugins.responder._check_assignee_allowed", lambda ticket_key: (True, "agent-1"))
    monkeypatch.setattr("plugins.responder._check_status_gate", lambda ticket_key: True)
    monkeypatch.setattr("plugins.responder._check_age_gate", lambda ticket_key: True)
    monkeypatch.setattr("plugins.responder._check_content_gate", lambda ticket_key: True)
    monkeypatch.setattr(
        "core.pipeline.get_plugin_config",
        lambda plugin_name: {"confidence_threshold": 0.6} if plugin_name == "responder" else {},
    )

    route_called = False

    def _route_ticket(_ticket_key: str):
        nonlocal route_called
        route_called = True
        raise AssertionError("route_ticket should not be called when confidence gate fails")

    monkeypatch.setattr("analysis.router.route_ticket", _route_ticket)

    assert handle_new_ticket("TEST-4") is False
    assert route_called is False


@pytest.fixture(autouse=True)
def fresh_sentiment_db(tmp_path, monkeypatch):
    monkeypatch.setattr("db.DB_PATH", str(tmp_path / "test.db"))
    init_db()
    yield


def _seed_access_ticket(ticket_key: str, intensity: str | None):
    with get_db_conn() as conn:
        conn.execute(
            "INSERT INTO tickets (ticket_key, summary, description) VALUES (?, ?, ?)",
            (ticket_key, "Need access", "Please grant repo access"),
        )
        conn.execute(
            """
            INSERT INTO ticket_classifications
                (ticket_key, category, issue_type, confidence, sentiment_intensity, sentiment_score)
            VALUES (?, 'Access', 'Permission request', 0.9, ?, 0.85)
            """,
            (ticket_key, intensity),
        )


def _seed_classified_ticket(
    ticket_key: str,
    *,
    category: str = "Configuration",
    issue_type: str = "Bug report",
    sentiment_score: float | None = 0.85,
    sentiment_intensity: str | None = "high",
) -> None:
    with get_db_conn() as conn:
        conn.execute(
            "INSERT INTO tickets (ticket_key, summary, description) VALUES (?, ?, ?)",
            (ticket_key, "Help", "Please fix this broken application immediately"),
        )
        conn.execute(
            """
            INSERT INTO ticket_classifications
                (ticket_key, category, issue_type, confidence, sentiment_intensity, sentiment_score)
            VALUES (?, ?, ?, 0.9, ?, ?)
            """,
            (ticket_key, category, issue_type, sentiment_intensity, sentiment_score),
        )


class TestCustomerSentimentDraftGate:
    def test_disabled_when_threshold_zero(self):
        from plugins.responder.gates import _confidence_sentiment_draft_gate

        _seed_classified_ticket("S-1", sentiment_score=0.95)
        assert _confidence_sentiment_draft_gate("S-1", {"sentiment_gate_threshold": 0.0}) is True

    def test_blocks_when_score_exceeds_threshold(self, caplog):
        from plugins.responder.gates import _confidence_sentiment_draft_gate

        _seed_classified_ticket("S-2", sentiment_score=0.72, sentiment_intensity="high")

        with caplog.at_level(logging.INFO):
            assert _confidence_sentiment_draft_gate("S-2", {"sentiment_gate_threshold": 0.6}) is False

        assert "S-2" in caplog.text
        assert "0.720" in caplog.text
        assert "customer sentiment gate" in caplog.text

    def test_passes_at_threshold(self):
        from plugins.responder.gates import _confidence_sentiment_draft_gate

        _seed_classified_ticket("S-3", sentiment_score=0.6)
        assert _confidence_sentiment_draft_gate("S-3", {"sentiment_gate_threshold": 0.6}) is True

    def test_passes_below_threshold(self):
        from plugins.responder.gates import _confidence_sentiment_draft_gate

        _seed_classified_ticket("S-4", sentiment_score=0.41, sentiment_intensity="low")
        assert _confidence_sentiment_draft_gate("S-4", {"sentiment_gate_threshold": 0.6}) is True

    def test_fails_open_without_score(self):
        from plugins.responder.gates import _confidence_sentiment_draft_gate

        _seed_classified_ticket("S-5", sentiment_score=None, sentiment_intensity=None)
        assert _confidence_sentiment_draft_gate("S-5", {"sentiment_gate_threshold": 0.6}) is True

    def test_blocks_non_access_high_distress(self):
        """Distinct from _check_sentiment_gate — any category above score threshold."""
        from plugins.responder.gates import (
            _check_sentiment_gate,
            _confidence_sentiment_draft_gate,
        )

        _seed_classified_ticket(
            "S-6",
            category="Configuration",
            issue_type="Bug report",
            sentiment_score=0.9,
            sentiment_intensity="high",
        )

        with patch("plugins.responder.gates.get_plugin_config") as mock_cfg:
            mock_cfg.return_value = {"sentiment_escalation_threshold": "high"}
            assert _check_sentiment_gate("S-6") is True

        assert _confidence_sentiment_draft_gate("S-6", {"sentiment_gate_threshold": 0.6}) is False

    def test_get_sentiment_gate_threshold_reads_pipeline_yaml(self, tmp_path, monkeypatch):
        from core.pipeline import load_pipeline_config
        from plugins.responder.gates import _get_sentiment_gate_threshold

        pipeline_path = tmp_path / "pipeline.yaml"
        pipeline_path.write_text(
            "plugins:\n  responder:\n    enabled: true\n    sentiment_gate_threshold: 0.55\n"
        )
        monkeypatch.setenv("PIPELINE_CONFIG", str(pipeline_path))
        load_pipeline_config(pipeline_path)

        assert _get_sentiment_gate_threshold() == 0.55


class TestSentimentGate:
    @patch("plugins.responder.gates.get_plugin_config")
    def test_high_intensity_access_ticket_blocked(self, mock_cfg):
        from plugins.responder.gates import _check_sentiment_gate

        mock_cfg.return_value = {"sentiment_escalation_threshold": "high"}
        _seed_access_ticket("A-1", "high")
        assert _check_sentiment_gate("A-1") is False

    @patch("plugins.responder.gates.get_plugin_config")
    def test_low_intensity_access_ticket_passes(self, mock_cfg):
        from plugins.responder.gates import _check_sentiment_gate

        mock_cfg.return_value = {"sentiment_escalation_threshold": "high"}
        _seed_access_ticket("A-2", "low")
        assert _check_sentiment_gate("A-2") is True

    @patch("plugins.responder.gates.get_plugin_config")
    def test_high_intensity_non_access_passes(self, mock_cfg):
        from plugins.responder.gates import _check_sentiment_gate

        mock_cfg.return_value = {"sentiment_escalation_threshold": "high"}
        with get_db_conn() as conn:
            conn.execute(
                "INSERT INTO tickets (ticket_key, summary, description) VALUES ('C-1', 'Bug', 'App broken')"
            )
            conn.execute(
                """
                INSERT INTO ticket_classifications
                    (ticket_key, category, issue_type, confidence, sentiment_intensity, sentiment_score)
                VALUES ('C-1', 'Configuration', 'Bug report', 0.9, 'high', 0.9)
                """,
            )
        assert _check_sentiment_gate("C-1") is True

    @patch("plugins.responder.gates.get_plugin_config")
    def test_missing_intensity_fails_open(self, mock_cfg):
        from plugins.responder.gates import _check_sentiment_gate

        mock_cfg.return_value = {"sentiment_escalation_threshold": "high"}
        _seed_access_ticket("A-3", None)
        assert _check_sentiment_gate("A-3") is True

    @patch("plugins.responder.gates.get_plugin_config")
    def test_medium_threshold_blocks_high_only(self, mock_cfg):
        from plugins.responder.gates import _check_sentiment_gate

        mock_cfg.return_value = {"sentiment_escalation_threshold": "medium"}
        _seed_access_ticket("A-4", "medium")
        assert _check_sentiment_gate("A-4") is False

        _seed_access_ticket("A-5", "low")
        assert _check_sentiment_gate("A-5") is True

    @patch("plugins.responder.gates.get_plugin_config")
    def test_issue_type_access_match(self, mock_cfg):  # noqa: D102
        from plugins.responder.gates import _check_sentiment_gate

        mock_cfg.return_value = {"sentiment_escalation_threshold": "high"}
        with get_db_conn() as conn:
            conn.execute(
                "INSERT INTO tickets (ticket_key, summary, description) VALUES ('A-6', 'Help', 'Need help')"
            )
            conn.execute(
                """
                INSERT INTO ticket_classifications
                    (ticket_key, category, issue_type, confidence, sentiment_intensity, sentiment_score)
                VALUES ('A-6', 'Permissions', 'Access request', 0.9, 'high', 0.9)
                """,
            )
        assert _check_sentiment_gate("A-6") is False


class TestGateEdgeCases:
    def test_check_content_gate_invalid_regex_fails_open(self, monkeypatch):
        """Test E — invalid regex in AUTO_DRAFT_NOISE_PATTERNS must not raise; gate fails open."""
        from plugins.responder import gates
        from plugins.responder.gates import _check_content_gate

        # Inject a broken regex pattern alongside a valid one so we also exercise the
        # short-circuit: the invalid pattern must be skipped, not allowed to propagate.
        monkeypatch.setattr(config_mod, "AUTO_DRAFT_NOISE_PATTERNS", ["[invalid", r"\btest\b"])
        monkeypatch.setattr(gates, "AUTO_DRAFT_NOISE_PATTERNS", ["[invalid", r"\btest\b"])

        with get_db_conn() as conn:
            conn.execute(
                "INSERT INTO tickets (ticket_key, summary, description, status) "
                "VALUES ('EDGE-1', 'Normal summary', "
                "'This description is long enough to pass the content gate check.', 'Open')"
            )

        # Should return True (fail-open) rather than raising re.error.
        result = _check_content_gate("EDGE-1")
        assert result is True

    def test_check_age_gate_malformed_created_at_fails_open(self):
        """Test F — unparseable created_at must not raise; age gate fails open."""
        from plugins.responder.gates import _check_age_gate

        with get_db_conn() as conn:
            conn.execute(
                "INSERT INTO tickets (ticket_key, summary, description, status, created_at) "
                "VALUES ('EDGE-2', 'Summary', 'Description', 'Open', 'not-a-date')"
            )

        # Should return True (fail-open) rather than raising ValueError.
        result = _check_age_gate("EDGE-2")
        assert result is True
