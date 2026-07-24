"""Tests for plugins/alerting/anomaly.py — ANTSE-314."""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from db import get_db_conn, init_db


@pytest.fixture(autouse=True)
def use_memory_db(tmp_path):
    db_path = str(tmp_path / "test.db")
    with patch("db.DB_PATH", db_path):
        init_db()
        yield


def _insert_ticket(conn, ticket_key: str, created_at: str, category: str = "Access"):
    conn.execute(
        """
        INSERT INTO tickets (ticket_key, summary, description, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (ticket_key, f"Summary {ticket_key}", f"Description for {ticket_key}", created_at),
    )
    conn.execute(
        """
        INSERT INTO ticket_classifications (ticket_key, category, issue_type, confidence)
        VALUES (?, ?, ?, ?)
        """,
        (ticket_key, category, "Request", 0.9),
    )


def _seed_baseline_history(conn, start: datetime, days: int, tickets_per_hour: int = 1):
    """Seed normal weekday business-hour traffic."""
    for day_offset in range(days):
        day = start + timedelta(days=day_offset)
        if day.weekday() >= 5:
            continue
        for hour in range(8, 18):
            ts = day.replace(hour=hour, minute=30, second=0, microsecond=0, tzinfo=timezone.utc)
            for i in range(tickets_per_hour):
                key = f"N-{day.strftime('%Y%m%d')}-{hour}-{i}"
                _insert_ticket(conn, key, ts.isoformat())


def _seed_spike(conn, day: str, hour: int, count: int):
    base = datetime.fromisoformat(f"{day}T{hour:02d}:00:00+00:00")
    for i in range(count):
        offset = timedelta(minutes=i % 60)
        ts = (base + offset).isoformat()
        _insert_ticket(conn, f"S-{day.replace('-', '')}-{hour}-{i}", ts)


class TestSegmentKey:
    def test_tuesday_business(self):
        from plugins.alerting.anomaly import segment_key

        dt = datetime(2026, 3, 17, 14, 0, tzinfo=timezone.utc)
        assert segment_key(dt) == "tue_business"

    def test_saturday_early(self):
        from plugins.alerting.anomaly import hour_bucket, segment_key

        dt = datetime(2026, 3, 14, 3, 0, tzinfo=timezone.utc)
        assert hour_bucket(3) == "early"
        assert segment_key(dt) == "sat_early"


class TestControlChartDetector:
    def test_build_baseline_writes_segments(self):
        from plugins.alerting.anomaly import ControlChartDetector

        ref = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
        with get_db_conn() as conn:
            _seed_baseline_history(conn, ref - timedelta(days=90), 90, tickets_per_hour=2)
            detector = ControlChartDetector(sigma=2.0)
            written = detector.build_baseline(conn, reference_time=ref)

        assert written > 0
        with get_db_conn() as conn:
            rows = conn.execute("SELECT segment FROM anomaly_baseline").fetchall()
        segments = {r["segment"] for r in rows}
        assert "tue_business" in segments

    def test_score_detects_spike(self):
        from plugins.alerting.anomaly import ControlChartDetector

        ref = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
        with get_db_conn() as conn:
            _seed_baseline_history(conn, ref - timedelta(days=90), 90, tickets_per_hour=1)
            _seed_spike(conn, "2026-03-16", 14, 40)
            detector = ControlChartDetector(sigma=2.0)
            detector.build_baseline(conn, reference_time=ref)
            result = detector.score(
                conn,
                window_minutes=60,
                reference_time=datetime(2026, 3, 16, 15, 0, tzinfo=timezone.utc),
            )

        assert result.is_anomaly
        assert result.zscore > 2.0
        assert result.ticket_count >= 30

    def test_score_normal_window(self):
        from plugins.alerting.anomaly import ControlChartDetector

        ref = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
        with get_db_conn() as conn:
            _seed_baseline_history(conn, ref - timedelta(days=90), 90, tickets_per_hour=1)
            detector = ControlChartDetector(sigma=2.0)
            detector.build_baseline(conn, reference_time=ref)
            result = detector.score(
                conn,
                window_minutes=60,
                reference_time=datetime(2026, 4, 8, 10, 0, tzinfo=timezone.utc),
            )

        assert not result.is_anomaly

    def test_baseline_includes_zero_volume_hours(self):
        from plugins.alerting.anomaly import ControlChartDetector

        ref = datetime(2026, 5, 12, 12, 0, tzinfo=timezone.utc)  # Tuesday
        with get_db_conn() as conn:
            _insert_ticket(conn, "SPARSE-1", "2026-05-12T09:15:00+00:00")
            detector = ControlChartDetector(sigma=2.0)
            detector.build_baseline(conn, reference_time=ref)
            row = conn.execute(
                "SELECT rolling_mean FROM anomaly_baseline WHERE segment = ?",
                ("tue_business",),
            ).fetchone()

        # Tue/business has 10 hourly slots per week; one ticket should average close to 0.1.
        assert row is not None
        assert row["rolling_mean"] < 0.2

    def test_score_rejects_non_hour_windows(self):
        from plugins.alerting.anomaly import ControlChartDetector

        ref = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
        with get_db_conn() as conn:
            _seed_baseline_history(conn, ref - timedelta(days=90), 90, tickets_per_hour=1)
            detector = ControlChartDetector(sigma=2.0)
            detector.build_baseline(conn, reference_time=ref)
            with pytest.raises(ValueError):
                detector.score(conn, window_minutes=120, reference_time=ref)

    def test_store_score_persists(self):
        from plugins.alerting.anomaly import AnomalyResult, ControlChartDetector

        detector = ControlChartDetector(sigma=2.0)
        result = AnomalyResult(is_anomaly=True, zscore=3.5, segment="tue_business", ticket_count=10)
        with get_db_conn() as conn:
            detector.store_score(conn, result, window_minutes=60)
            row = conn.execute("SELECT is_anomaly, zscore FROM anomaly_scores").fetchone()
        assert row["is_anomaly"] == 1
        assert row["zscore"] == 3.5


class TestValidationMetrics:
    """TPR > 80% on labeled windows; FPR < 5% on 20 normal windows."""

    def test_tpr_and_fpr_targets(self):
        from plugins.alerting.anomaly import ControlChartDetector, validate_labeled_windows

        ref = datetime(2026, 6, 2, 12, 0, tzinfo=timezone.utc)
        with get_db_conn() as conn:
            _seed_baseline_history(conn, ref - timedelta(days=120), 120, tickets_per_hour=1)
            _seed_spike(conn, "2026-03-16", 14, 50)
            _seed_spike(conn, "2026-01-06", 10, 20)
            detector = ControlChartDetector(sigma=2.0)
            detector.build_baseline(conn, reference_time=ref)
            metrics = validate_labeled_windows(conn, detector)

        assert metrics["tpr"] > 0.80, f"TPR {metrics['tpr']} below 80% target"
        assert metrics["fpr"] < 0.05, f"FPR {metrics['fpr']} above 5% target"

    def test_configurable_sigma(self):
        from core.pipeline import get_plugin_config, load_pipeline_config

        load_pipeline_config()
        cfg = get_plugin_config("alerting")
        assert cfg.get("anomaly_sigma", 2.0) == 2.0
