"""Tests for plugins/alerting/isolation_forest.py — ANTSE-316."""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from db import get_db_conn, init_db
from plugins.alerting.anomaly import AnomalyResult
from plugins.alerting.isolation_forest import MIN_FPR_ROWS, store_control_chart_score

FIT_WINDOW_DAYS = 90
# ~64 weekdays × 10 business hours × 2 tickets/hour when using tickets_per_hour=2
MIN_TRAINING_TICKETS = 500


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


def _seed_baseline_history(conn, start: datetime, days: int, tickets_per_hour: int = 2):
    """Seed normal weekday business-hour traffic (matches test_anomaly volume)."""
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


def _seed_fpr_scores(
    conn,
    *,
    total_rows: int,
    false_positives: int,
    ref: datetime | None = None,
    zscore: float = 2.0,
):
    """Insert control-chart score rows within the FPR measurement window."""
    ref = ref or datetime.now(timezone.utc)
    for i in range(total_rows):
        ts = ref - timedelta(hours=i)
        store_control_chart_score(
            conn,
            AnomalyResult(
                is_anomaly=i < false_positives,
                zscore=zscore,
                segment="tue_business",
                ticket_count=5,
                category_breakdown={},
            ),
            window_minutes=60,
            scored_at=ts,
        )


class TestMeasureFpr:
    def test_returns_zero_when_below_min_rows(self):
        from plugins.alerting.isolation_forest import _measure_fpr

        ref = datetime.now(timezone.utc)
        with get_db_conn() as conn:
            _seed_fpr_scores(conn, total_rows=MIN_FPR_ROWS - 1, false_positives=MIN_FPR_ROWS - 1, ref=ref)
            assert _measure_fpr(conn, window_days=7) == 0.0

    def test_measures_fpr_at_exactly_min_rows(self):
        from plugins.alerting.isolation_forest import _measure_fpr

        ref = datetime.now(timezone.utc)
        with get_db_conn() as conn:
            _seed_fpr_scores(conn, total_rows=MIN_FPR_ROWS, false_positives=3, ref=ref)
            fpr = _measure_fpr(conn, window_days=7)

        assert fpr == pytest.approx(3 / MIN_FPR_ROWS)

    def test_excludes_high_zscore_spikes_from_fpr_numerator(self):
        from plugins.alerting.isolation_forest import _measure_fpr

        ref = datetime.now(timezone.utc)
        with get_db_conn() as conn:
            for i in range(MIN_FPR_ROWS):
                ts = ref - timedelta(hours=i)
                store_control_chart_score(
                    conn,
                    AnomalyResult(
                        is_anomaly=i < 6,
                        zscore=42.0 if i == 0 else 2.0,
                        segment="tue_business",
                        ticket_count=5,
                        category_breakdown={},
                    ),
                    scored_at=ts,
                )
            fpr = _measure_fpr(conn, window_days=7)

        # 5 false positives (z < 10) out of 48 rows; z=42 spike excluded
        assert fpr == pytest.approx(5 / MIN_FPR_ROWS)

    def test_measures_fpr_from_anomaly_rows_with_low_zscore(self):
        from plugins.alerting.isolation_forest import _measure_fpr

        ref = datetime.now(timezone.utc)
        with get_db_conn() as conn:
            _seed_fpr_scores(conn, total_rows=MIN_FPR_ROWS + 2, false_positives=10, ref=ref)
            fpr = _measure_fpr(conn, window_days=7)

        assert fpr == pytest.approx(10 / (MIN_FPR_ROWS + 2))


class TestActivation:
    def test_not_activated_when_insufficient_data(self):
        from plugins.alerting.isolation_forest import IsolationForestDetector

        ref = datetime.now(timezone.utc)
        detector = IsolationForestDetector(fpr_threshold=0.05, fpr_window_days=7)
        with get_db_conn() as conn:
            _seed_fpr_scores(conn, total_rows=10, false_positives=10, ref=ref)
            assert detector.is_activated(conn) is False

    def test_activated_when_fpr_exceeds_threshold(self):
        from plugins.alerting.isolation_forest import IsolationForestDetector

        ref = datetime.now(timezone.utc)
        detector = IsolationForestDetector(fpr_threshold=0.05, fpr_window_days=7)
        with get_db_conn() as conn:
            _seed_fpr_scores(
                conn,
                total_rows=MIN_FPR_ROWS + 2,
                false_positives=4,
                ref=ref,
            )
            assert detector.is_activated(conn) is True

    def test_not_activated_when_fpr_at_or_below_threshold(self):
        from plugins.alerting.isolation_forest import IsolationForestDetector

        ref = datetime.now(timezone.utc)
        detector = IsolationForestDetector(fpr_threshold=0.05, fpr_window_days=7)
        with get_db_conn() as conn:
            _seed_fpr_scores(
                conn,
                total_rows=MIN_FPR_ROWS + 2,
                false_positives=2,
                ref=ref,
            )
            assert detector.is_activated(conn) is False


class TestFitAndScore:
    def test_fit_uses_ninety_day_training_window(self):
        from plugins.alerting.isolation_forest import IsolationForestDetector

        ref = datetime.now(timezone.utc)
        detector = IsolationForestDetector(contamination=0.05)
        with get_db_conn() as conn:
            _seed_baseline_history(conn, ref - timedelta(days=FIT_WINDOW_DAYS), FIT_WINDOW_DAYS)
            detector.fit(conn, window_days=FIT_WINDOW_DAYS, reference_time=ref)
            ticket_count = conn.execute(
                """
                SELECT COUNT(*) FROM tickets
                WHERE created_at >= ?
                """,
                ((ref - timedelta(days=FIT_WINDOW_DAYS)).isoformat(),),
            ).fetchone()[0]

        assert ticket_count >= MIN_TRAINING_TICKETS
        assert detector._model is not None

    def test_detects_off_hours_anomaly_after_realistic_fit(self):
        from plugins.alerting.isolation_forest import IsolationForestDetector

        ref = datetime.now(timezone.utc)
        detector = IsolationForestDetector(contamination=0.05)
        with get_db_conn() as conn:
            _seed_baseline_history(conn, ref - timedelta(days=FIT_WINDOW_DAYS), FIT_WINDOW_DAYS)
            detector.fit(conn, window_days=FIT_WINDOW_DAYS, reference_time=ref)

            # Sunday 3am burst — far outside the business-hour training distribution
            anom_ref = ref.replace(hour=4, minute=0, second=0, microsecond=0)
            while anom_ref.weekday() != 6:
                anom_ref -= timedelta(days=1)
            for i in range(20):
                ts = (anom_ref - timedelta(minutes=i)).isoformat()
                _insert_ticket(conn, f"OFF-{i}", ts)

            anom_result = detector.score(conn, window_minutes=60, reference_time=anom_ref)

            # Tuesday 11am — matches training distribution
            norm_ref = ref.replace(hour=11, minute=0, second=0, microsecond=0)
            while norm_ref.weekday() >= 5:
                norm_ref -= timedelta(days=1)
            for i in range(3):
                ts = (norm_ref - timedelta(minutes=i * 10)).isoformat()
                _insert_ticket(conn, f"NORM-{i}", ts)
            norm_result = detector.score(conn, window_minutes=60, reference_time=norm_ref)

        assert anom_result.is_anomaly
        assert anom_result.ticket_count == 20
        assert not norm_result.is_anomaly

    def test_score_noop_when_model_not_fitted(self):
        from plugins.alerting.isolation_forest import IsolationForestDetector

        ref = datetime.now(timezone.utc)
        detector = IsolationForestDetector()
        with get_db_conn() as conn:
            result = detector.score(conn, window_minutes=60, reference_time=ref)

        assert result.is_anomaly is False
        assert result.zscore == 0.0
        assert result.ticket_count == 0

    def test_fit_graceful_with_no_training_data(self):
        from plugins.alerting.isolation_forest import IsolationForestDetector

        ref = datetime.now(timezone.utc)
        detector = IsolationForestDetector()
        with get_db_conn() as conn:
            detector.fit(conn, reference_time=ref)
            result = detector.score(conn, window_minutes=60, reference_time=ref)

        assert detector._model is None
        assert result.is_anomaly is False


class TestOnScheduleWiring:
    def test_on_schedule_uses_isolation_forest_when_fpr_high(self, caplog):
        import datetime as dt

        from plugins.alerting import TOPIC_CLUSTER_JOB, plugin

        ref = datetime.now(timezone.utc)
        with get_db_conn() as conn:
            _seed_baseline_history(conn, ref - timedelta(days=FIT_WINDOW_DAYS), FIT_WINDOW_DAYS)
            _seed_fpr_scores(
                conn,
                total_rows=MIN_FPR_ROWS + 2,
                false_positives=4,
                ref=ref,
            )
            conn.execute(
                "INSERT INTO job_state (job_name, last_run_date) VALUES (?, ?)",
                (TOPIC_CLUSTER_JOB, dt.date.today().isoformat()),
            )

        with caplog.at_level("INFO"):
            plugin.on_schedule()

        assert any("isolation forest active" in r.message for r in caplog.records)
        assert not any("control chart active" in r.message for r in caplog.records)

    def test_on_schedule_uses_control_chart_when_fpr_low(self, caplog):
        import datetime as dt

        from plugins.alerting import TOPIC_CLUSTER_JOB, plugin

        ref = datetime.now(timezone.utc)
        with get_db_conn() as conn:
            _seed_baseline_history(conn, ref - timedelta(days=FIT_WINDOW_DAYS), FIT_WINDOW_DAYS)
            _seed_fpr_scores(
                conn,
                total_rows=MIN_FPR_ROWS + 2,
                false_positives=1,
                ref=ref,
            )
            conn.execute(
                "INSERT INTO job_state (job_name, last_run_date) VALUES (?, ?)",
                (TOPIC_CLUSTER_JOB, dt.date.today().isoformat()),
            )

        with caplog.at_level("INFO"):
            plugin.on_schedule()

        assert any("control chart active" in r.message for r in caplog.records)
        assert not any("isolation forest active" in r.message for r in caplog.records)

    def test_on_schedule_falls_back_to_control_chart_when_if_unfitted(self, caplog):
        import datetime as dt

        from plugins.alerting import TOPIC_CLUSTER_JOB, plugin

        ref = datetime.now(timezone.utc)
        with get_db_conn() as conn:
            # Seed enough false positives to force IF activation.
            _seed_fpr_scores(
                conn,
                total_rows=MIN_FPR_ROWS + 2,
                false_positives=4,
                ref=ref,
            )
            # Skip clustering so this test isolates anomaly path.
            conn.execute(
                "INSERT INTO job_state (job_name, last_run_date) VALUES (?, ?)",
                (TOPIC_CLUSTER_JOB, dt.date.today().isoformat()),
            )

        # No tickets seeded, so IF fit should no-op and plugin should safely use CC result.
        with caplog.at_level("INFO"):
            plugin.on_schedule()

        assert any("falling back to control chart" in r.message for r in caplog.records)
        assert not any("isolation forest active" in r.message for r in caplog.records)


class TestPipelineConfig:
    def test_if_config_defaults_in_pipeline_yaml(self):
        from pathlib import Path

        from core.pipeline import get_plugin_config, load_pipeline_config

        load_pipeline_config(Path(__file__).resolve().parents[3] / "pipeline.yaml")
        cfg = get_plugin_config("alerting")
        assert cfg.get("if_contamination", 0.05) == 0.05
        assert cfg.get("if_fpr_threshold", 0.05) == 0.05
        assert cfg.get("if_window_days", 7) == 7


class TestLabeledValidationComparison:
    def test_compare_with_control_chart_on_same_labeled_set(self):
        from plugins.alerting.anomaly import ControlChartDetector
        from plugins.alerting.isolation_forest import (
            IsolationForestDetector,
            compare_with_control_chart,
        )

        ref = datetime(2026, 6, 2, 12, 0, tzinfo=timezone.utc)
        with get_db_conn() as conn:
            _seed_baseline_history(conn, ref - timedelta(days=120), 120, tickets_per_hour=1)
            _seed_spike(conn, "2026-03-16", 14, 50)
            _seed_spike(conn, "2026-01-06", 10, 20)

            control = ControlChartDetector(sigma=2.0)
            control.build_baseline(conn, reference_time=ref)
            if_detector = IsolationForestDetector(contamination=0.05)
            comparison = compare_with_control_chart(
                conn,
                if_detector,
                control,
                reference_time=ref,
            )

        assert set(comparison.keys()) == {
            "control_chart",
            "isolation_forest",
            "recommended_detector",
        }
        assert comparison["recommended_detector"] in {"control_chart", "isolation_forest"}
        assert comparison["control_chart"]["normal_windows_sampled"] > 0
        assert comparison["isolation_forest"]["normal_windows_sampled"] > 0
