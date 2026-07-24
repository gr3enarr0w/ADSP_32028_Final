"""Tests for CategoryAnomalyDetector and content anomaly alerts — ANTSE-449.

Validation targets:
  TPR >= 80% on 2026-03-16 category spike window
  FPR < 10% on 10 sampled normal windows
"""

from __future__ import annotations

import json
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


def _insert_ticket(
    conn,
    ticket_key: str,
    created_at: str,
    category: str = "Access",
) -> None:
    conn.execute(
        """
        INSERT INTO tickets (ticket_key, summary, description, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (ticket_key, f"Summary {ticket_key}", f"Desc {ticket_key}", created_at),
    )
    conn.execute(
        """
        INSERT INTO ticket_classifications (ticket_key, category, issue_type, confidence)
        VALUES (?, ?, ?, ?)
        """,
        (ticket_key, category, "Request", 0.9),
    )


def _seed_normal_traffic(
    conn,
    start: datetime,
    days: int,
    tickets_per_hour: int = 1,
    category: str = "Access",
) -> None:
    for day_offset in range(days):
        day = start + timedelta(days=day_offset)
        if day.weekday() >= 5:
            continue
        for hour in range(8, 18):
            ts = day.replace(hour=hour, minute=15, second=0, microsecond=0, tzinfo=timezone.utc)
            for i in range(tickets_per_hour):
                key = f"N-{day.strftime('%Y%m%d')}-{hour}-{i}-{category[:3]}"
                _insert_ticket(conn, key, ts.isoformat(), category)


def _seed_category_spike(
    conn,
    day: str,
    hour: int,
    count: int,
    category: str = "Access",
) -> None:
    base = datetime.fromisoformat(f"{day}T{hour:02d}:00:00+00:00")
    for i in range(count):
        offset = timedelta(minutes=i % 60)
        ts = (base + offset).isoformat()
        _insert_ticket(conn, f"CS-{day.replace('-', '')}-{hour}-{i}-{category[:3]}", ts, category)


def _insert_cluster(
    conn,
    cluster_id: int,
    run_date: str,
    ticket_keys: list[str],
    label: str,
    is_new: int,
    growth_rate: float,
    run_type: str = "incident",
) -> None:
    conn.execute(
        """
        INSERT INTO ticket_clusters
            (cluster_id, run_date, run_type, ticket_keys, label, size, is_new, growth_rate)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            cluster_id,
            run_date,
            run_type,
            json.dumps(ticket_keys),
            label,
            len(ticket_keys),
            is_new,
            growth_rate,
        ),
    )


class TestBuildCategoryBaselines:
    def test_writes_category_segment_rows(self):
        from plugins.alerting.anomaly import CategoryAnomalyDetector

        ref = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
        with get_db_conn() as conn:
            _seed_normal_traffic(conn, ref - timedelta(days=90), 90, tickets_per_hour=2)
            detector = CategoryAnomalyDetector(sigma=2.0)
            written = detector.build_category_baselines(conn, reference_time=ref)

        assert written > 0
        with get_db_conn() as conn:
            rows = conn.execute(
                "SELECT category, segment FROM category_anomaly_baselines"
            ).fetchall()
        pairs = {(r["category"], r["segment"]) for r in rows}
        assert ("Access", "mon_business") in pairs

    def test_excludes_baseline_exclude_dates(self):
        from plugins.alerting.anomaly import CategoryAnomalyDetector

        ref = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
        with get_db_conn() as conn:
            _seed_normal_traffic(conn, ref - timedelta(days=90), 90)
            _seed_category_spike(conn, "2026-03-16", 14, 200)
            detector = CategoryAnomalyDetector(sigma=2.0)
            detector.build_category_baselines(conn, reference_time=ref)
            row = conn.execute(
                "SELECT rolling_mean FROM category_anomaly_baselines "
                "WHERE category = 'Access' AND segment = 'mon_business'"
            ).fetchone()

        assert row is not None
        assert row["rolling_mean"] < 10

    def test_replaces_on_rebuild(self):
        from plugins.alerting.anomaly import CategoryAnomalyDetector

        ref = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
        with get_db_conn() as conn:
            _seed_normal_traffic(conn, ref - timedelta(days=90), 90)
            detector = CategoryAnomalyDetector()
            first = detector.build_category_baselines(conn, reference_time=ref)
            second = detector.build_category_baselines(conn, reference_time=ref)
            count = conn.execute(
                "SELECT COUNT(*) FROM category_anomaly_baselines"
            ).fetchone()[0]

        assert first == second
        assert count == second


class TestScoreCategories:
    def test_flags_category_spike(self):
        from plugins.alerting.anomaly import CategoryAnomalyDetector

        ref = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
        with get_db_conn() as conn:
            _seed_normal_traffic(conn, ref - timedelta(days=90), 90, tickets_per_hour=1)
            detector = CategoryAnomalyDetector(sigma=2.0)
            detector.build_category_baselines(conn, reference_time=ref)
            _seed_category_spike(conn, "2026-06-01", 9, 80)
            results = detector.score_categories(conn, window_days=3, reference_time=ref)

        spike_cats = [r.category for r in results if r.is_spike]
        assert "Access" in spike_cats

    def test_returns_empty_for_normal_traffic(self):
        from plugins.alerting.anomaly import CategoryAnomalyDetector

        ref = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
        with get_db_conn() as conn:
            _seed_normal_traffic(conn, ref - timedelta(days=90), 90, tickets_per_hour=1)
            detector = CategoryAnomalyDetector(sigma=2.0)
            detector.build_category_baselines(conn, reference_time=ref)
            results = detector.score_categories(conn, window_days=3, reference_time=ref)

        assert all(not r.is_spike for r in results)

    def test_novel_category_flagged(self):
        from plugins.alerting.anomaly import CategoryAnomalyDetector

        ref = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
        with get_db_conn() as conn:
            _seed_normal_traffic(conn, ref - timedelta(days=90), 90, category="Access")
            detector = CategoryAnomalyDetector(sigma=2.0)
            detector.build_category_baselines(conn, reference_time=ref)
            _seed_category_spike(conn, "2026-06-01", 9, 5, category="NewCategory")
            results = detector.score_categories(conn, window_days=3, reference_time=ref)

        novel = [r for r in results if r.is_novel]
        assert any(r.category == "NewCategory" for r in novel)


class TestDetectTopicSpikes:
    def test_flags_new_cluster_above_threshold(self):
        from plugins.alerting.anomaly import CategoryAnomalyDetector

        ref = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
        run_date = "2026-06-01"
        with get_db_conn() as conn:
            for i in range(5):
                _insert_ticket(conn, f"TK-{i}", f"{run_date}T09:00:00+00:00", "Access")
            _insert_cluster(
                conn,
                cluster_id=1,
                run_date=run_date,
                ticket_keys=[f"TK-{i}" for i in range(5)],
                label="Access spike",
                is_new=1,
                growth_rate=0.8,
            )
            detector = CategoryAnomalyDetector(sigma=2.0)
            results = detector.detect_topic_spikes(conn, window_days=3, reference_time=ref)

        assert len(results) >= 1
        assert results[0].is_spike is True

    def test_ignores_old_cluster_below_threshold(self):
        from plugins.alerting.anomaly import CategoryAnomalyDetector

        ref = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
        run_date = "2026-06-01"
        with get_db_conn() as conn:
            for i in range(3):
                _insert_ticket(conn, f"TK-{i}", f"{run_date}T09:00:00+00:00", "Access")
            _insert_cluster(
                conn,
                cluster_id=2,
                run_date=run_date,
                ticket_keys=[f"TK-{i}" for i in range(3)],
                label="small",
                is_new=0,
                growth_rate=0.1,
            )
            detector = CategoryAnomalyDetector(sigma=2.0)
            results = detector.detect_topic_spikes(conn, window_days=3, reference_time=ref)

        assert results == []


class TestDetectNovelCategories:
    def test_flags_unseen_category(self):
        from plugins.alerting.anomaly import CategoryAnomalyDetector

        ref = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
        with get_db_conn() as conn:
            _seed_normal_traffic(conn, ref - timedelta(days=90), 90, category="Access")
            detector = CategoryAnomalyDetector()
            detector.build_category_baselines(conn, reference_time=ref)
            _insert_ticket(conn, "NOVEL-1", f"{ref.isoformat()}", "BrandNewCategory")
            results = detector.detect_novel_categories(conn, window_days=3, reference_time=ref)

        novel_cats = [r.category for r in results]
        assert "BrandNewCategory" in novel_cats
        assert all(r.is_novel for r in results)

    def test_ignores_known_categories(self):
        from plugins.alerting.anomaly import CategoryAnomalyDetector

        ref = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
        with get_db_conn() as conn:
            _seed_normal_traffic(conn, ref - timedelta(days=90), 90, category="Access")
            detector = CategoryAnomalyDetector()
            detector.build_category_baselines(conn, reference_time=ref)
            _insert_ticket(conn, "KNOWN-1", f"{ref.isoformat()}", "Access")
            results = detector.detect_novel_categories(conn, window_days=3, reference_time=ref)

        assert not any(r.category == "Access" for r in results)


class TestValidationMetrics:
    """TPR >= 80% on 2026-03-16 spike; FPR < 10% on 10 normal windows."""

    def test_tpr_and_fpr_targets(self):
        from plugins.alerting.anomaly import (
            CategoryAnomalyDetector,
            validate_content_anomaly_detection,
        )

        ref = datetime(2026, 6, 2, 12, 0, tzinfo=timezone.utc)
        with get_db_conn() as conn:
            _seed_normal_traffic(conn, ref - timedelta(days=120), 120, tickets_per_hour=1)
            _seed_category_spike(conn, "2026-03-16", 14, 80, category="Access")
            detector = CategoryAnomalyDetector(sigma=2.0)
            detector.build_category_baselines(conn, reference_time=ref)
            metrics = validate_content_anomaly_detection(conn, detector, window_days=3)

        assert metrics["tpr"] >= 0.80, f"TPR {metrics['tpr']} below 80% target"
        assert metrics["fpr"] < 0.10, f"FPR {metrics['fpr']} above 10% target"


class TestFireContentAnomalyAlert:
    @patch("plugins.alerting.anomaly_alert.opsgenie.post_alert")
    def test_fires_p2_for_spike(self, mock_post):
        mock_post.return_value = True
        from plugins.alerting.anomaly import CategoryAnomalyResult
        from plugins.alerting.anomaly_alert import fire_content_anomaly_alert

        result = CategoryAnomalyResult(
            category="Access",
            segment="mon_business",
            recent_count=50,
            rolling_mean=5.0,
            rolling_std=1.0,
            zscore=4.5,
            is_spike=True,
            is_novel=False,
            run_date="2026-06-01",
        )
        assert fire_content_anomaly_alert(result) is True
        kwargs = mock_post.call_args.kwargs
        assert kwargs["priority"] == "P2"
        assert kwargs["alias"] == "content-anomaly-access-2026-06-01"
        assert "content" in kwargs["tags"]
        assert "Access" in kwargs["message"]

    @patch("plugins.alerting.anomaly_alert.opsgenie.post_alert")
    def test_fires_p3_for_novel(self, mock_post):
        mock_post.return_value = True
        from plugins.alerting.anomaly import CategoryAnomalyResult
        from plugins.alerting.anomaly_alert import fire_content_anomaly_alert

        result = CategoryAnomalyResult(
            category="NewCategory",
            segment="mon_business",
            recent_count=3,
            rolling_mean=0.0,
            rolling_std=0.0,
            zscore=0.0,
            is_spike=False,
            is_novel=True,
            run_date="2026-06-01",
        )
        assert fire_content_anomaly_alert(result) is True
        assert mock_post.call_args.kwargs["priority"] == "P3"

    @patch("plugins.alerting.anomaly_alert.opsgenie.post_alert")
    def test_alias_normalises_category_name(self, mock_post):
        mock_post.return_value = True
        from plugins.alerting.anomaly import CategoryAnomalyResult
        from plugins.alerting.anomaly_alert import fire_content_anomaly_alert

        result = CategoryAnomalyResult(
            category="Access / Permissions",
            segment="mon_business",
            recent_count=10,
            rolling_mean=2.0,
            rolling_std=0.5,
            zscore=3.0,
            is_spike=True,
            is_novel=False,
            run_date="2026-06-01",
        )
        fire_content_anomaly_alert(result)
        alias = mock_post.call_args.kwargs["alias"]
        assert " " not in alias
        assert alias.startswith("content-anomaly-access")

    @patch("plugins.alerting.anomaly_alert.opsgenie.post_alert")
    def test_fire_content_anomaly_alerts_returns_count(self, mock_post):
        mock_post.return_value = True
        from plugins.alerting.anomaly import CategoryAnomalyResult
        from plugins.alerting.anomaly_alert import fire_content_anomaly_alerts

        results = [
            CategoryAnomalyResult(
                category=f"Cat{i}",
                segment="mon_business",
                recent_count=10,
                rolling_mean=1.0,
                rolling_std=0.5,
                zscore=3.0,
                is_spike=True,
                is_novel=False,
                run_date="2026-06-01",
            )
            for i in range(3)
        ]
        sent = fire_content_anomaly_alerts(results)
        assert sent == 3
        assert mock_post.call_count == 3
