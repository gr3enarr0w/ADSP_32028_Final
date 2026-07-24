"""Tests for ANTSE-450 — weekly MRR snapshot and OpsGenie alerting."""

from __future__ import annotations

import datetime as dt
from unittest.mock import MagicMock, patch

import pytest

from db import get_db_conn, init_db


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _in_memory_db(monkeypatch, tmp_path):
    """Redirect all DB calls to a fresh SQLite in-memory database."""
    db_file = str(tmp_path / "test_mrr.db")
    monkeypatch.setenv("DATABASE_URL", "")
    import db as db_mod
    monkeypatch.setattr(db_mod, "DATABASE_URL", None)
    monkeypatch.setattr(db_mod, "DB_PATH", db_file)
    init_db()
    yield


@pytest.fixture()
def fake_mrr_scores():
    return {
        "bm25": 0.40,
        "dense": 0.45,
        "rrf": 0.55,
        "weighted": 0.52,
        "learned": 0.53,
    }


# ---------------------------------------------------------------------------
# _primary_mrr
# ---------------------------------------------------------------------------

def test_primary_mrr_picks_max_fusion():
    from plugins.responder.mrr_monitor import _primary_mrr

    scores = {"bm25": 0.9, "dense": 0.9, "rrf": 0.60, "weighted": 0.55, "learned": 0.58}
    assert _primary_mrr(scores) == pytest.approx(0.60)


def test_primary_mrr_ignores_bm25_and_dense():
    from plugins.responder.mrr_monitor import _primary_mrr

    scores = {"bm25": 0.99, "dense": 0.99, "rrf": 0.10, "weighted": 0.20, "learned": 0.15}
    assert _primary_mrr(scores) == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# run_mrr_snapshot — idempotency
# ---------------------------------------------------------------------------

def test_snapshot_skips_when_already_run_today(fake_mrr_scores):
    from plugins.responder.mrr_monitor import run_mrr_snapshot, JOB_NAME
    from db import set_last_run_date

    today = dt.date.today()
    with get_db_conn() as conn:
        set_last_run_date(conn, JOB_NAME, today)

    with patch("plugins.responder.mrr_monitor.evaluate_mrr") as mock_eval:
        result = run_mrr_snapshot()

    mock_eval.assert_not_called()
    assert result is None


def test_snapshot_writes_row_on_first_run(fake_mrr_scores):
    from plugins.responder.mrr_monitor import run_mrr_snapshot

    with patch("plugins.responder.mrr_monitor.evaluate_mrr", return_value=fake_mrr_scores):
        with patch("plugins.responder.mrr_monitor.resolve_eval_queries", return_value=["q"] * 42):
            result = run_mrr_snapshot()

    assert result == fake_mrr_scores

    with get_db_conn() as conn:
        row = conn.execute(
            "SELECT * FROM retrieval_quality_log ORDER BY run_date DESC LIMIT 1"
        ).fetchone()

    assert row is not None
    assert float(row["mrr_rrf"]) == pytest.approx(0.55)
    assert float(row["mrr_bm25"]) == pytest.approx(0.40)
    assert int(row["n_queries"]) == 42


def test_snapshot_is_idempotent_within_same_day(fake_mrr_scores):
    """Second call on the same day skips eval and returns None."""
    from plugins.responder.mrr_monitor import run_mrr_snapshot

    with patch("plugins.responder.mrr_monitor.evaluate_mrr", return_value=fake_mrr_scores):
        with patch("plugins.responder.mrr_monitor.resolve_eval_queries", return_value=[]):
            run_mrr_snapshot()
            result2 = run_mrr_snapshot()

    assert result2 is None

    with get_db_conn() as conn:
        count = conn.execute("SELECT COUNT(*) FROM retrieval_quality_log").fetchone()[0]
    assert count == 1


# ---------------------------------------------------------------------------
# OpsGenie alert gating
# ---------------------------------------------------------------------------

def test_no_alert_when_no_previous_snapshot(fake_mrr_scores):
    from plugins.responder.mrr_monitor import run_mrr_snapshot

    with patch("plugins.responder.mrr_monitor.evaluate_mrr", return_value=fake_mrr_scores):
        with patch("plugins.responder.mrr_monitor.resolve_eval_queries", return_value=[]):
            with patch("plugins.responder.mrr_monitor._fire_mrr_drop_alert") as mock_alert:
                run_mrr_snapshot()

    mock_alert.assert_not_called()


def test_alert_fires_when_drop_exceeds_threshold(fake_mrr_scores):
    from plugins.responder.mrr_monitor import run_mrr_snapshot

    yesterday = (dt.date.today() - dt.timedelta(days=7)).isoformat()
    with get_db_conn() as conn:
        conn.execute(
            """
            INSERT INTO retrieval_quality_log
                (run_date, mrr_rrf, mrr_weighted, mrr_learned, mrr_bm25, mrr_dense, n_queries)
            VALUES (?, 0.70, 0.68, 0.67, 0.50, 0.55, 100)
            """,
            (yesterday,),
        )

    degraded = {k: v - 0.20 for k, v in fake_mrr_scores.items()}

    with patch("plugins.responder.mrr_monitor.evaluate_mrr", return_value=degraded):
        with patch("plugins.responder.mrr_monitor.resolve_eval_queries", return_value=[]):
            with patch("plugins.responder.mrr_monitor._fire_mrr_drop_alert") as mock_alert:
                run_mrr_snapshot()

    mock_alert.assert_called_once()
    _prev, _curr, drop, _scores = mock_alert.call_args.args
    assert drop > 0.05


def test_no_alert_when_drop_within_threshold(fake_mrr_scores):
    from plugins.responder.mrr_monitor import run_mrr_snapshot

    yesterday = (dt.date.today() - dt.timedelta(days=7)).isoformat()
    with get_db_conn() as conn:
        conn.execute(
            """
            INSERT INTO retrieval_quality_log
                (run_date, mrr_rrf, mrr_weighted, mrr_learned, mrr_bm25, mrr_dense, n_queries)
            VALUES (?, 0.56, 0.53, 0.54, 0.41, 0.46, 100)
            """,
            (yesterday,),
        )

    # Current is only 0.02 below previous primary (0.56 → 0.55) — under threshold
    with patch("plugins.responder.mrr_monitor.evaluate_mrr", return_value=fake_mrr_scores):
        with patch("plugins.responder.mrr_monitor.resolve_eval_queries", return_value=[]):
            with patch("plugins.responder.mrr_monitor._fire_mrr_drop_alert") as mock_alert:
                run_mrr_snapshot()

    mock_alert.assert_not_called()


# ---------------------------------------------------------------------------
# get_mrr_trend
# ---------------------------------------------------------------------------

def test_get_mrr_trend_returns_rows_newest_first():
    from plugins.responder.mrr_monitor import get_mrr_trend

    dates = ["2026-06-01", "2026-06-08", "2026-05-25"]
    with get_db_conn() as conn:
        for d in dates:
            conn.execute(
                """
                INSERT INTO retrieval_quality_log
                    (run_date, mrr_rrf, mrr_weighted, mrr_learned, mrr_bm25, mrr_dense, n_queries)
                VALUES (?, 0.50, 0.48, 0.49, 0.35, 0.40, 50)
                """,
                (d,),
            )

    trend = get_mrr_trend(n_weeks=10)
    assert len(trend) == 3
    assert trend[0]["run_date"] == "2026-06-08"
    assert trend[-1]["run_date"] == "2026-05-25"


def test_get_mrr_trend_includes_primary_mrr():
    from plugins.responder.mrr_monitor import get_mrr_trend

    with get_db_conn() as conn:
        conn.execute(
            """
            INSERT INTO retrieval_quality_log
                (run_date, mrr_rrf, mrr_weighted, mrr_learned, mrr_bm25, mrr_dense, n_queries)
            VALUES ('2026-06-08', 0.60, 0.55, 0.58, 0.40, 0.45, 200)
            """
        )

    trend = get_mrr_trend()
    assert trend[0]["primary_mrr"] == pytest.approx(0.60)


def test_get_mrr_trend_respects_n_weeks_limit():
    from plugins.responder.mrr_monitor import get_mrr_trend

    with get_db_conn() as conn:
        for i in range(10):
            date_str = (dt.date(2026, 1, 1) + dt.timedelta(weeks=i)).isoformat()
            conn.execute(
                """
                INSERT INTO retrieval_quality_log
                    (run_date, mrr_rrf, mrr_weighted, mrr_learned, mrr_bm25, mrr_dense, n_queries)
                VALUES (?, 0.5, 0.5, 0.5, 0.4, 0.4, 100)
                """,
                (date_str,),
            )

    trend = get_mrr_trend(n_weeks=4)
    assert len(trend) == 4
