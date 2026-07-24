"""Unit tests for plugins/feedback/correlator.py — ANTSE-324."""

from __future__ import annotations

import numpy as np
import pytest

from db import get_db_conn, init_db
from plugins.feedback.correlator import (
    CV_VARIANCE_UNSTABLE_THRESHOLD,
    MIN_SAMPLES,
    bonferroni_correct,
    compute_category_correlations,
    compute_category_stats,
    cv_spearman_variance,
)


@pytest.fixture(autouse=True)
def use_memory_db(tmp_path, monkeypatch):
    """Force SQLite for each test, overriding any Postgres proxy from conftest."""
    import sqlite3
    import db as _db

    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr(_db, "DATABASE_URL", None)
    monkeypatch.setattr(_db, "DB_PATH", db_path)

    def _sqlite_get_db():
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    monkeypatch.setattr(_db, "get_db", _sqlite_get_db)
    init_db()
    yield db_path


def _insert_classification(conn, ticket_key: str, category: str) -> None:
    conn.execute(
        """
        INSERT INTO ticket_classifications (ticket_key, category, classified_at)
        VALUES (?, ?, '2026-01-01')
        """,
        (ticket_key, category),
    )


def _insert_csat(conn, ticket_key: str, score: int) -> None:
    conn.execute(
        """
        INSERT INTO ticket_csat (ticket_key, csat_score, submitted_at, ingested_at)
        VALUES (?, ?, '2026-01-01', '2026-01-01')
        """,
        (ticket_key, score),
    )


def _insert_feedback(
    conn,
    ticket_key: str,
    similarity: float,
    *,
    accepted: bool = True,
) -> None:
    conn.execute(
        """
        INSERT INTO ai_draft_feedback (
            ticket_key, draft_comment_id, response_type,
            draft_customer_response, similarity_score, actual_response
        ) VALUES (?, ?, 'customer', 'draft', ?, ?)
        """,
        (
            ticket_key,
            f"{ticket_key}-draft",
            similarity,
            "actual" if accepted else None,
        ),
    )


def _seed_category(
    conn,
    category: str,
    n: int,
    *,
    csat_fn,
    sim_fn,
    prefix: str = "T",
) -> None:
    for i in range(n):
        key = f"{prefix}-{category}-{i}"
        _insert_classification(conn, key, category)
        _insert_csat(conn, key, csat_fn(i))
        _insert_feedback(conn, key, sim_fn(i))


class TestInsufficientSamples:
    def test_skips_category_with_n_below_30(self):
        with get_db_conn() as conn:
            _seed_category(
                conn,
                "small_cat",
                29,
                csat_fn=lambda i: (i % 5) + 1,
                sim_fn=lambda i: i / 29.0,
            )
            results = compute_category_correlations(conn)
            rows = conn.execute(
                "SELECT category FROM category_csat_correlations WHERE category = ?",
                ("small_cat",),
            ).fetchall()

        assert results == []
        assert len(rows) == 1


class TestStrongPositiveCorrelation:
    def test_detects_strong_positive_spearman(self):
        with get_db_conn() as conn:
            _seed_category(
                conn,
                "positive_cat",
                35,
                csat_fn=lambda i: (i % 5) + 1,
                sim_fn=lambda i: (i % 5) / 4.0,
            )
            results = compute_category_correlations(conn)

        assert len(results) == 1
        assert results[0].spearman_r > 0.7


class TestNoCorrelation:
    def test_reports_near_zero_correlation(self):
        rng = np.random.default_rng(99)

        def random_sim(_i):
            return float(rng.random())

        with get_db_conn() as conn:
            _seed_category(
                conn,
                "noise_cat",
                40,
                csat_fn=lambda i: (i % 5) + 1,
                sim_fn=random_sim,
                prefix="N",
            )
            results = compute_category_correlations(conn)

        assert len(results) == 1
        assert abs(results[0].spearman_r) < 0.35


def _unstable_segment_csat_sim(i: int) -> tuple[int, float]:
    """Five alternating positive/negative segments — high fold-level Spearman variance."""
    segment = i // 13
    offset = i % 13
    csat = (offset % 5) + 1
    sim = csat / 5.0 if segment % 2 == 0 else 1.0 - csat / 5.0
    return csat, sim


class TestUnstableCV:
    def test_flags_high_cv_variance(self):
        """Alternating segment correlations produce divergent fold-level Spearman estimates."""
        n = 65

        with get_db_conn() as conn:
            for i in range(n):
                csat, sim = _unstable_segment_csat_sim(i)
                key = f"U-unstable_cat-{i}"
                _insert_classification(conn, key, "unstable_cat")
                _insert_csat(conn, key, csat)
                _insert_feedback(conn, key, sim)
            results = compute_category_correlations(conn)

        assert len(results) == 1
        assert results[0].cv_variance is not None
        assert results[0].cv_variance > CV_VARIANCE_UNSTABLE_THRESHOLD
        assert results[0].unstable is True

    def test_cv_variance_helper_is_high_for_mixed_population(self):
        csat = np.array([_unstable_segment_csat_sim(i)[0] for i in range(65)], dtype=float)
        sim = np.array([_unstable_segment_csat_sim(i)[1] for i in range(65)], dtype=float)
        assert cv_spearman_variance(csat, sim) > CV_VARIANCE_UNSTABLE_THRESHOLD


class TestBonferroniCorrection:
    def test_p_corrected_equals_p_times_n_categories(self):
        with get_db_conn() as conn:
            for cat in ("cat_a", "cat_b"):
                _seed_category(
                    conn,
                    cat,
                    32,
                    csat_fn=lambda i, c=cat: (i % 5) + 1,
                    sim_fn=lambda i, c=cat: (i % 5) / 4.0,
                    prefix=cat,
                )
            results = compute_category_correlations(conn)

        assert len(results) == 2
        for r in results:
            expected = bonferroni_correct(r.spearman_p, 2)
            assert r.spearman_p_corrected == pytest.approx(expected)
            assert r.spearman_p_corrected == pytest.approx(min(1.0, r.spearman_p * 2))

    def test_bonferroni_counts_only_tested_categories(self):
        with get_db_conn() as conn:
            _seed_category(
                conn,
                "tested_cat",
                32,
                csat_fn=lambda i: (i % 5) + 1,
                sim_fn=lambda i: (i % 5) / 4.0,
                prefix="tested",
            )
            # n >= 30 but no similarity scores: should be excluded from tested hypotheses.
            for i in range(32):
                key = f"nosim-cat-{i}"
                _insert_classification(conn, key, "no_similarity_cat")
                _insert_csat(conn, key, (i % 5) + 1)
            results = compute_category_correlations(conn)

        assert len(results) == 1
        only = results[0]
        assert only.category == "tested_cat"
        assert only.spearman_p_corrected == pytest.approx(only.spearman_p)


class TestComputeCategoryStats:
    def test_bonferroni_uses_supplied_n_categories(self):
        rows = [
            {
                "category": "x",
                "ticket_key": f"T-{i}",
                "csat_score": (i % 5) + 1,
                "similarity_score": (i % 5) / 4.0,
                "accepted": 1,
            }
            for i in range(MIN_SAMPLES)
        ]
        result = compute_category_stats("x", rows, "2026-06-03", n_categories_tested=3)
        assert result is not None
        assert result.spearman_p_corrected == pytest.approx(min(1.0, result.spearman_p * 3))


class TestSnapshotBehavior:
    def test_rerun_replaces_same_day_snapshot_rows(self):
        with get_db_conn() as conn:
            _seed_category(
                conn,
                "first_cat",
                35,
                csat_fn=lambda i: (i % 5) + 1,
                sim_fn=lambda i: (i % 5) / 4.0,
                prefix="first",
            )
            compute_category_correlations(conn)
            first_rows = conn.execute(
                "SELECT category FROM category_csat_correlations"
            ).fetchall()
            assert {r["category"] for r in first_rows} == {"first_cat"}

            conn.execute("DELETE FROM ticket_csat")
            conn.execute("DELETE FROM ticket_classifications")
            conn.execute("DELETE FROM ai_draft_feedback")
            _seed_category(
                conn,
                "second_cat",
                35,
                csat_fn=lambda i: (i % 5) + 1,
                sim_fn=lambda i: 1.0 - (i % 5) / 4.0,
                prefix="second",
            )
            compute_category_correlations(conn)
            second_rows = conn.execute(
                "SELECT category FROM category_csat_correlations"
            ).fetchall()

        assert {r["category"] for r in second_rows} == {"second_cat"}
