"""M12 CSAT correlator — Spearman rank correlation between CSAT and draft quality by category."""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold

log = logging.getLogger(__name__)

MIN_SAMPLES = 30
CV_MIN_SAMPLES = 60
BOOTSTRAP_RESAMPLES = 1000
CV_VARIANCE_UNSTABLE_THRESHOLD = 0.15

# CSAT scores are ordinal (1–5), not interval-scale; Spearman uses ranks and is the
# appropriate primary association measure. Pearson is retained only as a secondary check.
_CORRELATION_JOIN_SQL = """
    SELECT tc.category, tc.ticket_key, t.csat_score, f.similarity_score,
           CASE WHEN f.actual_response IS NOT NULL THEN 1 ELSE 0 END AS accepted
    FROM ticket_csat t
    JOIN ticket_classifications tc ON tc.ticket_key = t.ticket_key
    LEFT JOIN ai_draft_feedback f ON f.ticket_key = t.ticket_key
    WHERE t.csat_score IS NOT NULL
"""

_INSERT_SQL = """
    INSERT INTO category_csat_correlations (
        category, run_date, n_samples, spearman_r, spearman_p, spearman_p_corrected,
        pearson_r, ci_lower, ci_upper, cv_variance, mean_csat, std_csat,
        acceptance_rate, mean_similarity
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


@dataclass
class CategoryCorrelation:
    """Correlation statistics for one ticket category on a given run date."""

    category: str
    run_date: str
    n_samples: int
    spearman_r: float | None
    spearman_p: float | None
    spearman_p_corrected: float | None
    pearson_r: float | None
    ci_lower: float | None
    ci_upper: float | None
    cv_variance: float | None
    mean_csat: float | None
    std_csat: float | None
    acceptance_rate: float | None
    mean_similarity: float | None
    unstable: bool = False


def fetch_correlation_rows(conn) -> list[dict[str, Any]]:
    """Return joined CSAT / classification / draft rows for correlation analysis."""
    return [dict(r) for r in conn.execute(_CORRELATION_JOIN_SQL).fetchall()]


def _group_rows_by_category(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        category = row["category"]
        if not category:
            continue
        grouped.setdefault(category, []).append(row)
    return grouped


def bonferroni_correct(p_value: float, n_categories: int) -> float:
    """Apply Bonferroni correction: p * n_categories, capped at 1.0."""
    return min(1.0, p_value * n_categories)


def bootstrap_spearman_ci(
    csat_scores: np.ndarray,
    similarity_scores: np.ndarray,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap 95% CI for Spearman r using percentile method."""
    rng = np.random.default_rng(seed)
    n = len(csat_scores)
    boot_rs: list[float] = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        r, _ = spearmanr(csat_scores[idx], similarity_scores[idx])
        boot_rs.append(float(r))
    return float(np.percentile(boot_rs, 2.5)), float(np.percentile(boot_rs, 97.5))


def cv_spearman_variance(
    csat_scores: np.ndarray,
    similarity_scores: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> float:
    """Variance of Spearman r across K-fold splits (stability diagnostic)."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_rs: list[float] = []
    indices = np.arange(len(csat_scores))
    for _, test_idx in kf.split(indices):
        r, _ = spearmanr(csat_scores[test_idx], similarity_scores[test_idx])
        fold_rs.append(float(r))
    return float(np.var(fold_rs))


def _pairs_for_correlation(category_rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract CSAT, similarity, and acceptance arrays; correlation uses non-null similarity."""
    csat_all = np.array([r["csat_score"] for r in category_rows], dtype=float)
    accepted = np.array([r["accepted"] or 0 for r in category_rows], dtype=float)
    mask = np.array(
        [r["similarity_score"] is not None for r in category_rows],
        dtype=bool,
    )
    csat = csat_all[mask]
    sim = np.array([r["similarity_score"] for r in category_rows if r["similarity_score"] is not None], dtype=float)
    return csat, sim, accepted


def _summary_from_rows(category_rows: list[dict[str, Any]]) -> tuple[float | None, float | None, float | None, float | None]:
    """Compute summary stats from raw category rows."""
    csat_all = np.array([r["csat_score"] for r in category_rows], dtype=float)
    similarity_non_null = np.array(
        [r["similarity_score"] for r in category_rows if r["similarity_score"] is not None],
        dtype=float,
    )
    accepted = np.array([r["accepted"] or 0 for r in category_rows], dtype=float)
    mean_csat = float(np.mean(csat_all)) if len(csat_all) else None
    std_csat = float(np.std(csat_all, ddof=1)) if len(csat_all) > 1 else 0.0
    acceptance_rate = float(np.sum(accepted) / len(category_rows)) if category_rows else None
    mean_similarity = float(np.mean(similarity_non_null)) if len(similarity_non_null) else None
    return mean_csat, std_csat, acceptance_rate, mean_similarity


def compute_category_stats(
    category: str,
    category_rows: list[dict[str, Any]],
    run_date: str,
    n_categories_tested: int,
) -> CategoryCorrelation | None:
    """Compute correlation metrics for one category; None if n < MIN_SAMPLES."""
    n_samples = len(category_rows)
    if n_samples < MIN_SAMPLES:
        return None

    csat, similarity, accepted = _pairs_for_correlation(category_rows)
    if len(csat) < MIN_SAMPLES:
        return None
    if np.unique(csat).size < 2 or np.unique(similarity).size < 2:
        return None

    spearman_r, spearman_p = spearmanr(csat, similarity)
    pearson_r, _ = pearsonr(csat, similarity)
    ci_lower, ci_upper = bootstrap_spearman_ci(csat, similarity)
    p_corrected = bonferroni_correct(float(spearman_p), n_categories_tested)

    cv_variance: float | None = None
    unstable = False
    if n_samples >= CV_MIN_SAMPLES:
        cv_variance = cv_spearman_variance(csat, similarity)
        unstable = cv_variance > CV_VARIANCE_UNSTABLE_THRESHOLD

    mean_csat = float(np.mean(csat))
    std_csat = float(np.std(csat, ddof=1)) if len(csat) > 1 else 0.0
    acceptance_rate = float(np.sum(accepted) / len(category_rows))
    mean_similarity = float(np.mean(similarity))

    return CategoryCorrelation(
        category=category,
        run_date=run_date,
        n_samples=n_samples,
        spearman_r=float(spearman_r),
        spearman_p=float(spearman_p),
        spearman_p_corrected=p_corrected,
        pearson_r=float(pearson_r),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        cv_variance=cv_variance,
        mean_csat=mean_csat,
        std_csat=std_csat,
        acceptance_rate=acceptance_rate,
        mean_similarity=mean_similarity,
        unstable=unstable,
    )


def _write_correlation(conn, result: CategoryCorrelation) -> None:
    conn.execute(
        _INSERT_SQL,
        (
            result.category,
            result.run_date,
            result.n_samples,
            result.spearman_r,
            result.spearman_p,
            result.spearman_p_corrected,
            result.pearson_r,
            result.ci_lower,
            result.ci_upper,
            result.cv_variance,
            result.mean_csat,
            result.std_csat,
            result.acceptance_rate,
            result.mean_similarity,
        ),
    )


def list_insufficient_categories(conn, run_date: str | None = None) -> list[tuple[str, int]]:
    """Categories persisted for a run with n_samples below MIN_SAMPLES."""
    if run_date is None:
        row = conn.execute(
            "SELECT MAX(run_date) AS run_date FROM category_csat_correlations"
        ).fetchone()
        run_date = row["run_date"] if row else None
    if not run_date:
        return []
    rows = conn.execute(
        """
        SELECT category, n_samples
        FROM category_csat_correlations
        WHERE run_date = ? AND n_samples < ?
        ORDER BY n_samples ASC, category ASC
        """,
        (run_date, MIN_SAMPLES),
    ).fetchall()
    return [(r["category"], r["n_samples"]) for r in rows]


def compute_category_correlations(conn) -> list[CategoryCorrelation]:
    """Run daily Spearman correlation by category and persist to category_csat_correlations."""
    run_date = dt.date.today().isoformat()
    grouped = _group_rows_by_category(fetch_correlation_rows(conn))

    testable: dict[str, list[dict[str, Any]]] = {}
    skipped_low_n: dict[str, list[dict[str, Any]]] = {}
    skipped_low_similarity_pairs: dict[str, list[dict[str, Any]]] = {}
    skipped_constant_inputs: dict[str, list[dict[str, Any]]] = {}
    for cat, rows in grouped.items():
        if len(rows) < MIN_SAMPLES:
            skipped_low_n[cat] = rows
            continue
        csat, similarity, _accepted = _pairs_for_correlation(rows)
        if len(csat) < MIN_SAMPLES:
            skipped_low_similarity_pairs[cat] = rows
            continue
        if np.unique(csat).size < 2 or np.unique(similarity).size < 2:
            skipped_constant_inputs[cat] = rows
            continue
        testable[cat] = rows

    n_categories_tested = len(testable)
    log.info(
        "[feedback] CSAT correlator: %d categories tested (Bonferroni n=%d)",
        n_categories_tested,
        n_categories_tested,
    )

    for cat, rows in sorted(skipped_low_n.items()):
        log.info(
            "[feedback] CSAT correlator: skipped category %r — n=%d < %d (insufficient data)",
            cat,
            len(rows),
            MIN_SAMPLES,
        )
    for cat, rows in sorted(skipped_low_similarity_pairs.items()):
        csat, _sim, _accepted = _pairs_for_correlation(rows)
        log.info(
            "[feedback] CSAT correlator: skipped category %r — n_with_similarity=%d < %d",
            cat,
            len(csat),
            MIN_SAMPLES,
        )
    for cat in sorted(skipped_constant_inputs):
        log.info(
            "[feedback] CSAT correlator: skipped category %r — constant CSAT/similarity values",
            cat,
        )

    # Keep each run_date as a full, consistent snapshot.
    conn.execute(
        "DELETE FROM category_csat_correlations WHERE run_date = ?",
        (run_date,),
    )

    for cat, rows in {**skipped_low_n, **skipped_low_similarity_pairs, **skipped_constant_inputs}.items():
        mean_csat, std_csat, acceptance_rate, mean_similarity = _summary_from_rows(rows)
        _write_correlation(
            conn,
            CategoryCorrelation(
                category=cat,
                run_date=run_date,
                n_samples=len(rows),
                spearman_r=None,
                spearman_p=None,
                spearman_p_corrected=None,
                pearson_r=None,
                ci_lower=None,
                ci_upper=None,
                cv_variance=None,
                mean_csat=mean_csat,
                std_csat=std_csat,
                acceptance_rate=acceptance_rate,
                mean_similarity=mean_similarity,
                unstable=False,
            ),
        )

    results: list[CategoryCorrelation] = []
    for category, rows in sorted(testable.items()):
        result = compute_category_stats(category, rows, run_date, n_categories_tested)
        if result is None:
            continue
        _write_correlation(conn, result)
        results.append(result)
        if result.unstable:
            log.warning(
                "[feedback] CSAT correlator: unstable correlation for %r "
                "(cv_variance=%.3f > %.2f)",
                category,
                result.cv_variance,
                CV_VARIANCE_UNSTABLE_THRESHOLD,
            )

    log.info("[feedback] CSAT correlator: %d categories processed", len(results))
    return results
