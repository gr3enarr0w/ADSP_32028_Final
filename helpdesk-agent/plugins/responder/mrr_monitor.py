"""ANTSE-450: Weekly retrieval quality monitoring — MRR snapshot + OpsGenie alerting.

Evaluates MRR across all five retrieval strategies (BM25, dense, RRF, weighted,
learned) once per week and writes results to ``retrieval_quality_log``.  A P3
OpsGenie alert is fired when the primary fusion MRR drops more than 0.05 versus
the previous week's reading.

Primary fusion MRR is determined in this order:
  1. ``mrr_rrf``   (active when ``tune_fusion`` selects RRF)
  2. ``mrr_weighted``
  3. ``mrr_learned``

The simplest proxy — ``max(mrr_rrf, mrr_weighted, mrr_learned)`` — is used so
the monitor tracks whichever fusion strategy is winning, consistent with how
``evaluate_mrr`` is consumed by ``tune_fusion``.
"""

from __future__ import annotations

import datetime as dt
import logging
import re

from db import get_db_conn, get_last_run_date, set_last_run_date
from plugins.responder.retrieval import evaluate_mrr, resolve_eval_queries

log = logging.getLogger(__name__)

JOB_NAME = "mrr_snapshot"
MRR_DROP_THRESHOLD = 0.05

_CLOUD_ID_RE = re.compile(r"/ex/(?:jira|jsm)/([^/]+)$")


def _resolve_job_name() -> str:
    """Return the job-state key scoped to the current JSM cloud instance.

    Resolves ``get_cloud_base_url("jsm")`` at call time and appends the cloud
    UUID so that snapshots from different Atlassian sites never share a
    ``job_state`` row.

    Falls back to the bare ``JOB_NAME`` if the URL cannot be resolved or parsed,
    so the monitor still runs rather than crashing.
    """
    try:
        from ingest.oauth2lo import get_cloud_base_url

        base_url = get_cloud_base_url("jsm")
        match = _CLOUD_ID_RE.search(base_url)
        if match:
            return f"{JOB_NAME}:{match.group(1)}"
        log.warning(
            "[mrr_monitor] could not extract cloud_id from %r — using bare job name",
            base_url,
        )
    except Exception as exc:
        log.warning("[mrr_monitor] get_cloud_base_url('jsm') failed (%s) — using bare job name", exc)
    return JOB_NAME


def _primary_mrr(scores: dict[str, float]) -> float:
    """Return the best fusion MRR from a ``evaluate_mrr()`` result dict."""
    return max(scores.get("rrf", 0.0), scores.get("weighted", 0.0), scores.get("learned", 0.0))


def _get_last_week_mrr(conn) -> float | None:
    """Return the primary MRR from the most recent log entry before today, or None."""
    today = dt.date.today().isoformat()
    row = conn.execute(
        """
        SELECT mrr_rrf, mrr_weighted, mrr_learned
        FROM retrieval_quality_log
        WHERE run_date < ?
        ORDER BY run_date DESC
        LIMIT 1
        """,
        (today,),
    ).fetchone()
    if row is None:
        return None
    return max(
        float(row["mrr_rrf"] or 0.0),
        float(row["mrr_weighted"] or 0.0),
        float(row["mrr_learned"] or 0.0),
    )


def run_mrr_snapshot() -> dict[str, float] | None:
    """Evaluate MRR over the full corpus and persist results.

    Skips the run if a snapshot already exists for today (idempotent).

    Side effects:
        - Writes one row to ``retrieval_quality_log``.
        - Fires a P3 OpsGenie alert when primary fusion MRR drops > 0.05
          week-over-week.

    Returns:
        The ``evaluate_mrr()`` result dict, or ``None`` if skipped.
    """
    today = dt.date.today()

    _job_name = _resolve_job_name()

    with get_db_conn() as conn:
        last_run = get_last_run_date(conn, _job_name)
        if last_run == today:
            log.debug("[mrr_monitor] snapshot already recorded today — skipping")
            return None

        prev_mrr = _get_last_week_mrr(conn)

    log.info("[mrr_monitor] running evaluate_mrr() for weekly snapshot")
    scores = evaluate_mrr()

    n_queries: int = 0
    try:
        n_queries = len(resolve_eval_queries())
    except Exception:
        pass

    with get_db_conn() as conn:
        conn.execute(
            """
            INSERT INTO retrieval_quality_log
                (run_date, mrr_bm25, mrr_dense, mrr_rrf, mrr_weighted, mrr_learned, n_queries)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (run_date) DO UPDATE SET
                mrr_bm25    = excluded.mrr_bm25,
                mrr_dense   = excluded.mrr_dense,
                mrr_rrf     = excluded.mrr_rrf,
                mrr_weighted = excluded.mrr_weighted,
                mrr_learned = excluded.mrr_learned,
                n_queries   = excluded.n_queries
            """,
            (
                today.isoformat(),
                scores.get("bm25"),
                scores.get("dense"),
                scores.get("rrf"),
                scores.get("weighted"),
                scores.get("learned"),
                n_queries,
            ),
        )
        set_last_run_date(conn, _job_name, today)

    current_mrr = _primary_mrr(scores)
    log.info(
        "[mrr_monitor] snapshot recorded — rrf=%.3f weighted=%.3f learned=%.3f "
        "primary=%.3f n_queries=%d",
        scores.get("rrf", 0.0),
        scores.get("weighted", 0.0),
        scores.get("learned", 0.0),
        current_mrr,
        n_queries,
    )

    if prev_mrr is not None:
        drop = prev_mrr - current_mrr
        if drop > MRR_DROP_THRESHOLD:
            _fire_mrr_drop_alert(prev_mrr, current_mrr, drop, scores)

    return scores


def _fire_mrr_drop_alert(
    prev_mrr: float,
    current_mrr: float,
    drop: float,
    scores: dict[str, float],
) -> None:
    """Post a P3 OpsGenie alert for a retrieval quality regression."""
    try:
        from plugins.alerting.opsgenie import post_alert

        message = (
            f"[ai-helpdesk] Retrieval MRR drop detected: "
            f"{prev_mrr:.3f} → {current_mrr:.3f} (Δ={drop:.3f})"
        )
        description = (
            f"Weekly MRR snapshot recorded a regression exceeding the {MRR_DROP_THRESHOLD:.2f} "
            f"threshold.\n\n"
            f"Previous: {prev_mrr:.4f}\n"
            f"Current:  {current_mrr:.4f}\n"
            f"Drop:     {drop:.4f}\n\n"
            f"Per-strategy scores:\n"
            f"  BM25={scores.get('bm25', 0.0):.4f}  "
            f"dense={scores.get('dense', 0.0):.4f}  "
            f"rrf={scores.get('rrf', 0.0):.4f}  "
            f"weighted={scores.get('weighted', 0.0):.4f}  "
            f"learned={scores.get('learned', 0.0):.4f}\n\n"
            "Check eval set integrity, corpus changes, or re-run tune_fusion()."
        )
        fired = post_alert(
            message=message,
            alias=f"mrr-drop-{dt.date.today().isoformat()}",
            description=description,
            priority="P3",
            tags=["retrieval", "mrr", "regression"],
        )
        if fired:
            log.warning(
                "[mrr_monitor] P3 OpsGenie alert sent — MRR drop %.3f (%.3f → %.3f)",
                drop,
                prev_mrr,
                current_mrr,
            )
        else:
            log.error("[mrr_monitor] OpsGenie alert failed to send for MRR drop %.3f", drop)
    except Exception as exc:
        log.error("[mrr_monitor] could not fire OpsGenie alert: %s", exc)


def get_mrr_trend(n_weeks: int = 8) -> list[dict]:
    """Return the last N weekly MRR snapshots, newest first.

    Args:
        n_weeks: Maximum number of rows to return.

    Returns:
        List of dicts with keys: run_date, mrr_bm25, mrr_dense, mrr_rrf,
        mrr_weighted, mrr_learned, n_queries, primary_mrr.
    """
    with get_db_conn() as conn:
        rows = conn.execute(
            """
            SELECT run_date, mrr_bm25, mrr_dense, mrr_rrf, mrr_weighted, mrr_learned, n_queries
            FROM retrieval_quality_log
            ORDER BY run_date DESC
            LIMIT ?
            """,
            (n_weeks,),
        ).fetchall()

    result = []
    for row in rows:
        d = dict(row)
        d["primary_mrr"] = max(
            float(d.get("mrr_rrf") or 0.0),
            float(d.get("mrr_weighted") or 0.0),
            float(d.get("mrr_learned") or 0.0),
        )
        result.append(d)
    return result
