"""Seed synthetic ai_draft_feedback rows for ANTSE-325 validation.

Uses resolved tickets with public agent comments as the source of ``actual_response``
text, derives realistic ``similarity_score`` values from ``category_csat_correlations``
mean CSAT data, and inserts completed feedback rows so that ANTSE-325 validation
queries have enough signal to run without requiring live AI draft captures.

Usage::

    python -m scripts.seed_synthetic_feedback
    python -m scripts.seed_synthetic_feedback --count 300 --dry-run

The script is fully idempotent: it uses ``ON CONFLICT (ticket_key, draft_comment_id)
DO NOTHING`` so re-runs are safe.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from datetime import datetime, timezone

from db import get_db_conn

log = logging.getLogger(__name__)

# ── Similarity thresholds (mirrors _score_similarity in feedback.py) ─────────

_CATEGORY_THRESHOLDS = [
    (0.92, "as_is"),
    (0.75, "lightly_edited"),
    (0.45, "heavily_rewritten"),
    (0.0, "ignored"),
]

# Synthetic comment ID prefix so these rows are distinguishable from real ones.
_SYNTHETIC_PREFIX = "syn_"

# Response type assigned to synthetic rows (mirrors QUESTION_TYPE_MAP fallback).
_DEFAULT_RESPONSE_TYPE = "admin_action"


# ── Similarity score helpers ──────────────────────────────────────────────────

def _feedback_category(score: float) -> str:
    """Return the feedback category for a similarity score.

    Mirrors the threshold logic in ``plugins/responder/feedback._score_similarity``.
    """
    for threshold, label in _CATEGORY_THRESHOLDS:
        if score >= threshold:
            return label
    return "ignored"


def _sim_score_from_csat(mean_csat: float | None) -> float:
    """Derive a plausible similarity score from a mean CSAT value (1–5 scale).

    Formula (per spec):
      base = min(1.0, mean_csat / 5.0) + gauss(0, 0.08)
      clamped to [0.0, 1.0]

    If ``mean_csat`` is None (no CSAT data for the category), falls back to
    a uniform random draw from [0.5, 0.95].
    """
    if mean_csat is None:
        return random.uniform(0.50, 0.95)
    base = min(1.0, mean_csat / 5.0) + random.gauss(0, 0.08)
    return max(0.0, min(1.0, base))


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_csat_lookup(conn) -> dict[str, float | None]:
    """Return a mapping of category → mean_csat from the most recent run date.

    If the table is empty or a category has no rows, the value will be ``None``,
    which causes ``_sim_score_from_csat`` to fall back to a uniform draw.
    """
    rows = conn.execute(
        """
        SELECT category, mean_csat
        FROM category_csat_correlations
        WHERE run_date = (
            SELECT MAX(run_date) FROM category_csat_correlations
        )
        """
    ).fetchall()
    return {row["category"]: row["mean_csat"] for row in rows}


def _load_candidate_tickets(conn, count: int) -> list[dict]:
    """Return resolved tickets that have at least one qualifying public comment.

    Qualifying comment: public, not from the ticket reporter, body ≥ 80 chars.
    One comment per ticket is selected (the earliest qualifying agent comment).

    We request up to ``count * 3`` candidates to give plenty of headroom after
    de-duplication against existing ai_draft_feedback rows.
    """
    limit = count * 3
    rows = conn.execute(
        """
        SELECT
            t.ticket_key,
            tc_cls.category,
            tc_cls.question_type,
            cm.comment_id,
            cm.body AS agent_response
        FROM tickets t
        LEFT JOIN ticket_classifications tc_cls
               ON t.ticket_key = tc_cls.ticket_key
        JOIN ticket_comments cm
               ON t.ticket_key = cm.ticket_key
        WHERE t.resolution IS NOT NULL
          AND cm.is_public = 1
          AND cm.author_id != t.reporter_id
          AND LENGTH(cm.body) > 80
          AND cm.comment_id = (
              SELECT comment_id FROM ticket_comments
               WHERE ticket_key = t.ticket_key
                 AND is_public = 1
                 AND author_id != t.reporter_id
                 AND LENGTH(body) > 80
               ORDER BY created_at ASC
               LIMIT 1
          )
        ORDER BY t.resolved_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def _already_seeded_keys(conn) -> set[str]:
    """Return the set of ticket_keys already present in ai_draft_feedback."""
    rows = conn.execute(
        "SELECT DISTINCT ticket_key FROM ai_draft_feedback"
    ).fetchall()
    return {row["ticket_key"] for row in rows}


# ── Core seeding logic ────────────────────────────────────────────────────────

def seed(count: int = 200, dry_run: bool = False) -> int:
    """Seed up to ``count`` synthetic feedback rows and return the number inserted.

    Args:
        count:   Target number of rows to insert.
        dry_run: If True, compute rows but do not write to the database.

    Returns:
        Number of rows successfully inserted (0 for dry runs).
    """
    with get_db_conn() as conn:
        csat_lookup = _load_csat_lookup(conn)
        log.info(
            "Loaded CSAT lookup: %d categories — %s",
            len(csat_lookup),
            list(csat_lookup.keys()),
        )

        candidates = _load_candidate_tickets(conn, count)
        log.info("Found %d candidate tickets with qualifying agent comments", len(candidates))

        if not candidates:
            log.warning(
                "No resolved tickets with qualifying public comments found. "
                "Ensure the tickets and ticket_comments tables are populated."
            )
            return 0

        already_seeded = _already_seeded_keys(conn)
        log.info("%d tickets already present in ai_draft_feedback", len(already_seeded))

        # Filter out tickets already in ai_draft_feedback
        fresh = [c for c in candidates if c["ticket_key"] not in already_seeded]
        log.info("%d fresh candidates after excluding already-seeded tickets", len(fresh))

        if not fresh:
            log.info("All candidate tickets are already seeded — nothing to do.")
            return 0

        # Shuffle to get a variety of categories across runs
        random.shuffle(fresh)
        to_insert = fresh[:count]

        now_iso = datetime.now(timezone.utc).isoformat()
        inserted = 0

        for row in to_insert:
            ticket_key = row["ticket_key"]
            category = row.get("category")  # may be None if unclassified
            agent_response = row["agent_response"] or ""
            # Use the real comment_id prefixed to distinguish from live drafts
            real_comment_id = str(row["comment_id"])
            draft_comment_id = f"{_SYNTHETIC_PREFIX}{real_comment_id}"

            # Derive similarity score
            mean_csat = csat_lookup.get(category)  # None if category missing
            sim_score = _sim_score_from_csat(mean_csat)
            fb_category = _feedback_category(sim_score)

            if dry_run:
                log.debug(
                    "[dry-run] %s  category=%s  mean_csat=%s  sim=%.4f  fb=%s",
                    ticket_key,
                    category,
                    mean_csat,
                    sim_score,
                    fb_category,
                )
                inserted += 1
                continue

            conn.execute(
                """
                INSERT INTO ai_draft_feedback
                    (ticket_key, draft_comment_id, response_type,
                     draft_customer_response, draft_admin_steps,
                     actual_response, actual_comment_id,
                     similarity_score, feedback_category,
                     draft_mode, template_name, captured_at)
                VALUES
                    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (ticket_key, draft_comment_id) DO NOTHING
                """,
                (
                    ticket_key,
                    draft_comment_id,
                    _DEFAULT_RESPONSE_TYPE,
                    agent_response,           # draft_customer_response (synthetic)
                    None,                     # draft_admin_steps
                    agent_response,           # actual_response (same text — seeding)
                    real_comment_id,          # actual_comment_id
                    sim_score,
                    fb_category,
                    "synthetic_seed",         # draft_mode — marks provenance
                    None,                     # template_name
                    now_iso,
                ),
            )
            inserted += 1

        if not dry_run:
            log.info("Committed %d synthetic feedback rows", inserted)

    return inserted


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.seed_synthetic_feedback",
        description=(
            "Seed synthetic ai_draft_feedback rows for ANTSE-325 validation. "
            "Uses resolved tickets with public agent comments as the source of "
            "actual_response text and derives similarity scores from "
            "category_csat_correlations mean CSAT data."
        ),
    )
    parser.add_argument(
        "--count",
        type=int,
        default=200,
        help="Target number of rows to insert (default: 200).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute rows but do not write to the database.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m scripts.seed_synthetic_feedback``."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    mode = "DRY RUN" if args.dry_run else "LIVE"
    log.info("seed_synthetic_feedback  mode=%s  target=%d rows", mode, args.count)

    inserted = seed(count=args.count, dry_run=args.dry_run)

    if args.dry_run:
        print(
            f"[dry-run] Would insert up to {inserted} rows "
            f"(counted candidates, no DB writes performed)."
        )
    elif inserted < args.count:
        log.warning(
            "Only %d/%d rows inserted — not enough qualifying resolved tickets "
            "with public agent comments. Populate tickets/ticket_comments tables "
            "or lower --count.",
            inserted,
            args.count,
        )
        print(f"Inserted {inserted}/{args.count} rows (data-limited).")
        return 1
    else:
        print(f"Inserted {inserted} synthetic feedback rows successfully.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
