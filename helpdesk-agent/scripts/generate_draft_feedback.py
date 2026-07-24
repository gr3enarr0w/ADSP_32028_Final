"""Generate real AI drafts for all CSAT-linked tickets and store in ai_draft_feedback.

Usage:
    PYTHONPATH=. DATABASE_URL=postgresql://... python -m scripts.generate_draft_feedback [--workers N] [--limit N]

For each resolved ticket with CSAT data and at least one agent comment:
  1. Run the real Gemini draft pipeline (no Jira post)
  2. Compare AI draft vs actual agent comment via cosine similarity
  3. Upsert result into ai_draft_feedback

Skips tickets already present in ai_draft_feedback.
Uses ThreadPoolExecutor for concurrent Gemini calls.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np

from db import get_db_conn
from plugins.responder.drafting import _draft_response
from plugins.responder.lookup import lookup
from services.embedding import embed_text

log = logging.getLogger(__name__)

_FEEDBACK_CATEGORY_THRESHOLDS = {
    "accepted": 0.85,
    "edited_minor": 0.70,
    "edited_major": 0.50,
}


def _similarity(v1: list[float], v2: list[float]) -> float:
    a = np.array(v1, dtype=np.float32)
    b = np.array(v2, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _feedback_category(score: float) -> str:
    if score >= _FEEDBACK_CATEGORY_THRESHOLDS["accepted"]:
        return "accepted"
    if score >= _FEEDBACK_CATEGORY_THRESHOLDS["edited_minor"]:
        return "edited_minor"
    if score >= _FEEDBACK_CATEGORY_THRESHOLDS["edited_major"]:
        return "edited_major"
    return "rejected"


def _load_pending_tickets(limit: int | None = None) -> list[dict]:
    """Return tickets with CSAT + agent comments not yet in ai_draft_feedback."""
    with get_db_conn() as conn:
        # Pick the longest public comment per ticket as the "actual response"
        sql = """
            SELECT
                t.ticket_key,
                t.summary,
                t.description,
                c.comment_id,
                c.body AS actual_response
            FROM ticket_csat tc
            JOIN tickets t ON t.ticket_key = tc.ticket_key
            JOIN (
                SELECT DISTINCT ON (ticket_key)
                    ticket_key, comment_id, body
                FROM ticket_comments
                WHERE is_public = 1
                  AND body IS NOT NULL
                  AND LENGTH(TRIM(body)) > 20
                ORDER BY ticket_key, LENGTH(body) DESC
            ) c ON c.ticket_key = tc.ticket_key
            WHERE tc.csat_score IS NOT NULL
              AND t.ticket_key NOT IN (
                  SELECT DISTINCT ticket_key FROM ai_draft_feedback
                  WHERE draft_comment_id LIKE 'batch:%'
              )
            ORDER BY t.ticket_key
        """
        if limit:
            sql += f" LIMIT {limit}"
        rows = conn.execute(sql).fetchall()
    return [dict(r) for r in rows]


def _process_ticket(ticket: dict) -> dict | None:
    """Generate draft + similarity for one ticket. Returns row dict or None on error."""
    key = ticket["ticket_key"]
    summary = ticket.get("summary") or ""
    description = ticket.get("description") or ""
    actual = ticket.get("actual_response") or ""

    try:
        matches = lookup(f"{summary}\n\n{description}".strip())
        draft_result = _draft_response(summary, description, matches)
        if not draft_result:
            log.debug("[%s] no draft produced", key)
            return None

        customer_response = draft_result.get("customer_response") or ""
        admin_steps = draft_result.get("admin_steps") or ""
        response_type = draft_result.get("response_type") or "unknown"

        if not customer_response.strip():
            log.debug("[%s] empty customer_response", key)
            return None

        v_draft = embed_text(customer_response, task_type="document")
        v_actual = embed_text(actual, task_type="document")
        score = _similarity(v_draft, v_actual)

        return {
            "ticket_key": key,
            # draft_comment_id is NOT NULL; use a synthetic id for batch rows
            # since no Jira comment is posted.
            "draft_comment_id": f"batch:{key}",
            "actual_comment_id": ticket.get("comment_id"),
            "response_type": response_type,
            "draft_customer_response": customer_response,
            "draft_admin_steps": admin_steps,
            "actual_response": actual,
            "similarity_score": score,
            "feedback_category": _feedback_category(score),
            "draft_mode": "batch_generate",
            "captured_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        log.error("[%s] failed: %s", key, exc)
        return None


def _upsert_row(row: dict) -> None:
    # Unique constraint is (ticket_key, draft_comment_id); batch rows have
    # draft_comment_id=NULL so ON CONFLICT won't fire. Deduplication is
    # handled upstream by _load_pending_tickets excluding existing keys.
    with get_db_conn() as conn:
        conn.execute(
            """
            INSERT INTO ai_draft_feedback (
                ticket_key, draft_comment_id, actual_comment_id, response_type,
                draft_customer_response, draft_admin_steps,
                actual_response, similarity_score, feedback_category,
                draft_mode, captured_at
            ) VALUES (
                %(ticket_key)s, %(draft_comment_id)s, %(actual_comment_id)s, %(response_type)s,
                %(draft_customer_response)s, %(draft_admin_steps)s,
                %(actual_response)s, %(similarity_score)s, %(feedback_category)s,
                %(draft_mode)s, %(captured_at)s
            )
            ON CONFLICT (ticket_key, draft_comment_id) DO UPDATE SET
                response_type           = EXCLUDED.response_type,
                draft_customer_response = EXCLUDED.draft_customer_response,
                draft_admin_steps       = EXCLUDED.draft_admin_steps,
                actual_response         = EXCLUDED.actual_response,
                actual_comment_id       = EXCLUDED.actual_comment_id,
                similarity_score        = EXCLUDED.similarity_score,
                feedback_category       = EXCLUDED.feedback_category,
                draft_mode              = EXCLUDED.draft_mode,
                captured_at             = EXCLUDED.captured_at
            """,
            row,
        )


def run(workers: int = 10, limit: int | None = None) -> dict:
    tickets = _load_pending_tickets(limit)
    total = len(tickets)
    log.info("generate_draft_feedback: %d tickets to process (%d workers)", total, workers)
    if not total:
        return {"processed": 0, "stored": 0, "errors": 0}

    # Pre-warm the embedding model so all threads share an already-loaded
    # instance (avoids MPS/meta-tensor race on Apple Silicon).
    log.info("Warming up embedding model...")
    embed_text("warmup", task_type="query")
    log.info("Embedding model ready.")

    stored = errors = 0
    start = time.monotonic()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_ticket, t): t["ticket_key"] for t in tickets}
        for i, future in enumerate(as_completed(futures), start=1):
            key = futures[future]
            try:
                row = future.result()
                if row:
                    _upsert_row(row)
                    stored += 1
                else:
                    errors += 1
            except Exception as exc:
                log.error("[%s] upsert failed: %s", key, exc)
                errors += 1

            if i % 50 == 0:
                elapsed = time.monotonic() - start
                rate = i / elapsed
                eta = (total - i) / rate if rate else 0
                log.info(
                    "progress: %d/%d stored=%d errors=%d rate=%.1f/s eta=%.0fs",
                    i, total, stored, errors, rate, eta,
                )

    elapsed = time.monotonic() - start
    log.info(
        "generate_draft_feedback done: processed=%d stored=%d errors=%d elapsed=%.1fs",
        total, stored, errors, elapsed,
    )
    return {"processed": total, "stored": stored, "errors": errors}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=10, help="Concurrent Gemini threads (default: 10)")
    parser.add_argument("--limit", type=int, default=None, help="Cap number of tickets (default: all)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )
    stats = run(workers=args.workers, limit=args.limit)
    print(f"\nDone: {stats}")


if __name__ == "__main__":
    main()
