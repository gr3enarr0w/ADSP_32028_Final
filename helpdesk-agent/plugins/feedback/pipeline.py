"""Sentiment scoring pipeline — score classified tickets and persist to DB.

Runs independently of the analysis/responder plugins. Reads from
ticket_classifications + tickets + ticket_comments, writes sentiment_score
and sentiment_intensity back to ticket_classifications.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.pipeline import get_plugin_config
from core.routing import INTENSITY_RANK
from db import get_db_conn
from plugins.feedback.sentiment import score_ticket

log = logging.getLogger(__name__)

TEXT_MAX_CHARS = 2000


def score_to_intensity(score: float, high_threshold: float = 0.6) -> str:
    """Map a 0–1 distress score to low / medium / high."""
    if score >= high_threshold:
        return "high"
    medium_threshold = high_threshold / 2.0
    if score >= medium_threshold:
        return "medium"
    return "low"


def _get_latest_customer_comment(conn, ticket_key: str, reporter_id: str | None) -> str:
    """Return the latest public customer comment body from the DB."""
    rows = conn.execute(
        """
        SELECT author_id, body FROM ticket_comments
        WHERE ticket_key = ? AND is_public = 1
        ORDER BY created_at DESC
        """,
        (ticket_key,),
    ).fetchall()

    for row in rows:
        author_id = row["author_id"] or ""
        if not reporter_id:
            continue  # can't identify customer — skip to avoid scoring agent text
        if author_id and author_id != reporter_id:
            continue
        body = (row["body"] or "").strip()
        if body:
            return body
    return ""


def build_sentiment_text(ticket_key: str) -> str:
    """Concatenate ticket description and latest customer comment for scoring."""
    with get_db_conn() as conn:
        ticket = conn.execute(
            "SELECT description, reporter_id FROM tickets WHERE ticket_key = ?",
            (ticket_key,),
        ).fetchone()
        if not ticket:
            return ""

        desc = (ticket["description"] or "").strip()
        comment = _get_latest_customer_comment(conn, ticket_key, ticket["reporter_id"])
        parts = [p for p in (desc, comment) if p]
        return "\n\n".join(parts)[:TEXT_MAX_CHARS]


def score_classification(ticket_key: str) -> bool:
    """Score sentiment for a classified ticket if not yet scored.

    Returns True when a score was written, False when skipped or already scored.
    """
    cfg = get_plugin_config("feedback")
    high_threshold = float(cfg.get("sentiment_intensity_threshold", 0.6))

    with get_db_conn() as conn:
        row = conn.execute(
            """
            SELECT sentiment_intensity FROM ticket_classifications
            WHERE ticket_key = ?
            """,
            (ticket_key,),
        ).fetchone()
        if not row:
            log.debug("[feedback] %s — no classification yet, skipping sentiment", ticket_key)
            return False
        if row["sentiment_intensity"] is not None:
            return False

    text = build_sentiment_text(ticket_key)
    result = score_ticket(text)
    intensity_score = float(result.get("intensity", 0.0))
    intensity_label = score_to_intensity(intensity_score, high_threshold)

    with get_db_conn() as conn:
        conn.execute(
            """
            UPDATE ticket_classifications
            SET sentiment_score = ?, sentiment_intensity = ?
            WHERE ticket_key = ?
            """,
            (round(intensity_score, 4), intensity_label, ticket_key),
        )

    log.info(
        "[feedback] %s sentiment scored: score=%.3f intensity=%s",
        ticket_key,
        intensity_score,
        intensity_label,
    )
    return True


def score_unscored_classifications(max_workers: int = 10) -> tuple[int, int]:
    """Score all classified tickets missing sentiment_intensity.

    Tickets are scored concurrently using a thread pool.  Each worker opens
    its own DB connection via ``get_db_conn()`` so no connection is shared
    across threads.

    Args:
        max_workers: Maximum number of concurrent Gemini scoring threads.
                     Defaults to 10.

    Returns:
        (scored_count, error_count).
    """
    with get_db_conn() as conn:
        rows = conn.execute(
            """
            SELECT ticket_key FROM ticket_classifications
            WHERE sentiment_intensity IS NULL
            ORDER BY classified_at
            """
        ).fetchall()

    ticket_keys = [row["ticket_key"] for row in rows]

    scored = 0
    errors = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_key = {
            executor.submit(score_classification, key): key
            for key in ticket_keys
        }
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                if future.result():
                    scored += 1
            except Exception as exc:
                errors += 1
                log.warning("[feedback] sentiment scoring failed for %s: %s", key, exc)

    if scored:
        log.info("[feedback] scored sentiment for %d tickets (%d errors)", scored, errors)
    return scored, errors
