"""CSAT ingestion — poll JSM feedback for recently resolved tickets."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

from core.oauth import get_cloud_auth, get_cloud_base_url

log = logging.getLogger(__name__)

LOOKBACK_DAYS = 7
FEEDBACK_PATH = "/rest/servicedeskapi/request/{key}/feedback"


def _parse_iso_date(raw: dict | str | None) -> str | None:
    """Extract an ISO8601 timestamp from Atlassian date objects."""
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw or None
    if isinstance(raw, dict):
        return raw.get("iso8601") or raw.get("jira") or None
    return str(raw)


def parse_feedback_payload(data: dict[str, Any]) -> tuple[int | None, str | None, str | None]:
    """Parse CSAT fields from the servicedeskapi feedback response."""
    rating = data.get("rating")
    if rating is None:
        rating = data.get("score")
    if rating is not None:
        rating = int(rating)

    comment = None
    comment_raw = data.get("comment")
    if isinstance(comment_raw, dict):
        comment = comment_raw.get("body") or comment_raw.get("text")
    elif isinstance(comment_raw, str):
        comment = comment_raw
    if comment is None:
        comment = data.get("feedbackComment") or data.get("commentBody")

    submitted_at = (
        _parse_iso_date(data.get("createdDate"))
        or _parse_iso_date(data.get("submittedDate"))
        or _parse_iso_date(data.get("ratingDate"))
    )
    return rating, comment, submitted_at


def get_recently_resolved_ticket_keys(conn, lookback_days: int = LOOKBACK_DAYS) -> list[str]:
    """Return ticket keys resolved within the lookback window."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()
    rows = conn.execute(
        """
        SELECT ticket_key FROM tickets
        WHERE resolution IS NOT NULL
          AND resolved_at IS NOT NULL
          AND resolved_at >= ?
        ORDER BY resolved_at DESC
        """,
        (cutoff,),
    ).fetchall()
    return [row["ticket_key"] for row in rows]


def fetch_ticket_feedback(ticket_key: str, session: requests.Session | None = None) -> dict[str, Any] | None:
    """Fetch CSAT feedback for a ticket. Returns None for 404/403."""
    headers = {**get_cloud_auth("jsm_feedback"), "X-ExperimentalApi": "opt-in"}
    base_url = get_cloud_base_url("jsm_feedback")
    url = f"{base_url}{FEEDBACK_PATH.format(key=ticket_key)}"

    http = session or requests
    resp = http.get(url, headers=headers, timeout=30)

    if resp.status_code == 404:
        return None
    if resp.status_code == 403:
        log.warning("CSAT feedback forbidden for %s — skipping", ticket_key)
        return None
    if resp.status_code != 200:
        log.error(
            "CSAT feedback error for %s: HTTP %d — %s",
            ticket_key,
            resp.status_code,
            resp.text[:500],
        )
        raise RuntimeError(f"CSAT feedback fetch failed for {ticket_key}: HTTP {resp.status_code}")

    return resp.json()


def upsert_csat(
    conn,
    ticket_key: str,
    csat_score: int | None,
    csat_comment: str | None,
    submitted_at: str | None,
) -> None:
    """Insert or update CSAT row for a ticket."""
    ingested_at = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO ticket_csat (ticket_key, csat_score, csat_comment, submitted_at, ingested_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(ticket_key) DO UPDATE SET
            csat_score = excluded.csat_score,
            csat_comment = excluded.csat_comment,
            submitted_at = excluded.submitted_at,
            ingested_at = excluded.ingested_at
        """,
        (ticket_key, csat_score, csat_comment, submitted_at, ingested_at),
    )



def get_all_resolved_ticket_keys(conn) -> list[str]:
    """Return all resolved/closed ticket keys with no time-window limit."""
    rows = conn.execute(
        """
        SELECT ticket_key FROM tickets
        WHERE resolution IS NOT NULL
          AND resolved_at IS NOT NULL
        ORDER BY resolved_at DESC
        """
    ).fetchall()
    return [row["ticket_key"] for row in rows]


def backfill_csat_all(conn, max_workers: int = 20) -> dict[str, int]:
    """Fetch CSAT for every historically resolved ticket — no LOOKBACK_DAYS limit.

    Unlike ingest_csat() which is bounded by LOOKBACK_DAYS=7, this queries ALL
    resolved tickets. Runs concurrently using a thread pool — each worker uses
    its own requests.Session so connections are reused per thread.

    Args:
        max_workers: Number of concurrent HTTP threads (default 20).

    Returns stats dict with keys: checked, found, errors.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from db import get_db_conn as _get_db_conn

    ticket_keys = get_all_resolved_ticket_keys(conn)
    if not ticket_keys:
        log.info("CSAT backfill: no resolved tickets found")
        return {"checked": 0, "found": 0, "errors": 0}

    log.info("CSAT backfill: checking %d resolved tickets (%d workers)", len(ticket_keys), max_workers)

    found = 0
    errors = 0

    def _check_one(key: str) -> tuple[str | None, int | None, str | None, str | None]:
        """Fetch feedback for one ticket; returns (key, score, comment, submitted_at) or None score on skip."""
        session = requests.Session()
        try:
            payload = fetch_ticket_feedback(key, session=session)
            if payload is None:
                return key, None, None, None
            score, comment, submitted_at = parse_feedback_payload(payload)
            return key, score, comment, submitted_at
        except Exception as exc:
            log.error("CSAT backfill failed for %s: %s", key, exc)
            return key, -1, None, None  # sentinel for error

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_check_one, k): k for k in ticket_keys}
        for future in as_completed(futures):
            key, score, comment, submitted_at = future.result()
            if score == -1:
                errors += 1
            elif score is not None:
                with _get_db_conn() as write_conn:
                    upsert_csat(write_conn, key, score, comment, submitted_at)
                found += 1

    log.info("CSAT backfill complete: checked=%d found=%d errors=%d", len(ticket_keys), found, errors)
    return {"checked": len(ticket_keys), "found": found, "errors": errors}


def ingest_csat(db) -> dict[str, int]:
    """Poll feedback for recently resolved tickets and upsert into ticket_csat."""
    stats = {"checked": 0, "found": 0, "errors": 0}

    ticket_keys = get_recently_resolved_ticket_keys(db)
    stats["checked"] = len(ticket_keys)
    if not ticket_keys:
        log.info("CSAT ingest: no recently resolved tickets to check")
        return stats

    session = requests.Session()
    for ticket_key in ticket_keys:
        try:
            payload = fetch_ticket_feedback(ticket_key, session=session)
            if payload is None:
                continue

            csat_score, csat_comment, submitted_at = parse_feedback_payload(payload)
            if csat_score is None:
                log.debug("CSAT ingest: no rating in feedback for %s — skipping", ticket_key)
                continue

            upsert_csat(db, ticket_key, csat_score, csat_comment, submitted_at)
            stats["found"] += 1
        except Exception as exc:
            stats["errors"] += 1
            log.error("CSAT ingest failed for %s: %s", ticket_key, exc)

    log.info(
        "CSAT ingest complete: checked=%d found=%d errors=%d",
        stats["checked"],
        stats["found"],
        stats["errors"],
    )
    return stats
