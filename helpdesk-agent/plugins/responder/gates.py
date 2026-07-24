"""Quality gates — status, age, content, confidence, sentiment, and dedup checks for auto-drafting."""

import logging
import re

from config import AGE_GATE_HOURS, AUTO_DRAFT_NOISE_PATTERNS
from core.pipeline import get_plugin_config
from core.routing import INTENSITY_RANK, is_access_related
from db import get_db_conn

log = logging.getLogger(__name__)

_SKIP_STATUSES = {"Resolved", "Closed", "Done", "Waiting for customer"}


def _check_status_gate(ticket_key: str) -> bool:
    """Return True if the ticket's status allows auto-drafting.

    Skips tickets that are resolved, closed, or waiting on the customer.
    Fails open (returns True) for unknown tickets so new arrivals aren't blocked.
    """
    with get_db_conn() as conn:
        row = conn.execute(
            "SELECT status FROM tickets WHERE ticket_key = ?", (ticket_key,)
        ).fetchone()

    if not row:
        return True  # fail-open — ticket not yet ingested

    status = row["status"] or ""
    if status in _SKIP_STATUSES:
        log.info("Ticket %s is '%s' — skipping auto-draft (status gate)", ticket_key, status)
        return False
    return True


def _check_age_gate(ticket_key: str) -> bool:
    """Return True if the ticket is recent enough for auto-drafting.

    Skips tickets older than AGE_GATE_HOURS. Fails open for missing data.
    """
    from datetime import datetime, timezone, timedelta

    if AGE_GATE_HOURS <= 0:
        return True

    with get_db_conn() as conn:
        row = conn.execute(
            "SELECT created_at FROM tickets WHERE ticket_key = ?", (ticket_key,)
        ).fetchone()

    if not row or not row["created_at"]:
        return True  # fail-open

    try:
        created = datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))
        age = datetime.now(timezone.utc) - created
        if age > timedelta(hours=AGE_GATE_HOURS):
            log.info("Ticket %s is %.1f hours old (limit %d) — skipping auto-draft (age gate)",
                     ticket_key, age.total_seconds() / 3600, AGE_GATE_HOURS)
            return False
    except (ValueError, TypeError) as e:
        log.warning("Age gate: could not parse created_at for %s: %s", ticket_key, e)

    return True


def _check_content_gate(ticket_key: str) -> bool:
    """Return True if the ticket has enough content for auto-drafting.

    Skips tickets with very short descriptions or summaries matching
    configured noise patterns. Fails open for missing tickets.
    """
    with get_db_conn() as conn:
        row = conn.execute(
            "SELECT summary, description FROM tickets WHERE ticket_key = ?",
            (ticket_key,),
        ).fetchone()

    if not row:
        return True  # fail-open

    desc = (row["description"] or "").strip()
    if len(desc) < 20:
        log.info("Ticket %s description is %d chars (min 20) — skipping auto-draft (content gate)",
                 ticket_key, len(desc))
        return False

    summary = row["summary"] or ""
    for pattern in AUTO_DRAFT_NOISE_PATTERNS:
        try:
            if re.search(pattern, summary, re.IGNORECASE):
                log.info("Ticket %s summary matches noise pattern %r — skipping auto-draft (content gate)",
                         ticket_key, pattern)
                return False
        except re.error:
            log.debug("Invalid noise pattern %r — skipping", pattern)

    return True


def _get_float_threshold(
    config: dict | None,
    key: str,
    default: float = 0.0,
    warn_if_missing: bool = False,
) -> float:
    cfg = config if config is not None else get_plugin_config("responder")
    threshold_raw = cfg.get(key)
    if threshold_raw is None:
        if warn_if_missing:
            log.warning("responder.%s missing from pipeline config — gate disabled", key)
        return default
    try:
        return float(threshold_raw)
    except (TypeError, ValueError):
        log.warning("Invalid %s %r in pipeline config — gate disabled", key, threshold_raw)
        return default


def _get_confidence_threshold(config: dict | None = None) -> float:
    return _get_float_threshold(config, "confidence_threshold", warn_if_missing=True)


def _get_sentiment_gate_threshold(config: dict | None = None) -> float:
    """Return the distress-score threshold for the customer sentiment draft gate.

    0.0 disables the gate (default until M2 is calibrated).
    """
    return _get_float_threshold(config, "sentiment_gate_threshold", 0.0)


def _confidence_gate(ticket_key: str, config: dict | None, cls_row: dict | None = None) -> bool:
    """Return True when the ticket's classification confidence is acceptable."""
    threshold = _get_confidence_threshold(config)
    if threshold <= 0.0:
        return True

    if cls_row is None:
        with get_db_conn() as conn:
            cls_row = conn.execute(
                "SELECT confidence FROM ticket_classifications WHERE ticket_key = ?",
                (ticket_key,),
            ).fetchone()

    if not cls_row or cls_row["confidence"] is None:
        return True

    try:
        confidence = float(cls_row["confidence"])
    except (TypeError, ValueError) as e:
        log.debug("Could not parse confidence for %s: %s", ticket_key, e)
        return True

    if confidence < threshold:
        log.info(
            "Ticket %s confidence %.2f below threshold %.2f — skipping auto-draft (confidence gate)",
            ticket_key, confidence, threshold,
        )
        return False
    return True


def _confidence_sentiment_draft_gate(ticket_key: str, config: dict | None, cls_row: dict | None = None) -> bool:
    """Return True when customer distress is low enough to allow auto-drafting.

    Suppresses drafting for any ticket whose stored distress score exceeds
    ``sentiment_gate_threshold``. Distinct from ``_check_sentiment_gate`` (access-
    related escalation) and from the M5 router (human_review routing).

    Reads ``sentiment_score`` / ``sentiment_intensity`` from ticket_classifications
    only — no import from the feedback plugin. Disabled when threshold <= 0.0.
    """
    threshold = _get_sentiment_gate_threshold(config)
    if threshold <= 0.0:
        return True

    if cls_row is None:
        with get_db_conn() as conn:
            cls_row = conn.execute(
                """
                SELECT sentiment_score, sentiment_intensity
                FROM ticket_classifications
                WHERE ticket_key = ?
                """,
                (ticket_key,),
            ).fetchone()

    if not cls_row or cls_row["sentiment_score"] is None:
        return True  # fail-open — not yet scored

    try:
        score = float(cls_row["sentiment_score"])
    except (TypeError, ValueError) as e:
        log.debug("Could not parse sentiment_score for %s: %s", ticket_key, e)
        return True

    if score > threshold:
        intensity = (cls_row["sentiment_intensity"] or "unknown").lower()
        log.info(
            "Ticket %s sentiment_score=%.3f (intensity=%s) exceeds "
            "sentiment_gate_threshold %.3f — skipping auto-draft (customer sentiment gate)",
            ticket_key, score, intensity, threshold,
        )
        return False
    return True


def check_classification_gates(ticket_key: str, config: dict | None) -> bool:
    """Run all classification-dependent gates with a single DB query.

    Fetches the ticket_classifications row once and passes it to
    ``_confidence_gate`` and ``_confidence_sentiment_draft_gate``, avoiding
    redundant round-trips during auto-draft sweeps. Fails open if no
    classification row exists yet.
    """
    with get_db_conn() as conn:
        cls_row = conn.execute(
            """
            SELECT confidence, sentiment_score, sentiment_intensity
            FROM ticket_classifications
            WHERE ticket_key = ?
            """,
            (ticket_key,),
        ).fetchone()

    if not _confidence_gate(ticket_key, config, cls_row=cls_row):
        return False
    if not _confidence_sentiment_draft_gate(ticket_key, config, cls_row=cls_row):
        return False
    return True


def _check_sentiment_gate(ticket_key: str) -> bool:
    """Return True if sentiment allows auto-drafting (access-related escalation check).

    Escalates (returns False) when sentiment_intensity meets the configured
    threshold on an access-related ticket. Reads from ticket_classifications
    only — no import from the feedback plugin.

    Note: in the production pipeline this logic is covered by M5 router Rule 1
    (``analysis.router.route_ticket``). This function is retained for direct
    unit testing of the intensity-rank comparison path.
    """
    cfg = get_plugin_config("responder")
    threshold_label = str(cfg.get("sentiment_escalation_threshold", "high")).lower()
    threshold_rank = INTENSITY_RANK.get(threshold_label, INTENSITY_RANK["high"])

    with get_db_conn() as conn:
        row = conn.execute(
            """
            SELECT category, issue_type, sentiment_intensity, sentiment_score
            FROM ticket_classifications
            WHERE ticket_key = ?
            """,
            (ticket_key,),
        ).fetchone()

    if not row or not row["sentiment_intensity"]:
        return True  # fail-open — not yet scored

    if not is_access_related(row["category"], row["issue_type"]):
        return True

    intensity = (row["sentiment_intensity"] or "low").lower()
    current_rank = INTENSITY_RANK.get(intensity, 0)
    if current_rank >= threshold_rank:
        log.info(
            "Ticket %s sentiment_intensity=%s (score=%s) on access-related ticket "
            "— skipping auto-draft (sentiment gate, threshold=%s)",
            ticket_key, intensity, row["sentiment_score"], threshold_label,
        )
        return False
    return True


def _has_pending_draft(ticket_key: str) -> bool:
    """Check if there's already any AI draft record for this ticket.

    Once a ticket has any row in ai_draft_feedback (whether actioned or not),
    we treat it as already handled and never re-draft it.
    """
    with get_db_conn() as conn:
        row = conn.execute(
            "SELECT id FROM ai_draft_feedback WHERE ticket_key = ?",
            (ticket_key,),
        ).fetchone()
    return row is not None
