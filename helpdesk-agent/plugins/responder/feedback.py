"""Feedback loop — emoji feedback, similarity scoring, example harvesting."""

import logging
from datetime import datetime, timezone

from db import get_db_conn

log = logging.getLogger(__name__)

FEEDBACK_EMOJI_MAP = {
    "✅": "both_good",
    "👤": "customer_good",
    "🔧": "steps_good",
    "❌": "both_bad",
    "🔄": "wrong_type",
    "❓": "needs_info",
}

QUESTION_TYPE_MAP = {
    "how-to": "self_service",
    "configuration": "admin_action",
    "access-request": "admin_action",
    "troubleshooting": "admin_action",
    "bug-report": "admin_action",
}


def handle_feedback_emoji(ticket_key: str, comment_id: str, emoji: str) -> bool:
    """Record agent feedback on an AI draft via emoji comment.

    Finds the most recent draft for the ticket, records the feedback
    category, and deletes the emoji comment.
    """
    from .jira_comments import _delete_comment

    category = FEEDBACK_EMOJI_MAP.get(emoji)
    if not category:
        return False

    log.info("Feedback %s (%s) for %s", emoji, category, ticket_key)

    with get_db_conn() as conn:
        # Find the most recent draft for this ticket
        row = conn.execute(
            """SELECT id FROM ai_draft_feedback
               WHERE ticket_key = ?
               ORDER BY captured_at DESC LIMIT 1""",
            (ticket_key,),
        ).fetchone()

        if not row:
            log.warning("No draft found for %s to attach feedback", ticket_key)
            _delete_comment(ticket_key, comment_id)
            return False

        conn.execute(
            "UPDATE ai_draft_feedback SET agent_feedback = ? WHERE id = ?",
            (category, row["id"]),
        )

    log.info("Recorded %s feedback for %s (draft %d)", category, ticket_key, row["id"])
    _delete_comment(ticket_key, comment_id)
    return True


def _store_draft_record(ticket_key: str, comment_id: str, draft: dict):
    """Insert a pending feedback row after posting a draft."""
    with get_db_conn() as conn:
        conn.execute(
            """INSERT OR IGNORE INTO ai_draft_feedback
               (ticket_key, draft_comment_id, response_type,
                draft_customer_response, draft_admin_steps,
                draft_mode, template_name)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                ticket_key,
                comment_id,
                draft["response_type"],
                draft.get("customer_response", ""),
                draft.get("admin_steps"),
                draft.get("draft_mode"),
                draft.get("template_name"),
            ),
        )
    log.debug("Stored draft record for %s (comment %s)", ticket_key, comment_id)


def _score_similarity(draft: str, actual: str) -> tuple[float, str]:
    """Compare draft vs actual response using embedding cosine similarity.

    Returns (score, category).
    """
    from services.embedding import embed_batch

    if not draft.strip() or not actual.strip():
        return 0.0, "ignored"

    vecs = embed_batch([draft, actual])
    score = float(sum(a * b for a, b in zip(vecs[0], vecs[1])))
    score = max(0.0, min(1.0, score))

    if score >= 0.92:
        category = "as_is"
    elif score >= 0.75:
        category = "lightly_edited"
    elif score >= 0.45:
        category = "heavily_rewritten"
    else:
        category = "ignored"
    return score, category


def capture_feedback():
    """Check pending drafts and capture agent responses.

    For each draft without an actual_response, fetches comments from Jira,
    finds the first public comment posted after the AI draft, scores the
    similarity, and updates the feedback row.
    """
    from ingest.tickets import _extract_adf_text
    from ingest.oauth2lo import get_cloud_base_url
    from .jira_comments import _jira_request

    with get_db_conn() as conn:
        pending = conn.execute(
            "SELECT id, ticket_key, draft_comment_id, draft_customer_response "
            "FROM ai_draft_feedback WHERE actual_response IS NULL"
        ).fetchall()

    if not pending:
        return

    log.info("Checking feedback for %d pending drafts", len(pending))
    base = get_cloud_base_url("jsm")
    captured = 0

    for row in pending:
        try:
            resp = _jira_request(
                "get",
                f"{base}/rest/servicedeskapi/request/{row['ticket_key']}/comment",
                "jsm",
                params={"limit": 50},
            )
            if resp.status_code != 200:
                continue

            comments = resp.json().get("values", [])

            # Find the draft comment index, then look for the next public comment
            found_draft = False
            for comment in comments:
                if not found_draft:
                    if str(comment.get("id", "")) == str(row["draft_comment_id"]):
                        found_draft = True
                    continue

                # JSM uses public:bool (True=public, False=internal)
                if not comment.get("public", True):
                    continue

                # Found the first public comment after the draft
                body = comment.get("body", "")
                actual_text = _extract_adf_text(body) if isinstance(body, dict) else str(body)
                if not actual_text.strip():
                    continue

                score, category = _score_similarity(row["draft_customer_response"], actual_text)

                with get_db_conn() as conn:
                    conn.execute(
                        """UPDATE ai_draft_feedback
                           SET actual_response = ?, actual_comment_id = ?,
                               similarity_score = ?, feedback_category = ?,
                               captured_at = ?
                           WHERE id = ?""",
                        (actual_text, comment.get("id"), score, category,
                         datetime.now(timezone.utc).isoformat(), row["id"]),
                    )
                captured += 1
                break  # only capture the first public response

        except Exception as e:
            log.debug("Could not capture feedback for %s: %s", row["ticket_key"], e)

    if captured:
        log.info("Captured feedback for %d/%d drafts", captured, len(pending))


def get_few_shot_examples(response_type: str | None = None, limit: int = 3) -> list[dict]:
    """Retrieve high-quality draft examples for few-shot prompting.

    Prioritizes drafts with explicit agent approval (emoji feedback),
    then falls back to high-similarity draft/response pairs.
    Excludes drafts rated as both_bad, wrong_type, or needs_info.
    """
    with get_db_conn() as conn:
        type_filter = "AND response_type = ?" if response_type else ""
        params = [response_type, limit] if response_type else [limit]

        # Priority 1: agent-approved via emoji (both_good > customer_good/steps_good)
        rows = conn.execute(f"""
            SELECT response_type, draft_customer_response, draft_admin_steps,
                   actual_response, similarity_score, agent_feedback
            FROM ai_draft_feedback
            WHERE agent_feedback IN ('both_good', 'customer_good', 'steps_good')
              {type_filter}
            ORDER BY CASE agent_feedback
                WHEN 'both_good' THEN 0
                WHEN 'customer_good' THEN 1
                WHEN 'steps_good' THEN 2
            END
            LIMIT ?
        """, params).fetchall()

        if len(rows) < limit:
            # Priority 2: high-similarity pairs (existing logic)
            remaining = limit - len(rows)
            params2 = [response_type, remaining] if response_type else [remaining]
            rows2 = conn.execute(f"""
                SELECT response_type, draft_customer_response, draft_admin_steps,
                       actual_response, similarity_score, agent_feedback
                FROM ai_draft_feedback
                WHERE feedback_category IN ('as_is', 'lightly_edited')
                  AND similarity_score >= 0.75
                  AND (agent_feedback IS NULL OR agent_feedback NOT IN ('both_bad', 'wrong_type', 'needs_info'))
                  {type_filter}
                ORDER BY similarity_score DESC
                LIMIT ?
            """, params2).fetchall()
            rows = list(rows) + list(rows2)

    return [dict(r) for r in rows]


def harvest_response_examples() -> int:
    """Scan Cloud-related resolved tickets for organic agent response patterns.

    Cloud tickets are identified by any of:
    - Updated on or after CLOUD_CUTOVER_DATE (post-cutover)
    - UAT affect versions (Broad UAT 1, Broad UAT 2, Scoped UAT)
    - Live 1.0 affect version (production Cloud)
    - Cloud Migration request type
    DC-only tickets are excluded.
    Returns the count of newly harvested examples.
    """
    from config import CLOUD_CUTOVER_DATE

    with get_db_conn() as conn:
        rows = conn.execute("""
            SELECT
                t.ticket_key,
                tc.question_type,
                tc.category,
                cm.body AS agent_response,
                cm.author_id
            FROM tickets t
            LEFT JOIN ticket_classifications tc ON t.ticket_key = tc.ticket_key
            JOIN ticket_comments cm ON t.ticket_key = cm.ticket_key
            WHERE t.resolution IS NOT NULL
              AND (t.created_at >= ?
                   OR t.affect_version LIKE '%%UAT%%'
                   OR t.affect_version = 'Live 1.0'
                   OR t.request_type LIKE '%%Cloud%%')
              AND cm.is_public = 1
              AND cm.author_id != t.reporter_id
              AND LENGTH(cm.body) > 100
              AND t.ticket_key NOT IN (SELECT ticket_key FROM response_examples)
              AND cm.comment_id = (
                  SELECT comment_id FROM ticket_comments
                  WHERE ticket_key = t.ticket_key
                    AND is_public = 1
                    AND author_id != t.reporter_id
                    AND LENGTH(body) > 100
                  ORDER BY created_at ASC
                  LIMIT 1
              )
            ORDER BY t.resolved_at DESC
        """, (CLOUD_CUTOVER_DATE,)).fetchall()

        count = 0
        for row in rows:
            response_type = QUESTION_TYPE_MAP.get(row["question_type"] or "", "admin_action")
            agent_response = row["agent_response"]
            # Guard against legacy ADF JSON stored as plain text in ticket_comments
            if agent_response and agent_response.lstrip().startswith("{"):
                try:
                    import json as _json
                    parsed = _json.loads(agent_response)
                    if isinstance(parsed, dict) and parsed.get("type") == "doc":
                        from ingest.tickets import _extract_adf_text
                        agent_response = _extract_adf_text(parsed)
                except (ValueError, KeyError):
                    pass
            try:
                conn.execute(
                    """INSERT OR IGNORE INTO response_examples
                       (ticket_key, response_type, question_type, category,
                        agent_response, agent_id)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        row["ticket_key"],
                        response_type,
                        row["question_type"],
                        row["category"],
                        agent_response,
                        row["author_id"],
                    ),
                )
                count += 1
            except Exception as e:
                log.debug("Could not harvest example for %s: %s", row["ticket_key"], e)

    if count:
        log.info("Harvested %d organic response examples", count)
    return count


def get_organic_examples(response_type: str | None = None, limit: int = 3) -> list[dict]:
    """Retrieve organic agent response examples for few-shot prompting."""
    with get_db_conn() as conn:
        if response_type:
            rows = conn.execute(
                """SELECT response_type, agent_response
                   FROM response_examples
                   WHERE response_type = ?
                   ORDER BY LENGTH(agent_response) DESC
                   LIMIT ?""",
                (response_type, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT response_type, agent_response
                   FROM response_examples
                   ORDER BY LENGTH(agent_response) DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
    return [dict(r) for r in rows]


def get_feedback_stats() -> dict:
    """Return feedback statistics for the API endpoint."""
    with get_db_conn() as conn:
        categories = conn.execute(
            """SELECT feedback_category, COUNT(*) as count,
                      AVG(similarity_score) as avg_score
               FROM ai_draft_feedback
               WHERE feedback_category IS NOT NULL
               GROUP BY feedback_category"""
        ).fetchall()
        pending = conn.execute(
            "SELECT COUNT(*) FROM ai_draft_feedback WHERE actual_response IS NULL"
        ).fetchone()[0]
        total = conn.execute(
            "SELECT COUNT(*) FROM ai_draft_feedback"
        ).fetchone()[0]

        organic = conn.execute(
            "SELECT COUNT(*) FROM response_examples"
        ).fetchone()[0]
        organic_by_type = conn.execute(
            """SELECT response_type, COUNT(*) as count
               FROM response_examples GROUP BY response_type"""
        ).fetchall()

        emoji_feedback = conn.execute(
            """SELECT agent_feedback, COUNT(*) as count
               FROM ai_draft_feedback
               WHERE agent_feedback IS NOT NULL
               GROUP BY agent_feedback"""
        ).fetchall()

    return {
        "total_drafts": total,
        "pending_capture": pending,
        "categories": {
            r["feedback_category"]: {
                "count": r["count"],
                "avg_score": round(r["avg_score"], 3) if r["avg_score"] else None,
            }
            for r in categories
        },
        "agent_feedback": {r["agent_feedback"]: r["count"] for r in emoji_feedback},
        "organic_examples": {
            "total": organic,
            "by_type": {r["response_type"]: r["count"] for r in organic_by_type},
        },
    }
