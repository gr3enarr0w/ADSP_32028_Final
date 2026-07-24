"""Disposition review — classify ticket status and post review comments."""

import logging

from config import GEMINI_MODEL_GENERATION
from core.genai import get_genai_client
from db import get_db_conn
from ingest.oauth2lo import get_cloud_base_url

from .drafting import _get_ticket_context, _get_latest_customer_reply, _lookup_and_draft
from .jira_comments import _jira_request, _delete_comment, _post_review_comment

log = logging.getLogger(__name__)


def review_ticket(ticket_key: str) -> dict | None:
    """Review a ticket's conversation thread and classify its disposition.

    Returns dict with disposition, summary, recommendation, and
    optional customer_response (for close dispositions).
    """
    from ingest.tickets import _extract_adf_text

    ctx = _get_ticket_context(ticket_key)
    if not ctx:
        return None
    summary, desc = ctx

    # Fetch full comment thread via JSM servicedeskapi
    try:
        base = get_cloud_base_url("jsm")
        resp = _jira_request(
            "get",
            f"{base}/rest/servicedeskapi/request/{ticket_key}/comment",
            "jsm",
            params={"limit": 100},
        )
        if resp.status_code != 200:
            return None

        comments = resp.json().get("values", [])
        thread_parts = []
        for c in comments:
            # JSM uses public:bool (True=public, False=internal)
            is_internal = not c.get("public", True)
            label = "INTERNAL" if is_internal else "PUBLIC"
            author = c.get("author", {}).get("displayName", "Unknown")
            created_raw = c.get("created", {})
            if isinstance(created_raw, dict):
                created = (created_raw.get("iso8601") or "")[:10]
            else:
                created = str(created_raw)[:10]
            body = c.get("body", "")
            text = _extract_adf_text(body) if isinstance(body, dict) else str(body)
            if text.strip():
                thread_parts.append(f"[{label}] {author} ({created}): {text.strip()[:500]}")
    except Exception as e:
        log.warning("Could not fetch comments for review of %s: %s", ticket_key, e)
        return None

    if not thread_parts:
        return {"disposition": "stale", "summary": "No comments on ticket",
                "recommendation": "Close as stale or follow up with customer",
                "customer_response": None}

    thread = "\n\n".join(thread_parts)

    prompt = f"""Review this support ticket conversation and determine its current status.

TICKET: {summary}
DESCRIPTION: {(desc or '')[:2000]}

COMMENT THREAD (chronological):
{thread[:6000]}

Classify this ticket into ONE of these dispositions:
- "close": The issue has been resolved — customer confirmed it's working, the solution was provided and accepted, or the question has been fully answered.
- "sprint_work": This requires development, engineering, or configuration work beyond helpdesk scope — should be moved to a sprint backlog or internal project.
- "needs_action": Customer replied with new information, a follow-up question, or the issue is still unresolved and needs agent attention.
- "stale": No meaningful customer response after agent follow-up. Ticket appears abandoned.

Return valid JSON only (no markdown fencing):
{{
  "disposition": "close" | "sprint_work" | "needs_action" | "stale",
  "summary": "<2-3 sentence summary of what happened on this ticket>",
  "recommendation": "<specific next step for the agent>",
  "customer_response": "<closing message for the customer if disposition is close, otherwise null>"
}}"""

    try:
        client = get_genai_client()
        response = client.models.generate_content(
            model=GEMINI_MODEL_GENERATION,
            contents=prompt,
        )
        from utils.gemini import parse_json_response
        result = parse_json_response(response.text)
        if isinstance(result, list):
            result = result[0] if result else None
        if not isinstance(result, dict) or "disposition" not in result:
            return None
        return result
    except Exception as e:
        log.error("Failed to review ticket %s: %s", ticket_key, e)
        return None


def handle_review_trigger(ticket_key: str, trigger_comment_id: str) -> bool:
    """Handle an on-demand /ai-review trigger from an internal comment."""
    log.info("AI review triggered for %s", ticket_key)
    _delete_comment(ticket_key, trigger_comment_id)

    review = review_ticket(ticket_key)
    if not review:
        log.info("Could not review %s", ticket_key)
        return False

    comment_id = _post_review_comment(ticket_key, review)
    return comment_id is not None


def batch_review_tickets(ticket_keys: list[str]) -> dict:
    """Review multiple tickets and post internal comments.

    Returns summary of dispositions.
    """
    results = {"close": [], "sprint_work": [], "needs_action": [], "stale": [], "failed": []}

    for key in ticket_keys:
        review = review_ticket(key)
        if not review:
            results["failed"].append(key)
            continue

        disposition = review["disposition"]
        comment_id = _post_review_comment(key, review)
        if comment_id:
            results[disposition].append(key)
        else:
            results["failed"].append(key)

    return results


def handle_ai_assist(ticket_key: str, comment_id: str) -> bool:
    """Unified handler — drafts response + reviews disposition in one pass.

    Only fires on agent-side statuses (In Progress, Waiting for support, Pending).
    Fetches ticket status via API, then runs both the draft and review pipelines.
    """
    log.info("AI assist triggered for %s", ticket_key)
    _delete_comment(ticket_key, comment_id)

    # Check ticket status from DB — only fire on agent-side statuses
    try:
        with get_db_conn() as _conn:
            _row = _conn.execute(
                "SELECT status FROM tickets WHERE ticket_key = ?", (ticket_key,)
            ).fetchone()
        status = _row["status"] if _row else ""
        agent_statuses = {"Waiting for support", "In Progress", "Pending"}
        if status and status not in agent_statuses:
            log.info("Ticket %s is in '%s' — AI assist only fires on %s",
                     ticket_key, status, agent_statuses)
            return False
    except Exception as e:
        log.warning("Could not check status for %s: %s", ticket_key, e)
        return False

    # Draft response (with deep doc research via updated _draft_response)
    ctx = _get_ticket_context(ticket_key)
    if not ctx:
        return False
    summary, desc = ctx

    customer_reply = _get_latest_customer_reply(ticket_key)
    if customer_reply:
        desc = desc + "\n\nCUSTOMER FOLLOW-UP:\n" + customer_reply

    draft_posted = _lookup_and_draft(ticket_key, summary, desc, force=True)

    # Review disposition
    review = review_ticket(ticket_key)
    if review:
        _post_review_comment(ticket_key, review)

    return draft_posted or review is not None
