"""Jira API interactions — HTTP helper, comment posting and deletion."""

import logging
import re
import requests

from ingest.oauth2lo import get_cloud_auth, get_cloud_base_url, clear_cache as _clear_oauth_cache

log = logging.getLogger(__name__)


def _jira_request(method: str, url: str, product: str, **kwargs) -> requests.Response:
    """Wrapper around requests.get/post with one 401 -> token-refresh retry."""
    headers = get_cloud_auth(product)
    kwargs.setdefault("timeout", 30)
    resp = getattr(requests, method)(url, headers=headers, **kwargs)
    if resp.status_code == 401:
        log.warning("Got 401 on %s — clearing token cache and retrying", url)
        _clear_oauth_cache()
        headers = get_cloud_auth(product)
        resp = getattr(requests, method)(url, headers=headers, **kwargs)
    return resp


def _delete_comment(ticket_key: str, comment_id: str):
    """Delete a comment from a ticket (used to clean up trigger comments)."""
    try:
        base = get_cloud_base_url("jira_write")
        resp = _jira_request(
            "delete",
            f"{base}/rest/api/3/issue/{ticket_key}/comment/{comment_id}",
            "jsm",
        )
        if resp.status_code in (200, 204):
            log.info("Deleted trigger comment %s from %s", comment_id, ticket_key)
        else:
            log.warning("Could not delete comment %s from %s: %d",
                        comment_id, ticket_key, resp.status_code)
    except Exception as e:
        log.debug("Could not delete comment %s from %s: %s", comment_id, ticket_key, e)


def _post_draft_comment(ticket_key: str, draft: dict) -> str | None:
    """Post a structured internal comment based on response type.

    Formats the comment differently for each response type:
    - self_service: customer draft only
    - admin_action: admin steps panel + customer draft panel
    - needs_info: customer draft (requesting info)

    Returns the comment ID on success, None on failure.
    """
    response_type = draft["response_type"]
    customer_response = draft.get("customer_response", "")
    admin_steps = draft.get("admin_steps")

    type_labels = {
        "self_service": "Self-Service Response",
        "admin_action": "Admin Action Required",
        "needs_info": "More Information Needed",
    }
    type_label = type_labels.get(response_type, "AI-Drafted Response")

    # Build plain-text body for JSM API (public:false = internal)
    lines = [f"[AI DRAFT: {type_label}] — Review and edit before sending to customer:"]

    sources = draft.get("sources") or []
    if admin_steps:
        admin_section = admin_steps
        if sources:
            unique_sources = list(dict.fromkeys(sources))[:5]
            admin_section += "\n\n*Sources used:*\n" + "\n".join(f"- {u}" for u in unique_sources)
        lines += ["", "--- Technician Steps (do NOT send to customer) ---", admin_section]
    elif sources:
        # No admin section — append sources as a trailing block below the customer response
        unique_sources = list(dict.fromkeys(sources))[:5]
        sources_block = "\n\n*Sources used:*\n" + "\n".join(f"- {u}" for u in unique_sources)
        customer_response = customer_response + sources_block

    lines += ["", "--- Customer Response ---", customer_response]
    lines += [
        "",
        "---",
        "Rate this draft — post one emoji as an internal comment:",
        "✅ Both good │ 👤 Customer only │ 🔧 Steps only │ ❌ Both off │ 🔄 Wrong type │ ❓ Needs info",
    ]
    body_text = "\n".join(lines)

    base = get_cloud_base_url("jira_write")

    try:
        resp = _jira_request(
            "post",
            f"{base}/rest/servicedeskapi/request/{ticket_key}/comment",
            "jsm",
            json={"body": body_text, "public": False},
        )
        if resp.status_code in (200, 201):
            comment_id = resp.json().get("id")
            log.info("Posted %s draft to %s (comment %s)", response_type, ticket_key, comment_id)
            return comment_id
        else:
            log.error("Failed to post comment on %s: %d — %s",
                      ticket_key, resp.status_code, resp.text[:300])
            return None
    except Exception as e:
        log.error("Failed to post comment on %s: %s", ticket_key, e)
        return None


def _post_review_comment(ticket_key: str, review: dict) -> str | None:
    """Post a color-coded review comment on a ticket."""
    disposition = review["disposition"]
    summary = review.get("summary", "")
    recommendation = review.get("recommendation", "")
    customer_response = review.get("customer_response")

    panel_config = {
        "close": ("success", "Ready to Close"),
        "sprint_work": ("warning", "Sprint Work — Not Helpdesk"),
        "needs_action": ("note", "Needs Agent Action"),
        "stale": ("info", "Stale — No Customer Response"),
    }
    panel_type, label = panel_config.get(disposition, ("info", "Review"))

    # Build plain-text body for JSM API (public:false = internal)
    lines = [
        f"[AI REVIEW: {label}]",
        "",
        "Status Summary:",
        summary,
        "",
        "Recommended Next Step:",
        recommendation,
    ]
    if customer_response and disposition == "close":
        lines += ["", "--- Draft Closing Response ---", customer_response]
    lines += [
        "",
        "---",
        "Rate this review — post one emoji as an internal comment:",
        "✅ Agree │ ❌ Disagree │ 🔄 Wrong disposition",
    ]
    body_text = "\n".join(lines)

    base = get_cloud_base_url("jira_write")

    try:
        resp = _jira_request(
            "post",
            f"{base}/rest/servicedeskapi/request/{ticket_key}/comment",
            "jsm",
            json={"body": body_text, "public": False},
        )
        if resp.status_code in (200, 201):
            comment_id = resp.json().get("id")
            log.info("Posted review (%s) to %s (comment %s)", disposition, ticket_key, comment_id)
            return comment_id
        else:
            log.error("Failed to post review on %s: %d — %s",
                      ticket_key, resp.status_code, resp.text[:300])
            return None
    except Exception as e:
        log.error("Failed to post review on %s: %s", ticket_key, e)
        return None


def _post_self_check_and_decide(ticket_summary: str, ticket_description: str, draft: dict) -> tuple[bool, str]:
    """Evaluate draft quality with a cheap Flash model before posting.

    Checks three binary criteria:
      1. Is the response type correct for this ticket?
      2. Does the draft address the customer's question?
      3. Is there enough context to give a useful answer?

    Returns (pass, reason). Fails open on errors or when disabled.
    """
    from config import DRAFT_SELF_CHECK, GEMINI_FLASH_MODEL
    from core.genai import get_genai_client

    if not DRAFT_SELF_CHECK:
        return True, "disabled"

    customer_response = draft.get("customer_response", "")
    response_type = draft.get("response_type", "")
    if not customer_response:
        return False, "empty draft"

    prompt = f"""You are a quality gate for an AI helpdesk auto-responder. Evaluate this draft response.

TICKET SUMMARY: {ticket_summary}
TICKET DESCRIPTION: {(ticket_description or '')[:1500]}

DRAFT RESPONSE TYPE: {response_type}
DRAFT CUSTOMER RESPONSE: {customer_response[:1500]}

Evaluate THREE criteria (answer true/false for each):
1. type_correct: Is "{response_type}" the right classification? (self_service = user can fix it themselves, admin_action = admin must act, needs_info = missing details)
2. addresses_question: Does the draft actually address what the customer asked about?
3. sufficient_context: Does the draft contain enough specific information to be useful (not just generic platitudes)?

Return valid JSON only (no markdown fencing):
{{"type_correct": true/false, "addresses_question": true/false, "sufficient_context": true/false, "reason": "one-line explanation if any criterion is false, otherwise null"}}"""

    try:
        client = get_genai_client()
        response = client.models.generate_content(
            model=GEMINI_FLASH_MODEL,
            contents=prompt,
        )
        from utils.gemini import parse_json_response
        result = parse_json_response(response.text)
        if isinstance(result, list):
            result = result[0] if result else {}
        if not isinstance(result, dict):
            return True, "unparseable response — fail-open"

        if not result.get("type_correct", True):
            return False, f"wrong response type: {result.get('reason', 'type mismatch')}"
        if not result.get("addresses_question", True):
            return False, f"does not address question: {result.get('reason', 'off-topic')}"
        if not result.get("sufficient_context", True):
            return False, f"insufficient context: {result.get('reason', 'too generic')}"

        return True, "all checks passed"
    except Exception as e:
        log.warning("Self-check failed for draft on %s — passing through: %s", ticket_summary[:80], e)
        return True, f"error — fail-open: {e}"


def _text_to_adf_paragraphs(text: str) -> list[dict]:
    """Convert plain text (with optional Markdown links) to ADF paragraph nodes."""
    _md_link = re.compile(r"\[([^\]]+)\]\((https?://[^\)]+)\)")

    def _inline_nodes(line: str) -> list[dict]:
        nodes = []
        pos = 0
        for m in _md_link.finditer(line):
            if m.start() > pos:
                nodes.append({"type": "text", "text": line[pos:m.start()]})
            nodes.append({
                "type": "text",
                "text": m.group(1),
                "marks": [{"type": "link", "attrs": {"href": m.group(2)}}],
            })
            pos = m.end()
        if pos < len(line):
            nodes.append({"type": "text", "text": line[pos:]})
        return nodes or [{"type": "text", "text": line}]

    paragraphs = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        paragraphs.append({"type": "paragraph", "content": _inline_nodes(line)})
    return paragraphs or [{"type": "paragraph", "content": [{"type": "text", "text": text}]}]


def _has_agent_response(ticket_key: str, assignee_account_id: str | None = None) -> bool:
    """Check if the current assignee already posted a public comment.

    Escalated tickets may have public comments from a previous agent —
    those don't count. Only skips if the current assignee (by accountId)
    has already responded publicly. If no assignee_account_id is provided,
    falls back to checking for any public comment.
    """
    try:
        base = get_cloud_base_url("jira_write")
        resp = _jira_request(
            "get",
            f"{base}/rest/servicedeskapi/request/{ticket_key}/comment",
            "jsm",
            params={"limit": 50},
        )
        if resp.status_code != 200:
            log.debug("Could not fetch comments for %s: %d", ticket_key, resp.status_code)
            return False

        comments = resp.json().get("values", [])
        for comment in comments:
            # JSM API uses public:bool (True=public, False=internal)
            if not comment.get("public", True):
                continue

            # If we know the assignee, only match their comments
            if assignee_account_id:
                author_id = comment.get("author", {}).get("accountId", "")
                if author_id == assignee_account_id:
                    log.info("Ticket %s already has a public comment from current assignee — skipping",
                             ticket_key)
                    return True
            else:
                log.info("Ticket %s already has a public comment — skipping auto-response",
                         ticket_key)
                return True
        return False
    except Exception as e:
        log.debug("Could not check comments for %s: %s", ticket_key, e)
        return False


def _check_assignee_allowed(ticket_key: str) -> tuple[bool, str | None]:
    """Check if the ticket's assignee is in the allowed list.

    Reads assignee_id (accountId) from the DB — populated on every ingest.
    accountId is an opaque technical identifier, not PII, so the scrubber
    never anonymizes it (unlike display names).  Returns (allowed, account_id).
    """
    from config import AUTO_RESPOND_ASSIGNEES, AUTO_DRAFT_ALL
    from db import get_db_conn

    if AUTO_DRAFT_ALL:
        return True, None  # all tickets allowed

    if not AUTO_RESPOND_ASSIGNEES:
        return True, None  # no restriction configured

    with get_db_conn() as conn:
        row = conn.execute(
            "SELECT assignee_id FROM tickets WHERE ticket_key = ?", (ticket_key,)
        ).fetchone()

    if row:
        account_id = row["assignee_id"] or ""
        if account_id in AUTO_RESPOND_ASSIGNEES:
            return True, account_id
        if account_id:
            log.info("Ticket %s assigned to %s — not in allowed list, skipping",
                     ticket_key, account_id)
            return False, account_id

    log.info("Ticket %s assignee unknown (not yet ingested) — skipping", ticket_key)
    return False, None
