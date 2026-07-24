"""Auto-responder — drafts context-aware responses for new tickets.

When a ticket comes in, looks up matching FAQ/KB/resolved tickets,
then uses Gemini to classify the response type and draft accordingly:

  1. SELF_SERVICE — user can resolve it themselves; draft links them to docs
  2. ADMIN_ACTION — an admin/elevated role needs to make changes;
     draft gives the agent step-by-step admin instructions AND a separate
     customer response confirming what was done
  3. NEEDS_INFO — not enough detail to act; draft asks the customer
     for the specific missing information

Posts the draft as an internal comment for agent review.

Only fires for tickets assigned to agents in AUTO_RESPOND_ASSIGNEES.
FAQ lookup, gap analysis, and source gathering are unaffected by this gate.
"""

import json
import logging
import os
import re
import requests

from config import CLOUD_URL, GEMINI_PROJECT, GEMINI_LOCATION, GEMINI_MODEL_GENERATION, GOOGLE_SERVICE_ACCOUNT_JSON, AUTO_RESPOND_ASSIGNEES, DOC_SOURCE_DOMAINS, DOC_CACHE_DAYS, apply_cloud_terminology, AGE_GATE_HOURS, DRAFT_SELF_CHECK, GEMINI_FLASH_MODEL, AUTO_DRAFT_NOISE_PATTERNS
from db import get_db_conn
from faq.lookup import lookup
from ingest.oauth2lo import get_cloud_auth, get_cloud_base_url, clear_cache as _clear_oauth_cache

log = logging.getLogger(__name__)

QUESTION_TYPE_MAP = {
    "how-to": "self_service",
    "configuration": "admin_action",
    "access-request": "admin_action",
    "troubleshooting": "admin_action",
    "bug-report": "admin_action",
}

AI_LOOKUP_TRIGGER = "/ai-lookup"
AI_REVIEW_TRIGGER = "/ai-review"

FEEDBACK_EMOJI_MAP = {
    "✅": "both_good",
    "👤": "customer_good",
    "🔧": "steps_good",
    "❌": "both_bad",
    "🔄": "wrong_type",
    "❓": "needs_info",
}


AI_ASSIST_EMOJI = "🤖"


def _jira_request(method: str, url: str, product: str, **kwargs) -> requests.Response:
    """Wrapper around requests.get/post with one 401 → token-refresh retry."""
    headers = get_cloud_auth(product)
    kwargs.setdefault("timeout", 30)
    resp = getattr(requests, method)(url, headers=headers, **kwargs)
    if resp.status_code == 401:
        log.warning("Got 401 on %s — clearing token cache and retrying", url)
        _clear_oauth_cache()
        headers = get_cloud_auth(product)
        resp = getattr(requests, method)(url, headers=headers, **kwargs)
    return resp


def _fetch_doc_content(url: str) -> str | None:
    """Fetch and cache doc page content from configured source domains.

    Only fetches from DOC_SOURCE_DOMAINS allowlist. Returns plain text
    content (HTML stripped), or None if not fetchable.
    """
    from urllib.parse import urlparse
    import re
    from datetime import datetime, timezone, timedelta

    parsed = urlparse(url)
    if parsed.hostname not in DOC_SOURCE_DOMAINS:
        return None

    # Check cache
    with get_db_conn() as conn:
        cached = conn.execute(
            "SELECT content, fetched_at FROM doc_content_cache WHERE url = ?",
            (url,),
        ).fetchone()
        if cached:
            fetched = datetime.fromisoformat(cached["fetched_at"].replace("Z", "+00:00"))
            if datetime.now(timezone.utc) - fetched < timedelta(days=DOC_CACHE_DAYS):
                return cached["content"]

    # Fetch fresh
    try:
        resp = requests.get(url, timeout=15, headers={
            "User-Agent": "JSM-Modeling-Bot/1.0 (internal tooling)"
        })
        if resp.status_code != 200:
            return None

        # Strip HTML to plain text
        text = re.sub(r'<script[^>]*>.*?</script>', '', resp.text, flags=re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = text[:3000]

        # Cache it
        now = datetime.now(timezone.utc).isoformat()
        with get_db_conn() as conn:
            conn.execute(
                """INSERT INTO doc_content_cache (url, content, fetched_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT (url) DO UPDATE SET content=EXCLUDED.content, fetched_at=EXCLUDED.fetched_at""",
                (url, text, now),
            )

        return text
    except Exception as e:
        log.debug("Could not fetch doc content from %s: %s", url, e)
        return None


def _get_genai_client():
    from google import genai

    sa_path = GOOGLE_SERVICE_ACCOUNT_JSON
    if not os.path.isabs(sa_path):
        sa_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", sa_path)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(sa_path)

    return genai.Client(
        vertexai=True,
        project=GEMINI_PROJECT,
        location=GEMINI_LOCATION,
    )


def _draft_response(ticket_summary: str, ticket_description: str, matches: dict) -> dict | None:
    """Use Gemini to classify the response type and draft accordingly.

    Returns a dict with:
        response_type: "self_service" | "admin_action" | "needs_info"
        customer_response: str — the draft to send to the customer
        admin_steps: str | None — step-by-step admin instructions (admin_action only)
        missing_info: str | None — what info is needed (needs_info only)
    Returns None if no draft could be produced.
    """
    if not matches.get("found"):
        return None

    # Build context from matches
    context_parts = []

    for faq in matches.get("faq_matches", []):
        context_parts.append(f"FAQ: {faq.get('title', '')}\n{faq.get('body_html', '')[:1000]}")

    fetched_count = 0
    for kb in matches.get("kb_matches", []):
        title = kb.get("title", "")
        url = kb.get("url", "")
        entry = f"KB Article: {title}\nURL: {url}"
        if url and fetched_count < 3:
            content = _fetch_doc_content(url)
            if content:
                entry += f"\nContent: {content[:1500]}"
                fetched_count += 1
        context_parts.append(entry)

    for t in matches.get("ticket_matches", [])[:3]:
        summary = t.get("summary", "")
        resolution = t.get("resolution_summary", "")
        context_parts.append(f"Resolved Ticket: {summary}\nResolution: {resolution}")

    for doc in matches.get("atlassian_matches", []):
        title = doc.get("title", "")
        url = doc.get("url", "")
        entry = f"Official Atlassian Documentation: {title}\nURL: {url}"
        if url and fetched_count < 3:
            content = _fetch_doc_content(url)
            if content:
                entry += f"\nContent: {content[:1500]}"
                fetched_count += 1
        context_parts.append(entry)

    if not context_parts:
        return None

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are a friendly, knowledgeable IT help desk agent for the Atlassian Cloud migration team.
A user has submitted a support ticket. Analyze the ticket and reference material, then classify and draft the appropriate response.

TICKET:
Summary: {ticket_summary}
Description: {(ticket_description or '')[:3000]}

REFERENCE MATERIAL (from FAQ, KB articles, previously resolved tickets, and official Atlassian documentation):
{context[:4000]}

{_build_few_shot_block()}STEP 1 — CLASSIFY the response type:

- "self_service": The user can resolve this themselves with documentation or steps they can perform with their own permissions (e.g. adjusting their own settings, following a guide, using a feature differently).
- "admin_action": An administrator or someone with elevated project/org roles needs to make changes on behalf of the user (e.g. modifying project settings, changing permission schemes, updating workflow configurations, adding groups to roles). The user cannot do this themselves.
- "needs_info": The ticket does not contain enough information to determine the solution or take action. Specific details are needed from the customer before proceeding.

IMPORTANT CONTEXT:
- Our team uses the Atlassian Cloud platform, Rovo MCP Server, and related tools successfully. When customers report issues with these tools, the problem is typically on THEIR side (configuration, permissions, setup).
- There are no OAuth tokens on personal Atlassian accounts. API tokens are the personal auth method.
- Many issues involve BOTH admin and customer actions. Classify based on what the PRIMARY fix requires.

STEP 2 — DRAFT the response. ALWAYS produce BOTH technician_steps AND customer_response:

TECHNICIAN_STEPS (always required — internal diagnostic notes for the support agent):
  - Numbered diagnostic and action steps for the agent/admin to investigate
  - Be specific — include exact navigation paths (e.g. "admin.atlassian.com > Security > API tokens")
  - ALWAYS include branching logic: "If X is already enabled/done, then check Y"
  - Cover multiple scenarios — don't assume the first check will be the fix
  - Include when to escalate
  - CITE YOUR SOURCES: For each step, include the URL from the reference material that supports it. Format: "See: [title](URL)" at the end of the step.
  - Example format:
    1. Check setting A at [path]. If disabled, enable it. See: [Configure SLAs](https://support.atlassian.com/...)
    2. If A was already enabled, check setting B at [path]. See: [Manage Permissions](https://support.atlassian.com/...)
    3. If both are correct, the issue is likely [C] — escalate to [team].

CUSTOMER_RESPONSE (always required — what gets sent to the customer):
  - This is a BRIEF acknowledgment, NOT detailed technical steps
  - For "admin_action" or issues being investigated: Simply acknowledge the issue and confirm the team is looking into it (e.g. "We are looking into this and will follow up with you shortly.")
  - For "self_service" where the user truly can fix it themselves: Provide clear numbered steps with inline links to source documentation
  - For "needs_info": Acknowledge the issue, list specific missing info, explain why it's needed
  - The customer response should NEVER contain the admin/technician diagnostic steps
  - Keep it concise — 1-3 sentences for acknowledgments, up to 300 words for self-service steps

FORMATTING RULES (all types):
- Be warm, professional, and concise
- If the solution involves steps for the customer, use a numbered list
- MANDATORY: Embed source links inline using markdown format [Title](URL). Use the URLs provided in the REFERENCE MATERIAL above. Every instruction or claim MUST cite its source URL.
- Do NOT mention that this is an AI-drafted response
- Do NOT reference internal ticket numbers or internal tools
- Do NOT include any sign-off, greeting, or signature
- TERMINOLOGY: In Jira Cloud, "projects" have been renamed to "spaces". Always use "space" / "spaces" instead of "project" / "projects". Exception: keep "project key" and "project category" as-is.

Return valid JSON only (no markdown fencing):
{{
  "response_type": "self_service" | "admin_action" | "needs_info",
  "customer_response": "<brief acknowledgment OR self-service steps for the customer — include source links>",
  "admin_steps": "<diagnostic and action steps for the technician — include source links and branching logic>",
  "missing_info": "<summary of what's missing, or null if not needs_info>"
}}"""

    try:
        client = _get_genai_client()
        response = client.models.generate_content(
            model=GEMINI_MODEL_GENERATION,
            contents=prompt,
        )

        from utils.gemini import parse_json_response
        result = parse_json_response(response.text)

        if isinstance(result, list):
            result = result[0] if result else None
        if not isinstance(result, dict) or "response_type" not in result:
            # Fallback: treat the whole response as a self-service customer draft
            log.warning("Could not parse structured response, falling back to plain text")
            return {
                "response_type": "self_service",
                "customer_response": response.text.strip(),
                "admin_steps": None,
                "missing_info": None,
            }

        # Normalize response_type
        rt = result.get("response_type", "self_service").lower().replace(" ", "_")
        if rt not in ("self_service", "admin_action", "needs_info"):
            rt = "self_service"
        result["response_type"] = rt

        # Apply Cloud terminology to all text fields
        for key in ("customer_response", "admin_steps", "missing_info"):
            if result.get(key):
                result[key] = apply_cloud_terminology(result[key])

        return result
    except Exception as e:
        log.error("Failed to draft response: %s", e)
        return None


def _post_internal_comment(ticket_key: str, draft: dict) -> str | None:
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
    if admin_steps:
        lines += ["", "--- Technician Steps (do NOT send to customer) ---", admin_steps]
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
            "jira_write",
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


def _text_to_adf_paragraphs(text: str) -> list[dict]:
    """Convert plain text (with optional Markdown links) to ADF paragraph nodes."""
    import re
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
        base = get_cloud_base_url("jsm")
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
        log.debug("Could not parse created_at for %s: %s", ticket_key, e)

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


def _self_check_draft(ticket_summary: str, ticket_description: str, draft: dict) -> tuple[bool, str]:
    """Evaluate draft quality with a cheap Flash model before posting.

    Checks three binary criteria:
      1. Is the response type correct for this ticket?
      2. Does the draft address the customer's question?
      3. Is there enough context to give a useful answer?

    Returns (pass, reason). Fails open on errors or when disabled.
    """
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
        client = _get_genai_client()
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
        log.warning("Self-check failed for draft — passing through: %s", e)
        return True, f"error — fail-open: {e}"


def _get_ticket_context(ticket_key: str) -> tuple[str, str] | None:
    """Fetch ticket summary and description from DB or Jira API.

    Returns (summary, description) or None if unavailable.
    """
    with get_db_conn() as conn:
        ticket = conn.execute(
            "SELECT summary, description FROM tickets WHERE ticket_key = ?",
            (ticket_key,),
        ).fetchone()

    if ticket:
        return ticket["summary"], ticket["description"] or ""

    # Not in DB — fetch from JSM API
    try:
        base = get_cloud_base_url("jsm")
        resp = _jira_request(
            "get",
            f"{base}/rest/servicedeskapi/request/{ticket_key}",
            "jsm",
            params={"expand": "requestFieldValues"},
        )
        if resp.status_code != 200:
            log.warning("Could not fetch %s from JSM: %d", ticket_key, resp.status_code)
            return None

        data = resp.json()
        summary = data.get("summary", "")
        desc = ""
        for field in data.get("requestFieldValues", []):
            if field.get("fieldId") == "description":
                val = field.get("value", "")
                if isinstance(val, dict):
                    from ingest.tickets import _extract_adf_text
                    val = _extract_adf_text(val)
                desc = str(val)
                break
        return summary, desc
    except Exception as e:
        log.warning("Could not fetch %s: %s", ticket_key, e)
        return None


def _get_latest_customer_reply(ticket_key: str) -> str | None:
    """Fetch the most recent public customer comment on a ticket.

    Walks comments in reverse to find the latest public comment
    that is NOT from an agent (i.e., not internal). Returns the
    text or None.
    """
    from ingest.tickets import _extract_adf_text

    try:
        base = get_cloud_base_url("jsm")

        # Get reporter from DB
        reporter_id = None
        with get_db_conn() as conn:
            row = conn.execute(
                "SELECT reporter_id FROM tickets WHERE ticket_key = ?", (ticket_key,)
            ).fetchone()
            if row:
                reporter_id = row["reporter_id"]

        resp = _jira_request(
            "get",
            f"{base}/rest/servicedeskapi/request/{ticket_key}/comment",
            "jsm",
            params={"limit": 50},
        )
        if resp.status_code != 200:
            return None

        comments = resp.json().get("values", [])
        for comment in reversed(comments):
            if not comment.get("public", True):
                continue  # skip internal comments

            author_id = comment.get("author", {}).get("accountId", "")
            if reporter_id and author_id != reporter_id:
                continue  # skip agent public comments, want customer reply

            body = comment.get("body", "")
            text = _extract_adf_text(body) if isinstance(body, dict) else str(body)
            if text.strip():
                return text.strip()

        return None
    except Exception as e:
        log.debug("Could not fetch customer reply for %s: %s", ticket_key, e)
        return None


def _delete_comment(ticket_key: str, comment_id: str):
    """Delete a comment from a ticket (used to clean up trigger comments)."""
    try:
        base = get_cloud_base_url("jira_write")
        resp = _jira_request(
            "delete",
            f"{base}/rest/api/3/issue/{ticket_key}/comment/{comment_id}",
            "jira_write",
        )
        if resp.status_code in (200, 204):
            log.info("Deleted trigger comment %s from %s", comment_id, ticket_key)
        else:
            log.warning("Could not delete comment %s from %s: %d",
                        comment_id, ticket_key, resp.status_code)
    except Exception as e:
        log.debug("Could not delete comment %s from %s: %s", comment_id, ticket_key, e)


def _has_pending_draft(ticket_key: str) -> bool:
    """Check if there's already an unactioned AI draft for this ticket."""
    with get_db_conn() as conn:
        row = conn.execute(
            "SELECT id FROM ai_draft_feedback WHERE ticket_key = ? AND actual_response IS NULL",
            (ticket_key,),
        ).fetchone()
    return row is not None


def _lookup_and_draft(ticket_key: str, summary: str, desc: str, force: bool = False) -> bool:
    """Shared logic: FAQ lookup → Gemini draft → post internal comment.

    Returns True if a draft was posted. Skips if a pending draft already
    exists unless force=True (e.g., triggered via /ai-lookup).
    """
    if not force and _has_pending_draft(ticket_key):
        log.info("Ticket %s already has a pending AI draft — skipping", ticket_key)
        return False
    matches = lookup(summary)
    if not matches.get("found"):
        if desc:
            words = " ".join(desc.split()[:20])
            matches = lookup(words)

    if not matches.get("found"):
        log.info("No matches found for %s — skipping auto-response", ticket_key)
        return False

    draft = _draft_response(summary, desc, matches)
    if not draft:
        log.info("Could not draft response for %s", ticket_key)
        return False

    passed, reason = _self_check_draft(summary, desc, draft)
    if not passed:
        log.info("Draft for %s failed self-check: %s — not posting", ticket_key, reason)
        return False

    log.info("Drafted %s response for %s", draft["response_type"], ticket_key)

    comment_id = _post_internal_comment(ticket_key, draft)
    if comment_id:
        _store_draft_record(ticket_key, comment_id, draft)
        return True
    return False


def handle_new_ticket(ticket_key: str) -> bool:
    """Main entry point — look up matches, draft response, post to ticket.

    Only drafts responses for tickets assigned to allowed agents
    (AUTO_RESPOND_ASSIGNEES). Returns True if a draft was posted, False otherwise.
    """
    log.info("Auto-responder processing %s", ticket_key)

    allowed, assignee_id = _check_assignee_allowed(ticket_key)
    if not allowed:
        return False

    if not _check_status_gate(ticket_key):
        return False

    if not _check_age_gate(ticket_key):
        return False

    if not _check_content_gate(ticket_key):
        return False

    from core.pipeline import get_plugin_config
    from plugins.responder.gates import (
        _get_confidence_threshold,
        check_classification_gates,
    )

    responder_cfg = get_plugin_config("responder")
    confidence_threshold = _get_confidence_threshold(responder_cfg)
    if not check_classification_gates(ticket_key, responder_cfg):
        return False

    # M5 Router: confidence + sentiment + novelty gate
    from analysis.router import route_ticket
    routing = route_ticket(ticket_key, confidence_threshold=confidence_threshold)
    if routing.route == "human_review":
        log.info(
            "Ticket %s → human_review (M5 router: %s) — skipping auto-draft",
            ticket_key, routing.reason,
        )
        return False
    if routing.route == "flag_only":
        log.info(
            "Ticket %s → flag_only (M5 router: %s) — skipping auto-draft",
            ticket_key, routing.reason,
        )
        return False

    if _has_agent_response(ticket_key, assignee_account_id=assignee_id):
        return False

    context = _get_ticket_context(ticket_key)
    if not context:
        return False
    summary, desc = context

    return _lookup_and_draft(ticket_key, summary, desc)


def handle_ai_lookup_trigger(ticket_key: str, trigger_comment_id: str) -> bool:
    """Handle an on-demand /ai-lookup trigger from an internal comment.

    Skips the _has_agent_response check (technician explicitly requested).
    Includes the latest customer reply as additional context.
    Deletes the trigger comment after processing.
    """
    log.info("AI lookup triggered for %s (comment %s)", ticket_key, trigger_comment_id)

    allowed, _ = _check_assignee_allowed(ticket_key)
    if not allowed:
        _delete_comment(ticket_key, trigger_comment_id)
        return False

    context = _get_ticket_context(ticket_key)
    if not context:
        _delete_comment(ticket_key, trigger_comment_id)
        return False
    summary, desc = context

    # Append latest customer reply as follow-up context
    customer_reply = _get_latest_customer_reply(ticket_key)
    if customer_reply:
        desc = desc + "\n\nCUSTOMER FOLLOW-UP:\n" + customer_reply

    result = _lookup_and_draft(ticket_key, summary, desc, force=True)
    _delete_comment(ticket_key, trigger_comment_id)
    return result


def handle_feedback_emoji(ticket_key: str, comment_id: str, emoji: str) -> bool:
    """Record agent feedback on an AI draft via emoji comment.

    Finds the most recent draft for the ticket, records the feedback
    category, and deletes the emoji comment.
    """
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


# ── Ticket review ─────────────────────────────────────────────


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
        client = _get_genai_client()
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
            "jira_write",
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
    """Unified 🤖 handler — drafts response + reviews disposition in one pass.

    Only fires on agent-side statuses (In Progress, Waiting for support, Pending).
    Fetches ticket status via API, then runs both the draft and review pipelines.
    """
    log.info("AI assist (🤖) triggered for %s", ticket_key)
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
            log.info("Ticket %s is in '%s' — 🤖 only fires on %s", ticket_key, status, agent_statuses)
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


# ── Feedback loop ──────────────────────────────────────────────


def _store_draft_record(ticket_key: str, comment_id: str, draft: dict):
    """Insert a pending feedback row after posting a draft."""
    with get_db_conn() as conn:
        conn.execute(
            """INSERT OR IGNORE INTO ai_draft_feedback
               (ticket_key, draft_comment_id, response_type,
                draft_customer_response, draft_admin_steps)
               VALUES (?, ?, ?, ?, ?)""",
            (
                ticket_key,
                comment_id,
                draft["response_type"],
                draft.get("customer_response", ""),
                draft.get("admin_steps"),
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
    from datetime import datetime, timezone
    from ingest.tickets import _extract_adf_text

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


def _build_few_shot_block() -> str:
    """Build few-shot examples block for the Gemini prompt.

    Pulls up to 5 total examples, prioritizing draft-feedback pairs
    (which have correction signal) and filling remaining slots with
    organic agent response examples from resolved tickets.
    Returns empty string during cold start with no examples.
    """
    examples = []
    for rt in ("self_service", "admin_action", "needs_info"):
        examples.extend(get_few_shot_examples(rt, limit=2))
        if len(examples) >= 5:
            break

    # Fill remaining slots with organic examples
    remaining = 5 - len(examples)
    if remaining > 0:
        organic = get_organic_examples(response_type=None, limit=remaining)
        examples.extend(organic)

    if not examples:
        return ""

    lines = ["EXAMPLES FROM PAST RESPONSES (these show the style our agents prefer):\n"]
    for ex in examples[:5]:
        rt_label = ex["response_type"].upper().replace("_", " ")
        if "draft_customer_response" in ex:
            # Draft-feedback pair
            lines.append(f"[{rt_label} example]")
            lines.append(f"AI Draft: {ex['draft_customer_response'][:500]}")
            lines.append(f"Agent's Final Version: {ex['actual_response'][:500]}")
        else:
            # Organic agent response
            lines.append(f"[{rt_label} example]")
            lines.append(f"Agent Response: {ex['agent_response'][:500]}")
        lines.append("")
    lines.append("")
    return "\n".join(lines)


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
                   OR t.affect_version LIKE '%UAT%'
                   OR t.affect_version = 'Live 1.0'
                   OR t.request_type LIKE '%Cloud%')
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


