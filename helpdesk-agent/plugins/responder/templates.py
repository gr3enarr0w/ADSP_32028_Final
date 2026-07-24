"""Response template system — template-fill draft mode for high-confidence ticket types.

When a ticket's classification confidence meets ``template_confidence_threshold``
and a matching row exists in ``response_templates``, slot values are filled from
lookup matches and ticket text — bypassing full Gemini generation.
"""

import hashlib
import logging
import re
from html.parser import HTMLParser

import requests

from config import apply_cloud_terminology
from core.pipeline import get_plugin_config
from db import get_db_conn
from ingest.oauth2lo import get_cloud_auth, get_cloud_base_url

from .feedback import QUESTION_TYPE_MAP

log = logging.getLogger(__name__)

SLOT_PATTERN = re.compile(r"\[(STEPS|SYSTEM_NAME|KB_URL)\]")
_CATEGORY_LINE = re.compile(r"^Category:\s*(.+)$", re.IGNORECASE | re.MULTILINE)

SEED_TEMPLATES: list[dict[str, str]] = [
    {
        "template_name": "access_request",
        "category": "Access",
        "issue_type": "*",
        "template_body": """---CUSTOMER---
Thank you for your access request regarding [SYSTEM_NAME]. Our team is reviewing your request and will follow up shortly.

---ADMIN---
Access verification steps:
[STEPS]""",
    },
    {
        "template_name": "permissions_change",
        "category": "Permissions",
        "issue_type": "*",
        "template_body": """---CUSTOMER---
We received your permissions request for [SYSTEM_NAME]. A team member will verify the required changes and update you soon.

---ADMIN---
Permission change checklist:
[STEPS]""",
    },
    {
        "template_name": "configuration_howto",
        "category": "Configuration",
        "issue_type": "*",
        "template_body": """---CUSTOMER---
For configuration guidance on [SYSTEM_NAME], please follow these steps:

[STEPS]

---ADMIN---
If the customer still needs help after following the steps above:
1. Verify the customer's project role and product access in admin.atlassian.com
2. Check for org-level policies that may override project settings
3. Escalate to the platform team if the configuration requires admin-level changes""",
    },
]


def seed_templates() -> int:
    """Insert default templates (idempotent). Returns number of rows inserted."""
    inserted = 0
    with get_db_conn() as conn:
        for tpl in SEED_TEMPLATES:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO response_templates
                    (template_name, category, issue_type, template_body)
                VALUES (?, ?, ?, ?)
                """,
                (tpl["template_name"], tpl["category"], tpl["issue_type"], tpl["template_body"]),
            )
            if getattr(cur, "rowcount", 0):
                inserted += cur.rowcount

    if inserted:
        log.info("Seeded %d response template(s)", inserted)
    return inserted


class _TemplateHTMLParser(HTMLParser):
    """Minimal HTML parser that splits Confluence storage HTML into h2-delimited sections."""

    def __init__(self):
        super().__init__()
        self.sections = []
        self._in_h2 = False
        self._current_title = []
        self._current_body = []

    def handle_starttag(self, tag, attrs):
        if tag == "h2":
            if self._current_title or self._current_body:
                self.sections.append(("".join(self._current_title), "".join(self._current_body)))
            self._current_title = []
            self._current_body = []
            self._in_h2 = True
        elif tag == "br":
            if not self._in_h2:
                self._current_body.append("\n")

    def handle_endtag(self, tag):
        if tag == "h2":
            self._in_h2 = False
        elif tag in ("p", "li", "h1", "h3", "h4", "h5", "h6", "div", "tr"):
            if not self._in_h2:
                self._current_body.append("\n")

    def handle_data(self, data):
        if self._in_h2:
            self._current_title.append(data)
        else:
            self._current_body.append(data)

    def get_sections(self):
        if self._current_title or self._current_body:
            self.sections.append(("".join(self._current_title), "".join(self._current_body)))
        return self.sections


def _clean_plain_text(text: str) -> str:
    """Strip trailing whitespace from each line and normalise surrounding whitespace."""
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines).strip()


def _parse_category_fields(category_line: str) -> tuple[str, str]:
    """Parse ``Category: …`` line into (category, issue_type)."""
    rest = category_line.split(":", 1)[-1].strip()
    if "/" in rest:
        category, issue_type = (part.strip() for part in rest.split("/", 1))
        return category, issue_type or "*"
    return rest, "*"


def _split_category_and_body(section_plain: str) -> tuple[str, str, str] | None:
    """Return (category, issue_type, template_body) from a template section."""
    match = _CATEGORY_LINE.search(section_plain)
    if not match:
        return None

    category, issue_type = _parse_category_fields(match.group(0))
    body = section_plain[match.end():].strip()
    if not category or not body:
        return None
    return category, issue_type, body


def _parse_confluence_templates(storage_html: str) -> list[dict[str, str]]:
    """Parse Confluence storage HTML into response template dicts."""
    parser = _TemplateHTMLParser()
    parser.feed(storage_html)

    templates: list[dict[str, str]] = []

    # Ignore the first implicit section before the first <h2>
    sections = parser.get_sections()
    if sections and not sections[0][0]:
        sections = sections[1:]

    for raw_title, raw_body in sections:
        template_name = _clean_plain_text(raw_title)
        section_plain = _clean_plain_text(raw_body)

        parsed = _split_category_and_body(section_plain)
        if not template_name or not parsed:
            log.warning("Skipping malformed template section %r", template_name or section_plain[:80])
            continue

        category, issue_type, template_body = parsed
        templates.append(
            {
                "template_name": template_name,
                "category": category,
                "issue_type": issue_type,
                "template_body": template_body,
            }
        )

    return templates


def _template_sync_state_key(cloud_id: str, page_id: str) -> str:
    return f"template_sync:{cloud_id}:{page_id}"


def sync_templates_from_confluence(page_id: str) -> dict:
    """Fetch template page from Confluence, parse sections, upsert to DB.

    Uses ``job_state.last_run_date`` to store the SHA-256 hash of the page body
    (key ``template_sync:{cloud_id}:{page_id}``) so unchanged pages skip re-parsing.

    The hash check, upsert, stale-row delete, and hash write all execute inside a
    single ``get_db_conn()`` context so they are atomic — a crash mid-way rolls
    back cleanly and the next cycle retries from scratch.

    Returns ``{"synced": N, "skipped": 0|1, "error": str|None}``.
    """
    page_id = (page_id or "").strip()
    if not page_id:
        return {"synced": 0, "skipped": 0, "error": "missing page_id"}

    try:
        headers = get_cloud_auth("confluence_search")
        base = get_cloud_base_url("confluence_search")
        url = f"{base}/wiki/api/v2/pages/{page_id}?body-format=storage"
        resp = requests.get(url, headers=headers, timeout=60)

        if resp.status_code == 401:
            return {"synced": 0, "skipped": 0, "error": f"HTTP 401: {resp.text[:200]}"}
        if resp.status_code != 200:
            return {"synced": 0, "skipped": 0, "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}

        storage = resp.json().get("body", {}).get("storage", {}).get("value", "")
        body_hash = hashlib.sha256(storage.encode("utf-8")).hexdigest()
        # Derive cloud_id from the base URL (e.g. "https://<cloud_id>.atlassian.net")
        _cloud_id = base.split("//", 1)[-1].split(".")[0] if base else "unknown"
        state_key = _template_sync_state_key(_cloud_id, page_id)

        with get_db_conn() as conn:
            # Hash check, parse, upsert, delete, and hash write are all in one
            # transaction so a mid-cycle crash leaves the DB in a consistent state.
            row = conn.execute(
                "SELECT last_run_date FROM job_state WHERE job_name = ?",
                (state_key,),
            ).fetchone()
            if row and row["last_run_date"] == body_hash:
                return {"synced": 0, "skipped": 1, "error": None}

            templates = _parse_confluence_templates(storage)
            if not templates:
                return {"synced": 0, "skipped": 0, "error": "no valid template sections found"}

            template_names = [tpl["template_name"] for tpl in templates]
            placeholders = ",".join("?" for _ in template_names)

            params = [
                (
                    tpl["template_name"],
                    tpl["category"],
                    tpl["issue_type"],
                    tpl["template_body"],
                )
                for tpl in templates
            ]

            conn.executemany(
                """
                INSERT INTO response_templates
                    (template_name, category, issue_type, template_body)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(template_name) DO UPDATE SET
                  category = excluded.category,
                  issue_type = excluded.issue_type,
                  template_body = excluded.template_body
                """,
                params,
            )
            synced = len(params)

            conn.execute(
                f"DELETE FROM response_templates WHERE template_name NOT IN ({placeholders})",
                tuple(template_names),
            )

            conn.execute(
                """
                INSERT INTO job_state (job_name, last_run_date) VALUES (?, ?)
                ON CONFLICT(job_name) DO UPDATE SET
                  last_run_date = excluded.last_run_date
                """,
                (state_key, body_hash),
            )

        log.info(
            "Synced %d response template(s) from Confluence page %s",
            synced,
            page_id,
        )
        return {"synced": synced, "skipped": 0, "error": None}
    except Exception as exc:
        log.exception("Template sync failed for Confluence page %s", page_id)
        return {"synced": 0, "skipped": 0, "error": str(exc)}


def find_template(category: str, issue_type: str) -> dict | None:
    """Return the best matching template for a classification, or None."""
    category = (category or "").strip()
    issue_type = (issue_type or "").strip()
    if not category:
        return None

    with get_db_conn() as conn:
        if issue_type:
            row = conn.execute(
                """
                SELECT template_name, category, issue_type, template_body
                FROM response_templates
                WHERE category = ? AND issue_type = ?
                LIMIT 1
                """,
                (category, issue_type),
            ).fetchone()
            if row:
                return dict(row)

        row = conn.execute(
            """
            SELECT template_name, category, issue_type, template_body
            FROM response_templates
            WHERE category = ? AND issue_type = '*'
            LIMIT 1
            """,
            (category,),
        ).fetchone()

    return dict(row) if row else None


def fill_slots(template_body: str, slots: dict[str, str]) -> str:
    """Replace [STEPS], [SYSTEM_NAME], and [KB_URL] placeholders."""
    def _replace(match: re.Match) -> str:
        return slots.get(match.group(1), match.group(0))

    filled = SLOT_PATTERN.sub(_replace, template_body)
    lines = [
        line for line in filled.splitlines()
        if not line.endswith(": ")
    ]
    return "\n".join(lines)


def _extract_kb_link(matches: dict) -> tuple[str, str]:
    """Return (markdown_link, url) from lookup matches.

    Returns (\"\", \"\") when no source is available.
    """
    for source, url_key, title_key in (
        ("faq_matches", "confluence_url", "title"),
        ("kb_matches", "url", "title"),
        ("atlassian_matches", "url", "title"),
    ):
        for item in matches.get(source, []):
            url = item.get(url_key) or item.get("url")
            if url:
                title = item.get(title_key) or "documentation"
                return f"[{title}]({url})", url

    return "", ""


def _extract_steps(matches: dict, category: str = "") -> str:
    """Build numbered admin/self-service steps from lookup context.

    ``category`` selects category-specific fallback steps when no lookup matches
    are available. Accepted values: ``"Access"``, ``"Permissions"``,
    ``"Configuration"`` (case-insensitive); defaults to the Access fallback.
    """
    lines: list[str] = []

    for ticket in matches.get("ticket_matches", [])[:1]:
        resolution = (ticket.get("resolution_summary") or "").strip()
        if resolution:
            lines.append(f"1. Review prior resolution: {resolution}")

    for kb in matches.get("kb_matches", [])[:1]:
        title = kb.get("title") or "KB article"
        url = kb.get("url") or ""
        if url:
            lines.append(f"{len(lines) + 1}. Consult [{title}]({url})")
        else:
            lines.append(f"{len(lines) + 1}. Consult KB article: {title}")

    for faq in matches.get("faq_matches", [])[:1]:
        title = faq.get("title") or "FAQ article"
        url = faq.get("confluence_url") or ""
        if url:
            lines.append(f"{len(lines) + 1}. Follow [{title}]({url})")

    if not lines:
        cat = category.strip().lower()
        if cat == "configuration":
            lines = [
                "1. Identify the specific configuration setting being requested",
                "2. Check if the change affects a shared scheme used by multiple projects — if so, coordinate before modifying",
                "3. Make the change in Jira/Confluence admin and verify it works in the affected project",
                "4. Escalate to the platform team if global admin access is required",
            ]
        elif cat == "permissions":
            lines = [
                "1. Identify the exact permission change needed (project role, space permission, group membership, or product access)",
                "2. Verify the requestor has a valid business need — do not assign direct user permissions, use groups",
                "3. Make the change via admin.atlassian.com > Directory > Groups or the project/space permission scheme",
                "4. Confirm the change took effect and resolve the ticket",
            ]
        else:  # Access and default
            lines = [
                "1. Confirm the requestor's identity and role via their Rover profile",
                "2. Verify the resource they are requesting access to (Jira project, Confluence space, or application)",
                "3. Provision access and confirm with the requestor that it is working before resolving",
            ]

    return "\n".join(lines)


def _extract_system_name(summary: str, description: str) -> str:
    """Derive a short system/resource label from ticket text.

    Searches summary first, then description, to avoid greedy regex matching
    across sentence boundaries.
    """
    if not summary and not description:
        return "the requested resource"

    patterns = (
        r"access to (.+?)(?:\.|$|,|\n)",
        r"permission[s]? (?:for|to) (.+?)(?:\.|$|,|\n)",
        r"configure (.+?)(?:\.|$|,|\n)",
    )

    # Try matching in summary first (most specific context)
    for pattern in patterns:
        match = re.search(pattern, summary or "", re.IGNORECASE)
        if match:
            return match.group(1).strip()[:80]

    # Fall back to description if summary had no matches
    for pattern in patterns:
        match = re.search(pattern, description or "", re.IGNORECASE)
        if match:
            return match.group(1).strip()[:80]

    # Try quoted text in summary first, then description
    for text in (summary or "", description or ""):
        quoted = re.search(r'"([^"]+)"', text)
        if quoted:
            return quoted.group(1).strip()[:80]

    return (summary or "the requested resource")[:80]


def build_slots(summary: str, description: str, matches: dict, category: str = "") -> dict[str, str]:
    """Build slot values for template fill from ticket text and lookup matches.

    ``category`` is passed through to ``_extract_steps`` to select
    category-appropriate fallback steps when no lookup context is available.
    """
    kb_link, _url = _extract_kb_link(matches)
    return {
        "STEPS": _extract_steps(matches, category=category),
        "SYSTEM_NAME": _extract_system_name(summary, description),
        "KB_URL": kb_link,
    }


def _parse_filled_body(filled_body: str) -> tuple[str, str | None]:
    """Split a filled template into customer and admin sections."""
    if "---CUSTOMER---" in filled_body and "---ADMIN---" in filled_body:
        _, customer_part = filled_body.split("---CUSTOMER---", 1)
        customer_text, admin_text = customer_part.split("---ADMIN---", 1)
        return customer_text.strip(), admin_text.strip() or None

    return filled_body.strip(), None


def _response_type_for(question_type: str | None, has_admin: bool) -> str:
    mapped = QUESTION_TYPE_MAP.get((question_type or "").lower())
    if mapped:
        return mapped
    if has_admin:
        return "admin_action"
    return "self_service"


def try_template_draft(
    ticket_key: str,
    summary: str,
    description: str,
    matches: dict | None,
) -> dict | None:
    """Attempt template-fill drafting. Returns a draft dict or None to fall back to Gemini."""
    cfg = get_plugin_config("responder")
    if not cfg.get("templates_enabled", True):
        return None

    matches = matches or {}

    threshold_raw = cfg.get("template_confidence_threshold", 0.75)
    try:
        threshold = float(threshold_raw)
    except (TypeError, ValueError):
        log.warning(
            "Invalid template_confidence_threshold %r — template-fill disabled",
            threshold_raw,
        )
        return None

    with get_db_conn() as conn:
        cls = conn.execute(
            """
            SELECT category, issue_type, question_type, confidence
            FROM ticket_classifications
            WHERE ticket_key = ?
            """,
            (ticket_key,),
        ).fetchone()

    if not cls:
        return None

    confidence = float(cls["confidence"] or 0.0)
    if confidence < threshold:
        log.debug(
            "Ticket %s confidence %.2f below template threshold %.2f — skipping template-fill",
            ticket_key,
            confidence,
            threshold,
        )
        return None

    template = find_template(cls["category"], cls["issue_type"])
    if not template:
        return None

    slots = build_slots(summary, description or "", matches, category=cls["category"] or "")
    filled = fill_slots(template["template_body"], slots)
    customer_response, admin_steps = _parse_filled_body(filled)
    response_type = _response_type_for(cls["question_type"], admin_steps is not None)

    from .drafting import _extract_source_urls

    customer_text = apply_cloud_terminology(customer_response)
    admin_text = apply_cloud_terminology(admin_steps) if admin_steps else None
    draft = {
        "response_type": response_type,
        "customer_response": customer_text,
        "admin_steps": admin_text,
        "missing_info": None,
        "draft_mode": "template",
        "template_name": template["template_name"],
        "sources": _extract_source_urls(customer_text) + _extract_source_urls(admin_text),
    }

    log.info(
        "Template-fill draft for %s using %s (confidence %.2f)",
        ticket_key,
        template["template_name"],
        confidence,
    )
    return draft
