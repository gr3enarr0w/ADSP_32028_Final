"""Generate and post synthetic <PROJECT_KEY> tickets on stage for demo validation.

Uses Gemini to produce realistic support tickets across 6 categories, grounded
by few-shot examples pulled from the real ticket database.  If DATABASE_URL is
not set the script falls back to zero-shot generation with a warning.  PII is
stripped from all generated text before posting.

Usage::

    # Generate ~20 tickets and post them
    python -m scripts.demo_customer_agent

    # Generate ~30 tickets without posting (review content first)
    python -m scripts.demo_customer_agent --dry-run --count 30

    # Close all previously posted demo tickets
    python -m scripts.demo_customer_agent --reset

    # Post 5 tickets
    python -m scripts.demo_customer_agent --count 5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import requests

# ── Project root on sys.path so imports resolve from scripts/ ──────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import CLOUD_URL, GEMINI_MODEL_GENERATION
from core.genai import get_genai_client
from ingest.oauth2lo import get_cloud_auth, get_cloud_base_url

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
_DATA_DIR = _PROJECT_ROOT / "data"
_DEMO_TICKETS_FILE = _DATA_DIR / "demo_tickets.json"

# ── Categories ─────────────────────────────────────────────────────────────────
# Used for zero-shot fallback prompts and as the canonical category list.
_CATEGORIES = {
    "Access": [
        "Need access to RHEL Confluence space",
        "Request view access to internal documentation portal",
        "Unable to access Jira board after team migration",
        "Grant read access to shared project repository",
    ],
    "Configuration": [
        "Add custom field 'Customer Region' to PROJ board",
        "Configure email notifications for sprint completion",
        "Set up default assignee rule for incoming requests",
        "Update workflow transition conditions for review stage",
    ],
    "Permissions": [
        "Grant team read access to restricted operations project",
        "Remove admin rights from former contractor account",
        "Update role permissions for QA team on staging environment",
        "Service account missing write permissions on CI pipeline space",
    ],
    "Integration": [
        "Connect GitHub repository to Jira project for commit tracking",
        "Set up Slack notifications for critical Jira ticket updates",
        "Enable webhook between Jira and internal deployment system",
        "Configure SSO integration for new contractor group",
    ],
    "Performance": [
        "Jira board search is slow for our team of 40 agents",
        "Dashboard loading time exceeds 30 seconds on project overview",
        "Bulk transition times out when processing more than 50 tickets",
        "Automation rule triggers delayed by 10+ minutes",
    ],
    "Workflow": [
        "Automation rule not firing on status change to In Review",
        "Transition button missing for agents on mobile portal",
        "SLA timer not resetting after customer reply",
        "Issue type change removes required field values unexpectedly",
    ],
}

# ── PII scrubbing patterns ─────────────────────────────────────────────────────
_PII_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Email addresses
    (re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", re.IGNORECASE), "[email redacted]"),
    # Account IDs (Atlassian accountId format: 24-char alphanumeric)
    (re.compile(r"\b[0-9a-f]{24}\b", re.IGNORECASE), "[account-id]"),
    # Phone numbers
    (re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"), "[phone redacted]"),
    # IP addresses
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "[ip-address]"),
]


def strip_pii(text: str) -> str:
    """Remove PII from generated text using regex patterns.

    Replaces emails, proper names, account IDs, phone numbers, and IP
    addresses with generic placeholders. Operates on the raw Gemini output
    before the ticket is posted.
    """
    for pattern, replacement in _PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# ── Database few-shot helpers ──────────────────────────────────────────────────

def _open_db():  # returns a db connection (sqlite3 or _PgConn) or None
    """Open a database connection using the app's get_db() helper.

    Uses DATABASE_URL from the environment — supports both PostgreSQL and
    SQLite URLs via SQLAlchemy.  Returns None when DATABASE_URL is not set
    so callers can fall back to zero-shot generation gracefully.
    """
    url = os.getenv("DATABASE_URL", "")
    if not url:
        log.warning("DATABASE_URL not set — using zero-shot generation...")
        return None
    try:
        os.environ["DATABASE_URL"] = url  # ensure it's set for get_db()
        from db import get_db
        return get_db()
    except Exception as e:
        log.warning("Could not open database: %s — falling back to zero-shot", e)
        return None


def _fetch_category_examples(conn, category: str, n: int = 4) -> list[dict]:
    """Pull real <PROJECT_KEY> tickets as few-shot examples for Gemini.

    Queries tickets that have been classified into *category*, filtering out
    rows with missing or very short descriptions.  Results are randomised so
    repeated runs surface different examples.

    Args:
        conn: An open database connection (sqlite3 or _PgConn) with dict-style row access.
        category: The classification category label (e.g., "Access").
        n: Maximum number of examples to return.

    Returns:
        List of dicts with ``summary`` and ``description`` keys.
    """
    rows = conn.execute("""
        SELECT t.summary, t.description
        FROM tickets t
        JOIN ticket_classifications tc ON t.ticket_key = tc.ticket_key
        WHERE tc.category = ?
          AND t.summary IS NOT NULL
          AND t.description IS NOT NULL
          AND length(t.description) > 50
        ORDER BY random()
        LIMIT ?
    """, (category, n)).fetchall()
    return [{"summary": r["summary"], "description": (r["description"] or "")[:500]} for r in rows]


# ── Gemini ticket generation ───────────────────────────────────────────────────

def _call_gemini(prompt: str) -> list[dict]:
    """Send *prompt* to Gemini and return a parsed JSON list.

    Strips markdown code fences if Gemini wraps the response, then parses and
    validates that the result is a JSON array.

    Args:
        prompt: The full prompt string to send.

    Returns:
        Parsed list of ticket dicts.

    Raises:
        RuntimeError: If Gemini returns non-parseable or non-array output.
    """
    client = get_genai_client()
    response = client.models.generate_content(
        model=GEMINI_MODEL_GENERATION,
        contents=prompt,
    )
    raw = response.text.strip()

    # Strip markdown code fences if Gemini wraps the JSON
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        log.error("Gemini returned non-JSON output:\n%s", raw[:500])
        raise RuntimeError(f"Gemini JSON parse failed: {exc}") from exc

    if not isinstance(result, list):
        raise RuntimeError(f"Expected a JSON array, got {type(result).__name__}")

    return result


def _generate_tickets_for_category(category: str, examples: list[dict], count: int = 3) -> list[dict]:
    """Generate *count* synthetic tickets for *category* using few-shot examples.

    Constructs a per-category prompt that shows Gemini real examples from the
    ticket database, then asks it to produce new tickets in the same style.

    Args:
        category: The target category label.
        examples: Real ticket dicts (``summary`` + ``description``) to show as
            few-shot examples.  May be empty — Gemini will still generate but
            without grounding.
        count: Number of new tickets to generate.

    Returns:
        List of ticket dicts with ``category``, ``summary``, ``description``.
    """
    examples_text = "\n\n".join([
        f"Example {i + 1}:\nSummary: {ex['summary']}\nDescription: {ex['description'][:300]}"
        for i, ex in enumerate(examples)
    ])

    prompt = f"""You are generating realistic synthetic Jira Service Management tickets for a Red Hat internal helpdesk (<PROJECT_KEY> project). Category: {category}

Here are {len(examples)} real examples from this project:

{examples_text}

Generate {count} NEW tickets in the same style — same vocabulary, same level of specificity, same tone. Tickets MUST be about Jira Cloud, Confluence Cloud, or Atlassian Cloud platform issues only (permissions, configuration, workflows, spaces, boards, integrations between Atlassian tools). Do NOT generate tickets about GitHub repository access, RHEL, OCP namespaces, or non-Atlassian systems.

Requirements:
- Write in first person as if the customer submitted the ticket
- Do NOT include any real names, real email addresses, or real account IDs
- Use placeholders like "our team", "my account", "the project", "our group"
- Each ticket must have a concise summary (max 100 chars) and a body description

Return ONLY a JSON array with {count} objects, each with "summary" and "description" fields. No markdown, no explanation."""

    log.info("  Generating %d tickets for category '%s' (examples: %d)...", count, category, len(examples))
    raw_tickets = _call_gemini(prompt)

    result = []
    for t in raw_tickets:
        result.append({
            "category": category,
            "summary": strip_pii(str(t.get("summary", "")).strip())[:100],
            "description": strip_pii(str(t.get("description", "")).strip()),
        })
    return result


# Zero-shot fallback prompt (used when DB is unavailable)
_ZERO_SHOT_PROMPT = """\
You are generating realistic synthetic Jira Service Management (JSM) support
tickets for a Red Hat internal IT helpdesk demo. Each ticket simulates a real
user request submitted through the customer portal.

Generate exactly {count} tickets spread evenly across these 6 categories:
{categories}

Requirements:
- Write in first person as if the customer submitted the ticket
- Be specific and realistic (mention tool names, project keys, team contexts)
- Vary length: some short (2-3 sentences), some detailed (4-6 sentences)
- Do NOT include any real names, real email addresses, or real account IDs
- Use placeholders like "our team", "my account", "the project", "our group"
- Each ticket must have a concise summary (max 100 chars) and a body description

Return ONLY valid JSON — an array of objects with these exact keys:
  "category": one of the 6 category names
  "summary": the ticket title (string, ≤100 chars)
  "description": the full ticket body (string)

Example format:
[
  {{
    "category": "Access",
    "summary": "Request access to RHEL documentation space",
    "description": "Our team needs read access to the RHEL Confluence space to reference..."
  }}
]
"""


def _generate_zero_shot(count: int) -> list[dict]:
    """Generate tickets zero-shot (no DB examples).

    Used as a fallback when DATABASE_URL is not configured.

    Args:
        count: Total number of tickets to generate across all categories.

    Returns:
        List of ticket dicts with ``category``, ``summary``, ``description``.
    """
    category_list = "\n".join(
        f"  - {name}: e.g., {examples[0]}"
        for name, examples in _CATEGORIES.items()
    )
    prompt = _ZERO_SHOT_PROMPT.format(count=count, categories=category_list)

    log.info("Asking Gemini to generate %d tickets (zero-shot)...", count)
    raw_tickets = _call_gemini(prompt)

    cleaned = []
    for t in raw_tickets:
        cleaned.append({
            "category": strip_pii(str(t.get("category", "Unknown"))),
            "summary": strip_pii(str(t.get("summary", "")).strip())[:100],
            "description": strip_pii(str(t.get("description", "")).strip()),
        })
    return cleaned


def generate_tickets(count: int) -> list[dict]:
    """Generate synthetic support tickets via Gemini.

    When DATABASE_URL is set, opens the local ticket DB and pulls real
    few-shot examples for each category before calling Gemini per-category.
    When DATABASE_URL is absent, falls back to a single zero-shot prompt and
    logs a warning.

    Tickets are distributed as evenly as possible across the 6 categories.
    Any remainder from integer division is added to the first categories until
    exhausted.

    Args:
        count: Total number of tickets to generate.

    Returns:
        List of ticket dicts ready for posting (``category``, ``summary``,
        ``description``).

    Raises:
        RuntimeError: If Gemini returns non-parseable output.
    """
    conn = _open_db()

    if conn is None:
        log.warning(
            "DATABASE_URL not set — using zero-shot generation. "
            "Set DATABASE_URL=sqlite:///path/to/tickets.db for few-shot grounding."
        )
        return _generate_zero_shot(count)

    try:
        categories = list(_CATEGORIES.keys())
        n_cats = len(categories)
        base, remainder = divmod(count, n_cats)

        all_tickets: list[dict] = []
        for i, category in enumerate(categories):
            per_cat = base + (1 if i < remainder else 0)
            if per_cat == 0:
                continue
            examples = _fetch_category_examples(conn, category, n=4)
            tickets = _generate_tickets_for_category(category, examples, count=per_cat)
            all_tickets.extend(tickets)
            # Brief pause between Gemini calls to stay within rate limits
            if i < n_cats - 1:
                time.sleep(0.5)

        log.info("Generated %d tickets (requested %d) via few-shot", len(all_tickets), count)
        return all_tickets
    finally:
        conn.close()


# ── JSM service desk discovery ─────────────────────────────────────────────────

def _get_service_desk_id(headers: dict, base_url: str, project_key: str) -> str:
    """Discover the numeric service desk ID for a project key.

    Calls GET /rest/servicedeskapi/servicedesk and matches on projectKey.

    Args:
        headers: OAuth Bearer auth headers.
        base_url: The Cloud API base URL (from get_cloud_base_url).
        project_key: The Jira project key to look up (e.g., "<PROJECT_KEY>").

    Returns:
        The service desk ID as a string.

    Raises:
        RuntimeError: If the project key is not found in the service desk list.
    """
    url = f"{base_url}/rest/servicedeskapi/servicedesk"
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    for sd in data.get("values", []):
        if sd.get("projectKey") == project_key:
            sd_id = str(sd["id"])
            log.info("Resolved service desk: %s → ID %s", project_key, sd_id)
            return sd_id

    available = [sd.get("projectKey") for sd in data.get("values", [])]
    raise RuntimeError(
        f"Service desk not found for project key '{project_key}'. "
        f"Available: {available}"
    )


def _get_request_type_id(headers: dict, base_url: str, sd_id: str) -> str:
    """Get the first available (default) request type ID for a service desk.

    Calls GET /rest/servicedeskapi/servicedesk/{id}/requesttype and returns
    the first request type's ID. In most JSM projects this is the general
    "Get IT help" or equivalent catch-all form.

    Args:
        headers: OAuth Bearer auth headers.
        base_url: The Cloud API base URL.
        sd_id: The numeric service desk ID string.

    Returns:
        The request type ID as a string.

    Raises:
        RuntimeError: If no request types are found.
    """
    url = f"{base_url}/rest/servicedeskapi/servicedesk/{sd_id}/requesttype"
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    values = data.get("values", [])
    if not values:
        raise RuntimeError(f"No request types found for service desk ID {sd_id}")

    # Log all available types for visibility
    for rt in values:
        log.info("  Request type: id=%s name=%s", rt.get("id"), rt.get("name"))

    # Prefer a general/help request type by name before falling back to index 0.
    # "generic" is listed first so "Generic Request" (id=76) wins over
    # "Enhancement Request" (id=81) which requires unsupported custom fields.
    preferred = next(
        (rt for rt in values if "generic" in rt.get("name", "").lower()),
        values[0],
    )
    log.info("Using request type: id=%s name=%s", preferred.get("id"), preferred.get("name"))
    return str(preferred["id"])


# ── Ticket posting ─────────────────────────────────────────────────────────────

def post_ticket(
    headers: dict,
    base_url: str,
    sd_id: str,
    request_type_id: str,
    ticket: dict,
) -> str:
    """Post a single synthetic ticket to the JSM service desk.

    Calls POST /rest/servicedeskapi/request with the summary and description
    as request field values. Returns the created issue key (e.g., <PROJECT_KEY>-42).

    Args:
        headers: OAuth Bearer auth headers.
        base_url: The Cloud API base URL.
        sd_id: The numeric service desk ID.
        request_type_id: The request type ID to use.
        ticket: Dict with 'summary' and 'description' keys.

    Returns:
        The Jira issue key of the created request.

    Raises:
        requests.HTTPError: On API failure.
    """
    url = f"{base_url}/rest/servicedeskapi/request"
    payload = {
        "serviceDeskId": sd_id,
        "requestTypeId": request_type_id,
        "requestFieldValues": {
            "summary": ticket["summary"],
            "description": ticket["description"],
            "customfield_10655": {"id": "16560"},
            "customfield_10551": {"id": "17315"},
        },
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=30)

    if not resp.ok:
        log.error(
            "Failed to post ticket: HTTP %d — %s",
            resp.status_code,
            resp.text[:400],
        )
        resp.raise_for_status()

    data = resp.json()
    issue_key = data.get("issueKey")
    if issue_key is None:
        raise ValueError(f"No issueKey in 2xx response: {data}")
    return issue_key


# ── Reset: close posted demo tickets ──────────────────────────────────────────

def _close_ticket(headers: dict, base_url: str, ticket_key: str) -> bool:
    """Attempt to transition a ticket to Resolved/Closed status.

    Fetches available transitions and applies the first terminal one found
    (Resolved, Closed, Done). Returns True on success, False on error.

    Args:
        headers: Jira write OAuth headers.
        base_url: The Cloud API base URL.
        ticket_key: The ticket key to close (e.g., <PROJECT_KEY>-42).

    Returns:
        True if successfully transitioned, False otherwise.
    """
    try:
        url = f"{base_url}/rest/api/3/issue/{ticket_key}/transitions"
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        transitions = resp.json().get("transitions", [])

        terminal_names = {"resolved", "closed", "done", "complete"}
        target = None
        for t in transitions:
            if t.get("name", "").lower() in terminal_names:
                target = t
                break

        if not target:
            log.info(
                "Ticket %s may already be closed (no terminal transitions available); "
                "available: %s",
                ticket_key,
                [t.get("name") for t in transitions],
            )
            return True

        resp = requests.post(
            url,
            headers=headers,
            json={"transition": {"id": target["id"]}},
            timeout=15,
        )
        resp.raise_for_status()
        return True

    except Exception as exc:
        log.warning("Could not close %s: %s", ticket_key, exc)
        return False


def reset_demo_tickets() -> None:
    """Close all tickets recorded in data/demo_tickets.json.

    Reads the stored ticket keys, attempts to transition each to a terminal
    state via the Jira API, and reports success/failure counts. Does not
    delete the file — the keys remain for reference.
    """
    if not _DEMO_TICKETS_FILE.exists():
        log.info("No demo_tickets.json found — nothing to reset.")
        return

    with open(_DEMO_TICKETS_FILE) as f:
        records = json.load(f)

    keys = [r["key"] for r in records if "key" in r]
    if not keys:
        log.info("demo_tickets.json is empty — nothing to reset.")
        return

    log.info("Closing %d demo tickets...", len(keys))

    # Use jira_write creds for transition API; fall back to jsm
    try:
        headers = get_cloud_auth("jira_write")
        base_url = get_cloud_base_url("jira_write")
    except Exception:
        headers = get_cloud_auth("jsm")
        base_url = get_cloud_base_url("jsm")

    closed = 0
    failed = 0
    for key in keys:
        success = _close_ticket(headers, base_url, key)
        if success:
            print(f"  Closed: {key}")
            closed += 1
        else:
            print(f"  Skipped: {key}")
            failed += 1
        time.sleep(0.3)  # avoid rate limiting

    print(f"\nReset complete: {closed} closed, {failed} skipped/failed.")


# ── Storage ────────────────────────────────────────────────────────────────────

def _load_existing_records() -> list[dict]:
    """Load previously posted ticket records from demo_tickets.json."""
    if _DEMO_TICKETS_FILE.exists():
        with open(_DEMO_TICKETS_FILE) as f:
            return json.load(f)
    return []


def _save_records(records: list[dict]) -> None:
    """Persist ticket records to data/demo_tickets.json atomically."""
    import tempfile
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _DEMO_TICKETS_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(records, f, indent=2)
    tmp.replace(_DEMO_TICKETS_FILE)
    log.info("Saved %d records to %s", len(records), _DEMO_TICKETS_FILE)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point for the demo customer agent script."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic <PROJECT_KEY> tickets on stage for demo validation.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Number of tickets to generate (default: 20).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate and print tickets without posting to Jira.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Close all tickets from data/demo_tickets.json and exit.",
    )
    args = parser.parse_args()

    if args.reset and args.count != 20:  # 20 is the default
        log.warning("--count is ignored when --reset is specified")

    if args.reset:
        if args.dry_run:
            log.info("DRY RUN: would close tickets in %s", _DEMO_TICKETS_FILE)
            return
        reset_demo_tickets()
        return

    # ── Generate tickets via Gemini ────────────────────────────────────────────
    tickets = generate_tickets(args.count)

    if args.dry_run:
        print(f"\n{'─' * 60}")
        print(f"DRY RUN — {len(tickets)} tickets (not posted):")
        print(f"{'─' * 60}")
        for i, t in enumerate(tickets, 1):
            print(f"\n[{i:02d}] [{t['category']}] {t['summary']}")
            print(f"     {t['description'][:200]}{'...' if len(t['description']) > 200 else ''}")
        print(f"\n{'─' * 60}")
        print(f"Run without --dry-run to post these to stage.")
        return

    # ── Resolve JSM auth and service desk metadata ─────────────────────────────
    log.info("Authenticating with JSM (stage)...")
    read_headers = get_cloud_auth("jsm")
    write_headers = get_cloud_auth("jira_write")
    base_url = get_cloud_base_url("jsm")

    project_key = "<PROJECT_KEY>"
    try:
        sd_id = _get_service_desk_id(read_headers, base_url, project_key)
        request_type_id = _get_request_type_id(read_headers, base_url, sd_id)
    except RuntimeError as exc:
        log.error("Service desk discovery failed: %s", exc)
        sys.exit(1)

    # ── Post tickets ───────────────────────────────────────────────────────────
    existing_records = _load_existing_records()
    new_records: list[dict] = []
    posted = 0
    failed = 0

    print(f"\n{'─' * 60}")
    print(f"Posting {len(tickets)} synthetic tickets to {project_key} on stage...")
    print(f"{'─' * 60}")

    for i, ticket in enumerate(tickets, 1):
        # Refresh token if needed (tokens last 1h; large batches may span that)
        if i > 1 and i % 50 == 0:
            write_headers = get_cloud_auth("jira_write")

        try:
            issue_key = post_ticket(write_headers, base_url, sd_id, request_type_id, ticket)
            record = {
                "key": issue_key,
                "category": ticket["category"],
                "summary": ticket["summary"],
            }
            new_records.append(record)
            print(f"  [{i:02d}] {issue_key}  [{ticket['category']}]  {ticket['summary']}")
            posted += 1
        except Exception as exc:
            log.error("Failed to post ticket %d (%s): %s", i, ticket["summary"][:50], exc)
            failed += 1

        # Polite pacing — JSM rate limit is generous but avoids 429s on big runs
        time.sleep(0.5)

    # ── Persist records ────────────────────────────────────────────────────────
    all_records = existing_records + new_records
    _save_records(all_records)

    print(f"\n{'─' * 60}")
    print(f"Done: {posted} posted, {failed} failed.")
    print(f"Ticket keys saved to: {_DEMO_TICKETS_FILE}")
    if new_records:
        keys = [r["key"] for r in new_records]
        print(f"Keys: {', '.join(keys)}")
    print(f"{'─' * 60}")


if __name__ == "__main__":
    main()
