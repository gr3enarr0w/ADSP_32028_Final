"""Pull every <PROJECT_KEY> ticket via JQL and upsert into the database.

The regular ingest pipeline uses the JSM queue API which only returns tickets
currently in active queues — historical and resolved tickets age out.  This
script uses the Jira REST v3 search/jql endpoint to pull the full project
history, including comments, and upserts everything idempotently.

Usage:
    python scripts/backfill_corpus.py              # pull all tickets
    python scripts/backfill_corpus.py --limit 200  # test with fewer
    python scripts/backfill_corpus.py --since 2025-10-01  # only since a date

Run from the project root.  Reads credentials from .env / environment.
DATABASE_URL controls which database is used (see db.py).
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

# Allow running as a script from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from db import get_db_conn, init_db
from ingest.tickets import api_get, _parse_ticket_cloud, _upsert_ticket, _extract_adf_text
from ingest.oauth2lo import get_cloud_auth, get_cloud_base_url

CLOUD_ID = "2b9e35e3-6bd3-4cec-b838-f4249ee02432"
_JIRA_API_BASE = f"https://api.atlassian.com/ex/jira/{CLOUD_ID}/rest/api/3"


def _personal_token_headers() -> dict | None:
    """Return Basic-auth headers from a personal API token if one is set."""
    import base64
    token = os.environ.get("ATLASSIAN_API_TOKEN") or os.environ.get("JIRA_API_TOKEN")
    email = os.environ.get("ATLASSIAN_EMAIL", "user@example.com")
    if not token:
        return None
    creds = base64.b64encode(f"{email}:{token}".encode()).decode()
    return {
        "Authorization": f"Basic {creds}",
        "Content-Type": "application/json",
    }

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BATCH_SIZE = 100
FIELDS = [
    "summary", "description", "status", "resolution", "comment",
    "components", "labels", "issuetype", "created", "updated",
    "reporter", "assignee", "priority", "customfield_10010",
    "versions", "resolutiondate",
]


def _upsert_comments(conn, ticket_key: str, comments_raw: list) -> int:
    """Upsert comments from REST API comment list format."""
    count = 0
    for c in comments_raw:
        body = c.get("body", "")
        if isinstance(body, dict):
            body = _extract_adf_text(body)
        if not body or not body.strip():
            continue

        created_raw = c.get("created", "")
        author = c.get("author") or {}
        is_public = c.get("jsdPublic", True)

        conn.execute("""
            INSERT OR IGNORE INTO ticket_comments
                (comment_id, ticket_key, author_id, author_name, body, is_public, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            str(c["id"]), ticket_key,
            author.get("accountId", ""), author.get("displayName", ""),
            body.strip(), 1 if is_public else 0, created_raw,
        ))
        count += 1
    return count


def backfill(limit: int | None = None, since: str | None = None):
    init_db()

    # Auth: prefer personal API token (full Jira read scope) over OAuth 2LO
    # OAuth 2LO apps may not have read:jira-work on the /search endpoint.
    # Set ATLASSIAN_API_TOKEN + ATLASSIAN_EMAIL env vars to use personal token.
    headers = _personal_token_headers()
    using_personal_token = headers is not None
    if headers is None:
        log.info("No personal API token found — trying OAuth 2LO")
        headers = get_cloud_auth("jsm")

    search_url = f"{_JIRA_API_BASE}/search/jql"  # cursor-based, current Jira Cloud API

    jql = "project = <PROJECT_KEY> ORDER BY created ASC"
    if since:
        jql = f'project = <PROJECT_KEY> AND created >= "{since}" ORDER BY created ASC'

    log.info("Starting backfill — JQL: %s", jql)
    log.info("Auth: %s", "personal API token" if using_personal_token else "OAuth 2LO")

    tickets_upserted = 0
    comments_upserted = 0
    page = 0
    next_page_token = None
    start_time = time.time()

    import requests as _requests

    while True:
        params = {
            "jql": jql,
            "fields": ",".join(FIELDS),
            "maxResults": BATCH_SIZE,
        }
        if next_page_token:
            params["nextPageToken"] = next_page_token

        resp = _requests.get(search_url, headers=headers, params=params, timeout=60)
        if resp.status_code == 401:
            log.error("401 Unauthorized — check ATLASSIAN_API_TOKEN / OAuth 2LO scopes")
            resp.raise_for_status()
        resp.raise_for_status()
        data = resp.json()

        issues = data.get("issues", [])
        if not issues:
            break

        total_reported = data.get("total", "?")
        page += 1

        with get_db_conn() as conn:
            for issue in issues:
                ticket = _parse_ticket_cloud(issue)
                _upsert_ticket(conn, ticket)
                tickets_upserted += 1

                comments_raw = (
                    issue.get("fields", {})
                    .get("comment", {})
                    .get("comments", [])
                )
                comments_upserted += _upsert_comments(conn, ticket["ticket_key"], comments_raw)

        elapsed = time.time() - start_time
        rate = tickets_upserted / elapsed if elapsed > 0 else 0
        log.info(
            "  Page %d — %d/%s tickets (%.0f/s) | %d comments",
            page, tickets_upserted, total_reported, rate, comments_upserted,
        )

        if limit and tickets_upserted >= limit:
            log.info("Reached --limit %d, stopping", limit)
            break

        next_page_token = data.get("nextPageToken")
        if not next_page_token:
            break

        time.sleep(0.1)

    elapsed = time.time() - start_time
    log.info(
        "Backfill complete — %d tickets, %d comments in %.1fs",
        tickets_upserted, comments_upserted, elapsed,
    )
    return tickets_upserted, comments_upserted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill all <PROJECT_KEY> tickets into the database")
    parser.add_argument("--limit", type=int, default=None, help="Stop after N tickets (for testing)")
    parser.add_argument("--since", type=str, default=None, help="Only tickets created on or after YYYY-MM-DD")
    args = parser.parse_args()

    backfill(limit=args.limit, since=args.since)
