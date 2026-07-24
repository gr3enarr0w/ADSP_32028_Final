"""Pull all <PROJECT_KEY> tickets via Jira REST API and cache locally.

Uses the Atlassian Cloud REST API with pagination to fetch all tickets.
Saves summary, description (as plain text), comments, status, resolution,
request type, components, and labels.

Usage:
    python -m eval.scripts.pull_jiraconfsd
    python -m eval.scripts.pull_jiraconfsd --limit 100   # test with fewer
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

CLOUD_ID = "2b9e35e3-6bd3-4cec-b838-f4249ee02432"
BASE_URL = f"https://api.atlassian.com/ex/jira/{CLOUD_ID}/rest/api/3"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_FILE = OUTPUT_DIR / "jiraconfsd_all.json"

FIELDS = [
    "summary", "description", "status", "resolution", "comment",
    "components", "labels", "issuetype", "created", "updated",
    "reporter", "assignee", "priority",
]

BATCH_SIZE = 100


EMAIL = os.environ.get("ATLASSIAN_EMAIL", "user@example.com")


def _get_auth_header():
    import base64
    token = os.environ.get("ATLASSIAN_API_TOKEN") or os.environ.get("JIRA_API_TOKEN")
    if not token:
        token_file = Path.home() / ".atlassian" / "token"
        if token_file.exists():
            token = token_file.read_text().strip()
    if not token:
        return None
    if token.startswith("ATATT"):
        creds = base64.b64encode(f"{EMAIL}:{token}".encode()).decode()
        return f"Basic {creds}"
    return f"Bearer {token}"


def _adf_to_text(node) -> str:
    if isinstance(node, str):
        return node
    if not isinstance(node, dict):
        return ""
    text = node.get("text", "")
    for child in node.get("content", []):
        text += _adf_to_text(child)
        if child.get("type") in ("paragraph", "heading", "listItem", "blockquote"):
            text += "\n"
    return text


def _extract_ticket(issue: dict) -> dict:
    f = issue["fields"]
    comments_raw = f.get("comment", {}).get("comments", [])
    comments = []
    for c in comments_raw:
        body = _adf_to_text(c.get("body", {})).strip()
        if len(body) > 10:
            comments.append({
                "author": c.get("author", {}).get("displayName", ""),
                "body": body,
                "created": c.get("created", ""),
                "is_public": c.get("jsdPublic", True),
            })

    return {
        "key": issue["key"],
        "summary": f.get("summary", ""),
        "description": _adf_to_text(f.get("description", {})).strip(),
        "status": f.get("status", {}).get("name", "") if isinstance(f.get("status"), dict) else "",
        "resolution": f.get("resolution", {}).get("name", "") if isinstance(f.get("resolution"), dict) else "",
        "issue_type": f.get("issuetype", {}).get("name", "") if isinstance(f.get("issuetype"), dict) else "",
        "components": [c["name"] for c in f.get("components", []) if isinstance(c, dict)],
        "labels": f.get("labels", []),
        "priority": f.get("priority", {}).get("name", "") if isinstance(f.get("priority"), dict) else "",
        "reporter": f.get("reporter", {}).get("displayName", "") if isinstance(f.get("reporter"), dict) else "",
        "created": f.get("created", ""),
        "updated": f.get("updated", ""),
        "comments": comments,
        "comment_count": len(comments),
    }


def pull_via_api(limit: int | None = None):
    auth = _get_auth_header()
    if not auth:
        print("No API token found. Set JIRA_API_TOKEN env var.")
        sys.exit(1)

    jql = "project = <PROJECT_KEY> ORDER BY created ASC"
    fields_param = ",".join(FIELDS)
    all_tickets = []
    next_page_token = None

    while True:
        params = {
            "jql": jql,
            "fields": fields_param,
            "maxResults": str(BATCH_SIZE),
        }
        if next_page_token:
            params["nextPageToken"] = next_page_token

        query = urllib.parse.urlencode(params)
        url = f"{BASE_URL}/search/jql?{query}"
        req = urllib.request.Request(url)
        req.add_header("Authorization", auth)
        req.add_header("Accept", "application/json")

        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            print(f"HTTP {e.code}: {e.read().decode()[:300]}")
            break

        issues = data.get("issues", [])
        if not issues:
            break

        for issue in issues:
            all_tickets.append(_extract_ticket(issue))

        total = data.get("total", "?")
        print(f"  Fetched {len(all_tickets)}/{total} tickets...", end="\r")

        if limit and len(all_tickets) >= limit:
            all_tickets = all_tickets[:limit]
            break

        next_page_token = data.get("nextPageToken")
        if not next_page_token:
            break

        time.sleep(0.1)

    print(f"\nFetched {len(all_tickets)} tickets total.")
    return all_tickets


def pull_via_mcp_files():
    """Fallback: load from existing jsm-modeling raw JSON files."""
    raw_dir = Path.home() / "Development" / "jsm-modeling"
    tickets = []
    for f in sorted(raw_dir.glob("jiraconfsd_raw_*.json")):
        data = json.loads(f.read_text())
        for issue in data:
            tickets.append(_extract_ticket(issue))
    print(f"Loaded {len(tickets)} tickets from jsm-modeling raw files")
    return tickets


if __name__ == "__main__":
    import urllib.parse

    parser = argparse.ArgumentParser(description="Pull all <PROJECT_KEY> tickets")
    parser.add_argument("--limit", type=int, default=None, help="Max tickets to pull")
    parser.add_argument("--from-files", action="store_true", help="Load from existing raw JSON files instead of API")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.from_files:
        tickets = pull_via_mcp_files()
    else:
        tickets = pull_via_api(limit=args.limit)

    OUTPUT_FILE.write_text(json.dumps(tickets, indent=2))
    print(f"Saved to {OUTPUT_FILE}")

    # Stats
    statuses = {}
    has_desc = sum(1 for t in tickets if t["description"])
    has_comments = sum(1 for t in tickets if t["comment_count"] > 0)
    resolved = sum(1 for t in tickets if t["resolution"])
    for t in tickets:
        statuses[t["status"]] = statuses.get(t["status"], 0) + 1

    print(f"\nStats:")
    print(f"  With description: {has_desc}")
    print(f"  With comments: {has_comments}")
    print(f"  With resolution: {resolved}")
    print(f"  Statuses: {statuses}")
