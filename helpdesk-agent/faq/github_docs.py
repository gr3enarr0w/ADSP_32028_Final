"""GitHub repository documentation indexer.

Indexes markdown files from configured GitHub repos into the atlassian_docs
table for use in FAQ lookups. Fetches file listings via GitHub API, stores
URL + title for each doc file. Same lightweight approach as the sitemap indexer.

Configured via GITHUB_DOC_REPOS in config — list of "owner/repo" strings.
"""

import logging
import re
from datetime import datetime, timezone

import requests

from config import GITHUB_DOC_REPOS
from db import get_db_conn

log = logging.getLogger(__name__)

# File patterns to index (markdown docs, not code)
_DOC_PATTERNS = re.compile(
    r"(^README\.md$|^CONTRIBUTING\.md$|^SECURITY\.md$|"
    r"^docs/.*\.md$|^skills/.*\.md$|"
    r"\.claude-plugin/.*\.md$|\.cursor-plugin/.*\.md$)",
    re.IGNORECASE,
)

# Files to skip
_SKIP_PATTERNS = re.compile(
    r"(LICENSE|CODE_OF_CONDUCT|CODEOWNERS|\.github/)",
    re.IGNORECASE,
)


def _title_from_path(path: str, repo: str) -> str:
    """Derive a readable title from a file path.

    e.g. "README.md" → "Atlassian Mcp Server Readme"
         "docs/setup-guide.md" → "Setup Guide"
    """
    filename = path.rsplit("/", 1)[-1]
    name = filename.replace(".md", "").replace(".txt", "")

    if name.upper() == "README":
        # Use repo name for README
        repo_name = repo.split("/")[-1]
        return repo_name.replace("-", " ").replace("_", " ").title() + " Readme"

    return name.replace("-", " ").replace("_", " ").title()


def _fetch_repo_tree(owner: str, repo: str) -> list[dict]:
    """Fetch the file tree from a GitHub repo and return doc file URLs.

    Uses the GitHub API to get the recursive tree, filters to doc files only.
    Returns list of {url, path, title} dicts.
    """
    api_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/main?recursive=1"

    try:
        resp = requests.get(api_url, timeout=30, headers={
            "User-Agent": "JSM-Modeling-Bot/1.0 (internal tooling)",
            "Accept": "application/vnd.github.v3+json",
        })
        resp.raise_for_status()
    except requests.RequestException as e:
        log.warning("Failed to fetch tree for %s/%s: %s", owner, repo, e)
        return []

    tree = resp.json().get("tree", [])
    docs = []

    for item in tree:
        if item.get("type") != "blob":
            continue
        path = item.get("path", "")

        if _SKIP_PATTERNS.search(path):
            continue
        if not _DOC_PATTERNS.search(path):
            continue

        html_url = f"https://github.com/{owner}/{repo}/blob/main/{path}"
        title = _title_from_path(path, f"{owner}/{repo}")
        docs.append({"url": html_url, "path": path, "title": title})

    log.info("GitHub %s/%s: %d doc files found", owner, repo, len(docs))
    return docs


def fetch_github_repo(repo_spec: str) -> int:
    """Index doc files from one GitHub repo.

    Args:
        repo_spec: "owner/repo" string (e.g. "atlassian/atlassian-mcp-server")

    Returns the number of URLs stored/updated.
    """
    parts = repo_spec.strip().split("/")
    if len(parts) != 2:
        log.warning("Invalid repo spec: %s (expected owner/repo)", repo_spec)
        return 0

    owner, repo = parts
    docs = _fetch_repo_tree(owner, repo)

    if not docs:
        log.info("No doc files found for %s/%s, skipping", owner, repo)
        return 0

    stored = 0
    now = datetime.now(timezone.utc).isoformat()
    product = f"github-{repo}"

    with get_db_conn() as conn:
        for doc in docs:
            conn.execute("""
                INSERT INTO atlassian_docs (url, product, title, last_modified, fetched_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                    title = excluded.title,
                    fetched_at = excluded.fetched_at
            """, (doc["url"], product, doc["title"], None, now))
            stored += 1

    log.info("Indexed GitHub %s/%s: %d doc URLs stored", owner, repo, stored)
    return stored


def fetch_all_github_repos() -> int:
    """Fetch all configured GitHub doc repos. Returns total URLs indexed."""
    if not GITHUB_DOC_REPOS:
        return 0

    total = 0
    for repo in GITHUB_DOC_REPOS:
        total += fetch_github_repo(repo)
    log.info("Total GitHub doc URLs indexed: %d", total)
    return total
