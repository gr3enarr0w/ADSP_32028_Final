#!/usr/bin/env python3
"""Bulk-index Confluence KB pages into kb_articles for BM25/dense retrieval.

Crawls configured Confluence spaces via CQL search, upserts plain-text rows into
kb_articles, then optionally runs vector backfill for kb_article entities.

Usage:
    python -m scripts.index_confluence_kb
    python -m scripts.index_confluence_kb --spaces HUB OMEGA
    python -m scripts.index_confluence_kb --dry-run
    python -m scripts.index_confluence_kb --no-embed
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from db import get_db_conn, init_db
from ingest.oauth2lo import clear_cache as _clear_oauth_cache
from ingest.oauth2lo import get_cloud_auth, get_cloud_base_url

log = logging.getLogger(__name__)

DEFAULT_SPACES = ("HUB", "OMEGA")
PAGE_SIZE = 50
MIN_BODY_CHARS = 100
EXPAND = "body.storage,metadata.labels"

TOPIC_TAXONOMY = (
    "Access",
    "Configuration",
    "Permissions",
    "Integration",
    "Workflow",
    "Data",
    "UI/UX",
    "Notifications",
    "Performance",
)

_UPSERT_SQL = """
    INSERT INTO kb_articles (
        page_id, space_key, title, body_text, url, labels, topics_covered, fetched_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(page_id) DO UPDATE SET
        space_key = excluded.space_key,
        title = excluded.title,
        body_text = excluded.body_text,
        url = excluded.url,
        labels = excluded.labels,
        topics_covered = excluded.topics_covered,
        fetched_at = excluded.fetched_at
"""


def _build_cql(spaces: tuple[str, ...]) -> str:
    """Build CQL for pages in the given space keys.

    Includes both current (published) and draft pages so AI-generated FAQ
    articles (published as drafts pending reviewer approval) are indexed
    and available for retrieval.
    """
    space_list = ",".join(f'"{s}"' for s in spaces)
    return f"space IN ({space_list}) AND type=page AND status IN (current, draft)"


def _strip_storage_html(html: str) -> str:
    """Strip Confluence storage-format HTML tags to plain text."""
    text = re.sub(r"<[^>]+>", " ", html or "")
    return re.sub(r"\s+", " ", text).strip()


def _extract_labels(page: dict) -> str:
    """Comma-join label names from metadata.labels.results."""
    labels = (
        page.get("metadata", {})
        .get("labels", {})
        .get("results", [])
    )
    names = [item.get("name", "") for item in labels if item.get("name")]
    return ",".join(names)


def _match_topics(title: str, body_text: str) -> str:
    """Return comma-joined taxonomy topics found in title + body (case-insensitive)."""
    haystack = f"{title} {body_text}".lower()
    matched = []
    for topic in TOPIC_TAXONOMY:
        needle = topic.lower()
        if needle in haystack:
            matched.append(topic)
    return ",".join(matched)


def _resolve_next_url(base: str, next_link: str) -> str:
    """Turn Confluence _links.next into an absolute request URL."""
    if next_link.startswith("http://") or next_link.startswith("https://"):
        return next_link
    if next_link.startswith("/"):
        # Gateway paths are rooted at api.atlassian.com, not under /ex/confluence/...
        if next_link.startswith("/ex/"):
            return f"https://api.atlassian.com{next_link}"
        return f"{base.rstrip('/')}{next_link}"
    return f"{base.rstrip('/')}/{next_link}"


def _get_search(
    url: str,
    *,
    headers: dict,
    params: dict | None,
) -> tuple[requests.Response, dict]:
    """GET content/search with 401 token refresh retry. Returns (response, headers)."""
    resp = requests.get(url, headers=headers, params=params, timeout=60)
    if resp.status_code == 401:
        log.warning("Confluence search got 401 — clearing token cache and retrying")
        _clear_oauth_cache()
        headers = get_cloud_auth("confluence_search")
        resp = requests.get(url, headers=headers, params=params, timeout=60)
    return resp, headers


def crawl_confluence_pages(spaces: tuple[str, ...]) -> list[dict]:
    """Paginate Confluence CQL search and return raw API page objects."""
    headers = get_cloud_auth("confluence_search")
    base = get_cloud_base_url("confluence_search")
    cql = _build_cql(spaces)

    url: str | None = f"{base}/wiki/rest/api/content/search"
    params: dict | None = {
        "cql": cql,
        "limit": PAGE_SIZE,
        "expand": EXPAND,
    }
    pages: list[dict] = []

    while url:
        resp, headers = _get_search(url, headers=headers, params=params)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Confluence search failed: HTTP {resp.status_code} — {resp.text[:500]}"
            )

        data = resp.json()
        batch = data.get("results", [])
        pages.extend(batch)
        log.info("Fetched %d pages (%d total so far)", len(batch), len(pages))

        next_link = data.get("_links", {}).get("next")
        if not next_link:
            break
        url = _resolve_next_url(base, next_link)
        params = None

    return pages


def parse_page(page: dict, base_url: str, fetched_at: str) -> dict | None:
    """Parse a Confluence page into a kb_articles row, or None if body too short."""
    page_id = str(page.get("id", ""))
    if not page_id:
        return None

    storage = (
        page.get("body", {})
        .get("storage", {})
        .get("value", "")
    )
    body_text = _strip_storage_html(storage)
    if len(body_text) < MIN_BODY_CHARS:
        return None

    space_key = page.get("space", {}).get("key", "") if page.get("space") else ""
    title = page.get("title", "") or ""
    url = f"{base_url}/wiki/spaces/{space_key}/pages/{page_id}"

    return {
        "page_id": page_id,
        "space_key": space_key,
        "title": title,
        "body_text": body_text,
        "url": url,
        "labels": _extract_labels(page),
        "topics_covered": _match_topics(title, body_text),
        "fetched_at": fetched_at,
    }


def upsert_kb_articles(rows: list[dict]) -> int:
    """Upsert parsed rows into kb_articles. Returns number of rows written."""
    if not rows:
        return 0
    params = [
        (
            row["page_id"],
            row["space_key"],
            row["title"],
            row["body_text"],
            row["url"],
            row["labels"],
            row["topics_covered"],
            row["fetched_at"],
        )
        for row in rows
    ]
    with get_db_conn() as conn:
        conn.executemany(_UPSERT_SQL, params)
    return len(rows)


def _count_pending_embeddings() -> int:
    """Return count of kb_articles rows with no embedding yet."""
    with get_db_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM kb_articles WHERE embedding IS NULL"
        ).fetchone()
    return int(row["n"])


def run_vector_backfill() -> tuple[int, int]:
    """Backfill kb_article embeddings. Returns (stored, failures)."""
    from scripts.backfill_vectors import backfill

    pending = _count_pending_embeddings()
    if pending == 0:
        return 0, 0

    try:
        stored = backfill("kb_article")
    except Exception:
        log.exception("Vector backfill failed")
        return 0, pending

    failures = max(0, pending - stored)
    return stored, failures


def _process_crawl(
    spaces: tuple[str, ...],
    *,
    dry_run: bool,
) -> tuple[Counter, int, int]:
    """Crawl and optionally parse/upsert rows. Returns (per_space, upserted, skipped)."""
    fetched_at = datetime.now(timezone.utc).isoformat()
    base_url = get_cloud_base_url("confluence_search")

    raw_pages = crawl_confluence_pages(spaces)
    per_space: Counter = Counter()
    skipped = 0
    rows: list[dict] = []

    for page in raw_pages:
        space_key = page.get("space", {}).get("key", "") if page.get("space") else ""
        if space_key:
            per_space[space_key] += 1
        else:
            per_space["<unknown>"] += 1

        if dry_run:
            continue

        parsed = parse_page(page, base_url, fetched_at)
        if parsed is None:
            skipped += 1
            continue
        rows.append(parsed)

    upserted = 0 if dry_run else upsert_kb_articles(rows)
    return per_space, upserted, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bulk-index Confluence KB pages into kb_articles.",
    )
    parser.add_argument(
        "--spaces",
        nargs="+",
        default=list(DEFAULT_SPACES),
        metavar="SPACE",
        help=f"Confluence space keys (default: {' '.join(DEFAULT_SPACES)})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count pages from Confluence only; no database writes or embeddings",
    )
    parser.add_argument(
        "--no-embed",
        action="store_true",
        help="Skip vector backfill after crawl",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    spaces = tuple(args.spaces)

    log.info("Crawling Confluence spaces: %s (dry_run=%s)", ", ".join(spaces), args.dry_run)

    if not args.dry_run:
        init_db()

    per_space, upserted, skipped = _process_crawl(spaces, dry_run=args.dry_run)

    embed_stored = 0
    embed_failures = 0
    if not args.dry_run and not args.no_embed:
        embed_stored, embed_failures = run_vector_backfill()

    print("\n=== Confluence KB index summary ===")
    print("Pages found per space:")
    for space_key in sorted(per_space):
        print(f"  {space_key}: {per_space[space_key]}")
    print(f"Total pages from API: {sum(per_space.values())}")
    if args.dry_run:
        print("Dry run — no database writes or embeddings.")
    else:
        print(f"Upserted: {upserted}")
        print(f"Skipped (body < {MIN_BODY_CHARS} chars): {skipped}")
        if args.no_embed:
            print("Embeddings: skipped (--no-embed)")
        else:
            print(f"Embeddings stored: {embed_stored}")
            print(f"Embed failures: {embed_failures}")

    return 1 if embed_failures > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
