"""Atlassian Cloud documentation indexer — sitemap-only URL+title index.

Fetches product sitemaps from support.atlassian.com and developer.atlassian.com,
filters out DC/Server URLs, derives titles from URL slugs, and stores
URL+title+product in the atlassian_docs table for fallback lookups.
No per-page fetching — lightweight sitemap-only index.

Supports two Atlassian doc domains:
  - support.atlassian.com — end-user product documentation
  - developer.atlassian.com — REST API docs, developer guides
"""

import logging
import re
from datetime import datetime, timezone
from urllib.parse import urlparse
from xml.etree import ElementTree

import requests

from config import ATLASSIAN_DOC_URLS, ATLASSIAN_DOCS_REFRESH_DAYS
from db import get_db_conn

log = logging.getLogger(__name__)

_SITEMAP_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

_ALLOWED_HOSTS = {"support.atlassian.com", "developer.atlassian.com"}

# Map resource base URLs to their actual sitemap XML files.
# Atlassian uses a root sitemap index; per-product sitemaps live at the root level.
_SITEMAP_MAP = {
    "https://support.atlassian.com/jira-software-cloud/resources/":
        "https://support.atlassian.com/jira-cloud.xml",
    "https://support.atlassian.com/confluence-cloud/resources/":
        "https://support.atlassian.com/confluence-cloud.xml",
    "https://support.atlassian.com/jira-service-management-cloud/resources/":
        "https://support.atlassian.com/jira-service-management-cloud.xml",
    # Developer docs — direct sitemap URLs
    "https://developer.atlassian.com/cloud/jira/platform/":
        "https://developer.atlassian.com/cloud/jira/platform/sitemap.xml",
    "https://developer.atlassian.com/cloud/jira/service-desk/":
        "https://developer.atlassian.com/cloud/jira/service-desk/sitemap.xml",
    "https://developer.atlassian.com/cloud/confluence/":
        "https://developer.atlassian.com/cloud/confluence/sitemap.xml",
    "https://developer.atlassian.com/cloud/admin/":
        "https://developer.atlassian.com/cloud/admin/sitemap.xml",
}


def _product_from_url(url: str) -> str:
    """Extract product name from an Atlassian documentation URL.

    Works with both support.atlassian.com and developer.atlassian.com URLs.
    """
    parsed = urlparse(url)
    host = parsed.hostname or ""

    if host == "developer.atlassian.com":
        # e.g. /cloud/jira/platform/rest/v3/... → "jira-platform-api"
        parts = [s for s in parsed.path.strip("/").split("/") if s]
        if len(parts) >= 3 and parts[0] == "cloud":
            return f"{parts[1]}-{parts[2]}-api"
        elif len(parts) >= 2 and parts[0] == "cloud":
            return f"{parts[1]}-api"
        return "developer-api"

    # support.atlassian.com
    match = re.search(r"support\.atlassian\.com/([^/]+)", url)
    return match.group(1) if match else "unknown"


def _is_cloud_url(url: str) -> bool:
    """Filter URLs to cloud-only content from Atlassian domains.

    Accepts:
    - support.atlassian.com (cloud product docs)
    - developer.atlassian.com (API docs, developer guides)

    Rejects:
    - Any other domain (blocks confluence.atlassian.com DC docs, etc.)
    - URLs with /server/ or /data-center/ path segments
    """
    parsed = urlparse(url)
    host = parsed.hostname or ""
    path = parsed.path.lower()

    if host not in _ALLOWED_HOSTS:
        return False

    # Reject DC/Server path segments
    dc_patterns = ("/server/", "/data-center/", "-server/", "-data-center/")
    if any(p in path for p in dc_patterns):
        return False

    return True


def _title_from_url(url: str) -> str:
    """Derive a readable title from a URL slug.

    e.g. /docs/configure-sla-policies/ → "Configure Sla Policies"
    """
    parsed = urlparse(url)
    # Take the last non-empty path segment
    segments = [s for s in parsed.path.strip("/").split("/") if s]
    slug = segments[-1] if segments else ""
    return slug.replace("-", " ").replace("_", " ").title()


def _resolve_sitemap_url(base_url: str) -> str:
    """Resolve a resource base URL to its sitemap XML URL.

    Uses the known mapping first, then falls back to deriving the
    sitemap name from the URL path.
    """
    normalized = base_url.rstrip("/") + "/"
    if normalized in _SITEMAP_MAP:
        return _SITEMAP_MAP[normalized]

    parsed = urlparse(base_url)
    host = parsed.hostname or ""

    # Developer docs: append sitemap.xml to the path
    if host == "developer.atlassian.com":
        return base_url.rstrip("/") + "/sitemap.xml"

    # Support docs: extract product slug, try as root-level XML
    product = _product_from_url(base_url)
    if product != "unknown":
        return f"https://support.atlassian.com/{product}.xml"

    return base_url.rstrip("/") + "/sitemap.xml"


def _fetch_sitemap_urls(base_url: str) -> list[dict]:
    """Fetch the sitemap XML for a doc site and return cloud article URLs.

    Handles both direct URL sitemaps and sitemap index files (which
    contain references to child sitemaps).

    Returns list of {url, lastmod} dicts.
    """
    sitemap_url = _resolve_sitemap_url(base_url)

    try:
        resp = requests.get(sitemap_url, timeout=30, headers={
            "User-Agent": "JSM-Modeling-Bot/1.0 (internal tooling)"
        })
        resp.raise_for_status()
    except requests.RequestException as e:
        log.warning("Failed to fetch sitemap %s: %s", sitemap_url, e)
        return []

    try:
        root = ElementTree.fromstring(resp.content)
    except ElementTree.ParseError as e:
        log.warning("Failed to parse sitemap XML from %s: %s", sitemap_url, e)
        return []

    # Check if this is a sitemap index (contains <sitemap> elements)
    child_sitemaps = root.findall("sm:sitemap", _SITEMAP_NS)
    if not child_sitemaps:
        child_sitemaps = root.findall("sitemap")

    if child_sitemaps:
        # Sitemap index — recursively fetch each child sitemap
        urls = []
        for sm in child_sitemaps:
            loc = sm.findtext("sm:loc", namespaces=_SITEMAP_NS)
            if not loc:
                loc = sm.findtext("loc")
            if loc and _is_cloud_url(loc):
                urls.extend(_fetch_child_sitemap(loc))
        log.info("Sitemap index %s: %d cloud URLs found across child sitemaps", sitemap_url, len(urls))
        return urls

    # Direct URL sitemap
    urls = []
    for url_elem in root.findall("sm:url", _SITEMAP_NS):
        loc = url_elem.findtext("sm:loc", namespaces=_SITEMAP_NS)
        lastmod = url_elem.findtext("sm:lastmod", namespaces=_SITEMAP_NS)
        if loc and _is_cloud_url(loc):
            urls.append({"url": loc, "lastmod": lastmod})

    # If no sitemap namespace matches, try without namespace
    if not urls:
        for url_elem in root.findall("url"):
            loc = url_elem.findtext("loc")
            lastmod = url_elem.findtext("lastmod")
            if loc and _is_cloud_url(loc):
                urls.append({"url": loc, "lastmod": lastmod})

    log.info("Sitemap %s: %d cloud URLs found", sitemap_url, len(urls))
    return urls


def _fetch_child_sitemap(sitemap_url: str) -> list[dict]:
    """Fetch a single child sitemap and return its URLs."""
    try:
        resp = requests.get(sitemap_url, timeout=30, headers={
            "User-Agent": "JSM-Modeling-Bot/1.0 (internal tooling)"
        })
        resp.raise_for_status()
        root = ElementTree.fromstring(resp.content)
    except Exception as e:
        log.debug("Failed to fetch child sitemap %s: %s", sitemap_url, e)
        return []

    urls = []
    for url_elem in root.findall("sm:url", _SITEMAP_NS):
        loc = url_elem.findtext("sm:loc", namespaces=_SITEMAP_NS)
        lastmod = url_elem.findtext("sm:lastmod", namespaces=_SITEMAP_NS)
        if loc and _is_cloud_url(loc):
            urls.append({"url": loc, "lastmod": lastmod})

    if not urls:
        for url_elem in root.findall("url"):
            loc = url_elem.findtext("loc")
            lastmod = url_elem.findtext("lastmod")
            if loc and _is_cloud_url(loc):
                urls.append({"url": loc, "lastmod": lastmod})

    return urls


def fetch_doc_site(base_url: str) -> int:
    """Index all article URLs from one Atlassian doc site.

    Returns the number of URLs stored/updated.
    """
    product = _product_from_url(base_url)
    sitemap_urls = _fetch_sitemap_urls(base_url)

    if not sitemap_urls:
        log.info("No sitemap URLs found for %s, skipping", base_url)
        return 0

    stored = 0
    now = datetime.now(timezone.utc).isoformat()

    with get_db_conn() as conn:
        for entry in sitemap_urls:
            url = entry["url"]
            lastmod = entry.get("lastmod")
            title = _title_from_url(url)
            article_product = _product_from_url(url)

            conn.execute("""
                INSERT INTO atlassian_docs (url, product, title, last_modified, fetched_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                    title = excluded.title,
                    last_modified = excluded.last_modified,
                    fetched_at = excluded.fetched_at
            """, (url, article_product, title, lastmod, now))
            stored += 1

    log.info("Indexed %s: %d article URLs stored/updated", product, stored)
    return stored


def fetch_all_doc_sites() -> int:
    """Fetch all configured Atlassian doc sites. Returns total URLs indexed."""
    total = 0
    for url in ATLASSIAN_DOC_URLS:
        total += fetch_doc_site(url)
    log.info("Total Atlassian doc URLs indexed: %d", total)
    return total


def read_atlassian_docs() -> list[dict]:
    """Read all stored Atlassian docs from DB."""
    with get_db_conn() as conn:
        rows = conn.execute("""
            SELECT url, product, title
            FROM atlassian_docs
            ORDER BY product, title
        """).fetchall()
    return [dict(r) for r in rows]


def search_atlassian_docs(conn, query_words: list[str], limit: int = 3) -> list[dict]:
    """Search atlassian_docs table by keyword matching on title.

    Args:
        conn: Active DB connection
        query_words: List of search terms (pre-filtered, len > 3)
        limit: Max results to return

    Returns:
        List of matching doc dicts with url, product, title
    """
    if not query_words:
        return []

    conditions = " OR ".join(["title LIKE ?"] * len(query_words))
    params = [f"%{w}%" for w in query_words]

    rows = conn.execute(f"""
        SELECT url, product, title
        FROM atlassian_docs
        WHERE {conditions}
        ORDER BY fetched_at DESC
        LIMIT ?
    """, [*params, limit]).fetchall()

    return [dict(r) for r in rows]
