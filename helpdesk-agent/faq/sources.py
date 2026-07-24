"""Aggregate all FAQ source material into a unified context."""

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

import requests

from config import FAQ_SOURCE_DOC_IDS, FAQ_SLIDES_ID, FAQ_CONFLUENCE_SPACES, FAQ_SOURCE_SHEET_IDS
from db import get_db_conn

log = logging.getLogger(__name__)

_GOOGLE_TIMEOUT_SECONDS = 15


def _call_google_source(label: str, fn, *args, **kwargs):
    """Call a Google Workspace source function with a hard wall-clock timeout.

    If ``*.googleapis.com`` is unreachable the Google auth libraries can hang for
    120 s × 3 retries.  This wrapper kills the call after ``_GOOGLE_TIMEOUT_SECONDS``
    and returns ``None`` so the pipeline can continue without Google content.

    Args:
        label: Human-readable name used in log messages (e.g. "Google Docs").
        fn: Callable that fetches the Google source.
        *args, **kwargs: Forwarded to ``fn``.

    Returns:
        The return value of ``fn``, or ``None`` on any failure.
    """
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(fn, *args, **kwargs)
            return future.result(timeout=_GOOGLE_TIMEOUT_SECONDS)
    except FuturesTimeoutError:
        log.warning(
            "  %s timed out after %ds — googleapis.com may be unreachable. Skipping.",
            label,
            _GOOGLE_TIMEOUT_SECONDS,
        )
    except (
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
    ) as exc:
        log.warning("  %s network error: %s. Skipping.", label, exc)
    except Exception as exc:  # noqa: BLE001 — includes google.auth.exceptions.*
        log.warning("  %s failed: %s. Skipping.", label, exc)
    return None


_CONFLUENCE_FETCH_TIMEOUT_SECONDS = 30
_CONFLUENCE_PAGE_LIMIT = 50
_CONFLUENCE_MAX_PAGES = 5  # Maximum pagination pages (up to 250 articles total)


def _strip_html(html: str) -> str:
    """Remove HTML tags from a string, collapsing whitespace.

    Args:
        html: Raw HTML string from Confluence body.storage.value.

    Returns:
        Plain-text representation with excess whitespace removed.
    """
    text = re.sub(r"<[^>]+>", " ", html or "")
    return re.sub(r"\s+", " ", text).strip()


def _fetch_confluence_pages(spaces: list[str]) -> list[dict]:
    """Fetch existing KB pages from configured Confluence spaces via CQL.

    Queries each configured space using the Confluence content search API with
    a broad CQL filter (``space IN (...) AND type=page``). Returns up to
    ``_CONFLUENCE_PAGE_LIMIT`` pages, each as a dict with keys ``id``,
    ``title``, ``text`` (first 500 chars of body, HTML stripped), and ``url``.

    This is distinct from the on-demand responder path in ``faq/lookup.py``
    which runs per-query CQL to answer individual tickets. Here we need the
    full corpus view for gap analysis coverage assessment.

    Args:
        spaces: List of Confluence space keys to include (e.g. ["HUB", "OMEGA"]).

    Returns:
        List of page dicts, or empty list if Confluence is unreachable or
        spaces is empty.
    """
    if not spaces:
        return []

    try:
        from ingest.oauth2lo import get_cloud_auth, get_cloud_base_url, clear_cache as _clear_oauth_cache

        headers = get_cloud_auth("confluence_search")
        base = get_cloud_base_url("confluence_search")

        spaces_cql = ",".join(f'"{s}"' for s in spaces)
        cql = (
            f"space IN ({spaces_cql}) AND type=page "
            f"ORDER BY lastmodified DESC"
        )
        params = {
            "cql": cql,
            "limit": _CONFLUENCE_PAGE_LIMIT,
            "expand": "body.storage",
        }

        resp = requests.get(
            f"{base}/wiki/rest/api/content/search",
            headers=headers,
            params=params,
            timeout=_CONFLUENCE_FETCH_TIMEOUT_SECONDS,
        )

        if resp.status_code == 401:
            log.warning("Confluence CQL fetch got 401 — clearing token cache and retrying")
            _clear_oauth_cache()
            headers = get_cloud_auth("confluence_search")
            resp = requests.get(
                f"{base}/wiki/rest/api/content/search",
                headers=headers,
                params=params,
                timeout=_CONFLUENCE_FETCH_TIMEOUT_SECONDS,
            )

        if resp.status_code in (429,) or resp.status_code >= 500:
            log.warning(
                "  Confluence pages fetch got HTTP %d — waiting 2s and retrying once",
                resp.status_code,
            )
            time.sleep(2)
            resp = requests.get(
                f"{base}/wiki/rest/api/content/search",
                headers=headers,
                params=params,
                timeout=_CONFLUENCE_FETCH_TIMEOUT_SECONDS,
            )
            if resp.status_code != 200:
                log.warning(
                    "  Confluence pages fetch failed after retry: HTTP %d."
                    " Gap analysis will lack KB coverage.",
                    resp.status_code,
                )
                return []

        if resp.status_code != 200:
            log.warning(
                "  Confluence pages fetch failed: HTTP %d. Gap analysis will lack KB coverage.",
                resp.status_code,
            )
            return []

        def _parse_results(data: dict, base_url: str) -> list[dict]:
            """Convert raw Confluence search results to page dicts."""
            out = []
            for page in data.get("results", []):
                page_url = page.get("_links", {}).get("webui", "")
                if page_url and not page_url.startswith("http"):
                    page_url = f"{base_url}{page_url}"
                raw_body = (
                    page.get("body", {})
                    .get("storage", {})
                    .get("value", "")
                )
                text_excerpt = _strip_html(raw_body)[:500]
                out.append({
                    "id": str(page.get("id", "")),
                    "title": page.get("title", ""),
                    "text": text_excerpt,
                    "url": page_url,
                })
            return out

        data = resp.json()
        base_url = data.get("_links", {}).get("base", base)
        pages = _parse_results(data, base_url)

        # Paginate: follow _links.next up to _CONFLUENCE_MAX_PAGES total pages.
        page_num = 1
        next_url = data.get("_links", {}).get("next")
        while next_url and page_num < _CONFLUENCE_MAX_PAGES:
            page_num += 1
            if not next_url.startswith("http"):
                next_url = f"{base}{next_url}"
            log.debug(
                "  Confluence pagination: fetching page %d (%s)",
                page_num,
                next_url,
            )
            next_resp = requests.get(
                next_url,
                headers=headers,
                timeout=_CONFLUENCE_FETCH_TIMEOUT_SECONDS,
            )
            if next_resp.status_code != 200:
                log.warning(
                    "  Confluence pagination page %d failed: HTTP %d. Stopping early.",
                    page_num,
                    next_resp.status_code,
                )
                break
            next_data = next_resp.json()
            pages.extend(_parse_results(next_data, base_url))
            next_url = next_data.get("_links", {}).get("next")

        log.info("  Confluence fetch complete: %d pages across %d API pages", len(pages), page_num)
        return pages

    except requests.exceptions.Timeout:
        log.warning(
            "  Confluence pages fetch timed out after %ds. Gap analysis will lack KB coverage.",
            _CONFLUENCE_FETCH_TIMEOUT_SECONDS,
        )
        return []
    except requests.exceptions.ConnectionError as exc:
        log.warning("  Confluence pages fetch network error: %s. Skipping.", exc)
        return []
    except Exception as exc:  # noqa: BLE001
        log.warning("  Confluence pages fetch failed: %s. Skipping.", exc)
        return []


def gather_all_sources() -> dict:
    """Gather all FAQ source material from configured sources.

    Returns dict with keys:
        google_doc: str — FAQ text from source Google Docs (combined)
        google_slides: str — Presentation content
        confluence_pages: list[dict] — KB articles from configured spaces
        slack_signals: list[dict] — Recent Slack questions
        ticket_resolutions: list[dict] — Resolved ticket answer data
    """
    sources = {
        "google_doc": "",
        "google_slides": "",
        "google_sheets": "",
        "confluence_pages": [],
        "slack_signals": [],
        "ticket_resolutions": [],
        "atlassian_docs": [],
    }

    # Google Docs/Slides/Sheets — disabled. Confluence HUB/OMEGA is now the
    # canonical KB source. External Google Workspace sources are future state (ANTSE-577).
    log.debug("  Google Workspace sources: disabled (Confluence is canonical KB)")

    # Confluence KB pages — pre-fetch for gap analysis corpus coverage.
    # The on-demand CQL path in faq/lookup.py serves the per-ticket responder;
    # here we need the full corpus view so the analyzer can assess what is
    # already published before deciding what is missing.
    if FAQ_CONFLUENCE_SPACES:
        pages = _fetch_confluence_pages(FAQ_CONFLUENCE_SPACES)
        sources["confluence_pages"] = pages
        log.info(
            "  Confluence pages: %d from spaces %s",
            len(pages),
            FAQ_CONFLUENCE_SPACES,
        )
    else:
        log.info("  Confluence pages: no spaces configured")

    # DB-sourced content
    with get_db_conn() as conn:

        # Slack signals — future state (ANTSE-577). Not yet active.
        sources["slack_signals"] = []
        log.debug("  Slack signals: disabled (future state)")

        rows = conn.execute("""
            SELECT t.ticket_key, t.summary, t.description, t.resolution,
                   c.issue_type, c.resolution_summary, c.category
            FROM tickets t
            JOIN ticket_classifications c ON t.ticket_key = c.ticket_key
            WHERE c.has_resolution = 1
            ORDER BY c.confidence DESC
            LIMIT 200
        """).fetchall()
        sources["ticket_resolutions"] = [dict(r) for r in rows]
        log.info("  Ticket resolutions: %d", len(rows))

        # Atlassian docs (vendor reference, not internal coverage)
        # atlassian_docs (support.atlassian.com, developer.atlassian.com) — future state (ANTSE-577).
        sources["atlassian_docs"] = []
        log.debug("  Atlassian external docs: disabled (future state)")

    return sources


def get_source_status() -> list[dict]:
    """Return status of all configured FAQ sources."""
    with get_db_conn() as conn:
        rows = conn.execute("""
            SELECT source_type, source_id, title, content_hash, last_fetched
            FROM faq_sources ORDER BY source_type, title
        """).fetchall()

    status = [dict(r) for r in rows]

    # Add unfetched sources
    configured = {(r["source_type"], r["source_id"]) for r in status}
    for doc_id in FAQ_SOURCE_DOC_IDS:
        if ("google_doc", doc_id) not in configured:
            status.append({"source_type": "google_doc", "source_id": doc_id,
                            "title": "(not yet fetched)", "last_fetched": None})
    if FAQ_SLIDES_ID and ("google_slides", FAQ_SLIDES_ID) not in configured:
        status.append({"source_type": "google_slides", "source_id": FAQ_SLIDES_ID,
                        "title": "(not yet fetched)", "last_fetched": None})
    for sheet_id in FAQ_SOURCE_SHEET_IDS:
        if ("google_sheet", sheet_id) not in configured:
            status.append({"source_type": "google_sheet", "source_id": sheet_id,
                            "title": "(not yet fetched)", "last_fetched": None})

    return status
