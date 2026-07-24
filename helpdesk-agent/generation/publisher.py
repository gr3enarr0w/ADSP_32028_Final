"""Confluence article publishing."""

import logging
import requests
from datetime import datetime, timezone

from config import CLOUD_URL, CONFLUENCE_KB_SPACE, CONFLUENCE_PARENT_PAGE_ID
from db import get_db_conn
from ingest.oauth2lo import get_cloud_auth, get_cloud_base_url, clear_cache as _clear_oauth_cache

log = logging.getLogger(__name__)

_space_id_cache: dict[str, str] = {}


def _resolve_space_id(space_key: str) -> str | None:
    """Resolve a Confluence space key to its numeric ID via CQL search."""
    if space_key in _space_id_cache:
        return _space_id_cache[space_key]

    try:
        base = get_cloud_base_url("confluence")
        headers = get_cloud_auth("confluence")
        params = {"cql": f"space={space_key} AND type=page", "limit": 1, "expand": "space"}
        resp = requests.get(
            f"{base}/wiki/rest/api/content/search", headers=headers,
            params=params, timeout=15,
        )
        if resp.status_code == 401:
            log.warning("Confluence space ID lookup got 401 — clearing token cache and retrying")
            _clear_oauth_cache()
            headers = get_cloud_auth("confluence")
            resp = requests.get(
                f"{base}/wiki/rest/api/content/search", headers=headers,
                params=params, timeout=15,
            )
        if resp.status_code == 200:
            results = resp.json().get("results", [])
            if results:
                space_id = str(results[0].get("space", {}).get("id", ""))
                if space_id:
                    _space_id_cache[space_key] = space_id
                    return space_id
    except Exception as e:
        log.warning("Failed to resolve space ID for '%s': %s", space_key, e)
    return None


# Group that gets read/edit access to AI-generated FAQ articles.
# Red Hat One group — members are the reviewing team.
_REVIEWER_GROUP = "Red Hat One"


def _apply_page_restrictions(confluence_base: str, headers: dict, page_id: str) -> None:
    """Restrict page view/edit to the reviewer group only.

    Raises RuntimeError if restrictions cannot be applied — callers must not
    mark the article as published if this fails, to prevent public exposure.
    """
    restriction_url = f"{confluence_base}/wiki/rest/api/content/{page_id}/restriction"
    payload = {
        "results": [
            {
                "operation": "read",
                "restrictions": {
                    "group": {"results": [{"type": "group", "name": _REVIEWER_GROUP}]},
                    "user": {"results": []},
                },
            },
            {
                "operation": "update",
                "restrictions": {
                    "group": {"results": [{"type": "group", "name": _REVIEWER_GROUP}]},
                    "user": {"results": []},
                },
            },
        ]
    }
    resp = requests.put(restriction_url, headers=headers, json=payload, timeout=15)
    if resp.status_code in (200, 204):
        log.debug("Restrictions applied to page %s (%s only)", page_id, _REVIEWER_GROUP)
    else:
        raise RuntimeError(
            f"Could not apply restrictions to page {page_id}: "
            f"HTTP {resp.status_code} — {resp.text[:200]}"
        )


def publish_article(article_id, space_key=None):
    """Publish a generated article to Confluence.

    Pages are published as status=current (visible in page tree) but restricted
    to the reviewer group (_REVIEWER_GROUP) via page restrictions. This keeps
    them out of public view while making them indexable by the KB crawler and
    visible to the reviewing team for approval.
    """
    with get_db_conn() as conn:
        article = conn.execute(
            "SELECT * FROM generated_articles WHERE id = ?", (article_id,)
        ).fetchone()

        if not article:
            log.error("Article %d not found", article_id)
            return False

        space = space_key or CONFLUENCE_KB_SPACE

        # Resolve space key → numeric ID (v2 requires ID, not key)
        space_id = _resolve_space_id(space)
        if not space_id:
            log.error("Could not resolve Confluence space '%s'", space)
            return False

        headers = get_cloud_auth("confluence_write")
        confluence_base = get_cloud_base_url("confluence_write")
        url = f"{confluence_base}/wiki/api/v2/pages"
        payload = {
            "spaceId": str(space_id),
            "title": article["title"],
            "body": {"storage": {"representation": "storage", "value": article["body_html"]}},
            "status": "current",
        }
        if CONFLUENCE_PARENT_PAGE_ID:
            payload["parentId"] = CONFLUENCE_PARENT_PAGE_ID

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            if resp.status_code == 401:
                log.warning("Confluence publish got 401 — clearing token cache and retrying")
                _clear_oauth_cache()
                headers = get_cloud_auth("confluence_write")
                resp = requests.post(url, headers=headers, json=payload, timeout=30)
            if resp.status_code == 409:
                # Title already exists — append article_id to deduplicate
                payload["title"] = f"{article['title']} ({article_id})"
                resp = requests.post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            result = resp.json()

            if "id" in result:
                page_id = str(result["id"])
                page_url = result.get("_links", {}).get("webui", "")
                if page_url and not page_url.startswith("http"):
                    base_url = result.get("_links", {}).get("base") or CLOUD_URL
                    page_url = f"{base_url}/wiki{page_url}" if not page_url.startswith("/wiki") else f"{base_url}{page_url}"

                # Apply restrictions BEFORE marking as published.
                # If this raises, the article stays unpublished for retry.
                _apply_page_restrictions(confluence_base, headers, page_id)

                conn.execute("""
                    UPDATE generated_articles
                    SET confluence_page_id = ?, confluence_url = ?,
                        status = 'published', published_at = ?
                    WHERE id = ?
                """, (page_id, page_url, datetime.now(timezone.utc).isoformat(), article_id))
                conn.commit()
                log.info("Published: %s -> %s (restricted to %s)", article["title"], page_url, _REVIEWER_GROUP)
                return True
        except Exception as e:
            log.error("Failed to publish article %d: %s", article_id, e)

    return False
