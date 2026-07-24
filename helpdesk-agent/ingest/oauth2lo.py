"""Atlassian OAuth 2LO (2-legged OAuth) client credentials flow.

Supports a single OAuth app for all products, with an optional separate
Confluence app if per-product credentials are needed.

Usage:
    from ingest.oauth2lo import get_cloud_auth, get_cloud_base_url

    headers = get_cloud_auth()               # Bearer token (Jira default)
    headers = get_cloud_auth("confluence")   # Uses Confluence app if configured
    jira_base = get_cloud_base_url("jira")
    conf_base = get_cloud_base_url("confluence")
"""

import logging
import threading
import time

import requests

from config import (
    CLOUD_URL,
    JSM_CLIENT_ID, JSM_CLIENT_SECRET,
    JIRA_CLIENT_ID, JIRA_CLIENT_SECRET,
    CONFLUENCE_CLIENT_ID, CONFLUENCE_CLIENT_SECRET,
    mask_id,
)

log = logging.getLogger(__name__)

_TOKEN_URL = "https://auth.atlassian.com/oauth/token"
_RESOURCES_URL = "https://api.atlassian.com/oauth/token/accessible-resources"
_API_BASE = "https://api.atlassian.com"

# Thread-safe per-product token cache
_lock = threading.Lock()
_token_cache: dict[str, tuple[str, float]] = {}  # key -> (token, expires_at)
_cloud_id_cache: dict[str, str] = {}              # key -> cloud_id

_CONFLUENCE_PRODUCTS = {"confluence", "confluence_write", "confluence_search"}
_JIRA_PRODUCTS = {"jira", "jira_write"}
# Everything else (jsm, jsm_feedback, opsgenie, default) → JSM token


def _get_creds(product: str = "jsm") -> tuple[str, str]:
    """Get OAuth credentials for a product.

    3-token product-based design (all credentials on the same service account):
      - confluence, confluence_write, confluence_search → CONFLUENCE token (7 scopes)
      - jira, jira_write                               → JIRA token (26 scopes)
      - jsm, jsm_feedback, opsgenie, (default)         → JSM token (20 scopes)
    """
    if product in _CONFLUENCE_PRODUCTS:
        return CONFLUENCE_CLIENT_ID, CONFLUENCE_CLIENT_SECRET
    if product in _JIRA_PRODUCTS:
        return JIRA_CLIENT_ID, JIRA_CLIENT_SECRET
    return JSM_CLIENT_ID, JSM_CLIENT_SECRET


def _cache_key(product: str) -> str:
    """Return cache key — one per credential pair."""
    if product in _CONFLUENCE_PRODUCTS:
        return "confluence"
    if product in _JIRA_PRODUCTS:
        return "jira"
    return "jsm"


def oauth_configured(product: str = "jira") -> bool:
    """Return True if OAuth 2LO credentials are set."""
    client_id, client_secret = _get_creds(product)
    return bool(client_id and client_secret)


def _fetch_token(product: str) -> tuple[str, int]:
    """Exchange client credentials for an access token."""
    client_id, client_secret = _get_creds(product)
    resp = requests.post(
        _TOKEN_URL,
        json={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
            "audience": "api.atlassian.com",
        },
        headers={"Content-Type": "application/json"},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    token = data["access_token"]
    expires_in = data.get("expires_in", 3600)
    log.info("OAuth 2LO token acquired for %s (expires in %ds)", _cache_key(product), expires_in)
    return token, expires_in


def _get_token(product: str = "jira") -> str:
    """Get a valid access token, refreshing if expired."""
    key = _cache_key(product)
    with _lock:
        cached = _token_cache.get(key)
        if cached:
            token, expires_at = cached
            if time.time() < (expires_at - 60):
                return token

        token, expires_in = _fetch_token(product)
        _token_cache[key] = (token, time.time() + expires_in)
        return token


def _get_cloud_id(product: str) -> str:
    """Discover the Atlassian Cloud site ID for API routing."""
    key = _cache_key(product)
    with _lock:
        cached = _cloud_id_cache.get(key)
        if cached:
            return cached

    token = _get_token(product)
    resp = requests.get(
        _RESOURCES_URL,
        headers={"Authorization": f"Bearer {token}"},
        timeout=15,
    )
    resp.raise_for_status()
    resources = resp.json()

    if not resources:
        raise RuntimeError(f"OAuth 2LO ({key}): no accessible resources found. "
                           "Ensure the app has the correct scopes and site access.")

    cloud_url_host = CLOUD_URL.replace("https://", "").replace("http://", "").rstrip("/") if CLOUD_URL else ""

    for resource in resources:
        if cloud_url_host and resource.get("url", "").rstrip("/").endswith(cloud_url_host):
            with _lock:
                _cloud_id_cache[key] = resource["id"]
            log.info("OAuth 2LO %s cloud ID resolved: %s (%s)",
                     key, mask_id(resource["id"]), resource.get("name", ""))
            return resource["id"]

    resource = resources[0]
    with _lock:
        _cloud_id_cache[key] = resource["id"]
    log.info("OAuth 2LO %s cloud ID (first resource): %s (%s)",
             key, mask_id(resource["id"]), resource.get("name", ""))
    return resource["id"]


def get_cloud_auth(product: str = "jira") -> dict:
    """Get OAuth 2LO Bearer authorization headers.

    Args:
        product: One of the accepted product strings:
                 - "jira"               — Jira Cloud REST API (read)
                 - "jira_write"         — Jira Cloud REST API (write scopes)
                 - "confluence"         — Confluence Cloud REST API (read)
                 - "confluence_write"   — Confluence Cloud REST API (write/publish)
                 - "confluence_search"  — Confluence Cloud search endpoints
                 - "jsm"                — Jira Service Management REST API
                 - "jsm_feedback"       — JSM customer feedback endpoints
                 - "opsgenie"           — OpsGenie REST API
                 Write products (jira_write, confluence_write) use the
                 JIRA_WRITE credential pair if configured; all others use
                 the primary ATLASSIAN_OAUTH credential pair.
    """
    if not oauth_configured(product):
        raise RuntimeError(
            "OAuth 2LO credentials not configured. "
            "Set ATLASSIAN_OAUTH_CLIENT_ID and ATLASSIAN_OAUTH_CLIENT_SECRET."
        )

    token = _get_token(product)
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def get_cloud_base_url(product: str = "jira") -> str:
    """Get the base URL for Cloud API calls.

    Args:
        product: One of the accepted product strings:
                 - "jira"               — routes through the jira gateway
                 - "jira_write"         — routes through the jira gateway (write)
                 - "confluence"         — routes through the confluence gateway
                 - "confluence_write"   — routes through the confluence gateway (write)
                 - "confluence_search"  — routes through the confluence gateway
                 - "jsm"                — routes through the jira gateway
                 - "jsm_feedback"       — routes through the jira gateway
                 - "opsgenie"           — routes through the jira gateway
                 Unmapped product strings are used as-is for the gateway segment.
    """
    cloud_id = _get_cloud_id(product)
    gateway = {"confluence_write": "confluence", "confluence_search": "confluence", "jira_write": "jira", "jsm": "jira", "jsm_feedback": "jira", "opsgenie": "jira"}.get(product, product)
    return f"{_API_BASE}/ex/{gateway}/{cloud_id}"


def clear_cache():
    """Clear all cached tokens and cloud IDs."""
    with _lock:
        _token_cache.clear()
        _cloud_id_cache.clear()
