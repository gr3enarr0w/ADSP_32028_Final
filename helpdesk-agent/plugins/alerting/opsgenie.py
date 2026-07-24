"""OpsGenie alert client — shared by volume anomaly (ANTSE-315) and cluster (ANTSE-319) alerts.

Auth strategy (in priority order):
  1. Atlassian OAuth 2LO via ``ATLASSIAN_OAUTH2_CLIENT_ID`` / ``ATLASSIAN_OAUTH2_CLIENT_SECRET``
     → posts to the Atlassian Operations API (``jsm/ops/invoke/{cloud_id}/v1/alerts``)
  2. Legacy OpsGenie API key via ``OPSGENIE_API_KEY``
     → posts directly to ``https://api.opsgenie.com/v2/alerts``

The module-level :func:`post_alert` wrapper tries OAuth first and silently falls back
to the API-key approach so existing callers need no changes.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import requests

import base64

from config import OPSGENIE_API_KEY, OPSGENIE_TOKEN, OPSGENIE_EMAIL

log = logging.getLogger(__name__)

# Legacy direct OpsGenie endpoint (API-key path).
_OPSGENIE_ALERTS_URL = "https://api.opsgenie.com/v2/alerts"

# Atlassian Operations API base — cloud_id is injected at runtime.
# Correct path is jsm/ops/api (not jsm/ops/invoke).
_ATLASSIAN_OPS_ALERTS_URL = "https://api.atlassian.com/jsm/ops/api/{cloud_id}/v1/alerts"

# Regex to extract the UUID cloud_id from get_cloud_base_url() output:
# e.g. "https://api.atlassian.com/ex/jira/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
_CLOUD_ID_RE = re.compile(r"/ex/jira/([^/]+)$")


def _extract_cloud_id_from_base_url(base_url: str) -> str:
    """Extract the raw cloud UUID from a get_cloud_base_url() return value.

    Args:
        base_url: URL of the form ``https://api.atlassian.com/ex/jira/<cloud_id>``.

    Returns:
        The UUID portion of the URL.

    Raises:
        ValueError: If the URL does not match the expected pattern.
    """
    match = _CLOUD_ID_RE.search(base_url)
    if not match:
        raise ValueError(
            f"Cannot extract cloud_id from base URL: {base_url!r}. "
            "Expected format: https://api.atlassian.com/ex/jira/<uuid>"
        )
    return match.group(1)


class OpsGenieBasicAuthClient:
    """JSM Operations API client using Atlassian service account Basic Auth.

    Uses OPSGENIE_EMAIL + OPSGENIE_TOKEN (ATSTT... format) to authenticate
    against the JSM Operations API. This is the preferred auth path when a
    dedicated service account API token is available.
    """

    def __init__(self, *, cloud_id: str | None = None, source: str = "ai-helpdesk-alerting", timeout: int = 30):
        self.cloud_id = cloud_id
        self.source = source
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(OPSGENIE_TOKEN and OPSGENIE_EMAIL)

    def _get_headers(self) -> dict:
        credentials = base64.b64encode(f"{OPSGENIE_EMAIL}:{OPSGENIE_TOKEN}".encode()).decode()
        return {
            "Authorization": f"Basic {credentials}",
            "Content-Type": "application/json",
        }

    def _get_cloud_id(self) -> str:
        if self.cloud_id:
            return self.cloud_id
        from ingest.oauth2lo import get_cloud_base_url
        base_url = get_cloud_base_url("opsgenie")
        match = _CLOUD_ID_RE.search(base_url)
        if not match:
            raise ValueError(f"Cannot extract cloud_id from: {base_url!r}")
        return match.group(1)

    def create_alert(
        self,
        *,
        message: str,
        description: str,
        alias: str | None = None,
        priority: str = "P3",
        tags: list[str] | None = None,
        details: dict[str, str] | None = None,
    ) -> dict[str, Any] | None:
        if not self.enabled:
            log.warning("[opsgenie/basic] OPSGENIE_TOKEN or OPSGENIE_EMAIL not set — skipping")
            return None

        payload: dict[str, Any] = {
            "message": message,
            "description": description,
            "source": self.source,
            "priority": priority,
        }
        if alias:
            payload["alias"] = alias
        if tags:
            payload["tags"] = tags
        if details:
            payload["details"] = details

        try:
            cloud_id = self._get_cloud_id()
            url = _ATLASSIAN_OPS_ALERTS_URL.format(cloud_id=cloud_id)
            resp = requests.post(url, headers=self._get_headers(), json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data: dict[str, Any] = {}
            if resp.content:
                try:
                    data = resp.json()
                except ValueError:
                    pass
            request_id = data.get("requestId") or data.get("data", {}).get("requestId") or "?"
            log.info("[opsgenie/basic] alert created (requestId=%s, alias=%s, status=%d)", request_id, alias, resp.status_code)
            return data or {"status": resp.status_code}
        except requests.RequestException as exc:
            body = exc.response.text[:500] if exc.response is not None else ""
            log.error("[opsgenie/basic] alert failed: %s %s", exc, body)
            return None


class OpsGenieAtlassianClient:
    """OpsGenie alert client backed by Atlassian OAuth 2LO.

    Uses the ``write:ops-alert:jira-service-management`` scope via
    ``ATLASSIAN_OAUTH2_CLIENT_ID`` / ``ATLASSIAN_OAUTH2_CLIENT_SECRET`` and
    posts to the Atlassian Operations API rather than the legacy OpsGenie endpoint.
    """

    def __init__(self, *, source: str = "ai-helpdesk-alerting", timeout: int = 30):
        self.source = source
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        """True when OAuth 2LO credentials for OpsGenie are configured."""
        return False  # ops-* OAuth scopes don't work for OpsGenie; use OPSGENIE_API_KEY

    def _get_alerts_url(self) -> str:
        """Resolve the Atlassian Operations API alerts URL for the configured cloud.

        Lazily imports oauth2lo to avoid circular-import issues at module load time.
        The cloud_id is extracted from the base URL returned by ``get_cloud_base_url``.
        """
        from ingest.oauth2lo import get_cloud_base_url  # local import — avoids circular deps

        base_url = get_cloud_base_url("opsgenie")
        cloud_id = _extract_cloud_id_from_base_url(base_url)
        return _ATLASSIAN_OPS_ALERTS_URL.format(cloud_id=cloud_id)

    def create_alert(
        self,
        *,
        message: str,
        description: str,
        alias: str | None = None,
        priority: str = "P3",
        tags: list[str] | None = None,
        details: dict[str, str] | None = None,
    ) -> dict[str, Any] | None:
        """Create an alert via the Atlassian Operations API.

        Args:
            message: Short summary shown in the alert title.
            description: Full details body.
            alias: Idempotency key; duplicate alias → de-duplicated alert.
            priority: OpsGenie priority (P1–P5).
            tags: Optional tag list.
            details: Optional key/value pairs for the alert details pane.

        Returns:
            Parsed JSON response on success (HTTP 202), or ``None`` on error.
        """
        if not self.enabled:
            log.warning("[opsgenie/oauth] OpsGenie OAuth disabled (ops-* scopes not supported) — skipping alert")
            return None

        from ingest.oauth2lo import get_cloud_auth  # local import — avoids circular deps

        payload: dict[str, Any] = {
            "message": message,
            "description": description,
            "source": self.source,
            "priority": priority,
        }
        if alias:
            payload["alias"] = alias
        if tags:
            payload["tags"] = tags
        if details:
            payload["details"] = details

        try:
            headers = get_cloud_auth("opsgenie")
            url = self._get_alerts_url()
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            # Atlassian Operations API returns 202 Accepted; body may be empty or minimal JSON.
            data: dict[str, Any] = {}
            if resp.content:
                try:
                    data = resp.json()
                except ValueError:
                    pass
            request_id = data.get("requestId") or data.get("data", {}).get("requestId") or "?"
            log.info(
                "[opsgenie/oauth] alert created (requestId=%s, alias=%s, status=%d)",
                request_id,
                alias,
                resp.status_code,
            )
            return data or {"status": resp.status_code}
        except requests.RequestException as exc:
            body = ""
            if exc.response is not None:
                body = exc.response.text[:500]
            log.error("[opsgenie/oauth] alert failed: %s %s", exc, body)
            return None


class OpsGenieClient:
    """Thin wrapper around the legacy OpsGenie Create Alert REST API (API-key auth).

    Kept for backward compatibility and as a fallback when OAuth 2LO credentials
    are not available.
    """

    def __init__(self, api_key: str, *, source: str = "ai-helpdesk-alerting", timeout: int = 30):
        self.api_key = api_key.strip()
        self.source = source
        self.timeout = timeout

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "OpsGenieClient":
        """Construct from a pipeline config dict."""
        return cls(
            OPSGENIE_API_KEY,
            source=str(cfg.get("opsgenie_source", "ai-helpdesk-alerting")),
        )

    @property
    def enabled(self) -> bool:
        """True when an API key is present."""
        return bool(self.api_key)

    def create_alert(
        self,
        *,
        message: str,
        description: str,
        alias: str | None = None,
        priority: str = "P3",
        tags: list[str] | None = None,
        details: dict[str, str] | None = None,
    ) -> dict[str, Any] | None:
        """Create an OpsGenie alert using a direct API key.

        Args:
            message: Short summary shown in the alert title.
            description: Full details body.
            alias: Idempotency key; duplicate alias → de-duplicated alert.
            priority: OpsGenie priority (P1–P5).
            tags: Optional tag list.
            details: Optional key/value pairs for the alert details pane.

        Returns:
            Parsed JSON response on success, or ``None`` on error.
        """
        if not self.enabled:
            log.warning("[opsgenie/apikey] API key not configured — skipping alert")
            return None

        payload: dict[str, Any] = {
            "message": message,
            "description": description,
            "source": self.source,
            "priority": priority,
        }
        if alias:
            payload["alias"] = alias
        if tags:
            payload["tags"] = tags
        if details:
            payload["details"] = details

        headers = {
            "Authorization": f"GenieKey {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            resp = requests.post(
                _OPSGENIE_ALERTS_URL,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            request_id = (
                data.get("requestId") or data.get("data", {}).get("requestId") or "?"
            )
            log.info(
                "[opsgenie/apikey] alert created (requestId=%s, alias=%s)",
                request_id,
                alias,
            )
            return data
        except requests.RequestException as exc:
            body = ""
            if exc.response is not None:
                body = exc.response.text[:500]
            log.error("[opsgenie/apikey] alert failed: %s %s", exc, body)
            return None


def post_alert(
    *,
    message: str,
    alias: str | None = None,
    description: str | None = None,
    priority: str = "P3",
    tags: list[str] | None = None,
) -> bool:
    """Module-level convenience wrapper used by anomaly_alert.py and cluster alerting.

    Tries Atlassian OAuth 2LO first (``ATLASSIAN_OAUTH2_CLIENT_ID``); falls back to
    the legacy OpsGenie API key (``OPSGENIE_API_KEY``) for backward compatibility.

    Args:
        message: Short alert summary.
        alias: Idempotency key.
        description: Full body text.
        priority: OpsGenie priority (P1–P5).
        tags: Optional tag list.

    Returns:
        ``True`` if the alert was accepted, ``False`` otherwise.
    """
    desc = description or ""

    # --- Primary path: Basic Auth (service account API token) ---
    basic_client = OpsGenieBasicAuthClient()
    if basic_client.enabled:
        result = basic_client.create_alert(
            message=message, description=desc, alias=alias, priority=priority, tags=tags,
        )
        if result is not None:
            return True
        log.warning("[opsgenie] Basic Auth alert failed; falling back to OAuth path")

    # --- Secondary path: Atlassian OAuth 2LO ---
    oauth_client = OpsGenieAtlassianClient()
    if oauth_client.enabled:
        result = oauth_client.create_alert(
            message=message, description=desc, alias=alias, priority=priority, tags=tags,
        )
        if result is not None:
            return True
        log.warning("[opsgenie] OAuth alert failed; falling back to API-key path")

    # --- Fallback: legacy OpsGenie API key ---
    api_key_client = OpsGenieClient(OPSGENIE_API_KEY)
    if api_key_client.enabled:
        result = api_key_client.create_alert(
            message=message, description=desc, alias=alias, priority=priority, tags=tags,
        )
        return result is not None

    log.error("[opsgenie] No alert sent — configure OPSGENIE_TOKEN+OPSGENIE_EMAIL, ATLASSIAN_OAUTH2_CLIENT_ID, or OPSGENIE_API_KEY")
    return False
