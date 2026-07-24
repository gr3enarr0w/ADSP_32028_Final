"""Ticket and comment fetching from JSM Cloud via servicedeskapi (OAuth 2LO compatible)."""

import json
import time
import logging
import requests
from datetime import datetime, timezone

from db import get_db_conn
from ingest.oauth2lo import get_cloud_auth, get_cloud_base_url, clear_cache as _clear_oauth_cache
from config import CLOUD_CUTOVER_DATE

log = logging.getLogger(__name__)

_SD_ID_CACHE: dict[str, str] = {}


def _cloud_headers():
    return get_cloud_auth("jsm")


def api_get(url, headers, params=None, max_retries=5, _product="jsm"):
    """GET with retry, rate-limit (429), and backoff handling."""
    _refreshed_token = False
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=30)
        except (requests.ConnectionError, requests.Timeout) as e:
            wait = min(2 ** attempt * 2, 60)
            log.warning("Connection error (attempt %d/%d): %s — retrying in %ds",
                        attempt + 1, max_retries, e, wait)
            time.sleep(wait)
            continue

        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 401 and not _refreshed_token:
            log.warning("Got 401 on %s — clearing token cache and retrying", url)
            _clear_oauth_cache()
            headers = get_cloud_auth(_product)
            _refreshed_token = True
            continue
        if resp.status_code == 429:
            retry_after = max(int(resp.headers.get("Retry-After", 2 ** attempt * 2)), 2)
            log.warning("Rate limited (429) — waiting %ds", retry_after)
            time.sleep(retry_after)
            continue
        if resp.status_code >= 500:
            wait = min(2 ** attempt * 2, 60)
            log.warning("Server error %d (attempt %d/%d) — retrying in %ds",
                        resp.status_code, attempt + 1, max_retries, wait)
            time.sleep(wait)
            continue

        log.error("API error %d: %s — %s", resp.status_code, url, resp.text[:500])
        resp.raise_for_status()

    raise RuntimeError(f"Failed after {max_retries} retries: {url}")


def _extract_adf_text(adf):
    """Recursively extract plain text from Atlassian Document Format."""
    if not adf or not isinstance(adf, dict):
        return ""
    parts = []
    for node in adf.get("content", []):
        if node.get("type") == "text":
            parts.append(node.get("text", ""))
        elif "content" in node:
            parts.append(_extract_adf_text(node))
    return "\n".join(parts)


def _is_uat_only(created_at: str | None) -> int:
    """Return 1 if created before the Cloud cutover date, 0 otherwise."""
    if not created_at:
        return 0
    try:
        created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        cutover = datetime.fromisoformat(CLOUD_CUTOVER_DATE).replace(tzinfo=timezone.utc)
        return 1 if created < cutover else 0
    except (ValueError, TypeError):
        return 0


def _normalize_status(raw_status: str) -> str:
    """Normalise Jira status names to a consistent set.

    Jira exposes two overlapping concepts: the *status name* (e.g. "Done",
    "Closed", "Resolved") and the *status category* (e.g. "Done", "In Progress").
    Some JSM cloud projects use "Done" as the literal status name instead of
    "Closed" or "Resolved".  We canonicalise those here so every consumer in
    the codebase can check for "Closed"/"Resolved" without also special-casing
    "Done".

    Additionally, the JSM queue endpoint has been observed to return tickets
    with a status whose ``name`` field is the literal string ``''`` (two single
    quotes).  These are Closed tickets whose status name was not correctly
    populated; they are mapped to "Closed".

    Mapping:
        "Done"  → "Closed"
        "''"    → "Closed"   (two-single-quote corruption from JSM queue API)
        anything else → unchanged
    """
    if raw_status in ("Done", "''"):
        return "Closed"
    return raw_status


def _parse_ticket_cloud(issue):
    """Parse a Jira Core /rest/api/3/search/jql issue object into a ticket dict."""
    fields = issue.get("fields", {})
    versions = fields.get("versions", [])
    components = fields.get("components", [])
    reporter = fields.get("reporter") or {}
    resolution = fields.get("resolution")
    created_at = fields.get("created")

    raw_status = (fields.get("status") or {}).get("name", "")

    return {
        "ticket_key": issue["key"],
        "summary": fields.get("summary", ""),
        "description": _extract_adf_text(fields.get("description")) if isinstance(fields.get("description"), dict) else (fields.get("description") or ""),
        "status": _normalize_status(raw_status),
        "resolution": resolution["name"] if resolution else None,
        "request_type": (fields.get("customfield_10010") or {}).get("requestType", {}).get("name")
                        if fields.get("customfield_10010") else None,
        "affect_version": versions[0]["name"] if versions else None,
        "components": json.dumps([c["name"] for c in components]) if components else "[]",
        "reporter_id": reporter.get("accountId", ""),
        "reporter_email": reporter.get("emailAddress", ""),
        "assignee_id": (fields.get("assignee") or {}).get("accountId", ""),
        "created_at": created_at,
        "resolved_at": fields.get("resolutiondate"),
        "updated_at": fields.get("updated"),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source": "cloud",
        "is_cloud": 1,
        "is_uat_only": _is_uat_only(created_at),
    }


def _parse_ticket_jsm_request(req):
    """Parse a JSM servicedeskapi /request object into a ticket dict.

    The servicedeskapi /request endpoint returns a different shape from the
    Jira Core /search/jql endpoint:

    - ``issueKey``          — ticket key (e.g. <PROJECT_KEY>-123)
    - ``requestFieldValues`` — list of {fieldId, label, value} objects
    - ``currentStatus``     — {status, statusDate}
    - ``reporter``          — {accountId, displayName}
    - ``requestType``       — {id, name, description}
    - ``serviceDeskId``     — numeric service desk id
    - ``createdDate``       — {iso8601, jira, friendly, epochMillis}

    Fields not available via this endpoint (no issuelinks, no versions, no
    resolution, no assignee) are set to None / empty defaults.  The description
    backfill via ``fetch_descriptions_cloud()`` handles the description gap.
    """
    key = req.get("issueKey", "")

    # Extract summary and description from requestFieldValues list
    summary = ""
    description = ""
    for field in req.get("requestFieldValues", []):
        fid = field.get("fieldId", "")
        val = field.get("value", "")
        if fid == "summary":
            summary = val if isinstance(val, str) else ""
        elif fid == "description":
            if isinstance(val, str):
                description = val
            else:
                log.debug("Ticket %s: description field %r is non-string type %s — skipped", req.get("issueKey", "?"), fid, type(val).__name__)

    # Status lives under currentStatus.status (a plain string in Cloud)
    raw_status = (req.get("currentStatus") or {}).get("status", "")

    # Created date — prefer iso8601 sub-field
    created_obj = req.get("createdDate") or {}
    created_at = created_obj.get("iso8601") if isinstance(created_obj, dict) else str(created_obj or "")

    # Reporter
    reporter = req.get("reporter") or {}

    # Request type name
    request_type = (req.get("requestType") or {}).get("name")

    return {
        "ticket_key": key,
        "summary": summary,
        "description": description,
        "status": _normalize_status(raw_status),
        "resolution": None,
        "request_type": request_type,
        "affect_version": None,
        "components": "[]",
        "reporter_id": reporter.get("accountId", ""),
        "reporter_email": reporter.get("emailAddress", ""),
        "assignee_id": "",
        "created_at": created_at,
        "resolved_at": None,
        "updated_at": None,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source": "cloud",
        "is_cloud": 1,
        "is_uat_only": _is_uat_only(created_at),
    }


def _upsert_ticket(conn, ticket):
    conn.execute("""
        INSERT INTO tickets (ticket_key, summary, description, status, resolution,
                             request_type, affect_version, components, reporter_id,
                             reporter_email, assignee_id, created_at, resolved_at,
                             updated_at, fetched_at, source, is_cloud, is_uat_only)
        VALUES (:ticket_key, :summary, :description, :status, :resolution,
                :request_type, :affect_version, :components, :reporter_id,
                :reporter_email, :assignee_id, :created_at, :resolved_at,
                :updated_at, :fetched_at, :source, :is_cloud, :is_uat_only)
        ON CONFLICT(ticket_key) DO UPDATE SET
            summary=excluded.summary, description=excluded.description,
            status=excluded.status, resolution=excluded.resolution,
            request_type=excluded.request_type, affect_version=excluded.affect_version,
            components=excluded.components, assignee_id=excluded.assignee_id,
            resolved_at=excluded.resolved_at, updated_at=excluded.updated_at,
            fetched_at=excluded.fetched_at,
            is_cloud=excluded.is_cloud, is_uat_only=excluded.is_uat_only
    """, ticket)


def _fetch_bad_data_tickets(conn, cloud_auth):
    """Find and re-fetch tickets with suspect/missing data from the DB.

    Queries for up to 200 tickets whose status is NULL, empty, or whose
    summary is missing despite being in a terminal state, then re-fetches
    each one from the JSM servicedeskapi endpoint and upserts the fresh data.

    Called once per pipeline cycle after the main incremental fetch so that
    data-quality gaps accumulated from earlier ingest problems are healed
    incrementally without requiring a full re-sync.

    Returns:
        int: Number of tickets successfully refreshed.
    """
    rows = conn.execute("""
        SELECT ticket_key FROM tickets
        WHERE status IS NULL
           OR status = ''
           OR (status IN ('Resolved', 'Closed') AND summary IS NULL)
           OR (created_at IS NULL)
        LIMIT 200
    """).fetchall()

    if not rows:
        return 0

    base_url = get_cloud_base_url("jsm")
    fixed = 0
    errors = 0

    for row in rows:
        key = row[0] if not hasattr(row, "keys") else row["ticket_key"]
        try:
            data = api_get(
                f"{base_url}/rest/servicedeskapi/request/{key}",
                cloud_auth,
                _product="jsm",
            )
        except Exception as exc:
            log.warning("Bad-data sweep: failed to fetch %s: %s", key, exc)
            errors += 1
            continue

        ticket = _parse_ticket_jsm_request(data)
        _upsert_ticket(conn, ticket)
        fixed += 1

    if fixed or errors:
        conn.commit()

    return fixed


def _get_service_desk_id(headers, base_url, project_key: str) -> str | None:
    """Discover JSM service desk ID for a project key."""
    if project_key in _SD_ID_CACHE:
        return _SD_ID_CACHE[project_key]
    try:
        data = api_get(f"{base_url}/rest/servicedeskapi/servicedesk", headers, _product="jsm")
        for sd in data.get("values", []):
            if sd.get("projectKey") == project_key:
                _SD_ID_CACHE[project_key] = str(sd["id"])
                log.info("Service desk %s → ID %s", project_key, _SD_ID_CACHE[project_key])
                return _SD_ID_CACHE[project_key]
    except Exception as e:
        log.warning("Could not get service desk ID for %s: %s", project_key, e)
    return None


def fetch_tickets_cloud(project_key, affect_version=None, last_run_date=None):
    """Fetch tickets from a JSM Cloud project, using incremental or full sync.

    Hybrid strategy:
    - **Full sync** (``last_run_date`` is None): runs
      ``GET /rest/servicedeskapi/request?requestStatus=ALL_REQUESTS`` which
      fetches all 8 000+ tickets regardless of status.  Used on first run or
      whenever the DB has no prior run timestamp for this project.
    - **Incremental sync** (``last_run_date`` is set): uses the Jira Core
      JQL endpoint with ``updated >= "<last_run_date>"`` so only tickets
      changed since the previous pipeline cycle are fetched.  This is the
      normal path after the initial sync.

    The JSM ``servicedeskapi/request`` endpoint does not support date filtering,
    so incremental fetches always go through the JQL path.

    ``affect_version`` is accepted for backwards compatibility but is NOT
    applied as a filter — the goal is to ingest every ticket so analysis and
    classification have a complete picture.

    Returns:
        tuple[int, list[str]]: (total_upserted, list_of_ticket_keys_touched).
        The key list is used by the pipeline to scope comment fetching to only
        tickets that were actually updated this cycle.
    """
    if affect_version:
        log.info(
            "fetch_tickets_cloud: affect_version=%r received but NOT applied as a filter "
            "— fetching all tickets for project %s.",
            affect_version, project_key,
        )

    headers = get_cloud_auth("jsm")
    base_url = get_cloud_base_url("jsm")

    if last_run_date is None:
        # ── Full sync (first run) ───────────────────────────────────────────
        # Prefer the native JSM endpoint which captures portal-only fields.
        # Fall back to JQL if the service desk ID cannot be resolved.
        sd_id = _get_service_desk_id(headers, base_url, project_key)

        if sd_id:
            log.info(
                "Full sync for %s via servicedeskapi/request "
                "(requestStatus=ALL_REQUESTS, serviceDeskId=%s)",
                project_key, sd_id,
            )
            total, ingested_keys = _fetch_via_jsm_request_api(project_key, sd_id, headers, base_url)
        else:
            log.warning(
                "Could not resolve service desk ID for %s — falling back to "
                "/rest/api/3/search/jql (full scan) with JSM credential.",
                project_key,
            )
            total, ingested_keys = _fetch_via_jql(project_key, headers, base_url)
    else:
        # ── Incremental sync ────────────────────────────────────────────────
        # Only fetch tickets updated on or after last_run_date.
        since_str = last_run_date.isoformat() if hasattr(last_run_date, "isoformat") else str(last_run_date)
        log.info(
            "Incremental sync for %s via JQL (updated >= %s)",
            project_key, since_str,
        )
        total, ingested_keys = _fetch_via_jql(
            project_key, headers, base_url,
            updated_since=last_run_date,
        )

    # ── Data-quality sweep ──────────────────────────────────────────────────
    # Re-fetch up to 200 tickets per cycle that have null/bad status or missing
    # created_at so that ingest gaps are healed without a full re-sync.
    with get_db_conn() as conn:
        try:
            bad_fixed = _fetch_bad_data_tickets(conn, headers)
            if bad_fixed:
                log.info("Refreshed %d tickets with bad/missing data", bad_fixed)
        except Exception as e:
            log.warning("Bad-data sweep failed (non-fatal): %s", e)
            bad_fixed = 0

    return total, ingested_keys


def _fetch_via_jsm_request_api(project_key, sd_id, headers, base_url):
    """Fetch all JSM requests via GET /rest/servicedeskapi/request.

    Uses offset-based pagination (``start`` / ``limit``).  Returns a tuple of
    (total_upserted, list_of_ticket_keys) so the pipeline can scope comment
    fetching to only the tickets touched in this cycle.

    The ``requestStatus=ALL_REQUESTS`` parameter instructs the API to return
    requests in every lifecycle state (open, in-progress, resolved, closed).
    This requires the authenticating principal to have agent-level access;
    customer-level tokens only see that user's own requests.

    Response shape differs from the Jira Core /search/jql payload — parsed by
    ``_parse_ticket_jsm_request`` which reads ``requestFieldValues``,
    ``currentStatus``, ``createdDate``, etc.
    """
    page_size = 100
    start = 0
    total_fetched = 0
    page_num = 0
    ingested_keys: list[str] = []

    with get_db_conn() as conn:
        while True:
            params = {
                "requestStatus": "ALL_REQUESTS",
                "serviceDeskId": sd_id,
                "limit": page_size,
                "start": start,
            }

            try:
                data = api_get(
                    f"{base_url}/rest/servicedeskapi/request",
                    headers, _product="jsm",
                    params=params,
                )
            except Exception as e:
                log.error(
                    "servicedeskapi/request failed at start=%d: %s — "
                    "check that JSM OAuth app has read:servicedesk-request scope.",
                    start, e,
                )
                raise

            values = data.get("values", [])
            is_last = data.get("isLastPage")
            if is_last is None:
                log.warning("isLastPage absent from JSM response at start=%d — treating as last page", start)
                is_last = True
            page_num += 1

            for req in values:
                key = req.get("issueKey")
                if not key:
                    continue
                ticket = _parse_ticket_jsm_request(req)
                _upsert_ticket(conn, ticket)
                # servicedeskapi/request does not include issuelinks — skip link parse
                ingested_keys.append(key)
                total_fetched += 1

            conn.commit()

            log.debug(
                "JSM request page %d (start=%d): got=%d total_so_far=%d is_last=%s",
                page_num, start, len(values), total_fetched, is_last,
            )

            if is_last or not values:
                break

            start += len(values)

    log.info(
        "Cloud fetch complete (servicedeskapi/request): %d tickets upserted "
        "for %s (%d pages)",
        total_fetched, project_key, page_num,
    )
    return total_fetched, ingested_keys


def _fetch_via_jql(project_key, headers, base_url, updated_since=None):
    """Fetch tickets via GET /rest/api/3/search/jql with cursor-based pagination.

    Used for two purposes:
    - Full-scan fallback when the service desk ID cannot be resolved.
    - Incremental sync when ``updated_since`` is provided; the JQL filter is
      ``project = <key> AND updated >= "<date>" ORDER BY updated ASC`` which
      limits results to tickets changed on or after that date.

    The Jira Core /rest/api/3/search/jql endpoint uses cursor-based pagination
    via ``nextPageToken`` (Atlassian removed the legacy startAt-based
    /rest/api/3/search endpoint in August 2025).
    """
    from ingest.links import parse_issue_links

    if updated_since is not None:
        since_str = updated_since.isoformat() if hasattr(updated_since, "isoformat") else str(updated_since)
        jql = f'project = "{project_key}" AND updated >= "{since_str}" ORDER BY updated ASC'
    else:
        jql = f'project = "{project_key}" ORDER BY created ASC'
    fields = (
        "summary,description,status,resolution,created,resolutiondate,updated,"
        "reporter,assignee,versions,components,customfield_10010"
    )

    next_page_token = None
    page_size = 100
    total_fetched = 0
    page_num = 0
    ingested_keys: list[str] = []

    with get_db_conn() as conn:
        while True:
            params = {
                "jql": jql,
                "maxResults": page_size,
                "fields": fields,
            }
            if next_page_token:
                params["nextPageToken"] = next_page_token

            data = api_get(
                f"{base_url}/rest/api/3/search/jql",
                headers, _product="jsm",
                params=params,
            )

            issues = data.get("issues", [])
            next_page_token = data.get("nextPageToken")
            page_num += 1

            for issue in issues:
                key = issue.get("key")
                if not key:
                    continue
                ticket = _parse_ticket_cloud(issue)
                _upsert_ticket(conn, ticket)
                parse_issue_links(conn, ticket["ticket_key"], issue)
                ingested_keys.append(key)
                total_fetched += 1

            conn.commit()

            log.debug(
                "JQL fallback page %d: got=%d total_so_far=%d has_more=%s",
                page_num, len(issues), total_fetched, bool(next_page_token),
            )

            if not issues or not next_page_token:
                break

    mode = "incremental JQL" if updated_since is not None else "JQL full-scan"
    log.info(
        "Cloud fetch complete (%s): %d tickets upserted for %s (%d pages)",
        mode, total_fetched, project_key, page_num,
    )
    return total_fetched, ingested_keys


def fetch_descriptions_cloud(ticket_keys):
    """Backfill ticket descriptions from JSM requestFieldValues.

    The queue/issue endpoint returns Jira field format where description is always
    empty for portal-created tickets. The real user description lives in
    requestFieldValues[fieldId=="description"].value on the servicedeskapi request
    endpoint. This function fetches and updates that field for all given keys.
    """
    with get_db_conn() as conn:
        headers = get_cloud_auth("jsm")
        base_url = get_cloud_base_url("jsm")
        updated = 0
        errors = 0

        for i, key in enumerate(ticket_keys):
            try:
                data = api_get(
                    f"{base_url}/rest/servicedeskapi/request/{key}",
                    headers, _product="jsm",
                )
            except Exception as e:
                log.warning("Failed to fetch request for %s: %s", key, e)
                errors += 1
                continue

            description = ""
            for field in data.get("requestFieldValues", []):
                if field.get("fieldId") == "description":
                    val = field.get("value", "")
                    description = val if isinstance(val, str) else ""
                    break

            if description:
                conn.execute(
                    "UPDATE tickets SET description = ? WHERE ticket_key = ?",
                    (description, key),
                )
                updated += 1

            if (i + 1) % 50 == 0:
                conn.commit()
                log.info("  Descriptions: %d/%d tickets processed (%d updated)",
                         i + 1, len(ticket_keys), updated)

        conn.commit()

    log.info("Description backfill complete: %d updated, %d errors", updated, errors)
    return updated


def fetch_comments_cloud(ticket_keys):
    """Fetch comments for tickets from JSM Cloud via servicedeskapi.

    Uses GET /rest/servicedeskapi/request/{key}/comment which returns:
      values[].{id, author, body (plain text), public (bool), created.iso8601}
    """
    with get_db_conn() as conn:
        headers = get_cloud_auth("jsm")
        base_url = get_cloud_base_url("jsm")
        total = 0

        for i, key in enumerate(ticket_keys):
            try:
                data = api_get(
                    f"{base_url}/rest/servicedeskapi/request/{key}/comment",
                    headers, _product="jsm",
                    params={"limit": 100},
                )
            except Exception as e:
                log.warning("Failed to fetch comments for %s: %s", key, e)
                continue

            for comment in data.get("values", []):
                author = comment.get("author", {})
                is_public = comment.get("public", True)

                body = comment.get("body", "")
                if isinstance(body, dict):
                    body = _extract_adf_text(body)

                created_raw = comment.get("created", {})
                if isinstance(created_raw, dict):
                    created_at = created_raw.get("iso8601") or created_raw.get("jira", "")
                else:
                    created_at = str(created_raw)

                conn.execute("""
                    INSERT OR IGNORE INTO ticket_comments
                        (comment_id, ticket_key, author_id, author_name, body, is_public, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(comment["id"]), key,
                    author.get("accountId", ""), author.get("displayName", ""),
                    body, 1 if is_public else 0, created_at,
                ))
                total += 1

            if (i + 1) % 50 == 0:
                conn.commit()
                log.info("  Comments: %d/%d tickets processed", i + 1, len(ticket_keys))

    log.info("Fetched %d comments from Cloud", total)
    return total
