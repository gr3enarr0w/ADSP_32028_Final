"""Scheduled JQL sweep — find undrafted open tickets and auto-draft responses."""

from __future__ import annotations

import logging

from analysis.router import route_ticket
from config import AUTO_DRAFT_ALL, AUTO_RESPOND_ASSIGNEES
from core.pipeline import get_plugin_config
from db import get_db_conn
from ingest.oauth2lo import get_cloud_base_url
from plugins.responder.jira_comments import _jira_request
from plugins.responder.drafting import (
    _get_latest_customer_reply,
    _get_ticket_context,
    _lookup_and_draft,
)
from plugins.responder.gates import (
    _check_age_gate,
    _check_content_gate,
    _check_status_gate,
    _get_confidence_threshold,
    check_classification_gates,
)

log = logging.getLogger(__name__)


def _auto_draft_sweep():
    """Find ALL open tickets with no AI draft and draft them.

    When AUTO_DRAFT_ALL is true, drafts for all open <PROJECT_KEY> tickets
    regardless of assignee. When false, only drafts for AUTO_RESPOND_ASSIGNEES.
    """
    base = get_cloud_base_url("jsm")

    if AUTO_DRAFT_ALL:
        jql = (
            'project = <PROJECT_KEY> '
            'AND status in ("Waiting for support", "In Progress", "New", "Escalated") '
            'ORDER BY created DESC'
        )
    elif AUTO_RESPOND_ASSIGNEES:
        assignee_filter = ", ".join(f'"{a}"' for a in AUTO_RESPOND_ASSIGNEES)
        jql = (
            f'project = <PROJECT_KEY> AND assignee in ({assignee_filter}) '
            f'AND status in ("Waiting for support", "In Progress", "New", "Escalated") '
            f'ORDER BY created DESC'
        )
    else:
        return

    search_url = f"{base}/rest/api/3/search/jql"
    search_params = {"jql": jql, "fields": "summary,status", "maxResults": 200}
    try:
        resp = _jira_request("get", search_url, "jsm", params=search_params, timeout=15)
        if resp.status_code != 200:
            log.error("Auto-draft sweep: JQL search failed (%d)", resp.status_code)
            return
        issues = resp.json().get("issues", [])
    except Exception as e:
        log.warning("Auto-draft sweep: %s", e)
        return

    log.info("Auto-draft sweep: JQL returned %d tickets", len(issues))
    if not issues:
        return

    # Check which already have drafts
    with get_db_conn() as conn:
        drafted = {r[0] for r in conn.execute(
            "SELECT DISTINCT ticket_key FROM ai_draft_feedback WHERE 1=1"
        ).fetchall()}

    undrafted = [i["key"] for i in issues if i["key"] not in drafted]
    log.info(
        "Auto-draft sweep: %d already drafted, %d undrafted to process",
        len(issues) - len(undrafted),
        len(undrafted),
    )
    if not undrafted:
        return

    responder_cfg = get_plugin_config("responder")
    confidence_threshold = _get_confidence_threshold(responder_cfg)

    posted = 0
    for key in undrafted:
        try:
            if not _check_status_gate(key):
                log.debug("Auto-draft sweep: %s — skipped (status gate)", key)
                continue
            if not _check_age_gate(key):
                log.debug("Auto-draft sweep: %s — skipped (age gate)", key)
                continue
            if not _check_content_gate(key):
                log.debug("Auto-draft sweep: %s — skipped (content gate)", key)
                continue
            if not check_classification_gates(key, responder_cfg):
                log.debug("Auto-draft sweep: %s — skipped (classification gate)", key)
                continue
            routing = route_ticket(key, confidence_threshold=confidence_threshold)
            if routing.route != "auto_draft":
                log.info(
                    "Auto-draft sweep: %s → %s (M5 router: %s) — skipping",
                    key, routing.route, routing.reason,
                )
                continue
            ctx = _get_ticket_context(key)
            if not ctx:
                log.warning("Auto-draft sweep: %s — no ticket context (API failure?)", key)
                continue
            summary, desc = ctx
            reply = _get_latest_customer_reply(key)
            if reply:
                desc = desc + "\n\nCUSTOMER FOLLOW-UP:\n" + reply
            if _lookup_and_draft(key, summary, desc):
                posted += 1
            else:
                log.info(
                    "Auto-draft sweep: %s — draft not posted (see drafting logs for reason)",
                    key,
                )
        except Exception:
            log.exception("Auto-draft sweep: error on %s", key)

    log.info("Auto-draft sweep: complete — %d/%d drafts posted", posted, len(undrafted))
