"""Check linked issues for known bugs/RFEs related to FAQ topics."""

import json
import logging

from config import LINK_FOLLOW_PROJECTS
from db import get_db

log = logging.getLogger(__name__)

_EMPTY_RESULT = {"open_bugs": [], "open_rfes": [], "resolved_bugs": []}


def check_linked_issues(theme: str) -> dict:
    """Check if a FAQ theme has linked bugs/RFEs in HATMOS, RH1, etc.

    Returns dict with:
        open_bugs: list — Active bugs (don't FAQ, team is fixing)
        open_rfes: list — Feature requests (mention as "planned")
        resolved_bugs: list — Fixed bugs (can FAQ with workaround/fix)
    """
    conn = get_db()
    try:
        theme_row = conn.execute(
            "SELECT sample_issue_types FROM kb_coverage WHERE theme = ?", (theme,)
        ).fetchone()

        if not theme_row or not theme_row["sample_issue_types"]:
            return dict(_EMPTY_RESULT)

        issue_types = json.loads(theme_row["sample_issue_types"])
        if not issue_types:
            return dict(_EMPTY_RESULT)

        placeholders = ",".join(["?"] * len(issue_types))
        ticket_keys = [r["ticket_key"] for r in conn.execute(f"""
            SELECT ticket_key FROM ticket_classifications
            WHERE issue_type IN ({placeholders})
        """, issue_types).fetchall()]

        if not ticket_keys:
            return dict(_EMPTY_RESULT)

        key_placeholders = ",".join(["?"] * len(ticket_keys))
        linked_keys = [r["linked_key"] for r in conn.execute(f"""
            SELECT DISTINCT linked_key FROM ticket_links
            WHERE ticket_key IN ({key_placeholders})
        """, ticket_keys).fetchall()]

        if not linked_keys:
            return dict(_EMPTY_RESULT)

        linked_placeholders = ",".join(["?"] * len(linked_keys))
        issues = conn.execute(f"""
            SELECT issue_key, project_key, summary, status, resolution,
                   priority, issue_type
            FROM linked_issues
            WHERE issue_key IN ({linked_placeholders})
        """, linked_keys).fetchall()
    finally:
        conn.close()

    result = {"open_bugs": [], "open_rfes": [], "resolved_bugs": []}

    for issue in issues:
        issue = dict(issue)
        itype = (issue.get("issue_type") or "").lower()
        status = (issue.get("status") or "").lower()
        resolution = issue.get("resolution") or ""

        is_resolved = status in ("closed", "done", "resolved") or bool(resolution)
        is_bug = "bug" in itype
        is_rfe = any(t in itype for t in ("story", "feature", "enhancement", "rfe"))

        if is_bug and is_resolved:
            result["resolved_bugs"].append(issue)
        elif is_bug and not is_resolved:
            result["open_bugs"].append(issue)
        elif is_rfe and not is_resolved:
            result["open_rfes"].append(issue)

    log.info("Theme '%s': %d open bugs, %d open RFEs, %d resolved bugs",
             theme, len(result["open_bugs"]), len(result["open_rfes"]),
             len(result["resolved_bugs"]))
    return result
