"""Issue link parsing and traversal."""

import logging
from datetime import datetime, timezone

log = logging.getLogger(__name__)


def parse_issue_links(conn, ticket_key, issue):
    """Parse and store issue links from a Jira issue."""
    fields = issue.get("fields", {})
    links = fields.get("issuelinks", [])
    if not links:
        return 0

    total = 0
    now = datetime.now(timezone.utc).isoformat()

    for link in links:
        link_type_obj = link.get("type", {})

        if "outwardIssue" in link:
            linked = link["outwardIssue"]
            conn.execute("""
                INSERT OR IGNORE INTO ticket_links
                    (ticket_key, link_type, linked_key, direction, linked_summary,
                     linked_status, linked_project, fetched_at)
                VALUES (?, ?, ?, 'outward', ?, ?, ?, ?)
            """, (
                ticket_key,
                link_type_obj.get("outward", link_type_obj.get("name", "relates to")),
                linked["key"],
                linked.get("fields", {}).get("summary", ""),
                (linked.get("fields", {}).get("status") or {}).get("name", ""),
                linked["key"].rsplit("-", 1)[0],
                now,
            ))
            total += 1

        if "inwardIssue" in link:
            linked = link["inwardIssue"]
            conn.execute("""
                INSERT OR IGNORE INTO ticket_links
                    (ticket_key, link_type, linked_key, direction, linked_summary,
                     linked_status, linked_project, fetched_at)
                VALUES (?, ?, ?, 'inward', ?, ?, ?, ?)
            """, (
                ticket_key,
                link_type_obj.get("inward", link_type_obj.get("name", "relates to")),
                linked["key"],
                linked.get("fields", {}).get("summary", ""),
                (linked.get("fields", {}).get("status") or {}).get("name", ""),
                linked["key"].rsplit("-", 1)[0],
                now,
            ))
            total += 1

    return total
