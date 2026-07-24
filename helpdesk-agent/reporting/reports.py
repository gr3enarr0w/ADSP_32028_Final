"""Report data structure builders."""

import json
import logging

from db import get_db

log = logging.getLogger(__name__)


def build_ticket_summary(version=None):
    """Build ticket summary report data."""
    conn = get_db()

    version_filter = ""
    params = []
    if version:
        version_filter = " WHERE tc.affect_version = ?"
        params.append(version)

    rows = conn.execute(f"""
        SELECT tc.category, tc.issue_type, COUNT(*) as cnt,
               ROUND(CAST(AVG(tc.confidence) AS NUMERIC), 2) as avg_conf,
               COUNT(cs.csat_score) as csat_responses,
               ROUND(CAST(AVG(cs.csat_score) AS NUMERIC), 2) as avg_csat
        FROM ticket_classifications tc
        LEFT JOIN ticket_csat cs ON tc.ticket_key = cs.ticket_key
        {version_filter}
        GROUP BY tc.category, tc.issue_type
        ORDER BY tc.category, cnt DESC
    """, params).fetchall()

    total = conn.execute(f"""
        SELECT COUNT(*)
        FROM ticket_classifications tc
        {version_filter}
    """, params).fetchone()[0]
    total_csat_responses = conn.execute(f"""
        SELECT COUNT(cs.csat_score)
        FROM ticket_classifications tc
        LEFT JOIN ticket_csat cs ON tc.ticket_key = cs.ticket_key
        {version_filter}
    """, params).fetchone()[0]

    conn.close()

    categories = {}
    for row in rows:
        cat = row["category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "csat_responses": 0, "issue_types": []}
        categories[cat]["total"] += row["cnt"]
        categories[cat]["csat_responses"] += row["csat_responses"]
        categories[cat]["issue_types"].append({
            "issue_type": row["issue_type"],
            "count": row["cnt"],
            "percentage": round(row["cnt"] / total * 100, 1) if total else 0,
            "avg_confidence": row["avg_conf"],
            "csat_responses": row["csat_responses"],
            "avg_csat": row["avg_csat"],
        })

    return {
        "total_classified": total,
        "total_csat_responses": total_csat_responses,
        "categories": categories,
        "version": version,
    }


def build_dashboard_data():
    """Build summary data for the dashboard API."""
    conn = get_db()

    stats = {
        "tickets": conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0],
        "comments": conn.execute("SELECT COUNT(*) FROM ticket_comments").fetchone()[0],
        "classifications": conn.execute("SELECT COUNT(*) FROM ticket_classifications").fetchone()[0],
        "csat_responses": conn.execute("SELECT COUNT(*) FROM ticket_csat").fetchone()[0],
        "links": conn.execute("SELECT COUNT(*) FROM ticket_links").fetchone()[0],
        "themes": conn.execute("SELECT COUNT(*) FROM kb_coverage").fetchone()[0],
        "articles": conn.execute("SELECT COUNT(*) FROM generated_articles").fetchone()[0],
    }

    coverage = conn.execute("""
        SELECT coverage_status, COUNT(*) as cnt
        FROM kb_coverage GROUP BY coverage_status
    """).fetchall()
    stats["coverage"] = {r["coverage_status"]: r["cnt"] for r in coverage}

    conn.close()
    return stats
