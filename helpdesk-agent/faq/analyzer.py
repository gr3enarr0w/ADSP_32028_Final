"""FAQ-specific gap analysis — extends analysis/gaps.py with Google Docs/Slides sources."""

import json
import logging
from datetime import datetime, timezone

from config import GEMINI_MODEL_ANALYSIS
from db import get_db_conn
from faq.issue_checker import check_linked_issues
from google.genai import types as genai_types

log = logging.getLogger(__name__)

FAQ_GAP_PROMPT = """You are analyzing FAQ coverage for an Atlassian DC-to-Cloud migration.

Compare the ticket patterns below against the existing documentation sources to identify
topics that need FAQ entries.

For each theme, assess:
1. Is this topic covered in the existing FAQ doc or slides?
2. Is it partially covered but missing detail?
3. Is it completely missing?

EXISTING FAQ DOCUMENT:
{google_doc_text}

EXISTING SLIDES/PRESENTATION:
{slides_text}

EXISTING DOCUMENTATION SPREADSHEETS (Q&A, guides, known issues):
{sheets_text}

EXISTING CONFLUENCE KB PAGES:
{confluence_text}

RECENT SLACK QUESTIONS:
{slack_text}

VENDOR REFERENCE — OFFICIAL ATLASSIAN DOCUMENTATION (for context only, NOT internal coverage):
{atlassian_docs_text}
NOTE: These are official Atlassian docs. They help you understand the product but do NOT
count as internal documentation coverage. Gaps should still be flagged if only vendor docs exist.

TICKET PATTERNS (grouped by issue type):
{ticket_patterns}

Return valid JSON array of themes:
[
  {{
    "theme": "<concise theme name>",
    "category": "<Access|Permissions|Data|Configuration|UI/UX|Integration|Performance|Workflow|Notifications|Other>",
    "ticket_count": <number of tickets matching>,
    "sample_issue_types": ["<issue_type_1>", "<issue_type_2>"],
    "coverage_status": "<missing|partial|covered>",
    "covered_by": "<which source covers it, or null>",
    "doc_section": "<which section, or null>",
    "gap_detail": "<what's missing or needs improvement>",
    "article_priority": "<high|medium|low>"
  }}
]

Priority rules:
- high: >15 tickets OR likely to recur post-migration OR has open linked bugs
- medium: 5-15 tickets
- low: <5 tickets or already partially covered
"""


def analyze_faq_gaps(sources: dict, client=None) -> list[dict]:
    """Compare all sources against ticket patterns to find FAQ gaps.

    Args:
        sources: Dict from sources.gather_all_sources()
        client: Optional pre-initialized Gemini client

    Returns:
        List of gap theme dicts stored in kb_coverage table.
    """
    with get_db_conn() as conn:
        rows = conn.execute("""
            SELECT c.issue_type, c.category, COUNT(*) as cnt,
                   SUM(CASE WHEN c.has_resolution = 1 THEN 1 ELSE 0 END) as resolved
            FROM ticket_classifications c
            JOIN tickets t ON t.ticket_key = c.ticket_key
            WHERE t.source = 'cloud'
            GROUP BY c.issue_type, c.category
            ORDER BY cnt DESC
        """).fetchall()

    ticket_patterns = "\n".join(
        f"- {r['issue_type']} ({r['category']}): {r['cnt']} tickets, {r['resolved']} resolved"
        for r in rows
    )

    # Build Confluence text summary
    confluence_text = "\n".join(
        f"- {p['title']}: {p.get('text', '')[:200]}..."
        for p in sources.get("confluence_pages", [])[:30]
    ) or "(no Confluence pages crawled)"

    # Build Slack text summary
    slack_text = "\n".join(
        f"- [{s.get('signal_type', 'question')}] {s.get('topic', '')}: {(s.get('message_text') or '')[:150]}"
        for s in sources.get("slack_signals", [])[:30]
    ) or "(no Slack signals)"

    # Build sheets text summary
    sheets_text = sources.get("google_sheets", "")
    if sheets_text:
        # Truncate to fit context but keep as much as possible
        sheets_text = sheets_text[:6000]
    else:
        sheets_text = "(no documentation spreadsheets configured)"

    # Build Atlassian docs reference text (vendor context, not coverage)
    atlassian_docs_text = "\n".join(
        f"- [{d.get('product', '')}] {d.get('title', '')} — {d.get('url', '')}"
        for d in sources.get("atlassian_docs", [])[:30]
    ) or "(no Atlassian docs indexed)"

    prompt = FAQ_GAP_PROMPT.format(
        google_doc_text=sources.get("google_doc", "(no FAQ document configured)")[:8000],
        slides_text=sources.get("google_slides", "(no slides configured)")[:4000],
        sheets_text=sheets_text,
        confluence_text=confluence_text,
        slack_text=slack_text,
        atlassian_docs_text=atlassian_docs_text,
        ticket_patterns=ticket_patterns,
    )

    if client is None:
        from core.genai import get_genai_client as _get_genai_client
        client = _get_genai_client()

    log.info("Running FAQ gap analysis with Gemini...")
    response = client.models.generate_content(
        model=GEMINI_MODEL_ANALYSIS,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            http_options=genai_types.HttpOptions(timeout=90_000),  # 90 s
        ),
    )

    from utils.gemini import parse_json_response
    themes = parse_json_response(response.text)
    now = datetime.now(timezone.utc).isoformat()

    # Enrich with linked issue data and store
    stored = 0
    with get_db_conn() as conn:
        for theme in themes:
            linked = check_linked_issues(theme["theme"])
            if linked["open_bugs"]:
                bug_keys = [b["issue_key"] for b in linked["open_bugs"]]
                theme["gap_detail"] = (theme.get("gap_detail") or "") + \
                    f" [OPEN BUGS: {', '.join(bug_keys)} — fix in progress]"
            if linked["open_rfes"]:
                rfe_keys = [r["issue_key"] for r in linked["open_rfes"]]
                theme["gap_detail"] = (theme.get("gap_detail") or "") + \
                    f" [PLANNED: {', '.join(rfe_keys)}]"

            conn.execute("""
                INSERT INTO kb_coverage
                    (theme, category, ticket_count, sample_issue_types,
                     coverage_status, covered_by, doc_section, gap_detail,
                     article_priority, assessed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(theme) DO UPDATE SET
                    category = excluded.category, ticket_count = excluded.ticket_count,
                    sample_issue_types = excluded.sample_issue_types,
                    coverage_status = excluded.coverage_status,
                    covered_by = excluded.covered_by, doc_section = excluded.doc_section,
                    gap_detail = excluded.gap_detail, article_priority = excluded.article_priority,
                    assessed_at = excluded.assessed_at
            """, (
                theme["theme"], theme.get("category", "Other"),
                theme.get("ticket_count", 0),
                json.dumps(theme.get("sample_issue_types", [])),
                theme.get("coverage_status", "missing"),
                theme.get("covered_by"),
                theme.get("doc_section"),
                theme.get("gap_detail"),
                theme.get("article_priority", "medium"),
                now,
            ))
            stored += 1

    log.info("FAQ gap analysis complete: %d themes identified", stored)
    return themes
