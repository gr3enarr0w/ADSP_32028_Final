#!/usr/bin/env python3
"""AI Helpdesk Agent CLI — ingest, classify, FAQ generation, and knowledge base management."""

import argparse
import json
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    FAQ_SOURCE_DOC_IDS, FAQ_OUTPUT_DOC_ID, FAQ_SLIDES_ID,
    FAQ_SOURCE_SHEET_IDS, FAQ_CONFLUENCE_SPACES,
    CLOUD_URL, GEMINI_MODEL_CLASSIFICATION, GEMINI_MODEL_GENERATION, GEMINI_MODEL_ANALYSIS,
    PROJECT_KEYS, DEFAULT_AFFECT_VERSION, mask_id,
)
from db import init_db, get_db

_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, _LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def cmd_ingest(args):
    """Fetch tickets and comments from Jira Cloud, scrub PII."""
    from ingest.tickets import fetch_tickets_cloud, fetch_comments_cloud
    from ingest.scrubber import scrub_database

    projects = [p.strip() for p in args.project.split(",")] if args.project else PROJECT_KEYS
    version = args.version or DEFAULT_AFFECT_VERSION or None

    if version:
        print(f"  Filtering by affectedVersion: {version}")

    total = 0
    for project_key in projects:
        print(f"\n  Fetching tickets from {project_key}...")
        fetched, _ = fetch_tickets_cloud(project_key, version)
        total += fetched

    print(f"\n  Total tickets fetched: {total}")

    # Fetch comments
    conn = get_db()
    all_keys = [r[0] for r in conn.execute("SELECT ticket_key FROM tickets").fetchall()]
    conn.close()
    if all_keys:
        print(f"  Fetching comments for {len(all_keys)} tickets...")
        fetch_comments_cloud(all_keys)

    # Auto-scrub PII
    print("  Scrubbing PII and anonymizing identities...")
    scrub_result = scrub_database(dry_run=False)
    scrubbed = scrub_result["scrubbed_tickets"] + scrub_result["scrubbed_comments"]
    anon = scrub_result["anon_reporters"] + scrub_result["anon_authors"]
    print(f"  Scrubbed {scrubbed} text fields, anonymized {anon} identity fields")
    print()


def cmd_classify(args):
    """Classify unclassified tickets with Gemini AI and update Jira ticket type."""
    from analysis.classifier import classify_unclassified
    from db import get_db_conn

    version = args.version or DEFAULT_AFFECT_VERSION or None

    # Mark newly-resolved tickets for re-classification
    with get_db_conn() as conn:
        stale = conn.execute("""
            SELECT COUNT(*) FROM tickets t
            JOIN ticket_classifications c ON t.ticket_key = c.ticket_key
            WHERE t.resolution IS NOT NULL
              AND (c.has_resolution = 0 OR c.has_resolution IS NULL)
        """).fetchone()[0]
        if stale > 0:
            conn.execute("""
                DELETE FROM ticket_classifications WHERE ticket_key IN (
                    SELECT t.ticket_key FROM tickets t
                    JOIN ticket_classifications c ON t.ticket_key = c.ticket_key
                    WHERE t.resolution IS NOT NULL
                      AND (c.has_resolution = 0 OR c.has_resolution IS NULL)
                )
            """)
            print(f"  Marked {stale} newly-resolved tickets for re-classification")

    classified, errors = classify_unclassified(version)
    print(f"\n  Classified: {classified}")
    print(f"  Errors:     {errors}")
    print()


def cmd_lookup(args):
    """Search FAQ entries, KB articles, and resolved tickets for a query."""
    from faq.lookup import lookup, format_not_found

    query = args.query
    print(f"\n  Looking up: {query}")

    result = lookup(query)
    if not result["found"]:
        print(f"\n  {format_not_found(query)}\n")
        return

    if result["faq_matches"]:
        print(f"\n  FAQ Matches ({len(result['faq_matches'])}):")
        for faq in result["faq_matches"]:
            print(f"    - {faq['title']}")

    if result["kb_matches"]:
        print(f"\n  KB Articles ({len(result['kb_matches'])}):")
        for kb in result["kb_matches"]:
            url = kb.get("url", "")
            print(f"    - {kb['title']}{f' ({url})' if url else ''}")

    if result["ticket_matches"]:
        print(f"\n  Similar Resolved Tickets ({len(result['ticket_matches'])}):")
        for t in result["ticket_matches"][:3]:
            print(f"    - {t['ticket_key']}: {t['summary']}")
            if t.get("resolution_summary"):
                print(f"      Resolution: {t['resolution_summary'][:150]}")

    if result.get("atlassian_matches"):
        print(f"\n  Atlassian Docs ({len(result['atlassian_matches'])}):")
        for doc in result["atlassian_matches"]:
            print(f"    - {doc['title']} ({doc['url']})")

    if result.get("response_draft"):
        print(f"\n  --- Draft Response ---")
        print(f"  {result['response_draft'][:500]}")

    print()


def cmd_sources(args):
    """Show all configured sources and their status."""
    from faq.sources import get_source_status

    print(f"\n  Cloud URL:  {CLOUD_URL or '(not configured)'}")
    print(f"  Auth:       OAuth 2LO (ants-engineering service account)")
    print()

    print(f"  Gemini Models:")
    print(f"    Classification: {GEMINI_MODEL_CLASSIFICATION}")
    print(f"    Generation:     {GEMINI_MODEL_GENERATION}")
    print(f"    Analysis:       {GEMINI_MODEL_ANALYSIS}")
    print()

    print("  Configured Sources:")
    for i, doc_id in enumerate(FAQ_SOURCE_DOC_IDS, 1):
        print(f"    Google Doc (source {i}): [{mask_id(doc_id)}]")
    if not FAQ_SOURCE_DOC_IDS:
        print("    Google Docs (source): (not set)")
    print(f"    Google Doc (output):  [{mask_id(FAQ_OUTPUT_DOC_ID)}]" if FAQ_OUTPUT_DOC_ID else "    Google Doc (output):  (not set)")
    print(f"    Google Slides:        [{mask_id(FAQ_SLIDES_ID)}]" if FAQ_SLIDES_ID else "    Google Slides:        (not set)")
    for i, sheet_id in enumerate(FAQ_SOURCE_SHEET_IDS, 1):
        print(f"    Google Sheet (source {i}): [{mask_id(sheet_id)}]")
    if not FAQ_SOURCE_SHEET_IDS:
        print("    Google Sheets (source): (not set)")
    print(f"    Confluence Spaces:    {', '.join(FAQ_CONFLUENCE_SPACES) or '(not set)'}")

    # Show dynamic link projects and Atlassian docs count
    conn = get_db()
    link_projects = conn.execute("""
        SELECT DISTINCT linked_project FROM ticket_links
        WHERE linked_project IS NOT NULL
        ORDER BY linked_project
    """).fetchall()
    atlassian_rows = conn.execute("""
        SELECT product, COUNT(*) as cnt FROM atlassian_docs GROUP BY product
    """).fetchall()
    conn.close()
    if link_projects:
        projects = [r["linked_project"] for r in link_projects]
        print(f"    Link Projects:        {', '.join(projects)} (auto-discovered)")
    if atlassian_rows:
        total = sum(r["cnt"] for r in atlassian_rows)
        breakdown = ", ".join(f"{r['product']}: {r['cnt']}" for r in atlassian_rows)
        print(f"    Atlassian Docs:       {total} article URLs ({breakdown})")
    else:
        print(f"    Atlassian Docs:       (not indexed — run 'crawl-docs')")
    print()

    status = get_source_status()
    if status:
        print("  Source Freshness:")
        for s in status:
            fetched = s.get("last_fetched") or "never"
            print(f"    [{s['source_type']}] {s.get('title', s['source_id'])}: last fetched {fetched}")
    else:
        print("  No sources fetched yet. Run 'analyze' to pull sources.")
    print()


def cmd_analyze(args):
    """Run FAQ gap analysis against all sources."""
    from faq.sources import gather_all_sources
    from faq.analyzer import analyze_faq_gaps

    print("\nGathering FAQ sources...")
    sources = gather_all_sources()

    print(f"\n  Sources collected:")
    print(f"    Google Doc:    {len(sources['google_doc']):,} chars")
    print(f"    Google Slides: {len(sources['google_slides']):,} chars")
    print(f"    Google Sheets: {len(sources['google_sheets']):,} chars")
    print(f"    Confluence:    {len(sources['confluence_pages'])} pages")
    print(f"    Slack signals: {len(sources['slack_signals'])}")
    print(f"    Resolutions:   {len(sources['ticket_resolutions'])} tickets")

    print("\nRunning FAQ gap analysis...")
    themes = analyze_faq_gaps(sources)

    missing = [t for t in themes if t.get("coverage_status") == "missing"]
    partial = [t for t in themes if t.get("coverage_status") == "partial"]
    covered = [t for t in themes if t.get("coverage_status") == "covered"]

    print(f"\n  Results: {len(themes)} themes")
    print(f"    MISSING: {len(missing)} themes ({sum(t.get('ticket_count', 0) for t in missing)} tickets)")
    print(f"    PARTIAL: {len(partial)} themes ({sum(t.get('ticket_count', 0) for t in partial)} tickets)")
    print(f"    COVERED: {len(covered)} themes ({sum(t.get('ticket_count', 0) for t in covered)} tickets)")

    if missing:
        print("\n  Missing FAQ coverage:")
        for t in missing:
            priority = t.get("article_priority", "medium")
            print(f"    [{priority.upper()}] {t['theme']} ({t.get('ticket_count', 0)} tickets)")
            if t.get("gap_detail"):
                print(f"          {t['gap_detail'][:120]}")
    print()


def cmd_generate(args):
    """Generate FAQ entries for gaps."""
    from faq.sources import gather_all_sources
    from faq.generator import generate_all_faq_entries

    print("\nGathering sources...")
    sources = gather_all_sources()

    theme = args.theme if hasattr(args, "theme") and args.theme else None
    print(f"\nGenerating FAQ entries{f' for: {theme}' if theme else ' for all gaps'}...")

    generated, errors = generate_all_faq_entries(sources, theme_filter=theme)
    print(f"\n  Generated: {generated}")
    print(f"  Errors:    {errors}")

    if generated > 0:
        conn = get_db()
        entries = conn.execute("""
            SELECT id, article_topic, title FROM generated_articles
            WHERE format = 'faq' ORDER BY generated_at DESC LIMIT ?
        """, (generated,)).fetchall()
        conn.close()

        print("\n  New entries:")
        for e in entries:
            print(f"    #{e['id']}: {e['title']}")
    print()


def cmd_export(args):
    """Write FAQ entries to output Google Doc."""
    from faq.google_docs import write_faq_entries
    from utils.html_parser import parse_faq_html as _parse_faq_html

    conn = get_db()
    rows = conn.execute("""
        SELECT article_topic, title, body_html FROM generated_articles
        WHERE format = 'faq' AND status = 'draft'
        ORDER BY generated_at
    """).fetchall()
    conn.close()

    if not rows:
        print("\nNo FAQ entries to export. Run 'generate' first.\n")
        return

    entries = []
    for r in rows:
        r = dict(r)
        parsed = _parse_faq_html(r["body_html"])
        entries.append({"topic": r["title"], **parsed})

    print(f"\nExporting {len(entries)} FAQ entries to Google Doc...")
    success = write_faq_entries(entries)
    if success:
        print(f"  Exported {len(entries)} entries to doc [{mask_id(FAQ_OUTPUT_DOC_ID)}]\n")
    else:
        print("  Export failed. Check FAQ_OUTPUT_DOC_ID configuration.\n")


def cmd_publish(args):
    """Publish FAQ entries to Confluence."""
    from generation.publisher import publish_article

    if not CLOUD_URL:
        print("\nNo Cloud URL configured. Cannot publish.\n")
        return

    conn = get_db()
    rows = conn.execute("""
        SELECT id, article_topic, title FROM generated_articles
        WHERE format = 'faq' AND status = 'draft'
        ORDER BY generated_at
    """).fetchall()
    conn.close()

    if not rows:
        print("\nNo FAQ entries to publish. Run 'generate' first.\n")
        return

    print(f"\nPublishing {len(rows)} FAQ entries to Confluence (Cloud)...")

    published = 0
    for r in rows:
        space = args.space if hasattr(args, "space") and args.space else None
        success = publish_article(r["id"], space_key=space)
        if success:
            published += 1
            print(f"  Published: {r['title']}")
        else:
            print(f"  FAILED: {r['title']}")

    print(f"\n  Published: {published}/{len(rows)}\n")


def cmd_crawl_docs(args):
    """Crawl Atlassian Cloud documentation sites and GitHub repos."""
    from faq.atlassian_docs import fetch_all_doc_sites
    from faq.github_docs import fetch_all_github_repos
    from config import ATLASSIAN_DOC_URLS, GITHUB_DOC_REPOS

    print(f"\nIndexing {len(ATLASSIAN_DOC_URLS)} Atlassian doc sites...")
    for url in ATLASSIAN_DOC_URLS:
        print(f"  - {url}")

    total = fetch_all_doc_sites()

    if GITHUB_DOC_REPOS:
        print(f"\nIndexing {len(GITHUB_DOC_REPOS)} GitHub repos...")
        for repo in GITHUB_DOC_REPOS:
            print(f"  - {repo}")
        total += fetch_all_github_repos()

    print(f"\n  Stored/updated: {total} article URLs")

    conn = get_db()
    rows = conn.execute("""
        SELECT product, COUNT(*) as cnt FROM atlassian_docs GROUP BY product
    """).fetchall()
    conn.close()

    if rows:
        print("\n  Article URLs by product:")
        for r in rows:
            print(f"    {r['product']}: {r['cnt']}")
    print()


def cmd_run(args):
    """Full pipeline: sources → analyze → generate → export."""
    print("\n=== FAQ Service — Full Pipeline ===")
    print(f"  Auth: OAuth 2LO (ants-engineering)")
    print()

    cmd_analyze(args)
    cmd_generate(args)
    cmd_export(args)

    print("=== Pipeline Complete ===\n")


def main():
    parser = argparse.ArgumentParser(
        description="FAQ Service — Atlassian Cloud Migration FAQ generation and management",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("sources", help="Show all configured sources and their status")
    sub.add_parser("analyze", help="Run FAQ gap analysis against all sources")

    gen_parser = sub.add_parser("generate", help="Generate FAQ entries for gaps")
    gen_parser.add_argument("--theme", help="Generate for a specific theme only")

    sub.add_parser("crawl-docs", help="Crawl Atlassian Cloud documentation sites")
    sub.add_parser("export", help="Write FAQ entries to output Google Doc")

    pub_parser = sub.add_parser("publish", help="Publish FAQ entries to Confluence")
    pub_parser.add_argument("--space", help="Override Confluence space key")

    run_parser = sub.add_parser("run", help="Full pipeline: sources → analyze → generate → export")
    run_parser.add_argument("--theme", help="Filter to a specific theme")

    args = parser.parse_args()

    init_db()

    commands = {
        "sources": cmd_sources,
        "crawl-docs": cmd_crawl_docs,
        "analyze": cmd_analyze,
        "generate": cmd_generate,
        "export": cmd_export,
        "publish": cmd_publish,
        "run": cmd_run,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
