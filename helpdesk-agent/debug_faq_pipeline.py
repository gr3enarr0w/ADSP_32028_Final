#!/usr/bin/env python3
"""Diagnostic script to debug FAQ auto-generation pipeline issues for Access/Permissions tickets.

Run on OCP cluster via:
  POD=$(oc get pods -n jira-messaging--runtime-ext -l app=ai-helpdesk-agent -o jsonpath='{.items[0].metadata.name}')
  oc exec -n jira-messaging--runtime-ext $POD -- python3 debug_faq_pipeline.py
"""

import sys
import json
from db import get_db_conn

def main():
    print("\n=== FAQ Pipeline Diagnostic ===\n")

    with get_db_conn() as conn:
        # ─────────────────────────────────────────────────────────────────
        # 1. What FAQ articles exist for Access/Permissions topics?
        # ─────────────────────────────────────────────────────────────────
        print("1. RECENT FAQ ARTICLES (last 20):")
        print("   " + "─" * 80)

        articles = conn.execute("""
            SELECT article_topic, title, status, confluence_page_id IS NOT NULL as published,
                   generated_at
            FROM generated_articles
            WHERE format = 'faq'
            ORDER BY generated_at DESC
            LIMIT 20
        """).fetchall()

        if not articles:
            print("   (no FAQ articles generated)")
        else:
            for a in articles:
                a = dict(a)
                status_marker = "[PUB]" if a["published"] else "[DRAFT]"
                print(f"   {status_marker} {a['title']}")
                print(f"         Topic: {a['article_topic']}")
                print(f"         Generated: {a['generated_at']}")

        # Count Access/Permissions-related FAQs
        access_faqs = conn.execute("""
            SELECT COUNT(*) as cnt FROM generated_articles
            WHERE format = 'faq'
              AND (article_topic ILIKE '%access%'
                OR article_topic ILIKE '%permission%'
                OR article_topic ILIKE '%configuration%')
        """).fetchone()[0]

        print(f"\n   Access/Permissions FAQs: {access_faqs}")

        # ─────────────────────────────────────────────────────────────────
        # 2. Do we have resolved Access/Permissions tickets with resolution_summary?
        # ─────────────────────────────────────────────────────────────────
        print("\n2. RESOLVED TICKETS BY CATEGORY:")
        print("   " + "─" * 80)

        tickets_by_category = conn.execute("""
            SELECT tc.category, COUNT(*) as total_cnt,
                   COUNT(CASE WHEN t.resolution IS NOT NULL THEN 1 END) as resolved_cnt,
                   COUNT(CASE WHEN t.resolution_summary IS NOT NULL THEN 1 END) as with_summary_cnt
            FROM ticket_classifications tc
            JOIN tickets t ON t.ticket_key = tc.ticket_key
            WHERE tc.category IN ('Access','Permissions','Configuration')
            GROUP BY tc.category
            ORDER BY total_cnt DESC
        """).fetchall()

        if not tickets_by_category:
            print("   (no Access/Permissions/Configuration tickets)")
        else:
            for row in tickets_by_category:
                row = dict(row)
                print(f"   {row['category']}:")
                print(f"     Total: {row['total_cnt']}")
                print(f"     Resolved: {row['resolved_cnt']}")
                print(f"     With resolution_summary: {row['with_summary_cnt']}")

        # ─────────────────────────────────────────────────────────────────
        # 3. What does kb_coverage say about Access/Permissions themes?
        # ─────────────────────────────────────────────────────────────────
        print("\n3. KB_COVERAGE FOR ACCESS/PERMISSIONS THEMES:")
        print("   " + "─" * 80)

        coverage = conn.execute("""
            SELECT theme, category, coverage_status, ticket_count, article_priority,
                   gap_detail, assessed_at
            FROM kb_coverage
            WHERE category IN ('Access','Permissions','Configuration')
                   OR theme ILIKE '%access%'
                   OR theme ILIKE '%permission%'
            ORDER BY ticket_count DESC
            LIMIT 15
        """).fetchall()

        if not coverage:
            print("   (no kb_coverage entries for Access/Permissions)")
        else:
            for c in coverage:
                c = dict(c)
                print(f"   [{c['coverage_status'].upper()}] {c['theme']}")
                print(f"     Category: {c['category']}")
                print(f"     Tickets: {c['ticket_count']}")
                print(f"     Priority: {c['article_priority']}")
                if c['gap_detail']:
                    print(f"     Gap: {c['gap_detail'][:100]}")

        # ─────────────────────────────────────────────────────────────────
        # 4. Check for gaps marked 'missing' but no corresponding FAQ article
        # ─────────────────────────────────────────────────────────────────
        print("\n4. MISSING GAPS WITHOUT FAQ ARTICLES:")
        print("   " + "─" * 80)

        missing_gaps = conn.execute("""
            SELECT DISTINCT kc.theme, kc.ticket_count, kc.article_priority
            FROM kb_coverage kc
            LEFT JOIN generated_articles ga ON kc.theme = ga.article_topic
                                             AND ga.format = 'faq'
            WHERE kc.coverage_status IN ('missing', 'partial')
              AND (kc.category IN ('Access','Permissions','Configuration')
                   OR kc.theme ILIKE '%access%'
                   OR kc.theme ILIKE '%permission%')
              AND ga.id IS NULL
            ORDER BY kc.ticket_count DESC
            LIMIT 10
        """).fetchall()

        if not missing_gaps:
            print("   (no missing gaps without FAQ articles)")
        else:
            for gap in missing_gaps:
                gap = dict(gap)
                print(f"   {gap['theme']}")
                print(f"     Tickets: {gap['ticket_count']}")
                print(f"     Priority: {gap['article_priority']}")

        # ─────────────────────────────────────────────────────────────────
        # 5. Check generator.py logic: min_tickets_per_theme threshold
        # ─────────────────────────────────────────────────────────────────
        print("\n5. GENERATOR LOGIC CHECK:")
        print("   " + "─" * 80)

        # Count ungenerated gaps by priority
        ungenerated = conn.execute("""
            SELECT
                kc.article_priority,
                COUNT(*) as theme_count,
                MIN(kc.ticket_count) as min_tickets,
                MAX(kc.ticket_count) as max_tickets,
                AVG(kc.ticket_count) as avg_tickets
            FROM kb_coverage kc
            LEFT JOIN generated_articles ga ON kc.theme = ga.article_topic
                                             AND ga.format = 'faq'
            WHERE kc.coverage_status IN ('missing', 'partial')
              AND ga.id IS NULL
            GROUP BY kc.article_priority
            ORDER BY CASE kc.article_priority
                    WHEN 'high' THEN 0
                    WHEN 'medium' THEN 1
                    ELSE 2 END
        """).fetchall()

        print("   Ungenerated gaps by priority:")
        for row in ungenerated:
            row = dict(row)
            print(f"   {row['article_priority'].upper()}: {row['theme_count']} themes")
            print(f"     Ticket range: {row['min_tickets']}-{row['max_tickets']}")
            print(f"     Average: {row['avg_tickets']:.1f}")

        # ─────────────────────────────────────────────────────────────────
        # 6. Identify possible pipeline issues
        # ─────────────────────────────────────────────────────────────────
        print("\n6. PIPELINE ISSUE ANALYSIS:")
        print("   " + "─" * 80)

        # Check if gap analysis has ever run
        all_coverage_count = conn.execute(
            "SELECT COUNT(*) FROM kb_coverage"
        ).fetchone()[0]

        if all_coverage_count == 0:
            print("   ISSUE: No kb_coverage entries at all")
            print("   FIX: Run 'python faq_service.py analyze' first")
        else:
            print(f"   Gap analysis has run: {all_coverage_count} themes identified")

            # Check if there are missing/partial gaps but no generated articles
            missing_no_articles = conn.execute("""
                SELECT COUNT(*) FROM kb_coverage
                WHERE coverage_status IN ('missing', 'partial')
                  AND theme NOT IN (
                    SELECT article_topic FROM generated_articles WHERE format = 'faq'
                  )
            """).fetchone()[0]

            if missing_no_articles > 0:
                print(f"   ISSUE: {missing_no_articles} missing/partial gaps have no FAQ articles")
                print("   Possible causes:")
                print("     1. Generator not run yet: 'python faq_service.py generate'")
                print("     2. Generator skipped due to:")
                print("        - No resolved tickets with resolution_summary")
                print("        - Semantic/structural duplicates detected")
                print("        - Gemini generation errors")
            else:
                print("   Generator appears to have processed all gaps")

        # Check for recent generation errors
        errors = conn.execute("""
            SELECT COUNT(*) FROM generated_articles
            WHERE format = 'faq' AND status = 'draft' AND generated_at > datetime('now', '-1 day')
        """).fetchone()[0]

        if errors >= 0:
            print(f"   Recent FAQ drafts (last 24h): {errors}")

        # ─────────────────────────────────────────────────────────────────
        # 7. Summary Report
        # ─────────────────────────────────────────────────────────────────
        print("\n7. SUMMARY:")
        print("   " + "─" * 80)

        total_faqs = conn.execute(
            "SELECT COUNT(*) FROM generated_articles WHERE format = 'faq'"
        ).fetchone()[0]

        published_faqs = conn.execute(
            "SELECT COUNT(*) FROM generated_articles WHERE format = 'faq' AND status = 'published'"
        ).fetchone()[0]

        total_themes = conn.execute(
            "SELECT COUNT(*) FROM kb_coverage"
        ).fetchone()[0]

        missing_themes = conn.execute(
            "SELECT COUNT(*) FROM kb_coverage WHERE coverage_status IN ('missing', 'partial')"
        ).fetchone()[0]

        print(f"   Total FAQ articles: {total_faqs}")
        print(f"   Published: {published_faqs}")
        print(f"   Drafts: {total_faqs - published_faqs}")
        print()
        print(f"   Total identified themes: {total_themes}")
        print(f"   Missing/Partial coverage: {missing_themes}")
        print(f"   Generation rate: {total_faqs}/{missing_themes if missing_themes > 0 else 'N/A'}")

    print("\n" + "=" * 84 + "\n")

if __name__ == "__main__":
    main()
