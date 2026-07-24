"""KB article generation using Gemini AI."""

import re
import json
import time
import logging
from datetime import datetime, timezone

from config import GEMINI_MODEL_GENERATION
from db import get_db
from faq.dedup import compute_embedding, compute_fingerprint, is_duplicate, is_semantic_duplicate

log = logging.getLogger(__name__)

ARTICLE_PROMPTS = {
    "how-to": """You are a technical writer for an IT help desk knowledge base.
Write a clear HOW-TO article based on the support tickets below.

Format:
1. Title: "How to [action]" format
2. Who this is for: Target audience
3. Prerequisites: What you need before starting
4. Steps: Numbered, one action per step
5. Expected result: What success looks like
6. Troubleshooting: Common issues and fixes

Requirements:
- Use direct "You" addressing, no filler paragraphs
- Use specific examples from tickets, not generic ones
- Mark screenshot placeholders with [SCREENSHOT: description]
- Format in Confluence storage format (HTML)

Tickets:
{ticket_data}

{reference_section}""",

    "troubleshooting": """You are a technical writer for an IT help desk knowledge base.
Write a TROUBLESHOOTING article based on the support tickets below.

Format:
1. Title: Descriptive problem statement
2. Symptoms: What the user sees
3. Root Cause: Why this happens
4. Resolution: Step-by-step fix
5. Prevention: How to avoid it

Requirements:
- Start with symptoms users actually reported
- Use actual resolution steps from agent comments
- Format in Confluence storage format (HTML)

Tickets:
{ticket_data}

{reference_section}""",

    "explainer": """You are a technical writer for an IT help desk knowledge base.
Write an EXPLAINER article based on the support tickets below.

Format:
1. Title: Clear topic statement
2. Context: Why this matters
3. Key Concepts: Terms and definitions
4. How It Works: Cloud vs Data Center
5. FAQ: Common questions from tickets

Requirements:
- Explain the "why" behind changes
- Compare DC vs Cloud behavior where relevant
- Format in Confluence storage format (HTML)

Tickets:
{ticket_data}

{reference_section}""",

    "reference": """You are a technical writer for an IT help desk knowledge base.
Write a REFERENCE article based on the support tickets below.

Format:
1. Title: Feature/topic name
2. Overview: What this feature does
3. Configuration: Settings and options
4. Examples: Common use cases
5. Known Limitations: Cloud-specific constraints

Requirements:
- Be concise and scannable, use tables for comparisons
- Format in Confluence storage format (HTML)

Tickets:
{ticket_data}

{reference_section}""",
}


def determine_article_format(question_types):
    if not question_types:
        return "how-to"
    type_counts = {}
    for qt in question_types:
        type_counts[qt] = type_counts.get(qt, 0) + 1
    most_common = max(type_counts, key=type_counts.get)
    return {"how-to": "how-to", "troubleshooting": "troubleshooting",
            "access-request": "how-to", "configuration": "reference",
            "bug-report": "troubleshooting"}.get(most_common, "how-to")


def gather_article_sources(conn, theme):
    """Gather top resolved tickets for an article theme."""
    sample_types = json.loads(theme["sample_issue_types"]) if theme["sample_issue_types"] else []
    if not sample_types:
        return [], [], []

    placeholders = ",".join(["?"] * len(sample_types))
    tickets = conn.execute(f"""
        SELECT t.ticket_key, t.summary, t.description, t.status, t.resolution,
               c.issue_type, c.question_type, c.resolution_summary, c.confidence
        FROM tickets t
        JOIN ticket_classifications c ON t.ticket_key = c.ticket_key
        WHERE c.issue_type IN ({placeholders}) AND c.has_resolution = 1
        ORDER BY c.confidence DESC LIMIT 5
    """, sample_types).fetchall()

    if not tickets:
        tickets = conn.execute(f"""
            SELECT t.ticket_key, t.summary, t.description, t.status, t.resolution,
                   c.issue_type, c.question_type, c.resolution_summary, c.confidence
            FROM tickets t
            JOIN ticket_classifications c ON t.ticket_key = c.ticket_key
            WHERE c.issue_type IN ({placeholders})
            ORDER BY c.confidence DESC LIMIT 5
        """, sample_types).fetchall()

    ticket_data_parts = []
    question_types = []
    ticket_keys = []

    for t in tickets:
        t = dict(t)
        ticket_keys.append(t["ticket_key"])
        question_types.append(t["question_type"])

        comments = conn.execute(
            "SELECT author_name, body, is_public FROM ticket_comments WHERE ticket_key = ? ORDER BY created_at",
            (t["ticket_key"],)
        ).fetchall()

        comment_text = ""
        for c in comments:
            prefix = "[Public]" if c["is_public"] else "[Internal]"
            comment_text += f"  {prefix} {c['author_name']}: {(c['body'] or '')[:300]}\n"

        ticket_data_parts.append(
            f"Key: {t['ticket_key']}\nSummary: {t['summary']}\n"
            f"Description: {(t['description'] or '')[:500]}\n"
            f"Status: {t['status']} | Resolution: {t['resolution'] or 'None'}\n"
            f"Resolution Summary: {t['resolution_summary']}\nComments:\n{comment_text}\n---"
        )

    return ticket_data_parts, question_types, ticket_keys


def generate_articles(theme_filter=None, client=None):
    """Generate KB article drafts for documentation gaps.

    Args:
        theme_filter: Optional specific theme name to generate for.
        client: Optional pre-initialized Gemini client.

    Returns:
        Tuple of (generated_count, error_count).
    """
    conn = get_db()

    if theme_filter:
        themes = conn.execute("""
            SELECT * FROM kb_coverage
            WHERE theme = ? AND coverage_status IN ('missing', 'partial')
        """, (theme_filter,)).fetchall()
    else:
        themes = conn.execute("""
            SELECT * FROM kb_coverage
            WHERE coverage_status IN ('missing', 'partial')
            ORDER BY
                CASE article_priority WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END,
                ticket_count DESC
        """).fetchall()

    if not themes:
        log.info("No documentation gaps found")
        conn.close()
        return 0, 0

    existing = set(r[0] for r in conn.execute(
        "SELECT article_topic FROM generated_articles"
    ).fetchall())
    themes = [t for t in themes if t["theme"] not in existing]

    if not themes:
        log.info("All gap themes already have generated articles")
        conn.close()
        return 0, 0

    if client is None:
        from core.genai import get_genai_client as _get_genai_client
        client = _get_genai_client()

    log.info("Generating articles for %d documentation gaps...", len(themes))
    generated = 0
    errors = 0

    for theme in themes:
        theme = dict(theme)
        log.info("  Generating article for: %s", theme["theme"])

        ticket_data_parts, question_types, ticket_keys = gather_article_sources(conn, theme)
        if not ticket_data_parts:
            log.warning("    No source tickets found for theme: %s", theme["theme"])
            continue

        article_format = determine_article_format(question_types)
        prompt_template = ARTICLE_PROMPTS.get(article_format, ARTICLE_PROMPTS["how-to"])

        reference_section = ""
        if theme.get("covered_by"):
            reference_section = f"Relevant existing documentation: {theme['covered_by']}"
            if theme.get("doc_section"):
                reference_section += f" — Section: {theme['doc_section']}"
            if theme.get("gap_detail"):
                reference_section += f"\nGap to fill: {theme['gap_detail']}"

        prompt = prompt_template.format(
            ticket_data="\n\n".join(ticket_data_parts),
            reference_section=reference_section,
        )

        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL_GENERATION, contents=prompt,
            )
            body_html = response.text.strip()

            title_match = re.search(r'<h[12][^>]*>(.*?)</h[12]>', body_html)
            title = re.sub(r'<[^>]+>', '', title_match.group(1)).strip() if title_match else theme["theme"]

            dup, dup_topic = is_duplicate(body_html, exclude_topic=theme["theme"])
            if dup:
                log.warning("Skipping near-duplicate article for '%s' (matches '%s')",
                            theme["theme"], dup_topic)
                errors += 1
                continue

            sem_dup, sem_topic, sem_sim = is_semantic_duplicate(
                body_html, exclude_topic=theme["theme"],
            )
            if sem_dup:
                log.warning(
                    "Skipping semantic-duplicate article for '%s' (cosine=%.3f vs '%s')",
                    theme["theme"], sem_sim, sem_topic,
                )
                errors += 1
                continue

            fingerprint = compute_fingerprint(body_html)
            embedding = json.dumps(compute_embedding(body_html))
            conn.execute("""
                INSERT INTO generated_articles
                    (article_topic, title, body_html, format, source_ticket_keys,
                     reference_urls, structural_fingerprint, semantic_embedding,
                     status, generated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'draft', ?)
                ON CONFLICT (article_topic) DO UPDATE SET
                    title = EXCLUDED.title,
                    body_html = EXCLUDED.body_html,
                    format = EXCLUDED.format,
                    source_ticket_keys = EXCLUDED.source_ticket_keys,
                    reference_urls = EXCLUDED.reference_urls,
                    structural_fingerprint = EXCLUDED.structural_fingerprint,
                    semantic_embedding = EXCLUDED.semantic_embedding,
                    status = EXCLUDED.status,
                    generated_at = EXCLUDED.generated_at
            """, (
                theme["theme"], title, body_html, article_format,
                json.dumps(ticket_keys), json.dumps([]),
                fingerprint, embedding,
                datetime.now(timezone.utc).isoformat(),
            ))
            conn.commit()
            generated += 1
            log.info("    Generated: %s (%s format, %d source tickets)",
                     title, article_format, len(ticket_keys))
        except Exception as e:
            errors += 1
            log.warning("    Error generating article for %s: %s", theme["theme"], e)
            time.sleep(1)

    conn.close()
    return generated, errors
