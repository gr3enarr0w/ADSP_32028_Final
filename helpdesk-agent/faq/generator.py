"""FAQ entry generation — shorter format than full KB articles."""

import json
import logging
import time
from datetime import datetime, timezone
from html import escape

from config import GEMINI_MODEL_GENERATION, apply_cloud_terminology
from google.genai import types as genai_types
from db import get_db_conn
from faq.dedup import (
    compute_embedding,
    compute_fingerprint,
    is_duplicate,
    is_duplicate_of_sections,
    is_semantic_duplicate,
)
from faq.issue_checker import check_linked_issues

log = logging.getLogger(__name__)

FAQ_PROMPT = """You are writing a FAQ entry for end users migrating from Atlassian Data Center to Cloud.

Topic: {theme}
Gap detail: {gap_detail}

SOURCE TICKETS (real agent resolutions — use these as the primary basis for your answer):
{ticket_data}

LINKED ISSUES:
{linked_context}

EXISTING DOCUMENTATION CONTEXT:
{doc_context}

Write a single FAQ entry in this exact JSON format:
{{
  "topic": "{theme}",
  "question": "<clear question a user would ask>",
  "answer": "<direct, concise answer — 2-4 sentences max>",
  "steps": ["<step 1>", "<step 2>", ...],
  "known_limitations": "<any Cloud-specific limitations or differences, or empty string>"
}}

Rules:
- Use direct "You" addressing
- Pull actual resolution steps from the ticket comments — don't make them up
- If there are open bugs, note them under known_limitations
- If there are open RFEs/feature requests, mention as "planned improvement"
- Be specific to the Atlassian Cloud migration context
- Steps should be actionable, one action per step
- TERMINOLOGY: In Jira Cloud, "projects" have been renamed to "spaces". Always use "space" / "spaces" instead of "project" / "projects" when referring to Jira Cloud. Exception: keep "project key" and "project category" as-is (these are API/technical terms).
- Return ONLY the JSON object, no markdown fences
"""


def generate_faq_entry(gap: dict, sources: dict, client=None) -> dict | None:
    """Generate a single FAQ entry from gap analysis + source material.

    Args:
        gap: Dict from kb_coverage (theme, category, sample_issue_types, etc.)
        sources: Dict from sources.gather_all_sources()
        client: Optional pre-initialized Gemini client

    Returns:
        FAQ entry dict or None on error.
    """
    # Get resolved ticket data for this theme using multiple strategies:
    # 1. Exact issue_type match
    # 2. Category match (sample_issue_types often contain categories)
    # 3. Keyword match against theme name in ticket summaries
    issue_types = json.loads(gap.get("sample_issue_types", "[]"))
    ticket_data_parts = []
    seen_keys = set()

    with get_db_conn() as conn:
        def _add_tickets(rows):
            for t in rows:
                t = dict(t)
                if t["ticket_key"] in seen_keys:
                    continue
                seen_keys.add(t["ticket_key"])
                comments = conn.execute("""
                    SELECT author_name, body, is_public FROM ticket_comments
                    WHERE ticket_key = ? ORDER BY created_at LIMIT 10
                """, (t["ticket_key"],)).fetchall()

                comment_text = "\n".join(
                    f"  {'[Public]' if c['is_public'] else '[Internal]'} {c['author_name']}: "
                    f"{(c['body'] or '')[:300]}"
                    for c in comments
                )

                ticket_data_parts.append(
                    f"Key: {t['ticket_key']}\nSummary: {t['summary']}\n"
                    f"Resolution: {t['resolution'] or 'None'}\n"
                    f"Resolution Summary: {t['resolution_summary']}\n"
                    f"Comments:\n{comment_text}\n---"
                )

        # Strategy 1: exact issue_type match
        if issue_types:
            placeholders = ",".join(["?"] * len(issue_types))
            _add_tickets(conn.execute(f"""
                SELECT t.ticket_key, t.summary, t.description, t.resolution,
                       c.resolution_summary, c.question_type
                FROM tickets t
                JOIN ticket_classifications c ON t.ticket_key = c.ticket_key
                WHERE c.issue_type IN ({placeholders}) AND c.has_resolution = 1
                ORDER BY c.confidence DESC LIMIT 5
            """, issue_types).fetchall())

        # Strategy 2: category match (sample_issue_types often ARE categories)
        if len(seen_keys) < 5 and issue_types:
            placeholders = ",".join(["?"] * len(issue_types))
            _add_tickets(conn.execute(f"""
                SELECT t.ticket_key, t.summary, t.description, t.resolution,
                       c.resolution_summary, c.question_type
                FROM tickets t
                JOIN ticket_classifications c ON t.ticket_key = c.ticket_key
                WHERE c.category IN ({placeholders}) AND c.has_resolution = 1
                ORDER BY c.confidence DESC LIMIT 5
            """, issue_types).fetchall())

        # Strategy 3: keyword match from theme name against issue_type or summary
        if len(seen_keys) < 5:
            # Extract meaningful keywords from the theme name
            theme_words = [w.lower() for w in gap["theme"].replace("&", "").split()
                           if len(w) > 3 and w.lower() not in
                           ("after", "from", "with", "that", "this", "been", "have",
                            "does", "their", "they", "into", "when", "your", "some")]
            for word in theme_words[:3]:
                if len(seen_keys) >= 5:
                    break
                _add_tickets(conn.execute("""
                    SELECT t.ticket_key, t.summary, t.description, t.resolution,
                           c.resolution_summary, c.question_type
                    FROM tickets t
                    JOIN ticket_classifications c ON t.ticket_key = c.ticket_key
                    WHERE (c.issue_type LIKE ? OR t.summary LIKE ?) AND c.has_resolution = 1
                    ORDER BY c.confidence DESC LIMIT 3
                """, (f"%{word}%", f"%{word}%")).fetchall())

    if not ticket_data_parts:
        log.warning("No source tickets for theme: %s (tried issue_types: %s, "
                     "category match, and keyword match)",
                     gap["theme"], issue_types)
        return None

    # Check linked issues
    linked = check_linked_issues(gap["theme"])
    linked_parts = []
    if linked["open_bugs"]:
        linked_parts.append("OPEN BUGS (fix in progress):")
        for b in linked["open_bugs"]:
            linked_parts.append(f"  - {b['issue_key']}: {b['summary']} [{b['status']}]")
    if linked["open_rfes"]:
        linked_parts.append("PLANNED IMPROVEMENTS:")
        for r in linked["open_rfes"]:
            linked_parts.append(f"  - {r['issue_key']}: {r['summary']} [{r['status']}]")
    if linked["resolved_bugs"]:
        linked_parts.append("RESOLVED BUGS (include fix in answer):")
        for b in linked["resolved_bugs"]:
            linked_parts.append(f"  - {b['issue_key']}: {b['summary']} [{b['resolution']}]")
    linked_context = "\n".join(linked_parts) or "(no linked issues)"

    # Build doc context from sources
    doc_context_parts = []
    if sources.get("google_doc"):
        doc_context_parts.append(f"FAQ Document:\n{sources['google_doc'][:2000]}")
    if gap.get("covered_by"):
        doc_context_parts.append(f"Existing coverage: {gap['covered_by']}")
    if gap.get("doc_section"):
        doc_context_parts.append(f"Relevant section: {gap['doc_section']}")
    doc_context = "\n".join(doc_context_parts) or "(no existing documentation)"

    prompt = FAQ_PROMPT.format(
        theme=gap["theme"],
        gap_detail=gap.get("gap_detail", ""),
        ticket_data="\n\n".join(ticket_data_parts),
        linked_context=linked_context,
        doc_context=doc_context[:3000],
    )

    if client is None:
        from core.genai import get_genai_client as _get_genai_client
        client = _get_genai_client()

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL_GENERATION,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                http_options=genai_types.HttpOptions(timeout=840_000),  # 14 min — 2× gemini-2.5-pro max (5-7 min)
            ),
        )
        from utils.gemini import parse_json_response
        entry = parse_json_response(response.text)
        # Apply Cloud terminology to all text fields
        for key in ("topic", "question", "answer", "known_limitations"):
            if entry.get(key):
                entry[key] = apply_cloud_terminology(entry[key])
        if entry.get("steps"):
            entry["steps"] = [apply_cloud_terminology(s) for s in entry["steps"]]
        log.info("Generated FAQ entry: %s", entry.get("topic", gap["theme"]))
        return entry
    except Exception as e:
        log.error("Failed to generate FAQ entry for %s: %s", gap["theme"], e)
        return None


def generate_all_faq_entries(sources: dict, theme_filter: str | None = None,
                             client=None) -> tuple[int, int]:
    """Generate FAQ entries for all gaps (or a specific theme).

    Stores results in generated_articles with format='faq'.

    Returns:
        Tuple of (generated_count, error_count).
    """
    with get_db_conn() as conn:
        if theme_filter:
            gaps = conn.execute("""
                SELECT * FROM kb_coverage
                WHERE theme = ? AND coverage_status IN ('missing', 'partial')
            """, (theme_filter,)).fetchall()
        else:
            gaps = conn.execute("""
                SELECT * FROM kb_coverage
                WHERE coverage_status IN ('missing', 'partial')
                ORDER BY
                    CASE article_priority WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END,
                    ticket_count DESC
            """).fetchall()

        if not gaps:
            log.info("No FAQ gaps to generate entries for")
            return 0, 0

        # Skip themes that already have FAQ entries (exact article_topic match).
        # Also pre-load existing article topics for semantic title dedup below —
        # this catches the case where the LLM generates a semantically equivalent
        # theme name across pipeline cycles (e.g. "Notification Behavior in Cloud"
        # vs "Notification Management & Issues") that slips past the exact match.
        existing_rows = conn.execute(
            "SELECT article_topic FROM generated_articles WHERE format = 'faq'"
        ).fetchall()
        existing = set(r[0] for r in existing_rows)
        existing_topics: list[str] = [r[0] for r in existing_rows]
        gaps = [g for g in gaps if g["theme"] not in existing]

    if not gaps:
        log.info("All FAQ gaps already have generated entries")
        return 0, 0

    if client is None:
        from core.genai import get_genai_client as _get_genai_client
        client = _get_genai_client()

    generated = 0
    errors = 0

    with get_db_conn() as conn:
        for gap in gaps:
            gap = dict(gap)
            log.info("Generating FAQ entry for: %s", gap["theme"])

            entry = generate_faq_entry(gap, sources, client)
            if not entry:
                errors += 1
                continue

            # Build HTML version for storage
            body_html = (
                f"<h2>{escape(entry.get('question', ''))}</h2>\n"
                f"<p>{escape(entry.get('answer', ''))}</p>"
            )
            if entry.get("steps"):
                steps_html = "".join(f"<li>{escape(s)}</li>" for s in entry["steps"])
                body_html += f"\n<h3>Steps</h3>\n<ol>{steps_html}</ol>"
            if entry.get("known_limitations"):
                body_html += f"\n<h3>Known Limitations</h3>\n<p>{escape(entry['known_limitations'])}</p>"

            # Title-level semantic dedup: compare the new theme name against all
            # existing article_topic strings.  This catches LLM-generated synonym
            # themes ("Notification Behavior in Cloud" vs "Notification Management
            # & Issues") that share body cosine ~0.90 but are obviously the same
            # topic.  Threshold 0.82 is intentionally lower than the body-level
            # threshold because title-only pairs are shorter and more volatile.
            candidates = [t for t in existing_topics if t != gap["theme"]]
            title_dup, title_idx, title_sim = is_duplicate_of_sections(
                gap["theme"],
                candidates,
                threshold=0.82,
            )
            if title_dup and title_idx is not None:
                log.warning(
                    "Skipping title-level semantic duplicate FAQ for '%s' "
                    "(cosine=%.3f vs '%s')",
                    gap["theme"], title_sim, candidates[title_idx],
                )
                errors += 1
                continue

            dup, dup_topic = is_duplicate(body_html, exclude_topic=gap["theme"])
            if dup:
                log.warning("Skipping near-duplicate FAQ for '%s' (matches '%s')",
                            gap["theme"], dup_topic)
                errors += 1
                continue

            sem_dup, sem_topic, sem_sim = is_semantic_duplicate(
                body_html, exclude_topic=gap["theme"],
            )
            if sem_dup:
                log.warning(
                    "Skipping semantic-duplicate FAQ for '%s' "
                    "(cosine=%.3f vs '%s')",
                    gap["theme"], sem_sim, sem_topic,
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
                VALUES (?, ?, ?, 'faq', ?, ?, ?, ?, 'draft', ?)
                ON CONFLICT (article_topic) DO UPDATE SET
                    title=EXCLUDED.title, body_html=EXCLUDED.body_html,
                    format=EXCLUDED.format,
                    source_ticket_keys=EXCLUDED.source_ticket_keys,
                    reference_urls=EXCLUDED.reference_urls,
                    structural_fingerprint=EXCLUDED.structural_fingerprint,
                    semantic_embedding=EXCLUDED.semantic_embedding,
                    status=EXCLUDED.status, generated_at=EXCLUDED.generated_at
            """, (
                gap["theme"],
                entry.get("topic", gap["theme"]),
                body_html,
                json.dumps([]),
                json.dumps([]),
                fingerprint,
                embedding,
                datetime.now(timezone.utc).isoformat(),
            ))
            conn.commit()

            generated += 1
            log.info("  Generated FAQ: %s", entry.get("topic", gap["theme"]))
            time.sleep(0.5)  # Rate limit for Gemini API

    log.info("FAQ generation complete: %d generated, %d errors", generated, errors)
    return generated, errors
