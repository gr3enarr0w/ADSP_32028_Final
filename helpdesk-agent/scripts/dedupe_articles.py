"""Deduplicate generated_articles table by cosine similarity on stored embeddings.

Compares all existing FAQ articles pairwise, identifies near-duplicates, and
marks the lower-quality one of each pair as 'archived'. Resets confluence_page_id
on archived articles so the publish loop skips them on next run.

Usage:
    # Dry run — shows what would be archived, no changes
    python -m scripts.dedupe_articles --dry-run

    # Apply with default threshold (0.90)
    python -m scripts.dedupe_articles

    # Apply with custom threshold
    python -m scripts.dedupe_articles --threshold 0.88

Quality tiebreaker (when two articles are near-duplicates, Gemini judges which to keep):
    Gemini Flash Lite reads both articles and picks the one with more specific, actionable
    content. Falls back to lower article ID if Gemini call fails.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from db import get_db_conn
from core.genai import get_genai_client
from google.genai import types as genai_types

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


_JUDGE_PROMPT = """You are evaluating two FAQ articles about Atlassian Cloud helpdesk support. Both cover the same topic. Pick the one that is more useful to a support agent — more specific, more actionable, contains real technical details rather than vague guidance.

ARTICLE A (id={id_a}):
Title: {title_a}
{body_a}

ARTICLE B (id={id_b}):
Title: {title_b}
{body_b}

Reply with ONLY the letter A or B — the article that is higher quality and more useful."""


def _gemini_pick_better(art_a: dict, art_b: dict) -> dict:
    """Ask Gemini Pro to pick the higher-quality article. Returns the one to keep."""
    import re
    try:
        client = get_genai_client()
        prompt = _JUDGE_PROMPT.format(
            id_a=art_a["id"], title_a=art_a["title"],
            body_a=art_a["body_text"][:2000],
            id_b=art_b["id"], title_b=art_b["title"],
            body_b=art_b["body_text"][:2000],
        )
        resp = client.models.generate_content(
            model="gemini-2.5-pro-preview",
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                http_options=genai_types.HttpOptions(timeout=60_000),
            ),
        )
        answer = (resp.text or "").strip().upper()
        match = re.search(r"\b([AB])\b", answer)
        if match:
            choice = match.group(1)
            keep = art_a if choice == "A" else art_b
            drop = art_b if choice == "A" else art_a
            log.debug("Gemini chose %s: [%d] '%s'", choice, keep["id"], keep["title"][:50])
            return keep, drop
    except Exception as e:
        log.warning("Gemini judge failed (%s) — falling back to lower ID as keep", e)
    # Fallback: keep lower ID (older, more stable)
    keep = art_a if art_a["id"] < art_b["id"] else art_b
    drop = art_b if art_a["id"] < art_b["id"] else art_a
    return keep, drop


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def run_dedup(threshold: float = 0.90, dry_run: bool = True) -> None:
    log.info("Loading FAQ articles with embeddings (threshold=%.2f, dry_run=%s)", threshold, dry_run)

    with get_db_conn() as conn:
        rows = conn.execute("""
            SELECT id, title, article_topic, body_html, semantic_embedding, confluence_page_id, status
            FROM generated_articles
            WHERE format = 'faq'
            AND semantic_embedding IS NOT NULL
            ORDER BY id
        """).fetchall()

    articles = []
    for r in rows:
        emb = json.loads(r["semantic_embedding"]) if isinstance(r["semantic_embedding"], str) else r["semantic_embedding"]
        import re as _re
        body_text = _re.sub(r"<[^>]+>", " ", r["body_html"] or "")
        articles.append({
            "id": r["id"],
            "title": r["title"],
            "topic": r["article_topic"],
            "body_text": body_text,
            "embedding": emb,
            "confluence_page_id": r["confluence_page_id"],
            "status": r["status"],
        })

    log.info("Loaded %d articles", len(articles))

    # Find duplicate pairs
    to_archive: set[int] = set()
    duplicate_pairs: list[tuple[dict, dict, float]] = []

    for i in range(len(articles)):
        if articles[i]["id"] in to_archive:
            continue
        for j in range(i + 1, len(articles)):
            if articles[j]["id"] in to_archive:
                continue
            sim = _cosine_similarity(articles[i]["embedding"], articles[j]["embedding"])
            if sim >= threshold:
                # Tiebreaker: Gemini Pro judges which article is higher quality
                keep, drop = _gemini_pick_better(articles[i], articles[j])
                to_archive.add(drop["id"])
                duplicate_pairs.append((keep, drop, sim))
                log.info(
                    "DUPLICATE (%.3f): KEEP [%d] '%s' | ARCHIVE [%d] '%s'",
                    sim, keep["id"], keep["title"][:50], drop["id"], drop["title"][:50],
                )

    log.info("\n=== SUMMARY ===")
    log.info("Total articles: %d", len(articles))
    log.info("Duplicate pairs found: %d", len(duplicate_pairs))
    log.info("Articles to archive: %d", len(to_archive))
    log.info("Articles to keep: %d", len(articles) - len(to_archive))

    if not to_archive:
        log.info("No duplicates found at threshold %.2f. Nothing to do.", threshold)
        return

    if dry_run:
        log.info("\n--- DRY RUN --- No changes made. Run without --dry-run to apply.")
        log.info("Articles that WOULD be archived:")
        for art in articles:
            if art["id"] in to_archive:
                log.info("  [%d] %s", art["id"], art["title"])
        return

    # Apply: mark duplicates as archived, clear confluence_page_id so publish loop skips them
    log.info("\nApplying changes...")
    with get_db_conn() as conn:
        for article_id in to_archive:
            conn.execute("""
                UPDATE generated_articles
                SET status = 'archived',
                    confluence_page_id = NULL,
                    confluence_url = NULL
                WHERE id = %s
            """, (article_id,))
        log.info("Archived %d articles. Confluence page IDs cleared.", len(to_archive))
        log.info("The publish loop will now re-publish only the %d surviving articles on next FAQ phase run.", len(articles) - len(to_archive))


def main():
    parser = argparse.ArgumentParser(description="Deduplicate FAQ articles by cosine similarity")
    parser.add_argument("--threshold", type=float, default=0.90, help="Cosine similarity threshold (default: 0.90)")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Show what would be archived without making changes")
    args = parser.parse_args()
    run_dedup(threshold=args.threshold, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
