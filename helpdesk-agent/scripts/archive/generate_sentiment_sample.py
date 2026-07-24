"""Generate a 50-ticket stratified sentiment sample for human review — ANTSE-322.

Loads 10 tickets from each of the 5 highest-volume categories (Access,
Configuration, UI/UX, Integration, Permissions), scores them with the
cardiffnlp/twitter-roberta-base-sentiment-latest model, and writes the
results to analysis/sentiment_sample.json.

The JSON file is the AC artefact for human labelling:
  - Reviewer reads predicted_label / predicted_score
  - Annotates a ``human_label`` column (NEGATIVE / NEUTRAL / POSITIVE)
  - Disagreements feed back into threshold calibration

Usage:
    cd /path/to/ai-helpdesk-agent
    python scripts/generate_sentiment_sample.py
"""

import json
import logging
import os
import sys
from pathlib import Path

# Make sure the project root is on sys.path so db / plugins are importable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TARGET_CATEGORIES = ["Access", "Configuration", "UI/UX", "Integration", "Permissions"]
TICKETS_PER_CATEGORY = 10
OUTPUT_PATH = PROJECT_ROOT / "analysis" / "sentiment_sample.json"
DESCRIPTION_PREVIEW_CHARS = 500


def _load_tickets_from_db(categories: list[str], per_category: int) -> list[dict]:
    """Query the DB for tickets stratified by category.

    Falls back gracefully: if a category has fewer than ``per_category``
    tickets, takes all available.  If ticket_classifications is empty or the
    DB doesn't exist, raises with a clear message.

    Args:
        categories: List of category names to sample from.
        per_category: Maximum tickets per category.

    Returns:
        List of row dicts with keys: ticket_key, summary, description, category.
    """
    from db import get_db_conn

    rows: list[dict] = []
    with get_db_conn() as conn:
        for cat in categories:
            # Case-insensitive LIKE match so "UI/UX" matches "ui/ux" etc.
            batch = conn.execute(
                """
                SELECT
                    t.ticket_key,
                    t.summary,
                    t.description,
                    tc.category
                FROM tickets t
                JOIN ticket_classifications tc ON t.ticket_key = tc.ticket_key
                WHERE tc.category LIKE ?
                ORDER BY t.created_at DESC
                LIMIT ?
                """,
                (cat, per_category),
            ).fetchall()

            fetched = len(batch)
            log.info("  %-15s  %d ticket(s) fetched", cat, fetched)
            rows.extend([dict(r) for r in batch])

    return rows


def _build_score_text(row: dict) -> str:
    """Combine summary and description into a single scoring string.

    Args:
        row: DB row dict with summary and description keys.

    Returns:
        Concatenated text suitable for the sentiment model.
    """
    parts = []
    if row.get("summary"):
        parts.append(row["summary"].strip())
    if row.get("description"):
        parts.append(row["description"].strip())
    return " ".join(parts)


def _print_summary_table(records: list[dict]) -> None:
    """Print a formatted summary table to stdout.

    Args:
        records: List of scored sample dicts.
    """
    header = f"{'Ticket':<18} {'Category':<16} {'Label':<10} {'Intensity':>9}  Summary"
    print("\n" + "=" * 90)
    print(header)
    print("-" * 90)

    for r in records:
        summary_preview = (r["summary"] or "")[:45].replace("\n", " ")
        print(
            f"{r['ticket_key']:<18} "
            f"{r['category']:<16} "
            f"{r['predicted_label']:<10} "
            f"{r['predicted_score']:>9.4f}  "
            f"{summary_preview}"
        )

    # Distribution counts
    from collections import Counter
    label_counts = Counter(r["predicted_label"] for r in records)
    print("=" * 90)
    print(f"\nLabel distribution: {dict(label_counts)}")

    high_intensity = [r for r in records if r["predicted_score"] >= 0.6]
    print(f"High-intensity tickets (intensity >= 0.6): {len(high_intensity)}/{len(records)}")

    by_cat = Counter(r["category"] for r in records)
    print(f"Tickets per category: {dict(by_cat)}\n")


def main() -> None:
    """Entry point: load, score, save, and print."""
    log.info("Loading tickets from DB (stratified sample)…")
    tickets = _load_tickets_from_db(TARGET_CATEGORIES, TICKETS_PER_CATEGORY)

    if not tickets:
        log.error(
            "No tickets found.  Ensure ticket_classifications is populated "
            "and that jsm_data.db is reachable."
        )
        sys.exit(1)

    log.info("Loaded %d tickets total; scoring with sentiment model…", len(tickets))

    from plugins.feedback.sentiment import score_ticket

    records: list[dict] = []
    for i, t in enumerate(tickets, 1):
        text = _build_score_text(t)
        result = score_ticket(text)

        record = {
            "ticket_key": t["ticket_key"],
            "category": t.get("category", ""),
            "summary": t.get("summary", ""),
            "description_preview": (t.get("description") or "")[:DESCRIPTION_PREVIEW_CHARS],
            "predicted_label": result["label"],
            "predicted_score": result["intensity"],   # intensity = NEGATIVE score
            "dominant_score": result["score"],
            # Placeholder for human annotator
            "human_label": None,
        }
        records.append(record)

        if i % 10 == 0:
            log.info("  Scored %d/%d…", i, len(tickets))

    log.info("Saving %d records to %s", len(records), OUTPUT_PATH)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False)

    log.info("Saved: %s", OUTPUT_PATH)

    _print_summary_table(records)
    print(f"Output file: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
