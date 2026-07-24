"""Kaggle IT Support Tickets loader — ticket-to-answer retrieval pairs.

Loads IT support tickets from a local CSV (user downloads from Kaggle) and
creates query-document pairs mapping ticket subjects to resolutions.

The Kaggle dataset has no duplicate labels, so this is used for retrieval
evaluation (ticket → best answer), not dedup evaluation.

Download: https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset
Place CSV at: eval/data/customer_support_tickets.csv

Usage:
    python -m eval.loaders.kaggle_it
    python -m eval.loaders.kaggle_it --limit 30
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_CSV = CACHE_DIR / "customer_support_tickets.csv"
DEFAULT_POSITIVE = 20
DEFAULT_NEGATIVE = 10

SUBJECT_COL = "Ticket Subject"
DESC_COL = "Ticket Description"
RESOLUTION_COL = "Resolution"
TYPE_COL = "Ticket Type"


def load_pairs(
    csv_path: Path | None = None,
    pos_limit: int = DEFAULT_POSITIVE,
    neg_limit: int = DEFAULT_NEGATIVE,
    seed: int = 42,
) -> list[dict]:
    csv_path = csv_path or DEFAULT_CSV
    if not csv_path.exists():
        print(
            f"CSV not found at {csv_path}\n"
            f"Download from: https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset\n"
            f"Place as: {DEFAULT_CSV}",
            file=sys.stderr,
        )
        return []

    rng = random.Random(seed)

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r.get(RESOLUTION_COL, "").strip()]

    if not rows:
        print("No rows with resolutions found", file=sys.stderr)
        return []

    rng.shuffle(rows)

    pairs = []
    pair_id = 1

    for row in rows[:pos_limit]:
        subject = row.get(SUBJECT_COL, "").strip()
        resolution = row.get(RESOLUTION_COL, "").strip()
        ticket_type = row.get(TYPE_COL, "IT").strip()
        if not subject or not resolution:
            continue
        pairs.append({
            "id": pair_id,
            "query": subject[:500],
            "document": resolution[:500],
            "relevance": 5,
            "category": f"IT-{ticket_type}",
            "source": "kaggle_it",
        })
        pair_id += 1

    used_indices = set(range(pos_limit))
    neg_count = 0
    for i in range(min(pos_limit, len(rows))):
        if neg_count >= neg_limit:
            break
        subject = rows[i].get(SUBJECT_COL, "").strip()
        wrong_idx = rng.choice([j for j in range(len(rows)) if j not in used_indices and j != i])
        wrong_res = rows[wrong_idx].get(RESOLUTION_COL, "").strip()
        if not subject or not wrong_res:
            continue
        pairs.append({
            "id": pair_id,
            "query": subject[:500],
            "document": wrong_res[:500],
            "relevance": 1,
            "category": f"IT-Neg",
            "source": "kaggle_it",
        })
        pair_id += 1
        neg_count += 1

    print(f"Loaded {len(pairs)} pairs from Kaggle IT ({pos_limit} pos + {neg_count} neg)")
    return pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load Kaggle IT support ticket pairs")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--limit", type=int, default=DEFAULT_POSITIVE)
    parser.add_argument("--neg", type=int, default=DEFAULT_NEGATIVE)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    pairs = load_pairs(csv_path=args.csv, pos_limit=args.limit, neg_limit=args.neg)

    if args.output:
        out = Path(args.output)
        out.write_text(json.dumps(pairs, indent=2))
        print(f"Saved {len(pairs)} pairs to {out}")
    elif pairs:
        print(f"\nSample pairs:")
        for p in pairs[:3]:
            print(f"  [{p['id']}] rel={p['relevance']} cat={p['category']}")
            print(f"       Q: {p['query'][:80]}...")
            print(f"       D: {p['document'][:80]}...")
    else:
        print("No pairs loaded — ensure CSV is downloaded to eval/data/")
