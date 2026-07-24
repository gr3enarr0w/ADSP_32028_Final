"""Build training pairs for fine-tuning: ticket → Confluence article.

Loads all <PROJECT_KEY> tickets and Confluence articles, computes similarity
using a pre-trained bi-encoder, then builds positive pairs from two sources:

  1. Duplicate-summary groups — tickets that share a normalized summary are
     grouped together.  For each group, every ticket is scored against all
     articles; the article with >=50% majority-vote agreement becomes the
     positive for every ticket in that group.

  2. Singleton tickets — tickets not in any duplicate group.  If a ticket's
     top-1 article similarity is >= 0.6, that (ticket, article) becomes a
     positive pair.

Pairs are randomly split 80/10/10 into train/val/test and saved as JSON.

Usage:
    python -m eval.scripts.build_training_data
    python -m eval.scripts.build_training_data --model mpnet
    python -m eval.scripts.build_training_data --threshold 0.55
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from eval.loaders.jiraconfsd import (
    _ticket_text,
    _normalize_summary,
    _build_duplicate_groups,
)
from eval.compare_strategies import MODELS, _get_model

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TICKETS_FILE = DATA_DIR / "jiraconfsd_all.json"
ARTICLES_FILE = DATA_DIR / "confluence_articles.json"


def _article_text(article: dict, max_len: int = 500) -> str:
    title = article.get("title", "")
    body = article.get("body", "")
    text = title
    if body and len(body) > 30:
        text += ". " + body[:max_len - len(title) - 2]
    return text[:max_len]


def _encode_batch(model, texts: list[str], prefix: str = "",
                  batch_size: int = 256) -> np.ndarray:
    embeddings = []
    for start in range(0, len(texts), batch_size):
        batch = [prefix + t for t in texts[start:start + batch_size]]
        embs = model.encode(batch, normalize_embeddings=True,
                            show_progress_bar=False)
        embeddings.append(embs)
    return np.vstack(embeddings)


def build_pairs(model_name: str = "minilm", threshold: float = 0.6,
                seed: int = 42) -> list[dict]:
    tickets = json.loads(TICKETS_FILE.read_text())
    articles = json.loads(ARTICLES_FILE.read_text())
    print(f"Loaded {len(tickets)} tickets, {len(articles)} articles")

    model_info = MODELS[model_name]
    model = _get_model(model_name)
    q_prefix = model_info["q_prefix"]
    d_prefix = model_info["d_prefix"]

    ticket_texts = [_ticket_text(t) for t in tickets]
    article_texts = [_article_text(a) for a in articles]

    print(f"Encoding {len(ticket_texts)} tickets...")
    t_embs = _encode_batch(model, ticket_texts, prefix=q_prefix)
    print(f"Encoding {len(article_texts)} articles...")
    a_embs = _encode_batch(model, article_texts, prefix=d_prefix)

    print("Computing similarity matrix...")
    sim_matrix = t_embs @ a_embs.T  # (n_tickets, n_articles)

    dup_groups = _build_duplicate_groups(tickets)
    dup_ticket_indices = set()
    for indices in dup_groups.values():
        dup_ticket_indices.update(indices)

    pairs = []
    source_counts = Counter()

    # Source 1: duplicate-summary groups with majority-vote article
    print(f"Processing {len(dup_groups)} duplicate groups...")
    for group_key, indices in dup_groups.items():
        top_articles = []
        for idx in indices:
            best_article_idx = int(np.argmax(sim_matrix[idx]))
            top_articles.append(best_article_idx)

        vote_counts = Counter(top_articles)
        majority_article_idx, majority_count = vote_counts.most_common(1)[0]
        agreement = majority_count / len(indices)

        if agreement < 0.5:
            continue

        article_text = article_texts[majority_article_idx]
        for idx in indices:
            pairs.append({
                "anchor": ticket_texts[idx],
                "positive": article_text,
                "source": "dup-group",
                "group_key": group_key,
                "agreement": round(agreement, 3),
                "sim_score": round(float(sim_matrix[idx][majority_article_idx]), 4),
            })
            source_counts["dup-group"] += 1

    # Source 2: singleton tickets with high top-1 similarity
    print("Processing singleton tickets...")
    singleton_indices = [i for i in range(len(tickets))
                         if i not in dup_ticket_indices]
    for idx in singleton_indices:
        best_article_idx = int(np.argmax(sim_matrix[idx]))
        best_score = float(sim_matrix[idx][best_article_idx])

        if best_score >= threshold:
            pairs.append({
                "anchor": ticket_texts[idx],
                "positive": article_texts[best_article_idx],
                "source": "singleton",
                "sim_score": round(best_score, 4),
            })
            source_counts["singleton"] += 1

    print(f"\nTotal pairs: {len(pairs)}")
    print(f"  dup-group:  {source_counts['dup-group']}")
    print(f"  singleton:  {source_counts['singleton']}")

    return pairs


def split_and_save(pairs: list[dict], seed: int = 42):
    rng = random.Random(seed)
    rng.shuffle(pairs)

    n = len(pairs)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    train = pairs[:train_end]
    val = pairs[train_end:val_end]
    test = pairs[val_end:]

    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        out_path = DATA_DIR / f"training_pairs_{split_name}.json"
        out_path.write_text(json.dumps(split_data, indent=2))
        print(f"  {split_name}: {len(split_data)} pairs → {out_path.name}")

    return train, val, test


def main():
    parser = argparse.ArgumentParser(
        description="Build training pairs from tickets and articles")
    parser.add_argument("--model", choices=list(MODELS.keys()), default="minilm",
                        help="Bi-encoder model for similarity (default: minilm)")
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="Min similarity for singleton pairs (default: 0.6)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pairs = build_pairs(model_name=args.model, threshold=args.threshold,
                        seed=args.seed)
    if not pairs:
        print("No pairs generated.", file=sys.stderr)
        sys.exit(1)

    print(f"\nSplitting {len(pairs)} pairs (80/10/10, seed={args.seed})...")
    train, val, test = split_and_save(pairs, seed=args.seed)

    source_breakdown = Counter(p["source"] for p in pairs)
    print(f"\nSource breakdown:")
    for src, count in source_breakdown.most_common():
        print(f"  {src}: {count}")

    print(f"\nSplit sizes:")
    print(f"  train: {len(train)}")
    print(f"  val:   {len(val)}")
    print(f"  test:  {len(test)}")

    train_sources = Counter(p["source"] for p in train)
    val_sources = Counter(p["source"] for p in val)
    test_sources = Counter(p["source"] for p in test)
    print(f"\nPer-split source breakdown:")
    print(f"  train: {dict(train_sources)}")
    print(f"  val:   {dict(val_sources)}")
    print(f"  test:  {dict(test_sources)}")


if __name__ == "__main__":
    main()
