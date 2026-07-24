"""Large-scale retrieval eval: ticket summaries → Confluence articles.

Tests the real production scenario: given a new <PROJECT_KEY> ticket,
can the embedding model find the right FAQ/KB article from the
Confluence corpus (HUB + OMEGA spaces)?

Uses all 8000+ tickets as queries and 483 Confluence articles as docs.
Since we don't have gold-standard ticket→article mappings, we evaluate
using proxy metrics:
  1. Coverage: what % of tickets get a high-confidence article match?
  2. Coherence: do similar tickets map to the same article?
  3. Head-query recall: for known topic clusters (e.g. "confluence access"),
     do they consistently map to the same article?

Usage:
    python -m eval.scripts.run_article_retrieval
    python -m eval.scripts.run_article_retrieval --models mpnet bge
    python -m eval.scripts.run_article_retrieval --top-k 5
"""

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TICKETS_FILE = DATA_DIR / "jiraconfsd_all.json"
ARTICLES_FILE = DATA_DIR / "confluence_articles.json"


def _load():
    tickets = json.loads(TICKETS_FILE.read_text())
    articles = json.loads(ARTICLES_FILE.read_text())
    return tickets, articles


def run(model_names: list[str] | None = None, top_k: int = 5, query_limit: int | None = None):
    from eval.compare_strategies import MODELS, _get_model

    tickets, articles = _load()
    print(f"Loaded {len(tickets)} tickets, {len(articles)} articles")

    queries = []
    query_norm_summaries = []
    for t in tickets:
        s = t.get("summary", "").strip()
        if not s or len(s) < 10:
            continue
        text = s
        desc = t.get("description", "")
        if desc and len(desc) > 30:
            text += ". " + desc[:200]
        queries.append(text[:500])
        query_norm_summaries.append(s.lower().strip())

    if query_limit:
        queries = queries[:query_limit]
        query_norm_summaries = query_norm_summaries[:query_limit]

    docs = []
    doc_titles = []
    for a in articles:
        text = a["title"] + ". " + a.get("body", "")
        docs.append(text[:1000])
        doc_titles.append(a["title"])

    print(f"Queries: {len(queries)}, Docs: {len(docs)}")

    # Build duplicate-summary groups for coherence eval
    norm_groups: dict[str, list[int]] = defaultdict(list)
    for i, ns in enumerate(query_norm_summaries):
        norm_groups[ns].append(i)
    dup_groups = {k: v for k, v in norm_groups.items() if len(v) >= 3}
    print(f"Duplicate-summary groups (size >= 3): {len(dup_groups)}")

    model_names = model_names or ["minilm", "mpnet", "bge", "e5"]
    batch_size = 256

    for model_name in model_names:
        print(f"\n{'='*90}")
        print(f"  {model_name} — {len(queries)} queries × {len(docs)} articles")
        print(f"{'='*90}")

        model = _get_model(model_name)
        info = MODELS[model_name]

        t0 = time.perf_counter()
        print(f"  Encoding {len(docs)} articles...", end="", flush=True)
        d_embs = []
        for start in range(0, len(docs), batch_size):
            batch = [info["d_prefix"] + d for d in docs[start:start + batch_size]]
            d_embs.append(model.encode(batch, normalize_embeddings=True, show_progress_bar=False))
        d_embs = np.vstack(d_embs)
        print(f" done ({d_embs.shape})")

        print(f"  Encoding {len(queries)} queries...", end="", flush=True)
        q_embs = []
        for start in range(0, len(queries), batch_size):
            batch = [info["q_prefix"] + q for q in queries[start:start + batch_size]]
            q_embs.append(model.encode(batch, normalize_embeddings=True, show_progress_bar=False))
        q_embs = np.vstack(q_embs)
        encode_time = time.perf_counter() - t0
        print(f" done ({q_embs.shape}, {encode_time:.1f}s)")

        print(f"  Computing similarity...", end="", flush=True)
        sim = q_embs @ d_embs.T
        print(f" done")

        # --- Metric 1: Coverage ---
        top1_scores = sim.max(axis=1)
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        print(f"\n  Coverage (% of tickets with top-1 article score >= threshold):")
        for th in thresholds:
            pct = (top1_scores >= th).mean() * 100
            print(f"    >= {th:.1f}: {pct:5.1f}%  ({(top1_scores >= th).sum()}/{len(queries)})")

        print(f"\n  Score distribution:")
        print(f"    Mean:   {top1_scores.mean():.3f}")
        print(f"    Median: {np.median(top1_scores):.3f}")
        print(f"    Stdev:  {top1_scores.std():.3f}")
        print(f"    Min:    {top1_scores.min():.3f}")
        print(f"    Max:    {top1_scores.max():.3f}")

        # --- Metric 2: Top articles ---
        top1_indices = sim.argmax(axis=1)
        article_counts = Counter(top1_indices.tolist())
        print(f"\n  Top 15 most-matched articles:")
        for idx, count in article_counts.most_common(15):
            pct = count / len(queries) * 100
            print(f"    {count:5d} ({pct:4.1f}%)  {doc_titles[idx][:80]}")

        uncovered = len(docs) - len(article_counts)
        print(f"\n  Articles never matched as top-1: {uncovered}/{len(docs)} ({uncovered/len(docs)*100:.0f}%)")

        # --- Metric 3: Coherence ---
        print(f"\n  Coherence (do similar tickets map to same article?):")
        coherent = 0
        total_groups = 0
        for gk, indices in list(dup_groups.items())[:100]:
            if len(indices) < 3:
                continue
            total_groups += 1
            mapped_articles = [int(top1_indices[i]) for i in indices if i < len(top1_indices)]
            if not mapped_articles:
                continue
            most_common_article, most_common_count = Counter(mapped_articles).most_common(1)[0]
            agreement = most_common_count / len(mapped_articles)
            if agreement >= 0.5:
                coherent += 1

        if total_groups > 0:
            print(f"    Groups with >50% agreement: {coherent}/{total_groups} ({coherent/total_groups*100:.0f}%)")

        # Show some coherent groups
        print(f"\n  Sample group→article mappings:")
        shown = 0
        for gk, indices in sorted(dup_groups.items(), key=lambda x: -len(x[1])):
            if shown >= 10:
                break
            mapped = [int(top1_indices[i]) for i in indices if i < len(top1_indices)]
            if not mapped:
                continue
            mc_article, mc_count = Counter(mapped).most_common(1)[0]
            agreement = mc_count / len(mapped) * 100
            print(f"    \"{gk[:60]}\" ({len(indices)} tickets)")
            print(f"      → \"{doc_titles[mc_article][:70]}\" ({agreement:.0f}% agreement)")
            shown += 1

        # --- Metric 4: Low-confidence tickets ---
        low_conf = np.where(top1_scores < 0.3)[0]
        print(f"\n  Lowest-confidence tickets (score < 0.3): {len(low_conf)}")
        for i in low_conf[:5]:
            print(f"    Score {top1_scores[i]:.3f}: \"{queries[i][:80]}\"")
            print(f"      Best match: \"{doc_titles[top1_indices[i]][:70]}\"")

        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None,
                        help="Max queries (default: all 8000)")
    args = parser.parse_args()

    run(model_names=args.models, top_k=args.top_k, query_limit=args.limit)
