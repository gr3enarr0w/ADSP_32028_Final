"""Quick screening: rank individual strategies on a 5% sample of validation pairs.

Loads eval/data/training_pairs_val.json, samples 5% (~15 pairs) with seed=42,
generates negatives, and runs all 10 individual strategies to produce an MRR
ranking.  Designed to finish in under a minute on CPU.

Usage:
    python -m eval.scripts.screen_models
"""

import os
os.environ["USE_TF"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import math
import random
import statistics
import time
from collections import Counter
from pathlib import Path

import numpy as np

# Ensure project root is on sys.path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from eval.compare_strategies import (
    MODELS, _get_model, _eval_ranking, _embed_with_model,
    _jaccard_sim, _bm25_score, _bm25_idf,
)
from faq.dedup import (
    _normalize, _tokenize, _tfidf_vector, _dict_to_unit_vector,
    _cosine_sim, _build_idf, rerank_score,
)


def load_val_sample(seed: int = 42, frac: float = 0.05) -> list[dict]:
    """Load val pairs, sample 5%, and generate positive + negative pairs."""
    val_path = Path(__file__).resolve().parent.parent / "data" / "training_pairs_val.json"
    with open(val_path) as f:
        raw = json.load(f)

    rng = random.Random(seed)
    k = max(1, int(len(raw) * frac))
    sample = rng.sample(raw, k)
    print(f"Sampled {k} pairs from {len(raw)} validation pairs (seed={seed})")

    pairs = []
    # Positive pairs
    for entry in sample:
        pairs.append({
            "query": entry["anchor"],
            "document": entry["positive"],
            "relevance": 5,
        })

    # Negative pairs: pair each anchor with a random different positive
    n = len(sample)
    for i, entry in enumerate(sample):
        j = rng.randint(0, n - 2)
        if j >= i:
            j += 1
        pairs.append({
            "query": entry["anchor"],
            "document": sample[j]["positive"],
            "relevance": 1,
        })

    print(f"Total pairs: {len(pairs)} ({n} positive + {n} negative)")
    return pairs


def main():
    pairs = load_val_sample()
    n = len(pairs)

    model_names = ["minilm", "mpnet", "bge", "e5", "minilm-ft", "mpnet-ft"]
    results: dict[str, dict] = {}

    # -- TF-IDF --
    print("\n[1/10] TF-IDF...")
    all_texts_norm = [_normalize(p["query"]) for p in pairs] + \
                     [_normalize(p["document"]) for p in pairs]
    all_tokens = [_tokenize(t) for t in all_texts_norm]
    idf = _build_idf(all_tokens)
    stub_q = [_dict_to_unit_vector(_tfidf_vector(_tokenize(_normalize(p["query"])), idf))
              for p in pairs]
    stub_d = [_dict_to_unit_vector(_tfidf_vector(_tokenize(_normalize(p["document"])), idf))
              for p in pairs]
    stub_matrix = [[_cosine_sim(stub_q[qi], stub_d[di])
                    for di in range(n)] for qi in range(n)]
    results["TF-IDF"] = _eval_ranking(pairs, lambda qi, di: stub_matrix[qi][di])

    # -- BM25 --
    print("[2/10] BM25...")
    q_tokens = [_tokenize(_normalize(p["query"])) for p in pairs]
    d_tokens = [_tokenize(_normalize(p["document"])) for p in pairs]
    bm25_idf = _bm25_idf(d_tokens)
    avgdl = statistics.mean(len(t) for t in d_tokens) if d_tokens else 1
    bm25_matrix = [[_bm25_score(q_tokens[qi], d_tokens[di], bm25_idf, avgdl)
                    for di in range(n)] for qi in range(n)]
    results["BM25"] = _eval_ranking(pairs, lambda qi, di: bm25_matrix[qi][di])

    # -- MinHash --
    print("[3/10] MinHash...")
    jac_matrix = [[_jaccard_sim(pairs[qi]["query"], pairs[di]["document"])
                   for di in range(n)] for qi in range(n)]
    results["MinHash"] = _eval_ranking(pairs, lambda qi, di: jac_matrix[qi][di])

    # -- Bi-encoders --
    bi_matrices: dict[str, list[list[float]]] = {}
    for idx, mname in enumerate(model_names, 4):
        print(f"[{idx}/10] bi:{mname}...")
        q_embs = [_embed_with_model(p["query"], mname, is_query=True) for p in pairs]
        d_embs = [_embed_with_model(p["document"], mname, is_query=False) for p in pairs]
        mat = [[float(np.dot(q_embs[qi], d_embs[di]))
                for di in range(n)] for qi in range(n)]
        bi_matrices[mname] = mat
        label = f"bi:{mname}"
        results[label] = _eval_ranking(pairs, lambda qi, di, m=mat: m[qi][di])

    # -- Cross-encoder --
    print("[10/10] Cross-encoder...")
    cross_matrix = [[rerank_score(pairs[qi]["query"], pairs[di]["document"])
                     for di in range(n)] for qi in range(n)]
    results["cross-encoder"] = _eval_ranking(
        pairs, lambda qi, di: cross_matrix[qi][di])

    # -- Print ranked results --
    ranked = sorted(results.items(), key=lambda t: t[1]["mrr"], reverse=True)

    print()
    print("=" * 80)
    print("  INDIVIDUAL STRATEGY SCREENING (5% val sample)")
    print("=" * 80)
    print()
    print(f"  {'Rank':<6} {'Strategy':<22} {'MRR':>7} {'P@1':>7} {'NDCG@5':>8} "
          f"{'Spearman':>9} {'Pearson':>8}")
    print(f"  {'-'*6} {'-'*22} {'-'*7} {'-'*7} {'-'*8} {'-'*9} {'-'*8}")

    for rank, (name, m) in enumerate(ranked, 1):
        print(f"  {rank:<6} {name:<22} {m['mrr']:>7.4f} {m['p1']:>7.4f} "
              f"{m['ndcg5']:>8.4f} {m['rho']:>9.4f} {m['pearson']:>8.4f}")

    print()
    top8 = [name for name, _ in ranked[:8]]
    print(f"  Top 8: {', '.join(top8)}")
    print()


if __name__ == "__main__":
    main()
