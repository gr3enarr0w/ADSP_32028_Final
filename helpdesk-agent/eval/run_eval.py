"""Embedding model evaluation harness for ANTSE-292.

Loads curated query-document pairs from eval/pairs.json, embeds each with the
active provider (default: all-MiniLM-L6-v2), and reports retrieval quality
metrics: MRR, Precision@1/3/5, NDCG@5, and Spearman rank correlation against
human relevance labels.  Also reports per-call latency statistics.

Supports multiple data sources via --sources:
    domain       — hand-curated pairs.json (default, backward compat)
    cqadupstack  — BEIR CQADupStack duplicate Q&A pairs
    kaggle       — Kaggle IT support tickets (retrieval eval)
    jira         — resolved <PROJECT_KEY> tickets (domain-specific)
    all          — merge all available sources

Cross-encoder re-ranking via --rerank:
    Applies ms-marco-MiniLM-L-6-v2 cross-encoder to re-score bi-encoder
    candidates, reporting both before/after metrics for comparison.

Usage:
    python -m eval.run_eval                          # domain pairs only
    python -m eval.run_eval --sources all            # all sources
    python -m eval.run_eval --rerank                 # with cross-encoder re-ranking
    python -m eval.run_eval --rerank --sources all   # full benchmark
    EMBEDDING_PROVIDER=stub python -m eval.run_eval  # compare with stub TF-IDF
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faq.dedup import (
    _normalize, _embedding_to_vector, _compute_similarity,
    EMBEDDING_PROVIDER, rerank_score,
)


VALID_SOURCES = {"domain", "cqadupstack", "kaggle", "jira", "all"}


def load_pairs(path: Path | None = None) -> list[dict]:
    path = path or Path(__file__).resolve().parent / "pairs.json"
    with open(path) as f:
        pairs = json.load(f)
    for p in pairs:
        p.setdefault("source", "domain")
    return pairs


def load_external(sources: list[str]) -> list[dict]:
    all_pairs: list[dict] = []

    if "cqadupstack" in sources or "all" in sources:
        try:
            from eval.loaders.cqadupstack import load_pairs as load_cqa
            all_pairs.extend(load_cqa())
        except Exception as e:
            print(f"[WARN] CQADupStack loader failed: {e}")

    if "kaggle" in sources or "all" in sources:
        try:
            from eval.loaders.kaggle_it import load_pairs as load_kaggle
            pairs = load_kaggle()
            if pairs:
                all_pairs.extend(pairs)
            else:
                print("[WARN] Kaggle IT: no pairs loaded (CSV missing?)")
        except Exception as e:
            print(f"[WARN] Kaggle IT loader failed: {e}")

    if "jira" in sources or "all" in sources:
        try:
            from eval.loaders.jiraconfsd import load_pairs as load_jira
            pairs = load_jira()
            if pairs:
                all_pairs.extend(pairs)
            else:
                print("[WARN] <PROJECT_KEY>: no pairs loaded (cache missing?)")
        except Exception as e:
            print(f"[WARN] <PROJECT_KEY> loader failed: {e}")

    return all_pairs


def merge_pairs(domain_pairs: list[dict], external_pairs: list[dict]) -> list[dict]:
    merged = list(domain_pairs)
    next_id = max((p["id"] for p in merged), default=0) + 1
    for p in external_pairs:
        p["id"] = next_id
        next_id += 1
        merged.append(p)
    return merged


def embed_text(text: str):
    return _embedding_to_vector(text)


def cosine(a, b) -> float:
    return _compute_similarity(a, b)


def dcg(relevances: list[float], k: int) -> float:
    return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances[:k]))


def ndcg_at_k(relevances: list[float], ideal: list[float], k: int) -> float:
    ideal_dcg = dcg(sorted(ideal, reverse=True), k)
    if ideal_dcg == 0:
        return 0.0
    return dcg(relevances, k) / ideal_dcg


def spearman_rank(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 2:
        return 0.0

    def _ranks(vals):
        indexed = sorted(enumerate(vals), key=lambda t: t[1])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and indexed[j + 1][1] == indexed[j][1]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                ranks[indexed[k][0]] = avg_rank
            i = j + 1
        return ranks

    rx, ry = _ranks(x), _ranks(y)
    d_sq = sum((a - b) ** 2 for a, b in zip(rx, ry))
    return 1 - (6 * d_sq) / (n * (n * n - 1))


def _compute_metrics(pairs, query_embeds, doc_embeds, sims, label=""):
    n = len(pairs)
    relevances = [p["relevance"] for p in pairs]

    reciprocal_ranks = []
    precision_at = {1: [], 3: [], 5: []}
    ndcg_scores = []

    for qi in range(n):
        scores = [(cosine(query_embeds[qi], doc_embeds[di]), pairs[di]["relevance"], di)
                  for di in range(n)]
        scores.sort(key=lambda t: t[0], reverse=True)

        ranked_positions = [s[2] for s in scores]
        correct_rank = ranked_positions.index(qi) + 1
        reciprocal_ranks.append(1.0 / correct_rank)

        for k in precision_at:
            top_k_ids = ranked_positions[:k]
            precision_at[k].append(1.0 if qi in top_k_ids else 0.0)

        ranked_rels = [s[1] for s in scores]
        all_rels = [p["relevance"] for p in pairs]
        ndcg_scores.append(ndcg_at_k(ranked_rels, all_rels, 5))

    return {
        "mrr": statistics.mean(reciprocal_ranks),
        "p1": statistics.mean(precision_at[1]),
        "p3": statistics.mean(precision_at[3]),
        "p5": statistics.mean(precision_at[5]),
        "ndcg5": statistics.mean(ndcg_scores),
        "rho": spearman_rank(sims, [float(r) for r in relevances]),
        "reciprocal_ranks": reciprocal_ranks,
    }


def _compute_rerank_metrics(pairs, label=""):
    """Compute retrieval metrics using cross-encoder scores instead of bi-encoder cosine."""
    n = len(pairs)
    relevances = [p["relevance"] for p in pairs]

    rerank_sims = []
    rerank_latencies = []
    for p in pairs:
        t0 = time.perf_counter()
        s = rerank_score(p["query"], p["document"])
        rerank_latencies.append(time.perf_counter() - t0)
        rerank_sims.append(s)

    reciprocal_ranks = []
    precision_at = {1: [], 3: [], 5: []}
    ndcg_scores = []

    for qi in range(n):
        scores = []
        for di in range(n):
            s = rerank_score(pairs[qi]["query"], pairs[di]["document"])
            scores.append((s, pairs[di]["relevance"], di))
        scores.sort(key=lambda t: t[0], reverse=True)

        ranked_positions = [s[2] for s in scores]
        correct_rank = ranked_positions.index(qi) + 1
        reciprocal_ranks.append(1.0 / correct_rank)

        for k in precision_at:
            top_k_ids = ranked_positions[:k]
            precision_at[k].append(1.0 if qi in top_k_ids else 0.0)

        ranked_rels = [s[1] for s in scores]
        all_rels = [p["relevance"] for p in pairs]
        ndcg_scores.append(ndcg_at_k(ranked_rels, all_rels, 5))

    return {
        "mrr": statistics.mean(reciprocal_ranks),
        "p1": statistics.mean(precision_at[1]),
        "p3": statistics.mean(precision_at[3]),
        "p5": statistics.mean(precision_at[5]),
        "ndcg5": statistics.mean(ndcg_scores),
        "rho": spearman_rank(rerank_sims, [float(r) for r in relevances]),
        "reciprocal_ranks": reciprocal_ranks,
        "sims": rerank_sims,
        "latency_mean_ms": statistics.mean(rerank_latencies) * 1000,
        "latency_p50_ms": statistics.median(rerank_latencies) * 1000,
    }


def run(pairs: list[dict] | None = None, sources: list[str] | None = None,
        use_rerank: bool = False) -> dict:
    sources = sources or ["domain"]

    if pairs is None:
        domain_pairs = load_pairs()
        if sources == ["domain"]:
            pairs = domain_pairs
        else:
            external = load_external(sources)
            if "domain" in sources or "all" in sources:
                pairs = merge_pairs(domain_pairs, external)
            else:
                pairs = external
                for i, p in enumerate(pairs, 1):
                    p["id"] = i

    n = len(pairs)
    if n == 0:
        print("No pairs to evaluate.")
        return {}

    query_embeds = []
    doc_embeds = []
    latencies = []

    source_counts: dict[str, int] = {}
    for p in pairs:
        src = p.get("source", "domain")
        source_counts[src] = source_counts.get(src, 0) + 1

    print(f"Provider: {EMBEDDING_PROVIDER}")
    print(f"Rerank: {'cross-encoder/ms-marco-MiniLM-L-6-v2' if use_rerank else 'none'}")
    print(f"Pairs: {n} ({', '.join(f'{s}={c}' for s, c in sorted(source_counts.items()))})")
    print()

    for p in pairs:
        t0 = time.perf_counter()
        qe = embed_text(p["query"])
        de = embed_text(p["document"])
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)
        query_embeds.append(qe)
        doc_embeds.append(de)

    sims = [cosine(q, d) for q, d in zip(query_embeds, doc_embeds)]

    m = _compute_metrics(pairs, query_embeds, doc_embeds, sims)

    results = {
        "provider": EMBEDDING_PROVIDER,
        "num_pairs": n,
        "sources": source_counts,
        "mrr": m["mrr"],
        "precision_at_1": m["p1"],
        "precision_at_3": m["p3"],
        "precision_at_5": m["p5"],
        "ndcg_at_5": m["ndcg5"],
        "spearman_rho": m["rho"],
        "sim_mean": statistics.mean(sims),
        "sim_stdev": statistics.stdev(sims) if n > 1 else 0.0,
        "sim_min": min(sims),
        "sim_max": max(sims),
        "latency_mean_ms": statistics.mean(latencies) * 1000,
        "latency_p50_ms": statistics.median(latencies) * 1000,
        "latency_p95_ms": sorted(latencies)[int(n * 0.95)] * 1000 if n > 1 else latencies[0] * 1000,
    }

    print("=" * 60)
    print(f"  Bi-Encoder — {EMBEDDING_PROVIDER}")
    print("=" * 60)
    print()
    print("Retrieval Quality:")
    print(f"  MRR              {m['mrr']:.4f}")
    print(f"  Precision@1      {m['p1']:.4f}")
    print(f"  Precision@3      {m['p3']:.4f}")
    print(f"  Precision@5      {m['p5']:.4f}")
    print(f"  NDCG@5           {m['ndcg5']:.4f}")
    print(f"  Spearman ρ       {m['rho']:.4f}")
    print()
    print("Similarity Distribution:")
    print(f"  Mean             {results['sim_mean']:.4f}")
    print(f"  Stdev            {results['sim_stdev']:.4f}")
    print(f"  Min              {results['sim_min']:.4f}")
    print(f"  Max              {results['sim_max']:.4f}")
    print()
    print("Latency (ms):")
    print(f"  Mean             {results['latency_mean_ms']:.1f}")
    print(f"  p50              {results['latency_p50_ms']:.1f}")
    print(f"  p95              {results['latency_p95_ms']:.1f}")
    print()

    if use_rerank:
        print("Computing cross-encoder re-ranking scores...")
        print()
        rm = _compute_rerank_metrics(pairs)

        results["rerank_mrr"] = rm["mrr"]
        results["rerank_precision_at_1"] = rm["p1"]
        results["rerank_precision_at_3"] = rm["p3"]
        results["rerank_precision_at_5"] = rm["p5"]
        results["rerank_ndcg_at_5"] = rm["ndcg5"]
        results["rerank_spearman_rho"] = rm["rho"]
        results["rerank_latency_mean_ms"] = rm["latency_mean_ms"]
        results["rerank_latency_p50_ms"] = rm["latency_p50_ms"]

        print("=" * 60)
        print(f"  Cross-Encoder Re-Ranking — ms-marco-MiniLM-L-6-v2")
        print("=" * 60)
        print()
        print("Retrieval Quality:")
        print(f"  MRR              {rm['mrr']:.4f}")
        print(f"  Precision@1      {rm['p1']:.4f}")
        print(f"  Precision@3      {rm['p3']:.4f}")
        print(f"  Precision@5      {rm['p5']:.4f}")
        print(f"  NDCG@5           {rm['ndcg5']:.4f}")
        print(f"  Spearman ρ       {rm['rho']:.4f}")
        print()
        print("Latency (ms):")
        print(f"  Mean             {rm['latency_mean_ms']:.1f}")
        print(f"  p50              {rm['latency_p50_ms']:.1f}")
        print()

        print("=" * 60)
        print("  Comparison: Bi-Encoder vs Cross-Encoder")
        print("=" * 60)
        print()
        print(f"  {'Metric':<16} {'Bi-Enc':>8} {'Cross-Enc':>10} {'Delta':>8}")
        print(f"  {'-'*16} {'-'*8} {'-'*10} {'-'*8}")
        for metric, bi_key, re_key in [
            ("MRR", "mrr", "rerank_mrr"),
            ("Precision@1", "precision_at_1", "rerank_precision_at_1"),
            ("Precision@3", "precision_at_3", "rerank_precision_at_3"),
            ("Precision@5", "precision_at_5", "rerank_precision_at_5"),
            ("NDCG@5", "ndcg_at_5", "rerank_ndcg_at_5"),
            ("Spearman ρ", "spearman_rho", "rerank_spearman_rho"),
        ]:
            bi = results[bi_key]
            re = results[re_key]
            delta = re - bi
            sign = "+" if delta > 0 else ""
            print(f"  {metric:<16} {bi:>8.4f} {re:>10.4f} {sign}{delta:>7.4f}")
        print()

    # Per-source breakdown
    if len(source_counts) > 1:
        print("Per-source metrics:")
        for src in sorted(source_counts):
            src_indices = [i for i, p in enumerate(pairs) if p.get("source", "domain") == src]
            if not src_indices:
                continue
            src_sims = [sims[i] for i in src_indices]
            src_rrs = [m["reciprocal_ranks"][i] for i in src_indices]
            print(f"  {src}: n={len(src_indices)}, MRR={statistics.mean(src_rrs):.3f}, "
                  f"sim_mean={statistics.mean(src_sims):.3f}")
        print()

    print("Per-pair breakdown:")
    if use_rerank:
        print(f"  {'ID':>3}  {'Rel':>3}  {'BiSim':>6}  {'Rank':>4}  {'RRSim':>7}  {'RRank':>5}  {'Source':<12}  Category")
        print(f"  {'---':>3}  {'---':>3}  {'------':>6}  {'----':>4}  {'-------':>7}  {'-----':>5}  {'------':<12}  --------")
        for i, p in enumerate(pairs):
            scores_i = [(cosine(query_embeds[i], doc_embeds[j]), j) for j in range(n)]
            scores_i.sort(key=lambda t: t[0], reverse=True)
            rank = [s[1] for s in scores_i].index(i) + 1

            rr_scores = [(rerank_score(pairs[i]["query"], pairs[j]["document"]), j) for j in range(n)]
            rr_scores.sort(key=lambda t: t[0], reverse=True)
            rr_rank = [s[1] for s in rr_scores].index(i) + 1
            rr_sim = rm["sims"][i]

            src = p.get("source", "domain")
            print(f"  {p['id']:>3}  {p['relevance']:>3}  {sims[i]:>6.3f}  {rank:>4}  {rr_sim:>7.3f}  {rr_rank:>5}  {src:<12}  {p['category']}")
    else:
        print(f"  {'ID':>3}  {'Rel':>3}  {'Sim':>6}  {'Rank':>4}  {'Source':<12}  Category")
        print(f"  {'---':>3}  {'---':>3}  {'------':>6}  {'----':>4}  {'------':<12}  --------")
        for i, p in enumerate(pairs):
            scores_i = [(cosine(query_embeds[i], doc_embeds[j]), j) for j in range(n)]
            scores_i.sort(key=lambda t: t[0], reverse=True)
            rank = [s[1] for s in scores_i].index(i) + 1
            src = p.get("source", "domain")
            print(f"  {p['id']:>3}  {p['relevance']:>3}  {sims[i]:>6.3f}  {rank:>4}  {src:<12}  {p['category']}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embedding evaluation harness")
    parser.add_argument(
        "--sources", nargs="+", default=["domain"],
        choices=list(VALID_SOURCES),
        help="Data sources to include (default: domain)",
    )
    parser.add_argument(
        "--rerank", action="store_true",
        help="Include cross-encoder re-ranking comparison",
    )
    args = parser.parse_args()

    run(sources=args.sources, use_rerank=args.rerank)
