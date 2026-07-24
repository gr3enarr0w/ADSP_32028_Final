"""Error analysis — what does the model get wrong, and how wrong is it?

Shows per-query breakdown: rank of correct answer, what ranked higher,
and categorizes errors by severity (near-miss vs complete whiff).

Usage:
    python -m eval.error_analysis                        # domain pairs
    python -m eval.error_analysis --sources cqadupstack   # OOD pairs
    python -m eval.error_analysis --model bge             # specific model
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.compare_strategies import MODELS, _get_model, _embed_with_model
from eval.run_eval import load_pairs, load_external, merge_pairs, VALID_SOURCES
from faq.dedup import _normalize, _tokenize, _tfidf_vector, _dict_to_unit_vector, _cosine_sim


def _run_analysis(pairs, model_name: str, label: str):
    n = len(pairs)
    queries = [p["query"] for p in pairs]
    docs = [p["document"] for p in pairs]
    rels = [p["relevance"] for p in pairs]

    print(f"\nEmbedding {n} queries + {n} docs with {model_name}...")
    model = _get_model(model_name)
    info = MODELS[model_name]

    q_embs = np.array([
        model.encode(info["q_prefix"] + q, normalize_embeddings=True) for q in queries
    ])
    d_embs = np.array([
        model.encode(info["d_prefix"] + d, normalize_embeddings=True) for d in docs
    ])

    sim_matrix = q_embs @ d_embs.T

    # also compute TF-IDF baseline for comparison
    all_texts = queries + docs
    tok_lists = [_tokenize(_normalize(t)) for t in all_texts]
    from collections import Counter
    df = Counter()
    for tl in tok_lists:
        df.update(set(tl))
    idf = {w: np.log((len(all_texts) + 1) / (c + 1)) + 1 for w, c in df.items()}

    def tfidf_sim(a, b):
        va = _tfidf_vector(_tokenize(_normalize(a)), idf)
        vb = _tfidf_vector(_tokenize(_normalize(b)), idf)
        return _cosine_sim(_dict_to_unit_vector(va), _dict_to_unit_vector(vb))

    # per-query analysis
    dup_indices = [i for i in range(n) if rels[i] >= 4]
    neg_indices = [i for i in range(n) if rels[i] <= 2]

    print(f"\n{'='*100}")
    print(f"  ERROR ANALYSIS — {label} ({model_name})")
    print(f"  {len(dup_indices)} duplicate pairs, {len(neg_indices)} negative pairs, {n} total docs to rank")
    print(f"{'='*100}")

    ranks = []
    errors = []
    tfidf_ranks = []

    for qi in dup_indices:
        scores = sim_matrix[qi]
        ranked = np.argsort(-scores)
        rank = int(np.where(ranked == qi)[0][0]) + 1
        ranks.append(rank)

        # TF-IDF rank for same query
        tfidf_scores = [tfidf_sim(queries[qi], docs[j]) for j in range(n)]
        tfidf_ranked = np.argsort(tfidf_scores)[::-1]
        tfidf_rank = int(np.where(tfidf_ranked == qi)[0][0]) + 1
        tfidf_ranks.append(tfidf_rank)

        if rank > 1:
            errors.append({
                "qi": qi,
                "rank": rank,
                "tfidf_rank": tfidf_rank,
                "query": queries[qi],
                "correct_doc": docs[qi],
                "correct_score": float(scores[qi]),
                "top_doc": docs[ranked[0]],
                "top_score": float(scores[ranked[0]]),
                "top_idx": int(ranked[0]),
                "top_rel": rels[ranked[0]],
                "category": pairs[qi].get("category", "?"),
            })

    # summary stats
    print(f"\n  Rank distribution of correct answer (n={len(dup_indices)} duplicate queries):")
    from collections import Counter as C
    rank_dist = C(ranks)
    for r in sorted(rank_dist.keys()):
        bar = "#" * rank_dist[r]
        pct = rank_dist[r] / len(dup_indices) * 100
        print(f"    Rank {r:3d}: {rank_dist[r]:3d} ({pct:5.1f}%) {bar}")

    at1 = sum(1 for r in ranks if r == 1)
    at3 = sum(1 for r in ranks if r <= 3)
    at5 = sum(1 for r in ranks if r <= 5)
    at10 = sum(1 for r in ranks if r <= 10)
    mrr = sum(1/r for r in ranks) / len(ranks)

    print(f"\n  {model_name}: MRR={mrr:.3f}  P@1={at1/len(ranks):.3f}  P@3={at3/len(ranks):.3f}  P@5={at5/len(ranks):.3f}  P@10={at10/len(ranks):.3f}")

    tfidf_mrr = sum(1/r for r in tfidf_ranks) / len(tfidf_ranks)
    tfidf_at1 = sum(1 for r in tfidf_ranks if r == 1)
    print(f"  TF-IDF:  MRR={tfidf_mrr:.3f}  P@1={tfidf_at1/len(tfidf_ranks):.3f}")
    print(f"  Lift over naive: MRR +{mrr - tfidf_mrr:.3f} ({(mrr - tfidf_mrr) / tfidf_mrr * 100:+.0f}%)")

    # error severity breakdown
    near_miss = [e for e in errors if e["rank"] <= 5]
    mid_miss = [e for e in errors if 5 < e["rank"] <= 20]
    far_miss = [e for e in errors if e["rank"] > 20]

    print(f"\n  Error breakdown ({len(errors)} total errors out of {len(dup_indices)} queries):")
    print(f"    Near-miss (rank 2-5):  {len(near_miss):3d} ({len(near_miss)/len(dup_indices)*100:.0f}%)")
    print(f"    Mid-miss  (rank 6-20): {len(mid_miss):3d} ({len(mid_miss)/len(dup_indices)*100:.0f}%)")
    print(f"    Far-miss  (rank 21+):  {len(far_miss):3d} ({len(far_miss)/len(dup_indices)*100:.0f}%)")

    # show worst errors
    errors_sorted = sorted(errors, key=lambda e: -e["rank"])
    show = min(15, len(errors_sorted))
    print(f"\n  {'─'*96}")
    print(f"  Top {show} worst errors (highest rank = most wrong):")
    print(f"  {'─'*96}")

    for e in errors_sorted[:show]:
        tfidf_note = f"TF-IDF rank={e['tfidf_rank']}"
        beat = "BETTER" if e["tfidf_rank"] > e["rank"] else "WORSE" if e["tfidf_rank"] < e["rank"] else "SAME"
        print(f"\n  [{e['category']}] Correct at rank {e['rank']} (score {e['correct_score']:.3f}) | {tfidf_note} ({beat})")
        print(f"  Q: {e['query'][:120]}")
        print(f"  Correct D: {e['correct_doc'][:120]}")
        print(f"  Model picked (rank 1, rel={e['top_rel']}): {e['top_doc'][:120]}")
        print(f"  Score gap: {e['top_score']:.3f} vs {e['correct_score']:.3f} (Δ={e['top_score'] - e['correct_score']:.3f})")

    # category breakdown
    if any("category" in p for p in pairs):
        print(f"\n  {'─'*96}")
        print(f"  Per-category breakdown:")
        print(f"  {'─'*96}")
        cats = {}
        for qi, r in zip(dup_indices, ranks):
            cat = pairs[qi].get("category", "?")
            cats.setdefault(cat, []).append(r)
        for cat in sorted(cats, key=lambda c: sum(1/r for r in cats[c])/len(cats[c])):
            rs = cats[cat]
            cmrr = sum(1/r for r in rs) / len(rs)
            cat1 = sum(1 for r in rs if r == 1)
            print(f"    {cat:30s}  n={len(rs):3d}  MRR={cmrr:.3f}  P@1={cat1/len(rs):.3f}  ranks: {sorted(rs)[:10]}{'...' if len(rs)>10 else ''}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", default=["domain"], choices=list(VALID_SOURCES) + ["all"])
    parser.add_argument("--model", type=str, default="bge", choices=list(MODELS.keys()))
    args = parser.parse_args()

    sources = args.sources
    if "all" in sources:
        sources = VALID_SOURCES

    domain_pairs = load_pairs() if "domain" in sources else []
    ext_sources = [s for s in sources if s != "domain"]
    ext_pairs = load_external(ext_sources) if ext_sources else []
    pairs = merge_pairs(domain_pairs, ext_pairs)
    label = "+".join(sources)

    _run_analysis(pairs, args.model, label)
