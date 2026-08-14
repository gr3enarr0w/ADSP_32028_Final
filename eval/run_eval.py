"""
run_eval.py — retrieval evaluation harness for the Agentic RAG node.

Computes, over eval/gold_queries.jsonl:
  * Recall@k, Precision@k, MRR, nDCG@k   (retrieval quality)
  * filter-honor rate                    (does every hit respect price/rating constraints?)
  * provenance validity                  (does every doc_id resolve to a real catalog item?)

Also defines `groundedness_score()` — the answer-level metric the Answerer/Critic
harness uses once the graph is wired (fraction of answer citations that point to
retrieved doc_ids).

Usage:
    PYTHONPATH=src python eval/run_eval.py            # uses configured provider
    PYTHONPATH=src python eval/run_eval.py --k 5
    EMBEDDING_PROVIDER=hash USE_RERANKER=false PYTHONPATH=src python eval/run_eval.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from rag.config import get_config          # noqa: E402
from rag.rag_search import rag_search      # noqa: E402
from rag.retrieval import get_retriever    # noqa: E402

HERE = Path(__file__).resolve().parent
GOLD = HERE / "gold_queries.jsonl"
RESULTS_DIR = HERE / "results"


# ---- metrics ---------------------------------------------------------------

def recall_at_k(relevant: set, retrieved: list) -> float:
    if not relevant:
        return float("nan")
    return len(relevant & set(retrieved)) / len(relevant)


def precision_at_k(relevant: set, retrieved: list, k: int) -> float:
    if k == 0:
        return 0.0
    return len(relevant & set(retrieved[:k])) / k


def reciprocal_rank(relevant: set, retrieved: list) -> float:
    for i, r in enumerate(retrieved, start=1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(relevant: set, retrieved: list, k: int) -> float:
    dcg = 0.0
    for i, r in enumerate(retrieved[:k]):
        if r in relevant:
            dcg += 1.0 / math.log2(i + 2)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return (dcg / idcg) if idcg > 0 else 0.0


def groundedness_score(answer_citation_ids: list, retrieved_ids: list) -> float:
    """Answer-level groundedness: fraction of the answer's citations that are
    backed by a retrieved document. 1.0 == every claim is grounded; < 1.0 flags
    a potential hallucination for the Critic to reject.
    """
    if not answer_citation_ids:
        return 0.0
    retrieved = set(retrieved_ids)
    grounded = sum(1 for c in answer_citation_ids if c in retrieved)
    return grounded / len(answer_citation_ids)


# ---- harness ---------------------------------------------------------------

def load_gold(path: Path = GOLD) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def honors_filters(results: list[dict], filters: dict) -> bool:
    for r in results:
        if filters.get("price_max") is not None:
            if r.get("price") is None or r["price"] > filters["price_max"] + 1e-6:
                return False
        if filters.get("price_min") is not None:
            if r.get("price") is None or r["price"] < filters["price_min"] - 1e-6:
                return False
        if filters.get("min_rating") is not None:
            if r.get("rating") is None or r["rating"] < filters["min_rating"] - 1e-6:
                return False
    return True


def evaluate(k: int) -> dict:
    cfg = get_config()
    gold = load_gold()
    retriever = get_retriever(cfg)
    valid_docids = set(retriever.ids)
    catalog_skus = {m.get("sku") for m in retriever.metas}

    per_query, agg = [], {"recall": [], "precision": [], "mrr": [], "ndcg": [],
                          "filter_honor": [], "provenance": []}

    for g in gold:
        filters = g.get("filters") or None
        out = rag_search(g["query"], k=k, filters=filters)
        results = out["results"]
        retrieved_skus = [r["sku"] for r in results]
        relevant = set(g["relevant_skus"]) & catalog_skus  # guard against typos in gold
        rec = recall_at_k(relevant, retrieved_skus)
        prec = precision_at_k(relevant, retrieved_skus, k)
        rr = reciprocal_rank(relevant, retrieved_skus)
        nd = ndcg_at_k(relevant, retrieved_skus, k)
        honor = honors_filters(results, filters or {})
        prov = all(r["doc_id"] in valid_docids for r in results)

        per_query.append({
            "id": g["id"], "query": g["query"], "recall@k": round(rec, 3),
            "precision@k": round(prec, 3), "rr": round(rr, 3), "ndcg@k": round(nd, 3),
            "filter_honor": honor, "provenance_ok": prov,
            "top": retrieved_skus[:3],
        })
        agg["recall"].append(rec)
        agg["precision"].append(prec)
        agg["mrr"].append(rr)
        agg["ndcg"].append(nd)
        agg["filter_honor"].append(1.0 if honor else 0.0)
        agg["provenance"].append(1.0 if prov else 0.0)

    def mean(xs):
        xs = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
        return round(sum(xs) / len(xs), 4) if xs else float("nan")

    summary = {
        "provider": cfg.embedding_signature(),
        "reranker": cfg.reranker_model if cfg.use_reranker else "off",
        "k": k,
        "n_queries": len(gold),
        f"Recall@{k}": mean(agg["recall"]),
        f"Precision@{k}": mean(agg["precision"]),
        "MRR": mean(agg["mrr"]),
        f"nDCG@{k}": mean(agg["ndcg"]),
        "filter_honor_rate": mean(agg["filter_honor"]),
        "provenance_validity": mean(agg["provenance"]),
    }
    return {"summary": summary, "per_query": per_query}


def _print_table(report: dict):
    s = report["summary"]
    print("\n=== Retrieval Evaluation ===")
    for kk, vv in s.items():
        print(f"  {kk:20s}: {vv}")
    print("\n  per-query:")
    print(f"  {'id':4s} {'R@k':>5s} {'P@k':>5s} {'RR':>5s} {'nDCG':>5s} {'filt':>5s} {'prov':>5s}  query")
    for q in report["per_query"]:
        print(f"  {q['id']:4s} {q['recall@k']:5.2f} {q['precision@k']:5.2f} {q['rr']:5.2f} "
              f"{q['ndcg@k']:5.2f} {str(q['filter_honor']):>5s} {str(q['provenance_ok']):>5s}  {q['query'][:52]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=get_config().top_k)
    ap.add_argument("--out", default=str(RESULTS_DIR / "eval_report.json"))
    args = ap.parse_args()

    report = evaluate(args.k)
    _print_table(report)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
