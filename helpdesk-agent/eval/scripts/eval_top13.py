"""Evaluate 13 top strategies on the 95% validation split (~288 pairs).

Uses seed=42 to identify the 5% screening indices (same as screen_models.py),
then takes the REMAINING 95% for evaluation.

Usage:
    python -m eval.scripts.eval_top13
"""

import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc
import json
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from eval.compare_strategies import (
    MODELS, _get_model, _eval_ranking, _cv_grid_search,
    _loaded_models, _normalize_scores,
)
from faq.dedup import (
    _normalize, _tokenize, _tfidf_vector, _dict_to_unit_vector,
    _cosine_sim, _build_idf, _normalize_for_embedding,
    CROSS_ENCODER_NAME,
)


def load_val_95(seed: int = 42) -> list[dict]:
    """Load val pairs, exclude the 5% screening sample, return remaining 95%."""
    val_path = Path(__file__).resolve().parent.parent / "data" / "training_pairs_val.json"
    with open(val_path) as f:
        raw = json.load(f)

    # Reproduce the exact 5% sample that screen_models.py used
    rng = random.Random(seed)
    k = max(1, int(len(raw) * 0.05))
    screen_sample = rng.sample(raw, k)

    # Identify screen indices by object identity (rng.sample returns refs)
    screen_ids = set(id(entry) for entry in screen_sample)
    remaining = [entry for entry in raw if id(entry) not in screen_ids]

    print(f"Total val pairs: {len(raw)}", flush=True)
    print(f"Screen sample (5%): {k}", flush=True)
    print(f"Remaining (95%): {len(remaining)}", flush=True)

    # Build positive + negative pairs
    rng2 = random.Random(seed)
    pairs = []
    for entry in remaining:
        pairs.append({
            "query": entry["anchor"],
            "document": entry["positive"],
            "relevance": 5,
        })

    n = len(remaining)
    for i, entry in enumerate(remaining):
        j = rng2.randint(0, n - 2)
        if j >= i:
            j += 1
        pairs.append({
            "query": entry["anchor"],
            "document": remaining[j]["positive"],
            "relevance": 1,
        })

    n_pos = sum(1 for p in pairs if p["relevance"] == 5)
    n_neg = sum(1 for p in pairs if p["relevance"] == 1)
    print(f"Total eval pairs: {len(pairs)} ({n_pos} positive + {n_neg} negative)", flush=True)
    return pairs


def main():
    t_start = time.perf_counter()
    pairs = load_val_95()
    n = len(pairs)

    bi_model_names = ["minilm", "mpnet", "bge", "e5", "minilm-ft", "mpnet-ft"]

    # ── Bi-encoder matrices (batch encode, then free models) ──

    bi_matrices: dict[str, list[list[float]]] = {}
    for mname in bi_model_names:
        print(f"Encoding bi:{mname}...", flush=True)
        t0 = time.perf_counter()
        info = MODELS[mname]
        model = _get_model(mname)

        q_texts = [info["q_prefix"] + p["query"] for p in pairs]
        d_texts = [info["d_prefix"] + p["document"] for p in pairs]

        q_embs = model.encode(q_texts, normalize_embeddings=True,
                              show_progress_bar=False, batch_size=64)
        d_embs = model.encode(d_texts, normalize_embeddings=True,
                              show_progress_bar=False, batch_size=64)

        mat = (q_embs @ d_embs.T).tolist()
        bi_matrices[mname] = mat
        elapsed = time.perf_counter() - t0
        print(f"  Done in {elapsed:.1f}s", flush=True)

    # Free all bi-encoder models from memory before cross-encoder
    print("Freeing bi-encoder models from memory...", flush=True)
    _loaded_models.clear()
    gc.collect()

    # ── Cross-encoder matrix (batch predict) ──

    print("Computing cross-encoder matrix...", flush=True)
    from sentence_transformers import CrossEncoder
    print(f"  Loading cross-encoder: {CROSS_ENCODER_NAME}", flush=True)
    ce_model = CrossEncoder(CROSS_ENCODER_NAME)

    queries_norm = [_normalize_for_embedding(p["query"]) for p in pairs]
    docs_norm = [_normalize_for_embedding(p["document"]) for p in pairs]

    total = n * n
    print(f"  Scoring {total:,} pairs...", flush=True)

    # Process row-by-row to avoid giant memory allocation
    cross_matrix = []
    t0 = time.perf_counter()
    for qi in range(n):
        input_pairs = [(queries_norm[qi], docs_norm[di]) for di in range(n)]
        scores = ce_model.predict(input_pairs, batch_size=64, show_progress_bar=False)
        cross_matrix.append([float(s) for s in scores])
        if (qi + 1) % 50 == 0 or qi == 0:
            elapsed = time.perf_counter() - t0
            eta = elapsed / (qi + 1) * (n - qi - 1)
            print(f"  Row {qi+1}/{n} ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"  Cross-encoder done in {elapsed:.1f}s", flush=True)

    # Free cross-encoder
    del ce_model
    gc.collect()

    # ── TF-IDF matrix ──

    print("Computing TF-IDF scores...", flush=True)
    t0 = time.perf_counter()
    all_texts_norm = [_normalize(p["query"]) for p in pairs] + \
                     [_normalize(p["document"]) for p in pairs]
    all_tokens = [_tokenize(t) for t in all_texts_norm]
    idf = _build_idf(all_tokens)
    stub_q = [_dict_to_unit_vector(_tfidf_vector(_tokenize(_normalize(p["query"])), idf))
              for p in pairs]
    stub_d = [_dict_to_unit_vector(_tfidf_vector(_tokenize(_normalize(p["document"])), idf))
              for p in pairs]
    tfidf_matrix = [[_cosine_sim(stub_q[qi], stub_d[di])
                     for di in range(n)] for qi in range(n)]
    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f}s", flush=True)

    # ── Build score functions ──

    def _make_fn(matrix):
        def fn(qi, di):
            return matrix[qi][di]
        return fn

    bi_fns = {name: _make_fn(bi_matrices[name]) for name in bi_model_names}
    cross_fn = _make_fn(cross_matrix)
    tfidf_fn = _make_fn(tfidf_matrix)
    signal_fns = {**bi_fns, "cross": cross_fn, "tfidf": tfidf_fn}

    # ── Define the 13 strategies ──

    strategies_spec = [
        ("bi:minilm", "solo", ["minilm"]),
        ("bi:minilm-ft", "solo", ["minilm-ft"]),
        ("bi:mpnet-ft", "solo", ["mpnet-ft"]),
        ("minilm+e5+cross", "weighted", ["minilm", "e5", "cross"]),
        ("minilm+mpnet-ft+cross", "weighted", ["minilm", "mpnet-ft", "cross"]),
        ("minilm+mpnet+e5", "weighted", ["minilm", "mpnet", "e5"]),
        ("minilm+mpnet+cross", "weighted", ["minilm", "mpnet", "cross"]),
        ("minilm+minilm-ft+cross", "weighted", ["minilm", "minilm-ft", "cross"]),
        ("minilm+e5+tfidf", "weighted", ["minilm", "e5", "tfidf"]),
        ("minilm+e5", "weighted", ["minilm", "e5"]),
        ("minilm+cross", "weighted", ["minilm", "cross"]),
        ("minilm+bge+e5", "weighted", ["minilm", "bge", "e5"]),
        ("minilm+bge+cross", "weighted", ["minilm", "bge", "cross"]),
    ]

    # ── Evaluate each strategy ──

    results = {}

    for idx, (name, stype, signals) in enumerate(strategies_spec, 1):
        print(f"\n[{idx}/13] Evaluating {name}...", flush=True)
        t0 = time.perf_counter()

        if stype == "solo":
            mname = signals[0]
            metrics = _eval_ranking(pairs, bi_fns[mname])
            results[name] = {"metrics": metrics, "weights": None}
        elif stype == "weighted":
            fn_dict = {s: signal_fns[s] for s in signals}
            best_weights, best_result, cv_info = _cv_grid_search(
                pairs, fn_dict, k_folds=5, steps=11)
            results[name] = {
                "metrics": best_result,
                "weights": best_weights,
                "cv": cv_info,
            }

        elapsed = time.perf_counter() - t0
        m = results[name]["metrics"]
        print(f"  MRR={m['mrr']:.4f}  P@1={m['p1']:.4f}  NDCG@5={m['ndcg5']:.4f}  "
              f"({elapsed:.1f}s)", flush=True)
        if results[name].get("weights"):
            w = results[name]["weights"]
            print(f"  Weights: {', '.join(f'{k}={v:.2f}' for k, v in w.items())}", flush=True)

    # ── Print full comparison table ──

    total_elapsed = time.perf_counter() - t_start
    print(f"\nTotal time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)", flush=True)
    print(flush=True)

    w = 140
    print("=" * w, flush=True)
    print(f"  TOP-13 STRATEGY COMPARISON (95% val split, {n} pairs)", flush=True)
    print("=" * w, flush=True)
    print(flush=True)

    ranked = sorted(results.items(), key=lambda t: t[1]["metrics"]["mrr"], reverse=True)

    col = 30
    header = (f"  {'Rank':<5} {'Strategy':<{col}} {'MRR':>7} {'P@1':>7} {'P@3':>7} "
              f"{'P@5':>7} {'NDCG@5':>8} {'Sp_rho':>7} {'Prsn_r':>7}  Weights")
    print(header, flush=True)
    print("  " + "-" * (len(header) - 2), flush=True)

    for rank, (name, data) in enumerate(ranked, 1):
        m = data["metrics"]
        wt_str = ""
        if data.get("weights"):
            wt_str = ", ".join(f"{k}={v:.2f}" for k, v in data["weights"].items())
        print(f"  {rank:<5} {name:<{col}} {m['mrr']:>7.4f} {m['p1']:>7.4f} {m['p3']:>7.4f} "
              f"{m['p5']:>7.4f} {m['ndcg5']:>8.4f} {m['rho']:>7.4f} {m['pearson']:>7.4f}  {wt_str}",
              flush=True)

    # CV overfit check
    print(flush=True)
    print("=" * w, flush=True)
    print("  CV OVERFITTING CHECK", flush=True)
    print("=" * w, flush=True)
    print(flush=True)
    print(f"  {'Strategy':<{col}} {'Full MRR':>9} {'CV MRR':>8} {'Train':>8} {'Gap':>7}  Verdict",
          flush=True)
    print(f"  {'-'*col} {'-'*9} {'-'*8} {'-'*8} {'-'*7}  {'-'*12}", flush=True)

    for name, data in ranked:
        if "cv" not in data or not data["cv"]:
            continue
        cv = data["cv"]
        gap = cv["overfit_gap"]
        if gap > 0.10:
            verdict = "OVERFITTING"
        elif gap > 0.05:
            verdict = "MILD OVERFIT"
        else:
            verdict = "OK"
        print(f"  {name:<{col}} {cv['full_data_mrr']:>9.4f} {cv['cv_mrr']:>8.4f} "
              f"{cv['train_mrr']:>8.4f} {gap:>+7.4f}  {verdict}", flush=True)

    print(flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
