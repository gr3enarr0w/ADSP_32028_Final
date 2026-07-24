"""Compare scoring strategies — naive, neural, structural, and advanced ensembles.

Runs all strategies on the eval dataset and reports quality metrics (MRR, P@1,
NDCG@5, Spearman ρ, Pearson r) alongside cost metrics (latency, memory, calls
per query, estimated $/query at GKE pricing).

Ensemble weights are optimized via grid search with k-fold cross-validation to
detect overfitting.  The --compare flag runs domain and cross-domain (CQADupStack)
separately and prints a side-by-side table showing the domain→OOD generalization gap.

Multi-model bi-encoders:
    MiniLM-L6-v2 (384-dim), mpnet-base-v2 (768-dim), bge-base-en-v1.5 (768-dim),
    E5-small-v2 (384-dim) — each tested individually and in cross-model ensembles.

Usage:
    python -m eval.compare_strategies
    python -m eval.compare_strategies --compare          # side-by-side domain vs OOD
    python -m eval.compare_strategies --models minilm mpnet bge e5  # specific models
"""

import argparse
import json
import math
import random
import statistics
import sys
import time
import tracemalloc
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faq.dedup import (
    _embedding_to_vector, _compute_similarity, compute_minhash,
    rerank_score, EMBEDDING_PROVIDER,
    _normalize, _tokenize, _tfidf_vector, _dict_to_unit_vector, _cosine_sim,
    _build_idf, _shingle, NUM_PERM,
)
from eval.run_eval import (
    load_pairs, load_external, merge_pairs, dcg, ndcg_at_k, spearman_rank,
    VALID_SOURCES,
)
from datasketch import MinHash


# ── Multi-model bi-encoder registry ──

MODELS = {
    "minilm": {
        "id": "sentence-transformers/all-MiniLM-L6-v2",
        "dim": 384,
        "q_prefix": "",
        "d_prefix": "",
    },
    "mpnet": {
        "id": "sentence-transformers/all-mpnet-base-v2",
        "dim": 768,
        "q_prefix": "",
        "d_prefix": "",
    },
    "bge": {
        "id": "BAAI/bge-base-en-v1.5",
        "dim": 768,
        "q_prefix": "",
        "d_prefix": "",
    },
    "e5": {
        "id": "intfloat/e5-small-v2",
        "dim": 384,
        "q_prefix": "query: ",
        "d_prefix": "passage: ",
    },
    "minilm-ft": {
        "id": str(Path(__file__).resolve().parent / "data/models/minilm-finetuned"),
        "dim": 384,
        "q_prefix": "",
        "d_prefix": "",
    },
    "mpnet-ft": {
        "id": str(Path(__file__).resolve().parent / "data/models/mpnet-finetuned"),
        "dim": 768,
        "q_prefix": "",
        "d_prefix": "",
    },
}

_loaded_models: dict[str, object] = {}


def load_test_pairs(path: Path | None = None, seed: int = 42) -> list[dict]:
    """Load held-out test pairs and generate negatives for discrimination.

    Each entry in the test JSON has 'anchor' and 'positive' fields (positive
    pair). For each positive, we generate one negative by pairing the anchor
    with a random non-matching positive (document from a different pair).

    Returns pairs with fields: query, document, relevance (5 for positive,
    1 for negative).
    """
    path = path or Path(__file__).resolve().parent / "data" / "training_pairs_test.json"
    with open(path) as f:
        raw = json.load(f)

    rng = random.Random(seed)
    pairs = []

    # Positive pairs
    for entry in raw:
        pairs.append({
            "query": entry["anchor"],
            "document": entry["positive"],
            "relevance": 5,
        })

    # Negative pairs: for each anchor, pair with a random non-matching positive
    n = len(raw)
    for i, entry in enumerate(raw):
        j = rng.randint(0, n - 2)
        if j >= i:
            j += 1
        pairs.append({
            "query": entry["anchor"],
            "document": raw[j]["positive"],
            "relevance": 1,
        })

    return pairs


def _get_model(name: str):
    if name not in _loaded_models:
        from sentence_transformers import SentenceTransformer
        info = MODELS[name]
        _loaded_models[name] = SentenceTransformer(info["id"])
        print(f"  Loaded {info['id']} ({info['dim']}-dim)")
    return _loaded_models[name]


def _embed_with_model(text: str, model_name: str, is_query: bool = True) -> np.ndarray:
    info = MODELS[model_name]
    prefix = info["q_prefix"] if is_query else info["d_prefix"]
    model = _get_model(model_name)
    return model.encode(prefix + text, normalize_embeddings=True)


# ── Correlation helpers ──

def _pearson_r(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    if sx == 0 or sy == 0:
        return 0.0
    return cov / (sx * sy)


# ── Scoring helpers ──

def _jaccard_sim(text_a: str, text_b: str) -> float:
    mh_a = compute_minhash(text_a)
    mh_b = compute_minhash(text_b)
    if mh_a is None or mh_b is None:
        return 0.0
    return mh_a.jaccard(mh_b)


def _idf_weighted_minhash(text: str, idf: dict[str, float]) -> MinHash | None:
    """MinHash with IDF-weighted shingle insertion (more hash updates for rare shingles)."""
    norm = _normalize(text or "")
    if not norm:
        return None
    shingles = _shingle(norm)
    if not shingles:
        return None
    mh = MinHash(num_perm=NUM_PERM)
    weights = []
    for s in shingles:
        words = s.split()
        weights.append(sum(idf.get(w, 1.0) for w in words) / len(words))
    if not weights:
        return None
    min_w = min(weights)
    max_w = max(weights)
    spread = max_w - min_w if max_w > min_w else 1.0
    for s, w in zip(shingles, weights):
        scaled = 1 + int(4 * (w - min_w) / spread)
        encoded = s.encode("utf-8")
        for _ in range(scaled):
            mh.update(encoded)
    return mh


def _idf_jaccard_sim(text_a: str, text_b: str, idf: dict[str, float]) -> float:
    mh_a = _idf_weighted_minhash(text_a, idf)
    mh_b = _idf_weighted_minhash(text_b, idf)
    if mh_a is None or mh_b is None:
        return 0.0
    return mh_a.jaccard(mh_b)


def _bm25_score(query_tokens: list[str], doc_tokens: list[str],
                idf: dict[str, float], avgdl: float,
                k1: float = 1.5, b: float = 0.75) -> float:
    tf = Counter(doc_tokens)
    dl = len(doc_tokens)
    score = 0.0
    for qt in set(query_tokens):
        if qt not in tf:
            continue
        f = tf[qt]
        numerator = f * (k1 + 1)
        denominator = f + k1 * (1 - b + b * dl / avgdl)
        score += idf.get(qt, 0.0) * numerator / denominator
    return score


def _bm25_idf(doc_tokens_list: list[list[str]]) -> dict[str, float]:
    n = len(doc_tokens_list)
    df: Counter = Counter()
    for tokens in doc_tokens_list:
        df.update(set(tokens))
    return {w: math.log((n - count + 0.5) / (count + 0.5) + 1)
            for w, count in df.items()}


def _platt_scale(scores: list[float], labels: list[int]) -> tuple[float, float]:
    """Fit Platt scaling parameters A, B via gradient descent on log-loss.

    Initializes A positive (higher score = more relevant) and uses
    Adam-style adaptive learning rate for robust convergence.
    """
    a, b_param = 1.0, 0.0
    lr = 0.1
    m_a, m_b = 0.0, 0.0
    v_a, v_b = 0.0, 0.0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    n = len(scores)
    for t in range(1, 2001):
        grad_a, grad_b = 0.0, 0.0
        for s, y in zip(scores, labels):
            z = a * s + b_param
            z = max(-30, min(30, z))
            p = 1.0 / (1.0 + math.exp(-z))
            err = p - y
            grad_a += err * s
            grad_b += err
        grad_a /= n
        grad_b /= n
        m_a = beta1 * m_a + (1 - beta1) * grad_a
        m_b = beta1 * m_b + (1 - beta1) * grad_b
        v_a = beta2 * v_a + (1 - beta2) * grad_a ** 2
        v_b = beta2 * v_b + (1 - beta2) * grad_b ** 2
        mhat_a = m_a / (1 - beta1 ** t)
        mhat_b = m_b / (1 - beta1 ** t)
        vhat_a = v_a / (1 - beta2 ** t)
        vhat_b = v_b / (1 - beta2 ** t)
        a -= lr * mhat_a / (math.sqrt(vhat_a) + eps)
        b_param -= lr * mhat_b / (math.sqrt(vhat_b) + eps)
    return a, b_param


def _platt_transform(score: float, a: float, b: float) -> float:
    return 1.0 / (1.0 + math.exp(-(a * score + b)))


def _rrf_score(rank_lists: list[list[int]], doc_idx: int, k: int = 60) -> float:
    """RRF score for a document across multiple ranked lists."""
    score = 0.0
    for ranked in rank_lists:
        if doc_idx in ranked:
            rank = ranked.index(doc_idx) + 1
            score += 1.0 / (k + rank)
    return score


def _normalize_scores(scores: list[float]) -> list[float]:
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [0.5] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


# ── Evaluation ──

def _eval_ranking(pairs, score_fn) -> dict:
    """Evaluate a scoring function on retrieval metrics."""
    n = len(pairs)
    reciprocal_ranks = []
    precision_at = {1: [], 3: [], 5: []}
    ndcg_scores = []
    pair_sims = []

    t0 = time.perf_counter()
    for qi in range(n):
        scores = [(score_fn(qi, di), pairs[di]["relevance"], di) for di in range(n)]
        scores.sort(key=lambda t: t[0], reverse=True)

        ranked_positions = [s[2] for s in scores]
        correct_rank = ranked_positions.index(qi) + 1
        reciprocal_ranks.append(1.0 / correct_rank)

        for k in precision_at:
            precision_at[k].append(1.0 if qi in ranked_positions[:k] else 0.0)

        ranked_rels = [s[1] for s in scores]
        all_rels = [p["relevance"] for p in pairs]
        ndcg_scores.append(ndcg_at_k(ranked_rels, all_rels, 5))

        pair_sims.append(score_fn(qi, qi))
    total_time = time.perf_counter() - t0

    relevances = [float(p["relevance"]) for p in pairs]
    return {
        "mrr": statistics.mean(reciprocal_ranks),
        "p1": statistics.mean(precision_at[1]),
        "p3": statistics.mean(precision_at[3]),
        "p5": statistics.mean(precision_at[5]),
        "ndcg5": statistics.mean(ndcg_scores),
        "rho": spearman_rank(pair_sims, relevances),
        "pearson": _pearson_r(pair_sims, relevances),
        "total_time_s": total_time,
        "calls": n * n,
    }


def _eval_two_stage(pairs, bi_matrix, cross_matrix, top_k: int) -> dict:
    """Evaluate two-stage retrieve-then-rerank: bi-encoder top-K → cross-encoder rerank."""
    n = len(pairs)
    reciprocal_ranks = []
    precision_at = {1: [], 3: [], 5: []}
    ndcg_scores = []
    pair_sims = []

    for qi in range(n):
        bi_scores = [(bi_matrix[qi][di], di) for di in range(n)]
        bi_scores.sort(key=lambda t: t[0], reverse=True)
        candidates = [s[1] for s in bi_scores[:top_k]]

        reranked = [(cross_matrix[qi][di], pairs[di]["relevance"], di) for di in candidates]
        rest = [(bi_scores[i][0], pairs[bi_scores[i][1]]["relevance"], bi_scores[i][1])
                for i in range(top_k, n)]
        reranked.sort(key=lambda t: t[0], reverse=True)
        full_ranking = reranked + rest

        ranked_positions = [s[2] for s in full_ranking]
        correct_rank = ranked_positions.index(qi) + 1
        reciprocal_ranks.append(1.0 / correct_rank)

        for k in precision_at:
            precision_at[k].append(1.0 if qi in ranked_positions[:k] else 0.0)

        ranked_rels = [s[1] for s in full_ranking]
        all_rels = [p["relevance"] for p in pairs]
        ndcg_scores.append(ndcg_at_k(ranked_rels, all_rels, 5))

        pair_sims.append(cross_matrix[qi][qi] if qi in candidates else bi_matrix[qi][qi])

    relevances = [float(p["relevance"]) for p in pairs]
    return {
        "mrr": statistics.mean(reciprocal_ranks),
        "p1": statistics.mean(precision_at[1]),
        "p3": statistics.mean(precision_at[3]),
        "p5": statistics.mean(precision_at[5]),
        "ndcg5": statistics.mean(ndcg_scores),
        "rho": spearman_rank(pair_sims, relevances),
        "pearson": _pearson_r(pair_sims, relevances),
        "total_time_s": 0,
        "calls": n * top_k,
    }


def _eval_rrf(pairs, matrices: list[list[list[float]]]) -> dict:
    """Evaluate RRF fusion across multiple score matrices."""
    n = len(pairs)
    reciprocal_ranks = []
    precision_at = {1: [], 3: [], 5: []}
    ndcg_scores = []
    pair_sims = []

    for qi in range(n):
        rank_lists = []
        for matrix in matrices:
            scored = [(matrix[qi][di], di) for di in range(n)]
            scored.sort(key=lambda t: t[0], reverse=True)
            rank_lists.append([s[1] for s in scored])

        rrf_scores = [(_rrf_score(rank_lists, di), pairs[di]["relevance"], di)
                      for di in range(n)]
        rrf_scores.sort(key=lambda t: t[0], reverse=True)

        ranked_positions = [s[2] for s in rrf_scores]
        correct_rank = ranked_positions.index(qi) + 1
        reciprocal_ranks.append(1.0 / correct_rank)

        for k in precision_at:
            precision_at[k].append(1.0 if qi in ranked_positions[:k] else 0.0)

        ranked_rels = [s[1] for s in rrf_scores]
        all_rels = [p["relevance"] for p in pairs]
        ndcg_scores.append(ndcg_at_k(ranked_rels, all_rels, 5))

        pair_sims.append(_rrf_score(rank_lists, qi))

    relevances = [float(p["relevance"]) for p in pairs]
    return {
        "mrr": statistics.mean(reciprocal_ranks),
        "p1": statistics.mean(precision_at[1]),
        "p3": statistics.mean(precision_at[3]),
        "p5": statistics.mean(precision_at[5]),
        "ndcg5": statistics.mean(ndcg_scores),
        "rho": spearman_rank(pair_sims, relevances),
        "pearson": _pearson_r(pair_sims, relevances),
        "total_time_s": 0,
        "calls": 0,
    }


# ── Grid search / CV ──

def _make_weights_grid(n_signals: int, steps: int = 11):
    if n_signals == 2:
        return [(round(a / (steps - 1), 2), round(1.0 - a / (steps - 1), 2))
                for a in range(steps)]
    elif n_signals == 3:
        grid = []
        for a in range(steps):
            for b in range(steps - a):
                c = steps - 1 - a - b
                grid.append((round(a / (steps - 1), 2),
                             round(b / (steps - 1), 2),
                             round(c / (steps - 1), 2)))
        return grid
    return []


def _grid_search_weights(pairs, score_fns: dict[str, callable], steps=11):
    """Grid search over weight combinations for ensemble scoring."""
    keys = list(score_fns.keys())
    weights_grid = _make_weights_grid(len(keys), steps)
    if not weights_grid:
        return None, None

    best_mrr = -1
    best_weights = None
    best_result = None

    n = len(pairs)
    all_scores = {}
    for key, fn in score_fns.items():
        matrix = [[fn(qi, di) for di in range(n)] for qi in range(n)]
        for qi in range(n):
            matrix[qi] = _normalize_scores(matrix[qi])
        all_scores[key] = matrix

    for weights in weights_grid:
        def _ensemble_fn(qi, di, _w=weights):
            return sum(_w[i] * all_scores[k][qi][di] for i, k in enumerate(keys))

        result = _eval_ranking(pairs, _ensemble_fn)
        if result["mrr"] > best_mrr:
            best_mrr = result["mrr"]
            best_weights = dict(zip(keys, weights))
            best_result = result

    return best_weights, best_result


def _cv_grid_search(pairs, score_fns: dict[str, callable], k_folds=5, steps=11):
    """K-fold cross-validated grid search — train weights on k-1 folds, eval on held-out fold."""
    keys = list(score_fns.keys())
    weights_grid = _make_weights_grid(len(keys), steps)
    if not weights_grid:
        return None, None, {}

    n = len(pairs)
    all_scores = {}
    for key, fn in score_fns.items():
        matrix = [[fn(qi, di) for di in range(n)] for qi in range(n)]
        for qi in range(n):
            matrix[qi] = _normalize_scores(matrix[qi])
        all_scores[key] = matrix

    indices = list(range(n))
    rng = np.random.RandomState(42)
    rng.shuffle(indices)
    folds = [indices[i::k_folds] for i in range(k_folds)]

    fold_results = []
    train_mrrs = []

    for fold_idx in range(k_folds):
        test_idx = set(folds[fold_idx])
        train_idx = [i for i in range(n) if i not in test_idx]

        train_pairs = [pairs[i] for i in train_idx]
        n_train = len(train_pairs)

        train_scores = {}
        for key in keys:
            mat = [[all_scores[key][train_idx[qi]][train_idx[di]]
                    for di in range(n_train)] for qi in range(n_train)]
            for qi in range(n_train):
                mat[qi] = _normalize_scores(mat[qi])
            train_scores[key] = mat

        best_train_mrr = -1
        best_w = None
        for weights in weights_grid:
            def _fn(qi, di, _w=weights):
                return sum(_w[i] * train_scores[k][qi][di] for i, k in enumerate(keys))
            r = _eval_ranking(train_pairs, _fn)
            if r["mrr"] > best_train_mrr:
                best_train_mrr = r["mrr"]
                best_w = weights

        train_mrrs.append(best_train_mrr)

        test_list = sorted(test_idx)
        n_test = len(test_list)
        test_pairs = [pairs[i] for i in test_list]
        test_scores = {}
        for key in keys:
            mat = [[all_scores[key][test_list[qi]][test_list[di]]
                    for di in range(n_test)] for qi in range(n_test)]
            for qi in range(n_test):
                mat[qi] = _normalize_scores(mat[qi])
            test_scores[key] = mat

        def _test_fn(qi, di, _w=best_w):
            return sum(_w[i] * test_scores[k][qi][di] for i, k in enumerate(keys))

        test_result = _eval_ranking(test_pairs, _test_fn)
        fold_results.append(test_result["mrr"])

    cv_mrr = statistics.mean(fold_results)
    train_mrr = statistics.mean(train_mrrs)
    overfit_gap = train_mrr - cv_mrr

    best_weights, best_result = _grid_search_weights(pairs, score_fns, steps)

    cv_info = {
        "cv_mrr": cv_mrr,
        "cv_fold_mrrs": fold_results,
        "train_mrr": train_mrr,
        "overfit_gap": overfit_gap,
        "full_data_mrr": best_result["mrr"],
    }
    return best_weights, best_result, cv_info


# ── Main ──

def run(sources: list[str] | None = None, source_label: str = "",
        model_names: list[str] | None = None, save_path: str | None = None,
        test_pairs: list[dict] | None = None, max_combo_size: int = 0):
    sources = sources or ["domain"]
    model_names = model_names or ["minilm", "mpnet", "bge", "e5"]

    if test_pairs is not None:
        pairs = test_pairs
    else:
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
        return

    print(f"Models: {', '.join(model_names)}")
    print(f"Pairs: {n}")
    print()

    # ── Pre-compute all signals ──

    # Stub TF-IDF
    print("Pre-computing stub TF-IDF vectors...")
    tracemalloc.start()
    mem_before = tracemalloc.get_traced_memory()[1]

    t0 = time.perf_counter()
    all_texts_norm = [_normalize(p["query"]) for p in pairs] + [_normalize(p["document"]) for p in pairs]
    all_tokens = [_tokenize(t) for t in all_texts_norm]
    idf = _build_idf(all_tokens)
    stub_q = [_dict_to_unit_vector(_tfidf_vector(_tokenize(_normalize(p["query"])), idf)) for p in pairs]
    stub_d = [_dict_to_unit_vector(_tfidf_vector(_tokenize(_normalize(p["document"])), idf)) for p in pairs]
    stub_time = time.perf_counter() - t0

    mem_after_stub = tracemalloc.get_traced_memory()[1]
    stub_mem_mb = (mem_after_stub - mem_before) / (1024 * 1024)

    stub_matrix = [[_cosine_sim(stub_q[qi], stub_d[di]) for di in range(n)] for qi in range(n)]

    # BM25
    print("Pre-computing BM25 scores (grid-searching k1, b)...")
    t0 = time.perf_counter()
    q_tokens = [_tokenize(_normalize(p["query"])) for p in pairs]
    d_tokens = [_tokenize(_normalize(p["document"])) for p in pairs]
    bm25_idf = _bm25_idf(d_tokens)
    avgdl = statistics.mean(len(t) for t in d_tokens) if d_tokens else 1

    best_bm25_mrr = -1
    best_k1, best_b = 1.5, 0.75
    for k1_candidate in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
        for b_candidate in [0.3, 0.5, 0.75, 0.9]:
            trial_matrix = [[_bm25_score(q_tokens[qi], d_tokens[di], bm25_idf, avgdl, k1_candidate, b_candidate)
                             for di in range(n)] for qi in range(n)]

            def _trial_fn(qi, di, _m=trial_matrix): return _m[qi][di]
            trial_result = _eval_ranking(pairs, _trial_fn)
            if trial_result["mrr"] > best_bm25_mrr:
                best_bm25_mrr = trial_result["mrr"]
                best_k1, best_b = k1_candidate, b_candidate

    bm25_matrix = [[_bm25_score(q_tokens[qi], d_tokens[di], bm25_idf, avgdl, best_k1, best_b)
                    for di in range(n)] for qi in range(n)]
    bm25_time = time.perf_counter() - t0

    # Multi-model bi-encoders
    bi_matrices: dict[str, list[list[float]]] = {}
    bi_times: dict[str, float] = {}
    bi_mems: dict[str, float] = {}

    for mname in model_names:
        print(f"Pre-computing {mname} bi-encoder embeddings...")
        mem_pre = tracemalloc.get_traced_memory()[1]
        t0 = time.perf_counter()
        q_embs = [_embed_with_model(p["query"], mname, is_query=True) for p in pairs]
        d_embs = [_embed_with_model(p["document"], mname, is_query=False) for p in pairs]
        bi_times[mname] = time.perf_counter() - t0
        mem_post = tracemalloc.get_traced_memory()[1]
        bi_mems[mname] = (mem_post - mem_pre) / (1024 * 1024)
        bi_matrices[mname] = [[float(np.dot(q_embs[qi], d_embs[di]))
                               for di in range(n)] for qi in range(n)]

    # Cross-encoder
    print("Pre-computing cross-encoder scores...")
    mem_pre_cross = tracemalloc.get_traced_memory()[1]
    t0 = time.perf_counter()
    cross_matrix = [[rerank_score(pairs[qi]["query"], pairs[di]["document"]) for di in range(n)] for qi in range(n)]
    cross_time = time.perf_counter() - t0

    mem_after_cross = tracemalloc.get_traced_memory()[1]
    cross_mem_mb = (mem_after_cross - mem_pre_cross) / (1024 * 1024)

    # Platt-scaled cross-encoder
    print("Fitting Platt scaling on cross-encoder scores...")
    diag_scores = [cross_matrix[i][i] for i in range(n)]
    off_diag_scores = []
    off_diag_labels = []
    for qi in range(n):
        for di in range(n):
            if qi != di:
                off_diag_scores.append(cross_matrix[qi][di])
                off_diag_labels.append(1 if pairs[di]["relevance"] >= 4 else 0)
    all_calib_scores = diag_scores + off_diag_scores
    all_calib_labels = [1] * n + off_diag_labels
    platt_a, platt_b = _platt_scale(all_calib_scores, all_calib_labels)
    platt_matrix = [[_platt_transform(cross_matrix[qi][di], platt_a, platt_b)
                     for di in range(n)] for qi in range(n)]

    # MinHash (standard)
    print("Pre-computing MinHash Jaccard scores...")
    t0 = time.perf_counter()
    jac_matrix = [[_jaccard_sim(pairs[qi]["query"], pairs[di]["document"]) for di in range(n)] for qi in range(n)]
    jac_time = time.perf_counter() - t0

    mem_after_jac = tracemalloc.get_traced_memory()[1]
    jac_mem_mb = (mem_after_jac - mem_after_cross) / (1024 * 1024)

    # IDF-weighted MinHash
    print("Pre-computing IDF-weighted MinHash scores...")
    t0 = time.perf_counter()
    idf_jac_matrix = [[_idf_jaccard_sim(pairs[qi]["query"], pairs[di]["document"], idf)
                       for di in range(n)] for qi in range(n)]
    idf_jac_time = time.perf_counter() - t0

    tracemalloc.stop()

    stub_per_call = (stub_time / (2 * n)) * 1000
    bm25_per_call = (bm25_time / (n * n)) * 1000
    cross_per_call = (cross_time / (n * n)) * 1000
    jac_per_call = (jac_time / (n * n)) * 1000
    idf_jac_per_call = (idf_jac_time / (n * n)) * 1000

    # ── Evaluate all strategies ──
    strategies = {}
    strategy_order = []

    # Baselines
    strategy_order.extend(["stub (TF-IDF)", "BM25"])

    def _stub_fn(qi, di): return stub_matrix[qi][di]
    def _bm25_fn(qi, di): return bm25_matrix[qi][di]
    def _cross_fn(qi, di): return cross_matrix[qi][di]
    def _platt_fn(qi, di): return platt_matrix[qi][di]
    def _jac_fn(qi, di): return jac_matrix[qi][di]
    def _idf_jac_fn(qi, di): return idf_jac_matrix[qi][di]

    print("Evaluating individual strategies...")

    strategies["stub (TF-IDF)"] = _eval_ranking(pairs, _stub_fn)
    strategies["stub (TF-IDF)"]["cost"] = {
        "latency_per_call_ms": stub_per_call,
        "model_calls_per_query": 0, "models_loaded": 0, "mem_mb": stub_mem_mb,
        "note": "Naive word-frequency TF-IDF — no neural model",
    }

    strategies["BM25"] = _eval_ranking(pairs, _bm25_fn)
    strategies["BM25"]["cost"] = {
        "latency_per_call_ms": bm25_per_call,
        "model_calls_per_query": 0, "models_loaded": 0, "mem_mb": stub_mem_mb,
        "note": f"BM25 (k1={best_k1}, b={best_b}) — tuned lexical baseline",
    }

    # Per-model bi-encoder strategies
    bi_fns: dict[str, callable] = {}
    for mname in model_names:
        label = f"bi:{mname}"
        strategy_order.append(label)
        _mat = bi_matrices[mname]
        def _fn(qi, di, m=_mat): return m[qi][di]
        bi_fns[mname] = _fn
        bi_per = (bi_times[mname] / (2 * n)) * 1000
        strategies[label] = _eval_ranking(pairs, _fn)
        strategies[label]["cost"] = {
            "latency_per_call_ms": bi_per,
            "model_calls_per_query": 1, "models_loaded": 1,
            "mem_mb": bi_mems[mname],
            "note": f"{MODELS[mname]['id']} ({MODELS[mname]['dim']}-dim)",
        }

    # Cross-encoder and structural strategies
    strategy_order.extend(["cross-encoder", "cross-enc (Platt)", "minhash", "minhash (IDF)"])

    strategies["cross-encoder"] = _eval_ranking(pairs, _cross_fn)
    strategies["cross-encoder"]["cost"] = {
        "latency_per_call_ms": cross_per_call,
        "model_calls_per_query": n, "models_loaded": 1, "mem_mb": cross_mem_mb,
        "note": "Must score each (query, doc) pair — O(n) per query",
    }

    strategies["cross-enc (Platt)"] = _eval_ranking(pairs, _platt_fn)
    strategies["cross-enc (Platt)"]["cost"] = {
        "latency_per_call_ms": cross_per_call,
        "model_calls_per_query": n, "models_loaded": 1, "mem_mb": cross_mem_mb,
        "note": f"Platt-calibrated cross-encoder (A={platt_a:.3f}, B={platt_b:.3f})",
    }

    strategies["minhash"] = _eval_ranking(pairs, _jac_fn)
    strategies["minhash"]["cost"] = {
        "latency_per_call_ms": jac_per_call,
        "model_calls_per_query": 0, "models_loaded": 0, "mem_mb": jac_mem_mb,
        "note": "Pure CPU, no model — structural similarity only",
    }

    strategies["minhash (IDF)"] = _eval_ranking(pairs, _idf_jac_fn)
    strategies["minhash (IDF)"]["cost"] = {
        "latency_per_call_ms": idf_jac_per_call,
        "model_calls_per_query": 0, "models_loaded": 0, "mem_mb": jac_mem_mb,
        "note": "IDF-weighted shingles — rare terms get more hash updates",
    }

    # ── All ensemble combinations ──
    k_folds = min(5, n)

    # Signal registry: name → (score_fn, matrix, per_call_ms, mem_mb)
    all_signals = {}
    for mname in model_names:
        bi_per = (bi_times[mname] / (2 * n)) * 1000
        all_signals[mname] = {
            "fn": bi_fns[mname], "matrix": bi_matrices[mname],
            "per_call_ms": bi_per, "mem_mb": bi_mems[mname],
            "calls_per_q": 1, "models": 1, "type": "bi",
        }
    all_signals["cross"] = {
        "fn": _cross_fn, "matrix": cross_matrix,
        "per_call_ms": cross_per_call, "mem_mb": cross_mem_mb,
        "calls_per_q": n, "models": 1, "type": "cross",
    }
    all_signals["tfidf"] = {
        "fn": _stub_fn, "matrix": stub_matrix,
        "per_call_ms": stub_per_call, "mem_mb": stub_mem_mb,
        "calls_per_q": 0, "models": 0, "type": "lexical",
    }
    all_signals["bm25"] = {
        "fn": _bm25_fn, "matrix": bm25_matrix,
        "per_call_ms": bm25_per_call, "mem_mb": stub_mem_mb,
        "calls_per_q": 0, "models": 0, "type": "lexical",
    }
    all_signals["mhash"] = {
        "fn": _jac_fn, "matrix": jac_matrix,
        "per_call_ms": jac_per_call, "mem_mb": jac_mem_mb,
        "calls_per_q": 0, "models": 0, "type": "lexical",
    }
    all_signals["mhash-idf"] = {
        "fn": _idf_jac_fn, "matrix": idf_jac_matrix,
        "per_call_ms": idf_jac_per_call, "mem_mb": jac_mem_mb,
        "calls_per_q": 0, "models": 0, "type": "lexical",
    }

    signal_names = list(all_signals.keys())
    max_size = max_combo_size if max_combo_size > 0 else len(signal_names)
    max_size = min(max_size, len(signal_names))

    # Count total combos for progress
    total_combos = sum(
        1 for size in range(2, max_size + 1)
        for _ in combinations(signal_names, size)
    )
    print(f"Evaluating all ensemble combinations of {len(signal_names)} signals "
          f"(sizes 2..{max_size}, {total_combos} combos)...")

    combo_idx = 0
    # Every subset of size 2..max_size
    for size in range(2, max_size + 1):
        for combo in combinations(signal_names, size):
            combo_idx += 1
            combo_label = "+".join(combo)
            print(f"  Combo {combo_idx}/{total_combos}: {combo_label}")

            combo_bi = [s for s in combo if all_signals[s]["type"] == "bi"]
            has_cross = "cross" in combo
            combo_mats = [all_signals[s]["matrix"] for s in combo]
            total_per = sum(all_signals[s]["per_call_ms"] for s in combo)
            total_mem = sum(all_signals[s]["mem_mb"] for s in combo)
            total_calls = sum(all_signals[s]["calls_per_q"] for s in combo)
            total_models = len(combo_bi) + (1 if has_cross else 0)

            cost_base = {
                "latency_per_call_ms": total_per,
                "model_calls_per_query": total_calls,
                "models_loaded": total_models,
                "mem_mb": total_mem,
            }

            # RRF
            rrf_label = f"{combo_label} (RRF)"
            strategy_order.append(rrf_label)
            rrf_result = _eval_rrf(pairs, combo_mats)
            strategies[rrf_label] = rrf_result
            strategies[rrf_label]["cost"] = {
                **cost_base,
                "note": f"RRF(k=60): {'+'.join(combo)}",
            }

            # Weighted CV (only for 2 or 3 signals — grid search is O(steps^n))
            if size <= 3:
                w_label = f"{combo_label} (w)"
                strategy_order.append(w_label)
                fn_dict = {s: all_signals[s]["fn"] for s in combo}
                print(f"  CV grid-searching {w_label} ({k_folds}-fold)...")
                w_best, r_best, cv_info = _cv_grid_search(
                    pairs, fn_dict, k_folds=k_folds)
                if r_best:
                    strategies[w_label] = r_best
                    strategies[w_label]["weights"] = w_best
                    strategies[w_label]["cv"] = cv_info
                    wt_str = ", ".join(f"{k}={v:.2f}" for k, v in w_best.items())
                    strategies[w_label]["cost"] = {
                        **cost_base,
                        "note": f"Weighted: {wt_str}",
                    }

    # 2-stage: each bi-encoder → cross-encoder rerank
    for mname in model_names:
        bi_per = (bi_times[mname] / (2 * n)) * 1000
        for top_k in [5, 10]:
            label = f"2-stage:{mname} (K={top_k})"
            strategy_order.append(label)
            ts_result = _eval_two_stage(pairs, bi_matrices[mname], cross_matrix, top_k=top_k)
            strategies[label] = ts_result
            strategies[label]["cost"] = {
                "latency_per_call_ms": bi_per + cross_per_call * (top_k / n),
                "model_calls_per_query": 1 + top_k, "models_loaded": 2,
                "mem_mb": bi_mems[mname] + cross_mem_mb,
                "note": f"{mname} top-{top_k} → cross-encoder rerank",
            }

    _print_results(strategies, strategy_order, n, label=source_label)

    if save_path:
        serializable = {}
        for name in strategy_order:
            s = strategies[name]
            entry = {k: v for k, v in s.items()
                     if k in ("mrr", "p1", "p3", "p5", "ndcg5", "rho", "pearson", "cost")}
            if "weights" in s:
                entry["weights"] = s["weights"]
            if "cv" in s:
                entry["cv"] = s["cv"]
            serializable[name] = entry
        Path(save_path).write_text(json.dumps({
            "strategy_order": strategy_order,
            "strategies": serializable,
        }, indent=2))
        print(f"\nResults saved to {save_path}")

    return strategies


def _print_results(strategies, strategy_order, n, label=""):
    # ── Main comparison table ──
    w = 140
    print()
    print("=" * w)
    if label:
        print(f"  STRATEGY COMPARISON — {label}")
    else:
        print("  STRATEGY COMPARISON")
    print("=" * w)
    print()

    col = 30
    header = (f"  {'Strategy':<{col}} {'MRR':>6} {'P@1':>6} {'P@3':>6} {'P@5':>6} "
              f"{'NDCG@5':>7} {'Spρ':>6} {'Prsn':>6} "
              f"{'ms/call':>8} {'CE/Q':>6} {'Mdls':>5} {'MB':>6} {'$/1Kq':>7}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    stub_mrr = strategies["stub (TF-IDF)"]["mrr"]

    for name in strategy_order:
        s = strategies[name]
        c = s["cost"]
        delta = s["mrr"] - stub_mrr
        delta_str = f"({'+' if delta > 0 else ''}{delta:.3f})" if name != "stub (TF-IDF)" else "(baseline)"
        cost_1k = _estimate_cost_per_1k(c)
        print(f"  {name:<{col}} {s['mrr']:>6.3f} {s['p1']:>6.3f} {s['p3']:>6.3f} {s['p5']:>6.3f} "
              f"{s['ndcg5']:>7.3f} {s['rho']:>6.3f} {s['pearson']:>6.3f} "
              f"{c['latency_per_call_ms']:>8.2f} "
              f"{c['model_calls_per_query']:>6} {c['models_loaded']:>5} {c['mem_mb']:>6.1f} "
              f"{cost_1k:>7}  {delta_str}")

    # ── Cost breakdown ──
    print()
    print("=" * w)
    print("  COST BREAKDOWN")
    print("=" * w)
    print()
    print("  All models run self-hosted on CPU (GKE e2-standard-2, ~$0.067/hr).")
    print("  No API calls. Cost = pod time + memory reservation.")
    print("  $/1Kq = estimated dollars per 1,000 queries (CPU time only, not idle pod cost).")
    print()

    for name in strategy_order:
        s = strategies[name]
        c = s["cost"]
        cost_1k = _estimate_cost_per_1k(c)
        print(f"  {name}")
        print(f"    {c['note']}")
        print(f"    Latency: {c['latency_per_call_ms']:.2f} ms/call | "
              f"CE calls/query: {c['model_calls_per_query']} | "
              f"Models in memory: {c['models_loaded']} | "
              f"Peak mem: {c['mem_mb']:.1f} MB | "
              f"Est. cost: {cost_1k}/1K queries")
        if "weights" in s:
            wt = s["weights"]
            print(f"    Optimal weights: {', '.join(f'{k}={v:.2f}' for k, v in wt.items())}")
        if "cv" in s:
            cv = s["cv"]
            print(f"    CV MRR: {cv['cv_mrr']:.3f} (train={cv['train_mrr']:.3f}, "
                  f"gap={cv['overfit_gap']:+.3f}, folds={[round(f, 3) for f in cv['cv_fold_mrrs']]})")
        print()

    # ── Recommendation ──
    print("=" * w)
    print("  RECOMMENDATION")
    print("=" * w)
    print()
    ranked = sorted(strategies.items(), key=lambda t: t[1]["mrr"], reverse=True)
    best_name, best = ranked[0]
    bi_entries = [(n, s) for n, s in strategies.items() if n.startswith("bi:") and "+" not in n]
    best_bi_entry = max(bi_entries, key=lambda t: t[1]["mrr"]) if bi_entries else (best_name, best)
    bi_label, bi_s = best_bi_entry
    cheapest_good = None
    for name, s in ranked:
        if s["mrr"] >= bi_s["mrr"] and s["cost"]["models_loaded"] <= 1:
            cheapest_good = (name, s)
            break

    print(f"  Best quality:  {best_name} (MRR={best['mrr']:.4f})")
    if cheapest_good:
        cn, cs = cheapest_good
        print(f"  Best value:    {cn} (MRR={cs['mrr']:.4f}, {cs['cost']['models_loaded']} model(s))")
    print(f"  Best bi-enc:   {bi_label} (MRR={bi_s['mrr']:.4f})")
    print(f"  Naive:         stub TF-IDF (MRR={stub_mrr:.4f})")
    print()

    # ── Overfitting check ──
    has_cv = any("cv" in s for s in strategies.values())
    if has_cv:
        print("=" * w)
        print("  OVERFITTING CHECK (k-fold cross-validation)")
        print("=" * w)
        print()
        print(f"  {'Ensemble':<24} {'Full MRR':>9} {'CV MRR':>8} {'Train MRR':>10} {'Gap':>7}  {'Verdict'}")
        print(f"  {'-'*24} {'-'*9} {'-'*8} {'-'*10} {'-'*7}  {'-'*20}")
        for name in strategy_order:
            s = strategies[name]
            if "cv" not in s:
                continue
            cv = s["cv"]
            gap = cv["overfit_gap"]
            if gap > 0.10:
                verdict = "OVERFITTING"
            elif gap > 0.05:
                verdict = "MILD OVERFIT"
            else:
                verdict = "OK"
            print(f"  {name:<24} {cv['full_data_mrr']:>9.3f} {cv['cv_mrr']:>8.3f} "
                  f"{cv['train_mrr']:>10.3f} {gap:>+7.3f}  {verdict}")
        print()


def _estimate_cost_per_1k(cost: dict) -> str:
    """Estimate $/1K queries based on GKE e2-standard-2 pricing (~$0.067/hr CPU)."""
    ms_per_query = cost["latency_per_call_ms"]
    cpu_hours_per_1k = (ms_per_query * 1000) / (1000 * 60 * 60)
    dollars = cpu_hours_per_1k * 0.067
    if dollars < 0.001:
        return "<$0.01"
    return f"${dollars:.3f}"


def run_compare(model_names: list[str] | None = None,
                domain_json: str | None = None, ood_json: str | None = None):
    """Run domain and cross-domain evaluations side by side.

    If domain_json / ood_json paths are provided, loads pre-computed results
    from those files instead of re-running (for parallel subagent workflow).
    """
    model_names = model_names or list(MODELS.keys())

    if domain_json and ood_json:
        domain_data = json.loads(Path(domain_json).read_text())
        ood_data = json.loads(Path(ood_json).read_text())
        domain = domain_data["strategies"]
        ood = ood_data["strategies"]
        shared = [n for n in domain_data["strategy_order"] if n in ood]
    else:
        print("Running domain evaluation...")
        print()
        domain_pairs = load_pairs()
        domain = run(sources=["domain"],
                     source_label=f"Domain ({len(domain_pairs)} hand-curated pairs)",
                     model_names=model_names)

        print()
        print()
        print("Running cross-domain evaluation (CQADupStack)...")
        print()
        ood = run(sources=["cqadupstack"], source_label="Cross-Domain (CQADupStack)",
                  model_names=model_names)

        if not domain or not ood:
            return
        shared = [n for n in domain if n in ood]

    w = 130
    print()
    print("=" * w)
    print("  SIDE-BY-SIDE: Domain vs Cross-Domain (Generalization Gap)")
    print("=" * w)
    print()
    print(f"  {'Strategy':<30} {'Domain':>8} {'OOD':>8} {'Gap':>8}  "
          f"{'Domain':>8} {'OOD':>8} {'Gap':>8}  "
          f"{'Domain':>8} {'OOD':>8} {'Gap':>8}")
    print(f"  {'':30} {'MRR':>8} {'MRR':>8} {'':>8}  "
          f"{'P@1':>8} {'P@1':>8} {'':>8}  "
          f"{'Prsn r':>8} {'Prsn r':>8} {'':>8}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8}  "
          f"{'-'*8} {'-'*8} {'-'*8}  "
          f"{'-'*8} {'-'*8} {'-'*8}")

    for name in shared:
        d, o = domain[name], ood[name]
        mrr_gap = o["mrr"] - d["mrr"]
        p1_gap = o["p1"] - d["p1"]
        prsn_gap = o["pearson"] - d["pearson"]
        print(f"  {name:<30} "
              f"{d['mrr']:>8.3f} {o['mrr']:>8.3f} {mrr_gap:>+8.3f}  "
              f"{d['p1']:>8.3f} {o['p1']:>8.3f} {p1_gap:>+8.3f}  "
              f"{d['pearson']:>8.3f} {o['pearson']:>8.3f} {prsn_gap:>+8.3f}")

    print()
    print("  Negative gaps = performance drops on out-of-domain data.")
    print("  Large negative gaps on ensembles (but not components) suggest overfitting to domain data.")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare scoring strategies")
    parser.add_argument(
        "--sources", nargs="+", default=["domain"],
        choices=list(VALID_SOURCES),
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Run domain and cross-domain (CQADupStack) side by side",
    )
    parser.add_argument(
        "--models", nargs="+", default=list(MODELS.keys()),
        choices=list(MODELS.keys()),
        help="Bi-encoder models to compare (default: all)",
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Save results to JSON file (for parallel subagent workflow)",
    )
    parser.add_argument(
        "--compare-from", nargs=2, metavar=("DOMAIN_JSON", "OOD_JSON"),
        help="Build side-by-side table from pre-computed JSON files",
    )
    parser.add_argument(
        "--test-set", action="store_true",
        help="Evaluate on held-out test set (eval/data/training_pairs_test.json) "
             "instead of curated pairs. Generates positive+negative pairs (~608 total).",
    )
    parser.add_argument(
        "--val-set", action="store_true",
        help="Evaluate on validation set (eval/data/training_pairs_val.json) "
             "with generated negatives.",
    )
    parser.add_argument(
        "--val-screen", action="store_true",
        help="Like --val-set but uses only a 5%% random sample (seed=42) for fast screening.",
    )
    parser.add_argument(
        "--max-combo-size", type=int, default=0,
        help="Cap ensemble combination size (0 = no limit, evaluates all sizes).",
    )
    args = parser.parse_args()

    if args.compare_from:
        run_compare(model_names=args.models,
                    domain_json=args.compare_from[0],
                    ood_json=args.compare_from[1])
    elif args.compare:
        run_compare(model_names=args.models)
    elif args.val_set or args.val_screen:
        val_path = Path(__file__).resolve().parent / "data" / "training_pairs_val.json"
        tp = load_test_pairs(path=val_path, seed=42)
        if args.val_screen:
            # 5% sample of the positive pairs (seed=42), then regenerate negatives
            with open(val_path) as f:
                raw = json.load(f)
            rng = random.Random(42)
            sample_n = max(1, int(len(raw) * 0.05))
            sampled = rng.sample(raw, sample_n)
            tp = []
            for entry in sampled:
                tp.append({"query": entry["anchor"], "document": entry["positive"], "relevance": 5})
            for i, entry in enumerate(sampled):
                j = (i + 1) % len(sampled)
                tp.append({"query": entry["anchor"], "document": sampled[j]["positive"], "relevance": 1})
        n_pos = sum(1 for p in tp if p["relevance"] == 5)
        n_neg = sum(1 for p in tp if p["relevance"] == 1)
        label_kind = "VAL SCREEN 5%" if args.val_screen else "VAL SET"
        run(model_names=args.models, save_path=args.save,
            test_pairs=tp, max_combo_size=args.max_combo_size,
            source_label=f"{label_kind} ({n_pos} positive + {n_neg} negative = {len(tp)} pairs)")
    elif args.test_set:
        tp = load_test_pairs()
        n_pos = sum(1 for p in tp if p["relevance"] == 5)
        n_neg = sum(1 for p in tp if p["relevance"] == 1)
        run(model_names=args.models, save_path=args.save,
            test_pairs=tp, max_combo_size=args.max_combo_size,
            source_label=f"Held-out TEST SET ({n_pos} positive + {n_neg} negative = {len(tp)} pairs)")
    else:
        run(sources=args.sources, model_names=args.models, save_path=args.save,
            max_combo_size=args.max_combo_size)
