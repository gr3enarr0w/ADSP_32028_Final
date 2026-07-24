"""Hybrid retrieval fusion for responder lookups (BM25 + dense).

Eval set sizing (see plugins/responder/eval_set.py):
  - Test/dev:   25+ hand-seeded pairs (TEST_MIN_QUERIES)
  - Production: 200–500 auto-labeled pairs from resolved tickets + KB/FAQ/docs
  - Build:      python -m scripts.build_retrieval_eval

Tuning uses 5-fold cross-validation with inner grid search / logistic regression.
Folds are source-type stratified when class counts support it.
Re-run tune_fusion() after rebuilding the eval set or re-ingesting corpus data.

Fusion modes: search(..., method="rrf"|"weighted"|"learned").
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold

from core.pipeline import get_plugin_config

from . import bm25, dense_retrieval
from .eval_set import (
    PRODUCTION_MIN_QUERIES,
    PRODUCTION_TARGET_QUERIES,
    SEED_EVAL_QUERIES,
    TEST_MIN_QUERIES,
    get_eval_queries,
)

log = logging.getLogger(__name__)

FusionMethod = Literal["rrf", "weighted", "learned"]
DEFAULT_METHOD: FusionMethod = "rrf"

QueryExample = tuple[str, str, str]

# Populated by tune_fusion(); search() prefers these over raw pipeline defaults.
_TUNED: "TuningResult | None" = None
_LEARNED_ALPHA: float | None = None

# Backward-compatible alias; production code should call resolve_eval_queries().
EVAL_QUERIES: list[QueryExample] = list(SEED_EVAL_QUERIES)

ALPHA_GRID = tuple(round(v, 2) for v in np.linspace(0.05, 0.95, 19))
RRF_K_GRID = (10, 20, 30, 40, 60, 80)


def resolve_eval_queries(
    queries: list[QueryExample] | None = None,
    *,
    min_queries: int | None = None,
    max_queries: int | None = None,
) -> list[QueryExample]:
    """Load the eval set from disk or auto-build from the DB corpus."""
    if queries is not None:
        return list(queries)

    cfg = get_plugin_config("responder")
    eval_path = cfg.get("retrieval_eval_path")
    return get_eval_queries(
        min_queries=min_queries
        or int(cfg.get("retrieval_eval_min_queries", PRODUCTION_MIN_QUERIES)),
        max_queries=max_queries
        or int(cfg.get("retrieval_eval_max_queries", PRODUCTION_TARGET_QUERIES)),
        path=Path(eval_path) if eval_path else None,
        prefer_file=bool(cfg.get("retrieval_eval_prefer_file", True)),
    )


@dataclass(frozen=True, slots=True)
class CVResult:
    """Cross-validation outcome for one fusion strategy."""

    method: FusionMethod
    mean_mrr: float
    std_mrr: float
    fold_mrrs: tuple[float, ...]
    fold_params: tuple[dict[str, float | int], ...]


@dataclass(frozen=True, slots=True)
class TuningResult:
    """Best fusion strategy and params from nested cross-validation."""

    method: FusionMethod
    params: dict[str, float | int]
    cv_mrr_mean: float
    cv_mrr_std: float
    cv_results: tuple[CVResult, ...] = field(default_factory=tuple)
    full_set_mrr: float = 0.0


def _doc_key(hit: dict) -> tuple[str, str]:
    return (hit["source_type"], str(hit["doc_id"]))


def _normalize_scores(results: list[dict]) -> dict[tuple[str, str], float]:
    """Min-max normalize scores within a single retriever result list to [0, 1]."""
    if not results:
        return {}

    raw_scores = [float(hit["score"]) for hit in results]
    min_score = min(raw_scores)
    max_score = max(raw_scores)
    span = max_score - min_score

    normalized: dict[tuple[str, str], float] = {}
    for hit in results:
        key = _doc_key(hit)
        if span == 0:
            value = 1.0 if max_score > 0 else 0.0
        else:
            value = (float(hit["score"]) - min_score) / span
        normalized[key] = max(normalized.get(key, 0.0), value)
    return normalized


def _rrf_merge(
    bm25_results: list[dict],
    dense_results: list[dict],
    k: int,
    *,
    rrf_k: int = 60,
    bm25_weight: float = 1.0,
    dense_weight: float = 1.0,
) -> list[dict]:
    """Merge ranked lists with Reciprocal Rank Fusion."""
    fusion_scores: dict[tuple[str, str], float] = {}

    for rank, hit in enumerate(bm25_results, start=1):
        key = _doc_key(hit)
        fusion_scores[key] = fusion_scores.get(key, 0.0) + bm25_weight / (rrf_k + rank)

    for rank, hit in enumerate(dense_results, start=1):
        key = _doc_key(hit)
        fusion_scores[key] = fusion_scores.get(key, 0.0) + dense_weight / (rrf_k + rank)

    ranked_keys = sorted(fusion_scores, key=lambda key: fusion_scores[key], reverse=True)
    payloads = {_doc_key(hit): hit for hit in bm25_results + dense_results}
    merged: list[dict] = []
    for key in ranked_keys[:k]:
        payload = payloads[key]
        merged.append(
            {
                "doc_id": payload["doc_id"],
                "source_type": key[0],
                "text": payload["text"],
                "score": fusion_scores[key],
            }
        )
    return merged


def _weighted_merge(
    bm25_results: list[dict],
    dense_results: list[dict],
    k: int,
    *,
    alpha: float = 0.5,
) -> list[dict]:
    """Merge ranked lists with normalized weighted linear combination."""
    bm25_norm = _normalize_scores(bm25_results)
    dense_norm = _normalize_scores(dense_results)
    all_keys = set(bm25_norm) | set(dense_norm)

    fusion_scores: dict[tuple[str, str], float] = {}
    for key in all_keys:
        fusion_scores[key] = alpha * bm25_norm.get(key, 0.0) + (1.0 - alpha) * dense_norm.get(
            key, 0.0
        )

    ranked_keys = sorted(fusion_scores, key=lambda key: fusion_scores[key], reverse=True)
    payloads = {_doc_key(hit): hit for hit in bm25_results + dense_results}
    merged: list[dict] = []
    for key in ranked_keys[:k]:
        payload = payloads[key]
        merged.append(
            {
                "doc_id": payload["doc_id"],
                "source_type": key[0],
                "text": payload["text"],
                "score": fusion_scores[key],
            }
        )
    return merged


def _reciprocal_rank(results: list[dict], expected: tuple[str, str]) -> float:
    for rank, hit in enumerate(results, start=1):
        if _doc_key(hit) == expected:
            return 1.0 / rank
    return 0.0


def _retrieve_candidates(query: str, k: int) -> tuple[list[dict], list[dict]]:
    candidate_k = k * 3
    return bm25.search(query, k=candidate_k), dense_retrieval.search(query, k=candidate_k)


def _fuse_queries(
    queries: list[QueryExample],
    k: int,
    fuse_fn: Callable[[list[dict], list[dict], int], list[dict]],
) -> float:
    if not queries:
        return 0.0
    total = 0.0
    for query, source_type, doc_id in queries:
        bm25_hits, dense_hits = _retrieve_candidates(query, k)
        results = fuse_fn(bm25_hits, dense_hits, k)
        total += _reciprocal_rank(results, (source_type, doc_id))
    return total / len(queries)


def _build_pointwise_training_data(
    queries: list[QueryExample],
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (bm25_norm, dense_norm) → relevant label pairs for weight learning."""
    features: list[list[float]] = []
    labels: list[int] = []
    for query, source_type, doc_id in queries:
        bm25_hits, dense_hits = _retrieve_candidates(query, k)
        bm25_norm = _normalize_scores(bm25_hits)
        dense_norm = _normalize_scores(dense_hits)
        expected = (source_type, doc_id)
        for key in set(bm25_norm) | set(dense_norm):
            features.append([bm25_norm.get(key, 0.0), dense_norm.get(key, 0.0)])
            labels.append(1 if key == expected else 0)
    return np.asarray(features, dtype=np.float64), np.asarray(labels, dtype=np.int32)


def _fit_learned_alpha(train_queries: list[QueryExample], k: int) -> float:
    """Learn BM25/dense blend via logistic regression on pointwise features."""
    x_train, y_train = _build_pointwise_training_data(train_queries, k)
    if len(np.unique(y_train)) < 2 or len(x_train) < 4:
        return 0.5

    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
    model.fit(x_train, y_train)
    coef = model.coef_[0]
    magnitude = abs(float(coef[0])) + abs(float(coef[1]))
    if magnitude == 0.0:
        return 0.5
    return float(np.clip(abs(float(coef[0])) / magnitude, 0.05, 0.95))


def _grid_search_alpha(train_queries: list[QueryExample], k: int) -> tuple[float, float]:
    best_alpha = 0.5
    best_mrr = -1.0
    for alpha in ALPHA_GRID:
        mrr = _fuse_queries(
            train_queries,
            k,
            lambda bm25_hits, dense_hits, top_k, a=alpha: _weighted_merge(
                bm25_hits, dense_hits, top_k, alpha=a
            ),
        )
        if mrr > best_mrr:
            best_mrr = mrr
            best_alpha = float(alpha)
    return best_alpha, best_mrr


def _grid_search_rrf_k(train_queries: list[QueryExample], k: int) -> tuple[int, float]:
    best_rrf_k = 60
    best_mrr = -1.0
    for rrf_k in RRF_K_GRID:
        mrr = _fuse_queries(
            train_queries,
            k,
            lambda bm25_hits, dense_hits, top_k, rk=rrf_k: _rrf_merge(
                bm25_hits, dense_hits, top_k, rrf_k=rk
            ),
        )
        if mrr > best_mrr:
            best_mrr = mrr
            best_rrf_k = rrf_k
    return best_rrf_k, best_mrr


def _fit_fold_params(
    method: FusionMethod,
    train_queries: list[QueryExample],
    k: int,
) -> dict[str, float | int]:
    if method == "weighted":
        alpha, _ = _grid_search_alpha(train_queries, k)
        return {"alpha": alpha}
    if method == "learned":
        return {"alpha": _fit_learned_alpha(train_queries, k)}
    rrf_k, _ = _grid_search_rrf_k(train_queries, k)
    return {"rrf_k": rrf_k}


def _eval_with_params(
    queries: list[QueryExample],
    k: int,
    method: FusionMethod,
    params: dict[str, float | int],
) -> float:
    if method == "weighted" or method == "learned":
        alpha = float(params["alpha"])
        return _fuse_queries(
            queries,
            k,
            lambda bm25_hits, dense_hits, top_k, a=alpha: _weighted_merge(
                bm25_hits, dense_hits, top_k, alpha=a
            ),
        )
    rrf_k = int(params["rrf_k"])
    return _fuse_queries(
        queries,
        k,
        lambda bm25_hits, dense_hits, top_k, rk=rrf_k: _rrf_merge(
            bm25_hits, dense_hits, top_k, rrf_k=rk
        ),
    )


def cross_validate_fusion(
    method: FusionMethod,
    queries: list[QueryExample] | None = None,
    *,
    k: int = 10,
    n_folds: int = 5,
    seed: int = 42,
) -> CVResult:
    """K-fold CV: tune on train folds, report MRR on held-out queries each fold."""
    eval_queries = resolve_eval_queries(queries)
    if len(eval_queries) < n_folds:
        raise ValueError(
            f"Need at least {n_folds} labeled queries for {n_folds}-fold CV, got {len(eval_queries)}"
        )

    indices = np.arange(len(eval_queries))
    labels = np.asarray([source_type for _, source_type, _ in eval_queries], dtype=object)
    label_counts = Counter(labels.tolist())
    use_stratified = bool(label_counts) and all(count >= n_folds for count in label_counts.values())

    if use_stratified:
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        split_iter = splitter.split(indices, labels)
    else:
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        split_iter = splitter.split(indices)

    fold_mrrs: list[float] = []
    fold_params: list[dict[str, float | int]] = []

    for _train_idx, val_idx in split_iter:
        train_queries = [eval_queries[i] for i in _train_idx]
        val_queries = [eval_queries[i] for i in val_idx]
        params = _fit_fold_params(method, train_queries, k)
        fold_params.append(params)
        fold_mrrs.append(_eval_with_params(val_queries, k, method, params))

    return CVResult(
        method=method,
        mean_mrr=float(np.mean(fold_mrrs)),
        std_mrr=float(np.std(fold_mrrs)),
        fold_mrrs=tuple(fold_mrrs),
        fold_params=tuple(fold_params),
    )


def _fit_full_params(
    method: FusionMethod,
    queries: list[QueryExample],
    k: int,
) -> dict[str, float | int]:
    return _fit_fold_params(method, queries, k)


def tune_fusion(
    queries: list[QueryExample] | None = None,
    *,
    k: int = 10,
    n_folds: int | None = None,
    seed: int | None = None,
    min_queries: int | None = None,
    max_queries: int | None = None,
    apply: bool = True,
) -> TuningResult:
    """Select fusion method and hyperparameters via cross-validation.

    Compares RRF (rrf_k grid search), weighted (alpha grid search), and learned
    (logistic regression alpha). Refits the winner on the full query set.
    """
    bm25.build()
    dense_retrieval.build()

    cfg = get_plugin_config("responder")
    if n_folds is None:
        n_folds = int(cfg.get("retrieval_tune_folds", 5))
    if seed is None:
        seed = int(cfg.get("retrieval_tune_seed", 42))
    required_min_queries = (
        min_queries
        if min_queries is not None
        else int(cfg.get("retrieval_eval_min_queries", PRODUCTION_MIN_QUERIES))
    )

    eval_queries = resolve_eval_queries(
        queries,
        min_queries=required_min_queries,
        max_queries=max_queries,
    )
    if len(eval_queries) < required_min_queries:
        message = (
            f"Only {len(eval_queries)} eval queries available; "
            f"need >= {required_min_queries} for production tuning."
        )
        if apply and not bool(cfg.get("retrieval_allow_undersized_tuning", False)):
            raise ValueError(
                f"{message} Build labels with `python -m scripts.build_retrieval_eval` "
                "or set responder.retrieval_allow_undersized_tuning=true explicitly."
            )
        log.warning("%s Proceeding because apply=%s.", message, apply)
    cv_results = tuple(
        cross_validate_fusion(method, eval_queries, k=k, n_folds=n_folds, seed=seed)
        for method in ("rrf", "weighted", "learned")
    )
    best_cv = max(cv_results, key=lambda result: result.mean_mrr)
    final_params = _fit_full_params(best_cv.method, eval_queries, k)
    full_mrr = _eval_with_params(eval_queries, k, best_cv.method, final_params)

    result = TuningResult(
        method=best_cv.method,
        params=final_params,
        cv_mrr_mean=best_cv.mean_mrr,
        cv_mrr_std=best_cv.std_mrr,
        cv_results=cv_results,
        full_set_mrr=full_mrr,
    )

    if apply:
        global _TUNED, DEFAULT_METHOD
        _TUNED = result
        DEFAULT_METHOD = result.method
        log.info(
            "[retrieval] tune_fusion selected method=%s params=%s cv_mrr=%.3f±%.3f full_mrr=%.3f",
            result.method,
            result.params,
            result.cv_mrr_mean,
            result.cv_mrr_std,
            result.full_set_mrr,
        )

    return result


def get_tuned() -> TuningResult | None:
    return _TUNED


def _get_learned_alpha(k: int, queries: list[QueryExample] | None = None) -> float:
    """Return logistic-regression alpha, cached after first fit on the eval set."""
    global _LEARNED_ALPHA
    if _TUNED and _TUNED.method == "learned":
        return float(_TUNED.params["alpha"])
    if _LEARNED_ALPHA is None:
        _LEARNED_ALPHA = _fit_learned_alpha(resolve_eval_queries(queries), k)
    return _LEARNED_ALPHA


def _resolve_fusion_config(
    method: FusionMethod | None,
    k: int,
) -> tuple[FusionMethod, dict[str, float | int]]:
    cfg = get_plugin_config("responder")
    fusion_method = method or (_TUNED.method if _TUNED else DEFAULT_METHOD)

    if _TUNED and method is None:
        params = dict(_TUNED.params)
    elif fusion_method == "learned":
        params = {"alpha": _get_learned_alpha(k)}
    elif fusion_method == "weighted":
        params = {"alpha": float(cfg.get("retrieval_alpha", 0.5))}
    else:
        params = {"rrf_k": int(cfg.get("rrf_k", 60))}

    return fusion_method, params


def evaluate_mrr(
    k: int = 10,
    queries: list[QueryExample] | None = None,
    *,
    min_queries: int | None = None,
    max_queries: int | None = None,
) -> dict[str, float]:
    """Compute MRR for each retrieval strategy over the eval query set."""
    eval_queries = resolve_eval_queries(
        queries,
        min_queries=min_queries,
        max_queries=max_queries,
    )
    totals = {"bm25": 0.0, "dense": 0.0, "rrf": 0.0, "weighted": 0.0, "learned": 0.0}
    n = len(eval_queries)
    if n == 0:
        return totals

    cfg = get_plugin_config("responder")
    rrf_k = int(cfg.get("rrf_k", 60))
    alpha = float(cfg.get("retrieval_alpha", 0.5))
    candidate_k = k * 3

    learned_alpha = _fit_learned_alpha(eval_queries, k)

    for query, source_type, doc_id in eval_queries:
        expected = (source_type, doc_id)
        bm25_hits = bm25.search(query, k=candidate_k)
        dense_hits = dense_retrieval.search(query, k=candidate_k)
        totals["bm25"] += _reciprocal_rank(bm25.search(query, k=k), expected)
        totals["dense"] += _reciprocal_rank(dense_retrieval.search(query, k=k), expected)
        totals["rrf"] += _reciprocal_rank(
            _rrf_merge(bm25_hits, dense_hits, k, rrf_k=rrf_k), expected
        )
        totals["weighted"] += _reciprocal_rank(
            _weighted_merge(bm25_hits, dense_hits, k, alpha=alpha), expected
        )
        totals["learned"] += _reciprocal_rank(
            _weighted_merge(bm25_hits, dense_hits, k, alpha=learned_alpha), expected
        )

    return {name: total / n for name, total in totals.items()}


def search(
    query: str,
    k: int = 10,
    *,
    method: FusionMethod | None = None,
) -> list[dict]:
    """Hybrid search over BM25 and dense retrievers."""
    if not query or not query.strip():
        return []
    if k <= 0:
        return []

    bm25_results, dense_results = _retrieve_candidates(query, k)
    fusion_method, params = _resolve_fusion_config(method, k)

    if fusion_method == "rrf":
        return _rrf_merge(bm25_results, dense_results, k, rrf_k=int(params["rrf_k"]))
    if fusion_method in ("weighted", "learned"):
        return _weighted_merge(bm25_results, dense_results, k, alpha=float(params["alpha"]))

    raise ValueError(f"Unknown fusion method: {fusion_method!r}")


__all__ = [
    "ALPHA_GRID",
    "CVResult",
    "DEFAULT_METHOD",
    "EVAL_QUERIES",
    "RRF_K_GRID",
    "TuningResult",
    "cross_validate_fusion",
    "evaluate_mrr",
    "get_tuned",
    "resolve_eval_queries",
    "search",
    "tune_fusion",
    "_fit_learned_alpha",
    "_rrf_merge",
    "_weighted_merge",
]
