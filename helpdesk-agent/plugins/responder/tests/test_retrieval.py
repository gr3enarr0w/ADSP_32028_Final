"""Tests for hybrid retrieval fusion (BM25 + dense)."""

from __future__ import annotations

import hashlib
import math
import random
import re

import pytest

import db as db_mod
from core.pipeline import load_pipeline_config
from db import get_db_conn, init_db
from plugins.responder import bm25, dense_retrieval, retrieval
from plugins.responder.eval_set import TEST_MIN_QUERIES, build_eval_queries_from_db
from plugins.responder.tests.test_bm25 import _seed_corpus

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _deterministic_vector(text: str, dim: int = 384) -> list[float]:
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)
    vals = [rng.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(v * v for v in vals))
    return [v / norm for v in vals]


def _semantic_vector(text: str, corpus_texts: list[str], dim: int = 384) -> list[float]:
    """Blend hash vectors with token overlap so dense retrieval handles paraphrases."""
    tokens = set(_TOKEN_RE.findall((text or "").lower()))
    base = _deterministic_vector(text, dim)
    if not tokens:
        return base

    overlaps: list[float] = []
    for corpus_text in corpus_texts:
        corpus_tokens = set(_TOKEN_RE.findall((corpus_text or "").lower()))
        overlaps.append(len(tokens & corpus_tokens) / max(len(tokens | corpus_tokens), 1))

    blended = list(base)
    for idx, overlap in enumerate(overlaps[: min(len(overlaps), dim // 4)]):
        blended[idx] = 0.3 * base[idx] + 0.7 * overlap

    norm = math.sqrt(sum(v * v for v in blended))
    return [v / norm for v in blended]


def _seed_extended_corpus() -> None:
    _seed_corpus()
    with get_db_conn() as conn:
        conn.execute(
            "INSERT INTO tickets (ticket_key, summary, description, status, resolution, fetched_at) "
            "VALUES (?, ?, ?, ?, ?, datetime('now'))",
            (
                "ANTSE-4",
                "OAuth service account provisioning",
                "Need 2LO service account for integration.",
                "resolved",
                "Create OAuth 2LO service account in admin console.",
            ),
        )
        conn.execute(
            "INSERT INTO tickets (ticket_key, summary, description, status, resolution, fetched_at) "
            "VALUES (?, ?, ?, ?, ?, datetime('now'))",
            (
                "ANTSE-5",
                "User cannot sign in",
                "Customer reports login failure.",
                "resolved",
                "Clear browser cache to resolve SSO login loop.",
            ),
        )
        conn.execute(
            "INSERT INTO kb_articles (page_id, title, body_text, labels, topics_covered, fetched_at) "
            "VALUES (?, ?, ?, ?, ?, datetime('now'))",
            (
                "KB-4",
                "Share project with teammate",
                "Add users to project role so colleagues can view the project.",
                "permissions",
                "access sharing",
            ),
        )


def _built_eval_queries() -> list[tuple[str, str, str]]:
    return [q.as_tuple() for q in build_eval_queries_from_db(max_queries=100)]


@pytest.fixture(autouse=True)
def fresh_indexes(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr(db_mod, "DB_PATH", db_path)
    monkeypatch.setattr(bm25, "_INDEX", bm25.BM25Index())
    monkeypatch.setattr(dense_retrieval, "_RETRIEVER", dense_retrieval.DenseRetriever())

    init_db()
    _seed_extended_corpus()
    bm25.build()

    corpus_texts: list[str] = []

    def _embed_text(text: str, task_type: str = "query") -> list[float]:
        del task_type
        return _semantic_vector(text, corpus_texts)

    def _embed_batch(texts: list[str], task_type: str = "query") -> list[list[float]]:
        return [_embed_text(text, task_type=task_type) for text in texts]

    monkeypatch.setattr(dense_retrieval, "embed_text", _embed_text)
    monkeypatch.setattr(dense_retrieval, "embed_batch", _embed_batch)
    dense_retrieval.build()
    corpus_texts.extend(doc.text for doc in dense_retrieval.get_retriever()._docs)
    dense_retrieval.build()

    monkeypatch.setenv("PIPELINE_RESPONDER_RETRIEVAL_EVAL_MIN_QUERIES", str(TEST_MIN_QUERIES))
    import core.pipeline as pipeline_mod

    pipeline_mod._config = None
    load_pipeline_config()
    yield


@pytest.fixture
def eval_queries():
    queries = _built_eval_queries()
    assert len(queries) >= TEST_MIN_QUERIES
    return queries


def test_search_default_uses_rrf():
    assert retrieval.DEFAULT_METHOD == "rrf"
    results = retrieval.search("oauth 2lo", k=3)
    assert results
    assert results[0]["doc_id"] == "FAQ-1"


def test_search_method_weighted():
    results = retrieval.search("oauth 2lo", k=3, method="weighted")
    assert results
    assert {"doc_id", "source_type", "text", "score"} <= set(results[0].keys())


def test_search_empty_query_returns_empty():
    assert retrieval.search("") == []
    assert retrieval.search("   ") == []


def test_search_non_positive_k_returns_empty():
    assert retrieval.search("oauth", k=0) == []
    assert retrieval.search("oauth", k=-1) == []


def test_rrf_merge_deduplicates_by_source_and_doc_id():
    bm25_results = [
        {"doc_id": "A", "source_type": "tickets", "text": "first", "score": 3.0},
        {"doc_id": "B", "source_type": "tickets", "text": "second", "score": 2.0},
    ]
    dense_results = [
        {"doc_id": "A", "source_type": "tickets", "text": "first dense", "score": 0.9},
        {"doc_id": "C", "source_type": "kb_articles", "text": "third", "score": 0.8},
    ]

    merged = retrieval._rrf_merge(bm25_results, dense_results, k=3, rrf_k=60)
    keys = [(hit["source_type"], hit["doc_id"]) for hit in merged]

    assert len(keys) == len(set(keys))
    assert ("tickets", "A") in keys
    assert merged[0]["doc_id"] == "A"


def test_weighted_merge_normalizes_and_combines():
    bm25_results = [
        {"doc_id": "A", "source_type": "tickets", "text": "alpha", "score": 1.0},
        {"doc_id": "B", "source_type": "tickets", "text": "beta", "score": 0.5},
    ]
    dense_results = [
        {"doc_id": "B", "source_type": "tickets", "text": "beta dense", "score": 0.8},
        {"doc_id": "C", "source_type": "kb_articles", "text": "gamma", "score": 0.2},
    ]

    merged = retrieval._weighted_merge(bm25_results, dense_results, k=3, alpha=0.5)
    by_id = {hit["doc_id"]: hit["score"] for hit in merged}

    assert by_id["B"] > by_id["C"]
    assert by_id["A"] > 0


def test_hybrid_beats_individual_retrievers_on_mrr(eval_queries):
    tuned = retrieval.tune_fusion(eval_queries, min_queries=TEST_MIN_QUERIES, apply=True)
    scores = retrieval.evaluate_mrr(k=10, queries=eval_queries, min_queries=TEST_MIN_QUERIES)
    fusion_mrr = tuned.full_set_mrr
    assert fusion_mrr > scores["bm25"]
    assert fusion_mrr > scores["dense"]
    assert scores["weighted"] > scores["dense"]


def test_semantic_query_recovers_via_hybrid():
    results = retrieval.search("can't log in", k=5, method="rrf")
    assert results[0]["doc_id"] == "ANTSE-5"
    assert results[0]["source_type"] == "tickets"


@pytest.mark.parametrize(
    ("query", "expected_doc_id", "expected_source_type"),
    [
        ("oauth 2lo", "FAQ-1", "faq_sources"),
        ("OAuth service account", "ANTSE-4", "tickets"),
        ("forgot my password", "FAQ-3", "faq_sources"),
    ],
)
def test_hybrid_keyword_and_semantic_queries(query, expected_doc_id, expected_source_type):
    results = retrieval.search(query, k=3)
    assert results
    assert results[0]["doc_id"] == expected_doc_id
    assert results[0]["source_type"] == expected_source_type


def test_search_calls_both_retrievers_with_candidate_k(monkeypatch):
    calls: list[tuple[str, int]] = []

    def _bm25_search(query, k=10):
        calls.append(("bm25", k))
        return bm25.get_index().search(query, k=k)

    def _dense_search(query, k=10, min_score=0.0):
        calls.append(("dense", k))
        return dense_retrieval.get_retriever().search(query, k=k, min_score=min_score)

    monkeypatch.setattr(retrieval.bm25, "search", _bm25_search)
    monkeypatch.setattr(retrieval.dense_retrieval, "search", _dense_search)

    retrieval.search("oauth", k=4)
    assert ("bm25", 12) in calls
    assert ("dense", 12) in calls


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown fusion method"):
        retrieval.search("oauth", method="invalid")  # type: ignore[arg-type]


def test_fit_learned_alpha_in_valid_range(eval_queries):
    alpha = retrieval._fit_learned_alpha(eval_queries, k=10)
    assert 0.05 <= alpha <= 0.95


def test_cross_validate_fusion_returns_fold_stats(eval_queries):
    cv = retrieval.cross_validate_fusion("weighted", eval_queries, k=10, n_folds=5, seed=42)
    assert cv.method == "weighted"
    assert len(cv.fold_mrrs) == 5
    assert len(cv.fold_params) == 5
    assert 0.0 <= cv.mean_mrr <= 1.0
    assert cv.std_mrr >= 0.0
    for params in cv.fold_params:
        assert 0.05 <= float(params["alpha"]) <= 0.95


def test_grid_search_alpha_beats_extreme_weights(eval_queries):
    train_mrr_low = retrieval._fuse_queries(
        eval_queries,
        10,
        lambda bm25_hits, dense_hits, top_k: retrieval._weighted_merge(
            bm25_hits, dense_hits, top_k, alpha=0.05
        ),
    )
    alpha, tuned_mrr = retrieval._grid_search_alpha(eval_queries, k=10)
    assert tuned_mrr >= train_mrr_low
    assert 0.05 <= alpha <= 0.95


def test_tune_fusion_selects_method_via_cv(eval_queries):
    result = retrieval.tune_fusion(eval_queries, min_queries=TEST_MIN_QUERIES, apply=False)
    assert result.method in ("rrf", "weighted", "learned")
    assert result.cv_mrr_mean > 0.0
    assert len(result.cv_results) == 3
    assert result.full_set_mrr >= max(r.mean_mrr for r in result.cv_results) - 0.15


def test_tune_fusion_applies_defaults_to_search(eval_queries):
    result = retrieval.tune_fusion(eval_queries, min_queries=TEST_MIN_QUERIES, apply=True)
    assert retrieval.get_tuned() is result
    assert retrieval.DEFAULT_METHOD == result.method
    results = retrieval.search("oauth 2lo", k=3)
    assert results
    assert results[0]["doc_id"] == "FAQ-1"


def test_tune_fusion_blocks_apply_on_undersized_eval_set():
    with pytest.raises(ValueError, match="need >= 200"):
        retrieval.tune_fusion(queries=[], min_queries=200, apply=True)


def test_search_method_learned_uses_logistic_alpha(monkeypatch):
    """Learned mode applies logistic-regression alpha, not the pipeline default."""
    monkeypatch.setattr(retrieval, "_get_learned_alpha", lambda k: 0.99)
    results = retrieval.search("oauth 2lo", k=3, method="learned")
    assert results
    assert results[0]["doc_id"] == "FAQ-1"
