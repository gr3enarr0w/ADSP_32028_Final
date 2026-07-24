"""Tests for the responder dense embedding retrieval index."""

from __future__ import annotations

import hashlib
import logging
import math
import random
import threading

import pytest

import db as db_mod
from db import get_db_conn, init_db
from plugins.responder import dense_retrieval
from plugins.responder.tests.test_bm25 import _seed_corpus


def _deterministic_vector(text: str, dim: int = 384) -> list[float]:
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)
    vals = [rng.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(v * v for v in vals))
    return [v / norm for v in vals]


def _mock_embed_text(text: str, task_type: str = "query") -> list[float]:
    del task_type
    return _deterministic_vector(text)


def _mock_embed_batch(texts: list[str], task_type: str = "query") -> list[list[float]]:
    return [_mock_embed_text(text, task_type=task_type) for text in texts]


@pytest.fixture(autouse=True)
def fresh_db_and_mocks(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr(db_mod, "DB_PATH", db_path)
    monkeypatch.setattr(dense_retrieval, "_RETRIEVER", dense_retrieval.DenseRetriever())
    monkeypatch.setattr(dense_retrieval, "embed_text", _mock_embed_text)
    monkeypatch.setattr(dense_retrieval, "embed_batch", _mock_embed_batch)
    init_db()
    _seed_corpus()
    yield


def test_build_logs_source_counts(caplog):
    caplog.set_level(logging.INFO)
    dense_retrieval.build()
    assert any("faq_sources=3" in record.message for record in caplog.records)
    assert any("kb_articles=3" in record.message for record in caplog.records)
    assert any("tickets=3" in record.message for record in caplog.records)
    assert any("atlassian_docs=3" in record.message for record in caplog.records)


def test_build_persists_embeddings():
    dense_retrieval.build()
    with get_db_conn() as conn:
        count = conn.execute(
            "SELECT COUNT(*) AS n FROM responder_corpus_embeddings WHERE embedding IS NOT NULL"
        ).fetchone()["n"]
    assert count == 12


def test_exact_document_text_ranks_first():
    dense_retrieval.build()
    doc = dense_retrieval.get_retriever()._docs[0]
    results = dense_retrieval.search(doc.text, k=1)
    assert results[0]["doc_id"] == doc.doc_id
    assert results[0]["source_type"] == doc.source_type
    assert results[0]["score"] > 0.99


def test_search_result_shape():
    dense_retrieval.build()
    results = dense_retrieval.search("oauth", k=3)
    assert len(results) == 3
    for hit in results:
        assert {"doc_id", "source_type", "text", "score"} <= set(hit.keys())
        assert hit["score"] > 0


def test_search_empty_query_returns_empty():
    dense_retrieval.build()
    assert dense_retrieval.search("") == []
    assert dense_retrieval.search("   ") == []


def test_search_non_positive_k_returns_empty():
    dense_retrieval.build()
    assert dense_retrieval.search("oauth", k=0) == []
    assert dense_retrieval.search("oauth", k=-1) == []


def test_search_uses_vector_store_query(monkeypatch):
    dense_retrieval.build()
    called = {"value": False}
    real_query_similar = dense_retrieval.query_similar

    def _spy(vector, k=5, entity_type="ticket"):
        called["value"] = True
        return real_query_similar(vector, k=k, entity_type=entity_type)

    monkeypatch.setattr(dense_retrieval, "query_similar", _spy)
    results = dense_retrieval.search("OAuth 2LO setup guide", k=3)
    assert called["value"]
    assert results


def test_responder_register_builds_index(monkeypatch):
    from plugins.responder import ResponderPlugin

    monkeypatch.setattr(dense_retrieval, "embed_text", _mock_embed_text)
    monkeypatch.setattr(dense_retrieval, "embed_batch", _mock_embed_batch)

    assert not dense_retrieval.get_retriever()._built
    ResponderPlugin().register(None, {})
    assert dense_retrieval.get_retriever()._built
    assert len(dense_retrieval.search("OAuth 2LO setup guide", k=1)) == 1


def test_add_document_updates_search_results():
    text = (
        "Escalate Jira Service Management SLA breach notifications "
        "for service desk queues."
    )
    dense_retrieval.build()
    dense_retrieval.add_document("KB-NEW", "kb_articles", text)

    results = dense_retrieval.search(text, k=1)
    assert results[0]["doc_id"] == "KB-NEW"
    assert results[0]["source_type"] == "kb_articles"


def test_add_document_upsert_no_duplicate():
    dense_retrieval.build()
    index = dense_retrieval.get_retriever()

    dense_retrieval.add_document("KB-NEW", "kb_articles", "First version about oauth.")
    count_after_first = len(index._docs)

    dense_retrieval.add_document(
        "KB-NEW",
        "kb_articles",
        "Second version mentioning token refresh instead.",
    )
    count_after_second = len(index._docs)

    assert count_after_second == count_after_first
    results = dense_retrieval.search(
        "Second version mentioning token refresh instead.", k=3
    )
    assert results
    assert results[0]["doc_id"] == "KB-NEW"
    assert "token refresh" in results[0]["text"]


def test_search_concurrent_thread_safety():
    dense_retrieval.build()

    errors: list[Exception] = []
    results_per_thread: list[list[dict]] = [[] for _ in range(4)]

    def search_worker(idx: int) -> None:
        try:
            results_per_thread[idx] = dense_retrieval.search("oauth sso", k=5)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=search_worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    for results in results_per_thread:
        assert results
