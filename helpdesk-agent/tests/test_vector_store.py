"""Tests for the vector storage layer (services.vector_store)."""

import math
import sqlite3
import struct

import numpy as np
import pytest

from db import init_db, get_db_conn, DB_PATH
from services.vector_store import store_embedding, query_similar, _pack, _unpack


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    """Create an isolated in-memory-like DB for each test."""
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr("db.DB_PATH", db_path)
    monkeypatch.setattr("services.vector_store.get_db_conn", None)
    monkeypatch.setattr("services.vector_store.get_db", None)

    import db as db_mod
    import services.vector_store as vs_mod

    db_mod.DB_PATH = db_path
    vs_mod.get_db_conn = db_mod.get_db_conn
    vs_mod.get_db = db_mod.get_db

    init_db()

    with get_db_conn() as conn:
        conn.execute(
            "INSERT INTO tickets (ticket_key, summary, description) VALUES (?, ?, ?)",
            ("T-1", "Password reset", "User cannot reset password"),
        )
        conn.execute(
            "INSERT INTO tickets (ticket_key, summary, description) VALUES (?, ?, ?)",
            ("T-2", "SSO login failure", "SAML SSO not working after migration"),
        )
        conn.execute(
            "INSERT INTO tickets (ticket_key, summary, description) VALUES (?, ?, ?)",
            ("T-3", "Billing inquiry", "Need invoice for Q4"),
        )
        conn.execute(
            "INSERT INTO kb_articles (page_id, title, body_text) VALUES (?, ?, ?)",
            ("KB-1", "How to reset your password", "Step-by-step password reset guide"),
        )
        conn.execute(
            "INSERT INTO generated_articles (article_topic, title, body_html) VALUES (?, ?, ?)",
            ("password-reset", "Password Reset Guide", "<p>Reset your password here</p>"),
        )

    yield


def _make_vec(seed: int, dim: int = 384) -> list[float]:
    """Create a deterministic unit-normalized vector from a seed."""
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v.tolist()


class TestPackUnpack:
    def test_round_trip(self):
        vec = _make_vec(42)
        blob = _pack(vec)
        recovered = _unpack(blob)
        np.testing.assert_allclose(recovered, vec, atol=1e-7)

    def test_blob_size(self):
        vec = _make_vec(1)
        blob = _pack(vec)
        assert len(blob) == 384 * 4  # float32 = 4 bytes each


class TestStoreEmbedding:
    def test_stores_to_ticket(self):
        vec = _make_vec(10)
        store_embedding("T-1", vec, "ticket")

        with get_db_conn() as conn:
            row = conn.execute(
                "SELECT embedding FROM tickets WHERE ticket_key = 'T-1'"
            ).fetchone()
        assert row["embedding"] is not None
        recovered = _unpack(row["embedding"])
        np.testing.assert_allclose(recovered, vec, atol=1e-7)

    def test_stores_to_kb_article(self):
        vec = _make_vec(20)
        store_embedding("KB-1", vec, "kb_article")

        with get_db_conn() as conn:
            row = conn.execute(
                "SELECT embedding FROM kb_articles WHERE page_id = 'KB-1'"
            ).fetchone()
        assert row["embedding"] is not None

    def test_stores_to_article(self):
        vec = _make_vec(30)
        store_embedding("password-reset", vec, "article")

        with get_db_conn() as conn:
            row = conn.execute(
                "SELECT embedding FROM generated_articles WHERE article_topic = 'password-reset'"
            ).fetchone()
        assert row["embedding"] is not None

    def test_invalid_entity_type_raises(self):
        with pytest.raises(ValueError, match="Unknown entity_type"):
            store_embedding("X-1", _make_vec(1), "invalid")

    def test_overwrite_existing_embedding(self):
        v1 = _make_vec(100)
        v2 = _make_vec(200)
        store_embedding("T-1", v1, "ticket")
        store_embedding("T-1", v2, "ticket")

        with get_db_conn() as conn:
            row = conn.execute(
                "SELECT embedding FROM tickets WHERE ticket_key = 'T-1'"
            ).fetchone()
        recovered = _unpack(row["embedding"])
        np.testing.assert_allclose(recovered, v2, atol=1e-7)


class TestQuerySimilar:
    def test_returns_sorted_by_similarity(self):
        base = _make_vec(50)
        similar = np.array(base, dtype=np.float32)
        similar[:10] += 0.01
        similar /= np.linalg.norm(similar)
        different = _make_vec(999)

        store_embedding("T-1", base, "ticket")
        store_embedding("T-2", similar.tolist(), "ticket")
        store_embedding("T-3", different, "ticket")

        results = query_similar(base, k=10, entity_type="ticket")
        assert len(results) == 3
        assert results[0][0] == "T-1"
        assert results[0][1] > results[1][1] > results[2][1]

    def test_self_similarity_near_one(self):
        vec = _make_vec(60)
        store_embedding("T-1", vec, "ticket")

        results = query_similar(vec, k=1, entity_type="ticket")
        assert len(results) == 1
        assert results[0][0] == "T-1"
        assert abs(results[0][1] - 1.0) < 1e-5

    def test_k_limits_results(self):
        for i, key in enumerate(["T-1", "T-2", "T-3"]):
            store_embedding(key, _make_vec(i), "ticket")

        results = query_similar(_make_vec(0), k=2, entity_type="ticket")
        assert len(results) == 2

    def test_empty_table_returns_empty(self):
        results = query_similar(_make_vec(1), k=5, entity_type="ticket")
        assert results == []

    def test_query_kb_articles(self):
        vec = _make_vec(70)
        store_embedding("KB-1", vec, "kb_article")

        results = query_similar(vec, k=5, entity_type="kb_article")
        assert len(results) == 1
        assert results[0][0] == "KB-1"

    def test_invalid_entity_type_raises(self):
        with pytest.raises(ValueError, match="Unknown entity_type"):
            query_similar(_make_vec(1), entity_type="invalid")

    def test_skips_null_embeddings(self):
        store_embedding("T-1", _make_vec(1), "ticket")
        # T-2 and T-3 have no embeddings

        results = query_similar(_make_vec(1), k=10, entity_type="ticket")
        assert len(results) == 1
        assert results[0][0] == "T-1"


class TestBackfillScript:
    def test_backfill_tickets(self, monkeypatch):
        call_log = []

        def mock_embed_batch(texts, task_type="query"):
            call_log.append(texts)
            return [_make_vec(i) for i in range(len(texts))]

        monkeypatch.setattr("scripts.backfill_vectors.embed_batch", mock_embed_batch)
        from scripts.backfill_vectors import backfill

        count = backfill("ticket")
        assert count == 3
        assert len(call_log) == 1  # all 3 in one batch (< BATCH_SIZE)

        results = query_similar(_make_vec(0), k=10, entity_type="ticket")
        assert len(results) == 3

    def test_backfill_skips_already_embedded(self, monkeypatch):
        store_embedding("T-1", _make_vec(99), "ticket")

        def mock_embed_batch(texts, task_type="query"):
            return [_make_vec(i) for i in range(len(texts))]

        monkeypatch.setattr("scripts.backfill_vectors.embed_batch", mock_embed_batch)
        from scripts.backfill_vectors import backfill

        count = backfill("ticket")
        assert count == 2  # T-2 and T-3 only
