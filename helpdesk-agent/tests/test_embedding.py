"""Tests for the shared embedding service (services.embedding)."""

import math

import pytest


class TestEmbedText:
    def test_returns_384_dim_list(self):
        from services.embedding import embed_text

        vec = embed_text("How do I migrate to Jira Cloud?")
        assert isinstance(vec, list)
        assert len(vec) == 384
        assert all(isinstance(v, float) for v in vec)

    def test_normalized_to_unit_vector(self):
        from services.embedding import embed_text

        vec = embed_text("Configure SAML SSO for your organization")
        magnitude = math.sqrt(sum(v * v for v in vec))
        assert abs(magnitude - 1.0) < 1e-5

    def test_deterministic(self):
        from services.embedding import embed_text

        a = embed_text("billing setup")
        b = embed_text("billing setup")
        assert a == b

    def test_different_texts_different_vectors(self):
        from services.embedding import embed_text

        a = embed_text("How to configure SAML SSO")
        b = embed_text("Billing and payment methods")
        assert a != b

    def test_task_type_accepted(self):
        from services.embedding import embed_text

        vec_q = embed_text("test query", task_type="query")
        vec_d = embed_text("test query", task_type="document")
        assert len(vec_q) == 384
        assert len(vec_d) == 384


class TestEmbedBatch:
    def test_batch_matches_single(self):
        from services.embedding import embed_batch, embed_text

        texts = ["alpha query", "beta query"]
        batch_result = embed_batch(texts)
        single_results = [embed_text(t) for t in texts]

        assert len(batch_result) == 2
        for batch_vec, single_vec in zip(batch_result, single_results):
            assert len(batch_vec) == 384
            for bv, sv in zip(batch_vec, single_vec):
                assert abs(bv - sv) < 1e-5

    def test_empty_list_returns_empty(self):
        from services.embedding import embed_batch

        assert embed_batch([]) == []

    def test_batch_normalization(self):
        from services.embedding import embed_batch

        vecs = embed_batch(["one", "two", "three"])
        for vec in vecs:
            mag = math.sqrt(sum(v * v for v in vec))
            assert abs(mag - 1.0) < 1e-5

    def test_large_batch_chunking(self):
        from services.embedding import embed_batch

        texts = [f"synthetic query number {i}" for i in range(300)]
        vecs = embed_batch(texts)
        assert len(vecs) == 300
        assert all(len(v) == 384 for v in vecs)


class TestSingletonModel:
    def test_model_loaded_once(self):
        import services.embedding as emb

        emb._model = None
        emb.embed_text("first call loads model")
        model_ref = emb._model
        assert model_ref is not None

        emb.embed_text("second call reuses model")
        assert emb._model is model_ref

    def test_env_var_override(self, monkeypatch):
        import services.embedding as emb

        emb._model = None
        monkeypatch.setattr(emb, "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        vec = emb.embed_text("test override")
        assert len(vec) == 384
