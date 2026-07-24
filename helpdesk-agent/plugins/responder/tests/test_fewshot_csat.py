"""Tests for CSAT-weighted few-shot prioritization (ANTSE-325)."""

from __future__ import annotations

import logging

import pytest

from db import get_db_conn, init_db


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr("db.DB_PATH", db_path)
    init_db()

    from plugins.responder.ann_fewshot import ANNFewShotIndex

    ANNFewShotIndex._last_rebuild_date = None
    ANNFewShotIndex._csat_cache = None
    ANNFewShotIndex._csat_cache_date = None
    ANNFewShotIndex._global_csat_stats = None
    ANNFewShotIndex._csat_alpha = None
    yield


def _fake_embed_text(text: str, task_type: str = "query") -> list[float]:
    lowered = text.lower()
    if "password" in lowered:
        return [1.0, 0.0, 0.0]
    if "sso" in lowered or "access" in lowered:
        return [0.0, 1.0, 0.0]
    if "billing" in lowered:
        return [0.0, 0.0, 1.0]
    return [0.33, 0.33, 0.34]


def _fake_embed_batch(texts: list[str], task_type: str = "query") -> list[list[float]]:
    return [_fake_embed_text(text, task_type=task_type) for text in texts]


def _seed_csat_and_examples(conn):
    conn.execute(
        """
        INSERT INTO category_csat_correlations
            (category, run_date, mean_csat, std_csat, acceptance_rate, n_samples)
        VALUES
            ('access', '2026-06-01', 4.5, 0.5, 0.9, 50),
            ('billing', '2026-06-01', 2.0, 0.8, 0.4, 50),
            ('sparse', '2026-06-01', 3.0, 0.5, 0.5, 10)
        """
    )
    conn.execute(
        "INSERT INTO tickets (ticket_key, summary, description, status) "
        "VALUES ('T-1', 'Reset password access', 'User cannot reset password', 'Open')"
    )
    conn.execute(
        "INSERT INTO tickets (ticket_key, summary, description, status) "
        "VALUES ('T-2', 'SSO access issue', 'User cannot sign in with SSO', 'Open')"
    )
    conn.execute(
        "INSERT INTO tickets (ticket_key, summary, description, status) "
        "VALUES ('T-3', 'Billing question', 'Invoice mismatch on billing page', 'Open')"
    )
    conn.execute(
        "INSERT INTO ticket_classifications (ticket_key, category, issue_type, confidence) "
        "VALUES ('T-1', 'access', 'access-request', 0.9)"
    )
    conn.execute(
        "INSERT INTO ticket_classifications (ticket_key, category, issue_type, confidence) "
        "VALUES ('T-2', 'access', 'access-request', 0.9)"
    )
    conn.execute(
        "INSERT INTO ticket_classifications (ticket_key, category, issue_type, confidence) "
        "VALUES ('T-3', 'billing', 'how-to', 0.9)"
    )
    conn.execute(
        """
        INSERT INTO ai_draft_feedback
            (ticket_key, draft_comment_id, response_type,
             draft_customer_response, actual_response, agent_feedback)
        VALUES ('T-1', 'c1', 'self_service', 'Reset password.', 'Reset password.', 'both_good')
        """
    )
    conn.execute(
        """
        INSERT INTO ai_draft_feedback
            (ticket_key, draft_comment_id, response_type,
             draft_customer_response, actual_response, agent_feedback)
        VALUES ('T-2', 'c2', 'admin_action', 'Check SSO config.', 'Check SSO config.', 'both_good')
        """
    )
    conn.execute(
        """
        INSERT INTO ai_draft_feedback
            (ticket_key, draft_comment_id, response_type,
             draft_customer_response, actual_response, agent_feedback)
        VALUES ('T-3', 'c3', 'self_service', 'Review billing.', 'Review billing.', 'both_good')
        """
    )


class TestApplyWeight:
    def test_baseline_returns_ann_score(self):
        from plugins.responder.ann_fewshot import _apply_weight

        csat = {"mean_csat": 4.0, "acceptance_rate": 0.8, "n_samples": 100}
        global_stats = {"mean": 3.0, "std": 1.0, "max_csat": 5.0}
        assert _apply_weight(0.8, "baseline", csat, global_stats, 0.5) == pytest.approx(0.8)

    def test_strategy_a_boosts_high_csat(self):
        from plugins.responder.ann_fewshot import _apply_weight

        csat = {"mean_csat": 5.0, "acceptance_rate": 0.9, "n_samples": 100}
        global_stats = {"mean": 3.0, "std": 1.0, "max_csat": 5.0}
        weighted = _apply_weight(0.8, "a", csat, global_stats, 0.5)
        assert weighted > 0.8

    def test_strategy_a_penalizes_low_csat(self):
        from plugins.responder.ann_fewshot import _apply_weight

        csat = {"mean_csat": 1.0, "acceptance_rate": 0.2, "n_samples": 100}
        global_stats = {"mean": 3.0, "std": 1.0, "max_csat": 5.0}
        weighted = _apply_weight(0.8, "a", csat, global_stats, 0.5)
        assert weighted < 0.8

    def test_strategy_b_uses_log_gate(self):
        import math

        from plugins.responder.ann_fewshot import _apply_weight

        csat = {"mean_csat": 4.0, "acceptance_rate": 0.8, "n_samples": 100}
        global_stats = {"mean": 3.0, "std": 1.0, "max_csat": 5.0}
        expected = 0.8 * math.log(1.0 + 4.0 / 5.0)
        assert _apply_weight(0.8, "b", csat, global_stats, 0.5) == pytest.approx(expected)

    def test_strategy_c_composite(self):
        from plugins.responder.ann_fewshot import _apply_weight

        csat = {"mean_csat": 4.0, "acceptance_rate": 0.5, "n_samples": 100}
        global_stats = {"mean": 3.0, "std": 1.0, "max_csat": 5.0}
        expected = 0.8 * (0.5 * (4.0 / 5.0))
        assert _apply_weight(0.8, "c", csat, global_stats, 0.5) == pytest.approx(expected)

    def test_fallback_when_n_samples_low(self):
        from plugins.responder.ann_fewshot import _apply_weight

        csat = {"mean_csat": 5.0, "acceptance_rate": 0.9, "n_samples": 5}
        global_stats = {"mean": 3.0, "std": 1.0, "max_csat": 5.0}
        assert _apply_weight(0.8, "a", csat, global_stats, 0.5) == pytest.approx(0.8)

    def test_fallback_when_csat_missing(self):
        from plugins.responder.ann_fewshot import _apply_weight

        global_stats = {"mean": 3.0, "std": 1.0, "max_csat": 5.0}
        assert _apply_weight(0.8, "c", None, global_stats, 0.5) == pytest.approx(0.8)


class TestRetrieveWeighting:
    def test_strategy_a_prefers_high_csat_category(self, monkeypatch):
        from plugins.responder import ann_fewshot
        from plugins.responder.ann_fewshot import ANNFewShotIndex

        monkeypatch.setattr(ann_fewshot, "embed_text", _fake_embed_text)
        monkeypatch.setattr(ann_fewshot, "embed_batch", _fake_embed_batch)
        monkeypatch.setattr(
            ann_fewshot,
            "get_plugin_config",
            lambda _name: {"csat_weight": 0.5, "fewshot_csat_strategy": "a"},
        )

        with get_db_conn() as conn:
            _seed_csat_and_examples(conn)

        ANNFewShotIndex.build()

        weighted = ANNFewShotIndex.retrieve_with_categories(
            "password reset help needed",
            k=2,
            similarity_floor=0.5,
            strategy="a",
        )

        assert weighted[0]["final_score"] >= weighted[-1]["final_score"]

    def test_fallback_logs_once_per_retrieve(self, monkeypatch, caplog):
        from plugins.responder import ann_fewshot
        from plugins.responder.ann_fewshot import ANNFewShotIndex

        monkeypatch.setattr(ann_fewshot, "embed_text", _fake_embed_text)
        monkeypatch.setattr(ann_fewshot, "embed_batch", _fake_embed_batch)
        monkeypatch.setattr(
            ann_fewshot,
            "get_plugin_config",
            lambda _name: {"csat_weight": 0.5, "fewshot_csat_strategy": "a"},
        )

        with get_db_conn() as conn:
            conn.execute(
                """
                INSERT INTO category_csat_correlations
                    (category, run_date, mean_csat, std_csat, acceptance_rate, n_samples)
                VALUES ('unknown', '2026-06-01', 3.0, 0.5, 0.5, 5)
                """
            )
            conn.execute(
                "INSERT INTO tickets (ticket_key, summary, description, status) "
                "VALUES ('T-9', 'Password reset', 'Need password help', 'Open')"
            )
            conn.execute(
                """
                INSERT INTO ai_draft_feedback
                    (ticket_key, draft_comment_id, response_type,
                     draft_customer_response, actual_response, agent_feedback)
                VALUES ('T-9', 'c9', 'self_service', 'Reset.', 'Reset.', 'both_good')
                """
            )

        ANNFewShotIndex.build()

        with caplog.at_level(logging.INFO):
            ANNFewShotIndex.retrieve("password reset help", strategy="a")

        fallback_logs = [r for r in caplog.records if "CSAT weighting fallback" in r.message]
        assert len(fallback_logs) == 1

    def test_exclude_ticket_keys_prevents_leakage(self, monkeypatch):
        from plugins.responder import ann_fewshot
        from plugins.responder.ann_fewshot import ANNFewShotIndex

        monkeypatch.setattr(ann_fewshot, "embed_text", _fake_embed_text)
        monkeypatch.setattr(ann_fewshot, "embed_batch", _fake_embed_batch)

        with get_db_conn() as conn:
            _seed_csat_and_examples(conn)

        ANNFewShotIndex.build()

        results = ANNFewShotIndex.retrieve_with_categories(
            "password reset help needed",
            k=5,
            similarity_floor=0.5,
            exclude_ticket_keys={"T-1"},
        )
        ticket_keys = {item["ticket_key"] for item in results}
        assert "T-1" not in ticket_keys

    def test_malformed_csat_row_skipped_by_load_csat_weights(self):
        """_load_csat_weights must skip rows with unconvertible values (defence-in-depth
        against type coercion failures from Postgres drivers or future schema changes).
        The DB NOT NULL constraint prevents NULL insertion via normal paths, but the
        in-code guard still protects against unexpected driver behaviour.
        """
        from unittest.mock import MagicMock
        from plugins.responder.ann_fewshot import _load_csat_weights

        bad_row = MagicMock()
        bad_row.__getitem__ = lambda self, key: {
            "category": "access",
            "mean_csat": None,
            "std_csat": 0.5,
            "acceptance_rate": 0.8,
            "n_samples": 50,
        }[key]

        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = [bad_row]

        result = _load_csat_weights(mock_conn)
        assert "access" not in result

    def test_missing_csat_category_falls_back_without_crash(self, monkeypatch):
        """Retrieval must not crash when an example's category has no CSAT entry."""
        from plugins.responder import ann_fewshot
        from plugins.responder.ann_fewshot import ANNFewShotIndex

        monkeypatch.setattr(ann_fewshot, "embed_text", _fake_embed_text)
        monkeypatch.setattr(ann_fewshot, "embed_batch", _fake_embed_batch)

        with get_db_conn() as conn:
            conn.execute(
                "INSERT INTO tickets (ticket_key, summary, description, status) "
                "VALUES ('T-10', 'Password reset', 'Need password help', 'Open')"
            )
            conn.execute(
                """
                INSERT INTO ai_draft_feedback
                    (ticket_key, draft_comment_id, response_type,
                     draft_customer_response, actual_response, agent_feedback)
                VALUES ('T-10', 'c10', 'self_service', 'Reset.', 'Reset.', 'both_good')
                """
            )

        ANNFewShotIndex.build()
        results = ANNFewShotIndex.retrieve("password reset help", strategy="a")
        assert len(results) >= 1


class TestCategoryDistribution:
    def test_gini_uniform(self):
        from plugins.responder.ann_fewshot import gini_coefficient

        assert gini_coefficient(["a", "b", "c"]) == pytest.approx(0.0, abs=0.01)

    def test_gini_concentrated(self):
        from plugins.responder.ann_fewshot import gini_coefficient

        assert gini_coefficient(["a", "a", "a", "b"]) == pytest.approx(0.25)

    def test_retrieve_with_categories_tracks_labels(self, monkeypatch):
        from plugins.responder import ann_fewshot
        from plugins.responder.ann_fewshot import ANNFewShotIndex, gini_coefficient

        monkeypatch.setattr(ann_fewshot, "embed_text", _fake_embed_text)
        monkeypatch.setattr(ann_fewshot, "embed_batch", _fake_embed_batch)

        with get_db_conn() as conn:
            _seed_csat_and_examples(conn)

        ANNFewShotIndex.build()
        results = ANNFewShotIndex.retrieve_with_categories(
            "password reset help needed",
            k=2,
            similarity_floor=0.5,
            strategy="baseline",
        )
        categories = [item["category"] or "unknown" for item in results]
        assert len(categories) >= 1
        assert gini_coefficient(categories) >= 0.0


class TestCrossValidationHelpers:
    def test_run_cross_validation_with_mock_draft(self, monkeypatch):
        from analysis.fewshot_csat_eval import run_cross_validation
        from plugins.responder import ann_fewshot
        from plugins.responder.ann_fewshot import ANNFewShotIndex

        monkeypatch.setattr(ann_fewshot, "embed_text", _fake_embed_text)
        monkeypatch.setattr(ann_fewshot, "embed_batch", _fake_embed_batch)
        monkeypatch.setattr(
            ann_fewshot,
            "get_plugin_config",
            lambda _name: {
                "fewshot_k": 2,
                "fewshot_similarity_floor": 0.5,
                "csat_weight": 0.5,
            },
        )

        corpus = []
        categories = ["access", "billing"]
        for idx in range(10):
            ticket_key = f"CV-{idx}"
            category = categories[idx % 2]
            with get_db_conn() as conn:
                conn.execute(
                    "INSERT INTO tickets (ticket_key, summary, description, status) "
                    "VALUES (?, ?, ?, 'Open')",
                    (ticket_key, f"{category} issue {idx}", f"Details for {category}"),
                )
                conn.execute(
                    "INSERT INTO ticket_classifications (ticket_key, category, issue_type, confidence) "
                    "VALUES (?, ?, 'how-to', 0.9)",
                    (ticket_key, category),
                )
                conn.execute(
                    """
                    INSERT INTO ai_draft_feedback
                        (ticket_key, draft_comment_id, response_type,
                         draft_customer_response, actual_response,
                         similarity_score, agent_feedback)
                    VALUES (?, ?, 'self_service', 'Draft', 'Actual response text', 0.9, 'both_good')
                    """,
                    (ticket_key, f"c{idx}"),
                )
                conn.execute(
                    """
                    INSERT INTO category_csat_correlations
                        (category, run_date, mean_csat, std_csat, acceptance_rate, n_samples)
                    VALUES (?, '2026-06-01', ?, 0.5, 0.8, 40)
                    ON CONFLICT(category, run_date) DO NOTHING
                    """,
                    (category, 4.0 if category == "access" else 2.5),
                )

            corpus.append(
                {
                    "ticket_key": ticket_key,
                    "summary": f"{category} issue {idx}",
                    "description": f"Details for {category}",
                    "actual_response": "Actual response text",
                    "category": category,
                }
            )

        ANNFewShotIndex.build()

        def _mock_draft(summary, description, matches):
            return {"response_type": "self_service", "customer_response": "Actual response text"}

        _, summaries = run_cross_validation(corpus, n_folds=5, seed=42, draft_fn=_mock_draft)
        assert set(summaries) == {"baseline", "a", "b", "c"}
        for strategy in summaries:
            assert summaries[strategy].mean_coverage >= 0.0
            assert summaries[strategy].mean_gini >= 0.0

    def test_run_cross_validation_falls_back_to_kfold_for_sparse_classes(self, monkeypatch):
        from analysis.fewshot_csat_eval import run_cross_validation
        from plugins.responder import ann_fewshot
        from plugins.responder.ann_fewshot import ANNFewShotIndex

        monkeypatch.setattr(ann_fewshot, "embed_text", _fake_embed_text)
        monkeypatch.setattr(ann_fewshot, "embed_batch", _fake_embed_batch)
        monkeypatch.setattr(
            ann_fewshot,
            "get_plugin_config",
            lambda _name: {
                "fewshot_k": 2,
                "fewshot_similarity_floor": 0.5,
                "csat_weight": 0.5,
            },
        )

        corpus = []
        categories = ["access", "access", "access", "access", "access", "rare"]
        for idx, category in enumerate(categories):
            ticket_key = f"CV-SPARSE-{idx}"
            with get_db_conn() as conn:
                conn.execute(
                    "INSERT INTO tickets (ticket_key, summary, description, status) "
                    "VALUES (?, ?, ?, 'Open')",
                    (ticket_key, f"{category} issue {idx}", f"Details for {category}"),
                )
                conn.execute(
                    "INSERT INTO ticket_classifications (ticket_key, category, issue_type, confidence) "
                    "VALUES (?, ?, 'how-to', 0.9)",
                    (ticket_key, category),
                )
                conn.execute(
                    """
                    INSERT INTO ai_draft_feedback
                        (ticket_key, draft_comment_id, response_type,
                         draft_customer_response, actual_response,
                         similarity_score, agent_feedback)
                    VALUES (?, ?, 'self_service', 'Draft', 'Actual response text', 0.9, 'both_good')
                    """,
                    (ticket_key, f"c-sparse-{idx}"),
                )
                conn.execute(
                    """
                    INSERT INTO category_csat_correlations
                        (category, run_date, mean_csat, std_csat, acceptance_rate, n_samples)
                    VALUES (?, '2026-06-01', 4.0, 0.5, 0.8, 40)
                    ON CONFLICT(category, run_date) DO NOTHING
                    """,
                    (category,),
                )

            corpus.append(
                {
                    "ticket_key": ticket_key,
                    "summary": f"{category} issue {idx}",
                    "description": f"Details for {category}",
                    "actual_response": "Actual response text",
                    "category": category,
                }
            )

        ANNFewShotIndex.build()

        def _mock_draft(summary, description, matches):
            return {"response_type": "self_service", "customer_response": "Actual response text"}

        _, summaries = run_cross_validation(corpus, n_folds=5, seed=42, draft_fn=_mock_draft)
        assert set(summaries) == {"baseline", "a", "b", "c"}

    def test_apply_winner_to_pipeline_preserves_comments(self, tmp_path):
        from analysis.fewshot_csat_eval import apply_winner_to_pipeline

        pipeline_text = """
plugins:
  responder:
    csat_weight: 0.5                   # alpha for Strategy A linear CSAT weighting
    fewshot_csat_strategy: baseline    # auto / baseline / a / b / c
""".strip()
        path = tmp_path / "pipeline.yaml"
        path.write_text(pipeline_text)

        apply_winner_to_pipeline("a", str(path))
        updated = path.read_text()
        assert "fewshot_csat_strategy: a" in updated
        assert "# auto / baseline / a / b / c" in updated
