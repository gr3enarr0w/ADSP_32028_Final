"""Tests for prompt variant generation, CV fold splitting, and similarity scoring."""

from __future__ import annotations

import json

import pytest

import db as db_mod
from config import GEMINI_MODEL_ANALYSIS
from core.pipeline import load_pipeline_config
from db import get_db_conn, init_db
from plugins.responder import drafting
from plugins.responder.drafting import (
    PROMPT_VARIANT_BASELINE,
    PROMPT_VARIANT_COT,
    PROMPT_VARIANT_LEAN,
    PROMPT_VARIANT_STRUCTURED,
    PROMPT_VARIANT_XML_COT,
    PROMPT_VARIANT_LEAN_COT,
    PROMPT_VARIANT_KITCHEN_SINK,
    PROMPT_VARIANTS,
    WINNING_PROMPT_VARIANT,
    _build_draft_prompt,
    _build_match_sections,
    _max_per_source_for_variant,
    build_stratified_folds,
    evaluate_draft_with_llm,
    evaluate_prompt_variants,
    load_ground_truth_corpus,
    run_prompt_variant_cv,
    set_winning_prompt_variant,
    select_winning_variant,
)


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr(db_mod, "DB_PATH", db_path)
    monkeypatch.setattr(db_mod, "DATABASE_URL", None)
    init_db()
    load_pipeline_config()
    yield


def _sample_matches(*, n: int = 5) -> dict:
    return {
        "found": True,
        "faq_matches": [
            {"title": f"FAQ {i}", "body_html": f"FAQ body {i}"} for i in range(n)
        ],
        "kb_matches": [
            {"title": f"KB {i}", "url": f"https://example.com/kb/{i}"} for i in range(n)
        ],
        "ticket_matches": [
            {"summary": f"Ticket {i}", "resolution_summary": f"Resolution {i}"}
            for i in range(n)
        ],
        "atlassian_matches": [
            {"title": f"Doc {i}", "url": f"https://support.atlassian.com/doc/{i}"}
            for i in range(n)
        ],
    }


def _seed_ground_truth_corpus() -> None:
    response_types = ("self_service", "admin_action", "needs_info")
    with get_db_conn() as conn:
        for idx in range(15):
            ticket_key = f"ANTSE-{100 + idx}"
            response_type = response_types[idx % 3]
            conn.execute(
                """
                INSERT INTO tickets (ticket_key, summary, description, status)
                VALUES (?, ?, ?, 'Open')
                """,
                (
                    ticket_key,
                    f"Issue type {response_type} number {idx}",
                    f"Detailed description for {ticket_key}",
                ),
            )
            conn.execute(
                """
                INSERT INTO ai_draft_feedback
                    (ticket_key, draft_comment_id, response_type,
                     draft_customer_response, actual_response, similarity_score)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    ticket_key,
                    f"c-{idx}",
                    response_type,
                    f"Draft for {ticket_key}",
                    f"Actual response for {ticket_key}",
                    0.8,
                ),
            )


class TestPromptVariantGeneration:
    def test_baseline_prompt_orders_ticket_before_reference(self):
        prompt = _build_draft_prompt(
            PROMPT_VARIANT_BASELINE,
            "Reset password",
            "User locked out",
            _sample_matches(),
            few_shot_block="",
        )
        assert prompt is not None
        ticket_pos = prompt.index("TICKET:")
        ref_pos = prompt.index("REFERENCE MATERIAL")
        assert ticket_pos < ref_pos
        assert "STEP 1 — CLASSIFY" in prompt

    def test_structured_context_labels_sections_before_ticket(self):
        prompt = _build_draft_prompt(
            PROMPT_VARIANT_STRUCTURED,
            "Reset password",
            "User locked out",
            _sample_matches(),
            few_shot_block="",
        )
        assert prompt is not None
        assert "<kb_articles>" in prompt
        assert "<faq_entries>" in prompt
        assert "<resolved_tickets>" in prompt
        assert "<atlassian_docs>" in prompt
        assert "<ticket>" in prompt
        assert prompt.index("<kb_articles>") < prompt.index("<ticket>")
        assert "TICKET:\nSummary:" not in prompt

    def test_chain_of_thought_adds_step_back_instruction(self):
        prompt = _build_draft_prompt(
            PROMPT_VARIANT_COT,
            "Reset password",
            "User locked out",
            _sample_matches(),
            few_shot_block="",
        )
        assert prompt is not None
        assert "identify the ticket type" in prompt
        assert "likely resolution pattern" in prompt

    def test_lean_context_limits_hits_per_source(self):
        matches = _sample_matches(n=5)
        lean_sections = _build_match_sections(
            matches, max_per_source=_max_per_source_for_variant(PROMPT_VARIANT_LEAN)
        )
        default_sections = _build_match_sections(
            matches, max_per_source=_max_per_source_for_variant(PROMPT_VARIANT_BASELINE)
        )
        assert len(lean_sections["faq_entries"]) == 3
        assert len(default_sections["faq_entries"]) == 5

    def test_baseline_keeps_similar_ticket_limit_of_three(self):
        prompt = _build_draft_prompt(
            PROMPT_VARIANT_BASELINE,
            "Reset password",
            "User locked out",
            _sample_matches(n=5),
            few_shot_block="",
        )
        assert prompt is not None
        assert prompt.count("Resolved Ticket:") == 3

    def test_winning_variant_is_baseline(self):
        assert WINNING_PROMPT_VARIANT == PROMPT_VARIANT_STRUCTURED

    def test_all_variants_registered(self):
        assert len(PROMPT_VARIANTS) == 7

    def test_xml_plus_cot_uses_xml_tags_and_cot_preamble(self):
        prompt = _build_draft_prompt(
            PROMPT_VARIANT_XML_COT,
            "Reset password",
            "User locked out",
            _sample_matches(),
            few_shot_block="",
        )
        assert prompt is not None
        # XML structural tags from structured context
        assert "<kb_articles>" in prompt
        assert "<faq_entries>" in prompt
        assert "<resolved_tickets>" in prompt
        assert "<ticket>" in prompt
        # COT preamble
        assert "identify the ticket type" in prompt
        assert "likely resolution pattern" in prompt

    def test_lean_plus_cot_limits_hits_and_adds_cot_preamble(self):
        matches = _sample_matches(n=5)
        prompt = _build_draft_prompt(
            PROMPT_VARIANT_LEAN_COT,
            "Reset password",
            "User locked out",
            matches,
            few_shot_block="",
        )
        assert prompt is not None
        # COT preamble present
        assert "identify the ticket type" in prompt
        assert "likely resolution pattern" in prompt
        # Lean limits: at most 3 FAQ entries in the prompt body
        assert prompt.count("FAQ: FAQ") <= 3

    def test_kitchen_sink_uses_xml_tags_lean_limits_and_cot_preamble(self):
        matches = _sample_matches(n=5)
        prompt = _build_draft_prompt(
            PROMPT_VARIANT_KITCHEN_SINK,
            "Reset password",
            "User locked out",
            matches,
            few_shot_block="",
        )
        assert prompt is not None
        # XML structural tags
        assert "<kb_articles>" in prompt
        assert "<faq_entries>" in prompt
        assert "<resolved_tickets>" in prompt
        assert "<ticket>" in prompt
        # COT preamble
        assert "identify the ticket type" in prompt
        assert "likely resolution pattern" in prompt
        # Lean limits: at most 3 FAQ entries in the prompt body
        assert prompt.count("FAQ: FAQ") <= 3


class TestStratifiedFoldSplit:
    def test_build_stratified_folds_preserves_response_type_counts(self):
        _seed_ground_truth_corpus()
        corpus = load_ground_truth_corpus()
        folds = build_stratified_folds(corpus, n_splits=5, random_state=42)

        assert len(folds) == 5
        for train_idx, test_idx in folds:
            assert len(train_idx) + len(test_idx) == len(corpus)
            assert len(set(train_idx) & set(test_idx)) == 0

            test_labels = [corpus[i]["response_type"] for i in test_idx]
            assert len(test_labels) == 3

    def test_build_stratified_folds_requires_minimum_per_class(self):
        with get_db_conn() as conn:
            conn.execute(
                """
                INSERT INTO ai_draft_feedback
                    (ticket_key, draft_comment_id, response_type,
                     draft_customer_response, actual_response, similarity_score)
                VALUES ('ANTSE-1', 'c1', 'self_service', 'd', 'a', 0.9)
                """
            )
        corpus = load_ground_truth_corpus()
        with pytest.raises(ValueError, match="at least 5"):
            build_stratified_folds(corpus, n_splits=5)


class TestLLMJudgeComputation:
    def test_evaluate_draft_with_llm_returns_score(self, monkeypatch):
        class MockResponse:
            text = '{"customer_score": 5, "admin_score": 5}'
        
        class MockModels:
            def generate_content(self, model, contents):
                return MockResponse()
        
        class MockClient:
            models = MockModels()

        monkeypatch.setattr(
            "plugins.responder.drafting.get_genai_client",
            lambda: MockClient(),
        )
        assert evaluate_draft_with_llm("draft", "internal notes", "actual") == 1.0

    def test_evaluate_draft_with_llm_averages_customer_and_admin_scores(self, monkeypatch):
        class MockResponse:
            text = '{"customer_score": 5, "admin_score": 1}'
        
        class MockModels:
            def generate_content(self, model, contents):
                return MockResponse()
        
        class MockClient:
            models = MockModels()

        monkeypatch.setattr(
            "plugins.responder.drafting.get_genai_client",
            lambda: MockClient(),
        )
        assert evaluate_draft_with_llm("draft", "internal notes", "actual") == 0.5

    def test_evaluate_draft_with_llm_handles_error(self, monkeypatch):
        def fake_client():
            raise Exception("API Error")

        monkeypatch.setattr("plugins.responder.drafting.get_genai_client", fake_client)
        assert evaluate_draft_with_llm("draft", "internal notes", "actual") == 0.0

    def test_empty_text_returns_zero(self):
        assert evaluate_draft_with_llm("", "internal notes", "actual") == 0.0
        assert evaluate_draft_with_llm("draft", "internal notes", "") == 0.0


class TestPromptVariantEvaluation:
    def test_evaluate_prompt_variants_with_mock_draft_fn(self, monkeypatch, tmp_path):
        _seed_ground_truth_corpus()

        def fake_lookup(summary: str) -> dict:
            return _sample_matches(n=3)

        monkeypatch.setattr(
            "plugins.responder.drafting._lookup_matches_for_ticket",
            lambda summary, description: fake_lookup(summary),
        )
        monkeypatch.setattr(
            "plugins.responder.drafting.evaluate_draft_with_llm",
            lambda *args, **kwargs: 0.75,
        )

        def fake_draft_fn(variant, summary, description, matches):
            return {
                "response_type": "self_service",
                "customer_response": f"{variant}: {summary}",
                "admin_steps": "1. Check the customer's permissions.",
                "missing_info": None,
            }

        results = evaluate_prompt_variants(
            draft_fn=fake_draft_fn,
            checkpoint_file=str(tmp_path / "test_checkpoint.jsonl"),
        )
        assert set(results) == set(PROMPT_VARIANTS)
        for variant in PROMPT_VARIANTS:
            assert results[variant].n_samples == 15
            assert results[variant].mean_judge_score == pytest.approx(0.75)

    def test_evaluate_prompt_variants_rejects_incompatible_checkpoint(self, tmp_path):
        _seed_ground_truth_corpus()
        checkpoint_path = tmp_path / "old_checkpoint.jsonl"
        checkpoint_path.write_text(
            json.dumps(
                {
                    "ticket_key": "ANTSE-100",
                    "fold_idx": 0,
                    "results": [[0, PROMPT_VARIANT_BASELINE, 0.9, True, 10, 1.0]],
                }
            )
            + "\n"
        )

        with pytest.raises(ValueError, match="incompatible"):
            evaluate_prompt_variants(checkpoint_file=str(checkpoint_path))

    def test_checkpoint_resume_handles_duplicate_ticket_keys(self, monkeypatch, tmp_path):
        corpus = [
            {
                "feedback_row_id": 101,
                "ticket_key": "ANTSE-DUP",
                "response_type": "self_service",
                "actual_response": "Actual response A",
            },
            {
                "feedback_row_id": 202,
                "ticket_key": "ANTSE-DUP",
                "response_type": "self_service",
                "actual_response": "Actual response B",
            },
        ]

        monkeypatch.setattr(
            drafting,
            "build_stratified_folds",
            lambda corpus, n_splits=5, random_state=42: [([], [0, 1])],
        )
        monkeypatch.setattr(
            drafting,
            "_get_ticket_context_from_db",
            lambda ticket_key: ("Duplicate ticket", "Same Jira issue, different feedback rows"),
        )
        monkeypatch.setattr(
            drafting,
            "_lookup_matches_for_ticket",
            lambda summary, description: _sample_matches(n=2),
        )

        def fake_draft_fn(variant, summary, description, matches, few_shot_block=None):
            return {
                "response_type": "self_service",
                "customer_response": "Customer draft",
                "admin_steps": "1. Review permissions.",
                "missing_info": None,
            }

        checkpoint_path = tmp_path / "duplicate_ticket_checkpoint.jsonl"

        monkeypatch.setattr(
            drafting,
            "evaluate_draft_with_llm",
            lambda customer_response, admin_steps, actual_response, **kwargs: (
                0.25 if actual_response.endswith("A") else 0.75
            ),
        )
        first_results = evaluate_prompt_variants(
            corpus=corpus,
            variants=(PROMPT_VARIANT_BASELINE,),
            checkpoint_file=str(checkpoint_path),
            draft_fn=fake_draft_fn,
        )
        assert first_results[PROMPT_VARIANT_BASELINE].n_samples == 2
        assert first_results[PROMPT_VARIANT_BASELINE].mean_judge_score == pytest.approx(0.5)

        def fail_if_judge_runs(*args, **kwargs):
            pytest.fail("checkpoint replay should not re-run judged tasks")

        monkeypatch.setattr(drafting, "evaluate_draft_with_llm", fail_if_judge_runs)
        replay_results = evaluate_prompt_variants(
            corpus=corpus,
            variants=(PROMPT_VARIANT_BASELINE,),
            checkpoint_file=str(checkpoint_path),
            draft_fn=fake_draft_fn,
        )
        assert replay_results[PROMPT_VARIANT_BASELINE].n_samples == 2
        assert replay_results[PROMPT_VARIANT_BASELINE].mean_judge_score == pytest.approx(0.5)

    def test_select_winning_variant_keeps_baseline_without_significance(self):
        from plugins.responder.drafting import PromptVariantCVResult

        baseline = PromptVariantCVResult(
            variant=PROMPT_VARIANT_BASELINE,
            mean_judge_score=0.8,
            std_judge_score=0.05,
            fold_means=[0.8],
            fold_stds=[0.05],
            p_value_vs_baseline=None,
            n_samples=10,
            response_type_accuracy=0.7,
            draft_length_p50=120,
            draft_length_p95=180,
            latency_p95=5.0,
        )
        challenger = PromptVariantCVResult(
            variant=PROMPT_VARIANT_STRUCTURED,
            mean_judge_score=0.81,
            std_judge_score=0.05,
            fold_means=[0.81],
            fold_stds=[0.05],
            p_value_vs_baseline=0.2,
            n_samples=10,
            response_type_accuracy=0.75,
            draft_length_p50=130,
            draft_length_p95=190,
            latency_p95=5.5,
        )
        winner = select_winning_variant(
            {PROMPT_VARIANT_BASELINE: baseline, PROMPT_VARIANT_STRUCTURED: challenger}
        )
        assert winner == PROMPT_VARIANT_BASELINE

    def test_run_prompt_variant_cv_applies_winner(self, monkeypatch):
        monkeypatch.setattr(
            drafting,
            "load_ground_truth_corpus",
            lambda: [{"ticket_key": "ANTSE-1", "response_type": "self_service"}],
        )
        monkeypatch.setattr(
            drafting,
            "evaluate_prompt_variants",
            lambda corpus, **kwargs: {},
        )
        monkeypatch.setattr(
            drafting,
            "select_winning_variant",
            lambda results: PROMPT_VARIANT_STRUCTURED,
        )
        applied = {}

        def _fake_apply(results, *, winner, evaluated_at, module_path=None):
            applied["winner"] = winner
            drafting.set_winning_prompt_variant(winner)

        monkeypatch.setattr(drafting, "apply_prompt_tuning_results_to_module", _fake_apply)
        monkeypatch.setattr(drafting, "WINNING_PROMPT_VARIANT", PROMPT_VARIANT_BASELINE)

        winner, _ = run_prompt_variant_cv()
        assert winner == PROMPT_VARIANT_STRUCTURED
        assert applied["winner"] == PROMPT_VARIANT_STRUCTURED
        assert drafting.WINNING_PROMPT_VARIANT == PROMPT_VARIANT_STRUCTURED

    def test_set_winning_prompt_variant_rejects_unknown(self):
        with pytest.raises(ValueError, match="Unsupported prompt variant"):
            set_winning_prompt_variant("unknown")

    def test_build_draft_prompt_raises_for_unknown_variant(self):
        with pytest.raises(ValueError, match="Unhandled prompt variant"):
            _build_draft_prompt(
                "totally_unknown",
                "summary",
                "desc",
                _sample_matches(),
                few_shot_block="",
            )


class TestGeminiModelSelection:
    def test_invoke_uses_analysis_model(self, monkeypatch):
        class _FakeResponse:
            text = '{"response_type":"self_service","customer_response":"ok","admin_steps":null,"missing_info":null}'

        class _FakeModels:
            captured_model = None

            def generate_content(self, *, model, contents):
                self.captured_model = model
                return _FakeResponse()

        class _FakeClient:
            models = _FakeModels()

        fake_client = _FakeClient()
        monkeypatch.setattr(drafting, "get_genai_client", lambda: fake_client)

        result = drafting._invoke_gemini_draft("prompt")
        assert result is not None
        assert fake_client.models.captured_model == GEMINI_MODEL_ANALYSIS
