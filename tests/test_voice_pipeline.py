"""Tests for ui/pipeline.py — the ASR → graph → TTS integration layer.

Runs fully offline: no ANTHROPIC_API_KEY (so `rag.llm.call_llm` takes its
MockLLM path), hash embedder via `tests/conftest.py`, and `synthesize=False`
so no audio backend is required in CI.

The point of these tests is the *contract the UI depends on* — that a turn
yields a Trace whose steps, citations and answer are consistent with the graph
state — not the graph's own behaviour, which `tests/test_graph.py` covers.
"""
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ui"))

from pipeline import _citations_from, run_turn  # noqa: E402

from rag.tts import fits_budget  # noqa: E402

DEMO_TRANSCRIPT = "Recommend an eco-friendly stainless-steel cleaner under fifteen dollars."


@pytest.fixture(autouse=True)
def _no_anthropic_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


@pytest.fixture(scope="module")
def turn():
    return run_turn(transcript=DEMO_TRANSCRIPT, synthesize=False)


def test_requires_some_input():
    with pytest.raises(ValueError):
        run_turn()


def test_trace_has_every_graph_node(turn):
    trace, _ = turn
    nodes = [s.node for s in trace.steps]
    # One step per graph node, in graph order (graph.py: router -> planner ->
    # retriever -> answerer_critic, the last split into answerer + critic).
    assert nodes[:2] == ["router", "planner"]
    assert "retriever" in nodes
    assert nodes[-2:] == ["answerer", "critic"]


def test_trace_carries_transcript_and_answer(turn):
    trace, final_state = turn
    assert trace.transcript == DEMO_TRANSCRIPT
    assert trace.answer_text == final_state["answerer_output"]["speech"]
    assert trace.answer_text


def test_spoken_answer_fits_the_15s_budget(turn):
    """prompts/answerer_critic.md's ≤15s rule, enforced on a real retrieval
    result — the case that caught the over-budget mock Answerer."""
    trace, _ = turn
    assert fits_budget(trace.answer_text), trace.answer_text


def test_every_table_row_is_cited(turn):
    """The grounding contract: nothing shows on screen that isn't traceable."""
    trace, final_state = turn
    cited = {c.get("doc_id") for c in trace.citations if c.get("doc_id")}
    for row in final_state["answerer_output"]["comparison_table"]:
        if row.get("doc_id"):
            assert row["doc_id"] in cited


def test_elapsed_ms_is_the_sum_of_steps(turn):
    trace, _ = turn
    assert trace.elapsed_ms == pytest.approx(sum(s.elapsed_ms for s in trace.steps))


def test_no_audio_when_synthesis_is_off(turn):
    trace, _ = turn
    assert trace.audio_path is None


def test_citations_merge_private_and_live():
    """Live URLs come from reconciliation, not from the Answerer payload —
    the model is never asked to produce a URL it could invent."""
    answerer_output = {
        "citations": [{"doc_id": "abc", "title": "Private Cleaner", "url": None}],
        "comparison_table": [],
        "speech": "",
    }
    reconciled = {
        "items": [{
            "doc_id": "abc", "title": "Private Cleaner",
            "live_match": {"url": "https://example.com/x", "price": 14.99,
                           "availability": None},
        }],
        "unmatched_web": [],
    }
    citations = _citations_from(answerer_output, reconciled)
    sources = {c["source"] for c in citations}
    assert sources == {"private", "live"}
    assert any(c.get("url") == "https://example.com/x" for c in citations)


def test_citations_are_deduplicated():
    answerer_output = {
        "citations": [
            {"doc_id": "abc", "title": "X", "url": None},
            {"doc_id": "abc", "title": "X", "url": None},
        ],
        "comparison_table": [], "speech": "",
    }
    assert len(_citations_from(answerer_output, {"items": []})) == 1
