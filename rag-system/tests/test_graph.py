"""Smoke test for rag.graph.build_graph() — offline end-to-end.

No real ANTHROPIC_API_KEY is required: with no key set, rag.llm.call_llm
degrades to its MockLLM path (see llm.py), so every node in the compiled
graph can run without hitting a real API or the network (retriever_node's
call_web_search stays false for the sample transcript's planner mock, so it
never needs web-search-mcp's network path either).
"""
import asyncio

import pytest

from rag.graph import GraphState, build_graph


@pytest.fixture(autouse=True)
def _no_anthropic_key(monkeypatch):
    """Force the MockLLM path regardless of the environment running the
    suite, so this test is deterministic and offline everywhere."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def test_build_graph_compiles():
    compiled = build_graph()
    assert compiled is not None


def test_graph_runs_end_to_end_offline():
    compiled = build_graph()
    initial_state: GraphState = {
        "transcript": "Recommend an eco-friendly stainless-steel cleaner under fifteen dollars.",
    }
    # The graph contains an async node (retriever_node, which may await
    # web_search()), so LangGraph requires the async entry point even from a
    # sync test — .invoke() raises TypeError on a graph with any async node.
    final_state = asyncio.run(compiled.ainvoke(initial_state))

    assert "router_output" in final_state
    assert "planner_output" in final_state
    assert "reconciled" in final_state
    assert "answerer_output" in final_state
    assert "critic_output" in final_state

    answerer_output = final_state["answerer_output"]
    assert set(answerer_output.keys()) >= {"speech", "citations", "comparison_table"}
    assert isinstance(answerer_output["speech"], str) and answerer_output["speech"]

    critic_output = final_state["critic_output"]
    assert critic_output["action"] in {"accept", "revise"}
