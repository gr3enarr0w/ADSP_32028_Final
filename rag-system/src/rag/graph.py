"""
graph.py — single source of truth for the LangGraph orchestration (Final
deliverable): Router -> Planner -> Retriever -> Answerer/Critic, with a
bounded revise loop.

Other code (a future Streamlit UI, the eval harness) should import
`build_graph()` from here rather than re-wiring the graph anywhere else.
"""
from __future__ import annotations

from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .nodes import answerer_critic_node, planner_node, retriever_node, router_node

# Only one Answerer/Critic revision per answerer_critic.md's documented
# "the Answerer can regenerate once" rule.
MAX_REVISE_COUNT = 1


class GraphState(TypedDict, total=False):
    """Shared state threaded through every node. `total=False` because each
    field is written by a specific node and doesn't exist before that node
    runs (e.g. `router_output` doesn't exist until after `router_node`)."""

    transcript: str
    router_output: dict
    planner_output: dict
    rag_results: dict
    web_results: dict
    reconciled: dict
    answerer_output: dict
    critic_output: dict
    revise_count: int


def _route_after_answerer_critic(state: GraphState) -> str:
    """Conditional edge after answerer_critic: loop back on 'revise' (bounded
    by MAX_REVISE_COUNT, tracked in state["revise_count"] by the node
    itself), otherwise end."""
    critic_output = state.get("critic_output") or {}
    revise_count = state.get("revise_count", 0)
    if critic_output.get("action") == "revise" and revise_count <= MAX_REVISE_COUNT:
        return "revise"
    return "end"


def build_graph() -> CompiledStateGraph:
    """Wire and compile the full orchestration graph.

    router -> planner -> retriever -> answerer_critic -> (conditional)
        revise -> answerer_critic
        accept/END -> END
    """
    graph = StateGraph(GraphState)

    graph.add_node("router", router_node)
    graph.add_node("planner", planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("answerer_critic", answerer_critic_node)

    graph.set_entry_point("router")
    graph.add_edge("router", "planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "answerer_critic")
    graph.add_conditional_edges(
        "answerer_critic",
        _route_after_answerer_critic,
        {"revise": "answerer_critic", "end": END},
    )

    return graph.compile()
