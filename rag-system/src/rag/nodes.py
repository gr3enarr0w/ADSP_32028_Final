"""
nodes.py — one function per LangGraph node: Router, Planner, Retriever,
Answerer/Critic (Final deliverable — closes the "prompts exist but aren't
loaded by real code" gap).

Every node loads its own prompt file(s) fresh from disk on every call — never
hardcoded prompt text in this module — so Prompt Disclosure (`prompts/`)
genuinely equals what runs. `PROMPTS_DIR` is resolved the same way
`rag.config.REPO_ROOT` is: `Path(__file__).resolve().parents[2]` from
`src/rag/nodes.py` lands on `rag-system/`, then `/ "prompts"`.

Each node accepts an optional `llm_fn` (default `rag.llm.call_llm`) so tests
can inject a fake without hitting a real API or needing complex mocking —
standard dependency injection.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Callable, Optional

from .llm import call_llm
from .rag_search import rag_search
from .reconcile import reconcile

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"
FEWSHOTS_DIR = PROMPTS_DIR / "fewshots"

LlmFn = Callable[..., str]

# web_search.py lives in the sibling `web-search-mcp/` directory, not under
# this package. Add it to sys.path the same way
# web-search-mcp/combined_mcp_server.py adds rag-system/src to ITS path: climb
# from this file to the shared parent, then down into the sibling dir — never
# a hardcoded absolute path, so this works regardless of checkout location.
# From src/rag/nodes.py: parents[0]=src/rag, [1]=src, [2]=rag-system,
# [3]=<repo root containing both rag-system/ and web-search-mcp/>.
#
# Deliberately NOT imported at module level: web_search.py -> orchestrator.py
# -> providers/__init__.py unconditionally imports every provider module
# (exa_py, google-genai, newsdataapi, google-auth, ...), none of which are
# rag-system's own dependencies. Importing `rag.nodes` (or `rag.graph`, which
# imports it) should not hard-require that whole stack just to test
# router_node/planner_node — so the import is deferred into retriever_node's
# body, the only place that actually needs it.
_WEB_SEARCH_MCP_DIR = Path(__file__).resolve().parents[3] / "web-search-mcp"


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _system_assistant_prompt() -> str:
    return _read(PROMPTS_DIR / "system_assistant.md")


def _load_json(path: Path):
    return json.loads(_read(path))


def _parse_llm_json(raw: str, required_keys: set, node_name: str) -> dict:
    """Parse an LLM response as JSON and check it has the documented shape.

    Raises a clear ValueError (never silently swallowed) if the response
    isn't valid JSON, isn't an object, or is missing required keys —
    per the task spec: "Raise a clear error if the LLM response isn't valid
    JSON matching the expected shape."
    """
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"{node_name}: LLM response is not valid JSON: {e}. Raw response: {raw!r}"
        ) from e
    if not isinstance(parsed, dict):
        raise ValueError(
            f"{node_name}: LLM response JSON is not an object (got {type(parsed).__name__}): {raw!r}"
        )
    missing = required_keys - parsed.keys()
    if missing:
        raise ValueError(
            f"{node_name}: LLM response JSON is missing required keys {sorted(missing)}: {raw!r}"
        )
    return parsed


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

_ROUTER_REQUIRED_KEYS = {"task", "constraints", "keywords", "safety_flags"}


def _router_mock_response(transcript: str) -> dict:
    """A structurally-valid canned router output, used when call_llm has no
    real key/provider to hit. Not meant to be semantically perfect — just a
    valid shape so the rest of the graph can run offline end-to-end."""
    lowered = transcript.lower()
    words = [w.strip(".,!?$") for w in transcript.split()]
    keywords = [w for w in words if len(w) > 3][:5] or ["cleaner"]
    return {
        "task": "product_recommendation",
        "constraints": {
            "price_max": None,
            "price_min": None,
            "material": "stainless steel" if "stainless" in lowered else None,
            "brand": None,
            "eco_preference": "eco" in lowered or "natural" in lowered,
            "min_rating": None,
            "wants_live": any(w in lowered for w in ("now", "today", "current", "in stock")),
        },
        "keywords": keywords,
        "safety_flags": ["mixing_chemicals"] if "bleach" in lowered and "ammonia" in lowered else [],
    }


def router_node(state: dict, llm_fn: LlmFn = call_llm) -> dict:
    """Router node: ASR transcript -> intent/constraints/safety JSON.

    Prompt sources (loaded fresh from disk, per prompts/README.md's
    "Loading" convention): system_assistant.md + router_intent.md, with
    router_examples.json as few-shot context in the user message. Schema:
    router_intent.md's "Output schema" section.
    """
    transcript = state["transcript"]
    system = _system_assistant_prompt() + "\n\n" + _read(PROMPTS_DIR / "router_intent.md")
    examples = _load_json(FEWSHOTS_DIR / "router_examples.json")
    user = (
        "Few-shot examples (transcript -> expected JSON output):\n"
        + json.dumps(examples, indent=2)
        + "\n\nNow produce the JSON output for this transcript. Return ONLY the JSON object.\n\n"
        + f"Transcript: {transcript!r}"
    )
    raw = llm_fn(system, user, mock_response=_router_mock_response(transcript))
    router_output = _parse_llm_json(raw, _ROUTER_REQUIRED_KEYS, "router_node")
    state["router_output"] = router_output
    return state


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

_PLANNER_REQUIRED_KEYS = {
    "sources", "call_web_search", "filters", "query",
    "comparison_criteria", "k", "reconcile_on",
}


def _planner_mock_response(router_output: dict) -> dict:
    constraints = router_output.get("constraints", {}) or {}
    keywords = router_output.get("keywords", []) or []
    wants_live = bool(constraints.get("wants_live"))
    task = router_output.get("task")
    filters = {
        k: v for k, v in {
            "price_max": constraints.get("price_max"),
            "price_min": constraints.get("price_min"),
            "min_rating": constraints.get("min_rating"),
            "brand": constraints.get("brand"),
            "material": constraints.get("material"),
        }.items() if v is not None
    }
    call_web_search = wants_live or task == "availability_check"
    return {
        "sources": ["rag.search", "web.search"] if call_web_search else ["rag.search"],
        "call_web_search": call_web_search,
        "filters": filters,
        "query": " ".join(keywords) or "household cleaner",
        "comparison_criteria": ["price", "rating", "price_per_oz", "ingredients"],
        "k": 5 if task == "comparison" else 3,
        "reconcile_on": ["sku", "brand", "title"] if call_web_search else [],
    }


def planner_node(state: dict, llm_fn: LlmFn = call_llm) -> dict:
    """Planner node: Router JSON -> source/filter/comparison plan JSON.

    Prompt sources: planner.md, with planner_examples.json as few-shot
    context. Input: state["router_output"]. Schema: planner.md's "Output
    schema" section.
    """
    router_output = state["router_output"]
    system = _system_assistant_prompt() + "\n\n" + _read(PROMPTS_DIR / "planner.md")
    examples = _load_json(FEWSHOTS_DIR / "planner_examples.json")
    user = (
        "Few-shot examples (router output -> expected plan):\n"
        + json.dumps(examples, indent=2)
        + "\n\nNow produce the plan JSON for this Router output. Return ONLY the JSON object.\n\n"
        + f"Router output: {json.dumps(router_output)}"
    )
    raw = llm_fn(system, user, mock_response=_planner_mock_response(router_output))
    planner_output = _parse_llm_json(raw, _PLANNER_REQUIRED_KEYS, "planner_node")
    state["planner_output"] = planner_output
    return state


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

async def retriever_node(state: dict) -> dict:
    """Retriever node: calls `rag.search` (and `web.search` when the
    Planner's `call_web_search` field is true), then reconciles.

    No LLM call here — this node is pure tool-calling + reconciliation, per
    retriever_tool_instructions.md. Source: rag_search() from rag.rag_search
    (Shane), web_search() from web-search-mcp/web_search.py (Clark),
    reconcile() from rag.reconcile.
    """
    planner_output = state["planner_output"]
    query = planner_output["query"]
    k = planner_output.get("k", 3)
    filters = planner_output.get("filters") or {}

    rag_result = rag_search(query, k=k, filters=filters)
    # retriever_tool_instructions.md "Do / Don't": at most one relaxed retry
    # with fewer filters if the first call returns nothing.
    if rag_result["count"] == 0 and filters:
        print(
            "nodes.retriever_node: rag.search returned 0 results with filters; "
            "retrying once with filters relaxed.",
            file=sys.stderr,
        )
        rag_result = rag_search(query, k=k, filters=None)

    web_result = {"query": query, "count": 0, "results": []}
    if planner_output.get("call_web_search"):
        try:
            if str(_WEB_SEARCH_MCP_DIR) not in sys.path:
                sys.path.insert(0, str(_WEB_SEARCH_MCP_DIR))
            # web_search.py itself loads web-search-mcp/.env via an explicit
            # path (see its own module docstring / the fix cited in
            # combined_mcp_server.py's comment block), so importing it here
            # picks up provider keys regardless of this process's CWD.
            from web_search import web_search  # noqa: E402

            web_result = await web_search(query, k=k)
        except Exception as e:  # noqa: BLE001 - never let a flaky live search break the graph
            print(
                f"nodes.retriever_node: web_search failed ({e!r}); "
                f"continuing with rag.search results only.",
                file=sys.stderr,
            )

    reconciled = reconcile(rag_result["results"], web_result["results"])

    state["rag_results"] = rag_result
    state["web_results"] = web_result
    state["reconciled"] = reconciled
    return state


# ---------------------------------------------------------------------------
# Answerer / Critic
# ---------------------------------------------------------------------------

_ANSWERER_REQUIRED_KEYS = {"speech", "citations", "comparison_table"}
_CRITIC_REQUIRED_KEYS = {"grounded", "unsafe", "reasons", "action"}


def _answerer_mock_response(reconciled: dict) -> dict:
    items = (reconciled or {}).get("items", [])
    if not items:
        return {
            "speech": (
                "I couldn't find a match for that in our catalog. Want me to "
                "pull up some plant-based multi-surface sprays instead?"
            ),
            "citations": [],
            "comparison_table": [],
        }
    top = items[0]
    citations = [{
        "doc_id": top.get("doc_id"), "title": top.get("title"),
        "url": top.get("url"), "source": "private",
    }]
    comparison_table = [{
        "title": top.get("title"), "price": top.get("price"), "rating": top.get("rating"),
        "price_per_oz": top.get("price_per_oz"), "ingredients": top.get("ingredients"),
        "doc_id": top.get("doc_id"),
    }]
    price = top.get("price")
    rating = top.get("rating")
    price_str = f"${price:.2f}" if isinstance(price, (int, float)) else "an unlisted price"
    speech = (
        f"My top pick is {top.get('title', 'this product')} at {price_str}"
        + (f", {rating} stars" if rating else "")
        + f". I compared it with {max(len(items) - 1, 0)} alternatives — details and sources "
        "are on your screen. Want the most affordable or the highest rated?"
    )
    return {"speech": speech, "citations": citations, "comparison_table": comparison_table}


def _critic_mock_response() -> dict:
    # Mock always accepts — there's no real grounding to check offline, and
    # accepting keeps the offline graph run terminating without a live model.
    return {"grounded": True, "unsafe": False, "reasons": [], "action": "accept"}


def answerer_critic_node(state: dict, llm_fn: LlmFn = call_llm) -> dict:
    """Answerer, then Critic. One Answerer+Critic pass per call.

    Prompt source: answerer_critic.md (both roles), with
    answerer_examples.json as the Answerer's few-shot context. Input:
    state["reconciled"]. On `critic_output.action == "revise"`, this node
    bumps `state["revise_count"]` — the graph (graph.py) is what actually
    loops back to this node, bounded to exactly one revision per
    answerer_critic.md's "the Answerer can regenerate once" rule.
    """
    reconciled = state["reconciled"]
    revise_count = state.get("revise_count", 0)
    prior_critic = state.get("critic_output")

    system = _system_assistant_prompt() + "\n\n" + _read(PROMPTS_DIR / "answerer_critic.md")
    examples = _load_json(FEWSHOTS_DIR / "answerer_examples.json")

    answerer_user = (
        "Few-shot examples (retrieved results -> expected Answerer payload):\n"
        + json.dumps(examples, indent=2)
        + "\n\nNow produce the Answerer JSON payload for this reconciled retrieval "
        "result. Return ONLY the JSON object (speech, citations, comparison_table).\n\n"
        + f"Reconciled results: {json.dumps(reconciled)}"
    )
    if prior_critic and prior_critic.get("action") == "revise":
        answerer_user += (
            "\n\nThe Critic rejected your previous answer for these reasons: "
            f"{json.dumps(prior_critic.get('reasons', []))}. Revise accordingly."
        )
    raw_answer = llm_fn(system, answerer_user, mock_response=_answerer_mock_response(reconciled))
    answerer_output = _parse_llm_json(raw_answer, _ANSWERER_REQUIRED_KEYS, "answerer_critic_node (Answerer)")

    critic_user = (
        "Verify the following Answerer payload against the reconciled retrieval "
        "results it was built from. Return ONLY the JSON verdict object "
        "(grounded, unsafe, reasons, action).\n\n"
        f"Answerer payload: {json.dumps(answerer_output)}\n\n"
        f"Reconciled results it was built from: {json.dumps(reconciled)}"
    )
    raw_critic = llm_fn(system, critic_user, mock_response=_critic_mock_response())
    critic_output = _parse_llm_json(raw_critic, _CRITIC_REQUIRED_KEYS, "answerer_critic_node (Critic)")

    if critic_output.get("action") == "revise":
        revise_count += 1

    state["answerer_output"] = answerer_output
    state["critic_output"] = critic_output
    state["revise_count"] = revise_count
    return state
