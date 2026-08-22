"""Offline smoke tests for rag.nodes — no real LLM/network calls.

Follows the convention in tests/test_pipeline.py / tests/test_reconcile.py
(plain pytest, PYTHONPATH=src via conftest.py, offline hash embedder).
Every node test injects a fake `llm_fn` (the DI pattern nodes.py supports)
so nothing here hits a real API — retriever_node is exercised against the
real (offline, hash-embedder) rag_search with call_web_search=False so it
needs no network either.
"""
import asyncio
import json

import pytest

from rag.nodes import answerer_critic_node, planner_node, retriever_node, router_node


def _fake_llm(response: dict):
    """Build a fake llm_fn matching call_llm's signature that always returns
    the given dict as a JSON string, ignoring every other argument."""

    def _fn(system, user, model=None, max_tokens=1024, mock_response=None):
        return json.dumps(response)

    return _fn


def _bad_llm(raw: str):
    def _fn(system, user, model=None, max_tokens=1024, mock_response=None):
        return raw

    return _fn


# ---------------------------------------------------------------------------
# router_node
# ---------------------------------------------------------------------------

ROUTER_RESPONSE = {
    "task": "product_recommendation",
    "constraints": {
        "price_max": 15, "price_min": None, "material": "stainless steel",
        "brand": None, "eco_preference": True, "min_rating": None, "wants_live": False,
    },
    "keywords": ["eco-friendly", "stainless steel", "cleaner"],
    "safety_flags": [],
}


def test_router_node_writes_router_output():
    state = {"transcript": "Recommend an eco-friendly stainless-steel cleaner under fifteen dollars."}
    out = router_node(state, llm_fn=_fake_llm(ROUTER_RESPONSE))
    assert out["router_output"] == ROUTER_RESPONSE
    assert out is state  # mutates and returns the same state dict


def test_router_node_raises_on_malformed_json():
    state = {"transcript": "anything"}
    with pytest.raises(ValueError, match="not valid JSON"):
        router_node(state, llm_fn=_bad_llm("not json at all"))


def test_router_node_raises_on_missing_required_keys():
    state = {"transcript": "anything"}
    incomplete = {"task": "product_recommendation"}  # missing constraints/keywords/safety_flags
    with pytest.raises(ValueError, match="missing required keys"):
        router_node(state, llm_fn=_fake_llm(incomplete))


# ---------------------------------------------------------------------------
# planner_node
# ---------------------------------------------------------------------------

PLANNER_RESPONSE = {
    "sources": ["rag.search"],
    "call_web_search": False,
    "filters": {"price_max": 15, "material": "stainless steel"},
    "query": "eco-friendly plant-based stainless steel cleaner",
    "comparison_criteria": ["price", "rating", "price_per_oz", "ingredients"],
    "k": 3,
    "reconcile_on": [],
}


def test_planner_node_writes_planner_output():
    state = {"router_output": ROUTER_RESPONSE}
    out = planner_node(state, llm_fn=_fake_llm(PLANNER_RESPONSE))
    assert out["planner_output"] == PLANNER_RESPONSE


def test_planner_node_raises_on_malformed_json():
    state = {"router_output": ROUTER_RESPONSE}
    with pytest.raises(ValueError, match="not valid JSON"):
        planner_node(state, llm_fn=_bad_llm("{not valid"))


# ---------------------------------------------------------------------------
# retriever_node (real rag_search, offline hash embedder; no web search)
# ---------------------------------------------------------------------------

def test_retriever_node_no_web_search():
    state = {
        "planner_output": {
            "sources": ["rag.search"],
            "call_web_search": False,
            "filters": {},
            "query": "stainless steel cleaner",
            "comparison_criteria": ["price", "rating"],
            "k": 3,
            "reconcile_on": [],
        }
    }
    # retriever_node is async (it may await web_search()); run it with a
    # plain asyncio.run() rather than pulling in the pytest-asyncio plugin
    # just for this one offline test.
    out = asyncio.run(retriever_node(state))
    assert "rag_results" in out and "web_results" in out and "reconciled" in out
    assert out["rag_results"]["count"] > 0
    assert out["web_results"]["count"] == 0
    assert out["web_results"]["results"] == []
    # reconciled has one entry per rag result, all unmatched since no web results
    assert len(out["reconciled"]["items"]) == out["rag_results"]["count"]
    for item in out["reconciled"]["items"]:
        assert item["live_match"] is None
    assert out["reconciled"]["unmatched_web"] == []


# ---------------------------------------------------------------------------
# answerer_critic_node
# ---------------------------------------------------------------------------

RECONCILED_ONE_ITEM = {
    "items": [{
        "sku": "SKU-1", "title": "Steel-Safe Eco Cleaner", "price": 12.49, "rating": 4.6,
        "brand": "GreenGleam", "ingredients": "plant-based surfactants",
        "doc_id": "abc123", "url": "https://example.com/x", "price_per_oz": 0.78,
        "score": 2.9, "live_match": None, "discrepancy": None,
    }],
    "unmatched_web": [],
}

ANSWERER_RESPONSE = {
    "speech": "My top pick is the Steel-Safe Eco Cleaner — 4.6 stars, $12.49.",
    "citations": [{"doc_id": "abc123", "title": "Steel-Safe Eco Cleaner",
                   "url": "https://example.com/x", "source": "private"}],
    "comparison_table": [{"title": "Steel-Safe Eco Cleaner", "price": 12.49, "rating": 4.6,
                           "price_per_oz": 0.78, "ingredients": "plant-based surfactants",
                           "doc_id": "abc123"}],
}

CRITIC_ACCEPT = {"grounded": True, "unsafe": False, "reasons": [], "action": "accept"}
CRITIC_REVISE = {"grounded": False, "unsafe": False, "reasons": ["ungrounded price"], "action": "revise"}


def test_answerer_critic_node_accept_path():
    state = {"reconciled": RECONCILED_ONE_ITEM}
    responses = [ANSWERER_RESPONSE, CRITIC_ACCEPT]

    def llm_fn(system, user, model=None, max_tokens=1024, mock_response=None):
        return json.dumps(responses.pop(0))

    out = answerer_critic_node(state, llm_fn=llm_fn)
    assert out["answerer_output"] == ANSWERER_RESPONSE
    assert out["critic_output"] == CRITIC_ACCEPT
    assert out["revise_count"] == 0


def test_answerer_critic_node_revise_path_increments_revise_count():
    state = {"reconciled": RECONCILED_ONE_ITEM}
    # First pass: answerer, then critic says revise.
    responses = [ANSWERER_RESPONSE, CRITIC_REVISE]

    def llm_fn(system, user, model=None, max_tokens=1024, mock_response=None):
        return json.dumps(responses.pop(0))

    out = answerer_critic_node(state, llm_fn=llm_fn)
    assert out["critic_output"]["action"] == "revise"
    assert out["revise_count"] == 1


def test_grounding_floor_rejects_citation_not_in_retrieval():
    """The code-enforced grounding floor: a fabricated doc_id forces a revise
    verdict even when the (mock or live) Critic accepted."""
    state = {"reconciled": RECONCILED_ONE_ITEM}
    ungrounded = {
        **ANSWERER_RESPONSE,
        "citations": [{"doc_id": "ghost999", "title": "Invented Product",
                       "url": "https://example.com/ghost", "source": "private"}],
    }
    responses = [ungrounded, CRITIC_ACCEPT]

    def llm_fn(system, user, model=None, max_tokens=1024, mock_response=None):
        return json.dumps(responses.pop(0))

    out = answerer_critic_node(state, llm_fn=llm_fn)
    assert out["critic_output"]["action"] == "revise"
    assert out["critic_output"]["grounded"] is False
    assert out["revise_count"] == 1
    assert any("ghost999" in r for r in out["critic_output"]["reasons"])


def test_grounding_floor_checks_comparison_table_rows_too():
    state = {"reconciled": RECONCILED_ONE_ITEM}
    bad_table = {
        **ANSWERER_RESPONSE,
        "comparison_table": ANSWERER_RESPONSE["comparison_table"]
        + [{"title": "Phantom Cleaner", "price": 1.99, "rating": 5.0,
            "price_per_oz": 0.1, "ingredients": "?", "doc_id": "phantom42"}],
    }
    responses = [bad_table, CRITIC_ACCEPT]

    def llm_fn(system, user, model=None, max_tokens=1024, mock_response=None):
        return json.dumps(responses.pop(0))

    out = answerer_critic_node(state, llm_fn=llm_fn)
    assert out["critic_output"]["action"] == "revise"
    assert any("phantom42" in r for r in out["critic_output"]["reasons"])


def test_hazard_floor_prepends_caution_when_ingredients_combine_hazardously():
    """hazard_flags() is now enforced on the live answer path: two
    individually-safe retrieved products that jointly name bleach + ammonia
    must yield a spoken caution, placed first so budget truncation can't
    drop it."""
    from rag.safety import HAZARD_CAUTION

    base = RECONCILED_ONE_ITEM["items"][0]
    hazardous = {
        "items": [
            {**base, "ingredients": "Water, Sodium hypochlorite (bleach)"},
            {**base, "doc_id": "def456", "sku": "SKU-2",
             "title": "Glass Shine Ammonia Cleaner",
             "ingredients": "Water, Ammonia, Fragrance"},
        ],
        "unmatched_web": [],
    }
    answer = {
        **ANSWERER_RESPONSE,
        "citations": ANSWERER_RESPONSE["citations"]
        + [{"doc_id": "def456", "title": "Glass Shine Ammonia Cleaner",
            "url": "https://example.com/y", "source": "private"}],
    }
    state = {"reconciled": hazardous,
             "transcript": "can I use these two together on the counter?"}
    responses = [answer, CRITIC_ACCEPT]

    def llm_fn(system, user, model=None, max_tokens=1024, mock_response=None):
        return json.dumps(responses.pop(0))

    out = answerer_critic_node(state, llm_fn=llm_fn)
    assert out["answerer_output"]["speech"].startswith(HAZARD_CAUTION)
    assert out["answerer_output"]["safety_flags"] == ["bleach_ammonia"]
    assert out["critic_output"]["unsafe"] is True
    # mitigated with a caution, not blocked: the verdict still accepts
    assert out["critic_output"]["action"] == "accept"
    assert out["revise_count"] == 0


def test_answerer_critic_node_raises_on_malformed_critic_output():
    state = {"reconciled": RECONCILED_ONE_ITEM}
    responses = [json.dumps(ANSWERER_RESPONSE), "not json"]

    def llm_fn(system, user, model=None, max_tokens=1024, mock_response=None):
        return responses.pop(0)

    with pytest.raises(ValueError, match="not valid JSON"):
        answerer_critic_node(state, llm_fn=llm_fn)
