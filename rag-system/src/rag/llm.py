"""
llm.py — single LLM call-site for the LangGraph orchestration (Final deliverable).

`call_llm()` is the one function every node in `nodes.py` goes through. It
dispatches on `get_config().llm_provider` (config-driven / Model-Agnostic,
per config.py's own doc comment and the grading requirement).

Only the "anthropic" path is fully implemented, per the plan — other
providers raise NotImplementedError with a clear message rather than
pretending to work.

Graceful degradation (matches the pattern already used by
web-search-mcp/web_search.py's provider dispatch and rag-system/src/rag/tts.py's
provider fallback): if ANTHROPIC_API_KEY is not set, or the provider isn't
"anthropic", or the `anthropic` package/network call fails, we never raise —
we fall back to a MockLLM and log which path was taken to stderr, so the
whole graph (and the notebook that exercises it) runs end-to-end offline with
zero setup.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Optional

from .config import get_config


def call_llm(
    system: str,
    user: str,
    model: Optional[str] = None,
    max_tokens: int = 1024,
    mock_response: Optional[dict] = None,
) -> str:
    """Call the configured LLM provider and return its text response.

    Args:
        system: system prompt (already assembled by the caller, e.g.
            `system_assistant.md + router_intent.md`).
        user: user-turn content (transcript, few-shot examples, structured
            input JSON — whatever the node needs the model to see).
        model: overrides `get_config().llm_model` if given.
        max_tokens: passed through to the provider call.
        mock_response: when the real call can't be made (or provider isn't
            "anthropic"), the mock path returns `json.dumps(mock_response)`
            if given, or a minimal generic fallback dict otherwise. Callers
            in nodes.py should thread through a caller-appropriate dict here
            so tests are deterministic and don't need to sniff prompt text.

    Returns:
        The model's text response (expected to be a JSON string for every
        node in this graph, per system_assistant.md's "Output discipline").

    Never raises on a missing key or provider failure — always degrades to
    the mock, matching the repo's established graceful-degradation pattern
    (see web-search-mcp/web_search.py, rag/tts.py).
    """
    cfg = get_config()
    provider = (cfg.llm_provider or "anthropic").lower()

    if provider != "anthropic":
        print(
            f"llm.call_llm: provider='{provider}' is not yet implemented for "
            f"live calls; falling back to MockLLM.",
            file=sys.stderr,
        )
        return _mock_llm(mock_response)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(
            "llm.call_llm: ANTHROPIC_API_KEY is not set; falling back to "
            "MockLLM so the graph still runs offline.",
            file=sys.stderr,
        )
        return _mock_llm(mock_response)

    try:
        return _call_anthropic(system, user, model or cfg.llm_model, max_tokens, api_key)
    except Exception as e:  # noqa: BLE001 - never let a live-call failure break the graph
        print(
            f"llm.call_llm: anthropic call failed ({e!r}); falling back to MockLLM.",
            file=sys.stderr,
        )
        return _mock_llm(mock_response)


def _call_anthropic(system: str, user: str, model: str, max_tokens: int, api_key: str) -> str:
    """The real, live Anthropic API call. Only path that hits the network."""
    from anthropic import Anthropic  # local import: optional dependency

    client = Anthropic(api_key=api_key)
    print(f"llm.call_llm: calling anthropic model='{model}'", file=sys.stderr)
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return response.content[0].text


def _mock_llm(mock_response: Optional[dict]) -> str:
    """Return a plausible canned JSON response for offline/no-key runs.

    Prefers an explicit `mock_response` dict threaded through by the caller
    (nodes.py) — simpler and more testable than sniffing the system prompt
    for schema markers. Falls back to a minimal generic dict if none given,
    so `call_llm` never raises even when a caller forgets to supply one.
    """
    print("llm.call_llm: using MockLLM (no real API call made)", file=sys.stderr)
    if mock_response is not None:
        return json.dumps(mock_response)
    return json.dumps({"mock": True, "note": "no mock_response provided to call_llm"})
