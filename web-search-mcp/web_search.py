"""
web_search.py — the body of the `web.search` MCP tool.

Clark's combined MCP server imports `web_search()` and exposes it as the
`web.search` tool alongside Shane's `rag.search`. It is a thin,
JSON-serializable wrapper over `orchestrator.search()` so the tool layer
stays dumb and the routing/provider logic stays in orchestrator.py.

Unlike Shane's `rag_search()` (sync, wraps a sync retriever), this wrapper is
`async def` because the underlying `orchestrator.search()` is itself async
(it awaits provider HTTP calls) and this module is meant to be awaited
directly from another async FastMCP tool function — see server.py's
`research()` tool for the existing call pattern (`return await
orchestrator.search(...)`). Wrapping it in `asyncio.run()` here would break
if the combined server's event loop is already running, so we stay async and
let the caller `await` us.

Return contract (matches prompts/retriever_tool_instructions.md):
    {
      "query": str,
      "count": int,
      "results": [
        {title, url, snippet, price, availability},
        ...
      ]
    }
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

# Providers (providers/*.py) read API keys straight from os.environ with no
# dotenv call of their own, and neither does orchestrator.py — so unless
# something loads web-search-mcp/.env explicitly, `cp .env.example .env` per
# README_mcp_web.md silently has no effect and every provider init fails with
# a missing-API-key warning. Load it here, by path relative to this file, so
# it works regardless of caller/CWD (combined_mcp_server.py imports this
# module, and server.py / running this file directly both benefit too).
from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parent / ".env")

import orchestrator  # noqa: E402

# --- Request-level TTL cache + rate limit -----------------------------------
#
# This is separate from usage_tracker.py's monthly-quota bookkeeping (calls
# vs. limit per provider, persisted to disk). This is a simple in-process
# cache/throttle over web_search() calls themselves: avoid re-hitting a
# provider for a query asked again within a short window, and avoid firing
# calls back-to-back with no gap. In-memory only, no external dependency —
# matches how usage_tracker.py keeps its own persistence simple.

WEB_SEARCH_CACHE_TTL = float(os.environ.get("WEB_SEARCH_CACHE_TTL", 120))
WEB_SEARCH_MIN_INTERVAL = float(os.environ.get("WEB_SEARCH_MIN_INTERVAL", 1.0))

# {(normalized_query, k): (expires_at, result_dict)}
_cache: dict[tuple[str, int], tuple[float, dict]] = {}
_last_call_at: float = 0.0


def _cache_key(query: str, k: int) -> tuple[str, int]:
    return (query.strip().lower(), k)


async def web_search(query: str, k: int = 5, use_cache: bool = True) -> dict:
    """Search the live web for current product info (price/availability/general facts).

    Use this for CURRENT, live-web facts — not grounded in the private
    Amazon-2020 catalog. Prefer `rag.search` unless the user asks for
    *current* price/availability or something outside the private catalog.

    Args:
        query: natural-language search query, e.g. "current price of
            OXO Good Grips stainless steel cleaner".
        k: number of results to return (default 5).
        use_cache: if True (default), serve/store results in the in-memory
            TTL cache (see WEB_SEARCH_CACHE_TTL). Pass False to force a
            fresh provider call, bypassing the cache entirely.

    Returns:
        dict with keys: query, count, results (list of dicts with
        title, url, snippet, price, availability). `url` is the citation.

        NOTE (known limitation / future work): the underlying providers
        (Exa/Brave/Tavily/Gemini/Linkup/Newsdata) are general-purpose web
        search, not product-specific scrapers, so `price` and
        `availability` are always `None` here rather than fabricated —
        a future pass could parse them out of `snippet` text or run an
        Apify extract (see providers/apify_provider.py) on the returned
        product `url`s.
    """
    global _last_call_at

    key = _cache_key(query, k)
    if use_cache:
        cached = _cache.get(key)
        if cached and cached[0] > time.monotonic():
            print(f"[web_search] cache hit for query={query!r} k={k}", file=sys.stderr)
            return cached[1]

    # Simple back-to-back throttle: don't fire a provider call sooner than
    # WEB_SEARCH_MIN_INTERVAL seconds after the previous one.
    elapsed = time.monotonic() - _last_call_at
    if elapsed < WEB_SEARCH_MIN_INTERVAL:
        wait = WEB_SEARCH_MIN_INTERVAL - elapsed
        print(f"[web_search] rate-limit: sleeping {wait:.2f}s before provider call", file=sys.stderr)
        await asyncio.sleep(wait)
    _last_call_at = time.monotonic()

    raw = await orchestrator.search(query, mode="auto", max_results=k)

    # orchestrator.search() passes max_results through to the provider as a
    # request hint, not a hard cap it enforces itself — slice defensively so
    # `count` always matches the `k` contract regardless of provider behavior.
    sources = (raw.get("sources") or [])[:k]
    results = [
        {
            "title": s.get("title"),
            "url": s.get("url"),
            "snippet": s.get("snippet"),
            "price": None,  # not extractable from general web search results yet
            "availability": None,  # not extractable from general web search results yet
        }
        for s in sources
    ]

    result = {"query": query, "count": len(results), "results": results}

    if use_cache:
        _cache[key] = (time.monotonic() + WEB_SEARCH_CACHE_TTL, result)

    return result


if __name__ == "__main__":
    import json

    out = asyncio.run(web_search("current price of OXO Good Grips stainless steel cleaner", k=3))
    print(json.dumps(out, indent=2))

    # --- Offline self-test: cache + rate-limit, no real API calls ----------
    # Monkeypatch orchestrator.search with a fast fake that counts calls, so
    # we can verify the cache actually prevents a second provider hit and
    # that the rate-limit throttle delays a rapid-fire second call.
    async def _self_test() -> None:
        call_count = 0

        async def fake_search(query, mode="auto", max_results=10):
            nonlocal call_count
            call_count += 1
            return {"sources": [{"title": "t", "url": "u", "snippet": "s"}]}

        orchestrator.search = fake_search  # type: ignore[assignment]
        _cache.clear()
        globals()["_last_call_at"] = 0.0

        # 1. First call: cache miss, hits fake provider once.
        r1 = await web_search("  Test Query  ", k=2)
        assert call_count == 1, f"expected 1 provider call, got {call_count}"
        assert r1["count"] == 1

        # 2. Same query (different case/whitespace): cache HIT, no new call.
        t0 = time.monotonic()
        r2 = await web_search("test query", k=2)
        dt = time.monotonic() - t0
        assert call_count == 1, f"expected cache hit to skip provider call, got {call_count} calls"
        assert r2 == r1
        assert dt < 0.5, f"cache hit should return ~instantly, took {dt:.2f}s"

        # 3. use_cache=False forces a fresh call even for the same query.
        r3 = await web_search("test query", k=2, use_cache=False)
        assert call_count == 2, f"expected use_cache=False to force a new call, got {call_count}"

        # 4. Different query, back-to-back: rate limit should delay it by
        #    roughly WEB_SEARCH_MIN_INTERVAL since the previous provider call.
        t1 = time.monotonic()
        await web_search("another query", k=2)
        dt2 = time.monotonic() - t1
        assert dt2 >= WEB_SEARCH_MIN_INTERVAL * 0.9, (
            f"expected rate-limit sleep of ~{WEB_SEARCH_MIN_INTERVAL}s, only waited {dt2:.2f}s"
        )
        assert call_count == 3

        print(f"\nAll __main__ self-tests passed. (provider calls made: {call_count})")

    asyncio.run(_self_test())
