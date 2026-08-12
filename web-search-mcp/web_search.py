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

Return contract (matches rag-system/prompts/retriever_tool_instructions.md):
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


async def web_search(query: str, k: int = 5) -> dict:
    """Search the live web for current product info (price/availability/general facts).

    Use this for CURRENT, live-web facts — not grounded in the private
    Amazon-2020 catalog. Prefer `rag.search` unless the user asks for
    *current* price/availability or something outside the private catalog.

    Args:
        query: natural-language search query, e.g. "current price of
            OXO Good Grips stainless steel cleaner".
        k: number of results to return (default 5).

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

    return {"query": query, "count": len(results), "results": results}


if __name__ == "__main__":
    import asyncio
    import json

    out = asyncio.run(web_search("current price of OXO Good Grips stainless steel cleaner", k=3))
    print(json.dumps(out, indent=2))
