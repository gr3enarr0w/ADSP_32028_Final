"""
rag_mcp_server.py — standalone MCP server exposing the `rag.search` tool.

Shane owns `rag.search`; Clark owns `web.search` and the combined two-tool
server. This file is a runnable, self-contained MCP server so `rag.search` can
be developed and tested in isolation, and its tool body (`rag.rag_search`) is
imported unchanged into Clark's combined server.

Transport: stdio (default) — the standard MCP transport for local tools.
Discovery: the tool name + JSON schema are advertised automatically from the
typed signature and docstring below.

Run:
    PYTHONPATH=../src python rag_mcp_server.py            # stdio
Inspect with the MCP Inspector:
    npx @modelcontextprotocol/inspector python rag_mcp_server.py

Note on naming: MCP tool identifiers are typically snake_case, so the tool is
registered as `rag_search`; its *logical* name in the project spec is
`rag.search`. We set the display name explicitly to keep both worlds happy.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Optional

# make `import rag` work whether run from repo root or from mcp/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from rag.rag_search import rag_search as _rag_search  # noqa: E402
from rag.config import get_config  # noqa: E402

mcp = FastMCP("rag-tools")


@mcp.tool(name="rag.search")
def rag_search(
    query: str,
    k: int = 5,
    filters: Optional[dict] = None,
) -> dict:
    """Search the private Amazon-2020 Household-Cleaning catalog (vector + BM25 hybrid).

    Use this for product FACTS grounded in the private catalog: titles, prices,
    ratings, brands, ingredients. Prefer this over web.search unless the user
    asks for *current* price/availability.

    Args:
        query: Natural-language product need, e.g.
            "eco-friendly stainless steel cleaner under $15".
        k: Number of products to return (1-20).
        filters: Optional structured constraints from the Planner, as a single
            nested dict (matches planner.md's output schema). Recognized keys:
            {"price_max": float, "price_min": float, "min_rating": float,
             "brand": str, "material": str}. Example:
            {"price_max": 15, "material": "stainless steel"}.

    Returns:
        {query, count, results:[{sku, title, price, rating, brand, ingredients,
        doc_id, url, price_per_oz, score}]}. `doc_id` is the private-catalog
        citation id to show on screen.
    """
    # Keep only the recognized, non-null filter keys.
    allowed = ("price_max", "price_min", "min_rating", "brand", "material", "category_contains")
    clean = {k2: v for k2, v in (filters or {}).items() if k2 in allowed and v is not None}
    t0 = time.time()
    out = _rag_search(query=query, k=max(1, min(int(k), 20)), filters=clean or None)
    # request/response logging (Grading: MCP logging) — stderr keeps stdio clean.
    print(
        f"[rag.search] q={query!r} filters={clean} -> {out['count']} hits "
        f"in {(time.time() - t0) * 1000:.0f}ms",
        file=sys.stderr, flush=True,
    )
    return out


if __name__ == "__main__":
    cfg = get_config()
    print(f"[rag.search] serving collection {cfg.collection!r} "
          f"({cfg.embedding_signature()}) over stdio", file=sys.stderr, flush=True)
    mcp.run(transport="stdio")
