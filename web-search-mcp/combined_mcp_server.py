"""
combined_mcp_server.py — combined two-tool MCP server (Clark).

Serves BOTH tools from the project spec over a single stdio MCP connection:

  - `rag.search` (Shane) — private Amazon-2020 Household-Cleaning catalog
    hybrid retrieval. The tool body (`rag.rag_search`) is imported *unchanged*
    from src/rag/rag_search.py, per the note in
    mcp/rag_mcp_server.py.
  - `web.search` (Clark) — live web search, thin wrapper over
    orchestrator.search() defined locally in web_search.py.

Transport: stdio (default) — the standard MCP transport for local tools.
Discovery: tool names + JSON schemas are advertised automatically from the
typed signatures and docstrings below.

Run:
    # from web-search-mcp/ — no PYTHONPATH needed, this file adds
    # ../src to sys.path itself (mirrors rag_mcp_server.py).
    python combined_mcp_server.py            # stdio
Inspect with the MCP Inspector:
    npx @modelcontextprotocol/inspector python combined_mcp_server.py
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# make `import rag` work whether run from repo root or from web-search-mcp/.
# web-search-mcp/ and src/ are sibling directories directly under the repo
# root, so climb one level up from this file and back down into src/ (mirrors
# the relative path used by mcp/rag_mcp_server.py for its own directory).
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")),
)

# rag.config calls load_dotenv() with no arguments, which only walks *upward*
# from the current working directory looking for a .env file. Since
# web-search-mcp/ and the repo root are siblings-of-a-sibling (not
# ancestor/descendant from web-search-mcp/'s perspective), running this file
# with CWD=web-search-mcp/ (the documented way to run it) means that upward
# search never finds the repo root's .env. Load it explicitly here, by path
# relative to this file, BEFORE importing anything from rag.* so get_config()
# below sees the right env vars regardless of CWD. This is a safe no-op if
# the repo root's .env doesn't exist yet (e.g. fresh checkout before
# .env.example has been copied over).
from dotenv import load_dotenv  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env")

# The repo root .env's data paths (INDEX_DIR, PROCESSED_DIR, RAW_CSV) are
# written as relative paths, meant to be resolved against the repo root as
# CWD (that's how scripts/build_index.sh and rag_mcp_server.py are documented
# to run). vectorstore.py resolves them against the *process's actual* CWD,
# though, so when this server runs from web-search-mcp/ (its own documented
# Run command), a relative INDEX_DIR=data/index silently points at
# web-search-mcp/data/index instead — confirmed by reproducing "Collection
# household_cleaning not found" with CWD=web-search-mcp/. Rewrite the three
# data-path env vars to be absolute (relative to the repo root, not CWD)
# before rag.config reads them, without touching config.py/vectorstore.py.
for _env_key in ("INDEX_DIR", "PROCESSED_DIR", "RAW_CSV"):
    _val = os.environ.get(_env_key)
    if _val and not os.path.isabs(_val):
        os.environ[_env_key] = str((_REPO_ROOT / _val).resolve())

from mcp.server.fastmcp import FastMCP  # noqa: E402

from rag.rag_search import rag_search as _rag_search  # noqa: E402
from rag.config import get_config  # noqa: E402

from web_search import web_search as _web_search  # noqa: E402

mcp = FastMCP("combined-tools")


def _ts() -> str:
    """UTC timestamp for log lines (ISO-8601, millisecond precision)."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


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
             "brand": str, "material": str, "category_contains": str}. Example:
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
        f"{_ts()} [rag.search] q={query!r} filters={clean} -> {out['count']} hits "
        f"in {(time.time() - t0) * 1000:.0f}ms",
        file=sys.stderr, flush=True,
    )
    return out


@mcp.tool(name="web.search")
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

        NOTE (known limitation / future work): price/availability are always
        None — see web_search.py for details.
    """
    t0 = time.time()
    out = await _web_search(query=query, k=max(1, min(int(k), 20)))
    latency_ms = (time.time() - t0) * 1000
    urls = [r.get("url") for r in out.get("results", []) if r.get("url")]
    # timestamped request/response logging (Grading: MCP logging) — stderr
    # keeps stdio clean.
    print(
        f"{_ts()} [web.search] q={query!r} k={k} -> {out['count']} hits "
        f"in {latency_ms:.0f}ms",
        file=sys.stderr, flush=True,
    )
    # source-URL logging, required specifically for web.search per
    # mcp/README_mcp_rag.md's Logging section.
    print(
        f"{_ts()} [web.search] sources={urls}",
        file=sys.stderr, flush=True,
    )
    return out


if __name__ == "__main__":
    cfg = get_config()
    print(
        f"{_ts()} [combined-tools] serving rag.search (collection {cfg.collection!r}, "
        f"{cfg.embedding_signature()}) and web.search over stdio",
        file=sys.stderr, flush=True,
    )
    mcp.run(transport="stdio")
