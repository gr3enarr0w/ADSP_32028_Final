"""
rag_search.py — the body of the `rag.search` MCP tool.

Clark's MCP server imports `rag_search()` and exposes it as the `rag.search`
tool alongside `web.search`. It is a thin, JSON-serializable wrapper over the
HybridRetriever so the tool layer stays dumb and the retrieval logic stays here.

Return contract (matches the syllabus):
    {
      "query": str,
      "count": int,
      "results": [
        {sku, title, price, rating, brand, ingredients, doc_id, url, price_per_oz, score},
        ...
      ]
    }
"""
from __future__ import annotations

from typing import Optional

from .config import get_config
from .retrieval import get_retriever


def rag_search(query: str, k: int = 5, filters: Optional[dict] = None) -> dict:
    """Query the private Amazon-2020 Household-Cleaning index.

    Args:
        query: natural-language product need, e.g. "eco-friendly stainless
            steel cleaner under $15".
        k: number of products to return (default 5).
        filters: optional structured constraints from the Planner. Supported keys:
            price_max (float), price_min (float), min_rating (float),
            brand (str), material (str e.g. "stainless steel"),
            category_contains (str).

    Returns:
        dict with keys: query, count, results (list of product dicts with
        doc_id for citations).
    """
    cfg = get_config()
    retriever = get_retriever(cfg)
    docs = retriever.search(query, k=k, filters=filters)
    results = [r.to_dict() for r in retriever.as_results(docs)]
    return {"query": query, "count": len(results), "results": results}


if __name__ == "__main__":
    import json
    out = rag_search("eco-friendly stainless steel cleaner under $15",
                     k=3, filters={"price_max": 15, "material": "stainless steel"})
    print(json.dumps(out, indent=2))
