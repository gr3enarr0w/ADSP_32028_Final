#!/usr/bin/env python3
"""
CLI / Function for retrieving from Efficient Hermes RAG.
Returns formatted context ready for LLM prompting + structured results.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

sys.path.append(str(Path(__file__).parent))
from utils import EfficientRAG, load_config  # reuse load_config if moved


def format_context(results: List[Dict[str, Any]], query: str, retrieval_skipped: bool = False) -> str:
    """Format retrieved chunks into clean context for Hermes prompt.

    When parent-chunk expansion is enabled (retrieval.fetch_parents in
    config.yaml), each result's "text" field holds the fuller parent-chunk
    text rather than just the terse child chunk that matched the search —
    that's the point of hierarchical retrieval: small chunks for precise
    search, larger parents for full context handed to the LLM. The child
    chunk that actually matched is still available under "child_text" (when
    present) and is shown separately when it differs from "text". Older
    result shapes without a "child_text" key are handled gracefully (the
    field is simply omitted, no crash).

    `retrieval_skipped` (set when the Adaptive-RAG-lite gate — see
    EfficientRAG._classify_retrieval_need()/retrieve() in utils.py —
    classified the query as "none" and deliberately skipped retrieval
    entirely) is handled distinctly from "retrieval ran and found zero
    hits": an empty `results` list with `retrieval_skipped=True` produces a
    message explicitly about the query being answerable from general
    knowledge, not the generic "no relevant context found" message used
    when retrieval genuinely came up empty.
    """
    if not results:
        if retrieval_skipped:
            return (
                "No retrieval performed — query classified as answerable from general "
                "knowledge (Adaptive-RAG-lite gate, retrieval.adaptive_retrieval_enabled)."
            )
        return "No relevant context found in knowledge base."

    context_parts = [f"### Retrieved Context for Query: {query}\n"]
    for i, r in enumerate(results, 1):
        src = r.get("source", "unknown")
        score = r.get("score") or 0
        text = r.get("text", "")
        child_text = r.get("child_text")
        summary = r.get("summary", "")
        tags = ", ".join(r.get("tags", []))

        part = f"""
**Result {i}** (Score: {score:.3f} | Source: {src})
"""
        if summary:
            part += f"Summary: {summary}\n"
        if child_text and child_text != text:
            part += f"Matched excerpt: {child_text[:400]}...\n"
        part += f"Context: {text[:2000]}...\n"
        if tags:
            part += f"Tags: {tags}\n"
        context_parts.append(part)

    context_parts.append("\n--- End of Retrieved Context ---")
    return "\n".join(context_parts)


def retrieve_context(query: str, collection: str, config: Optional[Dict] = None, 
                     top_k: int = 8, filters: Optional[Dict] = None) -> Dict[str, Any]:
    """High-level function for Hermes to call via execute_code."""
    if config is None:
        config = load_config()  # fallback

    rag = EfficientRAG(config)
    results = rag.retrieve(query, collection, top_k=top_k, filters=filters)

    # Adaptive-RAG-lite: read the gate's outcome AFTER calling retrieve() — set on the
    # EfficientRAG instance rather than returned from retrieve() itself, so retrieve()'s
    # return type/shape is unchanged for every existing caller. None means the gate is
    # off (retrieval.adaptive_retrieval_enabled: false, the default); "none" means the
    # gate ran and deliberately skipped retrieval; "light" means the gate ran and
    # retrieval proceeded normally (with adaptive_light_top_k/oversampling applied).
    retrieval_strategy: Optional[str] = getattr(rag, "_last_retrieval_strategy", None)
    retrieval_skipped = retrieval_strategy == "none"

    formatted = format_context(results, query, retrieval_skipped=retrieval_skipped)

    return {
        "query": query,
        "collection": collection,
        "num_results": len(results),
        "formatted_context": formatted,
        "raw_results": results,  # For advanced use / citations
        "retrieval_skipped": retrieval_skipped,
        "retrieval_strategy": retrieval_strategy
    }


def main():
    parser = argparse.ArgumentParser(description="Retrieve from Efficient RAG")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--collection", required=True)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--filters", help="JSON filter dict, e.g. '{\"tags\": \"work\"}'")
    parser.add_argument("--config", help="config.yaml path")
    parser.add_argument("--json_output", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    config = load_config(args.config)
    filters = json.loads(args.filters) if args.filters else None

    result = retrieve_context(args.query, args.collection, config=config, top_k=args.top_k, filters=filters)

    if args.json_output:
        print(json.dumps(result, indent=2))
    else:
        print(result["formatted_context"])
        print(f"\nRetrieved {result['num_results']} chunks from '{args.collection}'.")


if __name__ == "__main__":
    main()
