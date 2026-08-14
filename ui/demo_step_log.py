"""
demo_step_log.py — standalone demo of the agent step-log panel, wired to the
REAL `rag.search` retriever so the panel is proven against live data.

Run the UI:
    PYTHONPATH=src streamlit run ui/demo_step_log.py

Generate a static sample trace (no Streamlit needed):
    PYTHONPATH=src python ui/demo_step_log.py --dump ui/sample_trace.json

The Router/Planner/Answerer logic here is intentionally lightweight — those
nodes are owned by Alison (Router) and Victoria (Planner/Answerer). The demo
uses just enough of each to exercise the panel end-to-end and to document the
`TraceBuilder` contract they instrument.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from agent_step_log import TraceBuilder, render_agent_step_log  # noqa: E402

MATERIALS = ["stainless steel", "glass", "wood", "tile", "grout", "chrome",
             "porcelain", "ceramic", "fabric", "multi-surface"]


def _lightweight_router(text: str) -> dict:
    """Placeholder intent extraction (Alison owns the real Router node)."""
    t = text.lower()
    price_max = None
    m = re.search(r"(?:under|below|less than|<)\s*\$?\s*(\d+(?:\.\d+)?)", t)
    if not m:
        m = re.search(r"\$\s*(\d+(?:\.\d+)?)", t)
    if m:
        price_max = float(m.group(1))
    # spelled-out "under fifteen dollars"
    words = {"five": 5, "ten": 10, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30}
    for w, v in words.items():
        if re.search(rf"(?:under|below|less than)\s+{w}", t):
            price_max = float(v)
    material = next((mat for mat in MATERIALS if mat in t), None)
    eco = any(k in t for k in ["eco", "eco-friendly", "plant-based", "natural", "green", "non-toxic"])
    return {"task": "product_recommendation", "constraints": {
        "price_max": price_max, "material": material, "eco_preference": eco},
        "safety_flags": []}


def _lightweight_planner(intent: dict) -> dict:
    """Placeholder plan (Victoria owns the real Planner node)."""
    c = intent["constraints"]
    filters = {}
    if c.get("price_max") is not None:
        filters["price_max"] = c["price_max"]
    if c.get("material"):
        filters["material"] = c["material"]
    return {
        "sources": ["rag.search"],           # private-first
        "call_web_search": False,            # set True if user asked for 'current' price/stock
        "filters": filters,
        "comparison_criteria": ["price", "rating", "price_per_oz", "ingredients"],
        "k": 3,
    }


def _lightweight_answer(query: str, results: list) -> tuple[str, list]:
    """Placeholder Answerer (Victoria owns the real node). Builds a short,
    cited recommendation + a citations list for the lineage block."""
    if not results:
        return "I couldn't find a matching product in the catalog.", []
    top = results[0]
    others = results[1:3]
    price = f"${top['price']:.2f}" if top.get("price") is not None else "an unlisted price"
    rating = f"{top['rating']}★" if top.get("rating") is not None else "unrated"
    alt = ", ".join(o["title"].split(",")[0] for o in others) or "no close alternatives"
    text = (f"My top pick is {top['title'].split(',')[0]} — {rating} average rating, "
            f"typically {price}. I compared it with {alt}. "
            f"Details and sources are on your screen. Want the most affordable or the highest rated?")
    citations = [{
        "doc_id": r["doc_id"], "title": r["title"], "url": r.get("url"), "source": "private",
    } for r in results[:3]]
    return text, citations


def build_demo_trace(query: str) -> dict:
    from rag.rag_search import rag_search

    tb = TraceBuilder(transcript=query, query_text=query)

    t0 = time.time()
    intent = _lightweight_router(query)
    tb.add("router", label="intent + constraints", input={"transcript": query},
           output=intent, elapsed_ms=(time.time() - t0) * 1000)

    t0 = time.time()
    plan = _lightweight_planner(intent)
    tb.add("planner", label="private-first plan", input=intent, output=plan,
           elapsed_ms=(time.time() - t0) * 1000,
           note="Prefer rag.search; add web.search only for current price/availability.")

    t0 = time.time()
    out = rag_search(query, k=plan["k"], filters=plan["filters"] or None)
    tb.add("retriever", label="rag.search", input={"query": query, "filters": plan["filters"]},
           output=out, elapsed_ms=(time.time() - t0) * 1000,
           status="ok" if out["count"] else "warn")

    t0 = time.time()
    answer, citations = _lightweight_answer(query, out["results"])
    tb.add("answerer", label="cited recommendation",
           input={"n_candidates": out["count"]},
           output={"answer": answer, "n_citations": len(citations)},
           elapsed_ms=(time.time() - t0) * 1000)

    # grounding check (Critic): every citation must come from the retrieved set
    retrieved_ids = {r["doc_id"] for r in out["results"]}
    grounded = all(c["doc_id"] in retrieved_ids for c in citations)
    tb.add("critic", label="grounding + safety",
           input={"cited": [c["doc_id"] for c in citations]},
           output={"grounded": grounded, "unsafe": False},
           status="ok" if grounded else "error")

    tb.set_answer(answer, citations)
    return tb.finalize().to_dict()


DEFAULT_QUERY = "Recommend an eco-friendly stainless steel cleaner under fifteen dollars."


def _run_streamlit():
    import streamlit as st

    st.set_page_config(page_title="Agent Step Log", page_icon="🧠", layout="wide")
    st.title("Voice-to-Voice Assistant — Agent Step Log (demo)")
    st.caption("Owner: Shane · reusable panel for the shared Streamlit UI")

    query = st.text_input("Simulated transcript (what the user said):", DEFAULT_QUERY)
    if st.button("Run pipeline", type="primary") or query:
        with st.spinner("Running agent graph..."):
            trace = build_demo_trace(query)
        render_agent_step_log(trace, st)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", metavar="PATH", help="Write a sample trace JSON and exit")
    ap.add_argument("--query", default=DEFAULT_QUERY)
    args, _ = ap.parse_known_args()
    if args.dump:
        trace = build_demo_trace(args.query)
        with open(args.dump, "w") as f:
            json.dump(trace, f, indent=2, default=str)
        print(f"wrote {args.dump}")
        return
    _run_streamlit()


if __name__ == "__main__":
    main()
