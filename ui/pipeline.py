"""
pipeline.py — the one voice turn: audio → ASR → LangGraph → TTS → Trace.

This is the integration layer `docs/UI_WIREFRAME.md` names as "the remaining
integration work":

    "nothing yet stitches ASR → the compiled graph → tts.speak() → this UI
     into one Streamlit callback — that wiring (calling
     build_graph().invoke(...) from a Streamlit button handler and feeding
     its Steps into TraceBuilder) is the remaining integration work."

It deliberately lives *outside* `ui/app.py` so the same code path is exercised
by the headless harness (`scripts/run_end_to_end.py`) that the live app uses —
an end-to-end test that tested a different code path than the demo would be
worth very little.

Design constraints it respects:

* `src/rag/graph.py` stays the single source of truth for the graph. This
  module never re-wires nodes; it consumes `build_graph()` and observes the
  run via LangGraph's `astream(..., stream_mode="updates")`, which yields one
  `{node_name: state_delta}` per completed node — enough to time each node and
  turn it into a `Step` without touching graph topology.
* The graph contains an async node (`retriever_node`), so LangGraph requires
  the async entry points (`ainvoke`/`astream`) — see `tests/test_graph.py`'s
  note. `run_turn()` is the sync wrapper Streamlit button handlers want.
* Only `answerer_output["speech"]` is ever synthesized, and only once the
  Critic returns `action == "accept"` — per `prompts/answerer_critic.md` and
  `docs/UI_WIREFRAME.md` region [5] ("citations and the comparison table are
  deliberately screen-only, never read aloud").
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Any, Optional

# `src/` is not installed as a package — every entry point in this repo puts it
# on the path (PYTHONPATH=src in the Makefile, sys.path juggling in
# web-search-mcp/combined_mcp_server.py). Streamlit is launched as
# `streamlit run ui/app.py`, which does NOT inherit the Makefile's export, so
# do it here rather than making the run command fragile.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from agent_step_log import Trace, TraceBuilder  # noqa: E402  (same dir as this file)

from rag.asr import ASRUnavailableError, transcribe  # noqa: E402
from rag.graph import build_graph  # noqa: E402
from rag.tts import fits_budget, speak  # noqa: E402


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


def _citations_from(answerer_output: dict, reconciled: dict) -> list[dict]:
    """Merge the Answerer's citations with live provenance from reconciliation.

    `prompts/answerer_critic.md` makes the Answerer responsible for citing the
    private catalog (`doc_id`). The *live* half of the lineage the assignment
    asks for ("private doc IDs + live links") only exists on the reconciled
    items' `live_match.url`, which the Answerer payload has no field for — so
    it is appended here rather than asked of the model, which would invite it
    to invent URLs.
    """
    citations = []
    seen: set[tuple] = set()

    for c in answerer_output.get("citations") or []:
        key = (c.get("doc_id"), c.get("url"))
        if key in seen:
            continue
        seen.add(key)
        citations.append({**c, "source": c.get("source") or "private"})

    for item in (reconciled or {}).get("items", []):
        live = item.get("live_match") or {}
        url = live.get("url")
        if not url or (None, url) in seen or (item.get("doc_id"), url) in seen:
            continue
        seen.add((None, url))
        citations.append({
            "doc_id": item.get("doc_id"),
            "title": item.get("title"),
            "url": url,
            "source": "live",
        })

    return citations


def _add_retriever_steps(tb: TraceBuilder, delta: dict, elapsed_ms: float,
                         planner_output: dict) -> None:
    """Split the retriever node's single delta into the three steps the step-log
    styles separately (`retriever`, `web`, `reconcile` in `NODE_STYLE`).

    The three sub-calls happen inside one LangGraph node, so LangGraph reports
    one combined duration; it is attributed to the `retriever` step and the
    derived steps are recorded at 0 ms rather than inventing a split.
    """
    rag_results = delta.get("rag_results") or {}
    web_results = delta.get("web_results") or {}
    reconciled = delta.get("reconciled") or {}

    tb.add(
        "retriever", label="rag.search", elapsed_ms=elapsed_ms,
        input={"query": planner_output.get("query"),
               "k": planner_output.get("k"),
               "filters": planner_output.get("filters")},
        output={"count": rag_results.get("count"),
                "results": rag_results.get("results", [])},
        status="ok" if rag_results.get("count") else "warn",
        note="" if rag_results.get("count") else "No private-catalog matches.",
    )

    if planner_output.get("call_web_search"):
        count = web_results.get("count", 0)
        tb.add(
            "web", label="web.search",
            input={"query": web_results.get("query")},
            output={"count": count, "results": web_results.get("results", [])},
            status="ok" if count else "warn",
            note="" if count else (
                "Live search returned nothing — no provider key configured, or the "
                "routed provider failed. Answer falls back to private catalog only "
                "(see web-search-mcp/HANDOFF.md, 'Known gaps')."
            ),
        )

        flagged = [i for i in reconciled.get("items", []) if i.get("discrepancy")]
        tb.add(
            "reconcile", label="private ↔ live",
            input={"rag_items": len(rag_results.get("results", [])),
                   "web_items": len(web_results.get("results", []))},
            output={"matched": sum(1 for i in reconciled.get("items", [])
                                   if i.get("live_match")),
                    "discrepancies": [
                        {"title": i.get("title"), **i["discrepancy"]} for i in flagged
                    ],
                    "unmatched_web": len(reconciled.get("unmatched_web", []))},
            status="warn" if flagged else "ok",
            note="Private fact always wins; discrepancies are flagged, never applied."
                 if flagged else "",
        )


async def _arun_turn(transcript: str, tb: TraceBuilder,
                     synthesize: bool = True) -> tuple[Trace, dict]:
    """Drive the compiled graph, recording one Step per node, then speak."""
    compiled = build_graph()
    state: dict[str, Any] = {"transcript": transcript}
    final_state: dict[str, Any] = dict(state)

    t0 = time.perf_counter()
    async for update in compiled.astream(state, stream_mode="updates"):
        elapsed = _ms(t0)
        for node_name, delta in update.items():
            # LangGraph's update deltas are the node's full returned state in
            # this graph (each node mutates and returns `state`), so read the
            # keys that node owns rather than assuming a minimal diff.
            final_state.update(delta or {})

            if node_name == "router":
                tb.add("router", label="intent + constraints", elapsed_ms=elapsed,
                       input={"transcript": transcript},
                       output=delta.get("router_output"),
                       status="warn" if (delta.get("router_output") or {}).get("safety_flags") else "ok",
                       note="Safety flags raised — Critic will enforce."
                            if (delta.get("router_output") or {}).get("safety_flags") else "")
            elif node_name == "planner":
                tb.add("planner", label="source + filter plan", elapsed_ms=elapsed,
                       input=final_state.get("router_output"),
                       output=delta.get("planner_output"))
            elif node_name == "retriever":
                _add_retriever_steps(tb, delta, elapsed,
                                     final_state.get("planner_output") or {})
            elif node_name == "answerer_critic":
                answerer_output = delta.get("answerer_output") or {}
                critic_output = delta.get("critic_output") or {}
                revision = (delta.get("revise_count") or 0) > 0
                tb.add("answerer", label="revision" if revision else "",
                       elapsed_ms=elapsed,
                       input={"reconciled_items": len((final_state.get("reconciled") or {}).get("items", []))},
                       output=answerer_output)
                tb.add("critic", label=critic_output.get("action", ""),
                       input={"speech": answerer_output.get("speech")},
                       output=critic_output,
                       status="ok" if critic_output.get("action") == "accept" else "warn",
                       note="; ".join(critic_output.get("reasons") or []))
            else:  # pragma: no cover - future nodes show up rather than vanish
                tb.add(node_name, elapsed_ms=elapsed, output=delta)
        t0 = time.perf_counter()

    answerer_output = final_state.get("answerer_output") or {}
    critic_output = final_state.get("critic_output") or {}
    reconciled = final_state.get("reconciled") or {}
    speech = answerer_output.get("speech", "")

    audio_path: Optional[str] = None
    if synthesize and speech and critic_output.get("action") == "accept":
        t_tts = time.perf_counter()
        try:
            out = speak(speech)
            audio_path = str(out)
            tb.add("tts", label="speak()", elapsed_ms=_ms(t_tts),
                   input={"speech": speech,
                          "fits_15s_budget": fits_budget(speech)},
                   output={"audio_path": audio_path})
        except Exception as e:  # noqa: BLE001 - a mute demo beats a crashed demo
            tb.add("tts", label="speak()", elapsed_ms=_ms(t_tts),
                   input={"speech": speech}, output={"error": repr(e)},
                   status="error",
                   note="TTS failed; the written answer below is unaffected.")
    elif synthesize and speech:
        tb.add("tts", label="skipped", input={"speech": speech},
               output={"reason": f"critic action = {critic_output.get('action')!r}"},
               status="warn",
               note="Nothing is spoken until the Critic accepts the answer.")

    tb.set_answer(speech, _citations_from(answerer_output, reconciled), audio_path)
    return tb.finalize(), final_state


def run_turn(transcript: Optional[str] = None,
             audio_path: Optional[str | Path] = None,
             synthesize: bool = True,
             allow_asr_fallback: Optional[bool] = None) -> tuple[Trace, dict]:
    """Run one complete voice turn and return (Trace, final graph state).

    Args:
        transcript: typed/known text. Skips ASR entirely when given without
            `audio_path`.
        audio_path: a finished audio file to transcribe first. If both are
            given, ASR runs and its transcript wins (the typed one is treated
            as a reference, not an override).
        synthesize: call `rag.tts.speak()` on the accepted answer.
        allow_asr_fallback: forwarded to `rag.asr.transcribe()`.

    Raises:
        ValueError: if neither `transcript` nor `audio_path` is provided.
    """
    if not transcript and not audio_path:
        raise ValueError("run_turn() needs either a transcript or an audio_path.")

    tb = TraceBuilder(transcript=transcript or "")

    if audio_path:
        t0 = time.perf_counter()
        try:
            asr_output = transcribe(audio_path, allow_fallback=allow_asr_fallback)
            transcript = asr_output["text"]
            tb.trace.transcript = transcript
            tb.trace.query_text = transcript
            # "faster-whisper:small.en" → "faster-whisper · small.en": the step
            # title is rendered as Streamlit markdown, which eats a bare colon
            # followed by a word as a directive/shortcode (observed: the label
            # displayed as "faster-whisper.en", losing the model size).
            engine_label = (asr_output.get("engine") or "").replace(":", " · ")
            tb.add("asr", label=engine_label, elapsed_ms=_ms(t0),
                   input={"source_file": str(audio_path)},
                   output={"text": transcript,
                           "language": asr_output.get("language"),
                           "language_probability": asr_output.get("language_probability"),
                           "n_segments": len(asr_output.get("segments", []))},
                   status="warn" if asr_output.get("engine") == "fallback-reference" else "ok",
                   note="Reference transcript, not a live ASR run."
                        if asr_output.get("engine") == "fallback-reference" else "")
        except (ASRUnavailableError, FileNotFoundError) as e:
            tb.add("asr", elapsed_ms=_ms(t0), input={"source_file": str(audio_path)},
                   output={"error": repr(e)}, status="error")
            if not transcript:
                raise

    return asyncio.run(_arun_turn(transcript, tb, synthesize=synthesize))
