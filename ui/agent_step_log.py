"""
agent_step_log.py — reusable "agent step log" panel for the Streamlit UI
(Shane, cross-training with Alison's UI shell).

Two things live here:

1. `TraceBuilder` / `Step` — a tiny, dependency-free instrumentation API each
   LangGraph node calls to record what it did (plan, tool I/O, citations,
   timing). This is the contract between the graph and the UI.

2. `render_agent_step_log(trace, st)` — draws the panel: the transcript, each
   agent step as an expandable card (input → output, status, latency), and the
   citations/data-lineage block (private `doc_id`s + live links).

The renderer imports Streamlit only through the `st` argument, so this module
has NO hard UI dependency and can be unit-tested headlessly
(`render_markdown(trace)`).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

# node -> (emoji, human label)
NODE_STYLE = {
    "asr": ("🎙️", "ASR / Transcription"),
    "router": ("🧭", "Router · Intent"),
    "planner": ("🗺️", "Planner"),
    "retriever": ("📚", "Retriever · rag.search"),
    "web": ("🌐", "Web · web.search"),
    "reconcile": ("⚖️", "Reconcile"),
    "answerer": ("✍️", "Answerer"),
    "critic": ("🔎", "Critic · grounding/safety"),
    "tts": ("🔊", "TTS"),
}
STATUS_ICON = {"ok": "🟢", "warn": "🟡", "error": "🔴"}


@dataclass
class Step:
    node: str
    label: str = ""
    input: Any = None
    output: Any = None
    status: str = "ok"          # ok | warn | error
    elapsed_ms: float = 0.0
    note: str = ""

    def title(self) -> str:
        emoji, human = NODE_STYLE.get(self.node, ("•", self.node.title()))
        lbl = f" — {self.label}" if self.label else ""
        return f"{emoji} {human}{lbl}"


@dataclass
class Trace:
    query_text: str = ""
    transcript: str = ""
    steps: list = field(default_factory=list)
    answer_text: str = ""
    citations: list = field(default_factory=list)   # [{doc_id,title,url,source}]
    audio_path: Optional[str] = None
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


class TraceBuilder:
    """Threaded through the graph so every node appends its step.

    Example (inside a node):
        tb.add("retriever", label="rag.search", input=filters,
                output=results, elapsed_ms=dt, status="ok")
    """

    def __init__(self, transcript: str = "", query_text: str = ""):
        self.trace = Trace(transcript=transcript, query_text=query_text or transcript)

    def add(self, node: str, label: str = "", input: Any = None, output: Any = None,
            status: str = "ok", elapsed_ms: float = 0.0, note: str = "") -> Step:
        s = Step(node=node, label=label, input=input, output=output,
                 status=status, elapsed_ms=elapsed_ms, note=note)
        self.trace.steps.append(s)
        return s

    def set_answer(self, text: str, citations: list, audio_path: Optional[str] = None):
        self.trace.answer_text = text
        self.trace.citations = citations
        self.trace.audio_path = audio_path

    def finalize(self) -> Trace:
        self.trace.elapsed_ms = sum(s.elapsed_ms for s in self.trace.steps)
        return self.trace


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _as_trace(trace) -> Trace:
    if isinstance(trace, Trace):
        return trace
    # accept a plain dict (e.g. loaded from JSON)
    t = Trace(**{k: trace.get(k) for k in
                 ["query_text", "transcript", "answer_text", "citations",
                  "audio_path", "elapsed_ms"] if k in trace})
    t.steps = [Step(**s) if not isinstance(s, Step) else s for s in trace.get("steps", [])]
    return t


def _fmt(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, indent=2, ensure_ascii=False, default=str)
    return str(value)


def render_agent_step_log(trace, st) -> None:
    """Render the panel into a Streamlit container `st`."""
    t = _as_trace(trace)

    st.subheader("🧠 Agent step log")
    if t.transcript:
        st.caption("Transcript")
        st.info(t.transcript)

    n_steps = len(t.steps)
    total = t.elapsed_ms or sum(s.elapsed_ms for s in t.steps)
    c1, c2, c3 = st.columns(3)
    c1.metric("Steps", n_steps)
    c2.metric("Total latency", f"{total:.0f} ms")
    c3.metric("Citations", len(t.citations))

    for i, s in enumerate(t.steps, start=1):
        icon = STATUS_ICON.get(s.status, "⚪")
        with st.expander(f"{i}. {s.title()}  ·  {icon} {s.elapsed_ms:.0f} ms", expanded=False):
            if s.note:
                st.caption(s.note)
            if s.input is not None:
                st.markdown("**Input**")
                st.code(_fmt(s.input), language="json")
            if s.output is not None:
                st.markdown("**Output**")
                st.code(_fmt(s.output), language="json")

    if t.answer_text:
        st.subheader("✅ Answer")
        st.write(t.answer_text)

    if t.citations:
        st.subheader("🔗 Citations & data lineage")
        for c in t.citations:
            src = c.get("source", "private")
            badge = "🗄️ private" if src == "private" else "🌐 live"
            title = c.get("title", c.get("doc_id", "source"))
            url = c.get("url")
            did = c.get("doc_id", "")
            if url:
                st.markdown(f"- {badge} · [{title}]({url})  \n  `doc_id={did}`")
            else:
                st.markdown(f"- {badge} · {title}  \n  `doc_id={did}`")


def render_markdown(trace) -> str:
    """Headless renderer (tests / logs / non-Streamlit contexts)."""
    t = _as_trace(trace)
    lines = ["# Agent step log", ""]
    if t.transcript:
        lines += [f"**Transcript:** {t.transcript}", ""]
    for i, s in enumerate(t.steps, start=1):
        lines.append(f"## {i}. {s.title()}  ({STATUS_ICON.get(s.status,'')} {s.elapsed_ms:.0f} ms)")
        if s.note:
            lines.append(f"_{s.note}_")
        if s.input is not None:
            lines += ["**Input**", "```json", _fmt(s.input), "```"]
        if s.output is not None:
            lines += ["**Output**", "```json", _fmt(s.output), "```"]
        lines.append("")
    if t.answer_text:
        lines += ["## Answer", t.answer_text, ""]
    if t.citations:
        lines.append("## Citations")
        for c in t.citations:
            lines.append(f"- [{c.get('source','private')}] {c.get('title','')} "
                         f"(doc_id={c.get('doc_id','')}) {c.get('url','')}")
    return "\n".join(lines)
