"""
app.py — the Streamlit demo app (Final deliverable: "User Interface").

Builds every region of `docs/UI_WIREFRAME.md` Screen 1, in the order that
wireframe draws them:

    [1] mic capture → [2] live transcript → [3] agent step log →
    [5] spoken answer + Play TTS → [4] comparison table → [6] citations & lineage

Run it:

    PYTHONPATH=src streamlit run ui/app.py

(`ui/pipeline.py` also puts `src/` on `sys.path` itself, so the PYTHONPATH
prefix is belt-and-braces rather than required.)

Division of labour, deliberately: this module is *only* presentation and
Streamlit state. Every bit of "what happens in a turn" lives in
`ui/pipeline.run_turn()`, which the headless harness
(`scripts/run_end_to_end.py`) drives too — so what the demo does and what the
end-to-end test verifies cannot drift apart.

Region [3] is a direct embed of `ui.agent_step_log.render_agent_step_log()`,
which already existed and was demoed by `ui/demo_step_log.py`. That module's
Router/Planner were "intentionally lightweight regex placeholders"; this app
drives the real compiled LangGraph instead, which is the swap
`docs/UI_WIREFRAME.md` called for under "Known gaps / next steps".
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_HERE.parent / "src") not in sys.path:
    sys.path.insert(0, str(_HERE.parent / "src"))

from agent_step_log import render_agent_step_log  # noqa: E402
from pipeline import run_turn  # noqa: E402

from rag.asr import REFERENCE_TRANSCRIPTS, sample_clips  # noqa: E402
from rag.config import get_config  # noqa: E402

st.set_page_config(page_title="Voice-to-Voice Product Assistant",
                   page_icon="🎙️", layout="wide")

# Keeps the "à la demo_step_log.py" header convention the wireframe cites.
st.title("🎙️ Voice-to-Voice Product Assistant")
st.caption(
    "Ask for a product out loud. Router → Planner → Retriever (private catalog "
    "+ live web) → Answerer/Critic → spoken summary, with citations."
)

if "trace" not in st.session_state:
    st.session_state.trace = None
    st.session_state.final_state = None
    st.session_state.history = []


# ---------------------------------------------------------------------------
# Sidebar — session-level controls only (per the wireframe: never per-turn)
# ---------------------------------------------------------------------------
cfg = get_config()
with st.sidebar:
    st.header("Session")
    st.markdown(
        f"- **ASR model:** `{cfg.asr_model}` ({cfg.asr_device})\n"
        f"- **TTS provider:** `{cfg.tts_provider}`\n"
        f"- **LLM:** `{cfg.llm_provider}` / `{cfg.llm_model}`\n"
        f"- **Embeddings:** `{cfg.embedding_signature()}`\n"
        f"- **Vector store:** `{cfg.vector_store}` · `{cfg.collection}`"
    )
    speak_answer = st.toggle("Synthesize spoken answer", value=True)
    allow_asr_fallback = st.toggle(
        "Allow reference-transcript fallback", value=False,
        help="If faster-whisper isn't installed, use the known transcript for "
             "the prerecorded audio/queryNN.wav clips. Off by default so the "
             "demo never silently fakes a transcript.",
    )

    if st.session_state.history:
        st.subheader("Past turns")
        for i, past in enumerate(reversed(st.session_state.history[-8:]), start=1):
            st.caption(f"{i}. {past}")


# ---------------------------------------------------------------------------
# [1] Mic capture
# ---------------------------------------------------------------------------
st.subheader("🎤 1 · Ask")

audio_bytes = None
audio_path: Path | None = None
transcript_text: str | None = None

# A radio, not tabs: Streamlit renders *every* tab body on every rerun and
# gives no way to ask which one is active, so a tabbed layout would leave the
# sample-clip selectbox silently setting an audio path while the user typed a
# question in another tab — the typed text would then be ignored. (Caught in
# browser testing: a typed query ran ASR on query01.wav instead.) One
# explicit mode selector makes the active input unambiguous.
mode = st.radio(
    "Input", ["🎤 Record", "📁 Upload audio", "🎧 Sample clip", "⌨️ Type instead"],
    horizontal=True, label_visibility="collapsed",
)

if mode == "🎤 Record":
    if hasattr(st, "audio_input"):
        rec = st.audio_input("Record your question")
        if rec is not None:
            audio_bytes = rec.getvalue()
    else:  # pragma: no cover - older Streamlit
        st.info(
            "This Streamlit build has no `st.audio_input` (needs ≥1.38). "
            "Use **Upload audio**, or upgrade Streamlit."
        )

elif mode == "📁 Upload audio":
    up = st.file_uploader("WAV / MP3 / M4A / WebM / FLAC / OGG",
                          type=["wav", "mp3", "m4a", "webm", "flac", "ogg"])
    if up is not None:
        audio_bytes = up.getvalue()

elif mode == "🎧 Sample clip":
    clips = sample_clips()
    if clips:
        names = [c.name for c in clips]
        chosen = st.selectbox(
            "Prerecorded demo clip", names,
            format_func=lambda n: f"{n} — {REFERENCE_TRANSCRIPTS.get(n, '')}",
        )
        audio_path = next(c for c in clips if c.name == chosen)
        st.audio(str(audio_path))
    else:
        st.info("No clips found in `audio/`.")

else:
    typed = st.text_input(
        "Question", placeholder="Recommend an eco-friendly stainless-steel cleaner under fifteen dollars.")
    if typed.strip():
        transcript_text = typed.strip()

# Recorded/uploaded bytes have to become a real file: rag.asr.transcribe()
# is fragment-based and takes a finished path, matching the spec's
# "record → send file to ASR".
if audio_bytes:
    suffix = ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(audio_bytes)
    tmp.close()
    audio_path = Path(tmp.name)

ready = bool(audio_path or transcript_text)
if st.button("▶︎ Run the assistant", type="primary", disabled=not ready):
    with st.spinner("Listening, planning, retrieving, answering…"):
        try:
            trace, final_state = run_turn(
                transcript=transcript_text,
                audio_path=audio_path,
                synthesize=speak_answer,
                allow_asr_fallback=allow_asr_fallback,
            )
            st.session_state.trace = trace
            st.session_state.final_state = final_state
            if trace.transcript:
                st.session_state.history.append(trace.transcript)
        except Exception as e:  # noqa: BLE001 - surface it in the UI, don't 500
            st.session_state.trace = None
            st.error(f"Turn failed: {e}")

trace = st.session_state.trace
final_state = st.session_state.final_state or {}

if trace is None:
    st.info("Record, upload, pick a sample clip, or type a question — then run.")
    st.stop()


# ---------------------------------------------------------------------------
# [2] Live transcript
# ---------------------------------------------------------------------------
st.subheader("📝 2 · Transcript")
st.info(trace.transcript or "—")

# ---------------------------------------------------------------------------
# [3] Agent step log (existing component, now driven by the real graph)
# ---------------------------------------------------------------------------
st.divider()
render_agent_step_log(trace, st)

# ---------------------------------------------------------------------------
# [5] Spoken answer + Play TTS
# ---------------------------------------------------------------------------
st.divider()
st.subheader("🔊 5 · Spoken answer")
critic = final_state.get("critic_output") or {}
if critic.get("action") == "accept":
    st.success(trace.answer_text or "—")
else:
    st.warning(
        f"Critic verdict: `{critic.get('action')}` — "
        f"{'; '.join(critic.get('reasons') or ['no reason given'])}"
    )
    st.write(trace.answer_text or "—")

if trace.audio_path and Path(trace.audio_path).exists():
    st.audio(str(trace.audio_path))
    st.caption(f"`{trace.audio_path}` · ≤15s spoken summary "
               "(citations and the table are screen-only, never read aloud)")
elif speak_answer:
    st.caption("No audio for this turn — see the TTS step above.")

# ---------------------------------------------------------------------------
# [4] Comparison table
# ---------------------------------------------------------------------------
st.divider()
st.subheader("📊 4 · Comparison")

answerer_output = final_state.get("answerer_output") or {}
rows = answerer_output.get("comparison_table") or []
if rows:
    df = pd.DataFrame(rows)

    # Join in the live half of the picture, per the wireframe's region-[6]
    # stretch goal: live price and any flagged discrepancy, keyed on doc_id.
    live_by_doc = {
        i.get("doc_id"): i for i in (final_state.get("reconciled") or {}).get("items", [])
    }
    if "doc_id" in df.columns:
        df["live_price"] = df["doc_id"].map(
            lambda d: (live_by_doc.get(d, {}).get("live_match") or {}).get("price"))
        df["flag"] = df["doc_id"].map(
            lambda d: (live_by_doc.get(d, {}).get("discrepancy") or {}).get("detail"))
        if df["live_price"].isna().all() and df["flag"].isna().all():
            df = df.drop(columns=["live_price", "flag"])

    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption("Private catalog is the grounded baseline. Live prices are shown "
               "alongside and flagged when they differ by >10% — never merged in.")
else:
    st.info("No comparison rows for this turn.")

# ---------------------------------------------------------------------------
# [6] Citations & data lineage (promoted out of the step-log panel)
# ---------------------------------------------------------------------------
st.divider()
st.subheader("🔗 6 · Citations & data lineage")
if trace.citations:
    for c in trace.citations:
        badge = "🗄️ **private**" if c.get("source") == "private" else "🌐 **live**"
        title = c.get("title") or c.get("doc_id") or "source"
        url = c.get("url")
        line = f"- {badge} · [{title}]({url})" if url else f"- {badge} · {title}"
        if c.get("doc_id"):
            line += f"  \n  `doc_id={c['doc_id']}`"
        st.markdown(line)
else:
    st.info("No citations for this turn.")
