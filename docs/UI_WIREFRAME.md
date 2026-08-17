# UI Wireframe — Voice-to-Voice Product Assistant

**Status: BUILT.** Every region below is implemented in `ui/app.py`
(`make app`, or `PYTHONPATH=src streamlit run ui/app.py`). This document
started as the checkpoint's text wireframe — ASCII box layout + region-by-region
notes — and is kept as the design rationale for the shipped screen. The
per-region "Draft — no real code yet" notes are preserved verbatim below as the
record of what was planned versus what exists; see **Implementation status**
at the bottom for what each region became.

## Screen 1 — Main assistant screen (single page, sidebar + main, à la `demo_step_log.py`)

`ui/demo_step_log.py` already establishes the convention this wireframe
continues: `st.set_page_config(layout="wide")`, a title/caption header, an
input control near the top, a primary action button, then the rendered
results stacked top-to-bottom in the main column. The wireframe below keeps
that single-column, top-to-bottom flow (mic → transcript → step log →
answer/audio → comparison table → citations) rather than introducing a
sidebar, since a single linear flow best matches how a voice turn actually
unfolds in time — the user does not need to navigate between panels mid-turn.
A thin sidebar is reserved for session-level, not per-turn, controls (voice
picker, provider status, past-turn history) so it doesn't compete with the
turn-by-turn reading order.

```
+==============================================================================+
| Voice-to-Voice Product Assistant                              [session: •] |  <- st.title / st.caption,
|------------------------------------------------------------------------------|     mirrors demo_step_log.py's
| SIDEBAR (session-level, collapsed by default)   | MAIN COLUMN                |     st.title + st.caption pattern
| - TTS provider: pyttsx3 (offline default)       |                            |
| - ASR model: faster-whisper                     |  [1] MIC CAPTURE           |
| - Past turns (session history)                  |  +----------------------+ |
|                                                  |  | (o) Record   [~~~~~] | |
|                                                  |  | [Upload audio file]   | |
|                                                  |  | Status: idle/recording| |
|                                                  |  +----------------------+ |
|                                                  |                            |
|                                                  |  [2] LIVE TRANSCRIPT        |
|                                                  |  +----------------------+ |
|                                                  |  | "Recommend an eco-  | |
|                                                  |  |  friendly stainless | |
|                                                  |  |  steel cleaner      | |
|                                                  |  |  under fifteen      | |
|                                                  |  |  dollars."          | |
|                                                  |  +----------------------+ |
|                                                  |                            |
|                                                  |  [3] AGENT STEP LOG          |
|                                                  |  +----------------------+ |
|                                                  |  | Steps:6  Latency:812ms| |
|                                                  |  | Citations: 3          | |
|                                                  |  | > 1. router           | |
|                                                  |  | > 2. planner          | |
|                                                  |  | > 3. retriever(rag)   | |
|                                                  |  | > 4. answerer         | |
|                                                  |  | > 5. critic           | |
|                                                  |  | > 6. tts              | |
|                                                  |  +----------------------+ |
|                                                  |                            |
|                                                  |  [5] SPOKEN ANSWER + TTS     |
|                                                  |  +----------------------+ |
|                                                  |  | "My top pick is...  | |
|                                                  |  |  [ ▶ Play TTS ]      | |
|                                                  |  +----------------------+ |
|                                                  |                            |
|                                                  |  [4] COMPARISON TABLE        |
|                                                  |  +----------------------------------------+ |
|                                                  |  | title | price | rating | price_per_oz |ingredients| doc_id| |
|                                                  |  |-------|-------|--------|--------------|-----------|-------| |
|                                                  |  | Steel-Safe Eco...| $12.49| 4.6 | 0.78   | Water,... | 74d9..| |
|                                                  |  | ...   | ...   | ...    | ...          | ...       | ...   | |
|                                                  |  +----------------------------------------+ |
|                                                  |                            |
|                                                  |  [6] CITATIONS & LINEAGE     |
|                                                  |  +----------------------+ |
|                                                  |  | 🗄️ private · Steel-  | |
|                                                  |  |   Safe Eco Cleaner   | |
|                                                  |  |   doc_id=74d9f614... | |
|                                                  |  | 🌐 live · OXO Good   | |
|                                                  |  |   Grips Cleaner      | |
|                                                  |  |   url→example.com/.. | |
|                                                  |  +----------------------+ |
+==============================================================================+
```

## Region-by-region annotations

### [1] Mic capture control
**Draft — no real code yet.** Fragment-based capture per the assignment spec
("record → send file to ASR"), not live streaming — matches how
`02_whisper_asr.ipynb`'s `transcribe(audio_file: str | Path)` works: it takes
a *finished* audio file, not a stream. Planned widgets: a record button
(`st.audio_input` in modern Streamlit, or a JS recorder component that
uploads a WAV/WebM blob on stop) plus a plain `st.file_uploader` fallback for
prerecorded clips (useful for the demo since `02_whisper_asr.ipynb` §3
already expects "Upload prerecorded audio files"). On stop/upload, the file
is handed to the ASR node, which calls `transcribe()`.

### [2] Live transcript display
**Draft — no real code yet.** Despite the label "live," this is populated
from the *result* of one fragment-based ASR call, not word-by-word
streaming — consistent with `02_whisper_asr.ipynb`'s `transcribe()` returning
`{"text": ..., "language": ..., "language_probability": ..., "segments": [...]}`
only after the whole file is processed. The panel shows `output["text"]`
verbatim as the user-facing transcript; `output["segments"][*]["words"]` (with
per-word start/end/confidence) is available for a future "highlight
low-confidence words" affordance but is out of scope for this draft. This
transcript string becomes `Trace.transcript` / `TraceBuilder(transcript=...)`
in `ui/agent_step_log.py`, so region 2 and the "Transcript" caption already
rendered inside region 3's panel (`render_agent_step_log`'s
`st.caption("Transcript"); st.info(t.transcript)`) show the same value —
in the real app region 2 is this value surfaced early/prominently while the
turn is still running, and region 3 repeats it as part of the full trace.

### [3] Agent step log panel
**Real code exists.** This is a direct embed of
`ui.agent_step_log.render_agent_step_log(trace, st)`, already built and
demoed via `ui/demo_step_log.py` (`PYTHONPATH=src streamlit run
ui/demo_step_log.py`). Nothing to design here — the wireframe box literally
is that component's existing layout: `st.subheader("🧠 Agent step log")` →
transcript caption → 3-up `st.metric` row (Steps / Total latency /
Citations, from `Trace.steps`, `Trace.elapsed_ms`, `Trace.citations`) →
one `st.expander` per `Step` (title from `Step.title()`, which maps
`Step.node` through `NODE_STYLE` to an emoji+label, e.g. `retriever` →
"📚 Retriever · rag.search"; body shows `Step.input`/`Step.output` as
formatted JSON via `_fmt()`). Each LangGraph node (router, planner,
retriever, web, reconcile, answerer, critic, tts) appends a `Step` via
`TraceBuilder.add(...)`, exactly as `demo_step_log.py`'s
`build_demo_trace()` shows end to end.

### [4] Comparison table
**Draft — no real code yet.** Maps directly to the Answerer's
`comparison_table` field, schema fixed in
`prompts/answerer_critic.md`:
```
{"title":"…","price":0.0,"rating":0.0,"price_per_oz":0.0,"ingredients":"…","doc_id":"…"}
```
Planned rendering: `st.dataframe(pd.DataFrame(answer["comparison_table"]))`,
same approach `scripts/build_tts_notebook.py`'s generated notebook already
uses for its own preview (`comparison_df = pd.DataFrame(example['output']
['comparison_table']); display(comparison_df)`). Row count is the Planner's
`k` (top-N, e.g. 3 per `demo_step_log.py`'s `_lightweight_planner`). Every row
must trace back to a `doc_id` that also appears in region 6's citations list —
`answerer_critic.md` requires every table/citation entry to come from the
retrieved tool results, and the Critic's `grounded` verdict is what
enforces that before this table is ever shown.

### [5] Play TTS button
**Draft — no real code yet, but the backing function is real and tested.**
`rag.tts.speak(text: str, out_path=None, provider=None, voice=None) -> Path`
(`src/rag/tts.py`) does fragment-based synthesis of the Answerer's `speech`
field to a finished `.wav`, defaulting to the offline `pyttsx3` provider —
see `scripts/build_tts_notebook.py`'s notebook cells for the
full walkthrough (budget check via `fits_budget()`, synth via `speak()`,
inline playback via `IPython.display.Audio(filename=str(out_path))`). In
Streamlit the equivalent is: call `speak(answer["speech"])` once the Critic
returns `action: "accept"`, cache the returned `Path` on the trace
(`Trace.audio_path`, already a field on the dataclass in
`ui/agent_step_log.py` — just not wired to a player yet), and render
`st.audio(str(audio_path))` next to the button. Per
`build_tts_notebook.py`'s "Align spoken audio with on-screen citations"
section, only `speech` is ever voiced — citations and the comparison table
are deliberately screen-only, never read aloud.

### [6] Citations / data lineage display
**Real rendering logic exists (inside region 3), not yet promoted to its own
top-level panel.** `render_agent_step_log()` already implements this exact
badge-style lineage list at the bottom of the step-log panel: for each item
in `Trace.citations` (`[{doc_id, title, url, source}]`), it shows a
`🗄️ private` badge when `source == "private"` (from `rag.search`, per
`mcp/README_mcp_rag.md` — `doc_id` is "the private-catalog citation id the
UI shows next to the spoken answer") or a `🌐 live` badge otherwise (from
`web.search`, per `web-search-mcp/README_mcp_web.md` — "`url` is the
citation for every `web.search` result"), plus a markdown link when `url` is
present and always the raw `` `doc_id=...` `` string. This wireframe pulls
that same list out into its own labeled region (6) directly under the
comparison table so private-vs-live provenance reads as a first-class answer
component per the assignment spec ("Show citations and data lineage —
private doc IDs + live links"), rather than being buried at the bottom of an
expandable debug panel. Reconciliation notes from `README_mcp_web.md`
("if prices differ by more than 10%... flag the discrepancy with both
citations") are a stretch goal for this region, not implemented in the
draft.

## Known gaps / next steps *(as of the draft — superseded, see below)*

Only **region 3 (agent step log)** has real, working, demoed code today
(`ui/agent_step_log.py` + `ui/demo_step_log.py`). Everything else in this
wireframe is planned layout, not implementation:

- **Mic capture (region 1):** no recorder/uploader widget wired up yet.
  Needs a Streamlit audio-input component (or JS recorder) that saves a
  file and calls `notebooks/02_whisper_asr.ipynb`'s `transcribe()`
  logic (needs porting from notebook into a callable `src/rag/asr.py` module
  — it currently only exists as notebook cells).
- **Live transcript (region 2):** depends on the ASR wiring above; also
  needs a decision on whether to show `output["text"]` only or expose
  per-word confidence from `output["segments"]`.
- **Comparison table (region 4):** no `st.dataframe`/`pd.DataFrame` call in
  any `.py` file yet — the only place `comparison_table` is currently
  rendered is inside the *notebook* built by `build_tts_notebook.py`, not
  the Streamlit app.
- **Play TTS button (region 5):** `tts.speak()` itself is real and tested
  (`tests/test_tts.py`), but nothing calls it from Streamlit yet, and
  `Trace.audio_path` is an unused field in `ui/agent_step_log.py` today —
  no `st.audio()` call exists in the UI code.
- **Citations as a standalone panel (region 6):** the rendering logic
  exists but only as the tail end of `render_agent_step_log()`; it has not
  been extracted into its own component or promoted above the fold the way
  this wireframe draws it.
- **Router/Planner/Answerer/Critic nodes themselves:** `ui/demo_step_log.py`'s
  own Router/Planner are still "intentionally lightweight" regex placeholders,
  but real implementations now exist and are tested: `src/rag/nodes.py` +
  `src/rag/graph.py` (a compiled LangGraph `StateGraph`, exercised end-to-end
  in `notebooks/04_orchestration.ipynb`) load every prompt from `prompts/`
  at call time and run the real Router → Planner → Retriever → Answerer/Critic
  flow, with a live Anthropic call when `ANTHROPIC_API_KEY` is set and a
  deterministic mock fallback otherwise. This UI does not call `build_graph()`
  yet — `ui/demo_step_log.py`'s placeholder nodes and `TraceBuilder` still
  need to be swapped over to drive from `rag.graph.build_graph()`'s real
  state instead of the regex stand-ins.
- **End-to-end orchestration:** the graph itself is real and tested
  (`notebooks/04_orchestration.ipynb`), but nothing yet stitches ASR → the
  compiled graph → `tts.speak()` → this UI into one Streamlit callback — that
  wiring (calling `build_graph().invoke(...)` from a Streamlit button handler
  and feeding its `Step`s into `TraceBuilder`) is the remaining integration
  work.


---

## Implementation status (superseding "Known gaps / next steps")

Everything in that list is now built. What closed each gap:

| Region | Status | Where |
|---|---|---|
| [1] Mic capture | **Built** | `ui/app.py` — `st.audio_input` recorder, `st.file_uploader`, a picker over the ten prerecorded `audio/queryNN.wav` clips, and a typed-text path. All four feed the same `run_turn()`. |
| [2] Live transcript | **Built** | `ui/app.py` region 2, from `rag.asr.transcribe()`'s `output["text"]`. Per-word confidence from `segments[*]["words"]` is captured in the step log's ASR step but still not surfaced as highlighting — deliberately out of scope. |
| [3] Agent step log | **Built (now real)** | Unchanged `ui.agent_step_log.render_agent_step_log()`, but driven by the *real* compiled graph via `ui/pipeline.py` instead of `demo_step_log.py`'s regex placeholders. |
| [4] Comparison table | **Built** | `st.dataframe(pd.DataFrame(answer["comparison_table"]))`, plus the draft's stretch goal: live price and any flagged discrepancy joined in on `doc_id`. |
| [5] Play TTS | **Built** | `rag.tts.speak()` is called only once the Critic returns `action: "accept"`; the path is cached on `Trace.audio_path` (previously an unused field) and rendered with `st.audio()`. |
| [6] Citations & lineage | **Built** | Promoted to its own top-level region, merging the Answerer's private `doc_id` citations with live `url`s taken from reconciliation — the model is never asked to produce a URL it could invent. |
| ASR module | **Built** | `src/rag/asr.py`, the port out of `notebooks/02_whisper_asr.ipynb` this document asked for. Same return contract, plus an `engine` key. |
| End-to-end wiring | **Built** | `ui/pipeline.py::run_turn()` — ASR → `build_graph().astream()` → `TraceBuilder` → `tts.speak()`. `graph.py` is still the only place the graph is wired. |

### Design decisions taken during implementation

* **Radio, not tabs, for the input mode.** Streamlit executes every tab body on
  every rerun and offers no way to ask which tab is active, so a tabbed input
  would have let the sample-clip selectbox set an audio path while the user
  typed a question elsewhere — the typed text was silently ignored. Caught in
  browser testing; replaced with one explicit mode selector.
* **The same code path is tested and demoed.** `ui/app.py` holds only
  presentation; every behavioural decision lives in `ui/pipeline.py`, which
  `scripts/run_end_to_end.py` also drives. A green end-to-end run is therefore
  evidence about the demo itself.
* **The turn never dies on a TTS failure.** Synthesis errors are recorded as a
  red step and the written answer still renders.

### Operational note for the demo

Embedded Qdrant (`QDRANT_URL` blank) is **single-process**: the Streamlit app
holds a lock on `data/index/qdrant`, so `make e2e` / `make eval` will fail with
"Storage folder ... is already accessed by another instance" while the app is
running. Stop the app first, or point `QDRANT_URL` at a server.
