# Demo runbook — 7-minute final presentation

Everything needed to set up, rehearse, and deliver the live demo, plus the
fallbacks for when the room's wifi or a provider key lets us down.

The grading split is **Presentation 10 pts** for a "clear, engaging ≤7-min demo:
architecture, results, limitations" — so the plan below spends most of the time
on the running system and reserves explicit slots for results and limitations.

---

## 1. Setup (do this the night before, not in the room)

```bash
pip install -r requirements-rag.txt
sudo apt-get install espeak-ng          # Linux only; macOS TTS works out of the box
cp .env.example .env
bash scripts/build_index.sh             # sample data -> parquet -> vector index
make test                               # 53 tests, all offline
make e2e                                # 10/10 structural pass, ~80 ms/turn
```

Then pre-warm the two model downloads so the demo machine never hits the
network mid-presentation:

```bash
PYTHONPATH=src python -c "from rag.asr import get_model; get_model()"     # faster-whisper
PYTHONPATH=src python -c "from rag.embeddings import get_embedder; get_embedder()"
```

Optional but recommended:

* `ANTHROPIC_API_KEY` set → the Router/Planner/Answerer/Critic run on a real
  model. Without it everything still runs on the deterministic mock, and the
  step log says so — the demo does not depend on a key.
* A `BRAVE_SEARCH_API_KEY` or `TAVILY_API_KEY` in `web-search-mcp/.env` →
  `web.search` returns real live results and reconciliation has something to
  reconcile. Without one, the web step shows 0 results with a clear note.

**Launch:**

```bash
make app        # PYTHONPATH=src streamlit run ui/app.py
```

> Stop the app before running `make eval` or `make e2e`. Embedded Qdrant is
> single-process and will refuse the second one with "Storage folder … is
> already accessed by another instance."

---

## 2. Seven-minute plan

| Time | Segment | Who | What's on screen |
|---|---|---|---|
| 0:00–0:40 | **Problem + one-line pitch** | 1 speaker | Title slide. "Ask out loud, get a grounded, cited answer back out loud." |
| 0:40–1:30 | **Architecture** | same | Architecture diagram slide: ASR → Router → Planner → Retriever (rag.search + web.search via one MCP server) → Answerer/Critic → TTS. Name the two MCP tools explicitly. |
| 1:30–4:00 | **Live demo** | driver + narrator | The Streamlit app. Script in §3. |
| 4:00–5:15 | **How it's grounded** | narrator | Expand the retriever and critic steps in the step log; point at `doc_id`s in the citations panel matching the table rows. |
| 5:15–6:15 | **Results** | 1 speaker | Metrics slide (§4). |
| 6:15–7:00 | **Limitations + what's next** | same | Limitations slide (§5). Close on one sentence. |

Rehearse twice with a timer. The demo segment is the one that overruns; if
you're at 4:00 and still talking, cut straight to the citations panel.

---

## 3. Live demo script (2.5 minutes)

1. **Record the query live.** Input mode → **🎤 Record**, say:
   *"Recommend an eco-friendly stainless-steel cleaner under fifteen dollars."*
   Click **Run the assistant**.
   * *Fallback if the mic misbehaves:* switch to **🎧 Sample clip** →
     `query01.wav`, which is the same sentence prerecorded. Do not debug the
     mic on stage.
2. **Transcript appears** (region 2) — one line: "that's Whisper, running
   locally, no API key."
3. **Step log** (region 3) — scroll it while it fills. Call out the node names
   as they land: Router, Planner, Retriever, Answerer, Critic, TTS. Note the
   per-step latency badges.
4. **Press play on the spoken answer** (region 5). Let the whole ≤15s clip
   play — this is the deliverable, don't talk over it.
5. **Comparison table** (region 4) — three products, price, rating,
   price-per-oz, ingredients. Mention price-per-oz is normalized at ingest so
   sizes compare fairly.
6. **Citations** (region 6) — every row's `doc_id` is here; if a live result
   matched, a 🌐 badge with a real URL sits next to the 🗄️ private one.

**Second query, only if you're ahead of schedule** (~20s): pick
`query08.wav` ("Highest rated stainless steel cleaner") to show the Planner
choosing different filters for a different intent.

---

## 4. Results slide — the numbers to say out loud

Retrieval, `eval/run_eval.py --k 5` over the 10 gold queries:

| Metric | Value |
|---|---|
| Recall@5 | **0.91** |
| MRR | **1.00** |
| nDCG@5 | **0.93** |
| Filter-honor rate | **1.00** |
| Provenance validity | **1.00** |

End-to-end, `scripts/run_end_to_end.py --source audio` over the 10 prerecorded
clips (real Whisper + real TTS, production embeddings):

| Metric | Value |
|---|---|
| Turns passing all structural checks | **10 / 10** |
| Mean wall-clock per turn | **~3.0 s** steady state (4.5 s including first-turn model load) |
| Spoken answers within the ≤15 s budget | **10 / 10** |
| ASR word error rate (`small.en`) | **18.3 %** surface — see limitations |

Tests: **53 passing**, fully offline (hash embedder, mock LLM, no network).

---

## 5. Limitations to own before you're asked

* **Catalog is a 24-product sample slice.** The pipeline reads the full Kaggle
  *Amazon Product Dataset 2020* by pointing `RAW_CSV` at it — nothing else
  changes — but the committed sample is what the numbers above describe.
  Recall@5 on 24 products is not Recall@5 on 10,000.
* **That 18.3 % WER is mostly formatting, not misunderstanding.** Six of ten
  clips differ from the reference, but four are numeral normalization
  ("fifteen dollars" → "$15", "four-point-five" → "4.5") which the Router
  parses *better* than the spelled-out form. Two are genuine errors: the brand
  "Weiman" → "women", and "fragrance-free" → "Freakin's Free". Bigger model
  (`ASR_MODEL=medium.en`) or a domain phrase-bias list would fix both.
* **`web.search` routes to one provider per call.** If the routed provider's
  key is missing or fails, that call returns zero results rather than falling
  through to another configured key (`web-search-mcp/HANDOFF.md`). The graph
  degrades to private-catalog-only and says so in the step log.
* **The offline Critic always accepts.** With no `ANTHROPIC_API_KEY`, the mock
  Critic cannot actually verify grounding — it returns `accept` so the graph
  terminates. Run the demo with a key if you want the revise loop to be live.
* **Embedded Qdrant is single-process** — fine for a demo, not for concurrent
  users.

---

## 6. If something breaks on stage

| Symptom | Do this |
|---|---|
| Mic records nothing / browser blocks it | Switch to **🎧 Sample clip**. Same query, prerecorded. |
| "Storage folder … already accessed" | Another process holds the index. `pkill -f streamlit`, relaunch. |
| No audio plays | The written answer is still on screen — read it aloud yourself and move on. The TTS step will be red in the log; that's a talking point, not a failure. |
| Web step shows 0 results | Expected without a provider key. Say so: "no live key in this environment, so it's answering from the private catalog only — the reconciliation path is in the tests." |
| The whole app won't start | `make e2e` in a terminal produces the same trace as text. Worst case, show `eval/results/end_to_end_*.json`. |

---

## 7. Division of labour on the day

Two people at the front: one **drives** (never narrates), one **narrates**
(never touches the keyboard). Everyone else stays seated. This is the single
biggest thing that keeps a 7-minute demo to 7 minutes.
