# Voice-to-Voice Product Discovery Assistant

Ask for a product out loud; get a grounded, cited recommendation back out loud.

Speech goes in, a LangGraph multi-agent pipeline routes → plans → retrieves →
answers → checks, and a ≤15-second spoken summary comes back with every claim
traceable to a private catalog record or a live URL on screen.

**ADSP 32028 · Applied Generative AI · Final project** — Shane, Clark, Alison, Victoria.

```
 audio ─▶ ASR ─▶ Router ─▶ Planner ─▶ Retriever ─▶ Answerer ─▶ Critic ─▶ TTS ─▶ audio
                                        │                        │
                                 rag.search + web.search      revise once
                                 (one MCP server, two tools)
```

---

## Quickstart

```bash
pip install -r requirements-rag.txt
sudo apt-get install espeak-ng      # Linux only — the offline TTS backend
cp .env.example .env                # defaults: local embeddings, offline TTS, no keys needed
bash scripts/build_index.sh         # sample data → parquet → vector index
make app                            # the full assistant at localhost:8501
```

That runs with **zero API keys**: local embeddings, local Whisper, offline
TTS, and a deterministic mock in place of the LLM. Add `ANTHROPIC_API_KEY` to
`.env` for real Router/Planner/Answerer/Critic reasoning, and a
`BRAVE_SEARCH_API_KEY` or `TAVILY_API_KEY` to `web-search-mcp/.env` for live
price comparison.

`make help` lists every target. The demo walkthrough — setup, a timed
seven-minute plan, the numbers to quote, and on-stage fallbacks — is
[`docs/DEMO_RUNBOOK.md`](docs/DEMO_RUNBOOK.md).

> **One process at a time.** Embedded Qdrant holds an exclusive lock on
> `data/index/`, so stop the Streamlit app before running `make eval` or
> `make e2e` (or point `QDRANT_URL` at a Qdrant server).

---

## What's here

| Area | Entry point | Notes |
|---|---|---|
| Orchestration | `src/rag/graph.py`, `src/rag/nodes.py` | `build_graph()` is the single source of truth; every node loads its prompt from `prompts/` at call time |
| Private retrieval | `src/rag/retrieval.py`, `src/rag/rag_search.py` | Hybrid vector + BM25 with RRF fusion, metadata filters, optional cross-encoder reranker |
| Live comparison | `web-search-mcp/web_search.py` | Multi-provider search with TTL cache, rate limiting and source-URL logging |
| MCP server | `web-search-mcp/combined_mcp_server.py` | One server, both tools, real discovery + JSON schemas over stdio |
| Reconciliation | `src/rag/reconcile.py` | SKU → brand → fuzzy title, one-to-one; private fact always wins, >10% price gaps flagged |
| Speech in / out | `src/rag/asr.py`, `src/rag/tts.py` | Fragment-based both directions: finished file in, finished file out |
| Safety | `src/rag/safety.py` | Domain allowlist enforced in code, deterministic hazardous-mixture flags |
| UI | `ui/app.py`, `ui/pipeline.py`, `ui/agent_step_log.py` | Presentation, turn pipeline, and the reusable step-log panel |
| Evaluation | `eval/run_eval.py`, `scripts/run_end_to_end.py` | Retrieval metrics, and structural checks on the whole voice turn |
| Prompts | `prompts/` | The Prompt-Disclosure artifact — and literally what executes |

---

## Common commands

```bash
make build       # ingest raw CSV → parquet → build the vector index
make app         # the full voice-to-voice Streamlit app
make test        # 71 tests, fully offline (hash embedder, mock LLM, no network)
make eval        # retrieval metrics over eval/gold_queries.jsonl
make e2e         # end-to-end structural checks (text in, no ASR)
make e2e-voice   # the same, through real Whisper ASR + real TTS
make mcp         # serve rag.search over stdio
```

Inspect the MCP server interactively:

```bash
cd web-search-mcp && npx @modelcontextprotocol/inspector python combined_mcp_server.py
```

---

## Configuration

Everything is environment-driven — see `.env.example` for the annotated list.
Nothing below is hard-coded in a node, so any piece can be swapped without
touching application code:

| Knob | Default | Alternatives |
|---|---|---|
| `LLM_PROVIDER` / `LLM_MODEL` | `anthropic` / `claude-sonnet-5` | any provider you implement in `src/rag/llm.py` |
| `EMBEDDING_PROVIDER` | `local` (sentence-transformers) | `openai`, `hash` (offline, for tests) |
| `VECTOR_STORE` | `qdrant` (embedded) | `chroma`, or a Qdrant server via `QDRANT_URL` |
| `ASR_MODEL` | `small.en` | `tiny.en`, `base.en`, `medium.en` |
| `TTS_PROVIDER` | `pyttsx3` (offline) | `openai`, `elevenlabs` |
| `RAW_CSV` | the 24-row sample slice | the full Kaggle *Amazon Product Dataset 2020* CSV |
| `WEB_SEARCH_ALLOWLIST` | curated retailer/manufacturer list | comma-separated hosts, or `*` to disable |

---

## Results

Retrieval over the ten gold queries (`make eval`, k = 5): Recall@5 **0.91**,
MRR **1.00**, nDCG@5 **0.93**, filter-honor **1.00**, provenance **1.00**.

End to end over the ten prerecorded clips with real ASR and real TTS
(`make e2e-voice`): **10/10** turns pass every structural check, **10/10**
spoken answers fit the 15-second budget, **~3.0 s** per turn steady state.

Limitations we'd fix first — a 24-product sample catalog, an 18.3% surface
word error rate that is mostly numeral formatting, single-provider live search
with no failover, and a mock Critic that cannot really verify grounding
offline — are written up in `docs/DEMO_RUNBOOK.md` §5.

---

## Documentation

- [`docs/DEMO_RUNBOOK.md`](docs/DEMO_RUNBOOK.md) — setup, seven-minute plan, fallbacks
- [`docs/UI_WIREFRAME.md`](docs/UI_WIREFRAME.md) — screen design and what each region became
- [`README_shane.md`](README_shane.md) — the RAG subtree in detail
- [`mcp/README_mcp_rag.md`](mcp/README_mcp_rag.md) · [`web-search-mcp/README_mcp_web.md`](web-search-mcp/README_mcp_web.md) — tool schemas
- [`web-search-mcp/HANDOFF.md`](web-search-mcp/HANDOFF.md) — live-search internals and known gaps
- [`eval/rag_eval_plan.md`](eval/rag_eval_plan.md) — evaluation design
- [`prompts/README.md`](prompts/README.md) — prompt → node map
