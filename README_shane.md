# Shane's parts — Agentic RAG + tooling

My six deliverables for the Voice-to-Voice Product Assistant, as a self-contained
subtree that plugs into the shared repo and Clark's MCP server.

| # | Deliverable | Where | Checkpoint |
|---|---|---|---|
| 1 | Dataset slice + ingestion notebook | `notebooks/01_ingestion.ipynb`, `src/rag/ingest.py`, `data/raw/SAMPLE_*.csv` | CP1 |
| 2 | Embedding index (FAISS/Chroma) | `src/rag/embeddings.py`, `src/rag/index.py`, `scripts/build_index.sh` | — |
| 3 | Hybrid retrieval + reranker → `rag.search` | `src/rag/retrieval.py`, `src/rag/rag_search.py`, `mcp/rag_mcp_server.py` | — |
| 4 | RAG eval plan + harness | `eval/rag_eval_plan.md`, `eval/gold_queries.jsonl`, `eval/run_eval.py` | CP2 |
| 5 | Agent step-log UI panel | `ui/agent_step_log.py`, `ui/demo_step_log.py` | — |
| 6 | `prompts/` folder + build script | `prompts/`, `scripts/build_index.sh`, `Makefile` | Final |
| 7 | ASR module (ported from the notebook) | `src/rag/asr.py` | Final |
| 8 | Full Streamlit app + turn pipeline | `ui/app.py`, `ui/pipeline.py` | Final |
| 9 | End-to-end harness + demo runbook | `scripts/run_end_to_end.py`, `docs/DEMO_RUNBOOK.md` | Final |

## Quickstart

```bash
pip install -r requirements-rag.txt
cp .env.example .env                 # defaults: local embeddings + Claude LLM
bash scripts/build_index.sh          # sample data -> parquet -> vector index
PYTHONPATH=src python eval/run_eval.py --k 5      # retrieval metrics
python mcp/rag_mcp_server.py                      # serve rag.search (stdio)
PYTHONPATH=src streamlit run ui/app.py            # the full voice-to-voice app
PYTHONPATH=src python scripts/run_end_to_end.py   # end-to-end structural checks
```

`make help` lists all targets (`build`, `eval`, `mcp`, `app`, `e2e`,
`e2e-voice`, `test`, …). The demo walkthrough is `docs/DEMO_RUNBOOK.md`.

> Embedded Qdrant is single-process: stop the Streamlit app before running
> `make eval` / `make e2e`, or point `QDRANT_URL` at a server.

## Regenerating the CP1 notebook

The executed notebook is produced deterministically from a script:

```bash
python scripts/build_notebook.py
jupyter nbconvert --to notebook --execute --inplace notebooks/01_ingestion.ipynb
```

Last run: 24 products, 72 reviews, 100% price/rating/ingredients coverage.

## Using the real Kaggle data instead of the sample

The pipeline ships with a 24-row **sample** so everything runs offline. To use
the full *Amazon Product Dataset 2020*:

1. Download `marketing_sample_for_amazon_com-ecommerce__20200101_20200131__10k_data.csv`
   from Kaggle into `data/raw/`.
2. Set `RAW_CSV=data/raw/marketing_sample_..._10k_data.csv` in `.env`
   (or export it) and re-run `bash scripts/build_index.sh`.

Nothing else changes — same columns, same code. Optionally widen
`Config.slice_keywords` to tune the category filter.

## How my parts connect to the team

- **Clark** imports `rag.rag_search:rag_search` into the combined two-tool MCP
  server (alongside `web.search`). Tool schema: `mcp/README_mcp_rag.md`.
- **Victoria's** Planner emits `filters` that `rag.search` consumes; her
  Answerer/Critic uses `groundedness_score()` from `eval/run_eval.py`.
- **Alison's** Streamlit shell imports `ui/agent_step_log.py`
  (`render_agent_step_log`) and instruments nodes with `TraceBuilder`.
- The `prompts/` folder is the whole team's Prompt-Disclosure artifact; the
  map is in `prompts/README.md`.

## Config knobs (all in `.env`)

Provider (`local`/`openai`/`hash`), reranker on/off, hybrid `alpha`, `TOP_K`,
`CANDIDATE_K`, collection name, data paths. The embedding space is stamped in
`data/index/chroma/manifest.json` and verified on load so you can't accidentally
query an index built with a different model.

## Current results (sample slice)

Recall@5 **0.91**, MRR **1.00**, nDCG@5 **0.93**, filter-honor **1.00**,
provenance **1.00** — details and ablation in `eval/rag_eval_plan.md`.
The demo query returns exactly the syllabus's intended top pick:
*Steel-Safe Eco Stainless Steel Cleaner, $12.49, 4.6★.*

## Tests

```bash
make test        # 53 offline tests (hash embedder, mock LLM, no network)
make e2e         # 10/10 end-to-end structural checks over the gold queries
make e2e-voice   # the same, but through real Whisper ASR + real TTS
```

`scripts/run_end_to_end.py` asserts what the assignment grades: a spoken answer
exists, fits the ≤15s budget, the Critic returned a verdict, nothing was spoken
before it accepted, and every comparison-table row traces to a cited `doc_id`.
