# Hermes RAG Efficient — Storage-Optimized RAG Extension for Hermes Agent

**Early-stage, low-storage RAG skill for Hermes Agent** using Qdrant with hierarchical parent-child chunking, binary/scalar quantization, hybrid search, multi-source ingestion (local files, URLs, APIs), and built-in RAGAS evaluation. The pipeline is functional and has been exercised end-to-end at multi-document scale, but hyperparameters are still being validated and a serious multi-file ingest data-loss bug was only just fixed (see the status note below) — treat retrieval-quality claims as directional, not proven, until the in-progress tuning pass reports back.

This directly addresses community requests for a full user-configured knowledge base RAG system in Hermes.

## Why This Extension?
- **Aims for low-storage RAG** via quantization + hierarchy (often 4-32x smaller than naive full-text embedding approaches) — quality/recall trade-offs are still being validated (see Status note above).
- Full context preservation via hierarchical design.
- Supports any documents from local devices, web URLs, and API calls.
- Seamless Hermes integration via skills system.
- RAGAS-instrumented so quality vs. competitors (official Qdrant/Chroma skills, rag-memory-plugin, Qmd, etc.) can be measured, not just asserted — no comparative numbers are published yet.
- Built on Qdrant quantization + oversampling/rescoring, parent-child retrieval, and hybrid search patterns from current Qdrant documentation.

## Key Features
- **Modes**: `light` (binary quantization + hierarchy — default), `balanced` (INT8 + hybrid), `full`.
- **Ingestion**: Recursive directories, single files (PDF, DOCX, TXT, MD), web URLs, APIs (GET/POST with headers).
- **Hierarchical Chunking**: Small precise children for search + larger parents for full context.
- **Quantization**: Binary (max savings) or Scalar INT8 with `always_ram` + oversampling/rescoring for high recall — with an automatic dimension guard that falls back to INT8 when binary quantization isn't safe for the configured embedding size (override with `force_binary`).
- **Hybrid Search**: Dense (`BAAI/bge-small-en-v1.5`) + sparse (`Qdrant/bm25` via fastembed) vectors, retrieved as two independent `query_points()` legs and fused **client-side** with Reciprocal Rank Fusion (tunable `retrieval.rrf_k`, default `60`). Client-side fusion was chosen deliberately over Qdrant's server-side `FusionQuery(fusion=Fusion.RRF)`, which has no tunable `k` (see [Qdrant issue #5116](https://github.com/qdrant/qdrant/issues/5116)).
- **Reranking**: Optional cross-encoder reranking stage (`sentence-transformers` `CrossEncoder`, default `cross-encoder/ms-marco-MiniLM-L-6-v2`) over the fused candidate pool for higher precision.
- **Retrieval**: Formatted context with sources, summaries, citations. Metadata filtering.
- **Evaluation**: Real RAGAS metrics (context precision/recall, faithfulness, answer relevancy) driven by `retrieve_context()` against a live collection. Judge LLM/embeddings default to direct Anthropic/OpenAI when API keys are set, and fall back to Claude-on-Vertex + Vertex embeddings (Google Application Default Credentials, no API key needed) otherwise.
- **Hermes-Native**: Callable via `execute_code`, procedures in SKILL.md, self-improving friendly.

> **Status note**: The core pipeline (ingest, hierarchical chunking, hybrid search, reranking, quantization guard, RAGAS eval) is implemented and has been exercised end-to-end, including two ingest bugs found and fixed at multi-document scale — see [`docs/CHANGELOG.md`](docs/CHANGELOG.md). A hyperparameter tuning pass (chunk size × rerank pool size, RRF `k` sweep, baseline-vs-tuned validation) against a 120-document corpus is in progress; results are not yet available.

## Installation

### 1. Prerequisites
```bash
# Qdrant (recommended with persistent volume)
docker run -d --name qdrant \
  -p 6333:6333 -p 6334:6334 \
  -v $HOME/qdrant_storage:/qdrant/storage \
  qdrant/qdrant

# Python deps
pip install -r requirements.txt
```

### 2. Install the Skill
Copy the entire folder to your Hermes skills directory:

```bash
cp -r /path/to/hermes-rag-qdrant-efficient-skill ~/.hermes/skills/knowledge/rag-qdrant-efficient
```

Or clone from GitHub when published and use `hermes skills install`.

Restart Hermes or run `hermes skills list` to verify.

### 3. Configure (optional)
Edit `config/config.yaml` or pass via CLI.

## Quick Start

### Ingest Documents
```bash
# Local directory (recursive)
python scripts/ingest.py --path ~/Documents/my-project/ --recursive --collection myproject --mode light --tags "work,important"

# Single PDF or URL
python scripts/ingest.py --path report.pdf --collection reports
python scripts/ingest.py --url https://example.com/guide --collection web_knowledge

# API
python scripts/ingest.py --api "https://api.example.com/data" --method GET \
  --headers '{"Authorization": "Bearer xxx"}' --collection api_data
```

### Retrieve Context (for Hermes or CLI)
```bash
python scripts/retrieve.py --query "What are the key risks?" --collection myproject --top_k 5
```

In Hermes conversation:
> "Use the RAG skill on collection 'myproject' in light mode to answer: [your question]"

Or call the high-level function via `execute_code`.

### Evaluate with RAGAS
```bash
# Smoke test (no live API calls needed unless ANTHROPIC_API_KEY is set)
python scripts/evaluate_ragas.py

# Real evaluation against a collection
export ANTHROPIC_API_KEY=...   # judge LLM + (with --generate) answer generation
export OPENAI_API_KEY=...      # needed for the answer_relevancy metric's embeddings
python scripts/evaluate_ragas.py \
  --testset testset.json \
  --collection myproject \
  --config config/config.yaml \
  --generate
```
`testset.json` is a JSON list of `{"question": "...", "ground_truths": ["..."]}` items. Without
`--generate`/`ANTHROPIC_API_KEY`, `faithfulness`/`answer_relevancy` are skipped automatically and
only context-based metrics (`context_precision`, `context_recall`) are computed. Compare `light`
vs `full` mode collections for quantitative benchmarks.

**No Anthropic/OpenAI API key?** If `ANTHROPIC_API_KEY`/`OPENAI_API_KEY` aren't set but
`ANTHROPIC_VERTEX_PROJECT_ID` is, the judge LLM falls back to Claude on Vertex AI
(`ChatAnthropicVertex`) and the answer-relevancy embeddings fall back to Vertex AI embeddings
(`VertexAIEmbeddings`, model `gemini-embedding-001` by default) — both authenticate via Google
Application Default Credentials (`gcloud auth application-default login`), no API key required.
See `ANTHROPIC_VERTEX_REGION`, `ANTHROPIC_VERTEX_MODEL`, and `VERTEX_EMBEDDING_MODEL` in
`scripts/evaluate_ragas.py` for the tunable env vars. Note: this GCP org's model allowlist policy
blocks `text-embedding-005`/`text-embedding-004`; only `gemini-embedding-001` is confirmed to work
as the Vertex embedding fallback.

### Reranking & Hybrid Search
Hybrid search and cross-encoder reranking are both configured in `config/config.yaml` under
`indexing.hybrid_search` and `retrieval.rerank` (see `SKILL.md` for the full option list and the
binary-quantization dimension guard behavior). Hybrid search issues the dense and sparse queries as
two separate `query_points()` calls and fuses them **client-side** with Reciprocal Rank Fusion using
the tunable `retrieval.rrf_k` (default `60`) — not Qdrant's server-side `FusionQuery`, which has no
tunable `k`.

### A Note on `requirements.txt` Pins
The RAGAS/LangChain dependency versions in `requirements.txt` are **deliberately pinned**, not
arbitrary — `ragas>=0.3.0,<0.4.0`, `langchain-anthropic`/`langchain-openai <1.0.0`, and
`langchain-google-vertexai>=2.0.28,<3.0.0` all avoid confirmed, live-tested import failures (ragas
0.4.x unconditionally imports a `langchain_community` submodule that no longer exists; the `>=1.0`
LangChain integration packages require a newer `langchain-core` than ragas 0.3.x's own `langchain`
dependency pulls in). **Do not `pip install --upgrade ragas` or the langchain-* packages without
re-reading the comments at the top of `requirements.txt` first** — see also
[`docs/CHANGELOG.md`](docs/CHANGELOG.md).

## Project Structure
```
hermes-rag-qdrant-efficient-skill/
├── SKILL.md                 # Hermes skill definition + usage guide
├── README.md                # This file
├── config/
│   └── config.yaml          # Default configuration
├── scripts/
│   ├── utils.py             # Core EfficientRAG class (ingest, retrieve, chunking, quantization)
│   ├── ingest.py            # CLI for multi-source ingestion
│   ├── retrieve.py          # CLI + high-level retrieve_context() for Hermes
│   └── evaluate_ragas.py    # RAGAS evaluation CLI (context precision/recall, faithfulness, answer relevancy)
├── docs/
│   ├── ARCHITECTURE.md      # Pipeline architecture, diagram, and design-decision rationale
│   └── CHANGELOG.md         # Fixed bugs, gotchas, and in-progress work
├── examples/                # Usage examples (add your own)
└── references/              # Additional docs
```

## How It Compares to Competitors
- **Official Qdrant/Chroma skills**: These provide the DB client. This skill adds full pipelines, hierarchy, quantization tuning, summaries, and easy multi-source ingestion.
- **rag-memory-plugin** (community): Great hybrid + hooks, but SQLite-limited. This uses scalable Qdrant with superior storage optimization and hierarchy.
- **Qmd / FlowState QMD**: Excellent for Markdown/notes. This is more general-purpose (PDFs, URLs, APIs) with vector DB power.

Run the included RAGAS benchmarks to generate your own comparison data.

## Advanced Usage
- Change modes per collection for different trade-offs.
- Use metadata filters in retrieve (tags, source, etc.).
- Extend `utils.py` for custom summarization by injecting a `summary_fn` (call your Hermes LLM) instead of the placeholder heuristic summary.
- Schedule ingestion with Hermes cron for live APIs/feeds.

## Benchmarking Recommendations
See the detailed plan in previous context or expand `evaluate_ragas.py`. Recommended comparisons:
- Your light vs balanced vs full modes
- Official Qdrant skill (manual pipeline)
- rag-memory-plugin
- Qmd
- Naive flat chunking

A live hyperparameter tuning pass (child/parent chunk size × rerank candidate pool size, RRF `k`
sweep, and a baseline-vs-tuned RAGAS validation run) is currently in progress against a
120-document real corpus. Numbers are **pending** — this README will be updated once the run
completes; treat any tuning recommendations elsewhere as provisional until then.

Track: RAGAS scores, storage size, latency, recall.

## Contributing & License
MIT. Feel free to fork, improve (e.g., add ColBERT, better reranking, multi-modal), and share back to the Hermes community.

This package turns Hermes into a powerful, efficient personal knowledge engine that grows with you — without the storage bloat.

**Questions or improvements?** Open an issue or enhance the skill directly in Hermes using its self-improvement tools.

---

*Built with research-backed techniques for the Hermes ecosystem (2026).*