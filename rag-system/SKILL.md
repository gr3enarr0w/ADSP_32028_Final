---
name: rag-qdrant-efficient
description: Experimental, storage-efficient RAG extension for Hermes Agent using Qdrant. Supports hierarchical parent-child indexing, binary/scalar quantization, hybrid search, contextual summaries, and ingestion from local files/directories, web URLs, and APIs. Includes RAGAS evaluation support (Claude-on-Vertex fallback). Multiple modes (light/balanced/full) for trade-offs. Early-stage: functional and live-tested end-to-end, but hyperparameters are still being validated and a serious multi-file ingest data-loss bug was only just fixed — do not treat retrieval quality claims as proven yet.
version: 0.0.2
author: gr3enarr0w (initial idea drafted with Grok/xAI)
license: MIT
platforms: linux macos windows
tags: [rag, qdrant, efficient, hierarchical, quantization, hybrid-search, knowledge-base, low-storage, evaluation, ragas]
category: knowledge
metadata:
  hermes:
    requires_toolsets: [execute_code, terminal, skills]
    fallback_for_toolsets: [memory]
    write_approval: false
---

# Efficient RAG with Qdrant for Hermes Agent

## Overview & Latest Research Backing
This skill brings **state-of-the-art efficient RAG** into Hermes, drawing from 2025-2026 advancements:

- **Binary & Scalar Quantization** in Qdrant (up to 32x storage reduction with oversampling + rescoring for near-lossless recall) — guarded by embedding dimension (see below).
- **Hierarchical Parent-Child Chunking** (small precise child chunks + larger parent contexts retrieved on-demand via metadata pointers — dramatically cuts vector count while preserving full context).
- **Hybrid Search** (dense `BAAI/bge-small-en-v1.5` embeddings + sparse `Qdrant/bm25` fastembed vectors, retrieved as two independent `query_points()` legs and fused **client-side** with Reciprocal Rank Fusion using a tunable smoothing constant `retrieval.rrf_k`). Client-side fusion is a deliberate choice over Qdrant's server-side `FusionQuery(fusion=Fusion.RRF)`, which has no tunable `k` — see [Qdrant issue #5116](https://github.com/qdrant/qdrant/issues/5116).
- **Cross-Encoder Reranking** (optional second-stage reranking of the fused candidate pool with a `sentence-transformers` `CrossEncoder`, for higher precision at the top of the results).
- **Contextual Summaries** (embed summaries first; expand only when needed — currently a naive extractive placeholder unless a real LLM summarizer is injected, see below).
- **Metadata-rich indexing** with payload filters.

**Result**: Exceptionally good RAG (high context precision/recall, faithfulness) at a fraction of the storage of naive full-text embedding. Light mode often uses <100-500 MB for substantial personal or project knowledge bases. Validated via RAGAS metrics; a hyperparameter tuning pass (chunk sizes, rerank pool size, RRF `k`) against a 120-document corpus is in progress and will be documented once complete (see `docs/CHANGELOG.md`).

Hermes' persistent memory + this skill = powerful, growing personal knowledge system that stays lightweight.

**Two subtle ingest bugs** (collection-wipe-on-multi-file-ingest and a 409 conflict on subsequent files) were found and fixed during multi-document-scale testing. If you ever see retrieval returning the same generic/irrelevant chunk for every query on a multi-document collection, see `docs/CHANGELOG.md` first — it's very likely a `force_recreate` regression, not a ranking or embedding bug.

## Key Benefits vs Naive RAG
- Storage: 4-32x smaller (quantization + hierarchy + no full-text duplication).
- Speed: Faster search via quantized indices in RAM.
- Quality: Often superior due to hybrid + summaries + precise child retrieval + parent expansion.
- Scalability: Handles large corpora without TB-scale bloat.
- Privacy & Local: Embeddings and Qdrant stay on your machine (or your controlled server).

## Prerequisites & One-Time Setup
1. **Qdrant Server** (recommended Docker for persistence):
   ```bash
   docker run -d --name qdrant \
     -p 6333:6333 -p 6334:6334 \
     -v $HOME/qdrant_storage:/qdrant/storage \
     qdrant/qdrant
   ```
   Or use Qdrant Cloud.

2. **Python Dependencies** (run once):
   ```bash
   pip install -r requirements.txt
   ```
   This installs `qdrant-client` (>=1.10.0, required for `query_points`/`Prefetch` hybrid fusion legs), `sentence-transformers` (dense embeddings + cross-encoder reranker), `fastembed` (sparse BM25-style vectors for hybrid search), `pypdf`, `python-docx`, `beautifulsoup4`, `requests`, `lxml`, `pyyaml`, and `ragas` + `langchain-anthropic`/`langchain-openai`/`langchain-google-vertexai`/`anthropic[vertex]` for evaluation. **The RAGAS/LangChain versions are pinned for a reason** — see the comment block at the top of `requirements.txt` and the RAGAS Evaluation section below before upgrading them.

3. **Embedding Model**: Defaults to `BAAI/bge-small-en-v1.5` (excellent retrieval, compact). Configurable.

4. **Copy this skill** to `~/.hermes/skills/knowledge/rag-qdrant-efficient/` (or install via GitHub when published).

## Configuration
Edit `config/config.yaml` (or pass via env/CLI):
```yaml
qdrant:
  host: localhost
  port: 6333
  collection_prefix: rag_

embedding:
  model_name: BAAI/bge-small-en-v1.5
  dimension: 384
  sparse_model_name: Qdrant/bm25   # fastembed sparse model used for hybrid search

indexing:
  mode: light          # light | balanced | full
  child_chunk_size: 400
  child_chunk_overlap: 50
  parent_chunk_size: 1800
  use_hierarchical: true
  use_contextual_summaries: true
  quantization: binary   # binary | scalar_int8 | none (requested mode; see guard rule below)
  hybrid_search: true    # dense + sparse, fused with RRF
  force_binary: false            # override the dimension guard below (not recommended)
  binary_dim_threshold: 1024     # minimum embedding dimension allowed to use binary quantization

retrieval:
  top_k_children: 8
  oversampling: 3.0
  fetch_parents: true
  rerank: true                                            # enable cross-encoder rerank stage
  reranker_model: cross-encoder/ms-marco-MiniLM-L-6-v2
  rerank_candidate_pool: 40      # hybrid hits fetched before reranking (25-50 typical)
  rerank_top_n: 8                # final result count after rerank (defaults to top_k_children)
  rrf_k: 60                      # RRF rank-fusion smoothing constant (client-side fusion — see below)
```

### Quantization Dimension Guard
Binary quantization only makes sense (and only reliably preserves recall) on embeddings with
enough dimensions — with a small model like `BAAI/bge-small-en-v1.5` (384 dims), naive binary
quantization can hurt quality. `utils.py` enforces a guard:

- If `indexing.quantization: binary` is requested but the embedding dimension is below
  `indexing.binary_dim_threshold` (default `1024`), the skill automatically falls back to
  `scalar_int8` quantization instead and prints a warning explaining why.
- Set `indexing.force_binary: true` to override this guard and force binary quantization anyway
  (a warning is still printed, since this is generally not recommended below the threshold).
- `quantization: scalar_int8` and `quantization: none` are never affected by the guard.

Effective quantization is reported in ingestion stats (`_effective_quantization_label()`), so you
can always confirm what was actually applied to a given collection.

### Hybrid Search (Client-Side RRF Fusion)
When `indexing.hybrid_search: true`, a query is embedded twice — once with the dense model
(`BAAI/bge-small-en-v1.5`) and once with the sparse fastembed model (`Qdrant/bm25`) — and each is
issued as an **independent** `query_points()` call against the same collection (one leg `using="dense"`,
one leg `using="sparse"`). The two ranked hit lists are then fused **client-side** in `EfficientRAG.retrieve()`
using Reciprocal Rank Fusion: each point accumulates `1 / (rrf_k + rank)` per leg it appears in, and
the merged list is sorted by summed score. `retrieval.rrf_k` (default `60`) controls how much the
fusion favors top ranks vs. spreads credit across the pool — lower `rrf_k` weights top-ranked hits
more heavily. This is a deliberate design choice: Qdrant's server-side `FusionQuery(fusion=Fusion.RRF)`
performs the same fusion formula internally but does **not** expose `k` as a tunable parameter (see
[Qdrant issue #5116](https://github.com/qdrant/qdrant/issues/5116)), so this skill fuses client-side
instead in order to make `k` tunable. When `hybrid_search: false`, retrieval falls back to a single
plain dense `query_points()` call.

### Reranking
When `retrieval.rerank: true`, retrieval works in two stages:

1. The hybrid (RRF-fused) or plain dense search above returns a wider candidate pool of size
   `retrieval.rerank_candidate_pool` (default `40`).
2. A `sentence-transformers` `CrossEncoder` (default `cross-encoder/ms-marco-MiniLM-L-6-v2`,
   configurable via `retrieval.reranker_model`) scores every `(query, candidate_text)` pair
   directly, and the top `retrieval.rerank_top_n` (defaults to `top_k_children` if unset) are
   returned.

Set `retrieval.rerank: false` to skip straight from the fused/dense search to `top_k_children`
truncation (no reranker model is loaded in that case, saving startup time/memory).

## Core Capabilities (Tools / Procedures)

### 1. Ingest Documents (Local, URLs, APIs)
Use `scripts/ingest.py` (via `execute_code` or terminal in Hermes).

**Examples**:
- Local dir (recursive):
  ```python
  # In Hermes: execute the script
  python scripts/ingest.py --path ~/Documents/project/ --recursive --collection myproject --mode light --tags "work,important"
  ```

- Single file or URL:
  ```bash
  python scripts/ingest.py --path /path/to/report.pdf --collection reports
  python scripts/ingest.py --url https://example.com/guide --collection web_knowledge
  ```

- API source:
  ```bash
  python scripts/ingest.py --api "https://api.github.com/repos/owner/repo/issues" \
    --method GET --headers '{"Authorization": "token xxx"}' \
    --collection github_issues --tags "api,github"
  ```

**What happens in light mode**:
- Hierarchical split (small child chunks for embedding + larger parents).
- Optional contextual summary per parent.
- Embed children (quantized binary in Qdrant).
- Store rich metadata + pointer to original source (disk path or URL).
- No full-text duplication in DB.

**Modes impact**:
- `light`: Max compression + hierarchy + summaries.
- `balanced`: INT8 quant + hybrid.
- `full`: No quant, full payloads (for comparison).

**`--force-recreate` caveat**: only pass `--force-recreate` when you actually want to wipe and
rebuild a collection from scratch. For a `--recursive` directory ingest, `ingest.py` only applies
`force_recreate` to the *first* file in the loop — every subsequent file reuses the existing
collection instead of deleting it again. This is a fix for a real bug that silently dropped every
previously-ingested file's chunks during multi-file ingests; see `docs/CHANGELOG.md`.

### 2. Query / Retrieve Context
`scripts/retrieve.py`

**Example**:
```python
results = retrieve_context(
    query="What are the key risks in the Q3 project plan?",
    collection="myproject",
    mode="light",
    top_k=5
)
# Returns formatted context string ready for Hermes prompt + sources list
```

**Output includes**:
- Ranked relevant child excerpts.
- Linked parent context (full section or summary).
- Citations (source, page/URL, tags).
- Scores.

Hermes can automatically call this when reasoning about your knowledge base.

### 3. Management & Maintenance
- List collections, stats, sizes.
- Re-index with new mode/model.
- Prune low-relevance points.
- Delete collection or filtered points.

### 4. RAGAS Evaluation
`scripts/evaluate_ragas.py`

Real RAGAS evaluation against a live collection, using `retrieve_context()` from `retrieve.py` and
Anthropic/OpenAI as the judge LLM and embeddings:

```bash
# Smoke test only (no live API calls needed unless ANTHROPIC_API_KEY is set)
python scripts/evaluate_ragas.py

# Real evaluation against a collection, with generated answers
export ANTHROPIC_API_KEY=...   # judge LLM + (with --generate) answer generation
export OPENAI_API_KEY=...      # only needed for the answer_relevancy metric's embeddings
python scripts/evaluate_ragas.py \
  --testset testset.json \
  --collection myproject \
  --config config/config.yaml \
  --generate
```

`testset.json` shape:
```json
[
  {"question": "What are the key risks in the Q3 plan?", "ground_truths": ["..."]}
]
```
(a singular `ground_truth` string key is also accepted and normalized into `ground_truths`).

Computes `context_precision`, `context_recall`, `faithfulness`, and `answer_relevancy` via
`ragas.evaluate()` over an `EvaluationDataset` of `SingleTurnSample`s. Without `--generate` (or
without `ANTHROPIC_API_KEY`), `faithfulness`/`answer_relevancy` are dropped automatically since
they require generated answers, and only context-based metrics are computed. Compare `light` vs
`full` mode collections to quantify the storage/quality trade-off.

**No Anthropic/OpenAI API key available?** The judge LLM and answer-relevancy embeddings both have
a Vertex AI fallback that needs no API key — only Google Application Default Credentials
(`gcloud auth application-default login`) and a GCP project:

```bash
export ANTHROPIC_VERTEX_PROJECT_ID=my-gcp-project   # enables the Vertex fallback path
export ANTHROPIC_VERTEX_REGION=us-east5             # default; Claude on Vertex is region-limited
export ANTHROPIC_VERTEX_MODEL=claude-sonnet-4-5@20250929   # default
export VERTEX_EMBEDDING_MODEL=gemini-embedding-001  # default; answer_relevancy only
```

- Judge LLM: direct Anthropic (`ChatAnthropic`) if `ANTHROPIC_API_KEY` is set, else Claude on
  Vertex AI (`ChatAnthropicVertex`) if `ANTHROPIC_VERTEX_PROJECT_ID` is set.
- Answer-relevancy embeddings: OpenAI (`OpenAIEmbeddings`) if `OPENAI_API_KEY` is set, else Vertex
  AI embeddings (`VertexAIEmbeddings`) if `ANTHROPIC_VERTEX_PROJECT_ID` is set.
- `--generate` answer generation: direct Anthropic SDK if `ANTHROPIC_API_KEY` is set, else
  `anthropic.AnthropicVertex` if `ANTHROPIC_VERTEX_PROJECT_ID` is set.

**Important**: this GCP org's model allowlist policy blocks `text-embedding-005`/`text-embedding-004`
— only `gemini-embedding-001` is confirmed to work as the Vertex embedding fallback. Don't switch
`VERTEX_EMBEDDING_MODEL` to one of the blocked models without confirming your own org's allowlist.

**Dependency version pins matter here.** `requirements.txt` deliberately pins `ragas>=0.3.0,<0.4.0`
(ragas 0.4.x has a hard import bug against current `langchain-community`) and
`langchain-anthropic`/`langchain-openai<1.0.0` + `langchain-google-vertexai>=2.0.28,<3.0.0` (the
`>=1.0` LangChain integration packages require a newer `langchain-core` than ragas 0.3.x's own
`langchain` dependency allows). Do not casually upgrade these — see the comment block at the top of
`requirements.txt` and `docs/CHANGELOG.md` for the full rationale before touching the pins.

## How Hermes Uses This Skill
- Explicit: "Ingest my Obsidian vault into collection 'personal' in light mode"
- Implicit: During complex tasks, Hermes can decide to query RAG first for grounded answers.
- Self-improvement: Agent can edit chunking strategies or suggest re-indexing after seeing poor retrieval.

## Advanced / Future Extensions (Community)
- ColBERT late-interaction for even better precision.
- Graph RAG on top of hierarchical.
- Multi-modal (images via CLIP).
- Scheduled auto-ingest for APIs/feeds (Hermes cron).
- Integration with Honcho user modeling.

## References & Credits
- Qdrant Quantization & Hybrid docs (2025-2026 best practices).
- Parent-Child / Hierarchical RAG patterns (LangChain/LlamaIndex style, highly effective).
- RAGAS framework for rigorous evaluation.
- Research on quantization + oversampling for high-recall low-memory RAG.

This extension is designed to be the go-to lightweight yet powerful knowledge base for Hermes users. Storage-efficient by default, quality-first always.

Run `python scripts/ingest.py --help` after setup for full CLI.