# Configuration Guide

This document explains how to configure the RAG system for your environment.

## Quick Start

1. **Copy the environment template:**
   ```bash
   cp .env.example .env
   ```

2. **Fill in required values** in `.env`:
   - `QDRANT_URL` — Qdrant server address (default: `http://localhost:6333`)
   - `QDRANT_API_KEY` — API key if Qdrant requires authentication
   - `ANTHROPIC_API_KEY` — For Claude LLM evaluation and answer generation
   - `OPENAI_API_KEY` — For answer relevancy embeddings (optional)

3. **Update `config/config.yaml`** for your use case:
   - Chunk sizes (`child_chunk_size`, `parent_chunk_size`)
   - Retrieval mode (`light`, `balanced`, or `full`)
   - Search parameters (`top_k_children`, `oversampling`)

## Environment Variables

### Qdrant Configuration

- `QDRANT_URL` — Full URL to Qdrant server (default: `http://localhost:6333`)
- `QDRANT_PORT` — Port number (default: `6333`, overridden by QDRANT_URL if set)
- `QDRANT_API_KEY` — API key for authentication (optional)

### LLM APIs

- `ANTHROPIC_API_KEY` — Anthropic Claude API key (for judge LLM and answer generation)
- `OPENAI_API_KEY` — OpenAI API key (optional, for embeddings in answer relevancy metric)
- `ANTHROPIC_VERTEX_PROJECT_ID` — GCP project ID to use Vertex AI instead of direct APIs

### Google Workspace Integration

- `GOOGLE_WORKSPACE_CLIENT_SECRET_PATH` — Path to OAuth2 client secret JSON
- `GOOGLE_WORKSPACE_TOKEN_PATH` — Path to cached OAuth2 token JSON

### Data Paths

- `DATA_INGESTION_PATH` — Directory for local document ingestion
- `CONFLUENCE_DUMP_DIR` — Directory containing pre-dumped document collections

### Logging

- `LOG_LEVEL` — Logging level (default: `INFO`)

## Configuration File (config/config.yaml)

### Qdrant Section

```yaml
qdrant:
  host: ${QDRANT_URL:-localhost}
  port: ${QDRANT_PORT:-6333}
  api_key: ${QDRANT_API_KEY:-}
  collection_prefix: rag_  # All collections prefixed with this (e.g., rag_myproject)
```

### Embedding Section

```yaml
embedding:
  model_name: BAAI/bge-small-en-v1.5      # HuggingFace model
  dimension: 384                           # Model output dimension
  sparse_model_name: Qdrant/bm25          # Sparse embedding model
```

### Indexing Section

**Mode Selection** (`light` | `balanced` | `full`):
- `light` — Binary quantization, 100x storage savings, acceptable quality
- `balanced` — Scalar INT8 quantization, ~20x savings, better quality
- `full` — No quantization, best quality, highest storage

**Chunk Sizing** (in tokens, using BAAI/bge-small-en-v1.5's tokenizer):
- `child_chunk_size` — Small chunks for precise retrieval (default: 100 tokens)
- `child_chunk_overlap` — Overlap between child chunks (default: 20 tokens)
- `parent_chunk_size` — Large chunks for full context (default: 400 tokens)

### Retrieval Section

```yaml
retrieval:
  top_k_children: 8           # Number of child chunks to retrieve
  oversampling: 3.0           # Factor for hybrid search candidate pool
  fetch_parents: true         # Expand retrieved children to full parent chunks
  rerank: true                # Use cross-encoder reranking
  reranker_model: cross-encoder/ms-marco-MiniLM-L-6-v2
  rrf_k: 60                   # RRF smoothing constant for hybrid fusion
```

### Advanced Retrieval Features

All disabled by default — enable and configure as needed:

- **CRAG** (`context_review_enabled`) — Sentence-level relevance filtering
- **Adaptive-RAG** (`adaptive_retrieval_enabled`) — Query classification for routing
- **HyDE** (`hyde_enabled`) — Hypothetical document embeddings for better vocabulary matching
- **MMR** (`mmr_enabled`) — Maximal Marginal Relevance for diversity-aware ranking

## Credentials Storage

All credentials are excluded from version control via `.gitignore`:

```
.env                          # Local environment variables
.credentials/                 # OAuth2 tokens and secrets
.env.production.local         # Production-specific overrides
```

**Never commit** these files. Use environment variables or keep them in a secure, external location.

## Qdrant Setup

### Docker (Persistent Storage)

```bash
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v /path/to/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

### With Authentication

If running Qdrant with authentication enabled:

```bash
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -e QDRANT_API_KEY=your-secure-key \
  -v /path/to/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

Then set in `.env`:
```
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your-secure-key
```

## Google Workspace Configuration

### One-Time Setup

1. Create a Google Cloud project and enable the Workspace APIs (Drive, Docs, Sheets)
2. Create OAuth2 credentials (Desktop app type)
3. Download the client secret JSON and save to `.credentials/client_secret.json`
4. Run any ingestion command; the first run will prompt for interactive authorization
5. The token will be cached in `.credentials/token.json` for future runs

### Scopes

The connector requests **read-only** access to:
- Google Drive files
- Google Docs documents
- Google Sheets spreadsheets

## Data Source Configuration

### Local Files

Ingest with:
```bash
python scripts/ingest.py --path /path/to/documents --recursive --collection myproject
```

### URLs

```bash
python scripts/ingest.py --url https://example.com/docs --collection web_content
```

### APIs

```bash
python scripts/ingest.py \
  --api "https://api.example.com/data" \
  --method GET \
  --headers '{"Authorization": "Bearer <token>"}' \
  --collection api_data
```

### Pre-Dumped Collections

For Document Service or other structured exports:
```bash
python scripts/ingest.py --confluence-dump-dir /path/to/dumps --collection migrated_docs
```

## Collection Naming

All collections are automatically prefixed with `qdrant.collection_prefix` (default: `rag_`).

When you request a collection named `myproject`, it becomes `rag_myproject` in Qdrant.

## Troubleshooting

### Qdrant Connection Failed

Check:
- Docker container is running: `docker ps | grep qdrant`
- URL and port are correct: `QDRANT_URL=http://localhost:6333`
- API key is set if authentication is enabled

### Missing API Keys

Most operations work without API keys if you don't need:
- Answer generation (`ANTHROPIC_API_KEY`)
- Answer relevancy metrics (`OPENAI_API_KEY`)

These metrics will be automatically skipped if keys aren't set.

### Google Workspace Authentication

If `token.json` expires:
1. Delete `.credentials/token.json`
2. Run any ingestion command again
3. Follow the interactive authorization prompt

### Large Document Ingestion

For memory efficiency with large corpora:
- Use `--mode light` or `--mode balanced` instead of `full`
- Adjust `child_chunk_size` and `child_chunk_overlap` if needed
- Monitor disk usage in `qdrant_storage/`

## Next Steps

- See `README.md` for usage examples
- See `SKILL.md` for feature documentation
- See `docs/ARCHITECTURE.md` for technical details
