# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Hermes Agent "skill" package (not a standalone app): a storage-efficient RAG pipeline built on Qdrant. It's meant to be copied into a Hermes skills directory (`~/.hermes/skills/knowledge/rag-qdrant-efficient/`) where Hermes invokes the scripts via `execute_code`/terminal. There is no test suite, build step, or package manifest in this repo — it's a flat collection of Python scripts plus a `SKILL.md` skill definition consumed by Hermes.

## Setup & running

```bash
# Qdrant must be running (persistent volume recommended)
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
  -v $HOME/qdrant_storage:/qdrant/storage qdrant/qdrant

# Python deps (no requirements.txt/pyproject.toml exists — install manually)
pip install qdrant-client sentence-transformers pypdf python-docx beautifulsoup4 requests lxml pyyaml
```

Ingest and retrieve are run as standalone CLIs from `scripts/`:

```bash
python scripts/ingest.py --path ~/Documents/project/ --recursive --collection myproject --mode light --tags "work,important"
python scripts/ingest.py --url https://example.com/guide --collection web_knowledge
python scripts/ingest.py --api "https://api.example.com/data" --method GET --headers '{"Authorization": "Bearer xxx"}' --collection api_data

python scripts/retrieve.py --query "What are the key risks?" --collection myproject --top_k 5

python scripts/evaluate_ragas.py
```

## Architecture

- `scripts/utils.py` — the `EfficientRAG` class, the core of the whole skill: Qdrant collection creation (with quantization config), multi-source content loading (local files/PDF/DOCX, URLs via BeautifulSoup, arbitrary APIs), recursive hierarchical parent-child chunking, embedding via `sentence-transformers`, and search/retrieve with quantization-aware `SearchParams` (oversampling + rescoring).
- `scripts/ingest.py` — CLI wrapper around `EfficientRAG.ingest()`. Handles directory recursion, tag parsing, and per-source dispatch (`--path` / `--url` / `--api`). Defines its own `load_config()`.
- `scripts/retrieve.py` — CLI wrapper and the `retrieve_context()` high-level function meant to be called directly by Hermes via `execute_code`. Formats raw Qdrant hits into a citation-ready context string via `format_context()`.
- `scripts/evaluate_ragas.py` — stub for RAGAS-based quality evaluation (context precision/recall, faithfulness, answer relevancy); the actual `ragas.evaluate()` call is not yet wired in.
- `config/config.yaml` — default runtime config (Qdrant host/port, embedding model, chunking sizes, quantization mode, retrieval params). Each script layers its own hardcoded defaults under this file rather than sharing one config loader.
- `SKILL.md` — the actual Hermes skill manifest (frontmatter + usage docs) that Hermes reads to know what this skill does and how to invoke it. Treat this as the authoritative spec for expected CLI behavior/interface when changing scripts.

### Key domain concepts

- **Modes** (`light` / `balanced` / `full`) control the storage/quality tradeoff end-to-end: quantization type (binary / scalar INT8 / none), oversampling factor, and rescoring — set in `EfficientRAG.create_collection()` and mirrored in `retrieve()`'s `SearchParams`. Changing one without the other breaks the mode's intended tradeoff.
- **Hierarchical parent-child chunking**: `_hierarchical_chunk()` recursively splits documents into large "parent" chunks (full context, stored only as payload metadata, never embedded) and small "child" chunks (embedded and searched). Only children are upserted into Qdrant (`ingest()` explicitly skips non-child chunks when building points) — parents are reconstructed from metadata/pointers, not stored as separate vectors.
- **Collection naming**: all collections are transparently prefixed with `qdrant.collection_prefix` (default `rag_`) via `get_collection_name()` — pass the bare name everywhere, not the prefixed one.

### Known inconsistencies to be aware of when editing

- `scripts/ingest.py` references `Optional` in `load_config`'s type hint but never imports it from `typing` — this will raise `NameError` if that code path is exercised as-is.
- `scripts/retrieve.py` imports `load_config` from `utils`, but `load_config()` is actually only defined in `ingest.py`, not in `utils.py`. Fix by moving `load_config` into `utils.py` (as the comment in `retrieve.py` implies was intended) if working on either script.
- `_generate_summary()` in `utils.py` is a naive first-N-sentences placeholder, not an LLM call, despite "contextual summaries" being advertised as an LLM-driven feature in `SKILL.md`/`README.md`.
