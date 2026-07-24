# RAG System - Sanitized General-Purpose Release

This directory contains a cleaned, sanitized version of the hermes-rag-qdrant-efficient-skill project, ready for general-purpose use outside of any specific organization.

## What's Here

**`/rag-system/`** — A complete, storage-efficient RAG (Retrieval-Augmented Generation) pipeline built on:
- **Qdrant** — Vector database with quantization (binary, INT8, or full precision)
- **Sentence Transformers** — BAAI/bge-small-en-v1.5 embeddings
- **Hybrid Search** — Dense + sparse (BM25) retrieval with RRF fusion
- **Cross-Encoder Reranking** — Relevance-based result refinement
- **Hierarchical Chunking** — Parent-child document segmentation for context preservation

## Key Features

✓ **Storage-Efficient** — Light mode uses <100-500 MB for substantial knowledge bases (100x compression)
✓ **High Quality** — Context precision/recall validated via RAGAS metrics
✓ **Flexible Ingestion** — Local files, URLs, APIs, Google Workspace, pre-dumped collections
✓ **Advanced Retrieval** — Hybrid search, reranking, parent-chunk expansion, MMR diversity
✓ **No External Dependencies** — All credentials via environment variables

## Quick Start

1. **Read the configuration guide:**
   ```bash
   cat rag-system/CONFIGURATION.md
   ```

2. **Set up environment:**
   ```bash
   cd rag-system/
   cp .env.example .env
   # Edit .env with your API keys and Qdrant URL
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start Qdrant:**
   ```bash
   docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
     -v $HOME/qdrant_storage:/qdrant/storage qdrant/qdrant
   ```

5. **Ingest documents:**
   ```bash
   cd rag-system/
   python scripts/ingest.py --path /path/to/documents --collection myproject
   ```

6. **Retrieve context:**
   ```bash
   python scripts/retrieve.py --query "Your question here" --collection myproject
   ```

## Security & Privacy

This release has been aggressively sanitized for general use:

- ✓ No hardcoded credentials
- ✓ No internal company references
- ✓ No proprietary service URLs
- ✓ All paths relative or environment-based
- ✓ Comprehensive .gitignore for credential protection
- ✓ See `SANITIZATION_REPORT.md` for details

## Documentation

- **CONFIGURATION.md** — Complete setup guide, environment variables, troubleshooting
- **rag-system/README.md** — Feature overview and usage examples
- **rag-system/SKILL.md** — Detailed CLI documentation
- **rag-system/CLAUDE.md** — Architecture notes for developers
- **rag-system/docs/** — Technical deep dives (architecture, changelog, research notes)

## Files & Structure

```
/
├── README.md                    (this file)
├── SANITIZATION_REPORT.md       (detailed changes made)
└── rag-system/
    ├── .env.example             (environment template)
    ├── CONFIGURATION.md         (setup guide)
    ├── config/config.yaml       (runtime configuration)
    ├── scripts/                 (Python CLI tools)
    │   ├── ingest.py           (data ingestion)
    │   ├── retrieve.py         (context retrieval)
    │   ├── evaluate_ragas.py   (quality evaluation)
    │   ├── utils.py            (core EfficientRAG class)
    │   └── ...
    ├── docs/                    (technical documentation)
    ├── data/                    (example eval datasets)
    └── requirements.txt         (Python dependencies)
```

## Next Steps

1. **For local use:**
   - Follow the Quick Start above
   - Customize config/config.yaml for your needs
   - See rag-system/CONFIGURATION.md for all options

2. **For integration:**
   - Import `EfficientRAG` class from scripts/utils.py
   - Use `retrieve_context()` function for RAG retrieval
   - See rag-system/SKILL.md for API documentation

3. **For evaluation:**
   - Run `scripts/evaluate_ragas.py` to measure quality
   - Adjust hyperparameters based on RAGAS metrics
   - See docs/REAL_CORPUS_EVAL_V1_SUMMARY.md for benchmark results

## Support

For questions or issues:
1. Check rag-system/CONFIGURATION.md (Troubleshooting section)
2. Review rag-system/docs/ARCHITECTURE.md for technical details
3. Examine example commands in rag-system/SKILL.md

## License

This project is provided as-is without specific license restrictions. Use it freely in your organization or projects.

---

**Status:** Sanitized and ready for general use (July 23, 2026)
**Size:** 2.3 MB total (includes documentation, example datasets, no credentials)
**Dependencies:** Python 3.8+, Qdrant server, HuggingFace/PyTorch models
