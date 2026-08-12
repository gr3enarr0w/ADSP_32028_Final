#!/usr/bin/env bash
# build_index.sh — Final deliverable build script (Shane).
# Ingests the raw CSV -> parquet, then embeds & builds the vector index.
# Idempotent: safe to re-run. Honors .env / environment overrides.
#
#   bash scripts/build_index.sh
#   RAW_CSV=data/raw/marketing_sample_..._10k_data.csv bash scripts/build_index.sh
#   EMBEDDING_PROVIDER=openai bash scripts/build_index.sh
set -euo pipefail

# repo root = parent of this script's dir
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

# load .env if present
if [ -f .env ]; then set -a; . ./.env; set +a; fi

RAW_CSV="${RAW_CSV:-data/raw/SAMPLE_amazon_household_cleaning.csv}"

echo "▶ Voice-Commerce RAG index build"
echo "  root      : $ROOT"
echo "  raw csv   : $RAW_CSV"
echo "  store     : ${VECTOR_STORE:-qdrant}"
echo "  provider  : ${EMBEDDING_PROVIDER:-local}"
echo "  reranker  : ${USE_RERANKER:-true}"

if [ ! -f "$RAW_CSV" ]; then
  echo "  (raw csv not found — regenerating the sample slice)"
  python3 scripts/make_sample_data.py
fi

echo "▶ [1/2] Ingesting -> data/processed/*.parquet"
RAW_CSV="$RAW_CSV" python3 -m rag.ingest

echo "▶ [2/2] Building ${VECTOR_STORE:-qdrant} index -> ${INDEX_DIR:-data/index}"
python3 -m rag.index

echo "✔ Done. Manifest:"
cat "${INDEX_DIR:-data/index}/manifest.json"
echo
echo "Next: PYTHONPATH=src python eval/run_eval.py   |   python mcp/rag_mcp_server.py"
