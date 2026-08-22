# Makefile — Shane's RAG subtree. `make help` lists targets.
SHELL := /bin/bash
export PYTHONPATH := $(CURDIR)/src

.DEFAULT_GOAL := help

.PHONY: help setup sample ingest index build eval mcp ui app e2e e2e-voice demo-trace test clean

help: ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	 awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## install python dependencies
	pip install -r requirements-rag.txt

sample: ## (re)generate the sample Household-Cleaning CSV
	python3 scripts/make_sample_data.py

ingest: ## raw CSV -> data/processed/*.parquet
	python3 -m rag.ingest

index: ## build the vector index from processed parquet
	python3 -m rag.index

build: ## full build: ingest + index (calls build_index.sh)
	bash scripts/build_index.sh

eval: ## run the retrieval evaluation harness
	python3 eval/run_eval.py --k 5

mcp: ## run the rag.search MCP server (stdio)
	python3 mcp/rag_mcp_server.py

ui: ## launch the full voice-to-voice assistant (same as `make app`)
	streamlit run ui/app.py

ui-panel: ## launch the standalone step-log panel demo (superseded by `make app`)
	streamlit run ui/demo_step_log.py

app: ## launch the full voice-to-voice assistant (Streamlit)
	streamlit run ui/app.py

e2e: ## end-to-end harness over eval/gold_queries.jsonl (text in, no ASR)
	python3 scripts/run_end_to_end.py

e2e-voice: ## end-to-end harness over the 10 prerecorded clips in audio/ (real ASR + TTS)
	python3 scripts/run_end_to_end.py --source audio

demo-trace: ## write ui/sample_trace.json (no Streamlit needed)
	python3 ui/demo_step_log.py --dump ui/sample_trace.json

test: ## run the smoke tests (offline hash embedder)
	EMBEDDING_PROVIDER=hash USE_RERANKER=false python3 -m pytest tests/ -q

clean: ## remove generated artifacts
	rm -rf data/processed/*.parquet data/index eval/results ui/sample_trace.json \
	       .pytest_cache **/__pycache__
