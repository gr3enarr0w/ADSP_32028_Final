# RAG Evaluation Plan (Checkpoint 2)

**Owner:** Shane · **Component:** Agentic RAG (private retrieval) · **Deliverable:** CP2

This plan defines how we measure whether the private-catalog retriever returns
the *right* products and whether the assistant's spoken answers stay *grounded*
in what was retrieved. It covers the gold set, metrics, procedure, baselines,
current results on the sample slice, and the answer-level groundedness protocol
that runs once the full LangGraph is wired.

---

## 1. Objectives & success criteria

| Question | Metric | Target (full Kaggle slice) |
|---|---|---|
| Does the right product appear at all? | **Recall@5** | ≥ 0.85 |
| Is the top hit usually right? | **MRR** | ≥ 0.80 |
| Is the *ranking* good, not just the set? | **nDCG@5** | ≥ 0.80 |
| Are budget/rating constraints respected? | **Filter-honor rate** | 1.00 |
| Do citations point at real catalog items? | **Provenance validity** | 1.00 |
| Do spoken answers only claim retrieved facts? | **Groundedness** (answer-level) | ≥ 0.95 |

Retrieval quality (rows 1–5) is owned here. Groundedness (row 6) is measured
jointly with Victoria's Answerer/Critic node using the harness stub already in
`run_eval.py`.

## 2. Data under test

* **Corpus:** Household-Cleaning slice of *Amazon Product Dataset 2020*, produced
  by `notebooks/01_ingestion.ipynb` → `data/processed/products.parquet`
  (+ `reviews.parquet`). Sample = 24 curated products; full Kaggle file scales
  to the whole cleaning category with no code change.
* **Index:** Chroma, cosine space, embeddings over
  `title + features + top-3 review snippets + ingredients`.

### 2.1 Gold query set (`eval/gold_queries.jsonl`)

10 hand-labeled queries spanning the catalog's sub-categories (stainless,
glass, dish, degreaser, wood, bathroom, laundry, budget-constrained). Each row:

```json
{"id": "q01",
 "query": "eco-friendly stainless steel cleaner under $15",
 "filters": {"price_max": 15, "material": "stainless steel"},
 "relevant_skus": ["SKU-GREE-000", "SKU-PURE-001", "SKU-GREE-003",
                   "SKU-EVER-012", "SKU-NATU-018"],
 "notes": "distractors: ShineMaster ($16.75), MegaShine (not eco)"}
```

Relevance is **binary** and judged by a human against the product's title,
features, and ingredients. Deliberate **distractors** (a non-eco or over-budget
item in the same category) are recorded so precision is meaningful. On the full
Kaggle slice the gold set grows to ≥ 50 queries with two annotators and an
adjudication pass; inter-annotator agreement (Cohen's κ) is reported.

## 3. Metrics (definitions)

Let *R* = relevant SKUs for a query, *L* = ranked list returned by `rag.search`.

* **Recall@k** = |R ∩ L[:k]| / |R|
* **Precision@k** = |R ∩ L[:k]| / k
* **MRR** = 1 / rank of the first relevant hit (0 if none)
* **nDCG@k** = DCG@k / IDCG@k with binary gains, log₂ discount
* **Filter-honor rate** = fraction of queries where *every* returned item
  satisfies the query's `price_max` / `price_min` / `min_rating` constraints
* **Provenance validity** = fraction of returned `doc_id`s that resolve to a
  real indexed document (guards against citation drift)
* **Groundedness** (answer-level) = fraction of the Answerer's citation ids that
  belong to the retrieved set — `groundedness_score()` in `run_eval.py`

Precision is reported but not optimized: most queries have 1–2 truly relevant
items in a 24-item catalog, so Precision@5 is capped low by construction
(a single-relevant query maxes at 0.20). Recall@k, MRR, and nDCG@k are the
primary signals; Precision becomes informative on the larger slice.

## 4. Procedure

```bash
bash scripts/build_index.sh                 # ingest + embed + index
PYTHONPATH=src python eval/run_eval.py --k 5 # writes eval/results/eval_report.json
```

The harness calls the **exact** `rag.search` tool body the MCP server exposes,
so we evaluate the deployed path, not a parallel implementation. Every query is
run with its declared filters to jointly test retrieval + constraint handling.

### 4.1 Baselines / ablations

Controlled by env vars, so each config is one command:

| Config | `HYBRID_ALPHA` | `USE_RERANKER` |
|---|---|---|
| BM25 only | 0 | false |
| Vector only | 1 | false |
| Hybrid (RRF) | 0.5 | false |
| **Hybrid + cross-encoder rerank** (default) | 0.5 | true |

## 5. Current results (sample slice, 24 products, k=5)

From `eval/results/eval_report.json` — provider
`local:all-MiniLM-L6-v2`, reranker `ms-marco-MiniLM-L-6-v2`:

| Metric | Value |
|---|---|
| Recall@5 | **0.91** |
| Precision@5 | 0.32 |
| MRR | **1.00** |
| nDCG@5 | **0.93** |
| Filter-honor rate | **1.00** |
| Provenance validity | **1.00** |

### 5.1 Ablation (sample slice)

| Config | Recall@5 | nDCG@5 | MRR |
|---|---|---|---|
| BM25 only | 0.91 | 0.93 | 1.00 |
| Vector only | 0.91 | 0.93 | 1.00 |
| Hybrid | 0.91 | 0.93 | 1.00 |
| Hybrid + rerank | 0.91 | 0.93 | 1.00 |

**Reading this honestly:** on a 24-item catalog the retrieval task is nearly
*saturated* — the relevant items are so lexically and semantically separated
from distractors that all four channels place them in the top 5, so set-based
metrics don't move. The differences the ablation is designed to expose appear
(a) on the full ~10k-row Kaggle slice, where vocabulary mismatch and near
-duplicates make dense + sparse fusion and reranking matter, and (b) in
*ordering among relevant items*, which binary set-metrics can't see. Concrete
example of the latter: for q01 the cross-encoder promotes **"Steel-Safe Eco
Stainless Steel Cleaner, $12.49, 4.6★"** to rank 1 — exactly the syllabus's
intended top pick — whereas fusion-only ranks a smaller travel-size SKU first.

**Action for the full slice:** re-run §4.1 on the real Kaggle file; we expect
Hybrid+rerank to lead on nDCG@5 and MRR. Report the table again and keep the
config that wins; the winner is set via `.env` with no code change.

## 6. Known limitations / findings

* **Coarse `material` tags.** `material` is derived by keyword-matching the
  title/category (e.g. "tile"). q10's ScrubPro foam lacks "tile" in its title,
  so the `material:tile` filter drops it (Recall@5 = 0.50 on q10). Fix: derive
  `material` from features/ingredients too, or treat it as a soft boost instead
  of a hard filter. Tracked for the full-slice build.
* **Rating on real data.** The real Kaggle file has no rating column; `rating`
  is `null` until reviews are joined. `min_rating` filters therefore no-op on
  real data unless a reviews source is added. The sample includes ratings so the
  path is exercised end-to-end.
* **Small gold set.** 10 queries on the sample; expand to ≥ 50 with two
  annotators on the full slice before drawing config conclusions.

## 7. Answer-level groundedness (with Answerer/Critic)

Once the graph is wired, each demo/eval query also records the Answerer's spoken
text and the `doc_id`s it cited. `groundedness_score(answer_citation_ids,
retrieved_ids)` returns the fraction of cited ids that were actually retrieved;
the Critic node rejects and regenerates any answer scoring < 1.0. We report mean
groundedness and the count of Critic rejections across the gold set.

## 8. Reproducibility

* Deterministic given a fixed index; embedding space is stamped in
  `data/index/chroma/manifest.json` and verified on load.
* All knobs are env-driven (`.env.example`); every result line records provider,
  reranker, and k so runs are self-describing.
