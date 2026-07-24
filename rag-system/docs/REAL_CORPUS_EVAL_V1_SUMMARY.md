# Real-corpus evaluation v1 — closed-book vs. RAG vs. RAG+HyDE against `personal_corpus_v1`

**Date:** 2026-07-21
**Status:** Single preregistered confirmatory run. Completed, not a pilot/sweep.
**Full per-question results:** `data/real_corpus_eval_results_v1.json`
**Raw per-arm CSVs:** `/tmp/real_corpus_eval_v1/results_{closedbook,rag,rag_hyde}.csv`

## Why this run matters

Every prior RAGAS evaluation in this project (see `docs/CHANGELOG.md`'s HyDE/adaptive-RAG/MMR
entries) ran against NFCorpus, a public BEIR benchmark. That corpus is a plausible pretraining
contaminant for any modern LLM — a closed-book "baseline" that has already seen (or seen
similar) NFCorpus text during training is not a fair no-retrieval comparison, and every
"RAG helps" or "RAG doesn't help" conclusion drawn from it is confounded by that possibility.

This run instead uses `personal_corpus_v1` (2.38M Qdrant points), built from the user's actual
local files, 5 Document Service spaces (OMEGA/DDISDP/HUB/ANTSE team spaces + a personal space), and
Google Workspace content (Docs/Sheets/Slides) — content no LLM could have seen during training.
`data/real_corpus_eval_questions_v1.json` (39 hand-reviewed question/reference pairs, spanning
all of the above source types, security/PII questions pre-filtered by the user) is therefore the
**first genuinely uncontaminated test of whether this pipeline's retrieval adds real value.**

## Method

Reused the project's existing `scripts/evaluate_ragas.py` harness as-is (no reimplementation):

- **closed-book**: `run_closed_book_evaluation()` — `generate_closed_book_answer()`, zero
  retrieval, scored with `answer_relevancy` + `answer_similarity` (RAGAS metrics that don't
  require `retrieved_contexts`).
- **rag**: `run_ragas_evaluation()` with `_make_rag_retrieve_func("personal_corpus_v1",
  load_config())` — i.e. `config/config.yaml`'s defaults (hybrid dense+sparse search, RRF fusion,
  cross-encoder rerank, parent-chunk expansion, `hyde_enabled: false`, `adaptive_retrieval_enabled:
  false`) — and `generate_answer()` (the attribution-tagged `[C]`/`[G]` blending prompt).
- **rag_hyde**: identical to `rag`, except `retrieval.hyde_enabled = True` (this project's
  existing HyDE implementation, Gao et al. arXiv:2212.10496 — `EfficientRAG._generate_hyde_document()`
  generates a hypothetical answer passage via one Ollama call (`glm-5.2:cloud`), embeds that for
  the dense search leg only; sparse/BM25 and rerank still use the literal query).

Judge LLM / embeddings: whatever `_default_judge_llm()`/`_default_judge_embeddings()` defaulted
to in this environment — Claude on Vertex AI (`claude-sonnet-4-5@20250929`, no `ANTHROPIC_API_KEY`
set, only `ANTHROPIC_VERTEX_PROJECT_ID`) and Vertex AI embeddings (`gemini-embedding-001`, no
`OPENAI_API_KEY` set). No overrides — per the task's own guardrail, this run used whatever the
harness already defaults to.

Metrics: `faithfulness`, `answer_relevancy` (both raw and disclaimer-stripped, per this project's
existing `answer_relevancy` noncommittal-zero-gate fix —
`compute_disclaimer_stripped_answer_relevancy()`), `context_precision`, `context_recall` (scored
against the testset's `reference` field), and `answer_correctness`, for both RAG arms. Also
carried through this project's split-faithfulness diagnostic
(`faithfulness_c_only`/`groundedness_g_only`, see `compute_split_faithfulness()`), which was
already a default of the harness. closed-book cannot be scored on `faithfulness`/
`context_precision`/`context_recall`/`answer_correctness` (there is no retrieved context to
measure attribution against) — this is the harness's own established, deliberate convention
(`run_closed_book_evaluation()`'s docstring), not an omission in this run.

**N=39, single run, no repeats** — this is a preregistered confirmatory run on a fresh,
never-before-tested question set, not a multi-repeat pooled sweep like the earlier NFCorpus HyDE
replication. Statistics: paired Wilcoxon signed-rank (two-sided, alpha=0.05, uncorrected) +
bootstrap 95% CI (n_boot=10000, seed=42) on the per-question delta, matching this project's
established methodology (`/tmp/nfcorpus_eval_v2/analyze_hyde_confirmatory.py`, the reference
implementation). Deltas were computed by row **position**, not by joining on question text — all
three arms' DataFrames were built from the identical, order-preserved 39-item testset list by the
same driver script in the same process, so positional alignment is exact and avoids any
text-normalization mismatch a text join could introduce.

### Concurrency fix applied mid-run

The per-question split-faithfulness follow-up calls inside `run_ragas_evaluation()`
(`compute_split_faithfulness()`, one extra `evaluate()` call per question with at least one
`[C]`-tagged claim) originally ran in a plain sequential `for` loop. Mid-run, this was corrected
to `concurrent.futures.ThreadPoolExecutor(max_workers=9)` — these are independent, Claude/
Anthropic-based judge calls (each builds its own fresh judge LLM/`Faithfulness()` instance), not
NeuralWatt-hosted, so none of the NeuralWatt-specific 3-concurrent-request account-wide rate
limit or its `max_workers=1`-per-thread pattern applies here. Arm 1 (closed-book) and Arm 2 (RAG
baseline) had already completed under the old sequential code before this fix landed and were
**not** re-run (no wasted work); Arm 3 (RAG+HyDE) ran entirely under the parallelized version.
Measured wall-clock for Arm 3's full run (retrieval + main RAGAS `evaluate()` + parallelized
split-faithfulness + disclaimer-stripping) was **1406.8s (~23.4 min)** for 39 questions. Arm 2's
split-faithfulness step alone, under the old sequential code, took approximately 8 minutes for
the same 39 rows (~12s/row); the equivalent phase for Arm 3 (finishing the tail of the main
`evaluate()` call plus the fully parallelized split-faithfulness and batched disclaimer-stripping
step) completed in well under 2 minutes once retrieval was done — roughly a 4-5x wall-clock
reduction on that specific step, consistent with 9-way concurrency bounded by per-call latency
variance and judge-API contention (not the full theoretical 9x, since individual judge calls
still took 10-20+ seconds each and threads occasionally waited on each other). The fix has been
applied to `scripts/evaluate_ragas.py` itself (not just this run's driver script), so it benefits
every future caller of `run_ragas_evaluation()`.

## Results

### Descriptive means (N=39 per arm)

| Metric | closed_book | rag | rag_hyde |
|---|---|---|---|
| answer_relevancy | 0.3117 | 0.5281 | 0.5763 |
| answer_relevancy_disclaimer_stripped | 0.3333 | 0.6158 | 0.6814 |
| answer_similarity (closed-book only; NOT comparable to answer_correctness below) | 0.8280 | — | — |
| context_precision | N/A (no retrieval) | 0.4926 | 0.5031 |
| context_recall | N/A (no retrieval) | 0.6496 | 0.6667 |
| faithfulness | N/A (no retrieval) | 0.7595 | 0.7713 |
| answer_correctness | N/A (needs context) | 0.4964 | 0.4885 |
| faithfulness_c_only (diagnostic) | N/A | 0.8005 | 0.7686 |
| groundedness_g_only (diagnostic) | N/A | 0.7857 (n=valid G-claim rows only) | 0.5000 (n=4 valid rows only) |

**IMPORTANT caveat on `answer_similarity` vs. `answer_correctness`:** these are two *different*
metrics with different definitions and scales (embedding cosine similarity vs. an LLM-judged
TP/FP/FN atomic-statement decomposition) — this is the harness's own deliberate design
(`run_closed_book_evaluation()`'s docstring explains why `answer_correctness` isn't used for the
no-context arm). Do **not** read "closed-book's 0.828 > rag's 0.496" as "closed-book is more
correct than RAG" — that would be comparing two incommensurable numbers. No direct
correctness-vs-correctness comparison across closed-book and the RAG arms was possible with this
harness's existing metric set; `answer_relevancy` is the only truly apples-to-apples generation
metric shared across all three arms (plus its disclaimer-stripped variant).

### Pairwise statistical comparisons (paired Wilcoxon + bootstrap 95% CI, all computed metrics reported)

| Comparison | Metric | n | mean delta | 95% CI | Wilcoxon p | Significant? |
|---|---|---|---|---|---|---|
| closed_book → rag | answer_relevancy | 39 | +0.2164 | [+0.0307, +0.3947] | 0.0835 | No |
| closed_book → rag | answer_relevancy_disclaimer_stripped | 39 | +0.2825 | [+0.0913, +0.4623] | **0.0274** | **Yes** |
| closed_book → rag_hyde | answer_relevancy | 39 | +0.2647 | [+0.0841, +0.4394] | **0.0190** | **Yes** |
| closed_book → rag_hyde | answer_relevancy_disclaimer_stripped | 39 | +0.3480 | [+0.1550, +0.5224] | **0.0038** | **Yes** |
| rag → rag_hyde | answer_relevancy | 39 | +0.0483 | [+0.0017, +0.1156] | 0.1138 | No |
| rag → rag_hyde | answer_relevancy_disclaimer_stripped | 39 | +0.0656 | [-0.0244, +0.1591] | 0.3740 | No |
| rag → rag_hyde | context_precision | 39 | +0.0105 | [-0.0090, +0.0346] | 0.3980 | No |
| rag → rag_hyde | context_recall | 39 | +0.0171 | [-0.0684, +0.1026] | 0.7055 | No |
| rag → rag_hyde | faithfulness | 39 | +0.0118 | [-0.0490, +0.0720] | 0.6664 | No |
| rag → rag_hyde | answer_correctness | 39 | -0.0079 | [-0.0490, +0.0389] | 0.4596 | No |
| rag → rag_hyde | faithfulness_c_only (diagnostic) | 36 | -0.0626 | [-0.1582, +0.0278] | 0.2249 | No |
| rag → rag_hyde | groundedness_g_only (diagnostic) | 4 | -0.1250 | [-0.7500, +0.3750] | 1.0000 | No |

(`closed_book` has no `context_precision`/`context_recall`/`faithfulness`/`answer_correctness`
columns at all, so no comparison row exists for those metrics against closed-book — this is a
structural N/A, not a suppressed/non-significant result.)

## Interpretation

**RAG helps, and the effect is real on this genuinely-unseen corpus.** Both RAG arms answer
questions far more relevant to what was actually asked than closed-book Claude does
(`answer_relevancy` +0.216 to +0.265 raw, +0.283 to +0.348 disclaimer-stripped) — intuitively
correct, since these 39 questions ask about specific facts inside the user's own private
documents (Document Service pages, local files, Google Workspace content) that a general-purpose model
has no way to know without retrieval. The disclaimer-stripped `answer_relevancy` comparisons
(the metric this project's own noncommittal-zero-gate fix targets) reach significance for
**both** `rag` and `rag_hyde` vs. closed-book (p=0.0274, p=0.0038); the raw `answer_relevancy`
comparison reaches significance only for `rag_hyde` vs. closed-book (p=0.0190), with `rag` vs.
closed-book close but not significant on the raw metric (p=0.0835) — consistent with the raw
metric's own documented zero-gate noise (see `_strip_context_gap_disclaimer()`'s docstring),
which the disclaimer-stripped variant exists specifically to correct for.

**HyDE does not show a statistically confirmed improvement over baseline RAG on this corpus.**
Every `rag → rag_hyde` comparison has p > 0.1 (most p > 0.35), and several point estimates are
essentially flat or slightly negative (`answer_correctness` -0.008, `faithfulness_c_only` -0.063,
though the latter's n=36 and the diagnostic `groundedness_g_only`'s n=4 are both too small to
draw any real conclusion from). The one metric with a directionally suggestive but
NOT-statistically-significant trend is `answer_relevancy` (+0.048, 95% CI barely excludes zero
at the lower bound [+0.0017, +0.1156], but Wilcoxon p=0.1138 fails the preregistered alpha=0.05
threshold). This mirrors this project's own earlier NFCorpus HyDE confirmatory finding
(`docs/CHANGELOG.md`, 2026-07-20: "`ragHyde` vs `ragBaseline` was NOT significant either before
or after the [disclaimer] fix") — two independent corpora now agree that this specific HyDE
implementation, as currently configured (Ollama `glm-5.2:cloud` hypothetical-document generation,
dense leg only), does not reliably beat baseline hybrid+rerank retrieval, at least at this sample
size.

**Retrieval quality itself is moderate, not exceptional, on this corpus.** `context_precision`
(~0.49-0.50) and `context_recall` (~0.65-0.67) leave real room for improvement — roughly a third
of retrieved context is judged irrelevant to the question, and roughly a third of what the
reference answer needs isn't showing up in the top-k retrieved chunks. `faithfulness` (~0.76-0.77)
and the `faithfulness_c_only` diagnostic (~0.77-0.80, scoring only the `[C]`-tagged /
context-attributed portion of each answer) are healthy and consistent with each other, suggesting
the generation step is honoring its `[C]`/`[G]` attribution policy reasonably well. The
`groundedness_g_only` diagnostic's very small n (4 valid rows out of 39 for `rag`, versus a
`c_claim_count`-implied near-total coverage) indicates most RAG answers on this corpus were
fully or near-fully context-grounded, with little general-knowledge supplementation needed —
a good sign for a corpus this specific (private team/personal knowledge a general model
couldn't fill gaps in from its own training anyway).

**Bottom line:** this is the first non-benchmark-contaminated evidence that this pipeline's core
retrieve-then-generate loop delivers real, measurable value over closed-book Claude on
questions about genuinely private, previously-unseen content — confirmed on the
disclaimer-stripped `answer_relevancy` metric for both RAG arms, and on the raw metric for
RAG+HyDE. HyDE specifically remains unconfirmed as an improvement over baseline RAG, on this
corpus, at N=39, consistent with the project's prior NFCorpus finding — not a new negative
result, a replication of an existing one on a completely different, uncontaminated corpus.
