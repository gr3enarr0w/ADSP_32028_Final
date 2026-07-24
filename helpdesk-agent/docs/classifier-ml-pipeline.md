<!-- Mirrored from Confluence: https://<YOUR_DOMAIN>.atlassian.net/wiki/spaces/~ceverson/pages/412155992 -->
<!-- Last synced: 2026-06-18 -->

> **What this page covers:** How tickets get classified into categories, how the training pipeline works, and how models stay current.  
> **What it doesn't cover:** How routing decisions are made (see Architecture → M5 Router) or how drafts are generated (see Auto-Responder).

## Overview

Each incoming ticket is classified into one of 10 categories by a trained ensemble model. Classification drives routing (M5 Router), FAQ lookup, and draft generation. The ensemble is retrained weekly on the full ticket corpus.

## Corpus

| Property | Value |
| --- | --- |
| Total tickets | 8,384 (<PROJECT_KEY>) |
| Total comments | 42,400 |
| Date range | 2025-03-03 → 2026-05-28 |
| Backfill method | JQL search via `scripts/backfill_corpus.py` using personal API token (OAuth 2LO app lacks `read:jira-work` scope) |

**PII handling:** Text fields scrubbed (emails, credentials, bearer tokens, SSNs, credit cards). `reporter_email` cleared. Atlassian account IDs kept as opaque UUIDs — not reversible without org admin access.

## Categories (10)

Access, Configuration, Permissions, Integration, UI/UX, Workflow, Data, Performance, Notifications, Other

## Data Split (80/10/10)

| Split | Count | Fraction | Seed |
| --- | --- | --- | --- |
| Training | 6,705 | 80% | — |
| Validation | 839 | 10% | 6271 |
| Holdout (test) | 840 | 10% | 7919 |

Splits are stratified by category. The holdout set is SHA-locked (`7e8a345c9e80`) and integrity-verified before every retrain run. It is never used for training or hyperparameter selection.

## Ensemble Architecture

Three base models → weighted soft-vote → single category prediction.

| Model | Weight | Macro-F1 (holdout) | Latency |
| --- | --- | --- | --- |
| TF-IDF + Logistic Regression | 0.12 | 0.6022 | < 5ms |
| TF-IDF + Gradient Boosting | 0.25 | 0.5572 | < 10ms |
| DistilBERT (all-MiniLM-L6-v2) + LR head | 0.62 | 0.6783 | 20–30ms |
| **Ensemble (soft-vote)** | — | **0.7075** | **26–32ms** |

Weights are Optuna TPE-optimized using 5-fold OOF cross-validation on the training set (200 trials). DistilBERT dominates because its semantic embeddings provide better-calibrated probabilities for minority categories.

**Gemini fallback:** Notifications and Other have fewer than 10 training samples and use Gemini zero-shot classification instead of the ensemble (~2,600ms, higher accuracy on novel patterns).

## Validation Results

| Model | CV Macro-F1 (mean ± std) | Holdout Macro-F1 | Holdout Accuracy |
| --- | --- | --- | --- |
| TF-IDF + LR | 0.646 ± 0.025 | 0.6022 | 81.7% |
| TF-IDF + GB | 0.549 ± 0.025 | 0.5572 | 80.1% |
| DistilBERT + LR head | 0.673 ± 0.033 | 0.6783 | 87.0% |
| **Ensemble** | **0.703 ± 0.039** | **0.7075** | **89.8%** |

**Confidence calibration:** ECE = 0.161. At confidence ≥ 0.60 (the M5 router threshold), holdout accuracy is 96.4–100%. At confidence < 0.43, accuracy is 52.4% — these tickets correctly go to human_review.

**Learning curves:** Ensemble F1 plateaus at 50–80% training data (0.732 at 50%, 0.703 at 80%). The model is saturating current ticket vocabulary. Gains resume when ticket volume grows or category distributions shift after a migration wave.

## Resolution Summary Auto-Generation

Each time a ticket transitions to a resolved or closed state, the pipeline generates a concise 1–2 sentence summary of how it was resolved. The goal is not to summarize the ticket itself — the ticket summary and description already do that — but to capture the arc of the conversation: what the problem was and which agent actions or comments led to the solution.

The generation runs inside the standard 5-minute pipeline cycle, capped at 50 resolved tickets per cycle to stay comfortably within the cycle's time budget. Tickets that already have a `resolution_summary` are skipped, so the cap doesn't create a backlog problem — it just means a large surge of resolved tickets gets processed over several cycles rather than one.

The output is stored in `ticket_classifications.resolution_summary` and goes nowhere else. Nothing is written back to Jira. This is purely internal enrichment data.

**Why this matters for deduplication:** The resolution summary becomes the primary text source for the deduplication embedding. It's a more semantically precise signal than `summary + description` alone, because two tickets with the same root cause — say, "access removed during offboarding cycle" — will produce similar resolution summaries even when their original descriptions were phrased very differently. See the dedup calibration section below for how this feeds in.

**Backfill status:** A one-time backfill of 3,543 resolved tickets is running as of 2026-06-18. Once complete, dedup calibration will be re-run against the full resolution_summary corpus for a more accurate F1 baseline.

**Codebase:** `analysis/resolution_summary.py`

## Duplicate Detection — Calibration Status

The system includes a deduplication layer that checks whether an incoming ticket is semantically close enough to a recently seen ticket to be flagged as a likely duplicate. This uses cosine similarity over sentence embeddings to score each pair, then applies a learned threshold to decide "duplicate" vs "distinct".

The previous calibration attempt using `sentence-transformers/all-MiniLM-L6-v2` produced unusable results: the optimal threshold hit the 0.30 floor, with duplicate and distinct pairs nearly indistinguishable. The embedding backend has been replaced with **Vertex AI `gemini-embedding-001`**, which is fine-tuned on enterprise and support-domain text.

Results from the 2026-06-18 calibration run on 800 pairs (input: `summary + description`, pre-backfill):

| Metric | Value |
| --- | --- |
| Optimal threshold | 0.63 |
| F1 at threshold (test) | 0.626 |
| Precision / Recall | 0.569 / 0.695 |
| CV F1 mean (5-fold × 5-seed) | 0.589 ± 0.034 |
| Threshold std | 0.000 — STABLE, safe to deploy |

The current F1 of 0.626 is a **pre-backfill baseline**. Once the resolution summary backfill completes, calibration will be re-run using `resolution_summary` as the primary embedding text, which should produce tighter score separation and higher F1.

Calibration command: `scripts/calibrate_dedup.py --max-per-group 30 --min-pairs 800 --n-seeds 5 --n-folds 5 --save`  
Results stored in: `faq/calibration_result.json`

## Why This Approach

**Why ensemble over Gemini-only?**  
Gemini zero-shot baseline was ~0.35 Macro-F1 with 2,600ms latency. The ensemble reaches 0.7075 at 26–32ms for 8/10 categories. Cost is near-zero at current scale vs ~$X/month for full Gemini classification.

**Why all-MiniLM-L6-v2?**  
80MB, CPU-safe, 384-dim embeddings. No GPU required. Self-hosted on OpenShift with zero per-call cost. Evaluated against larger models — diminishing returns didn't justify the RAM or latency increase.

**Why 80/10/10 and not 80/20?**  
A 20% holdout from 974 tickets (196 samples) was too small and unrepresentative of the full corpus. After backfilling to 8,384 tickets, a proper 10% holdout (840 tickets) is large enough to give reliable Macro-F1 estimates with tight confidence intervals.

## Weekly Retrain

Every Monday 9am UTC. Pipeline: data guard → retrain LR/GB/DistilBERT → Optuna OOF weight optimization → holdout evaluation → promotion gate.

The promotion gate: new model must meet or exceed current production Macro-F1 within 1pp tolerance. If rejected, the old model stays. If promoted, pkl files and `models/production_metrics.json` are updated atomically.

Current production baseline: **Macro-F1 0.7075** (set in `models/production_metrics.json`).

## Codebase Locations

| Component | Path |
| --- | --- |
| Production classifier | `analysis/classifier.py` |
| M5 Router | `analysis/router.py` |
| Training scripts | `analysis/ml/` |
| Model artifacts | `models/*.pkl`, `models/distilbert_embeddings.npz` |
| Retrain entrypoint | `scripts/retrain.py` |
| Validation suite | `scripts/validate_models.py` |
| Holdout management | `analysis/ml/holdout.py` |
| Resolution summary generation | `analysis/resolution_summary.py` |
| Dedup calibration | `scripts/calibrate_dedup.py`, results in `faq/calibration_result.json` |

## Troubleshooting

**Symptom: All tickets routing to Gemini fallback**

* Likely cause: `ticket_classifications` table is nearly empty (was wiped or DB is fresh)
* Fix: the classifier auto-detects this and falls back to model class list; check logs for "DB has too few classifications — using model classes instead"; run `classify_unclassified()` to repopulate

**Symptom: Retrain promoted a model with worse per-class F1 on minority categories**

* Likely cause: promotion gate only checks overall Macro-F1, not per-class; a small regression on Notifications (n=39) can be masked
* Fix: after promotion, run `python -m scripts.validate_models --quick` and review per-class holdout F1; roll back by restoring previous pkl files from git history

**Symptom: Embedding cache SHA mismatch on retrain**

* Likely cause: training set size changed but cache wasn't rebuilt
* Fix: `rm models/distilbert_embeddings.npz && python -m analysis.distilbert train`
