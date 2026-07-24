<!-- Mirrored from Confluence: https://<YOUR_DOMAIN>.atlassian.net/wiki/spaces/~ceverson/pages/395806308 -->
<!-- Last synced: 2026-05-29 -->

# Architecture

## How the System Works

The AI Helpdesk Agent is a single FastAPI container running on OpenShift with a PostgreSQL database (StatefulSet) in the same namespace. It does two things continuously: processes tickets through a classification and drafting pipeline, and serves a REST API for lookups, manual triggers, and monitoring.

---

## A Ticket's Journey

**1. Arrival** — A ticket is created in <PROJECT_KEY>. A Jira Automation rule fires immediately and POSTs the ticket key to `/api/webhook/jira`. The 5-minute scheduled pipeline poll acts as a fallback.

**2. Ingestion** — Ticket and comments are fetched from JSM via the `servicedeskapi` queue endpoint (not Jira REST v3 — see [Authentication](authentication.md)).

**3. PII Scrubbing** — Email addresses, usernames, API keys, and IP addresses are stripped before anything reaches Gemini.

**4. Classification** — The ML ensemble classifier assigns a `category`. For 8/10 categories the ensemble runs at 26–32ms. For Notifications and Other (sparse training data), Gemini fallback applies.

**5. Routing** — The M5 Router evaluates classifier confidence and sentiment intensity to decide: `auto_draft`, `flag_only`, or `human_review`.

**6. Lookup** — Searches FAQ entries, Confluence KB articles, resolved tickets, and indexed Atlassian docs.

**7. Drafting** — Matched content + up to 5 few-shot examples → Gemini → draft posted as internal comment.

**8. Feedback** — Agents rate drafts with emoji. Agent responses are scored using embedding cosine similarity (all-MiniLM-L6-v2) and feed back into future few-shot examples.

---

## ML Classification Layer

### What it does
Classifies tickets into 10 categories: Access, Configuration, Permissions, Integration, UI/UX, Workflow, Data, Performance, Notifications, Other.

### Architecture

| Model | Weight | Notes |
|---|---|---|
| TF-IDF + Logistic Regression | 0.12 | Fast sparse baseline |
| TF-IDF + Gradient Boosting | 0.25 | Non-linear patterns |
| DistilBERT / all-MiniLM-L6-v2 LR head | 0.62 | Dominant contributor; 384-dim embeddings |

Weights are Optuna OOF-optimized. Production Macro-F1: **0.7075** on 840-ticket holdout.

### Where it lives

| Path | Purpose |
|---|---|
| `analysis/classifier.py` | Production inference |
| `analysis/ml/` | Training scripts |
| `models/*.pkl` | Serialized model artifacts |
| `models/distilbert_embeddings.npz` | Cached embeddings |
| `models/production_metrics.json` | Promotion gate baseline |

---

## M5 Router

Sits between the classifier and auto-responder. Routes each ticket to `auto_draft`, `flag_only`, or `human_review`.

### Rules (priority order)

| Priority | Condition | Outcome |
|---|---|---|
| 1 | Access ticket AND sentiment intensity ≥ 0.60 | `human_review` |
| 2 | Confidence < 0.60 AND novel/unknown category | `human_review` |
| 3 | Confidence < 0.60 AND known category | `flag_only` |
| 4 | Confidence ≥ 0.60 | `auto_draft` |

Lives in `analysis/router.py`, wired into `faq/auto_responder.py` and `main.py`.

---

## Weekly Retrain CronJob

**Schedule:** Monday 9am UTC (`0 9 * * 1`), 4-hour deadline.

**Pipeline:** retrain LR/GB/DistilBERT → Optuna OOF weight optimization → holdout evaluation → promotion gate (must meet or exceed production Macro-F1 within 1pp).

Lives in `scripts/retrain.py` + `deploy/openshift/retrain-cronjob.yaml`.

---

## Module Breakdown

| Module | What it does |
|---|---|
| `main.py` | FastAPI app, pipeline scheduler, all REST endpoints |
| `config.py` | Env var loading, credential routing |
| `db.py` | Dual-backend (PostgreSQL in production, SQLite locally); controlled by `DATABASE_URL` |
| `ingest/tickets.py` | Fetch tickets + comments from JSM servicedeskapi; ADF text extraction |
| `ingest/oauth2lo.py` | OAuth 2LO token exchange, caching, per-product credential selection |
| `ingest/scrubber.py` | PII removal before text reaches Gemini |
| `analysis/classifier.py` | Ensemble ML classification; Gemini fallback for Notifications/Other |
| `analysis/router.py` | M5 Router — maps classifier output + sentiment to handling path |
| `faq/auto_responder.py` | Full auto-responder pipeline: route → lookup → draft → post → feedback |
| `faq/lookup.py` | Multi-source FAQ/KB/ticket search |
| `faq/analyzer.py` | Gap analysis: ticket themes vs. FAQ coverage |
| `faq/generator.py` | Gemini-powered FAQ article generation |
| `scripts/retrain.py` | Weekly model retraining CronJob entrypoint |

---

## The Pipeline Cycle

Runs every 5 minutes (configurable via `PIPELINE_REFRESH_MINUTES`). Only one cycle runs at a time.

**Phase 1 — Ingest:** Fetch tickets from JSM queue.  
**Phase 2 — Scrub PII:** Strip emails, usernames, API keys.  
**Phase 3 — Classify:** Ensemble classifier; Gemini for Notifications/Other and resolved ticket summaries.  
**Phase 4 — Route and Draft:** M5 Router → auto-responder for `auto_draft` tickets.  
**Phase 5 — FAQ Pipeline:** Sources → gap analysis → FAQ generation → export.  
**Phase 6 — Background:** Feedback capture, response harvesting, doc index refresh (daily), gap analysis (daily), retrain (Monday).

---

## Database

**Production: PostgreSQL** — StatefulSet in `jira-messaging--runtime-ext`, 10Gi PVC. Connection via `DATABASE_URL` in `ai-helpdesk-agent-secrets` (`postgresql://helpdesk:<password>@postgres:5432/helpdesk`).

**Local development: SQLite** — falls back when `DATABASE_URL` is not set.

**Models/artifacts** — stored on `ai-helpdesk-agent-data` PVC mounted at `/app/models`.

---

## External Dependencies

| System | What we call | Why |
|---|---|---|
| JSM Cloud (`servicedeskapi`) | Ticket + comment read; internal comment write | Core data source and output |
| Gemini / Vertex AI | Classification fallback, drafting, FAQ generation, sentiment primary | The AI layer |
| Google Workspace | Read: FAQ source docs/sheets/slides. Write: output FAQ doc | Source material and export |
| Confluence REST | KB article search (CQL); article publish | Knowledge source and publication |
| Slack | Read: resolved Q&A threads. Write: `/jsm-assist` responses | Optional enrichment |
| OpenShift | Container runtime, persistent volumes, Routes, CronJob scheduler | Production deployment |
