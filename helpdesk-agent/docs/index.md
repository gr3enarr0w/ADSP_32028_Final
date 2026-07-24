<!-- Mirrored from Confluence: https://<YOUR_DOMAIN>.atlassian.net/wiki/spaces/~ceverson/pages/395806292 -->
<!-- Last synced: 2026-05-29 -->

# AI Helpdesk Agent

## Current system state

The AI Helpdesk Agent is a live FastAPI service running on OpenShift that automatically classifies, routes, and drafts responses for <PROJECT_KEY> support tickets. As of May 2026: the ensemble classifier (Macro-F1 0.7075) handles 8/10 categories; sentiment scoring (Gemini primary, 95.7%) gates high-frustration Access tickets to human review; and a weekly retrain CronJob keeps models current.

---

The AnTS Engineering support queue receives a steady flow of Tier 1 tickets — password resets, permission requests, "how do I do X in Jira Cloud?" — that an experienced agent can answer in two minutes but still has to read, triage, look up, and write a response for.

The AI Helpdesk Agent handles the repetitive part. When a ticket arrives in <PROJECT_KEY>, it reads the ticket, searches the knowledge base and resolved ticket history, drafts a response using Gemini, and posts it as an internal comment before the assigned agent even opens the ticket. Agents review, edit if needed, and send — or ignore it if it's off.

In parallel, the service continuously analyzes ticket patterns to identify gaps in FAQ and KB coverage, generates new articles to fill those gaps, and can publish directly to Confluence.

---

## What Happens When a Ticket Arrives

1. Ticket created/updated in <PROJECT_KEY>
2. Jira Automation rule fires webhook to the service immediately
3. Service ingests, scrubs PII, classifies with the ensemble (Gemini fallback for Notifications/Other)
4. M5 Router evaluates confidence + sentiment → `auto_draft`, `flag_only`, or `human_review`
5. Gemini drafts a response (self-service steps / admin action / needs info)
6. Draft posted as **internal comment** (not visible to customer)
7. Agent reviews, edits, sends — or writes their own response
8. Agents rate drafts with emoji reactions → feeds back into future draft quality

---

## Where to Start

**Support agent:** [Auto-Responder](auto-responder.md) — what drafts look like, rating emoji, trigger commands

**Deploying:** [Deployment](deployment.md) → [Configuration](configuration.md)

**Something broken:** [Runbook](runbook.md)

**Understanding the system:** [Architecture](architecture.md) → [Authentication](authentication.md)

**Integrating with the API:** [API Reference](api-reference.md) → [Webhook Integration](webhook.md)

**ML details:** [Classifier & ML Pipeline](classifier-ml-pipeline.md) → [Sentiment Model](sentiment-model.md) → [ML Operations](ml-operations.md)

---

## Current State

| Field | Value |
|---|---|
| Cluster | `prod-stable-spoke1-dc-iad2.itup.redhat.com` |
| Namespace | `jira-messaging--runtime-ext` |
| Database | PostgreSQL (StatefulSet + 10Gi PVC) |
| Pod status | 1/1 Running |
| Image | `ghcr.io/agile-tech-sol/ai-helpdesk-agent:latest` |

### Model status

| Metric | Value |
|---|---|
| Corpus | 8,384 tickets, 42,400 comments |
| Classifier | Ensemble (TF-IDF+LR + TF-IDF+GB + DistilBERT) |
| Macro-F1 (holdout, n=840) | 0.7075 |
| Accuracy | 89.8% |
| Sentiment | Gemini primary (95.7%), cardiffnlp+emotion fallback (83.99%) |
| Data split | 80/10/10 — 6,705 train / 839 val / 840 holdout |

### Deployed (Sprint 6, 2026-05-29)

- ANTSE-301 — DistilBERT embeddings
- ANTSE-302 — Optuna ensemble weight optimization
- ANTSE-303 — Ensemble in production
- ANTSE-304 — M5 Router
- ANTSE-305 — Retrain CronJob (Monday 9am UTC)
- ANTSE-322 — Sentiment model final
