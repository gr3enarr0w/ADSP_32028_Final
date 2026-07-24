<!-- Mirrored from Confluence: https://<YOUR_DOMAIN>.atlassian.net/wiki/spaces/~ceverson/pages/412123241 -->
<!-- Last synced: 2026-05-29 -->

> **What this page covers:** How frustration intensity is scored and how it feeds the M5 router's Access ticket escalation rule.  
> **What it doesn't cover:** The routing rules themselves (see Architecture → M5 Router) or the classification model (see Classifier & ML Pipeline).

## Overview

Each ticket is optionally scored for sentiment intensity — specifically, the probability that the user is frustrated. This score feeds one rule in the M5 Router: Access tickets with intensity ≥ 0.60 are escalated to human_review regardless of classifier confidence.

## Model Selection

We evaluated 4 models against a 50-ticket ground truth dataset labeled by Gemini 2.5 Flash (LLM-as-annotator methodology, Ziems et al. 2024).

| Model | Macro-F1 | Latency | Notes |
| --- | --- | --- | --- |
| **Gemini 2.5 Flash (zero-shot)** | **95.7%** | ~2,646ms | Primary |
| cardiffnlp + emotion/hartmann (soft-vote) | 83.99% | ~250ms | Fallback |
| cardiffnlp alone | 69.1% | 221ms | Baseline |
| distilbert-sst2 | 23.7% | 32ms | Failed — no NEUTRAL class |
| VADER | 26.0% | < 1ms | Failed — social media training doesn't generalise to IT language |

## Why Gemini Wins

IT helpdesk language is indirect and professional. Frustration is implied by context ("I have been waiting three weeks and nothing works") not by explicit sentiment words. Local models trained on social media or movie reviews don't detect this. Gemini's zero-shot prompt includes IT-specific label definitions, enabling it to interpret contextual frustration correctly.

## Why Recall > Precision

Missing a frustrated user and sending them an auto-draft is worse than over-escalating to human review. The 0.60 intensity threshold was chosen to prioritise recall — it is intentionally permissive on the false positive side.

## Production Setup

Controlled by the `SENTIMENT_BACKEND` environment variable in the ConfigMap:

| Value | Model | Macro-F1 | Latency | RAM |
| --- | --- | --- | --- | --- |
| `gemini` (default) | Gemini 2.5 Flash | 95.7% | ~2,646ms | API |
| `ensemble` | cardiffnlp + emotion soft-vote | 83.99% | ~250ms | ~1.6GB |
| `cardiffnlp` | cardiffnlp alone | 69.1% | 221ms | ~500MB |

Estimated cost at current traffic (~500 tickets/day): **~$3/month** for Gemini primary. Self-hosted break-even vs Gemini is ~250,000 requests/month — far beyond current scale.

## Output Format

```json
{"label": "NEGATIVE", "score": 0.94, "intensity": 0.94}
```

The `intensity` field is the NEGATIVE label confidence (0–1). The M5 router thresholds on `intensity ≥ 0.60` for Access tickets only.

## Codebase Location

`plugins/feedback/sentiment.py` — `score_ticket(text)` and `score_tickets(texts)` (batch).

## Troubleshooting

**Symptom: Access tickets always going to human_review**

* Likely cause: sentiment model returning high intensity for neutral tickets
* Fix: `python -c "from plugins.feedback.sentiment import score_ticket; print(score_ticket('Please reset my password'))"` — should return NEUTRAL with intensity ~0.0; if NEGATIVE with intensity > 0.60, check SENTIMENT_BACKEND and Gemini API key

**Symptom: Slow ticket processing (> 5s per ticket)**

* Likely cause: Gemini primary is experiencing high latency
* Fix: set `SENTIMENT_BACKEND=ensemble` in ConfigMap for local fallback (~250ms); monitor Vertex AI latency in GCP console

**Symptom: Pod OOMKilled after restart**

* Likely cause: ensemble fallback model (~1.6GB) loading alongside DistilBERT embeddings
* Fix: ensure pod memory limit ≥ 4Gi in `deploy/openshift/deployment.yaml`; the 4Gi limit is already set — verify it wasn't overridden
