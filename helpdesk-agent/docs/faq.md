<!-- Mirrored from Confluence: https://<YOUR_DOMAIN>.atlassian.net/wiki/spaces/~ceverson/pages/412221492 -->
<!-- Last synced: 2026-06-18 -->

> **What this page covers:** Common questions about operating, understanding, and extending the AI Helpdesk Agent.  
> **For step-by-step troubleshooting by symptom**, see the Operations Runbook.  
> **For system architecture**, see Architecture.

---

## Operations

**Q: Why isn't the system drafting responses for new tickets?**

Check in order: (1) Is the pipeline running? `GET /api/health` should return `{"status": "ok"}`. (2) Does the ticket pass the age gate? Tickets must be at least 4 hours old by default (`responder.age_gate_hours` in `pipeline.yaml`). (3) Did the M5 Router route it to `flag_only` or `human_review`? Check the pod logs for `[router] TICKET-KEY →`. (4) Is there already a pending AI draft on the ticket? The system skips tickets that already have an internal draft comment.

→ See also: Operations Runbook, Architecture → M5 Router

---

**Q: How do I know if the ML classifier is working correctly?**

Run `python -m scripts.validate_models --quick` on the pod or locally (requires DB access). This runs 5-fold CV and holdout evaluation and prints per-class F1. Current baseline: Ensemble Macro-F1 0.7075, Accuracy 89.8%. If scores are significantly lower, the training data may have label noise or the ensemble needs retraining.

→ See also: Classifier & ML Pipeline

---

**Q: The retrain ran but the model wasn't promoted — is that a problem?**

No. "Skipped" is a valid and expected outcome. It means the new model didn't beat the current production model within the 1pp tolerance. The old model stays in production. This is the promotion gate working correctly. Check the structured JSON log output for `new_macro_f1` vs `prev_macro_f1` to understand the delta.

→ See also: Classifier & ML Pipeline → Weekly Retrain, Operations Runbook → Retrain CronJob Issues

---

**Q: A ticket went to human_review but I think it should have been auto-drafted. Why?**

The M5 Router applies four rules in priority order: (1) high frustration intensity (≥ 0.60) on Access tickets → human_review; (2) classifier confidence < 0.60 on a novel category → human_review; (3) classifier confidence < 0.60 on a known category → flag_only; (4) confidence ≥ 0.60 → auto_draft. Check the pod logs for `[router] TICKET-KEY → human_review (reason)` to see which rule fired and why.

→ See also: Architecture → M5 Router

---

**Q: How do I check what backend the sentiment model is using?**

Check the `SENTIMENT_BACKEND` key in the `ai-helpdesk-agent-config` ConfigMap. Unset or `gemini` = Gemini primary. `ensemble` = local cardiffnlp+emotion fallback. `cardiffnlp` = cardiffnlp alone. You can also check the pod logs at startup — the sentiment module logs which backend is active on first use.

→ See also: Sentiment Model (M2)

---

**Q: How do I force a retrain outside of the Monday schedule?**

```shell
oc create job --from=cronjob/retrain-cronjob manual-retrain-$(date +%s) \
  -n jira-messaging--runtime-ext
```

Monitor with: `oc logs job/manual-retrain-<timestamp> -n jira-messaging--runtime-ext -f`

→ See also: Operations Runbook → Retrain CronJob Issues

---

**Q: How do I know if the deployment is healthy?**

```shell
# Pod status
oc get pods -n jira-messaging--runtime-ext

# Health endpoint
curl https://<route-url>/api/health

# Recent pipeline logs
oc logs deployment/ai-helpdesk-agent -n jira-messaging--runtime-ext --tail=50
```

A healthy pod shows `{"status": "ok", "db": "ok"}` on the health endpoint and logs a pipeline cycle completion every 5 minutes.

→ See also: Operations Runbook → Health Check Interpretation

---

## ML & Data

**Q: Why does the system use an ensemble instead of just Gemini for classification?**

Two reasons: cost and latency. Gemini zero-shot classification has ~2,600ms latency and costs ~$X/month at scale. The ensemble runs at 26–32ms with near-zero cost for 8/10 categories. The ensemble also outperforms Gemini zero-shot on Macro-F1 (0.7075 vs ~0.35 baseline) because it was trained on domain-specific ticket data.

→ See also: Classifier & ML Pipeline → Why This Approach

---

**Q: What does "confidence < 0.60" mean in practice?**

The ensemble outputs a probability distribution across all 10 categories. "Confidence" is the maximum probability in that distribution — how sure the model is about its top prediction. At confidence < 0.60, the model is hedging between two or more categories. From the holdout calibration: confidence < 0.43 → only 52% accuracy; confidence ≥ 0.60 → 96.4%+ accuracy. The 0.60 threshold was chosen by reading the calibration curve, not set arbitrarily.

→ See also: Classifier & ML Pipeline → Validation Results

---

**Q: How do I add a new ticket category?**

This requires: (1) labeling existing tickets with the new category in `ticket_classifications`, (2) waiting for a retrain run with ≥ 10 labeled samples so the ensemble picks it up (below 10 samples, Gemini handles it automatically), (3) updating the `CATEGORIES` list in `analysis/ml/tfidf_lr.py`. There is no config-only path — adding a category touches training data and code.

---

**Q: Why are Notifications and Other routed to Gemini and not the ensemble?**

They have fewer than 10 training samples (Notifications: 39 total but variable per split; Other: 5 total). The ensemble minimum sample threshold (`ensemble_min_samples: 10` in `pipeline.yaml`) prevents training on categories where there isn't enough signal. Below the threshold, the ensemble's predictions are unreliable and Gemini's zero-shot is more accurate.

→ See also: Classifier & ML Pipeline → Ensemble Architecture

---

**Q: How long before a retrain captures new ticket patterns?**

New tickets are classified immediately via the current ensemble. They feed into the training data from that point forward. The next Monday retrain will pick them up — so at most 7 days lag before new patterns influence the model. Categories with fewer than 10 samples continue to use Gemini until the threshold is crossed.

---

**Q: What is the dedup threshold and how was it calibrated?**

The dedup system uses **Vertex AI `gemini-embedding-001`** to embed each ticket's text, then applies a cosine similarity threshold to decide whether two tickets are likely duplicates. The current threshold is **0.63**, calibrated on 800 pairs from the JSM corpus using 5-fold × 5-seed cross-validation (F1=0.626, threshold std=0.000 — stable).

The threshold was previously uncalibrated because the earlier MiniLM embedding model produced no usable score separation on JSM ticket language: duplicate and distinct pairs landed at nearly identical cosine similarities, and the "optimal" threshold kept hitting the 0.30 floor. Switching to `gemini-embedding-001` — which is trained on enterprise and support-domain text — gave the first usable operating point.

The current F1 of 0.626 is a **pre-backfill baseline**. The calibration used `summary + description` as input text. Once the resolution summary backfill (3,543 tickets) completes, calibration will be re-run using `resolution_summary` as the primary embedding text, which should produce tighter score separation and higher F1 — resolution summaries capture the root cause of a ticket in a way that raw descriptions often don't.

→ See also: Classifier & ML Pipeline → Duplicate Detection, Resolution Summary Auto-Generation

---

## Deployment

**Q: What secrets does this service need and where do they come from?**

Secrets come from the `ai-helpdesk-agent-secrets` Secret in OpenShift (populated from Bitwarden SM in the CI/CD pipeline). Required: Atlassian OAuth client credentials (JSM, Jira write, Confluence), Google service account JSON (Vertex AI), database URL (Postgres), FAQ API token. See `deploy/openshift/secret.yaml` for the full list.

→ See also: Configuration, Authentication, Deployment

---

**Q: The webhook isn't triggering for new tickets. What do I check?**

Check: (1) Does the Jira webhook configuration send the `X-Webhook-Secret` header (not a `?secret=` query param — this was changed for security)? (2) Is the webhook URL pointing to the current route? (3) Check pod logs for incoming webhook requests — if the request arrives but fails validation, the pod logs the reason. (4) If no requests arrive at all, the Jira Automation rule may be disabled or the global webhook may not be configured.

→ See also: Webhook Integration, Operations Runbook → Webhook Not Triggering

---

**Q: The pod is in ImagePullBackOff — how do I fix it?**

Two common causes: (1) The `ghcr-pull-secret` is expired or has the wrong username — recreate it with a fresh classic GitHub PAT that has `read:packages` scope. (2) The secret exists but wasn't linked to the service account — run `oc secrets link default ghcr-pull-secret --for=pull -n jira-messaging--runtime-ext` after creating or recreating the secret, then restart the deployment.

→ See also: Operations Runbook → Pod Startup Failures

---

**Q: The pod crashes with an SCC error — "unable to validate against any security context constraint"**

The pod manifest has a hardcoded `runAsUser` or `fsGroup` that conflicts with the namespace's UID range (1002740000-1002749999). Remove `runAsUser` and `fsGroup` from the pod-level `securityContext` — OpenShift assigns these automatically from the namespace allocation. Keep the container-level `securityContext` with `allowPrivilegeEscalation: false`, `capabilities.drop: [ALL]`, and `seccompProfile.type: RuntimeDefault`.

→ See also: Operations Runbook → Pod Startup Failures
