<!-- Mirrored from Confluence: https://<YOUR_DOMAIN>.atlassian.net/wiki/spaces/~ceverson/pages/395806356 -->
<!-- Last synced: 2026-06-18 -->

# Operations Runbook

## Checking Health

```bash
curl https://ai-helpdesk-agent.apps.ext-waf.prod-stable-spoke1-dc-iad2.itup.redhat.com/api/health
```

| Field | Healthy | Investigate if... |
|---|---|---|
| `status` | `"ok"` | `"error"` — check pod logs |
| `pipeline_running` | `true` or `false` | Stuck on `true` for >10 minutes |
| `last_pipeline_run` | Updated every 5 minutes | More than 15 minutes old |
| `last_trigger` | `"webhook"` | `"schedule"` — webhook not firing |
| `oauth_configured.jira` | `true` | `false` — OAuth credentials missing |

---

## AI Drafts Stopped Posting

Check in order:
1. Is `AUTO_DRAFT_ALL=true` in ConfigMap?
2. Is the ticket assigned to an agent in `AUTO_RESPOND_ASSIGNEES`?
3. Does the ticket already have an agent public comment?
4. Does the FAQ lookup find any matches? — `POST /api/faq/lookup {"query": "<PROJECT_KEY>-1234"}`

---

## Pipeline Not Running

**Symptom:** `last_pipeline_run` stale or `pipeline_running` stuck on `true`.

```bash
oc logs deployment/ai-helpdesk-agent -n jira-messaging--runtime-ext --tail=100
```

If hung: `oc rollout restart deployment/ai-helpdesk-agent -n jira-messaging--runtime-ext`

---

## 401 Errors in Logs

**Normal / expected:**
- `_delete_comment() returned 401` — REST v3 delete doesn't work with service account tokens; trigger comment stays (cosmetic)
- `_auto_draft_sweep() returned 401` — JQL sweep uses REST v3; new tickets still caught via webhook

**Worth investigating:**
- `Got 401 on servicedeskapi/request/...` — JSM endpoints failing, core functionality broken

---

## Webhook Not Triggering

**Symptom:** `last_trigger` stays on `"schedule"`.

1. Confirm Jira Automation rule is enabled
2. Check rule URL matches current Route URL
3. Verify `X-Webhook-Secret` header matches `JIRA_WEBHOOK_SECRET` in Bitwarden SM
4. Test manually: `curl -X POST -H "X-Webhook-Secret: <secret>" -H "Content-Type: application/json" -d '{"webhookEvent":"jira:issue_created","issue":{"key":"<PROJECT_KEY>-1"}}' https://<route>/api/webhook/jira`

---

## Pod Won't Start

```bash
oc describe pod -l app=ai-helpdesk-agent -n jira-messaging--runtime-ext
oc logs -l app=ai-helpdesk-agent -n jira-messaging--runtime-ext --previous
```

| Symptom in logs | Cause | Fix |
|---|---|---|
| `service_account.json not found` | SA JSON secret not mounted | Apply `ai-helpdesk-agent-sa-json` secret |
| `ImagePullBackOff` | Can't pull from ghcr.io | Recreate `ghcr-pull-secret` + run `oc secrets link default ghcr-pull-secret --for=pull` |
| `unable to validate against any security context constraint` | Hardcoded `runAsUser`/`fsGroup` in pod spec | Remove pod-level securityContext; OpenShift assigns UIDs automatically (range: 1002740000-1002749999) |
| `sqlite3.OperationalError: near DROP: syntax error` | `DATABASE_URL` not set; app fell back to SQLite | Set `DATABASE_URL` in `ai-helpdesk-agent-secrets` pointing to Postgres |

---

## Accessing the Database

```bash
# Connect to PostgreSQL
oc exec -it statefulset/postgres -n jira-messaging--runtime-ext -- psql -U helpdesk helpdesk
```

Useful queries:
```sql
-- Ticket counts by status
SELECT status, COUNT(*) FROM tickets GROUP BY status;

-- Recent classifications
SELECT category, issue_type, COUNT(*) FROM ticket_classifications
GROUP BY category, issue_type ORDER BY COUNT(*) DESC LIMIT 20;

-- Draft quality breakdown
SELECT feedback_category, COUNT(*) FROM ai_draft_feedback
WHERE feedback_category IS NOT NULL GROUP BY feedback_category;
```

---

## Retrain CronJob Issues

**Symptom: Job hasn't run on schedule**
```bash
oc get jobs -n jira-messaging--runtime-ext | grep retrain
oc delete job <stale-job-name> -n jira-messaging--runtime-ext
# Force immediate run:
oc create job --from=cronjob/ai-helpdesk-retrain manual-retrain-$(date +%s) -n jira-messaging--runtime-ext
```

**Symptom: Promotion gate rejected the model** — Expected. No action needed. The old model stays.

**Symptom: `production_metrics.json` not found**
```bash
oc exec -it deployment/ai-helpdesk-agent -n jira-messaging--runtime-ext -- \
  bash -c 'echo "{\"macro_f1\": 0.7075, \"accuracy\": 0.8976, \"weighted_f1\": 0.9034}" > /app/models/production_metrics.json'
```

---

## M5 Router — Unexpected Routing

**High human_review rate:**
- Check confidence distribution: `python -m scripts.validate_models --quick` on pod
- Check sentiment backend: `oc get configmap ai-helpdesk-agent-config -n jira-messaging--runtime-ext -o yaml | grep SENTIMENT_BACKEND`

**Adjust thresholds** (in `analysis/router.py`, then redeploy):
```python
CONFIDENCE_THRESHOLD = 0.60
HIGH_SENTIMENT_THRESHOLD = 0.60
```

---

## Sentiment Model Issues

**Slow (>5s/ticket):** Set `SENTIMENT_BACKEND=ensemble` in ConfigMap → restart pod (switches to local cardiffnlp+emotion, ~250ms)

**Memory spike after restart:** Expected — cardiffnlp+emotion loads ~1.6GB RAM on first use. Allow 30-60s warmup.

**All Access tickets → human_review:**
```bash
oc exec -it deployment/ai-helpdesk-agent -n jira-messaging--runtime-ext -- \
  python -c "from plugins.feedback.sentiment import score_ticket; print(score_ticket('Please reset my password'))"
```
Expected: NEUTRAL with low intensity.

---

## Pod Startup Failures

**`ImagePullBackOff` — unauthorized:**
```bash
oc delete secret ghcr-pull-secret -n jira-messaging--runtime-ext
oc create secret docker-registry ghcr-pull-secret \
  --docker-server=ghcr.io --docker-username=<github-user> \
  --docker-password=<classic-pat-read-packages> \
  -n jira-messaging--runtime-ext
# CRITICAL: link to service account
oc secrets link default ghcr-pull-secret --for=pull -n jira-messaging--runtime-ext
oc rollout restart deployment/ai-helpdesk-agent -n jira-messaging--runtime-ext
```

---

## PostgreSQL Issues

**StatefulSet stuck at 0/1:** Check `postgres-secret` exists with POSTGRES_PASSWORD set.

**App DB errors:** Restart the app — `init_db()` initializes the schema automatically on startup. If password mismatch, verify DATABASE_URL password matches POSTGRES_PASSWORD.

---

## Resolution Summary Backfill

The `resolution_summary` column in `ticket_classifications` is populated automatically during each pipeline cycle for newly resolved tickets (up to 50 per cycle). For existing resolved tickets, a one-time backfill is run separately.

If resolved tickets have an empty `resolution_summary` — particularly after the initial 2026-06-18 backfill — the most likely cause is that the backfill was interrupted. Check coverage first:

```sql
SELECT
  COUNT(*) FILTER (WHERE resolution_summary IS NOT NULL) AS with_summary,
  COUNT(*) FILTER (WHERE resolution_summary IS NULL) AS without_summary
FROM ticket_classifications tc
JOIN tickets t ON tc.ticket_key = t.ticket_key
WHERE t.status IN ('Resolved', 'Closed');
```

To restart the backfill:

```python
python -c "from db import get_db; from analysis.resolution_summary import backfill_resolution_summaries; conn=get_db(); backfill_resolution_summaries(conn); conn.close()"
```

The backfill is resumable — it skips tickets that already have a summary — so it's safe to re-run at any time without duplicating work.

---

## Restarting

```bash
oc rollout restart deployment/ai-helpdesk-agent -n jira-messaging--runtime-ext
oc rollout status deployment/ai-helpdesk-agent -n jira-messaging--runtime-ext
oc logs -f deployment/ai-helpdesk-agent -n jira-messaging--runtime-ext
```
