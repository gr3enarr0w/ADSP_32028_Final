<!-- Mirrored from Confluence: https://<YOUR_DOMAIN>.atlassian.net/wiki/spaces/~ceverson/pages/395772381 -->
<!-- Last synced: 2026-05-29 -->

# Webhook Integration

## Why a Webhook

Without a webhook, a ticket created at minute 0 doesn't get a draft until the next 5-minute pipeline cycle. The webhook eliminates that gap — a Jira Automation rule fires immediately and triggers the pipeline within seconds.

## How It's Wired

**Jira Automation Rule** (configured, ready to activate after deploy):
- Fires on: ticket created, ticket updated
- Limitation: cannot fire on comment events

**Global Jira Webhook** (pending admin approval — ANTSE-193):
- Fires on: ticket created, ticket updated, comment created
- Needed for: `/ai-lookup`, `/ai-review`, `🤖` comment triggers to be instant

Until the global webhook is approved, comment triggers are processed on the next scheduled cycle.

## Activating After Deploy

**1. Get the Route URL:**
```bash
oc get route ai-helpdesk-agent -n jira-messaging--runtime-ext -o jsonpath='{.spec.host}'
```

**2. Update the Jira Automation rule** in <PROJECT_KEY> → Project Settings → Automation → "AI Helpdesk Webhook". Set the URL to:
```
https://<route-url>/api/webhook/jira
```
Set the `X-Webhook-Secret` header to the value from Bitwarden SM (`JIRA_WEBHOOK_SECRET`).

**3. Test it:**
```bash
curl https://<route-url>/api/health
```
Look for `"last_trigger": "webhook"`.

## How the Webhook Validates Requests

The `X-Webhook-Secret` header is validated via HMAC. If `JIRA_WEBHOOK_SECRET` is not configured, the endpoint accepts all requests (logs a startup warning). Always set it in production.

## What Each Event Does

| Event | Action |
|---|---|
| `jira:issue_created` / `issue_updated` | Full pipeline cycle + `handle_new_ticket()` for created |
| `comment_created` (global webhook only) | Routes to `/ai-lookup`, `/ai-review`, `🤖`, or emoji rating handler |

All handlers run in background threads. Always returns `{"status": "accepted"}` immediately.

## Pipeline Locking

Only one pipeline cycle runs at a time. If a webhook arrives during a running cycle, it's acknowledged but skipped — the running cycle picks up new tickets on its next ingest pass.

## Setting Up the Global Webhook (When Admin Approves)

Jira Administration → System → WebHooks → Create a WebHook:

| Field | Value |
|---|---|
| URL | `https://<route>/api/webhook/jira` with `X-Webhook-Secret` header |
| JQL filter | `project = <PROJECT_KEY>` |
| Events | Issue Created, Issue Updated, Comment Created |
