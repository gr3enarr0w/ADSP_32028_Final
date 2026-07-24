<!-- Mirrored from Confluence: https://<YOUR_DOMAIN>.atlassian.net/wiki/spaces/~ceverson/pages/395806340 -->
<!-- Last synced: 2026-05-29 -->

# API Reference

## Authentication

All endpoints except `/api/health` require a Bearer token matching the `FAQ_API_TOKEN` environment variable:

```
curl -H "Authorization: Bearer <FAQ_API_TOKEN>" https://<host>/api/tickets
```

If `FAQ_API_TOKEN` is not set, all endpoints are unprotected. A startup warning is logged.

The webhook endpoint uses a separate HMAC secret — see [Webhook Integration](webhook.md).

---

## Choosing the Right Endpoint

| What you want to do | Endpoint |
|---|---|
| Look up FAQ/KB content for a ticket or topic | `POST /api/faq/lookup` |
| Manually trigger an AI draft for a ticket | `POST /api/auto-respond/{key}` |
| Check if the service is healthy | `GET /api/health` |
| See how accurate AI drafts have been | `GET /api/feedback/stats` |
| Batch-review tickets for disposition | `POST /api/review-tickets` |
| Browse generated FAQ articles | `GET /api/articles` |
| Publish a generated article to Confluence | `POST /api/articles/{id}/publish` |
| See ticket counts and classification coverage | `GET /api/stats` |

---

## Health

### `GET /api/health`

No authentication required. Used for OpenShift liveness and readiness probes.

```json
{
  "status": "ok",
  "pipeline_running": false,
  "last_pipeline_run": "2026-04-22T14:30:00Z",
  "last_trigger": "webhook",
  "refresh_interval_minutes": 5,
  "oauth_configured": {"jira": true, "confluence": true}
}
```

| Field | What it means |
|---|---|
| `pipeline_running` | `true` means a cycle is currently active |
| `last_pipeline_run` | Should update every 5 minutes; if stale, pipeline may be hung |
| `last_trigger` | `"webhook"` = working correctly; `"schedule"` = webhook not firing |
| `oauth_configured.jira` | `false` means OAuth credentials are missing or invalid |

---

## Webhook

### `POST /api/webhook/jira`

Receives Jira events via the `X-Webhook-Secret` header (not a query parameter). See [Webhook Integration](webhook.md) for full setup.

Always returns 200 — the pipeline runs in a background thread.

---

## Tickets

### `GET /api/tickets`

| Param | Type | Description |
|---|---|---|
| `status` | string | Filter by ticket status |
| `limit` | int | Max results (default 50, max 500) |
| `offset` | int | Pagination offset |

### `GET /api/tickets/{key}`

Returns full ticket with classification, all comments, and linked issues.

### `POST /api/auto-respond/{key}`

Manually trigger an AI draft. Returns `{"status": "posted"}` or 404 if no content found.

### `POST /api/review-tickets`

Batch disposition review — returns `{close, sprint_work, needs_action, stale, failed}` lists.

### `GET /api/feedback/stats`

Auto-responder accuracy breakdown by category (as_is, lightly_edited, heavily_rewritten, ignored).

---

## Articles

### `GET /api/articles` — list generated FAQ/how-to articles (`status=draft|published`)
### `GET /api/articles/{id}` — single article with full `body_html`
### `POST /api/articles/{id}/publish` — publish to Confluence (`?space_key=HUB`)

---

## Analytics (read-only, no external calls)

### `GET /api/stats` — ticket counts by status, classification coverage
### `GET /api/gaps` — KB coverage gap analysis
### `GET /api/predictions` — risk predictions by issue type
### `GET /api/linked-issues` — issues linked from <PROJECT_KEY> tickets (`?project=RHCLOUD`)
### `GET /api/doc-improvements` — documentation gap suggestions
### `GET /api/kb-articles` — crawled Confluence KB articles (`?space=HUB`)

---

## FAQ

### `POST /api/faq/lookup`

Given a ticket key or free-text topic, returns matched content and a draft response.

```json
{
  "found": true,
  "query": "<PROJECT_KEY>-1234",
  "faq_matches": [...],
  "kb_matches": [...],
  "ticket_matches": [...],
  "response_draft": "Here's how to resolve this..."
}
```

### `GET /api/faq/entries` — paginated list of generated FAQ entries
### `GET /api/faq/entries/{id}` — single entry with full `body_html`
### `GET /api/faq/sources` — configured sources and freshness
### `GET /api/faq/gaps` — FAQ-specific gap analysis
### `POST /api/faq/generate` — trigger generation (`{"theme": "..."}` or `{}` for all gaps)
### `POST /api/faq/export` — export all draft entries to output Google Doc

---

## Slack

### `POST /api/slack/jsm-assist`

Handler for the `/jsm-assist` Slack slash command. Validates `X-Slack-Signature` header.
