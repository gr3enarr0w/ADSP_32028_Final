<!-- Mirrored from Confluence: https://<YOUR_DOMAIN>.atlassian.net/wiki/spaces/~ceverson/pages/395806324 -->
<!-- Last synced: 2026-05-29 -->

# Configuration Reference

## How Configuration Works

**Bitwarden Secrets Manager** holds sensitive values — OAuth credentials, API tokens, Google service account JSON. Pulled automatically into the CI deploy pipeline and applied as Kubernetes Secrets.

**OpenShift ConfigMap** (`deploy/openshift/configmap.yaml`) holds non-sensitive defaults — project keys, model names, feature flags. Version-controlled, safe to commit.

For local development, create a `.env` file in the project root.

---

## Minimum Required

| Variable | Where | What breaks without it |
|---|---|---|
| `ATLASSIAN_OAUTH_CLIENT_ID` | Bitwarden SM | All Atlassian API calls fail |
| `ATLASSIAN_OAUTH_CLIENT_SECRET` | Bitwarden SM | All Atlassian API calls fail |
| `JSM_CLOUD_URL` | Bitwarden SM | No base URL for API requests |
| `GOOGLE_SERVICE_ACCOUNT_JSON` | Bitwarden SM | Gemini calls fail |
| `DATABASE_URL` | Secret (manual) | Falls back to SQLite; `ALTER TABLE DROP COLUMN` crashes on UBI9 SQLite |
| `FAQ_API_TOKEN` | Bitwarden SM | API endpoints unprotected (warning logged) |
| `JIRA_WEBHOOK_SECRET` | Bitwarden SM | Webhook endpoint unprotected (warning logged) |

---

## Atlassian OAuth

| Variable | Required | Description |
|---|---|---|
| `JSM_CLOUD_URL` | **Yes** | e.g. `https://<YOUR_DOMAIN>.atlassian.net` |
| `ATLASSIAN_OAUTH_CLIENT_ID` | **Yes** | OAuth app client ID (`jsm-helpdesk-agent`) |
| `ATLASSIAN_OAUTH_CLIENT_SECRET` | **Yes** | OAuth app client secret |
| `JSM_OAUTH_CLIENT_ID/SECRET` | No | JSM-specific override; falls back to primary |
| `JIRA_WRITE_CLIENT_ID/SECRET` | No | Jira write override; falls back to primary |
| `CONFLUENCE_OAUTH_CLIENT_ID/SECRET` | No | Confluence read override; falls back to primary |
| `CONFLUENCE_WRITE_CLIENT_ID/SECRET` | No | Confluence write override; falls back to primary |

Currently all product-specific overrides are unset — the primary credentials cover all operations.

---

## Gemini / Vertex AI

| Variable | Required | Default | Description |
|---|---|---|---|
| `GOOGLE_SERVICE_ACCOUNT_JSON` | **Yes** | `service_account.json` | Path or content. In OpenShift: `/app/secrets/service_account.json` |
| `GEMINI_PROJECT` | No | `your-gcp-project` | GCP project ID |
| `GEMINI_LOCATION` | No | `global` | Vertex AI region |
| `GEMINI_MODEL` | No | `gemini-2.5-flash` | Model ID for all Gemini calls |

---

## Database

| Variable | Required | Description |
|---|---|---|
| `DATABASE_URL` | **Yes (production)** | PostgreSQL connection string. Format: `postgresql://helpdesk:<password>@postgres:5432/helpdesk`. Unset = SQLite fallback (local dev only). |
| `DATA_DIR` | No | Directory for model artifacts. In OpenShift: `/app/models` (mounted PVC). |

---

## Jira Projects

| Variable | Default | Description |
|---|---|---|
| `PROJECT_KEYS` | `<PROJECT_KEY>` | Comma-separated JSM project keys |
| `CLOUD_CUTOVER_DATE` | `2026-03-16` | Identifies Cloud vs UAT-only tickets |

---

## Confluence KB

| Variable | Default | Description |
|---|---|---|
| `CONFLUENCE_KB_SPACE` | `HUB` | Default space for publishing articles |
| `FAQ_CONFLUENCE_SPACES` | `HUB,OMEGA` | Spaces searched during lookup |

---

## FAQ Service & Auto-Responder

| Variable | Default | Description |
|---|---|---|
| `FAQ_API_TOKEN` | — | Bearer token for protected API endpoints. **If unset, all endpoints are unprotected.** |
| `AUTO_RESPOND_ASSIGNEES` | ceverson account ID | Atlassian `accountId` values (comma-separated). Only these agents get auto-drafts. |
| `AUTO_DRAFT_ALL` | `true` | When `true`, drafts for all open tickets regardless of assignee. |

---

## Sentiment Model

| Variable | Default | Description |
|---|---|---|
| `SENTIMENT_BACKEND` | `gemini` | `gemini` (primary, 95.7% accuracy), `ensemble` (local cardiffnlp+emotion, 83.99%, ~250ms), `cardiffnlp` (alone) |

---

## Pipeline

| Variable | Default | Description |
|---|---|---|
| `PIPELINE_REFRESH_MINUTES` | `5` | Scheduled pipeline interval |
| `DISABLE_DOCS` | `false` | Set `true` in production to disable OpenAPI UI |

---

## Slack (optional)

| Variable | Description |
|---|---|
| `SLACK_BOT_TOKEN` | Bot token (`xoxb-...`) |
| `SLACK_CHANNELS` | Channel IDs to monitor |

All Slack features degrade gracefully if unset.
