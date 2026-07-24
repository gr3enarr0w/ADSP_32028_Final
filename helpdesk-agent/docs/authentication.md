# Authentication

## Why a Service Account, Not User Login

The agent authenticates to Atlassian APIs as the `ants-engineering` service account using OAuth 2.0 Client Credentials — also called 2LO (two-legged OAuth). This is a deliberate choice worth understanding.

User-delegated OAuth (3LO) ties the service to an individual's account. If that person leaves, their tokens break. It also means every API call is attributed to a real user, which creates audit noise and requires the service to impersonate someone. A service account avoids all of that: it has its own identity in the Atlassian tenant, its own access scopes, and its credentials rotate independently of any individual.

For Atlassian Cloud, service account OAuth 2LO requires a registered OAuth app in the [Atlassian developer console](https://developer.atlassian.com/console/myapps/). The app is configured with granular scopes, and the service exchanges its client ID and secret for a Bearer token that lasts about an hour.

---

## How the Token Exchange Works

```
Service → POST https://api.atlassian.com/oauth/token
          {
            "grant_type": "client_credentials",
            "client_id": "<CLIENT_ID>",
            "client_secret": "<CLIENT_SECRET>",
            "audience": "api.atlassian.com"
          }
       ← Bearer token (valid ~1 hour)

Service → GET https://api.atlassian.com/ex/jira/<cloudId>/rest/...
          Authorization: Bearer <token>
```

The `audience: api.atlassian.com` field is not optional. If you omit it, Atlassian's token endpoint accepts the request and returns a token — but that token is silently scoped to the wrong audience and will be rejected with 401 on every subsequent API call. This was a hard-to-diagnose issue; the fix is a single line in the token request body, and `ingest/oauth2lo.py` always includes it.

Tokens are cached in memory and reused until they expire or return a 401. On a 401, the cache is cleared and the token is refreshed once before failing. This means transient token expirations are handled silently with no impact on callers.

---

## The OAuth App

**App name:** `jsm-helpdesk-agent`  
**Type:** OAuth 2.0 (2LO) — Client Credentials  
**Configured in:** [developer.atlassian.com/console](https://developer.atlassian.com/console/myapps/)  
**Credentials stored in:** Bitwarden Secrets Manager (`ATLASSIAN_OAUTH_CLIENT_ID`, `ATLASSIAN_OAUTH_CLIENT_SECRET`)

This is a single app covering all operations — JSM, Jira, and Confluence — through 50 granular scopes. The full scope list is in [`deploy/oauth-scopes.md`](../deploy/oauth-scopes.md).

| Category | Count | What it covers |
|----------|-------|----------------|
| Jira read | 22 | Issues, comments, users, projects, JQL |
| JSM read | 13 | Requests, queues, SLAs, customers, KB |
| Confluence read | 6 | Pages, spaces, labels |
| Jira write | 3 | Comments, comment properties, watchers |
| JSM write | 9 | Request comments, status transitions, participants |
| Confluence write | 1 | Pages (article publish) |

**No delete scopes are granted.** This means the cleanup step that deletes trigger comments (`/ai-lookup`, `🤖`) after processing will silently fail with a 401. The comment stays on the ticket — it's cosmetic, nothing breaks. This is a known limitation; adding a delete scope would require a separate OAuth app with Jira REST v3 access.

---

## Why JSM servicedeskapi, Not Jira REST v3

This is the most important technical detail in the authentication model.

Jira REST API v3 scoped endpoints — the ones you'd find in standard Atlassian documentation — return **401 "scope does not match"** when called with a service account OAuth 2LO token. This happens even with the correct scopes granted to the OAuth app. It's a known Atlassian issue that affects service account credentials specifically; user-delegated OAuth tokens work fine on the same endpoints.

The JSM `servicedeskapi` endpoints work correctly with the same credentials:

| Operation | Endpoint | Works? |
|-----------|---------|--------|
| Fetch tickets | `GET /rest/servicedeskapi/servicedesk/{id}/queue/{qid}/issue` | ✅ |
| Fetch comments | `GET /rest/servicedeskapi/request/{key}/comment` | ✅ |
| Post internal comment | `POST /rest/servicedeskapi/request/{key}/comment` | ✅ |
| Delete comment | `DELETE /rest/api/3/issue/{key}/comment/{id}` | ❌ 401 |
| JQL search | `GET /rest/api/3/search/jql` | ❌ 401 |

The JSM queue endpoint returns the same full Jira issue data as REST v3 search, so switching required no changes to how the data is parsed. The practical consequence is that the auto-draft sweep (which uses JQL to find undrafted tickets) will also 401 — new tickets are caught via webhook, so this only affects the scheduled fallback sweep.

### A Note on Confluence Paths

When calling the Confluence REST API through the `api.atlassian.com/ex/confluence/{cloudId}` gateway, all paths must include `/wiki/`:

```
✅  {base}/wiki/rest/api/content/search
❌  {base}/rest/api/content/search
```

This applies everywhere Confluence is called: KB article search in `faq/lookup.py` and article publishing in `generation/publisher.py`.

---

## How to Rotate Credentials

OAuth app credentials should be rotated if they're suspected to be compromised or as a matter of policy. The process is:

1. **Generate new credentials** in [developer.atlassian.com/console](https://developer.atlassian.com/console/myapps/) → `jsm-helpdesk-agent` → OAuth credentials → Rotate secret
2. **Update Bitwarden Secrets Manager** — edit the `ATLASSIAN_OAUTH_CLIENT_ID` and `ATLASSIAN_OAUTH_CLIENT_SECRET` secrets in the `ai-helpdesk-agent` project
3. **The deploy pipeline picks them up automatically** — the next push to `main` triggers a GitHub Actions run that pulls the updated secrets from Bitwarden SM and applies them to the OpenShift deployment
4. **Force an immediate rollout** if you can't wait for the next code push:
   ```bash
   oc rollout restart deployment/ai-helpdesk-agent -n ants-engineering
   ```

The in-memory token cache expires on its own — no manual cache clear is needed. After the pod restarts, the service fetches a new token with the rotated credentials on its first API call.
