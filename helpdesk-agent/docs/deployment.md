<!-- Mirrored from Confluence: https://<YOUR_DOMAIN>.atlassian.net/wiki/spaces/~ceverson/pages/395837969 -->
<!-- Last synced: 2026-05-29 -->

# Deployment

## How Deployments Work

### Image Build and Deploy (Automated via GitHub Actions)

Every push to `main` triggers the pipeline in `.github/workflows/build.yml`:
1. Builds from `Containerfile`
2. Pushes to `ghcr.io/agile-tech-sol/ai-helpdesk-agent:latest` and `:<git-sha>`
3. Retrieves secrets from Bitwarden SM
4. Logs into OpenShift via `OPENSHIFT_SERVER_URL` and `OPENSHIFT_TOKEN` GitHub Actions secrets
5. Applies ConfigMap and upserts the app Secret
6. Runs `oc set image` + `oc rollout status` to deploy the new image

Deploy is **imperative oc CLI** — there is no ArgoCD in this repo. See [GitOps and ArgoCD Setup](gitops-argocd.md) for historical context.

---

## Manual Deployment (VPN Required)

All `oc` commands require VPN access and a valid token for the current OCP cluster.

> **ACTION REQUIRED:** Replace `<OCP_API_URL>` below with the current live cluster URL.
> Run `oc whoami --show-server` while logged into the cluster to get it.
> Format: `https://api.<cluster-domain>:6443`

### One-Time Setup

**1. Log in to OpenShift**

Get a token from the console → your username → Copy Login Command.
```bash
oc login --token=<token> --server=<OCP_API_URL>
oc project jira-messaging--runtime-ext
```

**2. Create the image pull secret for ghcr.io**

Requires a **classic** GitHub PAT (not fine-grained) with `read:packages` scope.
```bash
oc create secret docker-registry ghcr-pull-secret \
  --docker-server=ghcr.io \
  --docker-username=<github-username> \
  --docker-password=<classic-pat> \
  -n jira-messaging--runtime-ext

# CRITICAL: must link to service account or pods cannot pull
oc secrets link default ghcr-pull-secret --for=pull -n jira-messaging--runtime-ext
```

**3. Create the Google service account JSON secret**

Get `service_account.json` from Bitwarden (search: "ai-helpdesk-agent service account").
```bash
oc create secret generic ai-helpdesk-agent-sa-json \
  --from-file=service_account.json=./service_account.json \
  -n jira-messaging--runtime-ext
```

**4. Deploy PostgreSQL**

Generate a strong password (`openssl rand -base64 24`). Store it in Bitwarden.
```bash
oc create secret generic postgres-secret \
  --from-literal=POSTGRES_PASSWORD=<generated-password> \
  -n jira-messaging--runtime-ext

oc apply -f deploy/openshift/postgres-service.yaml -n jira-messaging--runtime-ext
oc apply -f deploy/openshift/postgres-statefulset.yaml -n jira-messaging--runtime-ext
oc rollout status statefulset/postgres -n jira-messaging--runtime-ext --timeout=120s
```

**5. Set DATABASE_URL in the app secret**
```bash
oc patch secret ai-helpdesk-agent-secrets -n jira-messaging--runtime-ext \
  --type='json' \
  -p='[{"op":"add","path":"/data/DATABASE_URL","value":"<base64-encoded-connection-string>"}]'
# Encode: echo -n "postgresql://helpdesk:<password>@postgres:5432/helpdesk" | base64
```

**6. Apply remaining manifests**
```bash
oc apply -f deploy/openshift/pvc.yaml -n jira-messaging--runtime-ext
oc apply -f deploy/openshift/configmap.yaml -n jira-messaging--runtime-ext
oc apply -f deploy/openshift/service-route.yaml -n jira-messaging--runtime-ext
oc apply -f deploy/openshift/deployment.yaml -n jira-messaging--runtime-ext
oc apply -f deploy/openshift/cronjob.yaml -n jira-messaging--runtime-ext
oc apply -f deploy/openshift/retrain-cronjob.yaml -n jira-messaging--runtime-ext
```

**7. Verify**
```bash
oc get pods -n jira-messaging--runtime-ext -l app=ai-helpdesk-agent
# Replace <ROUTE_HOST> with the actual route hostname from: oc get route -n jira-messaging--runtime-ext
curl https://<ROUTE_HOST>/api/health
```

---

### Rolling Update (after code changes)

CI builds and pushes automatically on every push to `main`. After the build step succeeds, the deploy step handles the rollout automatically via `oc set image`. For a manual rollout:
```bash
oc login --token=<token> --server=<OCP_API_URL>
oc rollout restart deployment/ai-helpdesk-agent -n jira-messaging--runtime-ext
oc rollout status deployment/ai-helpdesk-agent -n jira-messaging--runtime-ext --timeout=120s
```

---

## OpenShift SCC Notes (restricted-v2)

This namespace enforces `restricted-v2`. Key rules:
- **Never** set `runAsUser` or `fsGroup` in pod-level `securityContext` — OpenShift assigns UIDs from range `1002740000-1002749999` automatically
- **Always** set in every container's `securityContext`:
  ```yaml
  securityContext:
    allowPrivilegeEscalation: false
    capabilities:
      drop: [ALL]
    seccompProfile:
      type: RuntimeDefault
  ```
- Violations → `FailedCreate` events with "unable to validate against any security context constraint"

---

## Secrets Reference

All secrets are in Bitwarden under the "ai-helpdesk-agent" collection.

| Secret in OpenShift | Where to find |
|---|---|
| `ai-helpdesk-agent-secrets` (OAuth credentials) | Bitwarden → ai-helpdesk-agent OAuth apps |
| `ai-helpdesk-agent-sa-json` | Bitwarden → ai-helpdesk-agent service account JSON |
| `ghcr-pull-secret` | Classic GitHub PAT with `read:packages` scope |
| `postgres-secret` | Bitwarden → ai-helpdesk-agent postgres password |
| `DATABASE_URL` (in ai-helpdesk-agent-secrets) | `postgresql://helpdesk:<postgres-password>@postgres:5432/helpdesk` |

---

## Secrets Management (CI/CD)

All sensitive values are stored in **Bitwarden Secrets Manager**, project `ai-helpdesk-agent`. The machine account (`ai-helpdesk-agent-ci`) authenticates via `BW_ACCESS_TOKEN` stored in GitHub.

The 7 secrets currently in Bitwarden SM: `JSM_CLOUD_URL`, `ATLASSIAN_OAUTH_CLIENT_ID`, `ATLASSIAN_OAUTH_CLIENT_SECRET`, `FAQ_API_TOKEN`, `GOOGLE_SHEET_ID`, `JIRA_WEBHOOK_SECRET`, `GOOGLE_SERVICE_ACCOUNT_JSON`.

The two OpenShift secrets (`OPENSHIFT_SERVER_URL`, `OPENSHIFT_TOKEN`) are stored directly as GitHub Actions secrets, not in Bitwarden SM.

---

## Local Development

```bash
cp .env.example .env  # fill in your credentials
python main.py        # or: uvicorn main:app --reload --port 8080
```

The `.env` file and `service_account.json` are in `.gitignore` — never commit them.
