<!-- Mirrored from Confluence: https://<YOUR_DOMAIN>.atlassian.net/wiki/spaces/~ceverson/pages/402714901 -->
<!-- Last synced: 2026-05-29 -->

# Deployment: Imperative oc CLI via GitHub Actions

## How Deployments Work

Every push to `main` triggers `.github/workflows/build.yml`:

1. Builds the container image from `Containerfile`
2. Pushes to `ghcr.io/agile-tech-sol/ai-helpdesk-agent:latest` and `:<git-sha>`
3. Retrieves runtime secrets from Bitwarden Secrets Manager via `BW_ACCESS_TOKEN`
4. Logs into OpenShift using `OPENSHIFT_SERVER_URL` and `OPENSHIFT_TOKEN` GitHub Actions secrets
5. Applies `deploy/openshift/configmap.yaml` and upserts `ai-helpdesk-agent-secrets`
6. Runs `oc set image` to roll out the new SHA, then waits for `oc rollout status`

Namespace: `jira-messaging--runtime-ext`

## GitHub Actions Secrets Required

| Secret | Description |
|---|---|
| `OPENSHIFT_SERVER_URL` | OCP API endpoint — format `https://api.<cluster>:6443`. Run `oc whoami --show-server` on the live cluster. |
| `OPENSHIFT_TOKEN` | Service account token for `jira-messaging--runtime-ext` |
| `BW_ACCESS_TOKEN` | Bitwarden SM machine account token |

## Historical Note: ArgoCD

An OpenShift GitOps (ArgoCD) instance was provisioned in `jira-messaging--runtime-ext` and configured to watch `deploy/openshift/` on `main`. Auto-sync was never enabled because the cluster egress firewall blocked outbound HTTPS to `github.com`. The egress firewall was later unblocked, but the team continued with the imperative GitHub Actions pipeline rather than activating ArgoCD. There are no ArgoCD `Application` manifests or `argocd` CLI calls in this repository.
