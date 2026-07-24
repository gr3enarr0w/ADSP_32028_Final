# ANTSE-291: Pod Memory and CPU Request Bump

## Context
ML stack additions (classifier, sentiment model, embedding model, FAISS index) need ~2-3 GB RAM. Current pod requests are 250m CPU / 512Mi which will OOMKill once models are loaded.

## Implementation

1. Modify `deploy/openshift/deployment.yaml`:
   - requests: `250m` → `1000m` CPU, `512Mi` → `3Gi` memory
   - limits: `1000m` → `2000m` CPU, `1Gi` → `4Gi` memory
2. Bump liveness probe `initialDelaySeconds` from 15 → 30 (model loading takes longer)

## Files
- Modify: `deploy/openshift/deployment.yaml`

## Verification
- `oc apply -f deploy/openshift/deployment.yaml`
- Pod starts successfully in ants-engineering namespace
- `/api/health` returns 200
- No OOMKills in `oc describe pod`
- Existing pipeline cycle completes without errors
