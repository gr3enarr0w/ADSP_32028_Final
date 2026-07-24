#!/usr/bin/env bash
# Run ai-helpdesk-agent locally using Podman (or Docker).
# Uses your .env file for secrets and mounts ./data for persistence.
set -euo pipefail

IMAGE="ai-helpdesk-agent:dev"
DATA_DIR="$(pwd)/data"
ENGINE="${CONTAINER_ENGINE:-podman}"

CMD="${1:-serve}"   # serve | ingest | classify | shell

mkdir -p "$DATA_DIR"

# Build if image doesn't exist or --build flag passed
if [[ "${2:-}" == "--build" ]] || ! $ENGINE image exists "$IMAGE" 2>/dev/null; then
  echo "Building $IMAGE..."
  $ENGINE build -f Containerfile -t "$IMAGE" .
fi

BASE_ARGS=(
  --rm
  --env-file .env
  -e DATA_DIR=/app/data
  -v "$DATA_DIR":/app/data:Z
)

# Mount service account JSON if present alongside .env
if [[ -f service_account.json ]]; then
  BASE_ARGS+=(-v "$(pwd)/service_account.json":/app/secrets/service_account.json:Z,ro)
  BASE_ARGS+=(-e GOOGLE_SERVICE_ACCOUNT_JSON=/app/secrets/service_account.json)
fi

case "$CMD" in
  serve)
    echo "Starting API server at http://localhost:8080 ..."
    $ENGINE run "${BASE_ARGS[@]}" -p 8080:8080 "$IMAGE"
    ;;
  ingest)
    echo "Running ingest pipeline..."
    $ENGINE run "${BASE_ARGS[@]}" "$IMAGE" python faq_service.py ingest
    ;;
  classify)
    echo "Running classifier..."
    $ENGINE run "${BASE_ARGS[@]}" "$IMAGE" python faq_service.py classify
    ;;
  shell)
    echo "Opening shell..."
    $ENGINE run -it "${BASE_ARGS[@]}" "$IMAGE" bash
    ;;
  *)
    echo "Usage: $0 [serve|ingest|classify|shell] [--build]"
    exit 1
    ;;
esac
