#!/usr/bin/env bash
# Non-interactive deploy to Hugging Face Spaces via OpenEnv.
# Requires a write token: https://huggingface.co/settings/tokens
#
# Usage:
#   export HF_TOKEN=hf_...
#   ./scripts/push_hf_space.sh
#
# Or add HF_TOKEN to .env, then: set -a && source .env && set +a && ./scripts/push_hf_space.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

if [[ -z "${HF_TOKEN:-}" ]] && [[ -n "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  export HF_TOKEN="$HUGGINGFACE_HUB_TOKEN"
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "Missing HF_TOKEN. Export it or add it to .env (see .env.example)." >&2
  echo "Create a token: https://huggingface.co/settings/tokens" >&2
  exit 1
fi

REPO_ID="${HF_SPACE_REPO_ID:-Adity00/sqlsage-env}"
exec openenv push -r "$REPO_ID" "$ROOT"
