#!/usr/bin/env bash
# Fast deploy to a Hugging Face Space via Git (small delta vs openenv push).
# Usage:
#   ./scripts/push_hf_fast.sh
#   HF_TOKEN=hf_... ./scripts/push_hf_fast.sh   # headless; token needs write on the Space
#
# Or use interactive auth once: `huggingface-cli login` then `make push-hf` from repo root.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
REMOTE="${HF_SPACE_GIT:-https://huggingface.co/spaces/Adity00/sqlsage-env}"
BRANCH="${HF_SPACE_BRANCH:-main}"

if [ -n "${HF_TOKEN:-}" ]; then
  # Token as HTTPS password; username is ignored for HF Git over HTTPS.
  path="${REMOTE#*huggingface.co}"
  exec git push "https://huggingface:${HF_TOKEN}@huggingface.co${path}" "$BRANCH"
fi

if git remote get-url hf-space &>/dev/null; then
  exec git push hf-space "$BRANCH"
fi

echo "Add remote: git remote add hf-space $REMOTE" >&2
echo "Or: HF_TOKEN=hf_... $0" >&2
exit 1
