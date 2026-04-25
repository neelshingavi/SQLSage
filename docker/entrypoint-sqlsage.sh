#!/usr/bin/env bash
set -euo pipefail

# Start PostgreSQL using the official image entrypoint (initializes data dir on first boot).
# Official image installs this helper on PATH.
docker-entrypoint.sh postgres &
until pg_isready -h 127.0.0.1 -U "${POSTGRES_USER:-postgres}" -d "${POSTGRES_DB:-sqlsage}" 2>/dev/null; do
  sleep 0.4
done

export POSTGRES_HOST="${POSTGRES_HOST:-127.0.0.1}"
export POSTGRES_PORT="${POSTGRES_PORT:-5432}"

cd /app
# Hugging Face Spaces set PORT (default 7860). Listening on 8000 when PORT is unset often causes
# external HTTPS to hang/time out while logs still show a healthy Uvicorn process.
exec uvicorn sqlsage.app:app --host 0.0.0.0 --port "${PORT:-7860}"
