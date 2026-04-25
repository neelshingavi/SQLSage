---
title: Sqlsage Env
emoji: 🐠
colorFrom: blue
colorTo: blue
sdk: docker
pinned: false
license: mit
---

# SQLSage Environment

SQLSage is an OpenEnv-compatible RL environment for SQL query optimization via PostgreSQL execution-plan reading.

## Database Mode

This project is configured for **local PostgreSQL via Docker**.

Use:

- `POSTGRES_HOST=127.0.0.1`
- `POSTGRES_PORT=5433`
- `POSTGRES_USER=postgres`
- `POSTGRES_PASSWORD=sqlsage`
- `POSTGRES_DB=sqlsage`
- `SQLSAGE_TIMEOUT_MS=120000` (SF=1 TPC-H can exceed 5s)

Start DB:

```bash
docker compose up -d
```

## Endpoints (Meta OpenEnv HTTP API)

- `GET /health`
- `POST /reset` — body: `{}` or `{"seed": 42}` (optional `episode_id`)
- `POST /step` — body wraps the SQL action under `action` (OpenEnv `StepRequest`):

```json
{
  "action": {
    "action": "push_filter",
    "rewritten_query": "SELECT 1"
  }
}
```

- `GET /state`
- `GET /schema`, `GET /metadata`, WebSocket `/ws` for session-based `reset` / `step`

## Local Run

```bash
POSTGRES_HOST=127.0.0.1 POSTGRES_PORT=5433 uvicorn sqlsage.app:app --reload --port 8000
```

## Single-container image (Postgres + API)

The root `Dockerfile` starts PostgreSQL then `uvicorn` (for Hugging Face Spaces). Load TPC-H into the data directory separately or bake init SQL if needed.

## Tests

```bash
pip install -e '.[dev]'
pytest tests/ -v
```

Integration tests use `POSTGRES_*` (default host `127.0.0.1`, port `5433`).

## Validation

```bash
openenv validate
```
