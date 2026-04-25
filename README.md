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

## Endpoints

- `GET /health`
- `POST /reset`
- `POST /step`
- `GET /state`

## Local Run

```bash
POSTGRES_HOST=127.0.0.1 POSTGRES_PORT=5433 uvicorn sqlsage.app:app --reload --port 8000
```

## Validation

```bash
openenv validate
```
