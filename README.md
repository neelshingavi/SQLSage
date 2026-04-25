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

### Person 1 — hours 8–14 (reference §8)

After TPC-H SF=1 is loaded, optionally drop helper indexes so the curriculum matches the reference (more Seq Scan headroom):

```bash
psql "host=127.0.0.1 port=5433 user=postgres dbname=sqlsage password=sqlsage" -f sql/scripts/drop_curriculum_indexes.sql
```

- **Level 2 tasks** are `Q3`, `Q5`, `Q10`, `Q12` style queries in `sqlsage/tasks/level2.py`. Train or stress **Level 2 only** with `SQLSageEnv(tasks=sqlsage.tasks.tasks_for_levels(2))` or `--levels 2` below.
- **Stress run** (sequential `reset` / optional identity `step`, no GRPO):

```bash
POSTGRES_HOST=127.0.0.1 POSTGRES_PORT=5433 python scripts/stress_env.py --episodes 50 --levels 2 --identity-step
```

OpenEnv HTTP uses a **single shared DB session**; the bridge serializes `reset` / `step` / `state` with a lock so multi-threaded Uvicorn does not interleave psycopg2 calls.

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

The root `Dockerfile` starts PostgreSQL then `uvicorn` (for Hugging Face Spaces). The entrypoint listens on **`$PORT`** (Hugging Face defaults this to **7860**). For a local `docker run` without `PORT`, the same default applies—set `-e PORT=8000 -p 8000:8000` if you want port 8000. Load TPC-H into the data directory separately or bake init SQL if needed.

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

## Training (Person 3)

Colab GRPO checklist, wandb rollout script, Hub upload, and baseline table: **`docs/PERSON3_PHASE8_MANUAL.md`**. Colab skeleton: **`notebooks/sqlsage_grpo_colab.ipynb`**. Quick smoke: `pip install -e '.[training]'` then `python scripts/rollout_wandb.py --episodes 5`.
