# Manual Next Steps (Sections 4-7)

This repo now contains the coding implementation through section 7.  
Run these steps manually on your machine (no training is auto-run here).

## 1) Python environment and dependencies

1. Create and activate a virtual environment.
2. Install dependencies:
   - `openenv`
   - `psycopg2-binary`
   - `fastapi`
   - `uvicorn`
   - `pytest`
   - `datasets`, `trl`, `unsloth`, `wandb` (for training phase later)

## 2) PostgreSQL setup

Option A (single command):
- `docker run --name sqlsage-pg -e POSTGRES_PASSWORD=sqlsage -e POSTGRES_DB=sqlsage -p 5433:5432 -d postgres:16`

Option B (repo helper):
- `docker compose up -d`

Then verify:
- `POSTGRES_HOST=127.0.0.1 POSTGRES_PORT=5433 python scripts/check_db_connection.py`

## 3) Load TPC-H data

Follow the TPC-H loading sequence from the reference:
- build `tpch-dbgen`
- generate data
- apply `dss.ddl`
- copy `.tbl` files
- drop selected indexes for optimization opportunities

## 4) Verify code and tests

- Compile: `python -m compileall sqlsage`
- Unit tests: `pytest -q tests/test_reward.py tests/test_anti_cheat.py`
- Static smoke: `python scripts/smoke_env.py`

## 5) Run environment API locally

- `uvicorn sqlsage.app:app --reload --port 8000`
- Test endpoints:
  - `GET /health`
  - `POST /reset`
  - `POST /step`
  - `GET /state`

## 6) Push environment (manual)

- Login to HF
- `openenv push your-hf-username/sqlsage-env`
- verify remote `/reset` endpoint

## 7) Training preparation only (manual)

- Print reference config:
  - `python scripts/training_stub.py --print-config`
- Use `sqlsage/dataset.py` to build records from real DB observations.
- Run GRPO training only after GPU/runtime setup is complete.
