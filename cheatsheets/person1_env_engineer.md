# SQLSage — Person 1: Environment Engineer
**Deadline: Hour 8 = HF Space live**

## Setup Commands
| Command | What it does |
|---------|-------------|
| `openenv init sqlsage-env` | Creates the scaffold (run from parent of the repo dir) |
| `docker compose up -d` | Start Postgres (see `docker-compose.yml`; default **port 5433** in README) |
| `uvicorn sqlsage.app:app --reload --port 8000` | Runs environment locally (set `POSTGRES_HOST` / `POSTGRES_PORT`) |
| `openenv push your-username/sqlsage-env` | Deploys to HF Spaces |
| `docker run --name sqlsage-pg -e POSTGRES_PASSWORD=sqlsage -e POSTGRES_DB=sqlsage -p 5432:5432 -d postgres:16` | Single-container Postgres 16 (alt. to compose) |
| `curl http://localhost:8000/reset` | Tests reset endpoint (local) |
| `curl -sS "$SQLSAGE_ENV_URL/reset"` | Tests deployed Space (set URL from `.env`) |
| `psql "host=127.0.0.1 port=5433 user=postgres dbname=sqlsage password=sqlsage" -c 'SELECT version();'` | Verify DB (project default port **5433**) |
| `openenv validate` | OpenEnv project validation |
| `python -m sqlsage.run p1-serve` | **Alias** → uvicorn (from `sqlsage.run`) |
| `python -m sqlsage.run p1-test` | **Alias** → curl local `/reset` \| `json.tool` |
| `python -m sqlsage.run p1-push` / `p1-deploy` | **Alias** → push + optional HF curl |

## TPC-H Data Load
| Command | What it does |
|---------|-------------|
| `git clone https://github.com/electrum/tpch-dbgen.git && cd tpch-dbgen && make` | Clone TPC-H tools |
| `./dbgen -s 1` | Generate SF=1 data (~1GB, ~2 min) |
| `./dbgen -s 0.1` | Generate SF=0.1 (fast, for training) |
| `psql -U postgres -d sqlsage -f dss.ddl` | Create schema (paths per your load process) |
| TPC-H bootstrap in Docker / HF image | `sql/bootstrap_tpch_schema.sql` auto-runs in Space; load data separately for real rewards |

## Drop Indexes (Creates Optimization Headroom)
```bash
psql "host=127.0.0.1 port=5433 user=postgres dbname=sqlsage password=sqlsage" \
  -f sql/scripts/drop_curriculum_indexes.sql
psql -U postgres -d sqlsage -c "DROP INDEX IF EXISTS idx_orders_status;"
psql -U postgres -d sqlsage -c "DROP INDEX IF EXISTS idx_lineitem_shipdate;"
```

## Stress & Levels (phases: curriculum / env hardening)
| Command | What it does |
|---------|-------------|
| `POSTGRES_HOST=127.0.0.1 POSTGRES_PORT=5433 python scripts/stress_env.py --episodes 50 --levels 2 --identity-step` | Stress `reset` / optional identity `step` (no GRPO) |
| `SQLSageEnv(tasks=tasks_for_levels(2))` | Level 2 only (see `sqlsage.tasks`) |
| `GET /health` | Health check (OpenEnv HTTP) |
| `GET /state`, `GET /schema`, `GET /metadata` | Session / schema metadata |

## Troubleshooting
| Symptom | Fix |
|---------|-----|
| `curl /reset` returns 500 | `docker ps` / `docker compose ps` — DB + API up; check logs |
| HF Space build failing | `Dockerfile` + entrypoint: Postgres as service, `uvicorn` on `$PORT` (7860 on HF) |
| TPC-H load too long | Use SF=0.1 for dev; SF=1 for final demo; raise `SQLSAGE_TIMEOUT_MS` if needed |
| Intermittent 500 under load | OpenEnv uses a **lock** for reset/step; avoid parallel `step` to same process |

## Hour Gates
- Hour 1: psql connects ✓
- Hour 4: `curl /reset` returns JSON ✓
- Hour 6: `step()` returns `{reward, done, state}` ✓
- Hour 8: **HF Space live** — **SHARE URL WITH TEAM** (`SQLSAGE_ENV_URL` in Colab) ✓
