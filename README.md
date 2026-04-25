# SQLSage Environment

SQLSage is an OpenEnv-compatible RL environment for SQL query optimization via PostgreSQL execution-plan reading.

## Endpoints

- `GET /health`
- `POST /reset`
- `POST /step`
- `GET /state`

## Local Run

```bash
uvicorn sqlsage.app:app --reload --port 8000
```

## Validation

```bash
openenv validate
```
