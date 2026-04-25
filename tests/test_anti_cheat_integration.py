"""Anti-cheat checks against a live PostgreSQL instance (Person 1 hours 8–14)."""

from __future__ import annotations

import os

import pytest

from sqlsage.anti_cheat import execute_read_only, validate_read_only_sql
from sqlsage.env import SQLSageEnv
from sqlsage.tasks import tasks_for_levels
from sqlsage.tasks.level1 import LEVEL1_TASKS


def _sqlsage_env_connect_kwargs() -> dict[str, object]:
    """Match tests/conftest.py defaults so SQLSageEnv uses the same DB as postgres_conn."""
    return {
        "db_host": os.getenv("POSTGRES_HOST", "127.0.0.1"),
        "db_port": int(os.getenv("POSTGRES_PORT", "5433")),
        "db_user": os.getenv("POSTGRES_USER", "postgres"),
        "db_password": os.getenv("POSTGRES_PASSWORD", "sqlsage"),
        "db_name": os.getenv("POSTGRES_DB", "sqlsage"),
    }


def test_execute_read_only_empty_result_differs_from_baseline(postgres_conn):
    _, h_full, n_full = execute_read_only(postgres_conn, "SELECT 1 AS x", timeout_ms=5000)
    _, h_empty, n_empty = execute_read_only(postgres_conn, "SELECT 1 AS x WHERE false", timeout_ms=5000)
    assert n_full >= 1
    assert n_empty == 0
    assert h_full != h_empty


def test_execute_read_only_rejects_ddl(postgres_conn):
    with pytest.raises(ValueError, match="blocked_or_non_read_only_sql"):
        execute_read_only(postgres_conn, "CREATE TABLE sqlsage_cheat (id int)", timeout_ms=5000)


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM lineitem WHERE false",
        "INSERT INTO lineitem SELECT * FROM lineitem WHERE false",
        "TRUNCATE lineitem",
        "VACUUM lineitem",
    ],
)
def test_validate_read_only_sql_blocks_write_paths(sql):
    assert not validate_read_only_sql(sql)


def test_env_step_penalizes_empty_wrapped_query(postgres_conn):
    """Rewriting to an empty result set must not pass hash verification (reference §4.2)."""
    _ = postgres_conn  # fixture ensures DB is up; env opens its own connection from env vars
    env = SQLSageEnv(tasks=[LEVEL1_TASKS[1]], max_steps=5, **_sqlsage_env_connect_kwargs())
    try:
        env.reset(seed=0)
        base = env.state().original_query.strip().rstrip(";")
        wrapped = f"SELECT * FROM (\n{base}\n) AS _sqlsage_sub WHERE 1 = 0"
        _obs, reward, _done, info = env.step("push_filter", wrapped)
        assert reward == -20.0
        assert info.get("error") == "result_changed"
    finally:
        env.close()


def test_env_level2_only_task_set_runs_reset(postgres_conn):
    _ = postgres_conn
    level2 = tasks_for_levels(2)
    assert len(level2) == 4
    env = SQLSageEnv(tasks=level2, max_steps=3, **_sqlsage_env_connect_kwargs())
    try:
        obs = env.reset(seed=1)
        assert obs.task_level == 2
        assert "JOIN" in obs.original_query.upper() or "join" in obs.original_query
    finally:
        env.close()


def test_env_syntax_error_for_ddl_step(postgres_conn):
    _ = postgres_conn
    env = SQLSageEnv(tasks=[LEVEL1_TASKS[0]], max_steps=3, **_sqlsage_env_connect_kwargs())
    try:
        env.reset(seed=0)
        _obs, reward, _done, info = env.step("push_filter", "CREATE INDEX bad_idx ON lineitem (l_orderkey)")
        assert reward == -15.0
        assert info.get("error") == "syntax_error"
    finally:
        env.close()
