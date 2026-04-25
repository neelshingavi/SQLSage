"""Main SQLSage environment implementation."""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass, field
from typing import Any

import psycopg2
from psycopg2 import errors
from psycopg2.extensions import connection

from .anti_cheat import execute_read_only, validate_read_only_sql
from .explain_parser import get_explain_dict
from .reward import compute_reward
from .tasks import ALL_TASKS, Task

VALID_ACTIONS = {
    "rewrite_join",
    "add_cte",
    "push_filter",
    "reorder_joins",
    "suggest_index",
    "limit_early",
    "revert",
}


@dataclass
class Observation:
    original_query: str
    explain_plan: dict[str, Any]
    execution_ms: float
    result_hash: str
    result_row_count: int
    schema_context: str
    previous_rewrites: list[str] = field(default_factory=list)
    step_count: int = 0
    task_level: int = 1


class SQLSageEnv:
    """Gym-like environment with reset(), step(), and state()."""

    def __init__(
        self,
        db_host: str | None = None,
        db_port: int | None = None,
        db_user: str | None = None,
        db_password: str | None = None,
        db_name: str | None = None,
        timeout_ms: int | None = None,
        max_steps: int = 5,
        tasks: list[Task] | None = None,
    ) -> None:
        resolved_host = db_host or os.getenv("POSTGRES_HOST", "localhost")
        resolved_port = int(db_port or os.getenv("POSTGRES_PORT", "5432"))
        resolved_user = db_user or os.getenv("POSTGRES_USER", "postgres")
        resolved_password = db_password or os.getenv("POSTGRES_PASSWORD", "sqlsage")
        resolved_db = db_name or os.getenv("POSTGRES_DB", "sqlsage")
        # SF=1 TPC-H queries can exceed 5s on modest hardware; override via SQLSAGE_TIMEOUT_MS.
        resolved_timeout_ms = int(timeout_ms or os.getenv("SQLSAGE_TIMEOUT_MS", "120000"))

        self.conn: connection = psycopg2.connect(
            host=resolved_host,
            port=resolved_port,
            user=resolved_user,
            password=resolved_password,
            dbname=resolved_db,
        )
        self.conn.autocommit = True
        self.timeout_ms = resolved_timeout_ms
        self.max_steps = max_steps
        self.tasks = tasks or ALL_TASKS
        self.state_obj: Observation | None = None
        self.current_task: Task | None = None
        self.best_rewrite: str | None = None
        self.best_ms: float | None = None

    def close(self) -> None:
        self.conn.close()

    def _fetch_schema_context(self) -> str:
        query = """
SELECT
  c.relname AS table_name,
  a.attname AS column_name,
  t.typname AS data_type
FROM pg_class c
JOIN pg_attribute a ON a.attrelid = c.oid
JOIN pg_type t ON a.atttypid = t.oid
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE c.relkind = 'r'
  AND a.attnum > 0
  AND NOT a.attisdropped
  AND n.nspname = 'public'
ORDER BY table_name, column_name;
"""
        with self.conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
        return "\n".join(f"{table}.{column}:{dtype}" for table, column, dtype in rows)

    def _validate_sql(self, query: str) -> bool:
        if not validate_read_only_sql(query):
            return False
        try:
            with self.conn.cursor() as cur:
                cur.execute("EXPLAIN " + query)
            return True
        except Exception:
            return False

    def execute_and_measure(self, query: str) -> tuple[float, str, int]:
        try:
            return execute_read_only(self.conn, query, timeout_ms=self.timeout_ms)
        except errors.QueryCanceled as exc:
            raise TimeoutError("query_timeout") from exc

    def reset(self, seed: int | None = None) -> Observation:
        if seed is not None:
            random.seed(seed)
        self.current_task = random.choice(self.tasks)
        base_query = self.current_task.query

        baseline_ms, baseline_hash, baseline_rows = self.execute_and_measure(base_query)
        baseline_plan = get_explain_dict(self.conn, base_query)
        schema_context = self._fetch_schema_context()

        self.state_obj = Observation(
            original_query=base_query,
            explain_plan=baseline_plan,
            execution_ms=baseline_ms,
            result_hash=baseline_hash,
            result_row_count=baseline_rows,
            schema_context=schema_context,
            previous_rewrites=[],
            step_count=0,
            task_level=self.current_task.level,
        )
        self.best_rewrite = base_query
        self.best_ms = baseline_ms
        return self.state_obj

    def state(self) -> Observation:
        if self.state_obj is None:
            raise RuntimeError("call reset() before state()")
        return self.state_obj

    def step(self, action: str, rewritten_query: str) -> tuple[Observation, float, bool, dict[str, Any]]:
        if self.state_obj is None:
            raise RuntimeError("call reset() before step()")

        if action not in VALID_ACTIONS:
            return self.state_obj, -10.0, False, {"error": "invalid_action"}

        if action == "revert":
            rewritten_query = self.best_rewrite or self.state_obj.original_query

        # 1. Validate the rewritten query is parseable SQL
        if not self._validate_sql(rewritten_query):
            return self.state_obj, -15.0, False, {"error": "syntax_error"}

        # 2. Run the new query and measure time
        try:
            new_ms, new_hash, new_rows = self.execute_and_measure(rewritten_query)
        except TimeoutError:
            return self.state_obj, -12.0, False, {"error": "timeout"}

        # 3. Anti-cheat: verify result set unchanged
        if new_hash != self.state_obj.result_hash or new_rows != self.state_obj.result_row_count:
            return self.state_obj, -20.0, False, {"error": "result_changed"}

        # 4. Get new EXPLAIN plan
        new_plan = get_explain_dict(self.conn, rewritten_query)

        # 5. Compute reward
        reward = compute_reward(self.state_obj.execution_ms, new_ms, new_plan, self.state_obj.explain_plan)

        # 6. Update state
        if self.best_ms is None or new_ms < self.best_ms:
            self.best_ms = new_ms
            self.best_rewrite = rewritten_query

        self.state_obj.execution_ms = new_ms
        self.state_obj.explain_plan = new_plan
        self.state_obj.previous_rewrites.append(rewritten_query)
        self.state_obj.step_count += 1

        done = bool(
            (self.current_task is not None and new_ms < self.current_task.target_ms)
            or (self.state_obj.step_count >= self.max_steps)
        )
        return self.state_obj, reward, done, {}

    def state_dict(self) -> dict[str, Any]:
        return asdict(self.state())
