"""OpenEnv environment for SQL query optimization with PostgreSQL."""

import hashlib
import json
import os
import random
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
import psycopg2.extras
try:
    from openenv import Environment
except ImportError:  # pragma: no cover - compatibility with older openenv exports
    class Environment:  # type: ignore[too-many-ancestors]
        """Fallback base when openenv.Environment is unavailable."""

        pass

from sqlsage.explain_parser import diagnose_bottleneck, get_explain_dict, get_result_hash
from sqlsage.reward import compute_reward


@dataclass
class Observation:
    """Serializable state returned to the policy at each step."""

    original_query: str
    explain_plan: dict
    execution_ms: float
    result_hash: str
    schema_context: str
    previous_rewrites: List[str] = field(default_factory=list)
    previous_rewards: List[float] = field(default_factory=list)
    step_count: int = 0
    task_level: int = 1
    bottleneck_diagnosis: str = "UNKNOWN"
    suggested_actions: List[str] = field(default_factory=list)
    cost_hotspot: str = ""
    rows_scanned_ratio: float = 0.0
    target_ms: float = 500.0

    def to_dict(self) -> dict:
        """Return all fields as a JSON-serializable dictionary."""
        return {
            "original_query": self.original_query,
            "explain_plan": self.explain_plan,
            "execution_ms": float(self.execution_ms),
            "result_hash": self.result_hash,
            "schema_context": self.schema_context,
            "previous_rewrites": list(self.previous_rewrites),
            "previous_rewards": [float(v) for v in self.previous_rewards],
            "step_count": int(self.step_count),
            "task_level": int(self.task_level),
            "bottleneck_diagnosis": self.bottleneck_diagnosis,
            "suggested_actions": list(self.suggested_actions),
            "cost_hotspot": self.cost_hotspot,
            "rows_scanned_ratio": float(self.rows_scanned_ratio),
            "target_ms": float(self.target_ms),
        }


VALID_ACTIONS = {
    "rewrite_join",
    "add_cte",
    "push_filter",
    "reorder_joins",
    "suggest_index",
    "limit_early",
    "revert",
}

DDL_KEYWORDS = {
    "CREATE",
    "DROP",
    "ALTER",
    "TRUNCATE",
    "INSERT",
    "UPDATE",
    "DELETE",
    "GRANT",
    "REVOKE",
    "VACUUM",
    "ANALYZE",
}

SCHEMA_SUMMARY = """
TPC-H Schema:
- lineitem(orderkey, partkey, suppkey, linenumber, quantity,
           extendedprice, discount, tax, returnflag, linestatus,
           shipdate, commitdate, receiptdate, shipinstruct,
           shipmode, comment)  ~6M rows
- orders(orderkey, custkey, orderstatus, totalprice, orderdate,
         orderpriority, clerk, shippriority, comment)  ~1.5M rows
- customer(custkey, name, address, nationkey, phone, acctbal,
           mktsegment, comment)  ~150K rows
- part(partkey, name, mfgr, brand, type, size, container,
       retailprice, comment)  ~200K rows
- supplier(suppkey, name, address, nationkey, phone, acctbal,
           comment)  ~10K rows
- partsupp(partkey, suppkey, availqty, supplycost, comment)  ~800K rows
- nation(nationkey, name, regionkey, comment)  25 rows
- region(regionkey, name, comment)  5 rows
"""

TABLE_SIZES = {
    "lineitem": 6001215,
    "orders": 1500000,
    "customer": 150000,
    "part": 200000,
    "supplier": 10000,
    "partsupp": 800000,
    "nation": 25,
    "region": 5,
}


class SQLSageEnv(Environment):
    """OpenEnv-compatible environment for SQL query optimization."""

    def __init__(
        self,
        db_host: Optional[str] = None,
        db_port: Optional[int] = None,
        db_user: Optional[str] = None,
        db_password: Optional[str] = None,
        db_name: Optional[str] = None,
        tasks: Optional[List[Any]] = None,
        max_steps: int = 5,
        **_: Any,
    ):
        self._db_host = db_host
        self._db_port = db_port
        self._db_user = db_user
        self._db_password = db_password
        self._db_name = db_name
        self.conn = None
        self._state: Optional[Observation] = None
        self.target_ms: float = 500.0
        self.best_query: str = ""
        self.best_ms: float = float("inf")
        self.max_steps: int = int(max_steps)
        self.timeout_ms: int = int(os.environ.get("SQLSAGE_TIMEOUT_MS", "120000"))
        self._connect()
        self._load_tasks()
        if tasks:
            self.all_tasks = list(tasks)

    def _connect(self):
        """Connect to PostgreSQL. Retry 3 times with 2s delay."""
        host = self._db_host or os.environ.get("PG_HOST") or os.environ.get("POSTGRES_HOST", "localhost")
        port = int(self._db_port or os.environ.get("PG_PORT") or os.environ.get("POSTGRES_PORT", 5432))
        user = self._db_user or os.environ.get("PG_USER") or os.environ.get("POSTGRES_USER", "postgres")
        password = (
            self._db_password or os.environ.get("PG_PASSWORD") or os.environ.get("POSTGRES_PASSWORD", "sqlsage")
        )
        dbname = self._db_name or os.environ.get("PG_DB") or os.environ.get("POSTGRES_DB", "sqlsage")

        for attempt in range(3):
            try:
                self.conn = psycopg2.connect(
                    host=host,
                    port=port,
                    user=user,
                    password=password,
                    dbname=dbname,
                    connect_timeout=10,
                )
                self.conn.autocommit = False
                return
            except psycopg2.OperationalError as error:
                if attempt == 2:
                    raise RuntimeError(
                        f"Cannot connect to PostgreSQL after 3 attempts: {error}"
                    )
                time.sleep(2)

    def _load_tasks(self):
        """Load all task levels. Import here to avoid circular imports."""
        from sqlsage.tasks.level1 import LEVEL1_TASKS
        from sqlsage.tasks.level2 import LEVEL2_TASKS
        from sqlsage.tasks.level3 import LEVEL3_TASKS

        self.all_tasks = LEVEL1_TASKS + LEVEL2_TASKS + LEVEL3_TASKS

    def close(self):
        """Close PostgreSQL connection."""
        if self.conn is not None:
            self.conn.close()

    def _get_task(self, level: int, seed: Optional[int] = None):
        """Return a task dict for the given level."""
        import random

        tasks = [task for task in self.all_tasks if task.level == level]
        if not tasks:
            tasks = self.all_tasks
        rng = random.Random(seed)
        return rng.choice(tasks)

    def validate_sql(self, query: str) -> bool:
        """
        Validate that query is safe and parseable.

        Returns False if:
        - query is empty or not a string
        - query contains DDL keywords (case-insensitive)
        - PostgreSQL cannot parse it (EXPLAIN fails)
        """
        if not query or not isinstance(query, str):
            return False
        query_upper = query.upper()

        import re

        for keyword in DDL_KEYWORDS:
            if re.search(r"\b" + keyword + r"\b", query_upper):
                return False
        try:
            cur = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(f"EXPLAIN {query}")
            self.conn.rollback()
            cur.close()
            return True
        except psycopg2.Error:
            self.conn.rollback()
            return False

    def execute_and_measure(self, query: str, timeout_s: int = 5) -> Tuple[float, str, int]:
        """
        Execute query in READ ONLY transaction.

        Returns (execution_ms, result_hash, row_count).
        Raises TimeoutError if execution exceeds timeout_s seconds.
        Raises psycopg2.Error on SQL error.
        """
        _ = signal.SIGALRM
        try:
            cur = self.conn.cursor()
            cur.execute(f"SET statement_timeout = {timeout_s * 1000}")
            cur.execute("BEGIN READ ONLY")

            start = time.perf_counter()
            cur.execute(query)
            rows = cur.fetchall()
            elapsed_ms = (time.perf_counter() - start) * 1000

            self.conn.rollback()
            cur.close()

            content = json.dumps(rows, sort_keys=True, default=str)
            result_hash = hashlib.md5(content.encode()).hexdigest()
            row_count = len(rows)

            return float(elapsed_ms), result_hash, int(row_count)

        except psycopg2.extensions.QueryCanceledError as error:
            self.conn.rollback()
            raise TimeoutError(f"Query exceeded {timeout_s}s timeout") from error
        except psycopg2.Error:
            self.conn.rollback()
            raise

    def _build_observation(self, query: str, level: int, target_ms: float) -> Observation:
        """
        Build a full Observation for a query.

        Runs EXPLAIN ANALYZE and computes all derived fields.
        """
        plan = get_explain_dict(self.conn, query, timeout_ms=self.timeout_ms)
        execution_ms, result_hash, _ = self.execute_and_measure(query)
        diagnosis = diagnose_bottleneck(plan, TABLE_SIZES)

        action_map = {
            "SEQ_SCAN_SELECTIVE_FILTER": ["push_filter", "suggest_index", "add_cte"],
            "NESTED_LOOP_HIGH_CARDINALITY": ["rewrite_join", "add_cte", "reorder_joins"],
            "MISSING_INDEX_FILTER": ["suggest_index", "push_filter"],
            "SUBQUERY_MATERIALIZABLE": ["add_cte", "push_filter"],
            "MULTI_JOIN_NO_STATS": ["reorder_joins", "add_cte", "rewrite_join"],
            "ALREADY_OPTIMAL": ["push_filter", "limit_early"],
        }
        suggested = action_map.get(diagnosis, ["push_filter"])

        hcn = plan.get("highest_cost_node", {})
        total_cost = float(plan.get("total_cost", 1) or 1)
        hcn_cost = float(hcn.get("Total Cost", 0) or 0)
        pct = (hcn_cost / total_cost * 100) if total_cost > 0 else 0
        cost_hotspot = f"{hcn.get('Node Type', 'unknown')} ({pct:.0f}% of total cost)"

        rows = float(plan.get("rows", 0) or 0)
        largest_table = max(TABLE_SIZES.values())
        rows_ratio = min(rows / largest_table, 1.0) if largest_table > 0 else 0.0

        return Observation(
            original_query=query,
            explain_plan=plan,
            execution_ms=float(execution_ms),
            result_hash=result_hash,
            schema_context=SCHEMA_SUMMARY,
            step_count=0,
            task_level=int(level),
            bottleneck_diagnosis=diagnosis,
            suggested_actions=suggested,
            cost_hotspot=cost_hotspot,
            rows_scanned_ratio=float(rows_ratio),
            target_ms=float(target_ms),
        )

    def reset(self, seed: Optional[int] = None) -> dict:
        """
        Start a new episode.

        Returns observation as dict (JSON-serializable).
        """
        # SQLSage-FIX: curriculum-gate (see fix_training.py / sqlsage-curriculum.json)
        level: int
        cpath = Path(__file__).resolve().parent.parent / "sqlsage-curriculum.json"
        if cpath.is_file():
            try:
                st = json.loads(cpath.read_text())
            except (json.JSONDecodeError, OSError, TypeError):
                st = {}
            if st.get("gating"):
                n = int(st.get("l1_count", 0) or 0) + 1
                m = int(st.get("l1_min", 200) or 200)
                st["l1_count"] = n
                try:
                    cpath.write_text(json.dumps(st, indent=2) + "\n")
                except OSError:
                    pass
                if n <= m:
                    level = 1
                else:
                    rng = random.Random(int(seed) if seed is not None else int(time.time()) % 2**31)
                    ul = int(st.get("unlocked_max_level", 3) or 3)
                    level = rng.randint(1, min(3, max(1, ul)))
            else:
                level = 1
        else:
            level = 1
        task = self._get_task(level=level, seed=seed)
        self.target_ms = float(task.target_ms)
        self.best_query = task.query
        self.best_ms = float("inf")

        self._state = self._build_observation(task.query, task.level, task.target_ms)
        return self._state

    def state(self) -> Observation:
        """Return the latest observation object."""
        if self._state is None:
            raise RuntimeError("Call reset() before state()")
        return self._state

    def step(self, action: str, rewritten_query: str) -> Tuple[dict, float, bool, dict]:
        """
        Execute one step in the episode.

        Returns:
            (observation_dict, reward, done, info)
        """
        if self._state is None:
            raise RuntimeError("Call reset() before step()")

        if action not in VALID_ACTIONS:
            return self._state.to_dict(), -15.0, False, {
                "error": "invalid_action",
                "message": f"Unsupported action: {action}",
            }

        if action == "revert":
            rewritten_query = self.best_query or self._state.original_query

        if not self.validate_sql(rewritten_query):
            return self._state.to_dict(), -15.0, False, {
                "error": "syntax_error",
                "message": "Invalid SQL or DDL statement detected",
            }

        try:
            new_ms, new_hash, _ = self.execute_and_measure(rewritten_query, timeout_s=5)
        except TimeoutError:
            return self._state.to_dict(), -12.0, False, {
                "error": "timeout",
                "message": "Query exceeded 5 second timeout",
            }
        except Exception as error:
            return self._state.to_dict(), -15.0, False, {
                "error": "execution_error",
                "message": str(error),
            }

        if new_hash != self._state.result_hash:
            return self._state.to_dict(), -20.0, False, {
                "error": "result_changed",
                "message": "Result set changed — reward hacking detected",
            }

        try:
            new_plan = get_explain_dict(self.conn, rewritten_query, timeout_ms=self.timeout_ms)
        except Exception:
            new_plan = self._state.explain_plan

        reward, breakdown = compute_reward(
            old_ms=self._state.execution_ms,
            new_ms=new_ms,
            new_plan=new_plan,
            old_plan=self._state.explain_plan,
            step_number=self._state.step_count,
            table_sizes=TABLE_SIZES,
        )

        if new_ms < self.best_ms:
            self.best_ms = float(new_ms)
            self.best_query = rewritten_query

        self._state.previous_rewrites.append(rewritten_query)
        self._state.previous_rewards.append(float(reward))
        self._state.execution_ms = float(new_ms)
        self._state.explain_plan = new_plan
        self._state.step_count += 1
        self._state.bottleneck_diagnosis = diagnose_bottleneck(new_plan, TABLE_SIZES)

        done = bool(new_ms <= self.target_ms or self._state.step_count >= self.max_steps)
        info = {
            "reward_breakdown": breakdown,
            "new_ms": float(new_ms),
            "best_ms": float(self.best_ms),
            "step_count": int(self._state.step_count),
        }
        return self._state.to_dict(), float(reward), done, info
