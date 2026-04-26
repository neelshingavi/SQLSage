"""EXPLAIN parsing and lightweight query diagnostics helpers."""

import json
import platform
import signal
from typing import Any

import psycopg2

IS_UNIX = platform.system() != "Windows"


class QueryTimeoutError(Exception):
    """Raised when EXPLAIN exceeds the configured timeout."""


class timeout_context:
    """Context manager that applies a SIGALRM timeout on Unix systems."""

    def __init__(self, seconds: int):
        """Initialize timeout context with timeout seconds."""
        self.seconds = max(int(seconds or 0), 0)
        self._previous_handler = None

    def _handle_timeout(self, signum: int, frame: Any) -> None:
        """Signal handler that raises QueryTimeoutError."""
        del signum, frame
        raise QueryTimeoutError("query execution exceeded timeout")

    def __enter__(self) -> "timeout_context":
        """Enter the timeout context and arm alarm when supported."""
        if not IS_UNIX or self.seconds <= 0:
            return self
        self._previous_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, self._handle_timeout)
        signal.alarm(self.seconds)
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        """Cancel alarm and restore previous signal handler."""
        del exc_type, exc_value, traceback
        if IS_UNIX:
            signal.alarm(0)
            if self._previous_handler is not None:
                signal.signal(signal.SIGALRM, self._previous_handler)
        return False


def count_node_type(plan: dict, node_type: str) -> int:
    """Recursively count occurrences of a specific PostgreSQL node type."""
    if not isinstance(plan, dict):
        return 0
    count = 1 if plan.get("Node Type") == node_type else 0
    children = plan.get("Plans", [])
    if not isinstance(children, list):
        return count
    for child in children:
        count += count_node_type(child, node_type)
    return int(max(count, 0))


def find_highest_cost(plan: dict) -> dict:
    """Return the plan node with the highest `Total Cost`."""
    if not isinstance(plan, dict) or not plan:
        return {"Node Type": "unknown", "Total Cost": 0}

    best_node = plan
    try:
        best_cost = float(plan.get("Total Cost", 0))
    except (TypeError, ValueError):
        best_cost = 0.0

    children = plan.get("Plans", [])
    if not isinstance(children, list):
        children = []
    for child in children:
        child_best = find_highest_cost(child)
        try:
            child_cost = float(child_best.get("Total Cost", 0))
        except (TypeError, ValueError):
            child_cost = 0.0
        if child_cost > best_cost:
            best_cost = child_cost
            best_node = child_best
    return best_node if isinstance(best_node, dict) else {"Node Type": "unknown", "Total Cost": 0}


def extract_key_fields(plan: dict) -> dict:
    """Extract normalized diagnostics fields from a plan node recursively."""
    if not isinstance(plan, dict):
        plan = {}

    try:
        total_cost = float(plan.get("Total Cost", 0))
    except (TypeError, ValueError):
        total_cost = 0.0
    try:
        actual_time_ms = float(plan.get("Actual Total Time", 0))
    except (TypeError, ValueError):
        actual_time_ms = 0.0
    try:
        rows = int(plan.get("Actual Rows", 0))
    except (TypeError, ValueError):
        rows = 0

    children_source = plan.get("Plans", [])
    if not isinstance(children_source, list):
        children_source = []

    return {
        "node_type": plan.get("Node Type", "unknown"),
        "total_cost": total_cost,
        "actual_time_ms": actual_time_ms,
        "rows": rows,
        "seq_scans": int(max(count_node_type(plan, "Seq Scan"), 0)),
        "index_scans": int(max(count_node_type(plan, "Index Scan"), 0)),
        "nested_loops": int(max(count_node_type(plan, "Nested Loop"), 0)),
        "hash_joins": int(max(count_node_type(plan, "Hash Join"), 0)),
        "highest_cost_node": find_highest_cost(plan),
        "children": [extract_key_fields(child) for child in children_source],
    }


def get_explain_dict(conn, query: str, timeout_ms: int | None = None) -> dict:
    """
    Run EXPLAIN (ANALYZE, FORMAT JSON) on query using conn.

    Args:
        conn: active psycopg2 connection
        query: SQL query string to analyze

    Returns:
        dict from extract_key_fields() — all keys always present

    Raises:
        QueryTimeoutError: if query exceeds 5000ms
        psycopg2.Error: if query is syntactically invalid
    """
    explain_query = f"EXPLAIN (ANALYZE, FORMAT JSON) {query}"
    timeout_seconds = 5 if timeout_ms is None else max(int(timeout_ms / 1000), 1)

    with timeout_context(timeout_seconds):
        cur = conn.cursor()
        try:
            cur.execute(explain_query)
            result = cur.fetchone()
        finally:
            cur.close()

    if not result:
        return extract_key_fields({})

    plan_json = result[0]
    if isinstance(plan_json, str):
        plan_json = json.loads(plan_json)

    root_plan = {}
    if isinstance(plan_json, list) and plan_json:
        first_item = plan_json[0]
        if isinstance(first_item, dict):
            root_plan = first_item.get("Plan", {}) if isinstance(first_item.get("Plan", {}), dict) else {}
    return extract_key_fields(root_plan)


def measure_execution_time(conn, query: str) -> float:
    """
    Run the query and return actual execution time in milliseconds.

    Uses EXPLAIN ANALYZE to get the real execution time.
    Returns 0.0 on error.
    """
    try:
        result_dict = get_explain_dict(conn, query)
        return float(result_dict.get("actual_time_ms", 0.0))
    except (QueryTimeoutError, psycopg2.Error, TypeError, ValueError, KeyError):
        return 0.0


def get_result_hash(conn, query: str) -> str:
    """
    Execute query and return MD5 hash of the result set.

    Used by anti-cheat system to verify result set unchanged.
    Returns hex string MD5 hash.
    Returns empty string on error.
    """
    import hashlib

    try:
        cur = conn.cursor()
        try:
            cur.execute(query)
            rows = cur.fetchall()
        finally:
            cur.close()
        content = json.dumps(rows, sort_keys=True, default=str)
        return hashlib.md5(content.encode()).hexdigest()
    except (psycopg2.Error, TypeError, ValueError):
        return ""


def diagnose_bottleneck(plan: dict, table_sizes: dict = None) -> str:
    """
    Analyse the plan dict and return a diagnosis string.

    Priority order:
    SEQ_SCAN_SELECTIVE_FILTER
    NESTED_LOOP_HIGH_CARDINALITY
    MISSING_INDEX_FILTER
    SUBQUERY_MATERIALIZABLE
    MULTI_JOIN_NO_STATS
    ALREADY_OPTIMAL
    """
    if table_sizes is None:
        table_sizes = {}
    del table_sizes

    if not isinstance(plan, dict):
        plan = {}

    seq_scans = int(plan.get("seq_scans", 0) or 0)
    nested_loops = int(plan.get("nested_loops", 0) or 0)
    index_scans = int(plan.get("index_scans", 0) or 0)
    rows = int(plan.get("rows", 0) or 0)
    actual_time = float(plan.get("actual_time_ms", 0) or 0)
    node_type = str(plan.get("node_type", "") or "")

    if seq_scans > 0 and actual_time > 500:
        return "SEQ_SCAN_SELECTIVE_FILTER"
    if nested_loops > 0 and rows > 50000:
        return "NESTED_LOOP_HIGH_CARDINALITY"
    if seq_scans > 0 and index_scans == 0:
        return "MISSING_INDEX_FILTER"
    if "Subquery" in node_type or "SubPlan" in node_type:
        return "SUBQUERY_MATERIALIZABLE"

    def join_depth(current: dict, depth: int = 0) -> int:
        """Compute maximum depth of join nodes in extracted plan format."""
        if not isinstance(current, dict):
            return depth
        current_depth = depth
        if current.get("node_type", "") in ("Nested Loop", "Hash Join", "Merge Join"):
            current_depth += 1
        max_depth = current_depth
        children = current.get("children", [])
        if not isinstance(children, list):
            return max_depth
        for child in children:
            max_depth = max(max_depth, join_depth(child, current_depth))
        return max_depth

    if join_depth(plan) > 2:
        return "MULTI_JOIN_NO_STATS"
    return "ALREADY_OPTIMAL"
