"""EXPLAIN ANALYZE parser helpers for SQLSage."""

from __future__ import annotations

from typing import Any

import psycopg2.extensions
from psycopg2 import errors


def count_node_type(plan: dict[str, Any], node_type: str) -> int:
    count = 1 if plan.get("Node Type") == node_type else 0
    for child in plan.get("Plans", []) or []:
        count += count_node_type(child, node_type)
    return count


def find_highest_cost(plan: dict[str, Any]) -> dict[str, Any]:
    best = {
        "node_type": plan.get("Node Type", ""),
        "total_cost": float(plan.get("Total Cost", 0.0)),
    }
    for child in plan.get("Plans", []) or []:
        child_best = find_highest_cost(child)
        if float(child_best.get("total_cost", 0.0)) > float(best["total_cost"]):
            best = child_best
    return best


def extract_key_fields(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "node_type": plan.get("Node Type", ""),
        "total_cost": float(plan.get("Total Cost", 0.0)),
        "actual_time_ms": float(plan.get("Actual Total Time", 0.0)),
        "rows": int(plan.get("Actual Rows", 0)),
        "seq_scans": count_node_type(plan, "Seq Scan"),
        "index_scans": count_node_type(plan, "Index Scan"),
        "nested_loops": count_node_type(plan, "Nested Loop"),
        "hash_joins": count_node_type(plan, "Hash Join"),
        "highest_cost_node": find_highest_cost(plan),
        "children": [extract_key_fields(c) for c in plan.get("Plans", []) or []],
    }


def get_explain_dict(
    conn: psycopg2.extensions.connection,
    query: str,
    timeout_ms: int = 5000,
) -> dict[str, Any]:
    """
    Run EXPLAIN (ANALYZE, FORMAT JSON) with a statement timeout.

    Uses a short read-only transaction so a long EXPLAIN cannot hang the server.
    Raises TimeoutError if ``statement_timeout`` fires (same semantics as reference).
    """
    with conn.cursor() as cur:
        cur.execute("BEGIN READ ONLY")
        try:
            cur.execute("SET LOCAL statement_timeout = %s", (timeout_ms,))
            cur.execute(f"EXPLAIN (ANALYZE, FORMAT JSON) {query}")
            raw = cur.fetchone()[0]
        except errors.QueryCanceled as exc:
            raise TimeoutError("explain_query_timeout") from exc
        finally:
            cur.execute("ROLLBACK")
    plan = raw[0]["Plan"]
    return extract_key_fields(plan)
