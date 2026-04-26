"""Anti-cheat and query safety checks."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import psycopg2.extensions


BLOCKED_KEYWORDS = {
    "create ",
    "alter ",
    "drop ",
    "truncate ",
    "insert ",
    "update ",
    "delete ",
    "grant ",
    "revoke ",
    "vacuum ",
    "analyze ",
}


def validate_read_only_sql(query: str) -> bool:
    lowered = query.strip().lower()
    if not lowered.startswith(("select", "with", "explain")):
        return False
    return not any(keyword in lowered for keyword in BLOCKED_KEYWORDS)


def normalize_rows(rows: list[tuple[Any, ...]]) -> bytes:
    # Deterministic serialization for stable hashing.
    payload = json.dumps(rows, default=str, separators=(",", ":"))
    return payload.encode("utf-8")


def get_result_hash(rows: list[tuple[Any, ...]]) -> str:
    return hashlib.md5(normalize_rows(rows)).hexdigest()


def execute_read_only(
    conn: psycopg2.extensions.connection,
    query: str,
    timeout_ms: int = 5000,
) -> tuple[float, str, int]:
    """
    Execute query in a read-only transaction with a statement timeout.

    Returns:
        (elapsed_ms, result_hash, row_count)
    """
    if not validate_read_only_sql(query):
        raise ValueError("blocked_or_non_read_only_sql")

    with conn.cursor() as cur:
        cur.execute("BEGIN READ ONLY")
        try:
            cur.execute("SET LOCAL statement_timeout = %s", (timeout_ms,))
            start = time.perf_counter()
            cur.execute(query)
            rows = cur.fetchall()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            result_hash = get_result_hash(rows)
            row_count = len(rows)
        finally:
            cur.execute("ROLLBACK")

    return elapsed_ms, result_hash, row_count
