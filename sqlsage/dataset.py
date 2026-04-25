"""Dataset construction helpers for SQLSage training."""

from __future__ import annotations

from typing import Any

from .anti_cheat import execute_read_only
from .explain_parser import get_explain_dict


def make_prompt(observation: dict[str, Any]) -> str:
    return (
        "You are SQLSage, an RL SQL optimizer.\n"
        "Given a SQL query, explain plan summary, runtime, and schema context, "
        "return JSON with keys: action and rewritten_query.\n\n"
        f"ORIGINAL_QUERY:\n{observation['original_query']}\n\n"
        f"EXPLAIN_PLAN:\n{observation['explain_plan']}\n\n"
        f"EXECUTION_MS: {observation['execution_ms']}\n"
        f"RESULT_HASH: {observation['result_hash']}\n\n"
        f"SCHEMA_CONTEXT:\n{observation['schema_context']}\n"
    )


def build_records(conn: Any, queries: list[str], schema_summary: str) -> list[dict[str, Any]]:
    """
    Build plain Python records for training.
    The caller can convert these records into a Hugging Face Dataset.
    """
    records: list[dict[str, Any]] = []
    for query in queries:
        execution_ms, result_hash, _rows = execute_read_only(conn, query)
        obs = {
            "original_query": query,
            "explain_plan": get_explain_dict(conn, query),
            "execution_ms": execution_ms,
            "result_hash": result_hash,
            "schema_context": schema_summary,
        }
        records.append({"prompt": make_prompt(obs), "query": query, "observation": obs})
    return records
