"""Dataset construction helpers for SQLSage training."""

from __future__ import annotations

from typing import Any

from .anti_cheat import execute_read_only
from .explain_parser import get_explain_dict
from .rewrite_patterns import detect_applicable_patterns, format_few_shot_for_prompt


def make_prompt(observation: dict[str, Any]) -> str:
    """
    Build the user prompt: query, plan summary, schema, and optional few-shot hints.

    When ``EXPLAIN`` suggests known bottlenecks, up to
    ``max_rewrite_pattern_shots`` (default 2) canonical rewrite examples are
    appended from :mod:`sqlsage.rewrite_patterns`. Set
    ``disable_rewrite_pattern_few_shot`` to a truthy value to skip.
    """
    out = (
        "You are SQLSage, an RL SQL optimizer.\n"
        "Given a SQL query, explain plan summary, runtime, and schema context, "
        "return JSON with keys: action and rewritten_query.\n\n"
        f"ORIGINAL_QUERY:\n{observation['original_query']}\n\n"
        f"EXPLAIN_PLAN:\n{observation['explain_plan']}\n\n"
        f"EXECUTION_MS: {observation['execution_ms']}\n"
        f"RESULT_HASH: {observation['result_hash']}\n\n"
        f"SCHEMA_CONTEXT:\n{observation['schema_context']}\n"
    )
    if observation.get("disable_rewrite_pattern_few_shot"):
        return out
    plan = observation.get("explain_plan") or {}
    if not isinstance(plan, dict):
        plan = {}
    applicable = detect_applicable_patterns(plan)
    n = int(observation.get("max_rewrite_pattern_shots", 2))
    n = max(0, n)
    few = format_few_shot_for_prompt(applicable, max_patterns=n) if n else ""
    if few:
        out += "\nREWRITE_PATTERN_HINTS (ground-truth rewrites; match plan signals; preserve semantics):\n\n"
        out += few
        out += "\n"
    return out


def build_records(
    conn: Any,
    queries: list[str],
    schema_summary: str,
    explain_timeout_ms: int = 120_000,
) -> list[dict[str, Any]]:
    """
    Build plain Python records for training.
    The caller can convert these records into a Hugging Face Dataset.
    """
    records: list[dict[str, Any]] = []
    for query in queries:
        execution_ms, result_hash, _rows = execute_read_only(conn, query)
        obs = {
            "original_query": query,
            "explain_plan": get_explain_dict(conn, query, timeout_ms=explain_timeout_ms),
            "execution_ms": execution_ms,
            "result_hash": result_hash,
            "schema_context": schema_summary,
        }
        records.append({"prompt": make_prompt(obs), "query": query, "observation": obs})
    return records
