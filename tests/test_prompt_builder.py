"""Unit tests for sqlsage.prompt_builder."""

from __future__ import annotations

import pytest

from sqlsage.env import Observation
from sqlsage.prompt_builder import (
    build_baseline_prompt,
    build_optimized_prompt,
    build_plan_only_prompt,
    format_few_shot_section,
    truncate_prompt_if_needed,
)
from sqlsage.rewrite_patterns import ALL_PATTERNS, detect_applicable_patterns


def _long_schema(n: int = 20_000) -> str:
    return "X" * n


def test_output_format_section_always_present() -> None:
    obs: dict = {
        "original_query": "SELECT 1",
        "schema_context": "small",
        "explain_plan": {
            "node_type": "Result",
            "seq_scans": 0,
            "nested_loops": 0,
            "hash_joins": 0,
            "total_cost": 1.0,
            "highest_cost_node": {"Node Type": "Result", "Total Cost": 1.0},
        },
        "execution_ms": 1.0,
        "result_hash": "a",
    }
    p = build_optimized_prompt(obs, None)
    assert "OUTPUT FORMAT" in p
    assert '"action"' in p
    assert '"rewritten_query"' in p
    b = build_baseline_prompt(obs)
    assert "OUTPUT FORMAT" in b or "rewritten_query" in b
    pl = build_plan_only_prompt(obs)
    assert "OUTPUT FORMAT" in pl
    assert '"action"' in pl


def test_prompt_under_1800_token_budget() -> None:
    """≈1 token = 4 chars → 1800 * 4 = 7200 characters."""
    obs: dict = {
        "original_query": "SELECT * FROM lineitem" * 200,
        "schema_context": _long_schema(25_000),
        "explain_plan": {
            "node_type": "Seq Scan",
            "seq_scans": 2,
            "nested_loops": 1,
            "hash_joins": 0,
            "total_cost": 99_000.0,
            "highest_cost_node": {
                "Node Type": "Seq Scan",
                "Total Cost": 50_000.0,
            },
        },
        "execution_ms": 8000.0,
        "result_hash": "h",
    }
    p = build_optimized_prompt(obs, None)
    assert len(p) <= 1800 * 4
    t = truncate_prompt_if_needed("x" * 10_000, max_tokens=1800)
    assert len(t) <= 1800 * 4


def test_pattern_injection_for_seq_scan_observation() -> None:
    plan = {
        "node_type": "Seq Scan",
        "seq_scans": 1,
        "nested_loops": 1,
        "rows": 600_000,
        "total_cost": 1_000.0,
        "hash_joins": 0,
        "highest_cost_node": {"Node Type": "Seq Scan", "Total Cost": 1_000.0},
    }
    assert len(detect_applicable_patterns(plan)) >= 1
    obs: dict = {
        "original_query": "SELECT 1",
        "schema_context": "schema",
        "explain_plan": plan,
        "execution_ms": 10.0,
        "result_hash": "a",
    }
    p = build_optimized_prompt(obs, None)
    assert "PROVEN OPTIMIZATION PATTERNS" in p
    assert "EXAMPLE OPTIMIZATION" in p or "Pattern:" in p
    assert format_few_shot_section(ALL_PATTERNS[:1]) != ""


def test_no_few_shot_for_clean_fast_plan() -> None:
    plan = {
        "node_type": "Index Scan",
        "seq_scans": 0,
        "nested_loops": 0,
        "rows": 10,
        "total_cost": 5.0,
        "hash_joins": 0,
        "index_scans": 1,
        "highest_cost_node": {"Node Type": "Index Scan", "Total Cost": 5.0},
    }
    assert len(detect_applicable_patterns(plan)) == 0
    obs: dict = {
        "original_query": "SELECT 1",
        "schema_context": "schema",
        "explain_plan": plan,
        "execution_ms": 0.5,
        "result_hash": "a",
    }
    p = build_optimized_prompt(obs, None)
    assert "PROVEN OPTIMIZATION PATTERNS" not in p
    p2 = build_plan_only_prompt(obs)
    assert "PROVEN" not in p2


def test_format_previous_attempts() -> None:
    from sqlsage.prompt_builder import format_previous_attempts

    o = Observation(
        original_query="SELECT 1",
        explain_plan={},
        execution_ms=1.0,
        result_hash="x",
        schema_context="",
        previous_rewrites=["SELECT 2", "SELECT 3"],
        previous_rewards=[0.1, -0.2],
    )
    t = format_previous_attempts(o)
    assert "Attempt 1" in t
    assert "SELECT 2" in t
    assert "No previous attempts" not in t
    o2 = Observation(
        original_query="SELECT 1",
        explain_plan={},
        execution_ms=1.0,
        result_hash="x",
        schema_context="",
    )
    assert "No previous attempts" in format_previous_attempts(o2)
