"""Integration tests for EXPLAIN JSON parsing (requires PostgreSQL)."""

from __future__ import annotations

import pytest

from sqlsage.explain_parser import extract_key_fields, get_explain_dict


def test_extract_key_fields_minimal():
    plan = {
        "Node Type": "Result",
        "Total Cost": 0.01,
        "Actual Total Time": 0.002,
        "Actual Rows": 1,
        "Plans": [],
    }
    out = extract_key_fields(plan)
    assert out["node_type"] == "Result"
    assert out["seq_scans"] == 0
    assert out["nested_loops"] == 0
    assert out["children"] == []


def test_get_explain_dict_simple_select(postgres_conn):
    plan = get_explain_dict(postgres_conn, "SELECT 1 AS x", timeout_ms=10_000)
    assert "node_type" in plan
    assert "total_cost" in plan
    assert isinstance(plan["seq_scans"], int)
    assert isinstance(plan["highest_cost_node"], dict)


def test_get_explain_dict_respects_short_timeout(postgres_conn):
    with pytest.raises(TimeoutError, match="explain_query_timeout"):
        get_explain_dict(postgres_conn, "SELECT pg_sleep(10)", timeout_ms=400)
