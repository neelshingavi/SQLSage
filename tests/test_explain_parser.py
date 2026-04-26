"""Tests for sqlsage.explain_parser using a live PostgreSQL connection."""

from sqlsage.explain_parser import (
    count_node_type,
    diagnose_bottleneck,
    find_highest_cost,
    get_explain_dict,
    get_result_hash,
)


def test_get_explain_dict_returns_all_keys(conn):
    """Ensure get_explain_dict returns the expected top-level keys."""
    result = get_explain_dict(conn, "SELECT 1")
    assert isinstance(result, dict)
    for key in (
        "node_type",
        "total_cost",
        "actual_time_ms",
        "rows",
        "seq_scans",
        "index_scans",
        "nested_loops",
        "hash_joins",
        "highest_cost_node",
        "children",
    ):
        assert key in result


def test_types_are_correct(conn):
    """Ensure extracted fields keep stable Python types."""
    result = get_explain_dict(conn, "SELECT 1")
    assert isinstance(result["total_cost"], float)
    assert isinstance(result["actual_time_ms"], float)
    assert isinstance(result["rows"], int)
    assert isinstance(result["seq_scans"], int)
    assert isinstance(result["index_scans"], int)
    assert isinstance(result["nested_loops"], int)
    assert isinstance(result["hash_joins"], int)
    assert isinstance(result["children"], list)
    assert isinstance(result["node_type"], str)


def test_seq_scan_detected(conn):
    """Validate that a table filter can surface sequential scan usage."""
    result = get_explain_dict(conn, "SELECT * FROM orders WHERE orderstatus = 'F'")
    assert result["seq_scans"] >= 1


def test_index_scan_detected(conn):
    """Create test index and verify index scan appears in plan counters."""
    cursor = conn.cursor()
    try:
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_test_pk ON orders(orderkey)")
        cursor.execute("SET enable_seqscan = off")
        conn.commit()
        result = get_explain_dict(conn, "SELECT * FROM orders WHERE orderkey = 1")
    finally:
        reset_cursor = conn.cursor()
        try:
            reset_cursor.execute("SET enable_seqscan = on")
            conn.commit()
        finally:
            reset_cursor.close()
        cursor.close()
    assert result["index_scans"] >= 1


def test_count_node_type_recursive():
    """Ensure recursive node counting traverses nested Plans correctly."""
    fake_plan = {
        "Node Type": "Hash Join",
        "Plans": [
            {"Node Type": "Seq Scan", "Plans": []},
            {"Node Type": "Hash", "Plans": [{"Node Type": "Seq Scan", "Plans": []}]},
        ],
    }
    assert count_node_type(fake_plan, "Seq Scan") == 2
    assert count_node_type(fake_plan, "Hash Join") == 1
    assert count_node_type(fake_plan, "Nested Loop") == 0


def test_find_highest_cost():
    """Ensure highest-cost node selection returns the most expensive node."""
    fake_plan = {
        "Node Type": "Hash Join",
        "Total Cost": 100.0,
        "Plans": [
            {"Node Type": "Seq Scan", "Total Cost": 500.0, "Plans": []},
            {"Node Type": "Hash", "Total Cost": 50.0, "Plans": []},
        ],
    }
    result = find_highest_cost(fake_plan)
    assert result["Node Type"] == "Seq Scan"
    assert result["Total Cost"] == 500.0


def test_get_result_hash_consistent(conn):
    """Ensure repeated hashing of identical query results is deterministic."""
    query = "SELECT orderkey FROM orders ORDER BY orderkey LIMIT 10"
    hash1 = get_result_hash(conn, query)
    hash2 = get_result_hash(conn, query)
    assert hash1 == hash2
    assert len(hash1) == 32
    assert hash1 != ""


def test_get_result_hash_differs_on_different_results(conn):
    """Ensure different result sets produce different result hashes."""
    hash1 = get_result_hash(conn, "SELECT 1")
    hash2 = get_result_hash(conn, "SELECT 2")
    assert hash1 != hash2


def test_diagnose_bottleneck_seq_scan():
    """Ensure seq-scan heavy high-latency pattern is prioritized."""
    fake_plan = {
        "seq_scans": 2,
        "nested_loops": 0,
        "index_scans": 0,
        "rows": 1000,
        "actual_time_ms": 1500.0,
        "node_type": "Seq Scan",
        "children": [],
    }
    assert diagnose_bottleneck(fake_plan) == "SEQ_SCAN_SELECTIVE_FILTER"


def test_diagnose_bottleneck_nested_loop():
    """Ensure high-cardinality nested loop is diagnosed correctly."""
    fake_plan = {
        "seq_scans": 0,
        "nested_loops": 1,
        "index_scans": 0,
        "rows": 200000,
        "actual_time_ms": 3000.0,
        "node_type": "Nested Loop",
        "children": [],
    }
    assert diagnose_bottleneck(fake_plan) == "NESTED_LOOP_HIGH_CARDINALITY"


def test_diagnose_bottleneck_already_optimal():
    """Ensure efficient plans return ALREADY_OPTIMAL diagnosis."""
    fake_plan = {
        "seq_scans": 0,
        "nested_loops": 0,
        "index_scans": 2,
        "rows": 100,
        "actual_time_ms": 10.0,
        "node_type": "Index Scan",
        "children": [],
    }
    assert diagnose_bottleneck(fake_plan) == "ALREADY_OPTIMAL"
