import math

import pytest

from sqlsage.reward import compute_reward


def make_plan(
    node_type="Seq Scan",
    total_cost=10000.0,
    actual_time_ms=5000.0,
    rows=100000,
    seq_scans=1,
    index_scans=0,
    nested_loops=0,
    hash_joins=0,
) -> dict:
    """Construct a normalized fake plan for reward tests."""
    return {
        "node_type": node_type,
        "total_cost": total_cost,
        "actual_time_ms": actual_time_ms,
        "rows": rows,
        "seq_scans": seq_scans,
        "index_scans": index_scans,
        "nested_loops": nested_loops,
        "hash_joins": hash_joins,
        "highest_cost_node": {},
        "children": [],
    }


def test_seq_scan_eliminated_gives_bonus():
    """Seq scan elimination should grant structural bonus and positive reward."""
    old = make_plan(seq_scans=1)
    new = make_plan(seq_scans=0, node_type="Index Scan", index_scans=1, actual_time_ms=4000.0)
    reward, bd = compute_reward(5000.0, 4000.0, new, old)
    assert bd["plan_type_bonus"] >= 8.0, "Expected at least one seq-scan elimination bonus."
    assert reward > 0.0, "Reward should be positive after removing a seq scan."


def test_nested_loop_eliminated_gives_bonus():
    """Nested loop elimination should grant structural bonus and positive reward."""
    old = make_plan(nested_loops=1, node_type="Nested Loop")
    new = make_plan(nested_loops=0, hash_joins=1, node_type="Hash Join", actual_time_ms=2000.0)
    reward, bd = compute_reward(8000.0, 2000.0, new, old)
    assert bd["plan_type_bonus"] >= 6.0, "Expected nested-loop elimination bonus to be applied."
    assert reward > 0.0, "Reward should be positive for substantial loop elimination improvement."


def test_50_percent_speedup():
    """A near-50% speedup should produce strong positive speedup reward."""
    old = make_plan()
    new = make_plan(actual_time_ms=2500.0)
    reward, bd = compute_reward(5000.0, 2500.0, new, old)
    assert bd["speedup_reward"] >= 15.0, "Base speedup reward should be at least 15 for 50% speedup."
    assert bd["speedup_reward"] >= 20.0, "Big-win bonus should fire for effective >0.5 speedup threshold."
    assert reward > 0.3, "Normalized reward should be meaningfully positive for this improvement."


def test_80_percent_speedup_exceptional():
    """An 80%+ speedup should include both big-win and exceptional bonuses."""
    old = make_plan()
    new = make_plan(actual_time_ms=1000.0)
    reward, bd = compute_reward(5000.0, 1000.0, new, old)
    assert bd["speedup_reward"] >= 28.0, "Expected speedup reward to include base + big + exceptional bonuses."
    assert reward > 0.6, "Normalized reward should be high for exceptional speedup."


def test_noop_step_0_penalty():
    """Step 0 no-op should receive the first progressive penalty."""
    plan = make_plan()
    reward, bd = compute_reward(5000.0, 5000.0, plan, plan, step_number=0)
    assert bd["noop_penalty"] == pytest.approx(-2.0), "Step 0 no-op penalty should be exactly -2.0."
    assert isinstance(reward, float), "Reward should still be a float for no-op cases."


def test_noop_step_4_penalty():
    """Step 4 no-op should receive the maximum progressive penalty."""
    plan = make_plan()
    reward, bd = compute_reward(5000.0, 5000.0, plan, plan, step_number=4)
    assert bd["noop_penalty"] == pytest.approx(-10.0), "Step 4 no-op penalty should be exactly -10.0."
    assert isinstance(reward, float), "Reward should still be a float for late-step no-op cases."


def test_rows_reduced_gives_signal():
    """Reducing processed rows should emit a positive shaping signal."""
    old = make_plan(rows=100000)
    new = make_plan(rows=40000)
    reward, bd = compute_reward(5000.0, 4800.0, new, old)
    assert bd["rows_signal"] > 0.0, "Rows signal should be positive when rows are reduced."
    assert reward > 0.0, "Reward should remain positive when rows and latency both improve."


def test_slowdown_gives_penalty():
    """Latency regressions should apply a negative slowdown component."""
    old = make_plan()
    new = make_plan(actual_time_ms=7000.0)
    reward, bd = compute_reward(5000.0, 7000.0, new, old)
    assert bd["slowdown_penalty"] < 0.0, "Slowdown penalty must be negative when runtime gets worse."
    assert reward < 0.0, "Overall reward should be negative for a clear slowdown."


def test_all_breakdown_keys_always_present():
    """Breakdown should always contain all required float-valued keys."""
    plan = make_plan()
    _, bd = compute_reward(1000.0, 800.0, plan, plan)
    required_keys = [
        "plan_type_bonus",
        "speedup_reward",
        "rows_signal",
        "cost_signal",
        "noop_penalty",
        "slowdown_penalty",
        "raw",
        "normalized",
    ]
    for key in required_keys:
        assert key in bd, f"Breakdown is missing required key: {key}."
        assert isinstance(bd[key], float), f"Breakdown value for {key} must be float."


@pytest.mark.parametrize(
    ("old_ms", "new_ms"),
    [
        (0.0, 0.0),
        (5000.0, 100.0),
        (100.0, 9999.0),
        (5000.0, 5000.0),
        (0.001, 0.001),
    ],
)
def test_normalized_always_in_range(old_ms, new_ms):
    """Normalized reward must stay finite and clipped to [-1, 1]."""
    plan = make_plan()
    reward, bd = compute_reward(old_ms, new_ms, plan, plan)
    assert -1.0 <= reward <= 1.0, "Reward must be clipped to [-1.0, 1.0]."
    assert -1.0 <= bd["normalized"] <= 1.0, "Breakdown normalized must be clipped to [-1.0, 1.0]."
    assert math.isfinite(reward), "Reward must be finite (no NaN/inf)."
    assert math.isfinite(bd["normalized"]), "Breakdown normalized must be finite (no NaN/inf)."


def test_all_values_are_floats():
    """Top-level reward and every breakdown entry should be float typed."""
    plan = make_plan()
    reward, bd = compute_reward(5000.0, 3000.0, plan, plan)
    assert isinstance(reward, float), "Returned reward value must be float."
    for key, val in bd.items():
        assert isinstance(val, float), f"{key} is not float: {type(val)}"


def test_combined_improvement_high_reward():
    """Strong combined plan and runtime improvements should yield high reward."""
    old = make_plan(seq_scans=2, nested_loops=1, rows=100000, total_cost=50000.0, actual_time_ms=8000.0)
    new = make_plan(
        seq_scans=0,
        nested_loops=0,
        hash_joins=1,
        index_scans=2,
        rows=5000,
        total_cost=5000.0,
        actual_time_ms=310.0,
        node_type="Hash Join",
    )
    reward, bd = compute_reward(8420.0, 310.0, new, old)
    assert reward > 0.7, "Combined improvements should produce high normalized reward (> 0.7)."
    assert bd["plan_type_bonus"] >= 14.0, "Plan bonus should include seq + nested-loop elimination bonuses."
