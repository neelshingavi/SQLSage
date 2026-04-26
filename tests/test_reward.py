from sqlsage.reward import compute_reward


def test_reward_speedup_and_plan_improvement():
    old_plan = {"seq_scans": 3, "nested_loops": 2, "total_cost": 1000.0}
    new_plan = {"seq_scans": 1, "nested_loops": 1, "total_cost": 700.0}
    reward = compute_reward(old_ms=1000.0, new_ms=500.0, new_plan=new_plan, old_plan=old_plan)
    # +10 speedup, +10 seq removed, +4 nested removed, +0.9 cost reduction
    assert reward == 24.9


def test_reward_slowdown_penalty():
    old_plan = {"seq_scans": 1, "nested_loops": 0, "total_cost": 100.0}
    new_plan = {"seq_scans": 1, "nested_loops": 0, "total_cost": 110.0}
    reward = compute_reward(old_ms=100.0, new_ms=125.0, new_plan=new_plan, old_plan=old_plan)
    assert reward == -2.0


def test_reward_noop_penalty():
    plan = {"seq_scans": 1, "nested_loops": 1, "total_cost": 100.0}
    reward = compute_reward(old_ms=100.0, new_ms=100.0, new_plan=plan, old_plan=plan)
    assert reward == -5.0


def test_reward_speedup_ratio_capped_at_one():
    old_plan = {"seq_scans": 1, "nested_loops": 0, "total_cost": 100.0}
    new_plan = {"seq_scans": 1, "nested_loops": 0, "total_cost": 100.0}
    reward = compute_reward(old_ms=100.0, new_ms=0.0, new_plan=new_plan, old_plan=old_plan)
    assert reward == 20.0


def test_reward_no_positive_speedup_when_slower():
    old_plan = {"seq_scans": 1, "nested_loops": 0, "total_cost": 100.0}
    new_plan = {"seq_scans": 1, "nested_loops": 0, "total_cost": 100.0}
    reward = compute_reward(old_ms=100.0, new_ms=120.0, new_plan=new_plan, old_plan=old_plan)
    assert reward < 0.0


def test_reward_seq_scan_bonus_only_for_reductions():
    old_plan = {"seq_scans": 1, "nested_loops": 0, "total_cost": 100.0}
    new_plan = {"seq_scans": 3, "nested_loops": 0, "total_cost": 100.0}
    reward = compute_reward(old_ms=100.0, new_ms=100.0, new_plan=new_plan, old_plan=old_plan)
    assert reward == 0.0


def test_reward_nested_loop_bonus_only_for_reductions():
    old_plan = {"seq_scans": 0, "nested_loops": 1, "total_cost": 100.0}
    new_plan = {"seq_scans": 0, "nested_loops": 2, "total_cost": 100.0}
    reward = compute_reward(old_ms=100.0, new_ms=100.0, new_plan=new_plan, old_plan=old_plan)
    assert reward == 0.0


def test_reward_cost_reduction_component():
    old_plan = {"seq_scans": 0, "nested_loops": 0, "total_cost": 100.0}
    new_plan = {"seq_scans": 0, "nested_loops": 0, "total_cost": 50.0}
    reward = compute_reward(old_ms=100.0, new_ms=100.0, new_plan=new_plan, old_plan=old_plan)
    assert reward == 1.5


def test_reward_no_cost_bonus_when_cost_increases():
    old_plan = {"seq_scans": 0, "nested_loops": 0, "total_cost": 100.0}
    new_plan = {"seq_scans": 0, "nested_loops": 0, "total_cost": 120.0}
    reward = compute_reward(old_ms=100.0, new_ms=100.0, new_plan=new_plan, old_plan=old_plan)
    assert reward == 0.0


def test_reward_slowdown_penalty_uses_ratio():
    old_plan = {"seq_scans": 0, "nested_loops": 0, "total_cost": 100.0}
    new_plan = {"seq_scans": 0, "nested_loops": 0, "total_cost": 100.0}
    reward = compute_reward(old_ms=200.0, new_ms=250.0, new_plan=new_plan, old_plan=old_plan)
    # slowdown ratio = (250-200)/200 = 0.25 => -8 * 0.25 = -2.0
    assert reward == -2.0


def test_reward_noop_penalty_not_applied_if_plan_changes():
    old_plan = {"seq_scans": 1, "nested_loops": 0, "total_cost": 100.0}
    new_plan = {"seq_scans": 0, "nested_loops": 0, "total_cost": 100.0}
    reward = compute_reward(old_ms=100.0, new_ms=100.0, new_plan=new_plan, old_plan=old_plan)
    assert reward == 5.0


def test_reward_handles_zero_old_latency_safely():
    old_plan = {"seq_scans": 0, "nested_loops": 0, "total_cost": 0.0}
    new_plan = {"seq_scans": 0, "nested_loops": 0, "total_cost": 0.0}
    reward = compute_reward(old_ms=0.0, new_ms=10.0, new_plan=new_plan, old_plan=old_plan)
    # no division by zero; only no-op doesn't apply because latencies differ
    assert reward == 0.0
