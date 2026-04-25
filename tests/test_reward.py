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
