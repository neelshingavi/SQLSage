"""Reward function for SQLSage RL environment."""

from __future__ import annotations


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def compute_reward(old_ms: float, new_ms: float, new_plan: dict, old_plan: dict) -> float:
    """
    Compute the scalar reward for a rewrite step.

    Formula follows the reference design:
    - +20 * speedup_ratio (cap 1.0)
    - +5 per Seq Scan removed
    - +4 per Nested Loop removed
    - +3 * cost_reduction_ratio
    - -8 * slowdown_ratio if latency got worse
    - -5 no-op penalty when plan and runtime are unchanged
    """
    reward = 0.0

    # Positive rewards
    speedup_ratio = min(_safe_ratio(old_ms - new_ms, old_ms), 1.0)
    reward += 20.0 * max(speedup_ratio, 0.0)

    seq_eliminated = float(old_plan.get("seq_scans", 0) - new_plan.get("seq_scans", 0))
    reward += 5.0 * max(seq_eliminated, 0.0)

    nested_eliminated = float(old_plan.get("nested_loops", 0) - new_plan.get("nested_loops", 0))
    reward += 4.0 * max(nested_eliminated, 0.0)

    cost_reduction = _safe_ratio(
        float(old_plan.get("total_cost", 0.0) - new_plan.get("total_cost", 0.0)),
        max(float(old_plan.get("total_cost", 0.0)), 1.0),
    )
    reward += 3.0 * max(cost_reduction, 0.0)

    # Penalties
    if new_ms > old_ms:
        slowdown = _safe_ratio(new_ms - old_ms, old_ms)
        reward -= 8.0 * slowdown

    if new_plan == old_plan and old_ms == new_ms:
        reward -= 5.0

    return float(reward)
