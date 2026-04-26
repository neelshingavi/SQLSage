"""Reward functions for SQL optimization reinforcement learning."""

from typing import Dict, Tuple
import math

MAX_SPEEDUP_REWARD = 15.0
BIG_WIN_BONUS = 5.0
EXCEPTIONAL_BONUS = 8.0
SEQ_SCAN_BONUS = 8.0
NESTED_LOOP_BONUS = 6.0
ROWS_SIGNAL_WEIGHT = 5.0
COST_SIGNAL_WEIGHT = 3.0
SLOWDOWN_WEIGHT = 10.0
NOOP_PENALTIES = [-2.0, -4.0, -6.0, -8.0, -10.0]
NORMALIZATION_DIVISOR = 44.0


def compute_reward(
    old_ms: float,
    new_ms: float,
    new_plan: dict,
    old_plan: dict,
    step_number: int = 0,
    table_sizes: dict = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute reward for a SQL query rewrite attempt.

    Args:
        old_ms: execution time before rewrite (milliseconds)
        new_ms: execution time after rewrite (milliseconds)
        new_plan: plan dict from explain_parser after rewrite
        old_plan: plan dict from explain_parser before rewrite
        step_number: which step in the episode (0-indexed, 0 to 4)
        table_sizes: dict of {table_name: row_count} for rows signal

    Returns:
        Tuple of:
          - normalized_reward: float in [-1.0, +1.0]
          - breakdown: dict with all reward components
    """
    if table_sizes is None:
        table_sizes = {}
    del table_sizes

    if not isinstance(new_plan, dict):
        new_plan = {}
    if not isinstance(old_plan, dict):
        old_plan = {}

    breakdown: Dict[str, float] = {
        "plan_type_bonus": 0.0,
        "speedup_reward": 0.0,
        "rows_signal": 0.0,
        "cost_signal": 0.0,
        "noop_penalty": 0.0,
        "slowdown_penalty": 0.0,
        "raw": 0.0,
        "normalized": 0.0,
    }

    old_seq = int(old_plan.get("seq_scans", 0) or 0)
    new_seq = int(new_plan.get("seq_scans", 0) or 0)
    seq_eliminated = max(old_seq - new_seq, 0)
    breakdown["plan_type_bonus"] += float(SEQ_SCAN_BONUS * seq_eliminated)

    old_nl = int(old_plan.get("nested_loops", 0) or 0)
    new_nl = int(new_plan.get("nested_loops", 0) or 0)
    nl_eliminated = max(old_nl - new_nl, 0)
    breakdown["plan_type_bonus"] += float(NESTED_LOOP_BONUS * nl_eliminated)

    old_ms_val = float(old_ms or 0.0)
    new_ms_val = float(new_ms or 0.0)
    if old_ms_val > 0 and new_ms_val > 0:
        speedup_ratio = min((old_ms_val - new_ms_val) / new_ms_val, 1.0)
        speedup_ratio = max(speedup_ratio, 0.0)
    elif old_ms_val > 0 and new_ms_val == 0:
        speedup_ratio = 1.0
    else:
        speedup_ratio = 0.0
    breakdown["speedup_reward"] = float(MAX_SPEEDUP_REWARD * speedup_ratio)
    if speedup_ratio > 0.5:
        breakdown["speedup_reward"] += float(BIG_WIN_BONUS)
    if speedup_ratio > 0.8:
        breakdown["speedup_reward"] += float(EXCEPTIONAL_BONUS)

    old_rows = float(old_plan.get("rows", 0) or 0.0)
    new_rows = float(new_plan.get("rows", 0) or 0.0)
    if old_rows > 0:
        rows_saved_ratio = max((old_rows - new_rows) / old_rows, 0.0)
        rows_saved_ratio = min(rows_saved_ratio, 1.0)
    else:
        rows_saved_ratio = 0.0
    breakdown["rows_signal"] = float(ROWS_SIGNAL_WEIGHT * rows_saved_ratio)

    old_cost = float(old_plan.get("total_cost", 0) or 0.0)
    new_cost = float(new_plan.get("total_cost", 0) or 0.0)
    if old_cost > 0:
        cost_ratio = max((old_cost - new_cost) / old_cost, 0.0)
        cost_ratio = min(cost_ratio, 1.0)
    else:
        cost_ratio = 0.0
    breakdown["cost_signal"] = float(COST_SIGNAL_WEIGHT * cost_ratio)

    plan_unchanged = (
        old_plan.get("node_type") == new_plan.get("node_type")
        and old_plan.get("seq_scans") == new_plan.get("seq_scans")
        and old_plan.get("nested_loops") == new_plan.get("nested_loops")
        and old_plan.get("hash_joins") == new_plan.get("hash_joins")
    )
    ms_unchanged = abs(old_ms_val - new_ms_val) < (old_ms_val * 0.02)
    if plan_unchanged and ms_unchanged:
        step_idx = max(0, min(int(step_number), len(NOOP_PENALTIES) - 1))
        breakdown["noop_penalty"] = float(NOOP_PENALTIES[step_idx])

    if new_ms_val > old_ms_val and old_ms_val > 0:
        slowdown_ratio = min((new_ms_val - old_ms_val) / old_ms_val, 2.0)
        breakdown["slowdown_penalty"] = float(-SLOWDOWN_WEIGHT * slowdown_ratio)

    raw = (
        breakdown["plan_type_bonus"]
        + breakdown["speedup_reward"]
        + breakdown["rows_signal"]
        + breakdown["cost_signal"]
        + breakdown["noop_penalty"]
        + breakdown["slowdown_penalty"]
    )
    breakdown["raw"] = float(raw)

    normalized = raw / NORMALIZATION_DIVISOR
    normalized = max(-1.0, min(1.0, normalized))
    if not math.isfinite(normalized):
        normalized = 0.0
    breakdown["normalized"] = float(normalized)

    return float(normalized), breakdown


def format_reward_breakdown(breakdown: Dict[str, float]) -> str:
    """
    Return a human-readable string of the reward breakdown.

    Used in logging and debugging.
    """
    lines = [
        f"  Plan type bonus:  {breakdown['plan_type_bonus']:+.2f}",
        f"  Speedup reward:   {breakdown['speedup_reward']:+.2f}",
        f"  Rows signal:      {breakdown['rows_signal']:+.2f}",
        f"  Cost signal:      {breakdown['cost_signal']:+.2f}",
        f"  No-op penalty:    {breakdown['noop_penalty']:+.2f}",
        f"  Slowdown penalty: {breakdown['slowdown_penalty']:+.2f}",
        "  ─────────────────────────────",
        f"  Raw total:        {breakdown['raw']:+.2f}",
        f"  Normalized:       {breakdown['normalized']:+.4f}",
    ]
    return "\n".join(lines)
