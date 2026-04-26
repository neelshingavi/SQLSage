"""Map SQLSage HTTP rollouts to wandb metric keys (reference §6.4 / Person 3 logging)."""

from __future__ import annotations

from typing import Any

from sqlsage.monitoring import METRIC_KEYS


def _seq_scans(plan: dict[str, Any]) -> float:
    try:
        return float(plan.get("seq_scans", 0))
    except (TypeError, ValueError):
        return 0.0


def episode_metrics(
    trajectory: list[dict[str, Any]],
    initial_obs: dict[str, Any],
) -> dict[str, float]:
    """
    Build one wandb log row (internal keys -> values) for a completed episode.

    Uses the same logical names as ``METRIC_KEYS`` values for direct ``wandb.log``.
    """
    if not trajectory:
        base_ms = float(initial_obs.get("execution_ms", 0.0))
        return {
            METRIC_KEYS["reward_mean"]: 0.0,
            METRIC_KEYS["reward_speedup_ratio"]: 0.0,
            METRIC_KEYS["penalty_result_changed"]: 0.0,
            METRIC_KEYS["penalty_syntax_error"]: 0.0,
            METRIC_KEYS["plan_seq_scans_removed"]: 0.0,
            METRIC_KEYS["episode_length"]: 0.0,
            "task_level": float(initial_obs.get("task_level", 1)),
        }

    rewards = [float(s["reward"]) for s in trajectory]
    mean_step_reward = float(sum(rewards) / max(len(rewards), 1))

    pen_res = sum(1 for s in trajectory if s.get("info", {}).get("error") == "result_changed")
    pen_syn = sum(1 for s in trajectory if s.get("info", {}).get("error") == "syntax_error")

    first_plan = initial_obs.get("explain_plan") or {}
    last_plan = trajectory[-1]["observation"].get("explain_plan") or {}
    seq0 = _seq_scans(first_plan)
    seq1 = _seq_scans(last_plan)
    seq_removed = max(0.0, seq0 - seq1)

    base_ms = float(initial_obs.get("execution_ms", 1.0))
    final_ms = float(trajectory[-1]["observation"].get("execution_ms", base_ms))
    if base_ms <= 0:
        speedup = 0.0
    else:
        speedup = max(0.0, min(1.0, (base_ms - final_ms) / base_ms))

    ep_len = float(len(trajectory))
    task_level = float(trajectory[-1]["observation"].get("task_level", initial_obs.get("task_level", 1)))

    return {
        METRIC_KEYS["reward_mean"]: mean_step_reward,
        METRIC_KEYS["reward_speedup_ratio"]: speedup,
        METRIC_KEYS["penalty_result_changed"]: float(pen_res),
        METRIC_KEYS["penalty_syntax_error"]: float(pen_syn),
        METRIC_KEYS["plan_seq_scans_removed"]: seq_removed,
        METRIC_KEYS["episode_length"]: ep_len,
        "task_level": task_level,
    }


def running_mean_reward(episode_returns: list[float]) -> float:
    if not episode_returns:
        return 0.0
    return float(sum(episode_returns) / len(episode_returns))
