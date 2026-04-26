"""Monitoring metric keys and helpers for section 6.4."""

from __future__ import annotations

METRIC_KEYS = {
    "reward_mean": "reward/mean",
    "reward_speedup_ratio": "reward/speedup_ratio",
    "penalty_result_changed": "penalty/result_changed",
    "penalty_syntax_error": "penalty/syntax_error",
    "plan_seq_scans_removed": "plan/seq_scans_removed",
    "episode_length": "episode_length",
}


def init_metrics() -> dict[str, float]:
    return {metric: 0.0 for metric in METRIC_KEYS.values()}
