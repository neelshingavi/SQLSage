"""Baseline vs trained comparison helpers (Person 3 hours 18–20)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class BaselineRow:
    label: str
    mean_episode_reward: float
    mean_execution_ms: float
    mean_speedup_ratio: float
    mean_syntax_penalties: float
    mean_result_changed_penalties: float
    episodes: int


def summarize_rollout_log(rows: list[dict[str, Any]]) -> BaselineRow:
    """``rows`` are per-episode dicts from ``scripts/rollout_wandb.py`` JSONL export."""
    n = len(rows)
    if n == 0:
        return BaselineRow("empty", 0.0, 0.0, 0.0, 0.0, 0.0, 0)
    return BaselineRow(
        label=str(rows[0].get("label", "run")),
        mean_episode_reward=float(sum(r["episode_total_reward"] for r in rows) / n),
        mean_execution_ms=float(sum(r["final_execution_ms"] for r in rows) / n),
        mean_speedup_ratio=float(sum(r["speedup_ratio"] for r in rows) / n),
        mean_syntax_penalties=float(sum(r["syntax_penalties"] for r in rows) / n),
        mean_result_changed_penalties=float(sum(r["result_penalties"] for r in rows) / n),
        episodes=n,
    )


def markdown_comparison_table(baseline: BaselineRow, trained: BaselineRow) -> str:
    lines = [
        "| Metric | Baseline | After training |",
        "| --- | ---: | ---: |",
        f"| Episodes | {baseline.episodes} | {trained.episodes} |",
        f"| Mean episode return (sum of step rewards) | {baseline.mean_episode_reward:.2f} | {trained.mean_episode_reward:.2f} |",
        f"| Mean final query latency (ms) | {baseline.mean_execution_ms:.1f} | {trained.mean_execution_ms:.1f} |",
        f"| Mean speedup ratio (0–1) | {baseline.mean_speedup_ratio:.3f} | {trained.mean_speedup_ratio:.3f} |",
        f"| Syntax penalties / episode | {baseline.mean_syntax_penalties:.2f} | {trained.mean_syntax_penalties:.2f} |",
        f"| Result-changed penalties / episode | {baseline.mean_result_changed_penalties:.2f} | {trained.mean_result_changed_penalties:.2f} |",
        "",
        "_Fill this table with numbers exported from real runs (`results/baseline.jsonl` vs `results/trained.jsonl`)._",
    ]
    return "\n".join(lines)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def baseline_row_to_dict(row: BaselineRow) -> dict[str, Any]:
    return asdict(row)
