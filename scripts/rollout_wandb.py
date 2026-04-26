#!/usr/bin/env python3
"""
Person 3 (hours 6–10 / 14–18): roll out episodes against the live SQLSage HTTP API and log to wandb.

Requires: pip install -e '.[training]'  (requests + wandb)
Environment: SQLSAGE_ENV_URL (default http://127.0.0.1:8000), POSTGRES_* on the server side.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from sqlsage.monitoring import METRIC_KEYS
from sqlsage.training.http_sqlsage_client import run_episode_http
from sqlsage.training.wandb_episode_metrics import episode_metrics


def policy_identity(obs: dict) -> tuple[str, str]:
    return "push_filter", str(obs.get("original_query", ""))


def policy_noisy_identity(obs: dict) -> tuple[str, str]:
    """Slight perturbation (still valid SQL) — use for smoke tests only."""
    q = str(obs.get("original_query", "")).rstrip()
    if q.endswith(";"):
        q = q[:-1].rstrip()
    return "push_filter", q + " /*rollout*/;"


def main() -> int:
    parser = argparse.ArgumentParser(description="Roll out SQLSage episodes and log to wandb.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--base-url", default=os.environ.get("SQLSAGE_ENV_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "sqlsage-grpo"))
    parser.add_argument("--run-name", default=os.environ.get("WANDB_RUN_NAME", "rollout-http"))
    parser.add_argument("--policy", choices=("identity", "noisy_identity"), default="identity")
    parser.add_argument("--out-jsonl", type=Path, default=Path("results/rollout_episodes.jsonl"))
    parser.add_argument("--seed-start", type=int, default=0)
    args = parser.parse_args()

    try:
        import wandb
    except ImportError:
        print("Install wandb: pip install wandb", file=sys.stderr)
        return 1

    entity = os.environ.get("WANDB_ENTITY", "").strip()
    wandb_kwargs: dict = {"project": args.project, "name": args.run_name, "reinit": True}
    if entity:
        wandb_kwargs["entity"] = entity

    policy = policy_identity if args.policy == "identity" else policy_noisy_identity

    run = wandb.init(**wandb_kwargs)
    rows_out: list[dict] = []
    episode_returns: list[float] = []

    try:
        for ep in range(args.episodes):
            obs0, traj = run_episode_http(args.base_url, policy, max_steps=5, seed=args.seed_start + ep)
            metrics = episode_metrics(traj, obs0)
            total_r = sum(float(s["reward"]) for s in traj) if traj else 0.0
            episode_returns.append(total_r)
            metrics[METRIC_KEYS["reward_mean"]] = float(sum(episode_returns) / len(episode_returns))

            wandb.log(metrics, step=ep)

            base_ms = float(obs0.get("execution_ms", 0.0))
            final_ms = float(traj[-1]["observation"].get("execution_ms", base_ms)) if traj else base_ms
            syn_n = sum(1 for s in traj if s.get("info", {}).get("error") == "syntax_error")
            res_n = sum(1 for s in traj if s.get("info", {}).get("error") == "result_changed")

            row = {
                "episode": ep,
                "label": args.policy,
                "episode_total_reward": total_r,
                "final_execution_ms": final_ms,
                "speedup_ratio": metrics[METRIC_KEYS["reward_speedup_ratio"]],
                "syntax_penalties": syn_n,
                "result_penalties": res_n,
                "task_level": obs0.get("task_level", 1),
            }
            rows_out.append(row)

        args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.out_jsonl.open("w", encoding="utf-8") as f:
            for row in rows_out:
                f.write(json.dumps(row, default=str) + "\n")
        print(f"Wrote {len(rows_out)} rows to {args.out_jsonl}", flush=True)
    finally:
        run.finish()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
