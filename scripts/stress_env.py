#!/usr/bin/env python3
"""Person 1 (hours 8–14): sequential stress run — reset + identity step, no trainer required.

Use after TPC-H is loaded and optional ``sql/scripts/drop_curriculum_indexes.sql`` has been applied.
Exits non-zero if any episode raises or returns unexpected errors.
"""

from __future__ import annotations

import argparse
import random
import sys


def main() -> int:
    from sqlsage.env import SQLSageEnv
    from sqlsage.tasks import ALL_TASKS, tasks_for_levels

    parser = argparse.ArgumentParser(description="Stress SQLSageEnv with repeated episodes.")
    parser.add_argument("--episodes", type=int, default=50, help="Number of reset() cycles")
    parser.add_argument(
        "--levels",
        type=str,
        default="1,2,3",
        help="Comma-separated task levels to include (e.g. '2' for Level 2 only)",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for task sampling")
    parser.add_argument(
        "--identity-step",
        action="store_true",
        help="After each reset, run one noop step (rewritten_query == original_query)",
    )
    args = parser.parse_args()

    levels = tuple(int(x.strip()) for x in args.levels.split(",") if x.strip())
    tasks = tasks_for_levels(*levels) if levels else list(ALL_TASKS)
    if not tasks:
        print("No tasks for levels:", args.levels, file=sys.stderr)
        return 2

    random.seed(args.seed)
    env = SQLSageEnv(tasks=tasks, max_steps=5)
    failures = 0
    try:
        for i in range(args.episodes):
            env.reset(seed=args.seed + i)
            obs = env.state()
            if args.identity_step:
                _o, reward, done, info = env.step("push_filter", obs.original_query)
                if info.get("error"):
                    print(f"episode {i}: unexpected error {info}", file=sys.stderr)
                    failures += 1
                    continue
                if done and not info:
                    pass
            if i % 10 == 0:
                print(f"episode {i}/{args.episodes} level={obs.task_level} ok")
    except Exception as exc:
        print(f"stress_env failed at episode loop: {exc}", file=sys.stderr)
        return 1
    finally:
        env.close()

    if failures:
        print(f"Completed with {failures} step failures", file=sys.stderr)
        return 1
    print(f"Completed {args.episodes} episodes on {len(tasks)} tasks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
