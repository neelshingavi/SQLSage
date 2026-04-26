"""Local smoke checks for SQLSage environment."""

from __future__ import annotations

import json

from sqlsage.tasks import LEVEL1_TASKS, LEVEL2_TASKS, LEVEL3_TASKS
from sqlsage.training.config import default_grpo_config


def main() -> None:
    summary = {
        "level1_tasks": len(LEVEL1_TASKS),
        "level2_tasks": len(LEVEL2_TASKS),
        "level3_tasks": len(LEVEL3_TASKS),
        "grpo_config": default_grpo_config(),
    }
    print(json.dumps(summary, indent=2))
    print("Smoke checks passed (static checks only).")


if __name__ == "__main__":
    main()
