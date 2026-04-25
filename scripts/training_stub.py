"""Training stub: wiring only, does not install or run by default."""

from __future__ import annotations

import argparse
import json

from sqlsage.training.config import default_grpo_config


def main() -> None:
    parser = argparse.ArgumentParser(description="SQLSage training wiring stub")
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args()

    if args.print_config:
        print(json.dumps(default_grpo_config(), indent=2))
        return

    print(
        "This is a coding-only training stub.\n"
        "Manual setup required: install unsloth/trl/openenv/wandb and run from a GPU environment."
    )


if __name__ == "__main__":
    main()
