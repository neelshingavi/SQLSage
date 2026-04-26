#!/usr/bin/env python3
"""Person 3 (hours 18–20): print a Markdown baseline vs trained table from two JSONL exports."""

from __future__ import annotations

import argparse
from pathlib import Path

from sqlsage.training.baseline_report import (
    load_jsonl,
    markdown_comparison_table,
    summarize_rollout_log,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--trained", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("results/baseline_vs_trained.md"))
    args = p.parse_args()

    b = summarize_rollout_log(load_jsonl(args.baseline))
    t = summarize_rollout_log(load_jsonl(args.trained))
    md = markdown_comparison_table(b, t)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8")
    print(md)
    print(f"\nWrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
