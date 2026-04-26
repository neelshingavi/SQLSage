# SQLSage — Person 2: Reward & Verifier Engineer
**Your job: reward pipeline works, no reward hacking, plots committed by Hour 16**

## Core Commands
| Command | What it does |
|---------|-------------|
| `pytest tests/ -v` | All unit tests |
| `pytest tests/test_reward.py -v` | Reward tests only |
| `pytest tests/test_explain_parser.py -v` | Parser tests only |
| `pytest tests/test_anti_cheat.py -v` | Anti-cheat tests only |
| `wandb login` | Authenticate W&B |
| `python monitor_training.py` | Live reward diagnostics (when configured) |
| `python plots/generate_plots.py` | **Actual path** — generate PNGs in `plots/` |
| `python -m sqlsage.run p2-reward` | **Alias** → quick `compute_reward` demo + breakdown |
| `python -m sqlsage.run p2-plots` | **Alias** → `plots/generate_plots.py` |
| `python -m sqlsage.run p2-anticheat` | **Alias** → `fix_training.py --issue reward_hacking` |
| `python -m sqlsage.status_checker --json` | Milestone + reward/plots/git checks (JSON) |

## Quick Reward Test
`compute_reward` returns **(normalized ∈ [-1,1], breakdown dict)** — use both:

```python
from sqlsage.reward import compute_reward, format_reward_breakdown
new_plan = {
    "seq_scans": 0, "nested_loops": 0, "hash_joins": 1, "total_cost": 1000,
    "rows": 1e5, "node_type": "x",
}
old_plan = {
    "seq_scans": 2, "nested_loops": 1, "hash_joins": 0, "total_cost": 50000,
    "rows": 1e6, "node_type": "y",
}
n, b = compute_reward(8420.0, 310.0, new_plan, old_plan, step_number=0)
print("normalized:", n)
print(format_reward_breakdown(b))
# Raw sum can be large; see breakdown["raw"] and breakdown["normalized"]
```

## Quick EXPLAIN Test
```bash
psql -h 127.0.0.1 -p 5433 -U postgres -d sqlsage -c \
"EXPLAIN (ANALYZE, FORMAT JSON) SELECT * FROM orders WHERE orderstatus='F' LIMIT 100;"
```

## Anti-Cheat Attack Vectors to Test Manually
1. `LIMIT 0` → strong negative / guard
2. `WHERE 1=0` → strong negative / guard
3. Remove `WHERE` clause (wrong result) → `result_changed` / penalty
4. `CREATE INDEX` / DDL → should be rejected / exception
5. Unchanged query → `noop_penalty` (escalates by step)

## Reward Component Reference (see `sqlsage/reward.py`)
| Component | Rule (simplified) |
|-----------|------------------|
| Speedup | Up to `MAX_SPEEDUP_REWARD` × speedup ratio + `BIG_WIN` / `EXCEPTIONAL` bonuses at high ratio |
| Plan | `+SEQ_SCAN_BONUS` per seq scan removed, `+NESTED_LOOP_BONUS` per nested loop removed |
| Rows / cost | `ROWS_SIGNAL_WEIGHT` × row reduction; `COST_SIGNAL_WEIGHT` × cost reduction |
| No-op | `NOOP_PENALTIES[step]` if plan+latency effectively unchanged |
| Slowdown | `-SLOWDOWN_WEIGHT` × slowdown ratio |
| Normalize | `raw / NORMALIZATION_DIVISOR` → **[-1, 1]** |

## If Reward Hacking Detected
```bash
python fix_training.py --issue reward_hacking
```

## Plots + Evidence (phases: Phase 8, rollout comparison)
| Command | What it does |
|---------|-------------|
| `export WANDB_ENTITY=...; export WANDB_PROJECT=sqlsage-grpo` | Match training project (see `docs/PHASE8_CLOSEOUT_CHECKLIST.md`) |
| `SQLSAGE_PLOTS_DEBUG=1 python plots/generate_plots.py` | Debug missing W&B data for plots |
| `python scripts/compare_rollouts.py --baseline results/baseline.jsonl --trained results/trained.jsonl` | Table when both JSONL exist |
