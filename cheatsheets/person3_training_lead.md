# SQLSage — Person 3: Training Lead
**Critical: Non-zero reward by Hour 12. 300+ episodes by Hour 18.**

## Colab Setup Sequence
1. Runtime → Change runtime type → **A100** GPU
2. `!pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git' -q`
3. `!pip install trl openenv wandb psycopg2-binary -q` — add `torch` as needed
4. `import torch; print(torch.cuda.get_device_name(0))` — must be A100 for Colab template
5. `pip install -e '.[training]'` (local) per README for `scripts/rollout_wandb.py`

## Env URL for rollouts
| Variable | What it does |
|----------|---------------|
| `SQLSAGE_ENV_URL` | Person 1 Space or local `http://127.0.0.1:8000` (see `README` / `.env.example`) |
| `WANDB_ENTITY`, `WANDB_PROJECT` | W&B org + project (e.g. `sqlsage-grpo`) |

## One-liners (phases: rollout, benchmark, closeout)
| Command | What it does |
|---------|-------------|
| `python scripts/rollout_wandb.py --episodes 5` | Quick smoke (see README) |
| `python run_benchmark.py` | TPC-H style benchmark; writes `results/benchmark_results.json` |
| `python -m sqlsage.run p3-benchmark` | **Alias** → `run_benchmark.py` |
| `python -m sqlsage.run p3-gpu` | **Alias** → CUDA check (`torch`) |
| `python -m sqlsage.run p3-verify` | Quick 5 checks (URL, W&B, plots, benchmark file, README links) |
| `python -m sqlsage.status_checker --json` | Full hackathon gate JSON |
| `docs/PERSON3_PHASE8_MANUAL.md` | **§7–8 + §9** — Colab, Hub, demo video, submission |
| `docs/PHASE8_CLOSEOUT_CHECKLIST.md` | All-roles Phase 8 closeout |
| `notebooks/sqlsage_grpo_colab.ipynb` | GRPO training skeleton |

## Model Save — CRITICAL
**CORRECT**
```python
model.save_pretrained_merged(
    "sqlsage-trained", tokenizer, save_method="merged_16bit"
)
```

**WRONG — can corrupt 4-bit quality**
```python
model.merge_and_unload()  # DO NOT USE on 4-bit as sole export path
```

## Common Fixes
| Symptom | Fix |
|---------|-----|
| CUDA OOM | `batch_size=2`, `max_completion_length=256`, `grad_accum=8` |
| Invalid JSON from model | Few-shot JSON example in prompt (see `fix_training` / notebook) |
| Reward flat after ~50 ep | `python fix_training.py --issue flat_reward` |
| `result_changed` penalty | `python fix_training.py --issue result_changed` |
| Episodes very slow | Smaller TPC-H SF (0.1), or lighter tasks |
| Curriculum stuck | `python fix_training.py --issue curriculum_stuck` |
| Syntax errors in actions | `python fix_training.py --issue syntax_error` |

## Reward is working when
- `reward/mean` trending up in W&B
- `plan/seq_scans_removed` (or similar) > 0 when appropriate
- `penalty/result_changed` near **zero** when results match
- Episode length / latency moving in the right direction for your run
- Benchmark: example target **Q5** latency under your documented threshold (adjust per SF)

## wandb metrics to watch
`reward/mean`, speedup-related scalars, `penalty/result_changed`, `penalty/syntax_error`, plan/scan metrics, `episode_length` — names depend on your logging callback (see `monitor_training.py` / notebook).
