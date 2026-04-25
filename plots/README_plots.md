# Judging plots (`plots/`)

**reward_curve.png** — Mean reward vs episode, 20-episode rolling mean, curriculum lines at ~100 / ~200. **penalty_dashboard.png** — Result-hash and syntax penalties vs episode (10-ep rolling overlay). **plan_improvement.png** — Seq scans removed (scatter + trend) and episode length vs training.

Regenerate: `python plots/generate_plots.py` (optional `WANDB_ENTITY` for live wandb).
