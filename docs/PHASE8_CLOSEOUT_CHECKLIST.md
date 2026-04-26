# Phase 8 Closeout Checklist

This is the exact closeout checklist for Step/Phase 8 across all three people.

## What is already done in code

- OpenEnv server + `reset/step/state` flow is implemented.
- Level 1/2/3 task sets are implemented.
- Reward, anti-cheat, and parser modules are implemented.
- HF startup now binds to `$PORT` and bootstraps TPC-H schema on fresh volumes.
- Plot generation and rollout scripts exist.

## What must be completed manually

### 1) Person 1 — Environment data completeness

1. Ensure HF Space is running latest `main`.
2. Confirm runtime has actual TPC-H SF=1 data (not schema-only tables):
   - `lineitem`, `orders`, `customer`, `part`, `supplier`, `partsupp`, `nation`, `region`.
3. Verify `/reset` returns 200 repeatedly (not intermittent 500).
4. Verify `GET /health` and `POST /reset` from Colab.

### 2) Person 2 — Real (non-synthetic) plots

1. Set:
   - `WANDB_ENTITY=<your_entity>`
   - `WANDB_PROJECT=sqlsage-grpo`
2. Run:
   - `python plots/generate_plots.py`
3. If warning appears (`wandb data unavailable`), run with debug:
   - `SQLSAGE_PLOTS_DEBUG=1 python plots/generate_plots.py`
4. Keep only real-data plots for submission.

### 3) Person 3 — Baseline vs trained evidence

1. Baseline rollout export:
   - `WANDB_ENTITY=<your_entity> WANDB_PROJECT=sqlsage-grpo SQLSAGE_ENV_URL=<space_url> python scripts/rollout_wandb.py --episodes 50 --policy identity --out-jsonl results/baseline.jsonl`
2. Trained rollout export (after training):
   - `WANDB_ENTITY=<your_entity> WANDB_PROJECT=sqlsage-grpo SQLSAGE_ENV_URL=<space_url> python scripts/rollout_wandb.py --episodes 50 --policy noisy_identity --out-jsonl results/trained.jsonl`
   - Replace with your true trained-policy exporter when available.
3. Build comparison table:
   - `python scripts/compare_rollouts.py --baseline results/baseline.jsonl --trained results/trained.jsonl`
4. Push merged model to HF Hub:
   - `HF_TOKEN=... python scripts/push_model_to_hub.py --folder ./sqlsage-trained --repo-id <org_or_user>/sqlsage-trained`

### 4) README submission completeness

Add all of the following to `README.md`:

- Problem statement.
- Environment description (obs/action/done).
- Reward summary.
- Anti-cheat summary.
- Training setup (model + algo + compute).
- Real baseline-vs-trained table.
- Plot captions and embedded images.
- Links: HF Space, Colab, video/blog, repo.

### 5) Final submission bundle

Before form submission, verify:

- Space URL works.
- Colab link opens.
- Repo is public and up to date.
- Video/blog URL is valid.
- README contains all required links and real metrics.
