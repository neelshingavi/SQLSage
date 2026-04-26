# Person 3 — Phase 8 (reference §8 hours 18–24 + §9 submission)

Repo automation covers **baseline export**, **wandb rollout logging**, **plot regeneration**, **Hub upload helper**, and **Colab notebook skeleton**. The items below **cannot** be automated from this codebase and must be done by you.

---

## What is already in the repo

| Artifact | Purpose |
| --- | --- |
| `sqlsage/training/http_sqlsage_client.py` | Sync HTTP `reset` / `step` against the OpenEnv API. |
| `sqlsage/training/wandb_episode_metrics.py` | Maps an episode to wandb keys used by `plots/generate_plots.py`. |
| `sqlsage/training/baseline_report.py` | JSONL helpers + Markdown comparison table. |
| `scripts/rollout_wandb.py` | Logs episodes to **wandb** project `sqlsage-grpo` (override with `WANDB_PROJECT`). |
| `scripts/compare_rollouts.py` | Builds `results/baseline_vs_trained.md` from two JSONL files. |
| `scripts/push_model_to_hub.py` | Uploads a merged model folder to the Hub (`HF_TOKEN`). |
| `scripts/train_grpo_with_env.py` | Prints GRPO + Colab checklist (no GPU trainer in-repo). |
| `notebooks/sqlsage_grpo_colab.ipynb` | Paste-friendly Colab cells for Unsloth + TRL wiring. |
| `plots/generate_plots.py` | Regenerate judging PNGs after real runs (`WANDB_ENTITY`). |

---

## What you must do manually

### 1. Accounts and secrets (one-time)

- [ ] **Hugging Face**: create/access token with **write** scope for model + Space. Store as `HF_TOKEN` (never commit it). Rotate any token that was ever pasted into a git remote URL.
- [ ] **Weights & Biases**: `wandb login` on every machine that trains or runs `rollout_wandb.py`. Set **`WANDB_ENTITY`** to your team or user slug so `plots/generate_plots.py` can pull runs.
- [ ] **Google Colab**: Runtime → **A100** GPU for Qwen 1.5B + Unsloth + TRL (reference §7.4).

### 2. Environment URL (Person 1 Space or local)

- [ ] Deploy or run the SQLSage OpenEnv server so it is reachable from Colab (public HTTPS preferred).
- [ ] Set **`SQLSAGE_ENV_URL`** to that base URL (no trailing slash required) when running rollouts or when wiring TRL to call `reset` / `step`.

### 3. Full GRPO training (300+ episodes)

- [ ] Open **`notebooks/sqlsage_grpo_colab.ipynb`** in Colab, follow cells: install stack, `wandb.init`, load model, connect reward to the live env (HTTP or OpenEnv `EnvClient` / WebSocket if you prefer sessions).
- [ ] Ensure logged metrics match: `reward/mean`, `reward/speedup_ratio`, `penalty/result_changed`, `penalty/syntax_error`, `plan/seq_scans_removed`, `episode_length`, `task_level` (see `sqlsage/monitoring.py`).
- [ ] Run the long job; monitor for **reward hacking** (`penalty/result_changed` rising).
- [ ] Save merged 16-bit weights per reference: `save_pretrained_merged(..., save_method='merged_16bit')`.

### 4. Baseline vs trained numbers (hours 18–20)

- [ ] Export a **baseline** JSONL (identity policy):  
  `WANDB_ENTITY=... python scripts/rollout_wandb.py --episodes 50 --policy identity --out-jsonl results/baseline.jsonl`
- [ ] After training, export a **trained** policy JSONL the same way (or from your trainer’s eval loop) to `results/trained.jsonl`.
- [ ] Generate the table:  
  `python scripts/compare_rollouts.py --baseline results/baseline.jsonl --trained results/trained.jsonl`
- [ ] Copy the table into the **main README** “Results” section with **real** ms and reward numbers.

### 5. Hugging Face Hub model (hours 20–22)

- [ ] Point `--folder` at the **merged** export directory that contains `config.json`, tokenizer files, and weights.
- [ ] Run:  
  `HF_TOKEN=... python scripts/push_model_to_hub.py --folder ./sqlsage-trained --repo-id YOUR_ORG/sqlsage-trained`
- [ ] In the Hub model card, add inference instructions and link to Space + Colab.

### 6. Judging plots (Person 2 artifact; you refresh after real wandb)

- [ ] `export WANDB_ENTITY=...` then `python plots/generate_plots.py` to overwrite PNGs in `plots/`.
- [ ] Commit updated PNGs if they reflect the **real** run.

### 7. Demo video + submission (hours 22–24 + §9)

- [ ] Record a **≤ 2 minute** demo (reference §10): slow query → plan → training clip → improved query → ms table → anti-cheat line.
- [ ] Upload to **YouTube** (unlisted is fine) **or** write a short **HF blog post** with the same story.
- [ ] Add **Space URL**, **Colab link**, **video/blog URL**, and **plot captions** to `README.md` (§9.3 checklist).
- [ ] Submit the Google Form before the deadline with all links.

---

## Quick smoke (local, no GPU)

```bash
pip install -e '.[training]'
# Terminal 1: Postgres + uvicorn (see README)
export SQLSAGE_ENV_URL=http://127.0.0.1:8000
export WANDB_ENTITY=your_entity
python scripts/rollout_wandb.py --episodes 5 --out-jsonl results/smoke.jsonl
python plots/generate_plots.py
```

If `WANDB_ENTITY` is unset, `rollout_wandb.py` still runs but omits the entity (wandb may use your default account).
