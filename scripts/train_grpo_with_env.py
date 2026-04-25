#!/usr/bin/env python3
"""
Person 3 (hours 6–18): GRPO training entry point — optional GPU stack (Unsloth + TRL).

This file documents the intended Colab/A100 layout from sqlsage_reference §6.2.
If ``torch``, ``trl``, ``unsloth``, and ``transformers`` are installed, it prints a
compact integration checklist; it does not duplicate a full 300-episode trainer here
(because that belongs in Colab with secrets and long runtimes).

For local smoke: use ``scripts/rollout_wandb.py`` against a running ``uvicorn`` server.
"""

from __future__ import annotations

import importlib.util
import textwrap


def main() -> int:
    missing: list[str] = []
    try:
        import torch  # noqa: F401
    except ImportError:
        missing.append("torch")
    try:
        import trl  # noqa: F401
    except ImportError:
        missing.append("trl")
    try:
        import transformers  # noqa: F401
    except ImportError:
        missing.append("transformers")
    # Do not import unsloth here: on CPU-only hosts it can raise after partial init.
    if importlib.util.find_spec("unsloth") is None:
        missing.append("unsloth")

    from sqlsage.training.config import default_grpo_config

    cfg = default_grpo_config()
    print(textwrap.dedent(f"""
    === SQLSage GRPO (Person 3) — integration checklist ===

    Reference GRPO defaults (sqlsage.training.config):
      {cfg}

    1) Start the environment (HF Space or local):
         export SQLSAGE_ENV_URL=https://...hf.space  # or http://127.0.0.1:8000

    2) In Colab (A100), install:
         pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'
         pip install trl openenv-core wandb transformers datasets accelerate huggingface_hub

    3) Load Qwen2.5-1.5B with Unsloth 4-bit + LoRA (see notebooks/sqlsage_grpo_colab.ipynb).

    4) Reward: call the Space over HTTP/WebSocket each GRPO batch — or materialize offline
       rollouts with scripts/rollout_wandb.py and adapt TRL reward_funcs.

    5) Log keys expected by plots/generate_plots.py:
         reward/mean, reward/speedup_ratio, penalty/result_changed, penalty/syntax_error,
         plan/seq_scans_removed, episode_length, task_level

    6) Save merged weights (reference):
         model.save_pretrained_merged('sqlsage-trained', tokenizer, save_method='merged_16bit')

    7) Push + inference smoke:
         python scripts/push_model_to_hub.py --folder ./sqlsage-trained --repo-id YOUR_ID/sqlsage-trained

    See docs/PERSON3_PHASE8_MANUAL.md for what you must do manually (video, HF tokens, Colab).
    """).strip())

    if missing:
        print("\nOptional GPU packages not installed in this env:", ", ".join(missing))
        print("(Expected on laptop — use Google Colab + A100 for full GRPO.)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
