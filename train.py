"""HF Spaces GRPO training entrypoint for SQLSage."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import requests


ProgressCallback = Optional[Callable[[float, str], None]]


def _emit_progress(progress_callback: ProgressCallback, progress: float, message: str) -> None:
    """Emit a normalized progress update when callback is provided."""
    if progress_callback is not None:
        progress_callback(float(max(0.0, min(1.0, progress))), message)


def _extract_sql_candidate(text: str) -> str:
    """Extract SQL from fenced block when present, else return raw text."""
    if not text:
        return "SELECT 1"
    match = re.search(r"```sql\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        candidate = match.group(1).strip()
        if candidate:
            return candidate
    return text.strip()


@dataclass
class SQLSageEnvClient:
    """Minimal HTTP client for deployed SQLSage OpenEnv Space."""

    base_url: str
    timeout_s: float = 180.0

    def _url(self, path: str) -> str:
        """Build absolute endpoint URL."""
        return f"{self.base_url.rstrip('/')}{path}"

    def reset(self, seed: Optional[int] = None) -> dict:
        """Call /reset and return observation payload."""
        payload: Dict[str, Any] = {}
        if seed is not None:
            payload["seed"] = int(seed)
        response = requests.post(self._url("/reset"), json=payload, timeout=self.timeout_s)
        response.raise_for_status()
        body = response.json()
        return dict(body.get("observation", body))

    def step(self, action: str, rewritten_query: str) -> tuple[dict, float, bool, dict]:
        """Call /step and return (observation, reward, done, info)."""
        payload = {"action": {"action": action, "rewritten_query": rewritten_query}}
        response = requests.post(self._url("/step"), json=payload, timeout=self.timeout_s)
        response.raise_for_status()
        body = response.json()
        observation = dict(body.get("observation", {}))
        reward = float(body.get("reward", 0.0))
        done = bool(body.get("done", False))
        info = dict(body.get("info", {}))
        return observation, reward, done, info


def run_training(progress_callback: ProgressCallback = None) -> dict:
    """
    Run SQLSage GRPO training against a deployed HF Space env server.

    The environment URL is read from SQLSAGE_ENV_URL.
    """
    env_url = os.environ.get("SQLSAGE_ENV_URL", "").strip()
    if not env_url:
        raise RuntimeError("Missing SQLSAGE_ENV_URL environment variable.")

    _emit_progress(progress_callback, 0.02, "Loading training dependencies")
    try:
        import wandb
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
        from unsloth import FastLanguageModel
    except ImportError as error:
        raise RuntimeError(
            "Training dependencies missing. Install: unsloth, trl, datasets, wandb, transformers, accelerate."
        ) from error

    _emit_progress(progress_callback, 0.08, "Connecting to SQLSage env Space")
    env_client = SQLSageEnvClient(base_url=env_url)
    env_client.reset(seed=0)

    model_name = os.environ.get("SQLSAGE_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
    output_dir = os.environ.get("SQLSAGE_OUTPUT_DIR", "sqlsage-trained")
    merged_dir = os.environ.get("SQLSAGE_MERGED_DIR", "sqlsage-trained-merged")
    wandb_project = os.environ.get("WANDB_PROJECT", "sqlsage-grpo")

    _emit_progress(progress_callback, 0.12, f"Loading model: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    _emit_progress(progress_callback, 0.22, "Preparing training prompts")
    prompt_rows = [
        {
            "prompt": (
                "You are SQLSage. Rewrite the SQL query for better performance while preserving result semantics. "
                "Return only SQL (or a fenced ```sql block)."
            )
        }
        for _ in range(256)
    ]
    train_dataset = Dataset.from_list(prompt_rows)

    _emit_progress(progress_callback, 0.30, "Initializing Weights & Biases")
    wandb.init(project=wandb_project, name=os.environ.get("WANDB_RUN_NAME", "sqlsage-grpo-hf-space"))

    def reward_from_env(prompts: List[str], completions: List[str], **_: Any) -> List[float]:
        """Compute GRPO reward by stepping the deployed SQLSage env."""
        rewards: List[float] = []
        for idx, completion in enumerate(completions):
            try:
                observation = env_client.reset(seed=idx)
                action_candidates = observation.get("suggested_actions", []) if isinstance(observation, dict) else []
                action = action_candidates[0] if action_candidates else "push_filter"
                sql_query = _extract_sql_candidate(completion)
                _, reward, _, info = env_client.step(action=action, rewritten_query=sql_query)
                rewards.append(float(reward))
                try:
                    wandb.log(
                        {
                            "reward/mean": float(reward),
                            "reward/env_step": float(reward),
                            "penalty/result_changed": float(info.get("error") == "result_changed"),
                            "penalty/syntax_error": float(info.get("error") == "syntax_error"),
                        }
                    )
                except Exception:
                    pass
            except Exception:
                rewards.append(-15.0)
        return rewards

    _emit_progress(progress_callback, 0.40, "Building GRPO trainer")
    training_args = GRPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,
        max_completion_length=512,
        num_generations=8,
        temperature=0.9,
        kl_coeff=0.05,
        num_train_epochs=3,
        report_to="wandb",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        reward_funcs=[reward_from_env],
    )

    _emit_progress(progress_callback, 0.55, "Starting GRPO training")
    train_result = trainer.train()

    _emit_progress(progress_callback, 0.88, "Saving LoRA adapter checkpoint")
    trainer.save_model(output_dir)

    _emit_progress(progress_callback, 0.94, "Saving merged 16-bit model")
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")

    try:
        wandb.finish()
    except Exception:
        pass

    _emit_progress(progress_callback, 1.0, "Training complete")
    return {
        "status": "ok",
        "env_url": env_url,
        "output_dir": output_dir,
        "merged_dir": merged_dir,
        "train_result": str(train_result),
    }


if __name__ == "__main__":
    run_training()
