"""HF Spaces GRPO training entrypoint for SQLSage."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import requests


ProgressCallback = Optional[Callable[[float, str], None]]


def _observation_to_make_prompt_dict(observation: dict[str, Any]) -> dict[str, Any]:
    """Map an env/HTTP observation dict to the keys :func:`sqlsage.dataset.make_prompt` expects."""
    ex = observation.get("explain_plan", {})
    if isinstance(ex, str):
        try:
            import json

            ex = json.loads(ex)
        except Exception:  # pragma: no cover
            ex = {}
    if not isinstance(ex, dict):
        ex = {}
    return {
        "original_query": str(observation.get("original_query", "")),
        "explain_plan": ex,
        "execution_ms": float(observation.get("execution_ms", 0.0)),
        "result_hash": str(observation.get("result_hash", "")),
        "schema_context": str(observation.get("schema_context", "")),
    }


def _build_env_prompt_rows(
    env_client: "SQLSageEnvClient", n: int
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """
    Call ``reset(seed=i)`` for ``i in range(n)``, build :func:`make_prompt` text per row, and
    a map *full prompt string* → *seed* so the reward can ``reset`` the same task as the model
    was shown (TRL may broadcast one prompt to many completions, so the map is by exact text).
    """
    from sqlsage.dataset import make_prompt

    rows: list[dict[str, str]] = []
    prompt_to_seed: dict[str, int] = {}
    for i in range(int(n)):
        obs = env_client.reset(seed=i)
        if not isinstance(obs, dict):
            obs = {}
        text = make_prompt(_observation_to_make_prompt_dict(obs))
        if text in prompt_to_seed and prompt_to_seed[text] != i:
            text = f"{text}\n# disambiguation: env_seed={i}\n"
        prompt_to_seed[text] = i
        rows.append({"prompt": text})
    return rows, prompt_to_seed


def _parse_action_and_sql(
    completion: str, suggested_actions: List[str]
) -> tuple[str, str]:
    """
    Prefer a JSON object with ``action`` and ``rewritten_query`` (matches ``make_prompt``);
    otherwise fall back to a ```sql`` fence / raw text and the first suggested action.
    """
    default_a = (suggested_actions[0] if suggested_actions else "push_filter") or "push_filter"
    s = (completion or "").strip()
    m = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", s, flags=re.IGNORECASE)
    blob = m.group(1).strip() if m else s
    start = blob.find("{")
    if start >= 0:
        try:
            from json import JSONDecoder

            data, _ = JSONDecoder().raw_decode(blob, start)
            if isinstance(data, dict):
                rq = str(data.get("rewritten_query", "")).strip()
                if rq:
                    act = str(data.get("action", "") or default_a).strip() or default_a
                    if act in (
                        "rewrite_join",
                        "add_cte",
                        "push_filter",
                        "reorder_joins",
                        "suggest_index",
                        "limit_early",
                        "revert",
                    ):
                        return act, rq
                    return default_a, rq
        except Exception:
            pass
    return default_a, _extract_sql_candidate(completion)


def _align_prompt(i: int, prompts: List[str]) -> str:
    """Index ``i`` into ``prompts``; if TRL broadcast a single prompt, use ``prompts[0]``."""
    if not prompts:
        return ""
    if i < len(prompts):
        return prompts[i]
    return prompts[0]


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

    The environment URL is read from ``SQLSAGE_ENV_URL``.

    **Prompts:** By default, the training set is built with :func:`sqlsage.dataset.make_prompt`
    and a fresh ``/reset?seed=`` / observation per row so the model sees the same query,
    EXPLAIN summary, schema, and rewrite-pattern hints that the reward step evaluates.
    Set ``SQLSAGE_TRAIN_PLAIN_PROMPT=1`` to use a short generic prompt for smoke tests (reward
    still uses index ``idx`` to ``reset``). ``SQLSAGE_TRAIN_DATASET_SIZE`` controls row count
    (default 256). Long plans + few-shots can exceed small ``SQLSAGE_MAX_SEQ_LENGTH``; increase
    it if you see truncation.
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

    max_seq = int(os.environ.get("SQLSAGE_MAX_SEQ_LENGTH", "2048"))
    _emit_progress(progress_callback, 0.12, f"Loading model: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq,
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

    n_rows = int(os.environ.get("SQLSAGE_TRAIN_DATASET_SIZE", "256"))
    use_plain = os.environ.get("SQLSAGE_TRAIN_PLAIN_PROMPT", "").lower() in (
        "1",
        "true",
        "yes",
    )
    prompt_to_seed: dict[str, int] = {}
    _emit_progress(
        progress_callback,
        0.22,
        "Preparing training prompts"
        + (" (plain generic)" if use_plain else " (make_prompt + env observations)"),
    )
    if use_plain:
        # Same text every row: reward must pair by example index, not by prompt string.
        generic = (
            "You are SQLSage. Rewrite the SQL query for better performance while preserving result semantics. "
            "Return only SQL (or a fenced ```sql block)."
        )
        prompt_rows = [{"prompt": generic} for _ in range(n_rows)]
    else:
        prompt_rows, prompt_to_seed = _build_env_prompt_rows(env_client, n_rows)
    train_dataset = Dataset.from_list(prompt_rows)

    _emit_progress(progress_callback, 0.30, "Initializing Weights & Biases")
    w_entity = os.environ.get("WANDB_ENTITY", "").strip()
    w_init: Dict[str, Any] = {
        "project": wandb_project,
        "name": os.environ.get("WANDB_RUN_NAME", "sqlsage-grpo-hf-space"),
    }
    if w_entity:
        w_init["entity"] = w_entity
    wandb.init(**w_init)

    def reward_from_env(prompts: List[str], completions: List[str], **_: Any) -> List[float]:
        """Compute GRPO reward by stepping the deployed SQLSage env."""
        rewards: List[float] = []
        for idx, completion in enumerate(completions):
            try:
                prompt_i = _align_prompt(idx, prompts)
                if use_plain or not prompt_to_seed:
                    seed = idx
                else:
                    seed = int(prompt_to_seed.get(prompt_i, idx))
                observation = env_client.reset(seed=seed)
                if not isinstance(observation, dict):
                    observation = {}
                action_candidates = list(observation.get("suggested_actions", []) or [])
                action, sql_query = _parse_action_and_sql(completion, action_candidates)
                _, reward, _, info = env_client.step(
                    action=action, rewritten_query=sql_query
                )
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
        per_device_train_batch_size=int(
            os.environ.get("SQLSAGE_PER_DEVICE_TRAIN_BATCH_SIZE", "2")
        ),
        gradient_accumulation_steps=int(
            os.environ.get("SQLSAGE_GRADIENT_ACCUMULATION_STEPS", "8")
        ),
        learning_rate=float(os.environ.get("SQLSAGE_LEARNING_RATE", "1e-5")),
        max_completion_length=int(
            os.environ.get("SQLSAGE_MAX_COMPLETION_LENGTH", "512")
        ),
        num_generations=int(os.environ.get("SQLSAGE_NUM_GENERATIONS", "8")),
        temperature=float(os.environ.get("SQLSAGE_GRPO_TEMPERATURE", "0.9")),
        kl_coeff=float(os.environ.get("SQLSAGE_KL_COEFF", "0.05")),
        num_train_epochs=int(os.environ.get("SQLSAGE_NUM_TRAIN_EPOCHS", "3")),
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
