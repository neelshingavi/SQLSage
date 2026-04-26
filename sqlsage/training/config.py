"""Reference GRPO configuration values from section 6."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GRPOConfigValues:
    output_dir: str = "./sqlsage-grpo"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    max_completion_length: int = 512
    num_generations: int = 8
    report_to: str = "wandb"


def default_grpo_config() -> dict:
    """Return config as a plain dict to keep this module dependency-light."""
    return asdict(GRPOConfigValues())
