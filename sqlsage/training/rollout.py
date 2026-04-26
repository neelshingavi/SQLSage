"""Rollout helpers bridging model output and environment step()."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any


def observation_to_dict(observation: Any) -> dict[str, Any]:
    """Normalize dataclass or Pydantic observations for logging / prompts."""
    if hasattr(observation, "model_dump"):
        return observation.model_dump()
    if is_dataclass(observation):
        return asdict(observation)
    if isinstance(observation, dict):
        return observation
    raise TypeError(f"unsupported observation type: {type(observation)!r}")


def parse_model_json(raw_text: str) -> dict[str, str]:
    parsed = json.loads(raw_text)
    action = parsed.get("action", "").strip()
    rewritten_query = parsed.get("rewritten_query", "").strip()
    if not action or not rewritten_query:
        raise ValueError("model_output_missing_action_or_query")
    return {"action": action, "rewritten_query": rewritten_query}


def rollout_episode(env: Any, model_generate: Any, max_steps: int = 5) -> list[dict[str, Any]]:
    """
    Run one episode and collect rollout tuples.
    model_generate is a callable that receives observation dict and returns JSON text.
    """
    obs = env.reset()
    trajectory: list[dict[str, Any]] = []

    for _ in range(max_steps):
        raw = model_generate(observation_to_dict(obs))
        action_payload = parse_model_json(raw)
        next_obs, reward, done, info = env.step(
            action_payload["action"], action_payload["rewritten_query"]
        )
        trajectory.append(
            {
                "observation": observation_to_dict(obs),
                "action": action_payload,
                "reward": reward,
                "done": done,
                "info": info,
            }
        )
        obs = next_obs
        if done:
            break

    return trajectory
