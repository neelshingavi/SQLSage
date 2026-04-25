"""Rollout helpers bridging model output and environment step()."""

from __future__ import annotations

import json
from typing import Any


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
        raw = model_generate(obs)
        action_payload = parse_model_json(raw)
        next_obs, reward, done, info = env.step(
            action_payload["action"], action_payload["rewritten_query"]
        )
        trajectory.append(
            {
                "observation": obs,
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
