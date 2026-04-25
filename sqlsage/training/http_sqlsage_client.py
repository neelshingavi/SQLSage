"""Minimal synchronous HTTP client for the SQLSage OpenEnv FastAPI server."""

from __future__ import annotations

from typing import Any, Callable

import requests


class SQLSageHTTPError(RuntimeError):
    pass


def _url(base: str, path: str) -> str:
    b = base.rstrip("/")
    return f"{b}{path}"


def http_reset(base_url: str, seed: int | None = None, timeout_s: float = 300.0) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if seed is not None:
        payload["seed"] = seed
    r = requests.post(_url(base_url, "/reset"), json=payload, timeout=timeout_s)
    if not r.ok:
        raise SQLSageHTTPError(f"reset failed: {r.status_code} {r.text[:500]}")
    data = r.json()
    return dict(data.get("observation", {}))


def http_step(
    base_url: str,
    action: str,
    rewritten_query: str,
    timeout_s: float = 600.0,
) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
    body = {"action": {"action": action, "rewritten_query": rewritten_query}}
    r = requests.post(_url(base_url, "/step"), json=body, timeout=timeout_s)
    if not r.ok:
        raise SQLSageHTTPError(f"step failed: {r.status_code} {r.text[:500]}")
    data = r.json()
    obs = dict(data.get("observation", {}))
    reward = float(data.get("reward") or 0.0)
    done = bool(data.get("done", False))
    info = dict(data.get("info") or {})
    return obs, reward, done, info


def run_episode_http(
    base_url: str,
    policy: Callable[[dict[str, Any]], tuple[str, str]],
    max_steps: int = 5,
    step_timeout_s: float = 600.0,
    seed: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Run one episode. ``policy`` maps observation -> (action_name, rewritten_sql).

    Returns ``(initial_observation, trajectory)`` where trajectory entries are:
    ``{obs_before, action, rewritten_query, observation, reward, done, info}``.
    """
    obs0 = http_reset(base_url, seed=seed)
    trajectory: list[dict[str, Any]] = []
    obs = obs0
    for _ in range(max_steps):
        act, q = policy(obs)
        obs_before = dict(obs)
        obs2, reward, done, info = http_step(base_url, act, q, timeout_s=step_timeout_s)
        trajectory.append(
            {
                "obs_before": obs_before,
                "action": act,
                "rewritten_query": q,
                "observation": obs2,
                "reward": reward,
                "done": done,
                "info": info,
            }
        )
        obs = obs2
        if done:
            break
    return obs0, trajectory
