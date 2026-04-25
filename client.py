"""OpenEnv client wrapper for SQLSage."""

from __future__ import annotations

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import SQLSageAction, SQLSageObservation


class SQLSageEnvClient(EnvClient[SQLSageAction, SQLSageObservation, State]):
    """Client for interacting with deployed SQLSage environments."""

    def _step_payload(self, action: SQLSageAction) -> Dict:
        return {"action": action.action, "rewritten_query": action.rewritten_query}

    def _parse_result(self, payload: Dict) -> StepResult[SQLSageObservation]:
        obs_data = payload.get("observation", {})
        observation = SQLSageObservation(
            original_query=obs_data.get("original_query", ""),
            explain_plan=obs_data.get("explain_plan", {}),
            execution_ms=obs_data.get("execution_ms", 0.0),
            result_hash=obs_data.get("result_hash", ""),
            result_row_count=obs_data.get("result_row_count", 0),
            schema_context=obs_data.get("schema_context", ""),
            previous_rewrites=obs_data.get("previous_rewrites", []),
            step_count=obs_data.get("step_count", 0),
            task_level=obs_data.get("task_level", 1),
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("observation", {}).get("step_count", 0),
        )
