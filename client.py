"""OpenEnv client wrapper for SQLSage."""

from __future__ import annotations

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import SQLSageAction, SQLSageObservation


class SQLSageEnvClient(EnvClient[SQLSageAction, SQLSageObservation, State]):
    """Client for interacting with deployed SQLSage environments over WebSocket."""

    def _step_payload(self, action: SQLSageAction) -> Dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[SQLSageObservation]:
        obs_data = dict(payload.get("observation", {}))
        obs_data["done"] = payload.get("done", False)
        obs_data["reward"] = payload.get("reward")
        observation = SQLSageObservation.model_validate(obs_data)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        return State.model_validate(payload)
