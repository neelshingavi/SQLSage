"""Meta OpenEnv Environment adapter over the gym-style SQLSageEnv."""

from __future__ import annotations

import threading
from dataclasses import asdict
from typing import Any, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata, State

from .env import Observation as CoreObservation
from .env import SQLSageEnv
from .openenv_types import SQLSageServerObservation, SQLSageStepAction


class SQLSageOpenEnvironment(Environment[SQLSageStepAction, SQLSageServerObservation, State]):
    """OpenEnv Environment base class wrapping SQLSageEnv for EnvClient / HTTPEnvServer."""

    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self, inner: SQLSageEnv | None = None) -> None:
        super().__init__(transform=None, rubric=None)
        self._inner = inner or SQLSageEnv()
        # Uvicorn may call reset/step/state from different threads; psycopg2 is not thread-safe.
        self._lock = threading.RLock()

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="SQLSageOpenEnvironment",
            description="RL environment for PostgreSQL query optimization via EXPLAIN plans",
            version="0.1.0",
        )

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SQLSageServerObservation:
        self._reset_rubric()
        with self._lock:
            core_obs = self._inner.reset(seed=seed)
        return self._to_server_obs(core_obs, reward=None, done=False, info={})

    def step(
        self,
        action: SQLSageStepAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> SQLSageServerObservation:
        with self._lock:
            core_obs, reward, done, info = self._inner.step(action.action, action.rewritten_query)
        return self._to_server_obs(core_obs, reward=reward, done=done, info=info)

    @property
    def state(self) -> State:
        with self._lock:
            core = self._inner.state()
            return State(episode_id=None, step_count=core.step_count)

    def close(self) -> None:
        """HTTPEnvServer calls close() after each HTTP request; keep the DB handle for singleton use."""
        return

    def shutdown(self) -> None:
        """Release the PostgreSQL connection (process shutdown or tests)."""
        with self._lock:
            self._inner.close()

    @staticmethod
    def _to_server_obs(
        core: CoreObservation,
        reward: float | None,
        done: bool,
        info: dict[str, Any],
    ) -> SQLSageServerObservation:
        d = asdict(core)
        return SQLSageServerObservation(
            original_query=d["original_query"],
            explain_plan=d["explain_plan"],
            execution_ms=d["execution_ms"],
            result_hash=d["result_hash"],
            result_row_count=d["result_row_count"],
            schema_context=d["schema_context"],
            previous_rewrites=list(d["previous_rewrites"]),
            step_count=d["step_count"],
            task_level=d["task_level"],
            reward=reward,
            done=done,
            metadata=dict(info) if info else {},
        )
