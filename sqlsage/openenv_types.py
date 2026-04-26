"""OpenEnv Pydantic action/observation types (HTTP/WebSocket API contract)."""

from __future__ import annotations

from typing import Any

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class SQLSageStepAction(Action):
    """Agent step: rewrite kind + SQL text (matches reference JSON shape)."""

    action: str = Field(..., description="rewrite_join | add_cte | push_filter | ...")
    rewritten_query: str = Field(..., description="Rewritten SQL")


class SQLSageServerObservation(Observation):
    """Observation returned over OpenEnv HTTP/WS (extends base reward/done/metadata)."""

    original_query: str = ""
    explain_plan: dict[str, Any] = Field(default_factory=dict)
    execution_ms: float = 0.0
    result_hash: str = ""
    result_row_count: int = 0
    schema_context: str = ""
    previous_rewrites: list[str] = Field(default_factory=list)
    step_count: int = 0
    task_level: int = 1
