"""OpenEnv client models for SQLSage."""

from __future__ import annotations

from pydantic import BaseModel


class SQLSageAction(BaseModel):
    action: str
    rewritten_query: str


class SQLSageObservation(BaseModel):
    original_query: str = ""
    explain_plan: dict = {}
    execution_ms: float = 0.0
    result_hash: str = ""
    result_row_count: int = 0
    schema_context: str = ""
    previous_rewrites: list[str] = []
    step_count: int = 0
    task_level: int = 1
    done: bool = False
    reward: float | None = None
