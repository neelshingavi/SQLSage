"""OpenEnv client models for SQLSage (aliases for shared server types)."""

from __future__ import annotations

from sqlsage.openenv_types import SQLSageServerObservation, SQLSageStepAction

# Backward-compatible names for client code
SQLSageAction = SQLSageStepAction
SQLSageObservation = SQLSageServerObservation

__all__ = ["SQLSageAction", "SQLSageObservation", "SQLSageStepAction", "SQLSageServerObservation"]
