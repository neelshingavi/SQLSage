"""FastAPI application for SQLSage (Meta OpenEnv HTTP/WebSocket API)."""

from __future__ import annotations

import atexit

from openenv.core.env_server.http_server import create_fastapi_app

from .openenv_bridge import SQLSageOpenEnvironment
from .openenv_types import SQLSageServerObservation, SQLSageStepAction

_singleton: SQLSageOpenEnvironment | None = None


def _sqlsage_factory() -> SQLSageOpenEnvironment:
    """Single long-lived env instance (HTTP handlers call close() after each request)."""
    global _singleton
    if _singleton is None:
        _singleton = SQLSageOpenEnvironment()
    return _singleton


def _shutdown_sqlsage() -> None:
    global _singleton
    if _singleton is not None:
        _singleton.shutdown()
        _singleton = None


atexit.register(_shutdown_sqlsage)

app = create_fastapi_app(
    _sqlsage_factory,
    SQLSageStepAction,
    SQLSageServerObservation,
)
