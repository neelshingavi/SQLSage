"""SQLSage OpenEnv package."""

from .tasks import tasks_for_levels
from .tpch import CURRICULUM, SCHEMA_OVERVIEW_SF1

try:
    from .env import Observation, SQLSageEnv
except Exception:  # pragma: no cover - depends on optional runtime extras
    Observation = None  # type: ignore[assignment]
    SQLSageEnv = None  # type: ignore[assignment]

try:
    # Optional import to keep lightweight modules (e.g., reward tests) usable
    # even when DB extras like psycopg2 are not installed.
    from .dataset import build_records, make_prompt
except Exception:  # pragma: no cover - depends on optional runtime extras
    build_records = None  # type: ignore[assignment]
    make_prompt = None  # type: ignore[assignment]

__all__ = [
    "Observation",
    "SQLSageEnv",
    "build_records",
    "make_prompt",
    "SCHEMA_OVERVIEW_SF1",
    "CURRICULUM",
    "tasks_for_levels",
]
