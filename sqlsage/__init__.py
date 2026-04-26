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

try:
    from .rewrite_patterns import (
        ALL_PATTERNS,
        RewritePattern,
        detect_applicable_patterns,
        format_few_shot_for_prompt,
        get_pattern_by_id,
        get_patterns_for_query,
    )
except Exception:  # pragma: no cover
    ALL_PATTERNS = ()  # type: ignore[assignment, misc]
    RewritePattern = None  # type: ignore[assignment, misc]
    detect_applicable_patterns = None  # type: ignore[assignment, misc]
    format_few_shot_for_prompt = None  # type: ignore[assignment, misc]
    get_pattern_by_id = None  # type: ignore[assignment, misc]
    get_patterns_for_query = None  # type: ignore[assignment, misc]

__all__ = [
    "ALL_PATTERNS",
    "Observation",
    "SQLSageEnv",
    "RewritePattern",
    "build_records",
    "make_prompt",
    "detect_applicable_patterns",
    "format_few_shot_for_prompt",
    "get_pattern_by_id",
    "get_patterns_for_query",
    "SCHEMA_OVERVIEW_SF1",
    "CURRICULUM",
    "tasks_for_levels",
]
