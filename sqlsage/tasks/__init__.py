"""Task definitions for SQLSage TPC-H curriculum."""

from dataclasses import dataclass


@dataclass
class Task:
    """Single SQL optimization task."""

    query: str
    target_ms: float
    level: int
    description: str


def _all_tasks() -> list[Task]:
    """Load all level task lists lazily to avoid import cycles."""
    from .level1 import LEVEL1_TASKS
    from .level2 import LEVEL2_TASKS
    from .level3 import LEVEL3_TASKS

    return [*LEVEL1_TASKS, *LEVEL2_TASKS, *LEVEL3_TASKS]


def tasks_for_levels(*levels: int) -> list[Task]:
    """Filter tasks by one or more curriculum levels."""
    wanted = set(levels)
    return [task for task in _all_tasks() if task.level in wanted]


ALL_TASKS = _all_tasks()

__all__ = ["Task", "ALL_TASKS", "tasks_for_levels"]
