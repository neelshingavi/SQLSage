"""Task sets for SQLSage curriculum."""

from .level1 import LEVEL1_TASKS, Task
from .level2 import LEVEL2_TASKS
from .level3 import LEVEL3_TASKS

ALL_TASKS = [*LEVEL1_TASKS, *LEVEL2_TASKS, *LEVEL3_TASKS]


def tasks_for_levels(*levels: int) -> list[Task]:
    """Filter curriculum tasks by level (1, 2, and/or 3) for staged training or stress runs."""
    wanted = set(levels)
    return [t for t in ALL_TASKS if t.level in wanted]


__all__ = [
    "Task",
    "LEVEL1_TASKS",
    "LEVEL2_TASKS",
    "LEVEL3_TASKS",
    "ALL_TASKS",
    "tasks_for_levels",
]
