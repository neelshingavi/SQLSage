"""Task sets for SQLSage curriculum."""

from .level1 import LEVEL1_TASKS, Task
from .level2 import LEVEL2_TASKS
from .level3 import LEVEL3_TASKS

ALL_TASKS = [*LEVEL1_TASKS, *LEVEL2_TASKS, *LEVEL3_TASKS]

__all__ = ["Task", "LEVEL1_TASKS", "LEVEL2_TASKS", "LEVEL3_TASKS", "ALL_TASKS"]
