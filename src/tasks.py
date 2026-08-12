"""Continual-learning task definitions for the Phase 1 baseline.

Each task is a ground-truth binary function over x1, x2 in [-1, 1].
The functions are chosen so their decision boundaries differ maximally,
forcing genuine weight re-wiring under sequential training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

LabelFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class Task:
    """A task: name + label function + optional parameters."""
    name: str
    label_fn: LabelFn
    params: dict = None

    def labels(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Binary labels (float 0/1) for the given coordinates."""
        return self.label_fn(x1, x2).float()


def _horizontal(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Task 1: label = 1 if x2 > 0 else 0. Boundary: y = 0."""
    return (x2 > 0).long()


def _vertical(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Task 2: label = 1 if x1 > 0 else 0. Boundary: x = 0."""
    return (x1 > 0).long()


def _circle(x1: torch.Tensor, x2: torch.Tensor, r: float = 0.55) -> torch.Tensor:
    """Task 3: label = 1 if x1^2 + x2^2 > r^2 else 0. Boundary: circle."""
    return (x1 ** 2 + x2 ** 2 > r * r).long()


def _diagonal(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Task 4: label = 1 if x1 + x2 > 0 else 0. Boundary: diagonal line."""
    return (x1 + x2 > 0).long()


def _xor(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Task 5: label = 1 if (x1 > 0) != (x2 > 0) else 0 (quadrants II+IV)."""
    return ((x1 > 0) != (x2 > 0)).long()


# Registry mapping config task-type strings to constructors.
# A constructor takes a params dict and returns a Task.
TASK_TYPES: dict[str, Callable[[dict], Task]] = {
    "horizontal": lambda p: Task("horizontal", _horizontal, p),
    "vertical": lambda p: Task("vertical", _vertical, p),
    "circle": lambda p: Task(
        "circle",
        lambda x1, x2, r=float(p.get("r", 0.55)): _circle(x1, x2, r),
        p,
    ),
    "diagonal": lambda p: Task("diagonal", _diagonal, p),
    "xor": lambda p: Task("xor", _xor, p),
}


def build_tasks(task_configs) -> list[Task]:
    """Build Task objects from config dataclasses (src/config.py)."""
    tasks = []
    for tc in task_configs:
        if tc.type not in TASK_TYPES:
            raise ValueError(
                f"unknown task type {tc.type!r}; known: {sorted(TASK_TYPES)}"
            )
        task = TASK_TYPES[tc.type](tc.params or {})
        # Prefer the name given in the config file.
        tasks.append(Task(name=tc.name, label_fn=task.label_fn, params=task.params))
    return tasks