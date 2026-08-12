"""Procedural dataset generation for the Phase 1 baseline.

No external data is downloaded. Fresh train/test samples are generated on
demand from independent random seeds so training and evaluation never share
samples, and the experiment has an effectively unlimited data supply.
"""

from __future__ import annotations

import torch
from torch.utils.data import TensorDataset

from .tasks import Task


def generate_dataset(
    task: Task,
    n_samples: int,
    base_seed: int,
    eval_split: bool,
    lo: float = -1.0,
    hi: float = 1.0,
) -> TensorDataset:
    """Generate a dataset for a task with a deterministic, independent seed.

    Args:
        task: the task whose boundary defines the labels.
        n_samples: number of samples.
        base_seed: experiment seed.
        eval_split: True -> evaluation split (independent seed family),
                    False -> training split.
        lo, hi: uniform sampling range for x1, x2.
    """
    # Independent seed families for train and eval (spec: generator must use
    # independent random seeds for training and evaluation).
    seed = base_seed * 10_000 + (10_000 if eval_split else 0) + n_samples % 1000
    gen = torch.Generator().manual_seed(seed)

    x = torch.rand(n_samples, 2, generator=gen) * (hi - lo) + lo
    x1, x2 = x[:, 0], x[:, 1]
    y = task.labels(x1, x2)
    return TensorDataset(x, y)


def scale_means(x: torch.Tensor) -> torch.Tensor:
    """Helper: report mean/std of a generated split (used in sanity checks)."""
    return x.mean(dim=0), x.std(dim=0)