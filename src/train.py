"""Training routine for the Phase 1 baseline.

The baseline performs plain, unrestricted gradient updates:

    loss.backward()
    optimizer.step()

No replay, no freezing, no adapters, no task-id. This is ordinary live weight
modification. The loop is written so that a future HRM controller can be
inserted (controller.before_update / after_forward / control_update /
after_update) without changing the experiment framework.
"""

from __future__ import annotations

import time

import torch
from torch import nn
from torch.utils.data import DataLoader


def prepare_batch(x: torch.Tensor, y: torch.Tensor, device: torch.device):
    """Move a batch to the compute device with the right dtypes."""
    return x.to(device).float(), y.to(device).float()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """Train for a single epoch.

    Returns:
        (mean loss, accuracy %, wall seconds).
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    start = time.perf_counter()

    for x, y in loader:
        x, y = prepare_batch(x, y, device)

        logits = model(x)          # after_forward hook insertion point
        loss = loss_fn(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # control_update hook insertion point (gradient scaling, masking...)
        optimizer.step()           # before_update/after_update hook points

        total_loss += loss.item() * x.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).long()
        correct += (preds == y.long()).sum().item()
        total += x.size(0)

    elapsed = time.perf_counter() - start
    mean_loss = total_loss / total if total else 0.0
    accuracy = 100.0 * correct / total if total else 0.0
    return mean_loss, accuracy, elapsed