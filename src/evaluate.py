"""Evaluation loop for the Phase 1 baseline."""

from __future__ import annotations

import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from .train import prepare_batch


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module = nn.BCEWithLogitsLoss(),
) -> tuple[float, float, float]:
    """Evaluate a model on a data loader.

    Returns:
        (accuracy %, mean loss, mean inference latency in ms per sample)
    """
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    latencies = []

    with torch.no_grad():
        for x, y in loader:
            x, y = prepare_batch(x, y, device)

            start = time.perf_counter()
            logits = model(x)
            latencies.append(time.perf_counter() - start)

            loss = loss_fn(logits, y)
            total_loss += loss.item() * x.size(0)

            preds = (torch.sigmoid(logits) >= 0.5).long()
            correct += (preds == y.long()).sum().item()
            total += x.size(0)

    accuracy = 100.0 * correct / total if total else 0.0
    mean_loss = total_loss / total if total else 0.0
    # Latency per single sample (a batch is processed at once).
    per_sample_ms = 1000.0 * (sum(latencies) / len(latencies)) / batch_size_of(loader)
    return accuracy, mean_loss, per_sample_ms


def batch_size_of(loader: DataLoader) -> int:
    return loader.batch_size if loader.batch_size else 1