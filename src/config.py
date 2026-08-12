"""Configuration loading for the EvMind Phase 1 baseline experiment.

All hyperparameters are configurable from a single YAML file (configs/).
This module defines typed config dataclasses and a loader that validates
the YAML contents.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class ModelConfig:
    """Architecture of the tiny numeric Transformer."""
    input_dim: int = 2          # number of numeric coordinates
    seq_len: int = 2            # sequence length (one token per coordinate)
    embedding_dim: int = 32     # d_model
    num_layers: int = 2         # Transformer encoder layers
    num_heads: int = 2          # attention heads
    ff_dim: int = 64            # feed-forward dimension
    dropout: float = 0.0        # dropout probability (baseline: none)


@dataclass
class TaskConfig:
    """A single continual-learning task definition."""
    name: str
    type: str                   # task type, see src/tasks.py
    params: dict[str, float] = field(default_factory=dict)  # e.g. r for circle


@dataclass
class TrainConfig:
    """Optimization settings."""
    seed: int = 0
    learning_rate: float = 1e-3
    batch_size: int = 128
    epochs_per_task: int = 5
    optimizer: str = "adamw"
    weight_decay: float = 0.0
    num_workers: int = 0


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""
    run_name: str = "baseline"
    train_samples: int = 10_000
    test_samples: int = 2_000
    tasks: list[TaskConfig] = field(default_factory=list)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @property
    def num_tasks(self) -> int:
        return len(self.tasks)

    def to_dict(self) -> dict[str, Any]:
        """Serializable dict (for run_config.json)."""
        return dataclasses.asdict(self)


def _build_tasks(raw_tasks: list[dict[str, Any]]) -> list[TaskConfig]:
    if not raw_tasks:
        raise ValueError("config must define at least one task")
    tasks = []
    for raw in raw_tasks:
        if not isinstance(raw, dict) or "name" not in raw or "type" not in raw:
            raise ValueError(f"invalid task definition: {raw!r}")
        tasks.append(
            TaskConfig(
                name=str(raw["name"]),
                type=str(raw["type"]),
                params={str(k): float(v) for k, v in raw.get("params", {}).items()},
            )
        )
    return tasks


def load_config(path: str) -> ExperimentConfig:
    """Load and validate a YAML experiment configuration."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"config file {path} must contain a YAML mapping")

    model = ModelConfig(**raw.get("model", {}))
    train = TrainConfig(**raw.get("train", {}))

    return ExperimentConfig(
        run_name=str(raw.get("run_name", "baseline")),
        train_samples=int(raw.get("train_samples", 10_000)),
        test_samples=int(raw.get("test_samples", 2_000)),
        tasks=_build_tasks(raw.get("tasks", [])),
        model=model,
        train=train,
    )