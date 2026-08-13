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
class GovernorConfig:
    """HRM intent governor (Phase 2) settings."""
    hidden_dim: int = 24        # hidden width of the gate network
    refine_steps: int = 2       # recursive refinement passes (§132.4)
    init_mask: float = 0.5      # starting gate value (neutral; sparse cost pushes down)
    granularity: str = "weight"  # "weight" = one gate per parameter (17,249 gates);
                                 # "module" = one gate per parameter tensor
    close_threshold: float = 0.02  # FIX 2: masks below this = closed weight
                                   # NODES (Adam moments zeroed so momentum
                                   # cannot keep forcing the node open)


@dataclass
class MetaConfig:
    """Meta-pretraining of the intent network (trained first, then frozen).

    Objective (parameter modification is itself expensive):
        L = L_new + lambda_old*L_old
            + lambda_sparse*(mean(M)-sparse_target)^2   # selective, not zero
            + lambda_delta*mean(|dW|/|W|)
    """
    steps: int = 200            # governor optimizer updates
    meta_batch: int = 8         # parallel meta-steps per update (GPU utilization)
    batch_size: int = 512       # meta task batch
    warmup_batches: int = 12    # plain updates on task A so it is well installed
    warmup_lr: float = 2e-3     # warmup lr (faster knowledge install than burst lr)
    burst_steps: int = 3        # SHORT burst: 20-step unrolls are chaotic
                                # (loss cliffs ~6 orders steeper than the
                                # FOMAML gradient predicted); ~3 steps make
                                # the trajectory smooth and gradients valid
    lr: float = 1e-3            # governor optimizer learning rate
    burst_optim: str = "adamw"  # burst update semantics: "adamw" = stateless
                                # AdamW matching the live stream's scale_update
                                # (mask scales AdamW's normalized delta; this is
                                # the transfer fix); "sgd" = legacy masked grad
    burst_lr: float = 3e-2      # masked update scale during the B burst if
                                # burst_optim=sgd; for adamw the burst uses the
                                # LIVE lr (meta.lr must equal the train lr)
    close_threshold: float = 0.02  # hard-close masks below this in the burst
                                # (mirrors governor.close_threshold in the
                                # live stream: zero movement + zero Adam
                                # moments for closed weight nodes)
    lambda_old: float = 1.0     # weight of old-task degradation penalty
    lambda_sparse: float = 1.0  # weight of (mean(M)-sparse_target)^2
    sparse_target: float = 0.3  # desired fraction of open gate mass
    lambda_ewc: float = 0.05    # vestigial; selectivity now comes from the
                                # gradient-relationship features (cos gA.gB,
                                # conflict magnitude, free-capacity ratio)
                                # + short-horizon unroll, not EWC pressure
    lambda_delta: float = 0.5   # weight of mean relative |dW| (change cost)
    distill_steps: int = 0      # node-level pretraining: regress the cell's
                                # region gates onto the frozen governor's
                                # per-weight masks (teacher distillation)
    distill_lr: float = 1e-2    # distillation optimizer learning rate
    second_order: bool = False  # short-horizon FOMAML is valid & stable here
    eval_pairs: int = 24        # paired gated-vs-ungated sanity check pairs
    old_tasks_max: int = 2      # persistent old-knowledge memory depth:
                                # 0..N prior tasks installed per meta step,
                                # each captured into a SensitivityMemory
    memory_agg: str = "ewc"     # importance aggregation across old tasks:
                                # ewc=mean g^2 (Fisher), max=|g| running max,
                                # recency=recency-weighted mean, ema=EMA


@dataclass
class RairawConfig:
    """RAIRAW-V1 (Experiment Series 2) settings.

    RAIRAW-V1 transforms the adaptive unit from an undifferentiated scalar
    weight into a small recursive model (R / A / I / Controller) holding
    local authority over a bounded region of main-model weights. The HRM
    governor keeps global authority (allocation); H_MEM records RAIRAW
    influence feedback for future allocation.
    """
    region_size: int = 1000      # max main weights per RAIRAW region (Exp 2.2)
    max_rairaw: int = 20         # bounded RAIRAW pool (Exp 2.3)
    h_dim: int = 16              # controller recurrent state dim (Exp 2.1)
    intent_dim: int = 4          # intent vocabulary: retain/learn/adapt/protect
    hmem_alpha: float = 0.5      # H_MEM influence EMA rate
    alloc_blend: float = 0.7     # HRM mask vs H_MEM prior in allocation
    sparse_target: float = 0.3   # allocation target: fraction of open mass
    close_threshold: float = 0.02  # gates below this = closed nodes


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""
    run_name: str = "baseline"
    train_samples: int = 10_000
    test_samples: int = 2_000
    tasks: list[TaskConfig] = field(default_factory=list)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    governor: GovernorConfig = field(default_factory=GovernorConfig)
    meta: MetaConfig = field(default_factory=MetaConfig)
    hmem_mode: str = "none"  # input-driven weight-influence channel:
                             # none | grad | random | shuffled | magnitude
    raira: RairawConfig = field(default_factory=RairawConfig)

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
    governor = GovernorConfig(**raw.get("governor", {}))
    meta = MetaConfig(**raw.get("meta", {}))
    raira = RairawConfig(**raw.get("raira", {}))

    return ExperimentConfig(
        run_name=str(raw.get("run_name", "baseline")),
        train_samples=int(raw.get("train_samples", 10_000)),
        test_samples=int(raw.get("test_samples", 2_000)),
        tasks=_build_tasks(raw.get("tasks", [])),
        model=model,
        train=train,
        governor=governor,
        meta=meta,
        raira=raira,
        hmem_mode=str(raw.get("hmem", {}).get("mode", "none")),
    )