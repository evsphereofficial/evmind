# EvMind — Phase 1 Baseline: Single-Node Live Learning Retention

This repository contains the **first experimental prototype** for the EvMind
continual-learning hypothesis (see `architecture_v2.md`).

This is **NOT** the full EvMind architecture. No nodes, no routing, no memory,
no tools, no multimodal, no HRM, no diffusion, no LLMs.

## The Experiment

Determine **how much a single small Transformer classifier forgets** when it is
continuously trained on a sequence of five different binary tasks without
resetting its weights.

One model, one parameter space, one continual training stream. No replay, no
freezing, no task-ID, no adapters, no protection mechanisms.

The result is the **baseline** that the future HRM-inspired recursive controller
(Phase 2, reserved) must beat.

## Setup

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
```

## Run

```bash
.venv/bin/python -m src.experiment --config configs/baseline.yaml
```

Results are written to `results/`:

```
results/
├── run_config.json       # full config + env + hardware measurements
├── task_accuracies.csv   # accuracy matrix: task x after-training-phase
├── forgetting.csv        # per-task initial/best/final accuracy + forgetting
├── training_log.csv      # per-epoch loss/accuracy
├── accuracy_matrix.png   # heatmap
├── forgetting_curve.png  # task retention vs training step
└── final_model.pt        # final weights
```

## Project Structure

```
src/
├── config.py      # YAML config loading (all hyperparameters live in configs/)
├── tasks.py       # 5 task definitions (boundary functions over [-1,1]^2)
├── dataset.py     # procedural dataset generator (unlimited fresh samples)
├── model.py       # tiny continuous numeric Transformer (<100K params)
├── train.py       # single-epoch training routine (plain backward/step)
├── evaluate.py    # evaluation loop with latency measurement
├── metrics.py     # forgetting / retention calculations
└── experiment.py  # continual orchestration, outputs, plots, summary
configs/
└── baseline.yaml  # all tunable hyperparameters
```

Design note: `model.py` is deliberately decoupled from the training loop so
that Phase 2 can wrap the model in an HRM controller (`before_update`,
`after_forward`, `control_update`, `after_update`) without rewriting the
experiment framework.

See `MASTER.md` for the running journal of every step and result.