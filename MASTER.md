# MASTER.md — EvMind Live-Learning Experiment Journal

> Master journal for the EvMind research program.
> Phase 1 (current): Single-node live-learning retention baseline.
> Phase 2 (reserved): HRM-inspired recursive controller. NOT implemented yet.
> Phase 3 (reserved): Controlled vs ordinary training comparison.

---

## Current Status (updated at every step)

- **Phase:** 1 (baseline)
- **Objective:** Measure how much a single small Transformer classifier forgets
  when continuously trained on a sequence of different tasks without resetting
  weights, no replay, no freezing, no adapters, no task-ID, no protection.
- **Do NOT build:** nodes, routing, memory, tools, multimodal, curiosity,
  HRM, diffusion, external datasets, LLMs.
- **Scope rule:** ONE model, ONE parameter space, ONE continual training stream.

---

## The Experimental Question

> Can an ordinary small neural model retain old knowledge while learning new
> knowledge online (live, single stream, no replay)?

This baseline is the number that the future HRM/controller experiment must beat:

```
Baseline:      Average forgetting = ?  (this experiment measures it)
HRM-controlled: Average forgetting = ? (must beat baseline to be meaningful)
```

## Scientific Rule

Do not attempt to make the baseline look good. Honest measurement of the
failure mode only. No mechanisms to improve retention until the baseline is
complete and reproducible.

---

## Journal

### Step 0 — Environment Setup (2026-08-12)

- Working dir: `/home/evsphere/projects/aiprojects/evmind`
- Repo exists, `architecture_v2.md` + `.git` already present, clean tree.
- System: Linux, 16 cores, 7.7 GB RAM, NVIDIA RTX 4070 12 GB (driver 596.36, CUDA 13.2).
- No system `pip`/`venv` for python3.12 → installed `uv` 0.12.3 to `~/.local/bin`.
- Created `.venv` with uv (Python 3.12.3), installed:
  - `torch==2.11.0+cu128` (CUDA works, device name confirmed)
  - `pyyaml`, `matplotlib`, `pandas`, `numpy`
- Created `.gitignore` (excludes `.venv/`, caches) and `requirements.txt`.

**Decision:** Use uv for env management (only rootless option available).
GPU available → experiments will run on CUDA.

### Step 1 — Experiment Plan Committed (2026-08-12)

- Created project docs: `README.md`, `MASTER.md` (this journal).
- Experiment structure per spec §16:

```
src/
├── config.py      # YAML config load/validation
├── tasks.py       # task defs (5 boundary functions)
├── dataset.py     # procedural generator (independent train/test seeds)
├── model.py       # tiny continuous numeric Transformer
├── train.py       # one training epoch (plain backward/step)
├── evaluate.py    # eval loop + latency
├── metrics.py     # forgetting math
└── experiment.py  # continual orchestration + outputs + plots
configs/baseline.yaml
results/           # generated artifacts
```

### Step 2 — Config + Data Layer (next)

(Log the implementation + any decisions here.)

---

## Key Design Decisions (living list)

| # | Decision | Rationale | Date |
|---|----------|-----------|------|
| D1 | Use uv instead of system pip | rootless env setup | 2026-08-12 |
| D2 | Run on CUDA (RTX 4070) | available hardware; tiny model anyway | 2026-08-12 |
| D3 | Fixed seeds for reproducibility | spec §8: one deterministic seed first | 2026-08-12 |

---

## Results (filled after first run)

Placeholder — will be replaced by the computed accuracy matrix, forgetting
table, and summary output of the first baseline run.