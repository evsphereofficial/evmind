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

### Step 2 — Implementation + First Baseline Run (2026-08-12)

Implemented per spec §16:

- `src/config.py` — YAML config schema (all hyperparameters per §8).
- `src/tasks.py` — 5 tasks: horizontal, vertical, circle (r=0.55),
  diagonal, xor_quadrant (spec §3). No task-ID anywhere.
- `src/dataset.py` — procedural generation, independent train/eval seed
  families (spec §2), 10k train / 2k test per task.
- `src/model.py` — `TinyNumericTransformer`: per-coordinate linear projection
  (x -> embedding), learned positional embedding, 2-layer TransformerEncoder
  (d=32, heads=2, ff=64), mean pooling, 1-output head.
  **Parameter count: 17,249 (< 100K target).**
- `src/train.py` — plain `loss.backward()` / `optimizer.step()`, no replay,
  no freezing, no adapters (spec §4, §10). Hook insertion points noted for
  Phase 2 (before_update / after_forward / control_update / after_update,
  spec §17) but NO controller implemented.
- `src/evaluate.py`, `src/metrics.py` — eval loop + spec §6 forgetting math
  (F_i = best_accuracy_i - final_accuracy_i).
- `src/experiment.py` — orchestrates training Task i then evaluating ALL
  previously learned tasks (spec §4 protocol), writes all §9 outputs
  (run_config.json, task_accuracies.csv, forgetting.csv, training_log.csv,
  accuracy_matrix.png, forgetting_curve.png, final_model.pt), prints the §9
  summary, records parameter count / training time / latency / peak VRAM / RAM.

Commit `350b6d2` — implementation. Run #1 executed on CUDA, 12.1 s total.

**RESULT — Run #1 (seed=0, deterministic):**

Accuracy matrix (%):  (rows = tasks, cols = after training phase)

```
                After T1   After T2   After T3   After T4   After T5
Task 1 horiz       99.45      48.70      49.25      75.15      50.40
Task 2 vert                  99.50      48.85      74.05      48.50
Task 3 circle                         98.65      50.05      49.90
Task 4 diag                                  99.10      48.45
Task 5 xor                                          99.10
```

Forgetting:

| Task | Initial | Best | Final | Forgetting |
|------|--------:|-----:|------:|-----------:|
| horizontal | 99.45 | 99.45 | 50.40 | 49.05 |
| vertical | 99.50 | 99.50 | 48.50 | 51.00 |
| circle | 98.65 | 98.65 | 49.90 | 48.75 |
| diagonal | 99.10 | 99.10 | 48.45 | 50.65 |
| xor_quadrant | 99.10 | 99.10 | 99.10 | 0.00 |

- **Average forgetting: 39.89%** (49.86% over the 4 tasks that were actually
  overwritten; the 5th task trivially forgets 0 because training ends).
- **Final average accuracy: 59.27%** (≈ chance on 4 of 5 tasks).
- Parameters: 17,249. Total training time: ~12 s. Peak VRAM: 22.3 MB.
  Peak RAM: ~1.2 GB. Latency: ~0.0003 ms/sample.

Scientific notes (honest observations, no tuning):

1. **Catastrophic forgetting is strong and immediate** — single-task accuracy
   collapses from ~99% to ~49% (chance) after ONE subsequent task, far worse
   than the illustrative matrix in the spec (§5, which showed 99→81→63→51).
   This is because the tasks share no aligned features and the Transformer
   head/pooling re-wiring overwrites prior decision structure.
2. **Task 4 (diagonal) partially re-teaches tasks 1 & 2** (75.15 / 74.05).
   x1+x2>0 linearly combines the x1 and x2 features used by the horizontal
   and vertical tasks, so diagonal training partially re-instantiates them —
   classic interference, not a mechanism, but a useful baseline detail.
3. **Task 5 (XOR) erases everything again** — XOR requires sign-combination
   features that rewrite the earlier ones; all four old tasks drop to chance.
4. The last task always shows 0 forgetting by construction (nothing trained
   after it); the meaningful number for the future HRM comparison is
   **~49.9% average forgetting over overwritten tasks**.

Rerun with identical seed produced identical numbers → baseline reproducible.

### Baseline Status

**Phase 1 baseline is complete and reproducible.** Ordinary single-node live
training of a 17K-parameter Transformer on 5 sequential synthetic tasks:
each task is learned to ~99% and then **destroyed to chance (~49%) by the next
task**. The catastrophic-forgetting failure mode is confirmed, honest, and
unmistakable. This is the number Phase 2 (HRM-inspired controller) must beat.

### Step 3 — To Do (Phase 2, RESERVED, DO NOT START)

- HRM-inspired recursive controller (spec §11, §17): investigate
  weight-update gates / masks / gradient scaling / block-level freeze,
  and test empirically whether parameter-level control is needed or
  layer/block/feature-level control suffices.
- Compare: ordinary training vs controlled live training (§12).
- Do not proceed to larger EvMind experiments until the retention question
  is understood (§12).

---

## Key Design Decisions (living list)

| # | Decision | Rationale | Date |
|---|----------|-----------|------|
| D1 | Use uv instead of system pip | rootless env setup | 2026-08-12 |
| D2 | Run on CUDA (RTX 4070) | available hardware; tiny model anyway | 2026-08-12 |
| D3 | Fixed seeds for reproducibility | spec §8: one deterministic seed first | 2026-08-12 |

---

## Results (filled after first run)

Full results in `results/`. Summary (Run #1, seed=0, deterministic, reproducible):

- Accuracy matrix, per-task forgetting, plots: see `results/`.
- **Average forgetting: 39.89%** (49.86% over overwritten tasks 1–4).
- **Final average accuracy: 59.27%.**
- All four overwritten tasks end at chance (~49%), each was ~99% after its
  own training phase.
- Confirmed: catastrophic forgetting is immediate and severe for this
  single-node baseline. See journal Step 2 for the full matrix and notes.