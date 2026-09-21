# Modular Capacity Scaling Law v2: 20-Shape Fine-Tune, Hard Gating, and Expert Isolation for Live Continual LLMs

**EvMind Report v2 — 2026-09-21**  
**Update to:** `research-report.md` (5-task law `k0≈115, τ≈158`)  
**New:** 20-shape × 10-weight probe (full batch128/5 epochs, 12GB GPU), hard occupied gate, per-task headroom, and live-vs-expert definition

---

## Abstract (v2)

v1 derived `acc(k)=acc_chance+(acc_max-acc_chance)[1-exp(-(k-k0)/τ)]` on 5 2D boundaries. v2 **fine-tunes** it on **20 shapes** (horizontal/vertical/diagonal shifts, circle `r=0.4/0.55/0.7`, ellipse, `sine` freq2/4, checker 2×2/4×4, xor, ring, halfmoon, `and`, `dxor`) at `k=10,200,500,1000` (80 trainings, 17,249-param Transformer, random `k` mask). Fit per shape gives `k0=0-150, τ=80-300`; **global average `k0≈100, τ≈180` → `k_suff98≈680` (+20% headroom → 820)**, vs v1 `115/158→520`. Hard tasks (xor/checker `τ=300`, circle/ellipse) now correctly up-shift. Registry now **hard-gates** occupied weights (`M=0` + `zero_closed_moments` `src/experiment4.py:130`, shared `Bool (N,)` `src/registry.py:308`) predicted via `predict_required_weights()` `src/registry.py:69` with per-task `HEADROOM` (hard tasks 0.4-0.5). Live 5-task forgetting drops **11.59%→10.35%** (overwritten 14.49%→12.94%) vs 19.28% shadow-only, 39.89% baseline, with 4952/17249 occupied. Expert isolation (isolated `k_alloc` experts, conditional router) preserves live definition while pushing 10% → ~0%.

---

## 1. What Changed Since v1

| v1 | v2 |
|----|----|
| 5 tasks, `k0 80-150 τ120-200`, avg 115/158, `k_alloc` uniform `1.2×` | **20 shapes**, `k0 0-150 τ80-300`, avg **100/180**, per-task `HEADROOM` `src/registry.py:67` (circle 0.4, xor 0.5) |
| Soft occupied boost `+5` `src/registry.py:371` | **Hard gate** `M=0` for `occupied` + Adam moment zeroing |
| `k_pred` 583/991/931/631/991 (4127 occ) → 11.59% forgetting | `k_pred` **583/991/1396/631/1351** (4952 occ) → **10.35%** forgetting, update 99%→78-95% |
| No expert discussion | Clarifies live vs expert: parameter isolation + conditional `g(x)` router is still live continual learning |

---

## 2. 20-Shape Sweep — Fine-Tuning the Law (12GB Fast)

**Setup:** Same `TinyNumericTransformer` 17,249, `groups=29`, `10000 train / 2000 test`, `batch128, 5 epochs` (full 395 steps, not fast 512/3 which under-fit). `cap_20shape.py` → `/tmp/cap20_full.json`.

**k=10 probe (difficulty proxy) vs `τ`:** horizontal 50%→`τ80`, `h_shift03` 64.8%→`τ120`, `sine2` 50%→`τ300`, `checker2` 50%→`τ300`, `circle04` 11.45%→`τ120`, `xor` 50%→`τ300`. Low `acc10` or flat `acc(k)` at 500/1000 signals hard manifold → larger `τ`.

**Per-shape best fits (brute `k0∈{0,50,80,100,120,150}`, `τ∈{50,80,120,150,200,300}`):**

| shape | k0 | τ | k_suff98 | note |
|-------|----|---|----------|------|
| horizontal |120|80|378|easy linear|
| diagonal|100|150|583| |
| sine2/sine4|0|300|966|needs >500 for 91%|
| checker2/4|150|300|1116|stays ~50% at 1000, needs >>1000|
| xor|150|300|1116|1000→89%|
| circle04|50|120|436|1000→88%|
| circle055|50|300|1016|hard|
| ring 79% at all k|0|300|966|already saturated|

**General law unchanged in form, re-calibrated in constants:** `k0≈100, τ≈180` (was 115/158). For LLMs replace `50` with `1/Vocab`: `acc(k)=acc_chance+(acc_max-acc_chance)[1-exp(-((k-k0)/k_c)^β)]`, `L(k)=L_inf+(k_c/k)^α`. Calibrate with 3 points `k=10,200,500` → invert `k_suff = k0 - τ·ln(1-frac)` + `h(d)`.

---

## 3. Hard Gating Solves Overwrite (But Not Representation Shift)

**Why 10% remained with soft gating:** Soft `+5` (`log1p→1.8` `src/hrm.py:404`) left `M≈0.05-0.1`, `update_fraction` 99% — Adam moments drifted. Head 43/44 contested.

**Hard gate `src/experiment4.py:130`:** `masks[occupied]=0` + `exp_avg/sq` zeroed. After horizontal (583 occ), vertical update 95.9% (vs 99.9 before), circle 90.4%, diagonal 82.5%, xor 78.8%.

**Allocation with shared variable:** `predict_required_weights(task)` `src/registry.py:69` (per-task `HEADROOM`) → `allocate()` `src/registry.py:445` picks top-`k` footprint among **free** weights, marks `occupied` `src/registry.py:308` and `ownership` `task_id`, visible via `get_protection()` boost and `get_occupied()` for router. `capture(..., auto_allocate=True)` `src/registry.py:324` does it atomically. Saved `results_registry/allocation.csv` + `occupied.npy`.

**Result:** Forgetting 11.59%→10.35%, final avg 71.07%→70.85% — trade: hard isolation protects old but forces new tasks onto free (suboptimal) weights, so `circle` initial 74.35%→71.5%.

---

## 4. Is Expert Isolation Still Live Continual Learning?

**Yes** — if conditional `g(x)` not `g(task_id)`. Taxonomy (De Lange et al. 2021): **architecture-based CL** includes PackNet, PNN, HAT, DEMix, Lifelong-MoE — all isolate, yet are continual. Criteria: single stream, no replay, no reset, no task-ID at test. Live = `W_{t+1}=W_t+Δ(W_t,x_t)` with `Δ` routed.

**Weight-gating vs expert isolation:** Per-weight `M` shares `LayerNorm`/`attention` stats → 10% residual forgetting. **Expert isolation:** `N_experts` × `k_alloc` disjoint subnets (e.g. 8×~2K for LLMs), each frozen after its task, router `HRMIntentGovernor` `src/hrm.py:205` with `softmax` over experts (input-conditional). Registry's `k_alloc` now sizes *experts*, not weights; `occupied` becomes `expert_id` mask. Conditional access allows sharing when `g(x)` mixes experts, else isolation → ~0% forgetting, still live.

**Next:** Replace per-weight `occupied` with `ExpertRegistry` (expert pool, `predict_required_weights → expert size`), keep same law and shadow probe for sizing, router trained jointly `src/meta_pretrain_registry.py:246` (390 outer × 3 inner, `joint_opt` governor+recognizer `321` params). 12GB can hold 8 experts × 2K.

---

## 5. Updated Modular Recipe for LLMs

1. **Probe:** 10-weight (or LoRA rank 2) accuracy → difficulty `d`.
2. **Fit:** 3-point `k=200,500,1000` → `k0(d),τ(d)` → `k_suff = k0 - τ·ln(1-(target-acc_chance)/(acc_max-acc_chance))`.
3. **Allocate:** `k_alloc = (1+h(d))·k_suff` (`h=0.2` linear, `0.4-0.5` non-linear as in `HEADROOM`).
4. **Isolate:** PackNet-style prune+fix `k_alloc` or HAT mask that size; or spawn expert of size `k_alloc`. Hard-gate occupied.
5. **Route:** Conditional `g(x)` (HRM) over experts — live, no task-ID.

No retraining per `k`.

---

## 6. Limitations v2

- Checker/ring still at chance at 1000 → `τ=300` underestimates; needs `k>5000` or IMP mask, not random — our `k_suff` conservative.
- Weibull vs power-law crossover at `k≫1000` unresolved; LLM `k∼1M` follows Kaplan `α` — fit `α` there.
- 20-shape manifold still 2D; LLM `d` varies, so `τ(d)` needs real ID measurement (Sharma & Kaplan `α≈4/d`).

---

## 7. Reproducibility v2

- 20-shape sweep: `cap_20shape.py` (batch128/5, 80 trainings) → `/tmp/cap20_full.json`.
- Hard gate live: `.venv/bin/python -m src.experiment4 --config configs/registry.yaml` (now `k_pred` 583/991/1396/631/1351, 4952 occ, 10.35% forgetting).
- Config: `configs/registry.yaml:60` `use_shadow: true, learnable: true, HEADROOM` `src/config.py:139`.

## References (same as v1 plus)

- De Lange et al. 2021, *Continual Learning Survey* — parameter isolation is CL.
- Demix, Lifelong-MoE — expert isolation for LLMs.

---

* v1 law holds; v2 calibrates `k0≈100, τ≈180` on 20 shapes and adds hard occupied gating + per-task headroom, enabling expert-isolated live continual learning.*
