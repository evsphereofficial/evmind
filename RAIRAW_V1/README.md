# RAIRAW-V1 — implementation status

**Retention-Attention Intent Recursive Adapted Weights** — Experiment Series 2.
Architecture thesis: `RAIRAW_V1_Architecture_Series_2.md` (repo root).

## Hierarchy (as implemented)

```
INPUT
  |
  v
HRM GOVERNOR (frozen, pretrained) ---- WHERE? ----
  |                                           |
  |  per-weight masks -> region importance    |
  |  + H_MEM prior (RAIRAW influence memory)  |
  v                                           v
allocation: top-K regions get a RAIRAW       [regions: contiguous chunks of
  (K = f(open HRM mask mass))                  the main-weight pool, <=1K w]
  |                                           |
  v                                           v
RAIRAW pool (weight-tied recursive cells,   each RAIRAW:
  one state per active region)                R (retention) / A (attention)
  |                                           I (intent) / C (controller)
  |  per-weight gates for its region          ~720 params < 1K budget
  v
gated AdamW update  (W = pre + M o dW_adam;  closed regions: gate 0 + zeroed
  moments — reuse HRMController enforcement)
  |
  v
influence report -> H_MEM (per-region EMA)   -----> future HRM allocation
```

- **HRM decides WHERE**: the pretrained HRM governor's per-weight masks are
  aggregated to region importance; the number of active RAIRAWs K is derived
  from the HRM's open mask mass. H_MEM (observed RAIRAW influence) blends
  into the importance as a prior after the first task.
- **RAIRAW decides HOW**: each active region's recursive controller emits the
  per-weight gates (local retention/attention/intent policy). Inactive
  regions are closed nodes (gate 0, Adam moments zeroed).
- **H_MEM**: begins empty; each RAIRAW reports its region's observed
  influence (relative gradient magnitude) after each task; EMA accumulates
  it into the next allocation (Experiments 2.5).
- **Adapters**: not in V1.0 — the Controller emits an `adapter_need` scalar
  per region (reported/measured) but no adapter pool exists yet
  (Experiment 2.6).

## Architecture-to-doc mapping

| Doc section | Implementation |
|---|---|
| §3 RAIRAW <= ~1K params | `RairawCell` ~721 params (configurable h_dim) |
| §3 <= ~1K assigned weights | `Region` region_size=1000, per-module chunks |
| §4.1 Retention | `R` head: sigmoid(W_r @ h) |
| §4.2 Attention | `A` head: sigmoid(W_a @ h) |
| §4.3 Intent | `I` head: softmax 4-dim {retain, learn, adapt, protect} |
| §4.4 Controller | per-weight gate head over [f_i; R; A; I; h] |
| §5 main pool 20,000 | 17,249-weight TinyNumericTransformer pool |
| §6 MAX_RAIRAW | `raira.max_rairaw` (default 20) |
| §7 H_MEM empty | `HmemMemory` starts zeroed |
| §9 RAIRAW -> HRM feedback | influence = region mean\|g\| / global mean\|g\| |
| §10 H_MEM generated | per-region EMA of feedback |
| §12 local learning | gated AdamW per region |
| §13 R/A decision | learned by controller (not a fixed policy) |
| §16 global/local authority | HRM = allocation, RAIRAW = local gates |
| §17 equations | `allocate()`, `RairawPool.gate()`, `HmemMemory.update()` |

## Experiments run

Status of the Series-2 experiment list (RAIRAW_V1_Architecture_Series_2.md §19):

- 2.1 RAIRAW size: TBD
- 2.2 assigned weight capacity: TBD
- 2.3 number of active RAIRAWs: TBD
- 2.4 dynamic allocation: TBD
- 2.5 H_MEM emergence: TBD
- 2.6 adapter pool: TBD (V2)
- 2.7 full continual-learning test: TBD

## Files

- `src/raira.py` — core: Region / RairawCell / RairawPool / HmemMemory / allocation
- `src/raira_meta.py` — meta-training of the shared RAIRAW controller
  (selective-plasticity objective, functional stateless-AdamW burst)
- `src/experiment3.py` — RAIRAW-V1 live 5-task stream + allocation measurement
- `configs/raira_v1.yaml` — RAIRAW-V1 experiment config
- `RAIRAW_V1/results/` — run outputs

## Key knobs (config `raira:`)

```yaml
raira:
  region_size: 1000     # max weights per RAIRAW region (Exp 2.2)
  max_rairaw: 20        # pool bound (Exp 2.3)
  h_dim: 16             # controller state dim (Exp 2.1)
  intent_dim: 4         # {retain, learn, adapt, protect}
  hmem_alpha: 0.5       # H_MEM EMA rate
  alloc_blend: 0.7      # HRM mask vs H_MEM prior weight in allocation
  close_threshold: 0.02 # gates below this = closed nodes (as HRM system)
```
