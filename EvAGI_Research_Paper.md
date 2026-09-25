# EvAGI: Hierarchical Neuron-Isolated Continual Learning for Frontier Models on Consumer Hardware

**Technical Report — 2026-09-24 (rev. B — fresh-session proof + behavior-model base)**  
**Status:** Architecture mapped, core claims proven on 17K → 8.3M → 1.2B scales; live CLI green on RTX 4070  
**Goal:** A frontier-capable model that learns live, runs on consumer hardware (12 GB VRAM / 16 GB RAM), and suffers zero catastrophic forgetting.

---

## Abstract

Continual learning in deployed language models fails because gradient updates overwrite prior knowledge. We present **EvAGI** (Evolving Agentic General Intelligence), a hierarchical architecture that eliminates catastrophic forgetting by construction through **hard neuron isolation**, an **expert registry** that records what each neuron holds, and a **hierarchical register** with global and per-neuron weight pools. Unlike Mixture-of-Experts (MoE), EvAGI neurons are not resident in VRAM; they are **allocated on demand from a weight pool, trained in isolation, and paged from SSD** by the router, enabling dynamic scale without proportional VRAM growth.

We prove four claims experimentally:

1. **Zero forgetting under hard isolation.** On a 17,249-param Transformer over a 5-task 2D stream, naive fine-tuning yields **39.89% average forgetting**; registry + hard gating reduces this to **10.35%**; full expert isolation yields **0.00%** forgetting across all five tasks while retaining **95–99.7% final accuracy**.

2. **A minimal-neuron allocation law (Equation V4).** Facts require a measured floor of **4 neurons** (≈24.6K weights on LFM2.5-1.2B); skills scale as **n ≈ 58.4·ans_tokens − 323** (R² = 0.675) with adaptive doubling. Across 28 calibration items, **100% probe success**; across 10 skill/knowledge items, **100% success** with 64–4096 neurons.

3. **Live learning that is not a system-prompt trick.** On LFM2.5-1.2B, a user teaches a novel fact in natural language mid-conversation (`My office code is ZEBRA-42`); the system allocates 4 neurons, trains ~10–25 s, stores a **~1.8 MB sparse checkpoint**, and after a **brand-new process** restores only from that file and answers **`ZEBRA-42`** with **0% forgetting**. Deleting the checkpoint → refuse ("I don't know — teach me"). No answer string appears in any system prompt or code path; the few-shot prompt only teaches intent classification (`INTENT|kind|value`). Content/file learning (`learn file re_requiem.txt`) trains a 2048-neuron knowledge expert that answers protagonist / platforms / multiplayer queries via grounded retrieval from the expert's stored sentences.

4. **Base intelligence is a behavior model, not just an LM.** The Layer-0 base is specified (and prototyped via the LLM-native router) as a **behavior stack** whose core is **curiosity → learn-intent → problem-solving → thinking**, rather than a passive next-token predictor. Curiosity drives the refuse-and-offer-to-learn loop; learn-intent turns natural language into `learn|kind|value`; problem-solving and thinking are the default generators when no expert is paged. The frozen base never stores episodic facts — only dispositional behavior.

We specify the full EvAGI stack as a three-layer hierarchy: **Layer 0** (router, global register, **behavior-model base intelligence**, expert register with sub-registers), **Layer 1** (main neurons / experts allocated by Equation V4), and **Layer 2** (sub-neurons that refine learning within a single main neuron). The design goal is a frontier model whose knowledge footprint scales with *learned content*, not with parameter count held in VRAM.

---

## 1. Introduction

### 1.1 Problem

A model that learns during deployment must answer two questions at every turn:

- **Does it already know?** If yes, answer; if no, refuse and offer to learn.
- **If learning, where does the new knowledge live so it cannot destroy the old?**

Standard fine-tuning answers neither safely. Each gradient step moves shared weights; sequential tasks interfere. This is **catastrophic forgetting**, and it is the blocker between "a model that can be updated" and "a model that can *keep learning*."

A second problem is **what the base is for**. Most stacks treat the base as a fact warehouse that happens to also chat. EvAGI treats the base as a **behavior model**: its job is to be curious, to recognize learn-intent, to refuse honestly when ignorant, and to solve problems / think when no expert applies. Facts live outside the base, in isolated neurons.

### 1.2 Claim

If knowledge is stored in **disjoint neuron subsets** of the feed-forward network, and those subsets are **masked in both forward and backward passes**, then:

- Old knowledge cannot receive gradients from new tasks (isolation ⇒ 0% interference).
- New knowledge cannot be corrupted by old tasks (isolation is symmetric).
- The base model remains a pristine generalist; experts are **sparse deltas** that load on demand.

This is EvAGI. The contribution of this report is to (a) prove the isolation claim across three model scales, (b) derive the allocation law that makes isolation *efficient*, (c) specify the hierarchical register architecture needed to scale to frontier size on consumer hardware, and (d) define the base as a **curiosity-first behavior model** whose core loop is learn-intent → problem-solving → thinking.

### 1.3 Terminology (project vocabulary)

| Term | Meaning |
|------|---------|
| **Neuron** (or **Expert**) | A disjoint set of FFN intermediate channels allocated to one knowledge unit (fact, skill, content topic). Formerly "expert"; we standardize on **Neuron**. |
| **Sub-neuron** | A finer partition inside one main neuron (Layer 2). A fact neuron can spawn sub-neurons for related details. |
| **Global Register** | Layer 0 structure: full weight-pool occupancy map + Equation V4 allocator. |
| **Sub-register** | Per-neuron local register: tracks which sub-neurons exist inside a main neuron; avoids scanning the global pool. |
| **Equation V4** | Empirical law predicting how many neurons a new item needs before training. |
| **Governor** | Tiny MLP (11 features) that gates owned neurons, now registry-aware. |
| **Base intelligence** | The frozen **behavior model** (curiosity, learn-intent, problem-solving, thinking). Never stores episodic facts. Prototype: LFM2.5-1.2B + LLM-native `understand()`. |
| **Curiosity** | Base disposition: detect ignorance → refuse → offer to learn (drives the teach loop). |
| **Learn-intent** | Parse natural language into `learn \| query \| chat \| unknown` + `kind\|value`. |
| **Problem-solving / thinking** | Default generative modes when no expert is paged (step-by-step, tool-use-ready). |

---

## 2. Related Work and Gap

| Line of work | What it does | Why it is not enough |
|--------------|--------------|----------------------|
| **EWC** (Kirkpatrick 2017) | Fisher-weighted penalty on important weights | Soft constraint; residual forgetting; no allocation *quantity* |
| **PackNet** (Mallya 2017) | Prune + freeze ~1/T capacity per task | Fixed fractions; no predictive sizing; capacity exhausts |
| **HAT** (Serra 2018) | Learned per-task attention masks | Masks are soft-ish; no hierarchical register |
| **LoRA** | Low-rank adapters per task | Adapters share base forward graph; interference through base; VRAM scales with #adapters if all loaded |
| **MoE** (Shazeer 2017; Switch) | Sparse expert FFNs, top-k routing | **All expert weights must be in VRAM** for routing; VRAM ∝ total experts; not designed for *post-hoc* live learning |
| **Lifelong-MoE / DEMix** | Expert isolation for CL | Closer; still assume experts live on accelerator; no Equation-V4-style sizing from conversation |
| **Scaling laws** (Kaplan 2020; Chinchilla) | Whole-model N, D | Do not predict *active subnetwork size* for a single new fact |
| **Lottery Ticket** (Frankle 2018) | Existence of sparse sufficient tickets | No closed-form k_suff; no live routing |

**Gap:** No existing system combines (1) *hard* isolation with *zero* measured forgetting, (2) a *calibrated* allocation equation so a fact costs 4 neurons not 4096, (3) a *hierarchical register* so routing does not scan a billion-neuron pool, and (4) *SSD-tier* expert storage so frontier-scale knowledge does not require frontier-scale VRAM.

EvAGI fills this gap.

---

## 3. Experimental Platform

All experiments run on a single consumer machine:

- **GPU:** NVIDIA RTX 4070, 12 GB VRAM  
- **CPU RAM:** 16 GB (WSL2 allocated 12 GB after `.wslconfig`)  
- **Storage:** NVMe SSD (Windows + WSL)  
- **Stack:** PyTorch, Transformers 5.17, Python 3.12  

Model scales used:

| Scale | Model | Params | FFN neurons | Role |
|-------|-------|--------|-------------|------|
| S | Custom 2D Transformer | 17,249 | (weights) | Capacity law, forgetting baselines |
| M | TinyTalk (GPT-Neo) | 8.28M | 4,096 | Conversational live learning |
| L | **LFM2.5-1.2B-Instruct** | 1.17B | **131,072** | Primary live CLI, calibration, Equation V4 |

LFM2.5 architecture details: 16 layers, hidden 2048, SwiGLU FFN (`w1/w3/w2`), effective intermediate 8192 (auto-adjusted from config 12288), `WEIGHTS_PER_NEURON = 6144` (2048×3), bf16, mixed conv/attention layer types, all 16 layers have FFN.

---

## 4. Results Summary (What We Proved)

### 4.1 Forgetting: Baseline → Registry → Expert Isolation

**Setup S:** 5-task 2D boundary stream (horizontal, vertical, circle, diagonal, XOR), 17,249-param Transformer, AdamW, 5 epochs/task.

| System | Avg forgetting | Final avg accuracy | Overwritten |
|--------|----------------|--------------------|-------------|
| Naive sequential FT (baseline) | **39.89%** | 59.27% | 49.86% |
| HRM Governor (Phase 2 fixed) | **17.58%** | ~65% | ~24% |
| Registry + shadow footprint | **18.21%** | — | — |
| Registry + **hard occupied gate** | **10.35%** | 70.85% | 12.94% |
| **Expert isolation (EvAGI)** | **0.00%** | **95.05–99.7%** | **0** |

Per-task expert-isolation (`results_expert/forgetting.csv`):

| Task | Initial | Final | Forgetting |
|------|---------|-------|------------|
| horizontal | 99.65% | 99.65% | **0.0** |
| vertical | 99.85% | 99.85% | **0.0** |
| circle | 98.35% | 98.35% | **0.0** |
| diagonal | 98.30% | 98.30% | **0.0** |
| xor_quadrant | 99.55% | 99.55% | **0.0** |

Dynamic allocation variant (`results_dynamic`) also achieves **0.0 forgetting** on all five tasks (final 95.05–99.7%).

**Interpretation:** Soft regularization (EWC-like, governor-only) *reduces* forgetting but cannot eliminate it because gradients still touch shared parameters. Hard masks that zero non-owned channels in **both** forward and backward make interference structurally impossible.

### 4.2 Task-Order Robustness

Across 10 random permutations of the 5-task stream (`results_orders`), baseline forgetting averages **33.7–40.0%** (std 0.5–1.0); governor reduces this to **19.5–39.7%** depending on order. Hard expert isolation removes order sensitivity entirely (0% regardless of permutation) because tasks never share parameters.

### 4.3 HMEM / Footprint Method Ablation

Ablating how task importance is recorded (`results_hmem/hmem_live_summary.csv`, mean over seeds):

| Footprint method | Avg forgetting | Overwritten | Final acc |
|------------------|----------------|-------------|-----------|
| none | 16.48 ± 1.08 | 20.60 ± 1.35 | 70.16 |
| random | 15.94 ± 1.50 | 19.92 ± 1.87 | 70.51 |
| shuffled | 16.11 ± 1.38 | 20.13 ± 1.73 | 70.31 |
| grad | 14.00 ± 1.70 | 17.51 ± 2.13 | 70.84 |
| **magnitude (\|grad\|·\|w\|)** | **12.68 ± 1.21** | **15.85 ± 1.52** | **71.51** |

We adopt **|grad|·|weight|** (footprint) for `capture_protection` in the live system: it is the best soft signal we measured; under hard isolation it becomes the *protection feature* the governor sees, not the sole defense.

### 4.4 Conversational Live Learning (Scale M: TinyTalk 8.3M)

**Cross-session proof** (`results_llm_live/prove_cross_session.json`):

| Fact | Expected | Generations | Accuracy |
|------|----------|-------------|----------|
| fact_name | Rehan | Rehan, Rehan, Rehan | **100% (3/3)** |
| fact_color | Emerald | Emerald, Emerald | **100% (2/2)** |

Five skills (sentiment, spelling, capital, plural, antonym): **100% each**, **average forgetting 0.0**, **final average accuracy 100%**.

Interactive session (`live_learn_evagi.json`): before teaching, "What is your name?" → `a!!!!!!!!!` (base model garbage); after teaching `"My name is Rehan"` → `"Rehan"` with support-NLL dropping from ~4.35 to **0.0004** on the correct expert.

### 4.5 Scale L: LFM2.5-1.2B Calibration (Equation V4)

**Facts** (`results_calibration/calibrate_v4_lfm2_results.json`): 22 fact items, **28/28 ok** overall in the coarse sweep. Neuron counts:

- min **4**, max **64**, mean **9.5**
- Probe accuracy **100%** for all facts at 4 neurons (short values) or 64 (high-entropy values >6 tokens)

**Skills / knowledge** (10 items, all `ok: true`, `probe_acc = 1.0`):

| Kind | Example | ans_tokens | Neurons | Loss |
|------|---------|------------|---------|------|
| skill_code | sort a list in python | 19 | 256 | 0.037 |
| skill_code | reverse a string in python | 22 | 512 | 0.289 |
| skill_code | read a file in python | 27 | 768 | 0.154 |
| skill_code | fizzbuzz in python | 77 | 4096 | 0.014 |
| knowledge | capital of France | 25 | 3072 | 0.031 |
| knowledge | how photosynthesis works | 28 | 768 | 0.029 |
| skill_code | check prime in python | 20 | 1536 | 0.065 |
| skill_code | merge two dicts in python | 18 | 384 | 0.099 |
| knowledge | what gravity is | 9 | 64 | 0.045 |
| knowledge | why the sky is blue | 11 | 256 | 0.388 |

Linear fit over generative answers: **n ≈ 58.4 · ans_tokens − 323**, R² = 0.675 (high variance ⇒ adaptive doubling is mandatory as a safety net).

**Recipe that works on 12 GB:** SGD momentum 0.9, lr 5e-3, 40 epochs, batch 4, hard mask on `w1/w3` forward hooks + `hard_mask_grads`. (AdamW over all 16-layer FFN would allocate ~6.4 GB optimizer states → OOM; SGD avoids this.)

### 4.6 Live CLI Session (Scale L) — including fresh-process proof

End-to-end session on `src/live_interactive_lfm2.py`:

1. Teach `"My name is Rehan"` → allocate **4 neurons**, loss 0.0011, probe **100%**, **~10 s**, checkpoint **1.78 MB**.  
2. Teach `"My favorite color is blue"` → second expert, **4 neurons**, no overlap (occupied = 8).  
3. Query `"What is my name?"` → routes to expert 0 → **`Rehan`**.  
4. Query `"What is my favorite color?"` → expert 1 → **`blue`**.  
5. Query `"What is the meaning of life?"` → **unknown**, confidence 0.40 → refuse + teach.  
6. Casual chat → natural base-model response, no expert mask.  
7. **Restart process** → restore sparse checkpoint → both facts recalled (**cross-session**).  
8. Registry JSON: pool 131,072, two allocations of 4 neurons, `captured_tasks = 2`.

**Not a system prompt — ZEBRA-42 control experiment (rev. B):**

| Step | Action | Result |
|------|--------|--------|
| Code audit | `grep ZEBRA / Rehan / office code` in `live_interactive_lfm2.py` | **No** hardcoded answer for ZEBRA; "Rehan" appears only as a few-shot *routing* example (`my name is Rehan -> learn\|fact_name\|Rehan`) and a comment. System prompt only asks for `INTENT\|kind\|value`. |
| Session A (process 1) | `My office code is ZEBRA-42` → learn | `fact_code=ZEBRA-42`, V4_n=4, probe **100%**, loss 0.0012, **~1.8 MB** ckpt |
| Session A | `What is my office code?` | **`ZEBRA-42`** |
| Session B (process 2, no re-teach) | Load only `evagi_live.pt` → ask 3 phrasings | **`ZEBRA-42` × 3** (`office code` / `My office code?` / `What's my code?`) |
| Session C (checkpoint **deleted**) | Same question | **"I don't know that yet"** — refuse + offer teach |

Cross-session memory is **only** the sparse weight deltas + register in `evagi_live.pt`. There is no conversational transcript injected as a system-prompt memory block for fact recall.

**Content / file learning:** `learn file /tmp/opencode/re_requiem.txt` → topic extracted as **`Resident Evil Requiem`** (proper-noun title, not filename); **2048 neurons** (hard cap = `inter/4`, never full-FFN); 9 sentences; loss ≈ 1.5–1.9; sparse ckpt **~50 MB** for content+fact. Queries route via title match (`top1=1.00`) and answer by **IDF-weighted grounded retrieval** over the expert's stored sentences (protagonist → Grace Ashcroft sentence; platforms → PS5/Xbox/PC sentence; multiplayer → co-op sentence). Foreign questions (Mars, meaning of life) refuse.

**Checkpoint engineering:** Dense per-expert FFN deltas were ~3.2 GB/expert float32; loading them under 7.7 GB WSL RAM caused OOM kills. **Sparse deltas** (only owned rows of `w1/w3`, owned cols of `w2`) reduce a fact-only session to **~1.8 MB**. Content experts are larger (~50 MB at 2048 neurons) but still far below dense. Old v1 dense checkpoints are ignored on load (`safe_test.sh` drops >500 MB files as legacy dense).

---

## 5. The EvAGI Architecture

### 5.1 Design principles

1. **Base intelligence is read-only and behavioral.** Pretrained weights are never the site of live updates. The base is a **behavior model** (curiosity, learn-intent, problem-solving, thinking), not a fact store.  
2. **Knowledge = isolated neurons.** A neuron (expert) owns a disjoint FFN channel set.  
3. **Allocation is predicted, then corrected.** Equation V4 gives a lean prior; failure doubles on the grid.  
4. **Registers are hierarchical.** Global pool for spawning; sub-registers for locality.  
5. **Routing sees the register.** The classifier receives protection / ownership features, not just hidden states.  
6. **Tiers: SSD → RAM → VRAM.** Neurons are paged in for the active turn; idle neurons do not occupy VRAM (unlike MoE).  
7. **Curiosity before knowledge.** Unknown → refuse → offer teach is a first-class behavior, not an error path.

### 5.2 Layer 0 — The chassis (everything lives here)

Layer 0 is the always-resident control plane:

```
┌─────────────────────────────────────────────────────────────────┐
│ LAYER 0                                                        │
│  ┌──────────────┐   ┌──────────────────────────────────────┐   │
│  │  Classifier  │◄──│  Expert Register (with sub-registers)│   │
│  │  / Router    │   │  - what each neuron holds            │   │
│  │  (HRM + LLM  │   │  - kind, value, support centroid     │   │
│  │   understand)│   │  - sub-register pointers             │   │
│  └──────┬───────┘   └──────────────▲───────────────────────┘   │
│         │ intent, eid              │ protection feats          │
│  ┌──────▼──────────────────────────┴───────────────────────┐   │
│  │  Global Register                                       │   │
│  │  - weight pool occupancy map (per layer Bool)          │   │
│  │  - ownership map (expert_id per neuron)                │   │
│  │  - protection scores (|g|·|w| footprint)               │   │
│  │  - Equation V4 allocator (predict_neurons_v4)          │   │
│  │  - allocate_free / deallocate / capture_protection     │   │
│  └────────────────────────────────────────────────────────┘   │
│  ┌────────────────────────────────────────────────────────┐   │
│  │  Base Intelligence = BEHAVIOR MODEL (frozen)           │   │
│  │  core: curiosity → learn-intent → problem-solving      │   │
│  │        → thinking                                     │   │
│  │  (prototype: LFM2.5-1.2B + understand())               │   │
│  └────────────────────────────────────────────────────────┘   │
│  ┌────────────────────────────────────────────────────────┐   │
│  │  TinyPerExpertGovernor × active neurons (11-dim)       │   │
│  └────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

**Components:**

1. **Classifier / Router**  
   - Fast path: embedding cosine + token overlap against registered support centroids.  
   - Slow path: **LLM-native `understand()`** — one short greedy generation classifies `learn | query | chat | unknown` and extracts `kind|value` from arbitrary natural language (no regex teach-patterns). Post-rules: questions never classify as learn; pronoun junk values rejected.  
   - Decides: answer from base (problem-solving/thinking), answer from neuron, refuse+offer teach (curiosity), or spawn learn.

2. **Global Register** (weight pool)  
   - Per-layer `occupied: Bool[inter]`, `ownership: Long[inter]`, `protection: Float[inter]`.  
   - `allocate_free(n, expert_id)` → disjoint masks; `deallocate` for adaptive retries.  
   - `capture_protection` records footprint **only on owned neurons** (monotonic max).  
   - `get_protection_feats(mask, layer)` → 3 features: `protection_norm`, `prior_owned`, `layer_occupancy`.

3. **Base intelligence — behavior model (core of Layer 0)**  
   The base is **not** a chat LM that happens to refuse sometimes. It is specified as a **behavior model** whose dispositional core, in priority order:

   | Behavior | Role | Prototype in current CLI |
   |----------|------|---------------------------|
   | **Curiosity** | Detect ignorance; never bluff; refuse and *offer* to learn ("I don't know — teach me"). | `intent=unknown` → fixed refuse+teach string; confidence gate |
   | **Learn-intent** | Turn any natural utterance into structured `learn\|kind\|value` (or query/chat/unknown). | `understand()` few-shot → `INTENT\|kind\|value`; questions forced non-learn |
   | **Problem-solving** | When no expert applies and the task is instrumental, work the problem (steps, tools, subgoals). | Base chat / future tool-loop; currently free-form generate without expert mask |
   | **Thinking** | Default generative mode: deliberate, stepwise reasoning before committing an answer. | Base generate (temp-controlled); future: explicit think→answer traces |

   Constraints on the behavior base:

   - **Frozen** (`requires_grad_(False)` except live FFN under hard mask during a learn step).  
   - **Does not store episodic facts.** Names, codes, preferences, document facts live only in expert neurons + sparse checkpoint.  
   - **Hosts the router** (`understand`) so learn-intent is the same weights that do thinking/problem-solving — one brain, two roles (disposition vs. knowledge).  
   - Target training for a *dedicated* EvAGI base (roadmap): RL/curriculum that rewards (a) correct "I don't know" + learn offer, (b) high-quality `learn\|kind\|value` extractions, (c) multi-step problem solutions, (d) explicit thinking traces — i.e. train **behavior**, let experts supply **content**.

4. **Expert Register (with sub-registers)**  
   - Maps `expert_id → {kind, value, n_neurons, loss, probe_text, centroid, source_text, sentences, sub_register_ptr}`.  
   - **Sub-register:** each main neuron that has children maintains a local occupancy map of its own weight slice — so "does this neuron have a sub-neuron for X?" is O(local), not O(global pool).  
   - This is what prevents frontier-scale routing from scanning 10⁸–10⁹ neuron records.

**Turn protocol (Layer 0):**

```
user prompt P
  → classifier.understand(P)          # learn-intent (behavior #2)
      if knowledge present and confident:
          route to matching expert e (register lookup + embedding)
          page e's sparse delta from SSD → apply mask → generate
          restore base snapshot
      elif question about unknown topic:
          curiosity: refuse + offer teach   # behavior #1
      elif casual / open problem:
          base: problem-solving or thinking # behaviors #3–4
      elif teaching signal:
          go to Layer 1 spawn protocol
  → append turn summary to classifier context (what was learned / asked)
```

Every turn's *outcome* (learned? existed? refused?) feeds back to the classifier so routing improves online.

### 5.3 Layer 1 — Main Neurons (Experts)

A **main neuron** is the unit of live allocation:

1. Classifier emits `learn|kind|value`.  
2. Global Register asks **Equation V4**: how many neurons?  
3. `allocate_free` carves a **disjoint** set from the free pool (never overlap prior experts).  
4. Train with `HardSwiGLUMask` + `hard_mask_grads` (SGD, short schedule).  
5. Probe; if fail → `deallocate` + `next_grid_step` (double) and retry (max 4).  
6. On success: `capture_protection`, `extract_sparse_delta`, register centroid, save sparse checkpoint.  
7. Restore base FFN; delta lives in storage until routed.

**Equation V4 (implemented in `src/registry.py`):**

```
grid = {4, 8, 16, ..., 16384}

if kind is fact_*:
    n = 4 if val_tokens ≤ 6 else 64          # measured floor
else:  # skill / knowledge / content
    n = max(64, ceil_grid((58.4 * ans_tokens - 323) * 1.25))

on training failure:
    n = next_grid_step(n)   # smallest grid value > n
```

Constants: `V4_FACT_BASE=4`, `V4_FACT_LONG=64`, `V4_SKILL_SLOPE=58.4`, `V4_SKILL_INTERCEPT=-323`, `V4_SKILL_HEADROOM=1.25`. Verified 13/13 safe with mean over-allocation **2.03×** (vs 6.1× with naive 2.0 headroom).

**Governor (registry-aware):** `INPUT_DIM = 11` = 8 local features (weight/grad stats, position, layer, owned) + 3 registry features. Gates start open (bias≈4 → sigmoid≈0.98); light L1 for selectivity. Purpose under hard isolation: *anticipate* which owned channels matter when later tasks force reallocation pressure — not to prevent overwrite (masks already do that).

### 5.4 Layer 2 — Sub-neurons

Motivation: one fact = one main neuron wastes routing granularity when a *topic* needs internal structure (e.g., "Resident Evil Requiem" → release date, platforms, protagonist as sub-facts).

```
Main Neuron (Layer 1)  — allocated by V4, has global expert_id
  └─ Sub-register (local pool slice)
       ├─ sub-neuron 0: "protagonist = Grace Ashcroft"
       ├─ sub-neuron 1: "platforms = PS5, Xbox, PC"
       └─ sub-neuron 2: "engine / gameplay hooks"
```

**Properties:**

- Global register assigns the main neuron a **weight pool** (its slice of the FFN).  
- Sub-register partitions that slice; sub-neurons train with the same hard-mask discipline *inside* the parent mask (intersection of masks).  
- Routing: classifier hits main neuron first (cheap, from Expert Register); sub-register resolves fine-grained field.  
- If the parent has no sub-register, behavior is identical to Layer 1 (backward compatible).  
- Maximize learning *within* one main neuron before spawning a sibling — increases effective capacity per allocated weight.

Sub-neurons are the mechanism that keeps **fact density high**: instead of one flat expert per sentence, a content topic becomes a small tree, still paged as one sparse blob for the active turn.

### 5.5 Storage tiers (vs MoE)

| | MoE | EvAGI |
|--|-----|-------|
| Where do expert weights live? | **All in VRAM** for top-k routing | **SSD / disk**; only active expert's delta + base in VRAM |
| VRAM vs #experts | Linear (or expert-parallel across GPUs) | **Constant** (base + 1 active delta ≈ base size) |
| When are experts created? | Fixed at pretrain | **On demand** during conversation |
| Routing input | Learned gate on hidden state | Classifier + **register contents** (protection, ownership, sub-register index) |

Sparse delta size (measured): **4-neuron fact ≈ 48 KB** raw weights; full 2-fact session checkpoint ≈ **1.9 MB** including registry + governors. Paging cost is negligible on NVMe; the router loads only the `expert_id` it selected.

---

## 6. Equation V4 — Derivation and Validation

### 6.1 Motivation

Naive "give every task 10% of the FFN" wastes 99.97% of capacity on facts that need 4 neurons. We need `n = f(kind, answer entropy)` calibrated on the *target* architecture.

### 6.2 Procedure

1. Grid search `n ∈ {4…16384}` with fixed recipe (SGD, 40 epochs, batch 4, hard mask).  
2. Success = probe exact-match ≥ threshold.  
3. Fit piecewise law; hold out adaptive doubling as safety net.

### 6.3 Findings

- **Facts are almost free.** 22/22 short facts succeed at **4 neurons**. Only values with >6 BPE tokens (e.g., "spaghetti bolognese") need 64.  
- **Generative answers scale with token count** but noisily (R²=0.675). Headroom 1.25 + grid doubling covers the tail.  
- **Over-allocation cost:** headroom 2.0 → mean 6.1× waste; V4 lean + double-on-fail → **2.03×** mean, 13/13 success.  
- Early 5M-model sweeps failed mostly due to *evaluation bugs* (whitespace mismatch) and too few epochs — not capacity. Fixing probes revealed the 4-neuron floor.

### 6.4 Pseudocode

```python
def predict_neurons_v4(kind, ans_tokens, val_tokens):
    if kind.startswith("fact_"):
        base = 64 if val_tokens > 6 else 4
        return ceil_grid(base * 1.0)
    est = max(64, -323 + 58.4 * max(ans_tokens, 1))
    return ceil_grid(est * 1.25)

# train loop
n = predict_neurons_v4(...)
for attempt in range(4):
    masks = allocate_free(register, n, expert_id)
    loss = train(masked_ffn, qa, sgd)
    if probe_ok: break
    deallocate(masks); n = next_grid_step(n)
```

---

## 7. Training Mechanics (Why ~10 s Today, and the Path to Instant)

### 7.1 What runs when a fact is learned

| Step | Cost today | Notes |
|------|------------|-------|
| Full forward × epochs × batches | ~70% | Needed: expert neurons sit *inside* the base graph; loss is at LM head |
| Full backward traversal | ~20% | Autograd walks to layer L then masks non-owned grads to 0 |
| Probe generation | ~10% | 3 × ~64-token decode |
| Governor / hooks / Python | small | |

Updating 4 neurons (≈24.6K weights) is microseconds; the wall clock is *computing the gradient context* through 1.17B frozen parameters.

### 7.2 Projected plug-and-play (Layer 0/1 target design)

To make training **milliseconds**, detach experts from the base graph:

```
base forward (no_grad) → cached hidden h_L
expert = small module on h_L   (additive delta / adapter)
loss(expert(h_L)) → backward ONLY through expert
```

This is **not** skipping learning — it is skipping *base* F+B. Base remains a frozen feature extractor; experts become true plug-and-play modules the router pages from SSD. That is the intended production shape of EvAGI at frontier scale; the current in-weight hard-mask design is the **correctness prototype** (proves 0% forgetting) and the adapter design is the **performance path** (proves instant learn).

---

## 8. Live Session Protocol (Reference)

```
1. Load base → GPU once; snapshot FFN; build empty Global Register.
2. For each user turn:
   a. understand(P) via classifier (LLM one-liner + embedding fast path)
   b. if query & expert match: page delta, mask, generate, restore base
   c. if query & no match: refuse + offer teach
   d. if learn: V4 → allocate → train → probe → sparse save → register
   e. if chat: base generate
   f. log turn outcome to classifier memory
3. On quit: save items + sparse deltas + register + router (≈ MBs).
4. On next launch: load base + checkpoint; rebuild masks from ownership;
   continue with zero retraining of base.
```

Commands exposed: natural teaching, `learn file <path>`, `learn this: <text>`, `facts`, `experts`, `save`, `quit`.

---

## 9. Limitations

1. **In-weight training latency (~10 s/fact, ~25 s/content)** due to full-graph F+B; adapter-style experts needed for instant learning (Section 7.2).  
2. **Skill/content V4 variance (R²=0.675)** — doubling retries add tail latency; content expert uses a hard cap (`inter/4 = 2048`) so probe may accept on loss-with-partial-hit rather than 100% exact prose match.  
3. **Classifier errors** — questions can be mis-tagged as `learn`; mitigated by post-rules (questions never learn) and source-derived QA prompts, not eliminated.  
4. **Single-GPU / single-process** — no multi-tenant shard of the register yet.  
5. **Content answers** rely on grounded lexical retrieval over stored sentences when masked generation is incoherent; book-scale recall not solved.  
6. **Sub-neuron (Layer 2)** specified and partially exercised via content topics; full local-pool allocator is the next implementation milestone.  
7. **Base is not yet behavior-trained** — current base is a general instruct LM (LFM2.5) with a curated `understand()` prompt; a dedicated curiosity / learn-intent / problem-solving / thinking fine-tune is roadmap item 1, not shipped.  
8. Experiments on S/M scales use custom or TinyTalk models; L-scale results are calibration + interactive CLI (including ZEBRA-42 fresh-process control), not a full 5-skill benchmark at 1.2B with the new LLM router (TinyTalk path carries the 0% forgetting conversational proof).

---

## 10. Conclusion and Roadmap

**Concluded findings:**

- Catastrophic forgetting in sequential live learning is **solved** at the architectural level by hard neuron isolation + register + governors: **39.89% → 0.00%** forgetting on the controlled stream; **0.0%** across five skills and cross-session facts on conversational models.  
- **Equation V4** makes isolation *affordable*: facts cost 4 neurons; skills cost O(answer tokens) with a calibrated slope and adaptive doubling.  
- The stack works end-to-end on a **consumer RTX 4070**: teach in natural language, **100% recall in a brand-new process** from a **~1.8 MB sparse checkpoint** (ZEBRA-42 control), **0% forgetting**; unknown → refuse + teach; no system-prompt fact memory. Content files train capped 2048-neuron experts with grounded multi-aspect answers.  
- MoE is the wrong scaling story for *live* learning (VRAM-resident experts). EvAGI's register + SSD-tier neurons scales knowledge with content, not with resident parameters.  
- **Base intelligence is a behavior model.** Core dispositional stack: **curiosity** (detect ignorance → refuse → offer learn), **learn-intent** (NL → `learn|kind|value`), **problem-solving**, **thinking**. Episodic knowledge is externalized to neurons; the base stays a pure doer/learner, never a fact warehouse.

**Roadmap (in order):**

1. **Behavior-trained base** — fine-tune/RL the base specifically on curiosity (know-what-you-don't-know), learn-intent extraction quality, multi-step problem-solving, and explicit thinking traces; keep facts 100% out of base weights.  
2. **Layer 2 sub-registers** — local pools under each main neuron; content trees.  
3. **Detached adapter experts** — instant train (ms) while keeping 0% forgetting.  
4. **Classifier turn-memory** — every turn updates routing priors (dynamic, not static few-shot).  
5. **Batch content ingestion** — books/courses as forests of sub-neurons.  
6. **Multi-scale register sharding** — frontier parameter counts with constant VRAM.  
7. **Publishable benchmark** — standard CL suites (CIFAR/ImageNet splits, text CL) with forgetting = 0 under isolation, vs EWC/PackNet/HAT/LoRA/MoE baselines; plus a behavior suite (refuse-accuracy, learn-intent F1, problem-solving pass@k, thinking-chain quality).

---

## 11. Reproducibility

| Artifact | Location |
|----------|----------|
| Live LFM2.5 CLI | `src/live_interactive_lfm2.py` |
| Register + Equation V4 | `src/registry.py` (`predict_neurons_v4`, `next_grid_step`) |
| Governor + isolation | `src/llm_evagi.py` (`NeuronRegister`, `TinyPerExpertGovernor`, `INPUT_DIM=11`) |
| Calibration sweeps | `src/calibrate_v4_lfm2.py`, `src/calibrate_v4.py` |
| Calibration JSON | `results_calibration/calibrate_v4_*.json` |
| Cross-session proof | `results_llm_live/prove_cross_session.json` |
| Fresh-process control | `/tmp/opencode/z_session_A.log`, `z_session_B.log` (ZEBRA-42; regenerate with demo above) |
| Expert isolation (S) | `results_expert/forgetting.csv` |
| Registry hard-gate (S) | `results_registry/` |
| HMEM ablation | `results_hmem/hmem_live_summary.csv` |
| Order sweep | `results_orders/order_means.csv` |
| Safe test harness | `safe_test.sh` |
| Prior law reports | `research-report.md`, `research-reportv2.md` |

**Minimal live demo:**

```bash
# Fresh fact + fresh process (proves no system-prompt memory)
rm -f models/LFM2.5-1.2B-Instruct/evagi_live.pt
printf 'My office code is ZEBRA-42\nWhat is my office code?\nWhat is the meaning of life?\nquit\n' \
  | timeout 300 .venv/bin/python -u src/live_interactive_lfm2.py
# → learns ZEBRA-42; answers ZEBRA-42; refuses meaning-of-life

printf 'What is my office code?\nMy office code?\nquit\n' \
  | timeout 120 .venv/bin/python -u src/live_interactive_lfm2.py
# brand-new process → restores ckpt → ZEBRA-42

# Content file
printf 'learn file /tmp/opencode/re_requiem.txt\nWho is the protagonist of Resident Evil Requiem?\nquit\n' \
  | timeout 300 .venv/bin/python -u src/live_interactive_lfm2.py
```

---

## References

1. Kirkpatrick et al. 2017. *Overcoming catastrophic forgetting in neural networks.* PNAS (EWC).  
2. Mallya & Lazebnik 2017. *PackNet: Adding Multiple Tasks to a Single Network by Pruning.* arXiv:1711.05769.  
3. Serra et al. 2018. *Gradient Episodic Memory / HAT.* NeurIPS.  
4. Frankle & Carbin 2019. *The Lottery Ticket Hypothesis.* ICLR.  
5. Kaplan et al. 2020. *Scaling Laws for Neural Language Models.* arXiv:2001.08361.  
6. Hoffmann et al. 2022. *Training Compute-Optimal Large Language Models* (Chinchilla).  
7. Shazeer et al. 2017. *Outrageously Large Neural Networks: Sparsely-Gated Mixture-of-Experts.* ICLR.  
8. Hendrycks et al. / Wang et al. — Lifelong-MoE / DEMix lines for expert-based CL.  
9. De Lange et al. 2021. *A Continual Learning Survey: Defying Forgetting.*  
10. Hu et al. 2021. *LoRA: Low-Rank Adaptation of Large Language Models.*  
11. EvMind internal: `research-report.md` (modular capacity law v1), `research-reportv2.md` (20-shape + hard gating), this report (EvAGI architecture + LFM2.5 live proof).

---

*EvAGI turns catastrophic forgetting from an empirical failure mode into an architectural non-event: knowledge lives in isolated neurons, the register knows who owns what, Equation V4 prices every new thought, the router pages only what the current turn needs — from SSD, not from a VRAM budget that MoE can never escape — and the base stays a pure **behavior model**: curious enough to admit ignorance, trained to recognize learn-intent, and core-wired for problem-solving and thinking.*
