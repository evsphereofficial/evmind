# Architecture Verified — Experiment Series 2

# RAIRAW-V1
## Retention-Attention Intent Recursive Adapted Weights

**Status:** Architecture thesis draft / Experiment Series 2  
**Predecessor:** HRM Governor + H_MEM continual-learning architecture  
**Primary objective:** Test whether making the adaptive unit a small recursive RAIRAW model, rather than an undifferentiated scalar weight, can improve stability, plasticity, dynamic parameter allocation, and continual-learning retention.

---

## 1. Thesis

RAIRAW-V1 proposes that catastrophic forgetting can be reduced by replacing undifferentiated parameter updates with a hierarchical system of small recursive learning models.

The global HRM Governor retains authority over the complete main weight pool.

The HRM does not directly treat every scalar parameter as an independent decision. Instead, it dynamically allocates a bounded number of **RAIRAW models**.

Each RAIRAW model is itself a small recursive model containing:

- Retention
- Attention
- Intent
- Controller

A RAIRAW model can receive a bounded pool of main-model weights underneath its authority. Those weights are the knowledge-bearing weights used for computation and training.

Therefore:

> **HRM decides WHERE computational/learning capacity should be allocated.**
>
> **RAIRAW decides HOW the allocated region should behave during learning.**
>
> **H_MEM records what the RAIRAW system discovers about the influence of those regions.**
>
> **Adapters provide additional plasticity without necessarily modifying protected main weights.**

---

# 2. Architectural hierarchy

The proposed hierarchy is:

```text
                         INPUT
                           |
                           v
                  +----------------+
                  | HRM GOVERNOR   |
                  | + H_MEM        |
                  +-------+--------+
                          |
                  global allocation
                          |
             +------------+------------+
             |            |            |
             v            v            v
          RAIRAW-1     RAIRAW-2     RAIRAW-N
             |            |            |
        tiny model    tiny model    tiny model
             |            |            |
        +----+----+  +----+----+  +----+----+
        | R  A  I |  | R  A  I |  | R  A  I |
        |    C    |  |    C    |  |    C    |
        +----+----+  +----+----+  +----+----+
             |            |            |
             v            v            v
        assigned main weights / computational region
             |            |            |
             +------------+------------+
                          |
                  influence feedback
                          |
                          v
                        H_MEM
                          |
                          v
                  future HRM allocation

                  Separate adapter pool
                          |
                          v
                  adaptive capacity
```

---

# 3. The RAIRAW model itself

A critical architectural distinction:

**RAIRAW is not merely a container around ordinary weights.**

A RAIRAW is a **small collection of tiny recursive models**.

The default experimental budget is approximately:

```text
RAIRAW model size <= ~1,000 parameters
```

This budget describes the **RAIRAW model/controller itself**.

A RAIRAW model can additionally receive:

```text
up to ~1,000 main-model weights
```

under its authority for computation and live training.

Therefore, a RAIRAW has two conceptually separate components:

```text
RAIRAW
|
+-- Tiny recursive control model
|      ~1K parameters
|
+-- Assigned main-weight region
       <= ~1K weights
```

The exact budgets are configurable and must be experimentally validated rather than assumed to be optimal.

---

# 4. RAIRAW's internal recursive components

Each RAIRAW contains four conceptual components.

## 4.1 Retention (R)

Retention estimates how strongly the allocated region should preserve existing knowledge.

It may incorporate:

- accumulated influence
- prior update history
- previous-task relevance
- stability requirements
- H_MEM feedback

Conceptually:

```text
R = "How dangerous is changing this region?"
```

High retention means stronger protection.

Low retention means greater plasticity is available.

---

## 4.2 Attention (A)

Attention estimates the importance of the allocated region for the current learning problem.

Conceptually:

```text
A = "How useful is this region for what I am learning now?"
```

High attention means the region is important for current learning.

Low attention means the region may require little adaptation.

---

## 4.3 Intent (I)

Intent represents the purpose/context of the requested update.

Potential states include:

- retain
- learn
- adapt
- specialize
- transfer
- modify
- protect

The exact representation is an experimental question.

The architecture must not assume that a predefined intent vocabulary is necessarily optimal.

---

## 4.4 Controller (C)

The Controller is the recursive decision mechanism inside the RAIRAW.

It receives information from:

```text
R
A
I
HRM context
local weight/update information
```

and decides how the allocated weights should be modified.

Conceptually:

```text
       R
       |
       v
A ---> C <--- I
       |
       v
learning / retention decision
```

The controller has a fixed parameter budget.

This prevents the recursive control system from growing without bound.

---

# 5. Main Weight Pool

The main model has a predefined pool of knowledge-bearing weights.

Example:

```text
MAIN_WEIGHT_POOL = 20,000
```

This is an architectural budget, not a mandatory value.

The HRM has authority over the complete pool.

The HRM can allocate this pool into RAIRAW-controlled regions.

For example:

```text
Main pool = 20,000

HRM allocates 5 RAIRAW models

RAIRAW-1 -> <= 1,000 assigned weights
RAIRAW-2 -> <= 1,000 assigned weights
RAIRAW-3 -> <= 1,000 assigned weights
RAIRAW-4 -> <= 1,000 assigned weights
RAIRAW-5 -> <= 1,000 assigned weights
```

Unused main weights remain available to the architecture.

The initial implementation should keep allocation simple and measurable before introducing complex dynamic partitioning.

---

# 6. RAIRAW model pool

The system contains a bounded pool of available RAIRAW models.

Example:

```text
MAX_RAIRAW = 20
```

The HRM decides how many RAIRAW models to activate.

For example:

```text
Simple input:
    3 RAIRAW models

Moderate input:
    6 RAIRAW models

Complex input:
    10 RAIRAW models
```

The number is dynamic but bounded.

Thus:

```text
0 < active_RAIRAW <= MAX_RAIRAW
```

The important architectural principle is:

> The HRM allocates a number of small learning units rather than assuming the entire model must participate equally.

---

# 7. H_MEM begins empty

At the beginning of a new model/task:

```text
H_MEM(0) = empty
```

H_MEM is therefore NOT required to predict influential regions before the architecture has interacted with the data.

This is intentional.

The first interaction is an information-acquisition event.

---

# 8. First interaction

The initial flow is:

```text
INPUT
  |
  v
HRM
  |
  | H_MEM is empty
  |
  v
allocate N RAIRAW models
  |
  +--> RAIRAW-1
  +--> RAIRAW-2
  +--> ...
  +--> RAIRAW-N
  |
  v
RAIRAW models process assigned regions
  |
  v
RAIRAW reports influence
```

No separate expensive direct "hit" experiment is required.

The architecture obtains influence information through normal communication between HRM and RAIRAW.

---

# 9. RAIRAW -> HRM feedback

After processing the first wave of data, each RAIRAW reports information about its assigned region.

For example:

```text
RAIRAW-1 -> influence = 0.82
RAIRAW-2 -> influence = 0.07
RAIRAW-3 -> influence = 0.64
RAIRAW-4 -> influence = 0.11
RAIRAW-5 -> influence = 0.91
```

The exact influence representation is an experimental variable.

The important architectural property is:

> RAIRAW models communicate their observed influence back to the global HRM.

---

# 10. H_MEM is generated from communication

The feedback becomes H_MEM.

Conceptually:

```text
RAIRAW influence feedback
            |
            v
          H_MEM
            |
            v
       future HRM decisions
```

For example:

```text
H_MEM:

RAIRAW-1 -> high influence
RAIRAW-2 -> low influence
RAIRAW-3 -> medium influence
RAIRAW-4 -> low influence
RAIRAW-5 -> very high influence
```

This creates a progressively accumulated representation of which RAIRAW regions have demonstrated computational importance.

Therefore:

> H_MEM is an observed memory of RAIRAW influence, not an initial influence predictor.

---

# 11. Subsequent interaction

Once H_MEM contains information:

```text
NEW INPUT
   |
   v
HRM
   |
   +--> consult H_MEM
   |
   +--> allocate RAIRAW models
   |
   v
RAIRAW processing
   |
   v
new influence feedback
   |
   v
H_MEM update
```

The architecture therefore forms a closed loop:

```text
HRM
 |
 v
allocation
 |
 v
RAIRAW
 |
 v
feedback
 |
 v
H_MEM
 |
 +------> HRM
```

The system can progressively improve its allocation based on what it has actually observed.

---

# 12. RAIRAW local learning

Once a RAIRAW receives its assigned main weights, its internal components jointly determine the local learning behavior.

Conceptually:

```text
assigned weights
      |
      v
+-------------+
| RAIRAW      |
|             |
| R  A  I     |
|     |       |
|     C       |
+------+------+
       |
       v
local update policy
```

The RAIRAW does not replace the main model.

It controls the learning behavior of its assigned region.

---

# 13. Retention-plasticity decision

The central local decision can be conceptualized as:

```text
High R + Low A
    -> protect

Low R + High A
    -> learn directly

High R + High A
    -> preserve main weight and consider adapter

Low R + Low A
    -> minimal update
```

This is not a fixed policy.

The recursive Controller is responsible for learning the appropriate behavior.

---

# 14. Separate Adapter Weight Pool

RAIRAW-V1 contains a second, independent parameter pool for newly created adaptation capacity.

Example:

```text
ADAPTER_WEIGHT_POOL = 10,000
```

This pool is separate from:

```text
MAIN_WEIGHT_POOL
```

Therefore adapter allocation does not consume the main knowledge-weight budget.

When a RAIRAW determines that a region is highly important for existing knowledge but also requires substantial new learning, the Controller can request adapter capacity.

Conceptually:

```text
RAIRAW
 |
 +-- Retention HIGH
 |
 +-- Attention HIGH
 |
 +-- Intent = adapt
 |
 v
Controller
 |
 v
request adapter
 |
 v
ADAPTER_WEIGHT_POOL
 |
 v
new adaptive parameters
```

A LoRA-like or other lightweight adapter mechanism can be investigated as the first implementation.

---

# 15. Why adapters matter

The stability/plasticity problem can be expressed as:

```text
Existing knowledge wants:
    DON'T CHANGE W

New task wants:
    CHANGE W
```

RAIRAW provides a third option:

```text
KEEP W
 +
ADD ADAPTATION
```

Therefore:

```text
Existing W
     |
     | protected
     v
   W_i
     +
 Adapter_i
     |
     v
New behavior
```

This creates a potential mechanism for learning new information without forcing the original representation to absorb all of the new task.

---

# 16. Global vs local authority

The architecture has explicit hierarchical authority.

### HRM

Global authority:

> Which RAIRAW models should be activated and where computational/weight capacity should be allocated?

### RAIRAW

Local authority:

> How should the assigned region behave during learning?

### H_MEM

Persistent influence memory:

> What has the RAIRAW system learned about the influence of allocated regions?

### Adapter Controller

Expandable plasticity:

> Should new parameters be allocated instead of modifying protected knowledge?

Thus:

```text
GLOBAL
  |
  v
HRM
  |
  +---------------------------+
  |                           |
  v                           v
RAIRAW allocation         Adapter allocation
  |
  v
LOCAL
  |
  v
R / A / I / Controller
```

---

# 17. Core architectural equation

A simplified abstraction is:

```text
HRM:
    allocation = f(input, H_MEM, resource_state)

RAIRAW:
    local_state = f_RAI(input, assigned_weights, context)

Controller:
    action = f_C(R, A, I, HRM_context)

H_MEM:
    H_MEM <- update(H_MEM, RAIRAW_feedback)

Adapter:
    A_weights <- allocate_if_required(action)
```

The exact mathematical implementation is intentionally left open for Experiment Series 2.

---

# 18. Key hypothesis

The Series-2 hypothesis is:

> If the model's learning capacity is organized into small recursive RAIRAW models, and the HRM dynamically allocates these models over a bounded main-weight pool, then the system should be able to concentrate learning on influential regions while preserving less relevant regions. RAIRAW feedback can create H_MEM naturally through normal model communication, allowing subsequent HRM allocations to become increasingly informed. A separate adapter-weight pool may provide additional plasticity when modifying retained knowledge is unsafe.

---

# 19. Experimental progression

## Experiment 2.1 — RAIRAW model size

Test:

```text
RAIRAW controller/model:
256
512
1,000
2,000
```

Measure:

- forgetting
- new-task accuracy
- final accuracy
- computational cost

---

## Experiment 2.2 — RAIRAW assigned weight capacity

Test:

```text
128 weights
256 weights
512 weights
1,000 weights
```

per RAIRAW model.

Determine whether small RAIRAW models can effectively control small weight regions.

---

## Experiment 2.3 — Number of active RAIRAW models

Test:

```text
1
2
5
10
20
```

and determine whether the HRM learns an appropriate allocation.

---

## Experiment 2.4 — Dynamic allocation

Compare:

```text
fixed number of RAIRAW models
vs
HRM-selected number of RAIRAW models
```

The objective is to establish whether dynamic allocation itself provides an advantage.

---

## Experiment 2.5 — H_MEM emergence

Compare:

```text
H_MEM disabled
H_MEM initialized empty
H_MEM generated by RAIRAW feedback
```

The key measurement is whether naturally generated influence memory improves future allocation.

---

## Experiment 2.6 — Adapter pool

Compare:

```text
direct modification
vs
RAIRAW + adapter
```

Measure:

- retention
- new-task acquisition
- final accuracy
- adapter parameter usage
- main-weight modification
- total parameter growth

---

## Experiment 2.7 — Full continual-learning test

Use the established protocol:

```text
5 tasks
10 seeds
paired baseline
multiple task orders
```

Report:

- average forgetting
- overwritten-task forgetting
- final average accuracy
- new-task acquisition
- per-task retention
- active RAIRAW count
- main weights modified
- adapter parameters allocated

---

# 20. Success criteria

Series 2 should not be considered successful merely because forgetting decreases.

The primary criterion remains the stability-plasticity balance:

```text
FORGETTING        ↓
NEW-TASK ACC      ↑
FINAL ACC         ↑
```

A successful RAIRAW system should ideally:

1. reduce forgetting relative to HRM + H_MEM,
2. maintain or improve new-task acquisition,
3. dynamically allocate fewer effective parameters than the complete main pool,
4. demonstrate that H_MEM can emerge from RAIRAW feedback,
5. use adapters selectively rather than indiscriminately,
6. remain robust across seeds and task orders.

---

# 21. Central research question

The fundamental question of Series 2 is:

> **Can a neural network become a hierarchy of small recursive learning systems that dynamically allocate, remember, protect, and expand its own effective parameter space instead of treating the entire parameter matrix as one homogeneous learning surface?**

RAIRAW-V1 is the first architecture proposed to test this hypothesis.

---

# 22. Architectural principle

The final abstraction is:

```text
                  HRM
                   |
             "WHERE?"
                   |
                   v
                RAIRAW
                   |
        +----------+----------+
        |          |          |
        R          A          I
        |          |          |
        +----------+----------+
                   |
                   v
              Controller
                   |
             "HOW?"
              /     \
             /       \
       modify W     adapter
             \       /
              \     /
               H_MEM
                  |
                  v
             future HRM
```

In one sentence:

> **RAIRAW-V1 transforms the model from a flat collection of weights into a hierarchy of small recursive learning elements, while preserving the HRM's global authority and allowing influence memory and adaptive capacity to emerge through communication.**

---

# 23. Scope boundary

RAIRAW-V1 is an experimental architecture.

The following are hypotheses, not established facts:

- that ~1K RAIRAW model size is optimal,
- that ~1K assigned weights per RAIRAW is optimal,
- that RAIRAW feedback is sufficient to create useful H_MEM,
- that dynamic RAIRAW allocation improves continual learning,
- that adapters will outperform direct weight modification,
- that the architecture scales beyond the tiny experimental model.

These must be established experimentally.

The initial objective is therefore **architecture verification**, not a claim of solving catastrophic forgetting universally.
