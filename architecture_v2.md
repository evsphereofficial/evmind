# EvMind Architecture
## A Dynamically Composable, Continually Learning Neural Architecture

> **Status:** Research / experimental architecture hypothesis
> **Revision:** Added recursive / HRM-inspired node architecture, adaptive computation depth, recursive controllers, and expanded node taxonomy.  
> **Project:** EvMind  
> **Author concept:** EV ecosystem / evolving neural-node architecture  
> **Document purpose:** Preserve the complete architectural hypothesis, motivations, assumptions, mechanisms, and experimental roadmap before implementation changes the original idea.

---

# 1. Executive Summary

EvMind is not intended to be a conventional LLM, a larger Transformer, or simply an MoE model with more experts.

The central hypothesis is that intelligence does not have to be represented as one monolithic parameterized neural network. Instead, an intelligent system can be represented as a **persistent, dynamically evolving population of small, independently trainable neural nodes** that can be composed into temporary computational graphs according to the requirements of each request.

The fundamental unit of the architecture is therefore the **node**, not the parameter count of one global model.

A node may be tiny: 10K, 50K, 100K, 500K, 1M parameters, or another size determined by the capability it represents. Nodes may use different neural architectures. They do not have to be Transformers. The initial direction discussed for EvMind is strongly interested in **diffusion-based nodes**, while allowing other architectures where appropriate.

The architecture is built around several principles:

1. **Nodes are independently trainable computational units.**
2. **Different nodes can specialize in different capabilities.**
3. **New learning can create new nodes instead of globally modifying established knowledge.**
4. **Nodes can be frozen, unfrozen, loaded, unloaded, merged, split, or retired.**
5. **A controller dynamically selects and composes nodes for each task.**
6. **Only the nodes needed for the current computation need to reside in VRAM.**
7. **The persistent node pool may be much larger than active computational memory.**
8. **Memory can itself be represented by specialized neural nodes rather than requiring a gigantic token/KV-cache context.**
9. **Highly overfitted nodes are not automatically a weakness; specialized memory nodes may intentionally overfit their learned distribution.**
10. **Intelligence is treated as emerging from interactions between cognitive nodes, knowledge/memory nodes, perception nodes, skill nodes, tool nodes, and controllers.**
11. **The architecture is extensible: new node types can be added without retraining the entire system.**
12. **The long-term objective is a practical path toward broad, continual, multimodal, resource-efficient intelligence rather than merely a larger language model.**

A useful analogy is a **Framework Laptop** or modular computer: the chassis and interface remain stable while computational capabilities are replaceable and extensible. Another analogy is **Tetris or a puzzle**: every request creates a different temporary arrangement of components from the persistent node pool.

---

# 2. Origin and Motivation

The original research question was not initially a complete architecture.

The starting ambition was roughly:

> Build something fundamentally different from current parameter-scaled frontier models, capable of live/continual learning while remaining small and usable on constrained consumer hardware.

The earlier goal included three broad interests:

- **Curiosity**
- **Understanding / thinking**
- **Memory**

At that stage, there was no concrete mechanism for achieving it.

The central dissatisfaction with conventional architectures was that increasing capability is usually framed as:

- larger parameter counts,
- larger context windows,
- more training,
- more compute,
- more GPUs,
- and increasingly large fixed models.

The desired alternative is:

> **Do not make one neural network contain everything. Build a system that can grow by adding specialized computational pieces.**

This leads to the EvMind architectural hypothesis.

---

# 3. Core Architectural Thesis

## 3.1 Intelligence Is Not the Same Thing as Knowledge

A central distinction in EvMind is:

> **Knowledge is a substrate. Intelligence is what operates on, organizes, questions, combines, and acts upon that knowledge.**

A knowledge node may be extremely narrow and highly specialized without being intelligent by itself.

For example:

- a knowledge node may contain a specialized physics distribution,
- a memory node may encode a particular interaction history,
- a perception node may understand visual structure,
- a thinking node may reason over representations supplied by other nodes,
- a curiosity node may identify uncertainty and knowledge gaps,
- a planning node may construct actions,
- a tool node may interact with an external environment.

This separates knowledge acquisition from cognition.

---

# 4. Node as the Fundamental Unit

## 4.1 Definition

A **node** is a small independently addressable computational model or computational capability.

A node may contain:

- parameters,
- learned state,
- embeddings or latent representations,
- specialization metadata,
- capability descriptors,
- confidence/reliability estimates,
- relationships with other nodes,
- resource requirements,
- activation policies,
- provenance/history,
- and optional persistent state.

The node is the architectural equivalent of a replaceable module.

## 4.2 Node Size

There is no assumption that every node has the same parameter count.

Potential node scales include:

- 10K parameters
- 50K parameters
- 100K parameters
- 500K parameters
- 1M parameters
- several million for complex specialist nodes

Tiny nodes are especially important because they may be cheap enough to move between storage, RAM, and VRAM dynamically.

A 1M-parameter node is roughly:

- ~2 MB for FP16 weights,
- ~1 MB for INT8 weights,
- ~0.5 MB for 4-bit weights,

before runtime activations, metadata, buffers, and other overhead.

These figures are illustrative; actual runtime residency is larger.

## 4.3 Node Count vs Parameter Count

Traditional model comparisons emphasize:

> 7B vs 70B vs 405B parameters.

EvMind instead emphasizes:

- node count,
- node specialization,
- node topology,
- node quality,
- node relationships,
- active node count,
- active compute,
- persistent node count,
- and dynamic resource allocation.

Total persistent intelligence may consist of thousands or hundreds of thousands of nodes while only a small subset is active.

---

# 5. Node Types

Node types are not necessarily fixed forever. The following are initial categories.

## 5.1 Thinking / Reasoning Nodes

Responsibilities:

- reasoning,
- logic,
- planning,
- problem solving,
- decision making,
- hypothesis formation,
- comparison,
- abstraction,
- verification,
- multi-step cognition.

These are not expected to contain all knowledge.

They should be able to reason over representations provided by other nodes.

Thinking nodes may use recurrent or recursive computation. An HRM-inspired node is one candidate: a compact model can repeatedly refine an internal reasoning state, potentially operating at different computational timescales instead of requiring a large permanently stacked parameter count.

## 5.2 Memory / Context Nodes

Memory nodes are central to the architecture.

They are intended to represent persistent context through learned neural state rather than relying exclusively on a conventional token window and KV cache.

Important properties:

- highly specialized,
- potentially deliberately overfitted,
- persistent,
- independently loadable,
- trainable over time,
- freezable,
- reactivatable,
- potentially representing extremely long-lived context.

A memory node may represent a conversation, project history, domain-specific knowledge, or another persistent context distribution.

## 5.3 Curiosity Nodes

Curiosity is treated as an explicit computational capability rather than something that must mysteriously emerge from next-token prediction.

Curiosity nodes can:

- identify uncertainty,
- detect missing information,
- ask questions,
- identify unexplained observations,
- seek evidence,
- request tools,
- request additional memory,
- propose exploration,
- initiate information acquisition.

Potential loop:

```text
Uncertainty
    ↓
Curiosity
    ↓
"What information is missing?"
    ↓
Memory / perception / tool request
    ↓
New observation
    ↓
Reasoning
    ↓
New hypothesis
    ↓
Further curiosity
```

## 5.4 Skill Nodes

Skill nodes contain specialized competencies such as:

- coding,
- mathematics,
- writing,
- domain-specific reasoning,
- debugging,
- scientific procedures,
- educational procedures,
- specialized transformations.

Skill nodes can be independently developed and added without retraining the entire system.

## 5.5 Perception Nodes

Responsibilities include:

- image understanding,
- video understanding,
- audio understanding,
- sensor interpretation,
- OCR,
- spatial perception,
- temporal perception,
- multimodal grounding.

Perception nodes transform raw sensory information into representations useful to cognitive nodes.

## 5.6 Tool Nodes

Tool nodes provide interfaces to external capabilities.

Examples:

- web search,
- code execution,
- filesystem,
- databases,
- APIs,
- browser,
- simulators,
- external environments,
- software applications,
- EvAgent.

A tool node does not necessarily need to be a neural model. It can be an interface exposed through the EvMind node protocol.

## 5.7 Generation Nodes

Generation can be specialized by modality:

- text generation,
- image generation,
- video generation,
- audio generation,
- image editing,
- video transformation,
- structured generation.

Diffusion-based generation nodes are a major direction.

## 5.8 Controller / Attention / Meta Nodes

Controller nodes determine:

- which nodes are relevant,
- which nodes should be loaded,
- which nodes should be frozen,
- which nodes should receive attention,
- which nodes should be combined,
- whether a new node is necessary,
- how resources should be allocated,
- whether nodes should be retired or merged,
- how new nodes relate to old nodes.

This is not necessarily identical to Transformer attention.

It is closer to **learned arbitration over a population of computational modules**.

---

# 6. Nodes Do Not Have to Be Transformers

EvMind is intentionally not restricted to a single model family.

Nodes may be:

- diffusion models,
- Transformers,
- CNNs,
- state-space models,
- recurrent / recursive models,
- HRM-style hierarchical recursive reasoning models,
- reinforcement-learning policies,
- specialized generative models,
- learned compressors,
- deterministic algorithms,
- tool interfaces,
- future architectures not yet invented.

The initial research direction places substantial emphasis on **diffusion-based neural nodes**.

This is motivated by the idea that a node may be more useful as a small generative/refinement/reconstruction process than as a miniature autoregressive language model.

---

# 7. Why Diffusion Nodes?

A Transformer language model typically operates through sequential token prediction.

A diffusion model instead learns iterative refinement or reconstruction from a noisy/incomplete state.

Conceptually:

```text
Noise / incomplete state
        ↓
Denoising
        ↓
Denoising
        ↓
Denoising
        ↓
Coherent latent / output
```

This may be useful for nodes responsible for:

- memory reconstruction,
- hypothesis generation,
- state refinement,
- plan refinement,
- visual generation,
- video generation,
- multimodal transformation,
- specialized distributions.

A diffusion node can still contain attention internally. “Diffusion node” does not mean “no attention.”

Likewise, a recursive node may contain attention, diffusion components, gated state updates, or another internal mechanism. The EvMind node abstraction is deliberately higher-level than the neural architecture used inside the node.

---

# 8. Deliberate Overfitting as a Feature

Traditional machine learning typically treats overfitting as a problem because the goal is broad generalization.

EvMind changes the objective for certain nodes.

A memory node may intentionally be trained to become extraordinarily specialized to its own distribution.

For example:

```text
Memory Node X

Training distribution:
- conversation history
- project history
- recurring terminology
- specific user interactions
- specialized documents
```

The goal is not:

> Generalize this information to everything.

The goal is:

> **Retain and reconstruct this specific distribution exceptionally well.**

Thus:

```text
Overfitting
    ↓
bad for universal general-purpose nodes
    ↓
potentially useful for specialized memory nodes
```

This is analogous to using a highly specialized component rather than forcing every component to be general.

---

# 9. Memory as Learned Neural State

A conventional context window is generally implemented through token processing and attention state, often represented at inference time by a KV cache.

EvMind explores replacing long-lived context with persistent learned nodes.

The key hypothesis is:

> **A persistent context can become a small executable neural representation instead of a massive activation cache.**

The intended mechanism is not simply retrieval from a raw database.

Instead:

```text
Experience
    ↓
Model produces KV state
    ↓
KV state is transferred to a specialized context-node trainer
    ↓
Node learns the state/distribution
    ↓
Node becomes persistent
    ↓
Original KV cache can be released
```

Later:

```text
Context Node
    ↓
load node
    ↓
reconstruct usable state
    ↓
continue inference
```

The context node is intended to become the **persistent neural representation of the context**.

It is not literally a conventional KV cache, because a KV cache contains per-layer keys and values for specific attention computations, but the research goal is functional equivalence:

> **Can the neural node replace the persistent need for the original KV state while allowing the model to behave as though the history remained available?**

---

# 10. Neural KV-State Replacement Hypothesis

Let the original KV cache be:

```text
C = {K1, V1, K2, V2, ..., KL, VL}
```

A specialized context node is trained from the state:

```text
Nθ ← Train(C)
```

Later, the node reconstructs a usable representation:

```text
Ĉ = Decoder(Nθ)
```

The goal is not necessarily bit-for-bit equality.

The important criterion is downstream behavior:

```text
Behavior(LLM, C)
≈
Behavior(LLM, Ĉ)
```

Potential loss:

```text
L =
  L_task
  + λ * L_state
```

where:

- `L_task` measures downstream behavior/performance,
- `L_state` encourages preservation of relevant state information.

The most important objective may ultimately be **functional fidelity**, not numerical equality of the original tensors.

---

# 11. Memory Capacity Hypothesis

A tiny model cannot losslessly store arbitrary unlimited data.

EvMind does not claim that it can.

Instead, the hypothesis is:

- real-world interactions often contain structure and redundancy,
- specialized nodes can learn distributions and associations,
- a tiny neural model may represent a surprisingly large amount of useful information,
- memory capacity should be measured empirically.

Potential experimental curve:

```text
Node size:
10K
50K
100K
500K
1M

Context:
8K
16K
32K
64K
128K
256K
1M
10M
100M+
```

Measure:

- factual recall,
- semantic recall,
- sequence recall,
- relationship recall,
- behavioral fidelity,
- downstream answer accuracy,
- reasoning continuity,
- compression ratio,
- latency,
- VRAM usage.

---

# 12. Functional Context vs Token Context

The important architectural shift is:

### Conventional

```text
Tokens
  ↓
Embedding
  ↓
Transformer
  ↓
KV cache
  ↓
future attention
```

### EvMind hypothesis

```text
Experience
  ↓
specialized context node
  ↓
persistent learned state
  ↓
load / reconstruct when needed
  ↓
thinking / generation nodes
```

This allows the concept of “context” to become:

> **a persistent computational object rather than merely a count of tokens.**

The effective context could therefore theoretically exceed conventional fixed context-window scales, subject to the actual capacity and fidelity of the learned node.

---

# 13. The Node Pool

The persistent node pool is the equivalent of a huge collection of modular computational components.

Example:

```text
Node Pool
├── N1   Thinking
├── N2   Memory
├── N3   Curiosity
├── N4   Vision
├── N5   Coding
├── N6   Physics
├── N7   Video
├── N8   Planning
├── N9   Tool
├── ...
└── N100000+
```

The entire pool does not need to be loaded into VRAM.

This is one of the most important differences from the conventional “one model loaded into memory” mentality.

---

# 14. Storage Hierarchy

EvMind treats hardware memory as a hierarchy.

```text
Persistent node pool
        ↓
      SSD
        ↓
       RAM
        ↓
      VRAM
        ↓
   Active compute
```

Potentially:

```text
SSD:
    enormous persistent node population

RAM:
    cached likely-to-be-used nodes

VRAM:
    currently active nodes

GPU compute:
    actively executing subset
```

The total persistent intelligence can therefore exceed the amount of simultaneously resident computation.

---

# 15. Dynamic Node Residency

Suppose a machine has 20 GB VRAM.

The persistent node pool may contain thousands of nodes.

Only a subset is active.

Example:

```text
Node 17    ACTIVE
Node 42    ACTIVE
Node 91    ACTIVE
Node 104   ACTIVE

Node 120   UNLOADED
Node 121   UNLOADED
...
Node 50000 UNLOADED
```

A new request arrives.

The controller determines that a new node is needed.

```text
Current active nodes
        ↓
VRAM pressure
        ↓
identify low-value node
        ↓
evict / freeze node
        ↓
load required node
        ↓
continue computation
```

Eviction does not mean deletion.

The node remains in persistent storage and can later be reactivated.

---

# 16. Dynamic Computation as a Puzzle / Tetris System

The best conceptual model for a request is:

> **Puzzle pieces / Tetris pieces.**

Each request is solved by assembling a different combination of nodes.

Example:

### Chatting

```text
Memory + Thinking + Curiosity + Language
```

### Mathematics

```text
Thinking + Mathematics Skill + Knowledge + Verification
```

### Image Analysis

```text
Perception + Thinking + Memory + Skill
```

### Research

```text
Curiosity + Planning + Tool + Knowledge + Thinking
```

### Video Physics

```text
Video Perception + Vision + Physics + Thinking + Memory
```

No single node must contain the complete capability.

The system composes multiple pieces.

---

# 17. Dynamic Node Graph

For a request, the controller constructs a temporary graph.

Example:

```text
User
 ↓
Controller
 ↓
Memory ──────────┐
                 │
Vision ──────────┼──→ Thinking
                 │      ↓
Knowledge ───────┤   Planning
                 │      ↓
Tool ────────────┘    Action
```

Another request may form a completely different graph.

Thus the model does not have one permanent fixed computation path.

---

# 18. Node Controller

The EvMind Controller is the system’s high-level orchestration layer.

Potential responsibilities:

- intent understanding,
- node selection,
- workflow planning,
- node routing,
- resource management,
- node residency,
- performance evaluation,
- node evolution,
- node ranking,
- node creation,
- node retirement,
- node merging,
- node freezing,
- node unfreezing.

Possible controller components:

```text
EvMind Controller
├── Intent Understanding
├── Node Selection
├── Workflow Planner
├── Resource Manager
├── Performance Evaluator
└── Evolution Manager
```

The controller may itself be composed of one or more meta-nodes.

---

# 19. Learned Node Attention / Arbitration

New learning should not automatically invalidate existing nodes.

Suppose:

```text
N_old = previous knowledge
N_new = newly learned knowledge
```

Instead of modifying:

```text
N_old → N_old'
```

the system can retain both.

The controller learns:

```text
Context A → N_old
Context B → N_new
Context C → N_old + N_new
```

The new node may eventually receive greater activation for relevant situations.

The fundamental operation becomes:

```text
Old node remains intact
New node is added
Controller learns when to use each
```

This is the proposed attack on catastrophic forgetting.

---

# 20. Continual Learning

Traditional continual learning often has a difficult tradeoff:

```text
New data
   ↓
modify existing parameters
   ↓
knowledge interference
   ↓
catastrophic forgetting
```

EvMind instead proposes:

```text
New experience
    ↓
evaluate existing nodes
    ↓
sufficient?
 /        \
yes        no
 |          |
update     spawn new node
 |          |
freeze   train/specialize
            ↓
         register node
            ↓
        connect node
```

Existing mature nodes can remain frozen while new nodes accumulate new knowledge.

This does not eliminate all interference. Routing, composition, controller learning, and node interactions can still fail. Those must be experimentally studied.

---

# 21. Node Lifecycle

Every node can potentially follow a lifecycle.

## 21.1 Birth

A node is:

- created from scratch,
- spawned from an existing node,
- cloned and specialized,
- or introduced externally.

## 21.2 Train / Evolve

The node learns from:

- new experiences,
- specific tasks,
- domain data,
- interactions,
- KV states,
- generated examples,
- tool observations,
- other nodes.

## 21.3 Specialize

The node becomes increasingly effective in its domain.

Specialization can be intentional.

A memory node can become highly overfit.

A reasoning node may remain broader.

## 21.4 Deploy

The node enters the persistent node pool.

The controller can activate it whenever relevant.

## 21.5 Freeze

A node may be frozen to protect learned behavior.

Freezing can mean:

- no parameter updates,
- no training,
- no computational execution,
- no modification,
- or simply not being resident in VRAM.

These are separate concepts and should be implemented distinctly.

## 21.6 Reactivate

A frozen/unloaded node can later be loaded and used when relevant.

## 21.7 Merge

Related nodes may potentially be merged if doing so provides a better representation.

## 21.8 Split

An overloaded or overly broad node may be divided:

```text
N42
 ↓
N42a
N42b
N42c
```

## 21.9 Retire

Outdated or redundant nodes may be archived or removed from active consideration.

---

# 22. Node Growth

Node growth does not have to mean one node becoming arbitrarily large.

Possible strategies:

### Expand

```text
100K → 250K → 500K
```

### Spawn

```text
N42 → N42a + N42b
```

### Specialize

```text
general node
    ↓
domain-specific child
```

### Connect

```text
existing node + new node
```

The goal is to let capability grow structurally.

---

# 23. Resource-Aware Node Scheduling

The scheduler may consider:

- relevance,
- confidence,
- capability,
- dependencies,
- latency,
- VRAM requirements,
- RAM requirements,
- storage location,
- expected future use,
- node compatibility.

A simplified score:

```text
Node Score =
    relevance
  + capability
  + confidence
  + expected utility
  - latency
  - memory cost
```

The exact formulation is an experimental design choice.

---

# 24. Parallel and Serial Node Loading

SSD/RAM-to-VRAM latency is one of the main engineering concerns.

EvMind can hide some latency through asynchronous scheduling.

Example:

```text
Input
 ↓
Controller predicts:
Batch A, Batch B, Batch C may be needed
 ↓
Load in parallel where possible
 ↓
Compute Batch A
 ↓
Batch B becomes available
 ↓
Compute Batch B
 ↓
Batch C becomes available
 ↓
Compute Batch C
```

The architecture can therefore use:

- parallel node loading,
- serial dependency loading,
- asynchronous prefetching,
- speculative node loading,
- cache retention,
- eviction prediction.

The controller may eventually learn:

```text
P(Node_i | current state)
```

and prefetch nodes before they are actually required.

---

# 25. Neural Virtual Memory Concept

EvMind can be viewed as having a neural equivalent of virtual memory:

```text
Persistent intelligence
        ↓
SSD / RAM
        ↓
predicted-needed nodes
        ↓
VRAM
        ↓
active cognition
```

Only a working set is resident.

This separates:

```text
total intelligence capacity
```

from:

```text
currently resident intelligence
```

and:

```text
currently executing computation
```

These are distinct quantities.

---

# 26. Universal Node Communication

Because nodes may operate across modalities, EvMind requires a common node communication protocol.

The goal is not necessarily one universal embedding vector for everything.

Instead, the likely architecture is:

> **Universal node interface + shared cognitive latent/protocol + modality-specific representations.**

A node could expose:

```text
NodeMessage
├── modality
├── representation
├── semantic state
├── spatial state
├── temporal state
├── confidence
├── metadata
└── requests
```

The internal representation remains node-specific where necessary.

---

# 27. Shared and Modality-Specific Latents

Potential architecture:

```text
                 Shared Cognitive Space
                         ↑
          ┌──────────────┼──────────────┐
          ↓              ↓              ↓
       image latent   text latent   video latent
          ↓              ↓              ↓
      Vision Node     Language      Video Node
                         Node
```

A Thinking Node can operate on evidence produced by other nodes.

This avoids forcing pixels, language, audio, video, memory, and tool results into one mathematically identical representation.

---

# 28. Multimodal Architecture

EvMind should support:

- text input,
- images,
- video,
- audio,
- sensor data,
- generated images,
- generated video.

Example:

```text
Image
 ↓
Vision Node
 ↓
Visual representation
 ↓
Thinking Node
```

For video:

```text
Video
 ↓
Video Perception Node
 ↓
Temporal representation
 ↓
Vision / Reasoning Nodes
```

A Thinking Node can request specific information:

```text
"I need the five seconds before the event."
```

The relevant video node activates only the needed segment.

---

# 29. Image and Video Generation

Generation nodes can be separate components.

Example:

```text
Thinking
   ↓
"I need a diagram."
   ↓
Image Generation Node
   ↓
Generated image
   ↓
Vision Node
   ↓
Critique
   ↓
Image Generation Node
   ↓
Revised image
```

For video:

```text
Planning
 ↓
Video Generation
 ↓
Video Perception
 ↓
Critique
 ↓
Revision
```

This creates a:

```text
generate → perceive → critique → regenerate
```

loop.

---

# 30. Tool Integration and EvAgent

EvMind can expose tool nodes.

EvAgent can serve as an execution/orchestration interface rather than forcing all tool use into neural weights.

Potential flow:

```text
Curiosity
 ↓
"I need external information."
 ↓
Tool Node
 ↓
EvAgent
 ↓
web / code / filesystem / APIs / applications
 ↓
Observation
 ↓
Knowledge / Memory Node
 ↓
Thinking
```

This allows EvMind to acquire information rather than requiring all information to be pre-trained into weights.

EvAgent can become one of the major capabilities in the EV ecosystem.

---

# 31. Curiosity as a Core Cognitive Primitive

Curiosity is one of the most important research directions.

Instead of:

```text
prompt → answer
```

EvMind can operate:

```text
goal
 ↓
current knowledge
 ↓
uncertainty detection
 ↓
curiosity
 ↓
information request
 ↓
tool / perception / memory
 ↓
new evidence
 ↓
hypothesis
 ↓
testing
 ↓
learning
```

A mature system should eventually be able to determine:

> “I don't know enough.”

and then:

> “What do I need to know?”

and then:

> “How can I obtain it?”

This is closer to autonomous learning than static prediction.

---

# 32. Example Curious Learning Loop

```text
Observation
   ↓
Thinking Node
   ↓
"Something does not fit."
   ↓
Curiosity Node
   ↓
Identify knowledge gap
   ↓
Search / tool / perception
   ↓
New observation
   ↓
Spawn or activate knowledge node
   ↓
Thinking
   ↓
Hypothesis
   ↓
Verification
   ↓
Persist learning
```

This is one of the major pathways toward general learning behavior.

---

# 33. Tools Are Part of the Cognitive Architecture

Tool nodes are not merely peripheral APIs.

They can be treated as action capabilities.

Examples:

```text
Search Node
Calculator Node
Python Node
Browser Node
Filesystem Node
Database Node
Simulation Node
EvAgent Node
Robotics Node
```

This enables:

```text
knowledge → reasoning → action → observation → learning
```

rather than knowledge-only cognition.

---

# 34. Framework Laptop Analogy

EvMind should be thought of as more like a **Framework Laptop** than a monolithic LLM.

The framework chassis and interface remain stable.

Capabilities are modular.

Example:

```text
Core framework
   +
Thinking Node
   +
Vision Node
   +
Video Node
   +
Physics Node
   +
Robotics Node
```

Adding a new node should not require retraining the entire system.

A node can be registered through the node protocol.

The controller can learn:

> “When is this node useful?”

without rebuilding the entire neural architecture.

This is an important design principle:

> **Adding capability should be an extension operation, not a global retraining operation.**

---

# 35. Endless Node Extensibility Hypothesis

Potential node types are not predetermined.

Future nodes could include:

- chemistry,
- biology,
- robotics,
- music,
- spatial reasoning,
- 3D reasoning,
- legal reasoning,
- scientific simulation,
- gaming,
- mathematical proof,
- hardware control,
- speech,
- emotion modeling,
- agentic planning,
- unknown future architectures.

If a computational capability can implement the node protocol, it can theoretically participate.

Therefore:

> The system is not limited to a single model family.

---

# 36. EvMind as a Framework, Not an LLM

The most accurate conceptual description is not:

> “An LLM with modular memory.”

It is closer to:

> **A dynamically composable neural computing framework containing persistent, trainable, specialized computational nodes that can be assembled into task-specific cognitive systems.**

The LLM, if used at all, is only one possible node type.

The architecture itself is independent of language models.

---

# 37. Difference From Conventional Parameter Scaling

Traditional scaling:

```text
More capability
    ↓
larger model
    ↓
more parameters
    ↓
more compute
    ↓
more memory
```

EvMind hypothesis:

```text
More capability
    ↓
new nodes
    ↓
specialization
    ↓
new relationships
    ↓
better routing
    ↓
larger persistent node population
```

Parameters remain an implementation detail of nodes rather than the primary abstraction of the entire intelligence.

---

# 38. Difference From Mixture-of-Experts

EvMind is not simply MoE.

MoE typically has a predefined expert population and a router that activates only some experts.

EvMind proposes:

- independently persistent nodes,
- arbitrary node architectures,
- dynamic node creation,
- node evolution,
- deliberate freezing,
- persistent node storage outside VRAM,
- dynamic residency,
- potentially extremely different node types,
- tool nodes,
- memory nodes,
- cognitive nodes,
- asynchronous node loading,
- architectural growth over time.

Most importantly:

> **An EvMind node is intended to be an independently meaningful computational object, not merely an expert partition inside a fixed giant model.**

---

# 39. Difference From Conventional Context Windows

Traditional context:

```text
tokens
 ↓
attention
 ↓
KV cache
 ↓
more attention
```

EvMind hypothesis:

```text
persistent learned context nodes
 ↓
load relevant node(s)
 ↓
neural reconstruction / state exposure
 ↓
thinking nodes
```

Context becomes a persistent neural capability rather than only a temporary sequence.

---

# 40. Difference From External Database Memory

EvMind memory nodes are not merely:

```text
database → retrieve text → inject prompt
```

The stronger hypothesis is:

```text
data / experience
 ↓
learned neural memory representation
 ↓
node itself becomes persistent learned state
 ↓
node participates directly in computation
```

Raw external storage can still exist as a backup or source of truth, but the primary research goal for some memory nodes is learned neural state.

---

# 41. Request Workflow

A request may follow this pipeline:

```text
1. User makes a request
        ↓
2. Controller understands intent
        ↓
3. Controller identifies required capabilities
        ↓
4. Best node combination is selected
        ↓
5. Nodes are loaded within VRAM constraints
        ↓
6. Nodes communicate and collaborate
        ↓
7. Result is produced
        ↓
8. Result / experience can update or spawn nodes
```

This process can dynamically change during inference.

---

# 42. Runtime Example

User:

> “Why did the object fall in this video?”

Potential execution:

```text
Input
 ↓
Controller
 ↓
Video Node
 ↓
Vision Node
 ↓
Physics Skill Node
 ↓
Thinking Node
 ↓
Memory Node
 ↓
Verification Node
 ↓
Answer
```

If something is missing:

```text
Thinking
 ↓
Curiosity
 ↓
Tool Node
 ↓
External information
 ↓
New knowledge node
 ↓
Thinking continues
```

---

# 43. Active Working Set

For a request, the system may keep only:

```text
10–100 active nodes
```

while maintaining:

```text
10,000+ persistent nodes
```

or more.

The exact ratios are experimental.

The system optimizes the **working set**, not the entire persistent model.

---

# 44. Node Ranking

Nodes should be ranked by more than parameter count.

Potential ranking dimensions:

- task relevance,
- capability,
- confidence,
- historical performance,
- specialization,
- compatibility with other nodes,
- latency,
- memory cost,
- reliability,
- recency,
- usefulness.

Example:

```text
Node #421
Capability: visual-spatial reasoning
Reliability: 0.93
VRAM cost: 40 MB
Latency: low
Compatible with: Physics, Thinking, Planning
```

The controller can use this information when assembling a graph.

---

# 45. Node Compatibility

Nodes may learn useful relationships.

Example:

```text
Vision Node
    ↕ 0.91
Physics Node

Physics Node
    ↕ 0.87
Thinking Node
```

The system can learn that certain combinations are particularly effective.

This creates a learned topology over time.

---

# 46. Node Pool as an Evolving Graph

The node pool is not merely a bag of independent models.

It can become a graph:

```text
N1 ─── N42 ─── N81
 │      │        │
 │      └── N120 ┤
 │               │
 N7 ─────────────┘
```

Edges can represent:

- communication,
- compatibility,
- shared capabilities,
- dependencies,
- learned utility,
- transfer,
- co-activation patterns.

The topology itself can become learned.

---

# 47. New Node Creation

A new node can be created when:

- no existing node has enough capability,
- an existing node is overloaded,
- a new domain appears,
- the controller detects a persistent knowledge gap,
- repeated experiences form a new specialization,
- an existing capability needs a specialized branch.

Potential process:

```text
Need detected
 ↓
Search node pool
 ↓
Candidate exists?
 /           \
yes          no
 |            |
reuse       spawn
             ↓
          train
             ↓
        evaluate
             ↓
        register
             ↓
       connect to graph
```

---

# 48. No Global Retraining Requirement

The intended architectural property is:

```text
Add Node X
    ↓
do NOT retrain all existing nodes
    ↓
register Node X
    ↓
evaluate it
    ↓
teach controller when to use it
```

This does not mean controllers can never be trained or that individual nodes cannot be updated. It means **global re-training is not structurally required to add capability**.

---

# 49. Node Freezing

Freezing is a foundational mechanism.

Nodes can be frozen when:

- they are stable,
- they contain critical memory,
- further training risks degradation,
- they are not currently needed,
- their computational state should be preserved,
- or they are being moved out of active VRAM.

Different freeze states may be tracked separately.

Example:

```text
TRAINABLE
ACTIVE
FROZEN
UNLOADED
ARCHIVED
```

---

# 50. Node Re-Activation

A frozen node can later become relevant.

Example:

```text
New query
 ↓
Controller
 ↓
detect old memory is relevant
 ↓
load node
 ↓
reactivate
 ↓
compose with current nodes
```

This supports persistent accumulation without forcing all learned components into active memory.

---

# 51. Memory Node Example

A memory node may represent an interaction history.

```text
Memory Node:
- 10K–1M parameters
- specialized to a particular context
- trained continuously
- highly overfit
- frozen between sessions
- loaded when context is needed
```

The node is intended to behave as an executable learned representation of the context.

---

# 52. Long-Term Context Goal

The goal is not “unlimited context.”

The goal is:

> **A context representation whose effective capacity is not directly tied to a conventional fixed token window.**

The actual capacity is determined experimentally by:

- node size,
- architecture,
- data entropy,
- training objective,
- reconstruction fidelity,
- node population,
- and available computation.

A population of specialized context nodes can also partition memory.

---

# 53. Hierarchical Memory

Rather than having one giant memory node, memory can be organized hierarchically:

```text
Memory Root
├── Conversation
│   ├── Session A
│   ├── Session B
│   └── Session C
├── Projects
│   ├── EvAgent
│   ├── EvStudy
│   └── EvsMMFPS
├── Knowledge
│   ├── Physics
│   ├── Mathematics
│   └── Programming
└── Experiences
    ├── Successes
    ├── Failures
    └── Procedures
```

The exact hierarchy does not have to be hard-coded.

It can potentially emerge from node organization.

---

# 54. AGI-Oriented Objective

EvMind should not be considered AGI simply because it:

- has many nodes,
- remembers a lot,
- passes an IQ test,
- performs well on benchmarks,
- has a large node pool,
- produces convincing text,
- or beats one domain.

The meaningful target is **general learning ability**.

---

# 55. Unfamiliar-Domain Test

A critical proposed experiment is:

1. Give the system a domain it has never explicitly encountered.
2. Do not manually construct domain-specific nodes.
3. Allow normal tool access.
4. Allow persistent memory.
5. Give it a general objective.
6. Observe whether it can identify its knowledge gaps.
7. Let curiosity seek information.
8. Allow new nodes to be created.
9. Allow reasoning over acquired knowledge.
10. Evaluate whether the resulting capability persists.

A strong result would look like:

```text
Unknown domain
      ↓
"I don't understand this."
      ↓
Curiosity
      ↓
"What information do I need?"
      ↓
Tool / exploration
      ↓
Observation
      ↓
Knowledge node
      ↓
Reasoning
      ↓
Skill acquisition
      ↓
Successful unfamiliar task
      ↓
Persistent capability
```

---

# 56. AGI Evidence Ladder

A practical progression:

### Level 0 — Node substrate
Nodes can be created, loaded, unloaded, frozen and communicate.

### Level 1 — Continual learner
New information does not destroy established knowledge.

### Level 2 — Persistent learner
Knowledge survives sessions and restarts.

### Level 3 — Self-organizing learner
The system chooses when to create, reuse, freeze, split or reactivate nodes.

### Level 4 — Curious learner
The system identifies knowledge gaps and seeks information.

### Level 5 — General learner
The same architecture learns substantially different domains.

### Level 6 — Autonomous problem solver
The system receives unfamiliar goals and constructs its own plans.

### Level 7 — Open-ended learner
The system continuously explores, learns, forms capabilities, and reorganizes itself.

Levels 5–7 would be strong evidence of general-learning behavior.

---

# 57. Evaluation Framework

The first serious experiments should compare EvMind against conventional baselines.

Measure:

## Continual Learning
- Task retention
- Catastrophic forgetting
- Performance over time
- Recovery after contradictory information

## Generalization
- Unseen-domain transfer
- Cross-domain reasoning
- Few-shot adaptation
- Zero-shot adaptation

## Memory
- Recall accuracy
- Long-range recall
- Context reconstruction
- Functional recall
- Memory compression

## Resource Use
- VRAM
- RAM
- SSD
- active node count
- total node count
- latency
- node-loading bandwidth

## Growth
- nodes spawned
- nodes retired
- node specialization
- topology changes
- controller adaptation

## Autonomy
- self-generated questions
- exploration behavior
- tool usage
- knowledge acquisition
- ability to recognize uncertainty

---

# 58. KV-State Experiment

This should be one of the first major experiments.

### Baseline

```text
Conversation
 ↓
Transformer
 ↓
KV Cache
 ↓
Continue inference
```

### EvMind experiment

```text
Conversation
 ↓
Transformer
 ↓
KV Cache
 ↓
Context Node Trainer
 ↓
Tiny context node
 ↓
Delete KV cache
 ↓
Load context node
 ↓
Reconstruct state
 ↓
Continue inference
```

Compare:

- original KV memory,
- node size,
- compression ratio,
- memory usage,
- latency,
- recall,
- reasoning continuity,
- answer agreement.

---

# 59. Compression Ratio

Define:

```text
Compression Ratio =
Original KV Cache Size / Context Node Size
```

Potential result:

```text
500 MB KV
     ↓
5 MB node
     ↓
100× compression
```

The actual useful result is not the ratio by itself.

A 1000× compression that destroys reasoning is worthless.

The goal is a useful point in the:

```text
Compression
    ↕
Fidelity
    ↕
Latency
    ↕
VRAM
```

tradeoff.

---

# 60. Exact vs Functional Memory

Two different tests should be performed.

## Exact memory

Does the node reconstruct numerical KV state close to the original?

## Functional memory

Does the model behave as though the original KV state were still available?

Functional memory is likely the more important target.

Example:

```text
KV similarity: 85%
Downstream performance: 99%
```

may be much more useful than:

```text
KV similarity: 99.9%
Downstream performance: 75%
```

---

# 61. Test Scale

Do not jump immediately to 100M tokens.

Start with:

```text
8K
16K
32K
64K
128K
256K
512K
1M
```

Then scale toward:

```text
10M
50M
100M+
```

Measure how node capacity scales.

The hypothesis is not that a 1M-parameter node magically stores arbitrary 100M-token information losslessly.

The hypothesis is that highly structured contexts may admit very high learned compression.

---

# 62. Latency Problem

Node loading introduces latency.

This is expected.

Potential solutions:

- SSD → RAM prefetch
- RAM → VRAM prefetch
- parallel loads
- serial loads only where dependencies require them
- predictive scheduling
- speculative execution
- node caching
- temporary residency
- node reuse
- asynchronous transfers
- batch loading

Latency is an engineering problem to measure and optimize rather than an immediate reason to reject the architecture.

---

# 63. Example Runtime With Prefetch

```text
Current Node
    ↓
predict next required nodes
    ↓
prefetch likely nodes
    ↓
current computation continues
    ↓
next node becomes available
    ↓
swap/execute
    ↓
predict again
```

This creates a neural computation pipeline similar to virtual memory and speculative execution.

---

# 64. Multi-Node Computation

A request can activate a combination of:

- memory,
- perception,
- thinking,
- skill,
- curiosity,
- tool,
- generation,
- planning,
- verification.

The final answer emerges from cooperation rather than from one monolithic model.

---

# 65. Example: Simple Chat

Input:

```text
"Hi"
```

Potential system:

```text
Language / interaction node
        ↓
Memory node
        ↓
Thinking node
        ↓
Response node
```

Only a few nodes may be loaded.

A complex request may cause dozens of node activations.

---

# 66. Example: Image + Question

Input:

```text
Image + "What is happening here?"
```

Potential execution:

```text
Image
 ↓
Perception Node
 ↓
Visual latent
 ↓
Thinking Node
 ↓
Memory / Knowledge Nodes
 ↓
Answer
```

If further detail is needed:

```text
Thinking
 ↓
request higher-resolution perception
 ↓
Perception Node
```

---

# 67. Example: Research

```text
User goal
 ↓
Curiosity
 ↓
Planning
 ↓
Tool Node
 ↓
Web / APIs
 ↓
Knowledge Node
 ↓
Thinking
 ↓
Verification
 ↓
Memory
 ↓
Result
```

The acquired information can become future capability.

---

# 68. Example: New Capability Appears

Suppose the system encounters many physics-video problems.

Initially:

```text
Vision
Thinking
Physics
```

Performance is insufficient.

Curiosity/controller identifies a repeated gap.

```text
Gap
 ↓
spawn VideoPhysicsNode
 ↓
train
 ↓
specialize
 ↓
evaluate
 ↓
register
```

Future requests can use it immediately without retraining the whole system.

---

# 69. Framework-Level Extensibility

A successful EvMind framework should make it possible to add:

```text
new node
→ define protocol
→ train node
→ register capability
→ evaluate
→ controller learns usage
```

rather than:

```text
new capability
→ retrain entire model
→ redeploy everything
```

This creates an extensible ecosystem.

---

# 70. EV Ecosystem Integration

EvMind can become a common cognitive substrate for the EV ecosystem.

Possible mapping:

```text
EvMind
├── EvsMem       → persistent memory capability
├── EvAgent      → agent execution / tool orchestration
├── EvStudy      → educational skills and knowledge
├── EvsMMFPS     → simulation / generative capabilities
└── Future nodes → new domains and modalities
```

The ecosystem can stop being merely a collection of isolated applications.

Products can become capabilities that participate in the same node ecosystem.

---

# 71. EV Node Ecosystem

Long term, nodes may come from:

- EV development,
- research,
- specialized internal projects,
- external researchers,
- third-party developers.

A node registry could store:

```text
Node ID
Version
Capability
Modality
Input schema
Output schema
Resource requirements
Performance
Reliability
Dependencies
Compatibility
```

EvMind can discover and use them.

---

# 72. Strategic Advantage Hypothesis

Instead of competing solely by training ever-larger monolithic models, EV could focus on:

> **A framework in which intelligence expands through modular capability addition.**

The potential moat becomes:

- node ecosystem,
- protocol,
- controller,
- routing,
- continual-learning mechanisms,
- persistent neural memory,
- dynamic resource manager,
- learned topology,
- and the accumulated library of specialized nodes.

---

# 73. Relationship to Existing EV Projects

EvMind conceptually unifies themes already present in EV work:

### EvsMem
Persistent memory and autonomous knowledge organization.

### EvAgent
Agentic orchestration, execution, tool use, and workflow.

### EvsMMFPS
Specialized generative models and model selection.

### EvStudy
Domain-specific educational intelligence.

EvMind can potentially treat these capabilities as external systems or eventually as node-compatible components.

---

# 74. Hardware Philosophy

EvMind is explicitly designed around constrained consumer hardware.

The goal is not:

> “Fit the entire intelligence into VRAM.”

The goal is:

> **Keep only the current computational working set in VRAM.**

Example:

```text
Persistent nodes: 10,000+
RAM cache:         1,000
VRAM resident:       50
Actively computing:  10
```

Numbers are illustrative.

The important relationship is:

```text
Persistent capacity >> Active capacity
```

---

# 75. Hardware-Aware Intelligence

Hardware management becomes part of cognition.

The controller should understand:

- how much VRAM is available,
- which nodes are expensive,
- which nodes are likely to be reused,
- which nodes can be evicted,
- how long transfers take,
- what can be prefetched,
- what computation can run in parallel.

Thus:

> **Resource management is a first-class cognitive function.**

---

# 76. Potential Node Metadata

Example:

```yaml
node_id: N042
version: 0.3

type: thinking
architecture: diffusion

parameters: 500000

capabilities:
  - visual-spatial reasoning
  - physics reasoning

modalities:
  input:
    - visual_latent
    - text_latent
  output:
    - reasoning_state

resource:
  vrAM_mb: 25
  ram_mb: 30
  latency_ms: 18

state:
  trainable: false
  active: false
  persistent: true
  frozen: true

relationships:
  compatible:
    - N017: 0.91
    - N091: 0.87
```

---

# 77. Node Protocol

A possible minimal conceptual interface:

```python
class Node:
    node_id: str
    node_type: str
    capabilities: list[str]

    def load(self):
        ...

    def unload(self):
        ...

    def activate(self, input_state):
        ...

    def freeze(self):
        ...

    def unfreeze(self):
        ...

    def train(self, experience):
        ...

    def evaluate(self, task):
        ...

    def metadata(self):
        ...
```

This is illustrative rather than a final implementation.

---

# 78. Cognitive Message Protocol

A possible conceptual message:

```python
class NodeMessage:
    modality: str
    latent: object
    semantic_state: object | None
    spatial_state: object | None
    temporal_state: object | None
    confidence: float
    metadata: dict
```

Nodes can communicate through this protocol without requiring identical internal architectures.

---

# 79. Potential Core Runtime

```text
EvMindRuntime
├── NodeRegistry
├── NodePool
├── NodeLoader
├── VRAMManager
├── RAMCache
├── SSDStore
├── NodeRouter
├── Controller
├── WorkflowPlanner
├── CuriosityEngine
├── MemoryManager
├── ToolManager
├── EvaluationEngine
└── EvolutionManager
```

---

# 80. Minimal Prototype

The first implementation does not need full AGI.

A strong minimal prototype could use:

```text
20–100 nodes
```

with:

- a few thinking nodes,
- memory nodes,
- one curiosity node,
- one controller,
- one loader,
- one tool interface,
- one persistent node store.

Demonstrate:

1. node creation,
2. node loading/unloading,
3. node freezing,
4. node communication,
5. node specialization,
6. dynamic routing,
7. continual learning,
8. no global retraining when adding a new capability.

---

# 81. First Killer Experiment

The initial core experiment should compare a conventional continually trained model against EvMind.

### Model A

```text
One model
 ↓
continual training
 ↓
observe forgetting
```

### Model B

```text
many tiny nodes
 ↓
continual node growth
 ↓
freeze old nodes
 ↓
controller learns routing
```

Test:

```text
Task 1
Task 2
Task 3
...
Task 100
```

Periodically test old tasks.

Track:

- retention,
- new-task performance,
- active memory,
- node count,
- global retraining count,
- training cost.

---

# 82. Second Killer Experiment: Neural Context

Compare:

```text
Normal KV cache
vs
Tiny context node
```

At:

```text
8K
32K
128K
512K
1M
```

Measure:

- VRAM,
- latency,
- recall,
- reasoning,
- compression,
- training cost.

---

# 83. Third Killer Experiment: Node Addition Without Retraining

Start:

```text
10 nodes
```

Then introduce a completely new capability.

Add one node.

Do not retrain the entire system.

Measure whether:

```text
Old performance = preserved
New capability = acquired
```

This directly tests the Framework Laptop principle.

---

# 84. Fourth Killer Experiment: Unfamiliar Domain

Give the system an unfamiliar domain.

Do not manually install a custom specialist for the task.

Allow:

- curiosity,
- tools,
- perception,
- memory,
- node creation.

Observe whether it can:

```text
recognize uncertainty
→ acquire information
→ create knowledge
→ reason
→ solve
→ retain
```

This is a major test of general learning.

---

# 85. Scientific Falsifiability

EvMind must remain testable.

It should be possible for experiments to show:

- tiny nodes are insufficient,
- compression is too lossy,
- node routing overhead dominates,
- diffusion nodes are too slow,
- node communication fails,
- continual learning still causes interference,
- the controller becomes unstable,
- node proliferation becomes unmanageable,
- active memory is still too high.

A failed experiment is useful if it identifies the architectural bottleneck.

---

# 86. Major Open Problems

## Representation Alignment
How do arbitrary nodes communicate reliably?

## Credit Assignment
How does the system know which node caused useful or harmful behavior?

## Routing
How does the controller select the right nodes?

## Node Creation
When should a new node be spawned?

## Node Size
How small can useful nodes become?

## Memory
How much information can a highly specialized node preserve?

## KV Replacement
Can a learned node actually replace persistent KV state?

## Latency
Can node loading and reconstruction be hidden effectively?

## Stability
Does the dynamic node population remain coherent?

## Node Proliferation
How many nodes can exist before routing becomes difficult?

## Verification
How does the system know a newly learned node is trustworthy?

## Security
Can malicious or corrupted nodes affect the system?

## Multimodal Alignment
Can vision, video, language, audio and reasoning nodes share enough common structure to cooperate?

---

# 87. Risks

Potential failure modes include:

### Routing collapse
The controller repeatedly chooses the same nodes.

### Node explosion
Too many tiny nodes make orchestration expensive.

### Fragmentation
Knowledge becomes too distributed to combine effectively.

### False specialization
Nodes become highly overfit but fail to remain useful.

### Memory hallucination
A reconstructed context may contain plausible but incorrect information.

### Reconstruction drift
Repeated training of a context node may gradually alter old information.

### Controller instability
The controller continually changes node assignments.

### Latency explosion
Storage transfers dominate compute.

### Compatibility failures
Two nodes produce representations that cannot be interpreted consistently.

---

# 88. Design Philosophy

EvMind should favor:

- modularity,
- empirical validation,
- hardware awareness,
- specialization,
- persistent state,
- continual evolution,
- composability,
- reversible changes,
- small units,
- clear interfaces.

It should avoid assuming:

- bigger is always better,
- every capability belongs in one model,
- every node must generalize,
- every memory must be token-based,
- every architecture must be a Transformer,
- every parameter must stay resident,
- new capabilities require global retraining.

---

# 89. Core Mental Models

## Framework Laptop

Stable core + replaceable capabilities.

## Tetris / Puzzle

Different requests assemble different node combinations.

## Neural Virtual Memory

Persistent nodes exist outside VRAM and are brought into working memory when needed.

## Brain-Like Reconstruction

Memory can be reconstructed from learned distributed state rather than replaying an enormous token sequence.

## Growing Organism

The system accumulates capabilities through experience and structural growth.

---

# 90. What EvMind Is Not

EvMind is not intended to be:

- a single giant Transformer,
- merely an MoE,
- merely a retrieval system,
- merely a database,
- merely an LLM wrapper,
- merely a context-window extension,
- merely a collection of plugins,
- merely a parameter compression scheme.

Those techniques may be used as components.

The architecture itself is the dynamic node ecosystem.

---

# 91. Central Hypothesis

The central hypothesis can be summarized as:

> **Intelligence may be more scalable, continually trainable, hardware-efficient, and extensible when represented as a dynamic population of small specialized neural/computational nodes rather than as one monolithic parameterized model.**

A stronger version:

> **A system whose architecture can grow, specialize, freeze, reactivate, and recombine may support continual learning with much less catastrophic interference than a system that must repeatedly modify one monolithic parameter space.**

And the strongest long-term hypothesis:

> **General intelligence may emerge from the ability to dynamically compose cognition, memory, curiosity, perception, skills, tools, and learned world knowledge into task-specific computational graphs, while continuously adding new capabilities without global retraining.**

---

# 92. Prototype Roadmap

## Phase 1 — Node Runtime

Implement:

- node registry,
- node loader,
- persistent storage,
- node metadata,
- load/unload,
- freeze/unfreeze,
- resource accounting.

## Phase 2 — Communication

Implement:

- node protocol,
- shared latent interface,
- typed messages,
- basic controller.

## Phase 3 — Specialization

Implement:

- multiple tiny nodes,
- independent training,
- node ranking,
- specialist creation.

## Phase 4 — Continual Learning

Implement:

- online learning,
- freezing,
- spawning,
- evaluation,
- retention testing.

## Phase 5 — Dynamic Scheduling

Implement:

- VRAM-aware loading,
- asynchronous prefetch,
- eviction,
- predictive scheduling.

## Phase 6 — Neural Memory

Implement:

- memory nodes,
- highly overfit context nodes,
- persistence,
- neural context reconstruction.

## Phase 7 — KV-State Research

Implement:

- KV capture,
- context-node training,
- state reconstruction,
- downstream equivalence testing.

## Phase 8 — Multimodal Nodes

Add:

- vision,
- video,
- audio,
- image generation,
- video generation.

## Phase 9 — Curiosity and Tools

Add:

- curiosity nodes,
- tool nodes,
- EvAgent integration,
- exploration loops.

## Phase 10 — Self-Evolution

Add:

- automated node creation,
- node splitting,
- node merging,
- controller learning,
- capability discovery.

## Phase 11 — General Learning Evaluation

Run:

- unfamiliar-domain experiments,
- persistent learning tests,
- open-ended environments,
- autonomous knowledge acquisition.

---

# 93. Example Full System

```text
                                 USER / ENVIRONMENT
                                         │
                              input / request / observation
                                         │
                                         ▼
                              ┌────────────────────┐
                              │   EVMIND CORE      │
                              │                    │
                              │ Intent              │
                              │ Planning            │
                              │ Node Selection      │
                              │ Resource Management │
                              │ Evolution           │
                              └─────────┬──────────┘
                                        │
                           dynamic node graph
                                        │
             ┌──────────────────────────┼───────────────────────────┐
             │                          │                           │
             ▼                          ▼                           ▼
       MEMORY NODES              PERCEPTION NODES             THINKING NODES
             │                          │                           │
             │                     image/video                    │
             │                          │                           │
             └──────────────┬───────────┴──────────────┬────────────┘
                            │                          │
                            ▼                          ▼
                      SKILL NODES                 CURIOSITY
                            │                          │
                            └──────────────┬───────────┘
                                           │
                                           ▼
                                      TOOL NODES
                                           │
                                    EvAgent / tools
                                           │
                                           ▼
                                      NEW DATA
                                           │
                                           ▼
                                  LEARNING / EXPERIENCE
                                           │
                              ┌────────────┴─────────────┐
                              ▼                          ▼
                         update node              spawn node
                              │                          │
                              └────────────┬─────────────┘
                                           ▼
                                     NODE POOL
                                           │
                              ┌────────────┼────────────┐
                              ▼            ▼            ▼
                             SSD          RAM          VRAM
                                                        │
                                                        ▼
                                                  ACTIVE COMPUTE
```

---

# 94. Practical Success Criteria

A successful early prototype does not need AGI.

It should demonstrate:

1. Tiny nodes can perform meaningful specialized tasks.
2. Nodes can cooperate.
3. Nodes can be loaded/unloaded cheaply enough to be useful.
4. Nodes can be frozen without losing their state.
5. New nodes can add capability without global retraining.
6. The controller can select useful node combinations.
7. Memory nodes can preserve useful context.
8. Node populations can remain stable as they grow.
9. The architecture can operate within constrained consumer VRAM.
10. Continual learning produces less destructive interference than comparable monolithic baselines.

---

# 95. Long-Term Vision

The long-term EvMind vision is not:

> Build the biggest model.

It is:

> **Build a computational substrate that can keep becoming more capable.**

The system should ideally evolve from:

```text
small node population
```

to:

```text
large persistent node ecosystem
```

while preserving a bounded active working set.

The system should be able to:

```text
remember
reason
question
explore
observe
use tools
learn
specialize
compose
create capabilities
freeze stable knowledge
reactivate old knowledge
and continuously evolve
```

without requiring a global restart or full-model retraining each time something new is learned.

---

# 96. Final Architectural Principle

The most concise expression of EvMind is:

> **Do not build one model that contains everything. Build a system that can assemble whatever it needs.**

And the second principle:

> **Do not overwrite knowledge when you can add capability.**

And the third:

> **Do not require all intelligence to be resident at once.**

And the fourth:

> **Treat computation, memory, cognition, tools, and specialized knowledge as composable nodes.**

And the fifth:

> **Let the architecture evolve as experience accumulates.**

---

# 97. Research Position

EvMind is currently a **hypothesis**, not a proven AGI architecture.

The claims that require empirical verification include:

- whether tiny diffusion nodes can provide sufficient useful cognition,
- whether deliberate overfitting produces effective persistent memory,
- whether neural nodes can replace large KV caches with acceptable functional fidelity,
- whether dynamic node routing can scale,
- whether asynchronous loading can keep latency manageable,
- whether continuous node spawning genuinely mitigates catastrophic forgetting,
- whether arbitrary node types can communicate reliably,
- and whether the resulting system can achieve broad generalization and autonomous learning.

The purpose of this architecture document is therefore not to declare that these questions are already solved.

It is to preserve a coherent hypothesis that can now be implemented, measured, falsified, and improved.

---

# 98. One-Sentence Definition

> **EvMind is a dynamically evolving, hardware-aware neural computing framework in which persistent specialized nodes—rather than a single monolithic parameterized model—are composed, learned, frozen, activated, and extended to create task-specific intelligence with continual learning, persistent neural memory, multimodal cognition, curiosity, tools, and structural growth.**




# 99. Recursive Reasoning Nodes and HRM-Inspired Architecture

Recursive models are a major candidate node family for EvMind, especially for **thinking, planning, verification, controller, curiosity, and meta-cognitive nodes**.

The central idea is to separate **parameter capacity** from **computation depth**.

A node does not necessarily become larger when a task becomes harder. Instead, the same learned transition function can be applied repeatedly:

```text
Input State
    ↓
Recursive Step 1
    ↓
Recursive Step 2
    ↓
Recursive Step 3
    ↓
...
    ↓
Recursive Step N
    ↓
Output State
```

This introduces another potential scaling axis for EvMind:

```text
Parameter capacity
        +
Computation depth
```

A compact node can therefore perform more work when a task requires it without requiring a proportionally larger permanent parameter footprint.

## 99.1 Why Recursion Fits EvMind

A 500K-parameter reasoning node could conceptually perform:

```text
Easy task:
500K × 5 recurrent steps

Hard task:
500K × 50 recurrent steps

Very hard task:
500K × 200 recurrent steps
```

The parameter set remains approximately the same while computation increases.

This is highly compatible with the EvMind principle of keeping nodes small while allowing the system to dynamically allocate computation.

## 99.2 HRM as an Inspiration

Hierarchical Reasoning Model (HRM) is an important architectural reference for this direction.

The relevant idea for EvMind is not that HRM is already a complete general-reasoning solution, but that **a compact recurrent architecture can repeatedly transform internal state and use different computational timescales** rather than relying only on a large feed-forward depth.

An HRM-derived or HRM-inspired node can therefore be used as:

- Thinking Node
- Planning Node
- Verification Node
- Controller Node
- Meta-Reasoning Node
- Curiosity Node
- Domain-specific recursive specialist

HRM is a candidate primitive to benchmark, not a fixed commitment.

## 99.3 EvMind-Adapted Recursive Node

A conceptual recursive node:

```text
                  NODE INPUT
                      │
                      ▼
             State Initialization
                      │
                      ▼
              ┌───────────────┐
              │ Fast Reasoner │
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │ Slow / Global │
              │ Reasoner      │
              └───────┬───────┘
                      │
                      ▼
                 State Update
                      │
                      ├───────→ continue recursion
                      │
                      ▼
                    Emit
```

The exact implementation may use:

- two recurrent timescales,
- one recurrent timescale,
- HRM-like hierarchy,
- adaptive computation time,
- attention,
- gated residual recurrence,
- state-space recurrence,
- diffusion refinement,
- or another learned mechanism.

## 99.4 Recursive Nodes Can Consume Other Nodes

A Thinking Node should not be isolated.

It may receive information from:

```text
Memory Node
Vision Node
Knowledge Node
Curiosity Node
Tool Node
Skill Node
```

Example:

```text
                    Thinking Node
                         ▲
           ┌─────────────┼─────────────┐
           │             │             │
       Memory          Vision       Knowledge
           │             │             │
           └─────────────┼─────────────┘
                         │
                  recursive reasoning
                         │
                         ▼
                       decision
```

The recursive state can be repeatedly updated as new evidence arrives.

This creates **iterative cognition** rather than one-pass node execution.

## 99.5 Dynamic Recursive Computation

The controller can determine how much recursion is needed.

Simple task:

```text
Request
  ↓
Thinking Node
  ↓
5 iterations
  ↓
confidence sufficient
  ↓
answer
```

Difficult task:

```text
Request
  ↓
Thinking Node
  ↓
20 iterations
  ↓
uncertain
  ↓
activate specialist
  ↓
additional reasoning
  ↓
Verification
  ↓
additional recursion
  ↓
answer
```

The system therefore treats inference depth itself as a resource.

## 99.6 Recursive Node + Tetris Composition

The Tetris/puzzle model becomes stronger when nodes are recursive.

The temporary cognition graph might be:

```text
Vision Node
     ↓
Thinking Node
 ↕   ↕   ↕
Memory / Physics / Knowledge
     ↓
Planning Node
     ↓
Verification Node
```

Each node can perform multiple internal iterations before passing a state onward.

The result is a **network of small recursive processes**, not simply a static sequence of experts.

## 99.7 Recursive Controller

The EvMind controller may itself become recursive:

```text
Input
 ↓
Select initial nodes
 ↓
Execute
 ↓
Inspect intermediate results
 ↓
Need more information?
 ├── yes → activate / load node
 │          ↓
 │       execute
 │          ↓
 │       inspect again
 │
 └── no → output
```

This allows the architecture to **reconfigure itself during cognition**.

## 99.8 Example: Video Physics

```text
User:
"Why did the object fall?"

        ↓

Controller
        ↓
Video Node
        ↓
visual / temporal state
        ↓
Thinking Node
        ↓
"I need physical information."
        ↓
Physics Node
        ↓
physics result
        ↓
Thinking Node recursively refines hypothesis
        ↓
"I need the frames immediately before the fall."
        ↓
Video Node provides targeted observation
        ↓
Thinking Node
        ↓
Verification Node
        ↓
final answer
```

This combines recursive reasoning with active perception and dynamic node loading.

## 99.9 Recursive Curiosity

Curiosity can itself operate recurrently:

```text
Current understanding
        ↓
identify uncertainty
        ↓
estimate information value
        ↓
select action
        ↓
observe result
        ↓
update state
        ↓
identify remaining uncertainty
        ↓
repeat
```

This creates a persistent information-seeking process instead of a single “curiosity score.”

## 99.10 Recursive Planning

A planning node can repeatedly refine a plan:

```text
Goal
 ↓
Plan v1
 ↓
simulate / evaluate
 ↓
identify failure
 ↓
Plan v2
 ↓
evaluate
 ↓
Plan v3
 ↓
confidence threshold
 ↓
execute
```

The same parameters are reused across planning iterations.

## 99.11 Recursive Verification

A verification node can challenge an answer repeatedly:

```text
Candidate Answer
      ↓
Verifier
      ↓
contradiction?
 ├── yes → send back to Thinking
 └── no  → increase confidence
      ↓
continue if needed
```

This supports iterative checking rather than relying on one forward pass.

## 99.12 Recursive Node + Diffusion Node

EvMind does not need to choose between recursion and diffusion.

A recursive thinking node can repeatedly invoke a diffusion specialist:

```text
Recursive Thinking
        ↓
hypothesis state
        ↓
Diffusion Specialist
        ↓
refined candidate
        ↓
Recursive Thinking
        ↓
new hypothesis
        ↓
Diffusion Specialist
        ↓
...
```

Likewise, a diffusion node could sit inside a recursive controller.

This allows each node family to perform the type of computation it is best suited for.

## 99.13 Recursive Node + Memory Node

A Thinking Node can interact with persistent learned memory over multiple reasoning iterations:

```text
Thinking iteration 1
      ↓
request Memory Node
      ↓
memory state
      ↓
Thinking iteration 2
      ↓
new request
      ↓
additional memory state
      ↓
Thinking iteration 3
      ↓
...
```

This is different from loading a complete life-history context into one monolithic attention window.

The cognition is an interaction between persistent neural state and recursive reasoning.

## 99.14 Recursive Node + Tool Node

Tools can become external cognitive actions:

```text
Thinking
 ↓
"I need to calculate X."
 ↓
Tool Node
 ↓
Python / calculator / API
 ↓
result
 ↓
Thinking
 ↓
"I need another experiment."
 ↓
Tool Node
 ↓
result
 ↓
Thinking
```

The loop continues until the controller decides the goal is sufficiently solved.

## 99.15 Recursive Node + Vision

A Thinking Node can request progressively more specific perception:

```text
Image
 ↓
Vision Node
 ↓
coarse representation
 ↓
Thinking
 ↓
"I need details around this region."
 ↓
Vision Node
 ↓
fine representation
 ↓
Thinking
```

This supports **active perception** and avoids forcing the entire raw modality into the reasoning node.

## 99.16 Adaptive Computation Depth

EvMind should eventually determine whether another recursive step is useful.

Possible stopping signals:

- confidence,
- state convergence,
- predicted improvement,
- verification success,
- contradiction count,
- uncertainty,
- latency budget,
- compute budget.

Conceptually:

```text
continue_score > threshold
        ↓
another recursive step

continue_score ≤ threshold
        ↓
emit
```

The same node can therefore spend radically different compute on different tasks.

## 99.17 Recursive Node Resource Model

Recursive computation changes the resource calculation.

A node can be tiny in persistent storage but expensive if it is run for many iterations.

The resource manager should therefore consider:

```text
persistent parameter cost
+
activation memory
+
recursion depth
+
expected latency
+
node-loading cost
+
external dependencies
```

Example:

```text
Node A:
1M params × 5 iterations

vs

Node B:
200K params × 50 iterations
```

The best option depends on the task and hardware budget.

## 99.18 Why Recursive Nodes Strengthen the EvMind Thesis

Recursive nodes reinforce several core goals:

### Small Nodes
A node can remain compact.

### Dynamic Compute
Difficulty can increase computation instead of parameter count.

### Modularity
Recursive nodes can be combined with arbitrary specialists.

### Hardware Efficiency
Persistent parameter footprint stays small.

### Continual Learning
A node's learned transition behavior can improve independently.

### Dynamic Cognition
The system can continue reasoning, request evidence, revise state, and continue.

## 99.19 HRM Should Be a Baseline, Not a Constraint

A useful research matrix is:

```text
A. Standard recurrent node
B. HRM-style hierarchical recurrent node
C. Recursive node + attention
D. Recursive node + diffusion refinement
E. Recursive node + external memory
F. Recursive node + node-to-node communication
G. Adaptive-computation recursive node
```

Compare them on:

- reasoning accuracy,
- out-of-distribution generalization,
- compute per solved task,
- latency,
- parameter count,
- VRAM,
- robustness,
- continual learning,
- transfer,
- cooperation with other node types.

EvMind should not assume that the HRM formulation is optimal; it should use HRM as a high-value experimental starting point.

## 99.20 Recursive Architecture Hypothesis

A specific EvMind hypothesis is:

> **A population of very small recursive neural nodes may provide greater effective cognitive capacity per resident parameter by reusing the same learned state-transition machinery over multiple inference steps and by composing with other specialized nodes.**

This is directly testable.

---

# 100. Expanded Node Taxonomy

With recursive models included, the node ecosystem becomes a set of computational primitives rather than a set of one-model experts:

```text
NODE FAMILIES
│
├── Generative / Reconstruction
│   ├── Diffusion
│   ├── Memory reconstruction
│   └── Image / Video generation
│
├── Recursive / Reasoning
│   ├── HRM-inspired
│   ├── Thinking
│   ├── Planning
│   ├── Verification
│   └── Controller
│
├── Perception
│   ├── Vision
│   ├── Video
│   ├── Audio
│   └── Sensors
│
├── Knowledge / Memory
│   ├── Context
│   ├── Episodic
│   ├── Semantic
│   └── Specialized learned distributions
│
├── Cognitive
│   ├── Curiosity
│   ├── Uncertainty
│   ├── Goal formation
│   └── Meta-reasoning
│
├── Skill
│   ├── Coding
│   ├── Mathematics
│   ├── Science
│   └── Domain skills
│
└── Tool / Action
    ├── Web
    ├── Python
    ├── Filesystem
    ├── APIs
    ├── EvAgent
    └── External environments
```

---

# 101. Recursive Nodes and the Node Pool

Recursive nodes follow the same residency rules as every other node.

A tiny recursive node can live on:

```text
SSD → RAM → VRAM → active compute
```

If inactive:

```text
Thinking Node
    ↓
freeze / unload
```

If a new problem requires it:

```text
Controller
    ↓
reactivate
    ↓
load
    ↓
recursive computation
```

This means recursion does not undermine the Framework Laptop model; it becomes another replaceable computational primitive.

---

# 102. Updated Core Architectural Principle

The concise EvMind principle is now:

> **Do not build one model that contains everything. Build a system that can assemble whatever it needs, and allow each computational piece to use the architecture most appropriate to its function.**

A recursive node may provide deep computation without becoming large.

A diffusion node may provide reconstruction or generation.

A memory node may provide persistent learned context.

A tool node may provide external action.

A perception node may provide sensory understanding.

A curiosity node may determine what the system needs to know next.

The architecture stays unified through the **node protocol and controller**, not by forcing every node to share one internal neural architecture.

---

# 103. Updated Experimental Priority

The addition of recursive nodes changes the early research sequence:

```text
1. Tiny standalone nodes
        ↓
2. Tiny recurrent / recursive node
        ↓
3. HRM-inspired node baseline
        ↓
4. Diffusion node
        ↓
5. Node-to-node communication
        ↓
6. Dynamic composition
        ↓
7. Adaptive recursion depth
        ↓
8. Memory node
        ↓
9. KV-state neuralization experiment
        ↓
10. Curiosity + tool loop
        ↓
11. Dynamic node creation
        ↓
12. Continual learning
        ↓
13. Unfamiliar-domain evaluation
```

The goal is to determine experimentally whether **small recursive nodes + specialized nodes + dynamic composition** provide a meaningful capability-per-resource advantage before building a large system.

---

# 104. Updated Research Position

EvMind now explicitly treats **recursion as an additional scaling axis** alongside:

```text
Node count
Node size
Node specialization
Node topology
Active node count
Recursion depth
Inference compute
Persistent memory
```

The conventional paradigm often emphasizes:

```text
parameters
+
context
+
compute
```

EvMind experiments with:

```text
small parameters
+
many specialized nodes
+
dynamic topology
+
variable recursion
+
persistent learned state
+
hardware-aware activation
```

The question is whether this combination can produce significantly greater **capability-per-active-resource** and **continual-learning capacity** than a monolithic baseline.

---

# 105. References / Inspiration

The HRM direction should be treated as external architectural inspiration rather than an established component whose assumptions are accepted unchanged.

Primary reference:

- Hierarchical Reasoning Model (HRM), *Hierarchical Reasoning Model*, arXiv:2506.21734 — https://arxiv.org/abs/2506.21734

Caution / follow-up analysis:

- arXiv:2601.10679 — https://arxiv.org/abs/2601.10679

These references are retained so future EvMind experiments can compare custom recursive-node designs against the original HRM formulation and later analyses.

---

# 106. Final EvMind Thesis

EvMind's core idea is increasingly clear:

> **A general intelligence does not necessarily need to be one giant model.**

It may instead be a **living computational ecosystem**:

```text
                    EVMIND
                       │
                 META CONTROLLER
                       │
      ┌────────────────┼─────────────────┐
      │                │                 │
      ▼                ▼                 ▼
  PERSISTENT        COGNITIVE        ENVIRONMENT
    NODES             NODES             NODES
      │                │                 │
 Memory          Thinking / HRM      Tools
 Knowledge       Curiosity           APIs
 Skills          Planning            EvAgent
 Perception      Verification        Sensors
 Generation      Meta-reasoning      External world
      │                │                 │
      └────────────────┼─────────────────┘
                       │
                DYNAMIC NODE GRAPH
                       │
                 ACTIVE WORKING SET
                       │
                    VRAM / RAM
                       │
                  CURRENT COMPUTE
```

The persistent ecosystem can grow without requiring the entire intelligence to become resident at once.

The computational graph can change from request to request.

Individual nodes can remain tiny.

Recursive nodes can trade parameter count for repeated computation.

Diffusion nodes can specialize in reconstruction and generation.

Memory nodes can preserve long-lived learned context.

Curiosity nodes can seek missing knowledge.

Tool nodes can act on the external world.

New nodes can be created when existing capabilities are insufficient.

Stable nodes can be frozen instead of repeatedly overwritten.

And the entire system can evolve structurally rather than relying on continual modification of one monolithic parameter tensor.

> **That is the EvMind research hypothesis: intelligence as a dynamically assembled, recursively computing, continually evolving ecosystem of small neural and computational nodes.**

---

# 107. Final One-Sentence Definition

> **EvMind is a dynamically evolving, hardware-aware neural computing framework in which persistent specialized nodes—including diffusion, recursive/HRM-inspired, perceptual, memory, cognitive, skill, generative, and tool nodes—are composed, learned, frozen, activated, and extended to create task-specific intelligence with continual learning, persistent neural memory, multimodal cognition, curiosity, adaptive computation, and structural growth.**


# 108. Generalized Node Definition: Capability, Not Model Type

A critical refinement to EvMind is that a **node is not synonymous with a small neural network**.

A node is the **architectural interface and capability container**. Its implementation can be a neural model, recursive model, diffusion model, deterministic algorithm, external tool, simulator, data-processing system, or an entirely new ML architecture.

> **A node is a pluggable computational capability, not necessarily a neural model.**

## 108.1 Capability → Node → Implementation

```text
Capability
    ↓
Node
    ↓
Implementation
```

Examples:

```text
Capability: exact arithmetic
Node: CalculatorNode
Implementation: deterministic calculator

Capability: deep iterative reasoning
Node: ThinkingNode
Implementation: HRM-inspired recursive model

Capability: image generation
Node: ImageGenerationNode
Implementation: diffusion model

Capability: persistent learned context
Node: ContextMemoryNode
Implementation: specialized neural memory model

Capability: real-time data preparation
Node: DataPipelineNode
Implementation: streaming / online ML model
```

The EvMind node protocol remains stable while the internal implementation can differ radically.

# 109. Heterogeneous Computational Substrate

EvMind should not require every node to use one model family. Possible implementations include:

```text
Transformer
Diffusion model
Recursive / HRM model
CNN
RNN
State-space model
Graph neural network
Reinforcement-learning policy
Classifier
Clustering model
Anomaly detector
Streaming model
Compression model
Simulator
Physics engine
Calculator
Search engine
Database interface
Browser controller
Computer-use system
External API
Deterministic algorithm
Hybrid ML system
Future architecture
```

The higher-level system cares about the node's interface, inputs, outputs, capabilities, resource requirements, and reliability—not necessarily its internal architecture.

# 110. Why a Calculator Is a Node

EvMind should not learn every capability unnecessarily. For exact arithmetic, a Calculator Node may be superior to a neural model because it is exact, fast, tiny, deterministic, and requires no training.

This establishes a core principle:

> **Use the most appropriate computational substrate for the task instead of forcing every capability into learned parameters.**

# 111. Capability Routing and Substrate Routing

The controller has two related questions:

1. What capability is required?
2. What implementation should provide it?

Thus EvMind performs both **capability routing** and **computational-substrate routing**.

# 112. New ML Architectures as Nodes

When a capability cannot be served effectively by the current node ecosystem, EvMind may eventually create a new computational implementation.

```text
Capability gap
      ↓
Model Architect Node
      ↓
architecture proposal
      ↓
training system
      ↓
candidate model
      ↓
evaluation
      ↓
register successful model as node
```

This creates two levels of growth:

```text
Level 1: add a new node using an existing architecture
Level 2: add a new node using a new architecture
```

# 113. Model Architect Node

A future Model Architect Node could decide what model family, training objective, representation, scale, and architecture best fits a newly discovered capability gap, then train and evaluate candidate implementations.

# 114. Real-Time Data-Processing Nodes

Continual learning creates a data-management bottleneck. Incoming conversations, images, video, web observations, tool outputs, sensor streams, feedback, failures, and successes cannot necessarily be consumed directly.

EvMind can create specialized data-processing nodes for real-time cleaning, filtering, deduplication, normalization, classification, clustering, prioritization, drift detection, and quality ranking.

```text
Live data stream
      ↓
Data Processing Node
      ↓
clean / filter / deduplicate / cluster / prioritize
      ↓
Training / Memory Pipeline
```

The implementation can be a streaming ML model, online clustering system, anomaly detector, recurrent model, deterministic pipeline, or hybrid.

# 115. Continuous Data Learning Loop

```text
Environment
    ↓
Raw observations
    ↓
Data Processing Node
    ↓
Cleaning / sorting / prioritization
    ↓
Research / Memory / Training
    ↓
Candidate knowledge
    ↓
Node creation or update
    ↓
Evaluation
    ↓
Persistent Node Pool
```

# 116. Open-Ended Capability Acquisition

The strongest interpretation of EvMind's "limitless potential" is not literally infinite compute. It is that the architecture does not prescribe a fixed, finite vocabulary of future computational capabilities.

When a new requirement appears, the system may potentially recognize the gap, search existing nodes, invoke deterministic tools, use external capabilities, acquire information, design or adapt a new implementation, train it, evaluate it, register it, and reuse it.

# 117. Unfamiliar-Domain Example: Gene Engineering

Suppose EvMind initially has strong software-engineering capability but insufficient specialization for a gene-engineering research task. It can recognize the capability gap, invoke curiosity, plan research, use browser/computer-use nodes, perceive pages through vision, retain useful information in memory nodes, and continue researching across many pages.

```text
User request
      ↓
Controller
      ↓
Capability-gap detection
      ↓
Curiosity
      ↓
Research planning
      ↓
Browser / Computer-Use
      ↓
Vision
      ↓
Evidence
      ↓
Memory / Knowledge
      ↓
Further research
```

# 118. Browser + Vision + Memory Research Loop

```text
Page 1 → perceive → retain → Page 2 → perceive → integrate → Page 3 → ...
```

Only the current working set needs to remain resident in VRAM while the accumulated research state can persist in neural memory nodes.

# 119. Computer Use as a General Capability

A generic computer-use capability can operate through:

```text
Screen
 ↓
Vision
 ↓
UI understanding
 ↓
Recursive reasoning
 ↓
Action selection
 ↓
Mouse / touch / keyboard
 ↓
New screen state
 ↓
Vision
 ↓
Reasoning
 ↓
Next action
```

This potentially generalizes across websites, desktop software, mobile applications, research tools, and other graphical interfaces, subject to authentication, permissions, safety controls, and action verification.

# 120. Research → Learning → New Specialist

After sufficient research, the system may decide a persistent specialist is worthwhile:

```text
Temporary Research Knowledge
          ↓
Repeated future demand
          ↓
Stable domain distribution
          ↓
Training-data construction
          ↓
Specialized ML model
          ↓
Evaluation
          ↓
GeneEngineeringNode v1
          ↓
Register into Node Pool
```

# 121. Hierarchical Specialization

A broad specialist can later produce narrower specialists:

```text
Gene Engineering
      ├── Gene Editing
      ├── CRISPR
      ├── Genomics
      ├── Protein Design
      ├── Experimental Methods
      └── Literature Analysis
```

The controller can choose between broad and narrow specialists according to task performance and resource cost.

# 122. Temporary Knowledge vs Persistent Capability

Not every observation should become a model. A possible progression is:

```text
Observation
  ↓
Memory Node
  ↓
Repeated relevance
  ↓
Stable knowledge
  ↓
Repeated task demand
  ↓
Specialized model worth creating?
  ↓
Yes → train and register node
```

# 123. Capability Creation Decision Tree

```text
New requirement
      ↓
Existing node sufficient?
   ┌──┴──┐
  yes    no
   │      │
 reuse   external tool sufficient?
            ┌──┴──┐
           yes    no
            │      │
          use    acquire information
                       ↓
                 existing model family?
                    ┌──┴──┐
                   yes    no
                    │      │
                 adapt   Model Architect
                           ↓
                     new implementation
```

# 124. Model Ecosystem vs Fixed Model

A model zoo is a fixed collection of models. EvMind instead maintains a living node population with capabilities, relationships, routing, resource state, continual training, and structural evolution.

# 125. Canonical Definition of a Node

> **A node is a standardized, persistent, addressable computational capability whose implementation may be neural, recursive, generative, algorithmic, tool-based, simulator-based, or otherwise executable, and which can be loaded, unloaded, frozen, trained, composed, evaluated, and evolved by the EvMind runtime.**

# 126. Framework Laptop Analogy — Expanded

```text
Framework Laptop ↔ EvMind

Common socket / interface ↔ Node protocol
RAM module ↔ Memory node
CPU specialist ↔ Recursive reasoning node
GPU accelerator ↔ Diffusion / vision / generation node
Peripheral ↔ Tool node
Storage ↔ Persistent node pool
Future expansion card ↔ New ML architecture
```

The stable platform remains the same while components can be added or replaced.

# 127. EvMind as a General AI Runtime

Possible core services:

```text
Node Registry
Node Store
Node Loader
VRAM Manager
RAM Cache
Node Router
Capability Registry
Workflow Planner
Curiosity Engine
Training Manager
Model Architect
Tool Manager
Memory Manager
Evaluation Engine
Evolution Manager
```

# 128. Open-Ended Capability Space

The key claim is:

> **EvMind does not require the designers to enumerate every future capability at initialization.**

A new capability can potentially become a new node, a new implementation, or even a new model family.

# 129. Capability Acquisition as an AGI-Relevant Property

A strong AGI-oriented property would be:

```text
Unknown problem
      ↓
recognize capability gap
      ↓
select tools / nodes
      ↓
gather information
      ↓
learn representation
      ↓
create appropriate implementation
      ↓
validate
      ↓
incorporate into node ecosystem
      ↓
reuse on future tasks
```

This is stronger than simply retrieving information: it is **capability acquisition**.

# 130. Updated Long-Term Vision

EvMind is a self-extensible computational ecosystem in which intelligence can acquire not only information and skills, but potentially new computational implementations.

A mature system may contain thinking, memory, curiosity, vision, video, audio, planning, verification, diffusion, recursive reasoning, calculators, search, browser control, data processing, simulation, scientific specialists, domain specialists, external tools, and generated ML architectures, while only a small working set is active at any moment.

# 131. Updated Central Hypothesis

> **General intelligence may be better approached as an extensible computational ecosystem than as a single fixed neural parameter space: the system can choose existing capabilities, invoke deterministic tools, acquire information, create new specialized models, and dynamically compose heterogeneous computational nodes into task-specific cognitive systems.**

The concise form is:

> **The unit of growth does not have to be a parameter. It can be a computational capability.**

The more ambitious long-term hypothesis is:

> **The system may eventually learn not only what it knows, but what kind of computational model it needs in order to learn what it does not yet know.**

# 132. HRM-Inside-the-Node Learning Governor

A new near-term EvMind prototype direction is to place a small **HRM-style recursive learning governor inside a node** and use it to control the node's live plasticity.

The purpose is not to make the HRM governor the main knowledge model. Its purpose is to decide how the knowledge-bearing parameters of the host node should change when new information arrives.

```text
                         NODE
┌──────────────────────────────────────────────┐
│  FROZEN HRM-STYLE LEARNING GOVERNOR          │
│  novelty • conflict • relevance • stability  │
│                     │                        │
│              update gates / masks             │
│                     ↓                        │
│  knowledge-bearing parameters W1...Wn         │
└──────────────────────────────────────────────┘
```

## 132.1 Learned Update Mask

A conceptual formulation is:

\[
M = f_{\theta_{HRM}}(x,s)
\]

and:

\[
W' = W - \eta \, M \odot \nabla_W L
\]

where `M ≈ 0` freezes a region, `M ≈ 1` allows normal learning, and intermediate values permit partial adaptation.

The goal is to replace blind global updating with **learned plasticity allocation**.

## 132.2 Hierarchical Update Granularity

Early prototypes should not gate every individual parameter. A practical progression is:

```text
Node → Module → Block → Layer → Head/Channel → Parameter group → Parameter
```

Module- or block-level control is likely the best starting point.

## 132.3 Frozen Governance + Mutable Knowledge

A node can be conceptually divided into:

```text
1. Frozen Governance Core
   - update policy
   - novelty detection
   - conflict detection
   - relevance
   - stability

2. Stable Knowledge
   - learned specialization
   - protected when appropriate

3. Live Adaptation
   - currently trainable capacity
```

The governance mechanism is therefore stable while the knowledge substrate remains selectively plastic.

## 132.4 Recursive Learning Governance

The HRM-style governor can evaluate its own proposed update strategy:

```text
new information
      ↓
HRM governor
      ↓
proposed update mask
      ↓
temporary adaptation
      ↓
evaluate old/new knowledge
      ↓
revise mask
      ↓
re-evaluate
      ↓
commit OR reject
```

This is more expressive than a static importance mask and gives the learning process an explicit internal feedback loop.

## 132.5 Old-vs-New Knowledge Arbitration

The governor should distinguish:

- reinforcement of existing knowledge,
- compatible specialization or conditional exceptions,
- genuine contradictions,
- information that exceeds the node's representational capacity.

For example:

```text
Old: X is true
New: X is false under condition Y

→ preserve X generally
→ learn the condition-specific exception
```

If safe coexistence is impossible, the node can escalate to an adaptation branch or a new node rather than blindly overwriting old knowledge.

## 132.6 Small Trainable Extension

The HRM-governed node may also use a small trainable parameter extension:

```text
100K frozen base
      +
10K live adaptation
```

The governor controls how the live portion learns. Once stable, an adaptation branch can be frozen and promoted into persistent knowledge.

## 132.7 Adaptation vs Structural Growth

A learning hierarchy can be:

```text
Can current parameters safely adapt?
        ↓ yes
Selective update

If no:
Can a small adaptation branch solve it?
        ↓ yes
Train branch

If no:
Can another existing node solve it?
        ↓ yes
Activate it

If no:
Spawn a new node
```

This gives EvMind two complementary growth mechanisms: **plasticity inside a capability** and **structural growth through new nodes**.

## 132.8 First Falsifiable Experiment

The first prototype should remain deliberately small:

```text
Base node: 50K–500K parameters
HRM-style governor: small recursive model
Learning stream: sequential tasks
Baseline: ordinary continual training
```

The first question is simply:

> **Does the HRM-governed node retain more previously learned capability while learning new information under comparable compute?**

A modest but reproducible improvement is already meaningful evidence for the mechanism.

## 132.9 Controlled Continual-Learning Stream

```text
Phase A → learn A → test A
Phase B → learn B → test A+B
Phase C → learn C → test A+B+C
Phase D → introduce contradiction → retest
Phase E → revisit A → measure retention
```

Record:

- old-task retention,
- new-task performance,
- update fraction,
- training time,
- latency,
- memory use,
- controller decisions.

## 132.10 Update Fraction

A key diagnostic is:

\[
UpdateFraction = \frac{\text{parameters or groups substantially changed}}{\text{total parameters}}
\]

A promising outcome would be substantially fewer changed regions with equal or better new-task performance and improved old-task retention.

The actual values must be measured rather than assumed.

## 132.11 Baselines

At minimum compare:

```text
A. Standard continual training
B. Frozen base + small trainable adapter
C. HRM-governed selective updating
```

Later variants can test:

```text
D. HRM governor + adapter
E. HRM governor + node spawning
F. HRM governor + memory nodes
G. HRM governor + full dynamic routing
```

## 132.12 Research Interpretation

The first prototype should not be judged by whether it solves AGI.

The initial falsifiable claim is narrower:

> **A recursive learning governor can measurably improve continual learning by controlling where and how a node updates.**

If that works repeatedly, it justifies adding adaptation branches, node spawning, persistent memory, dynamic routing, and the broader EvMind ecosystem.

## 132.13 Updated Live-Learning Principle

> **New information should not automatically overwrite established knowledge. The node should contain a stable mechanism capable of deciding where plasticity is appropriate, while preserving adaptation, coexistence, branching, or structural growth as separate options.**

A stronger statement is:

> **The system should learn not only new knowledge, but a policy for how new knowledge is allowed to modify existing knowledge.**
