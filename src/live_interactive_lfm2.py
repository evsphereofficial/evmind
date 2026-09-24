"""EvAGI Interactive — LFM2.5-1.2B-Instruct live learning CLI.

Full stack on the 1.2B base model:
  - Equation V4 lean prior + adaptive doubling (registry.predict_neurons_v4)
  - NeuronRegister with protection capture (governor sees old-task territory)
  - TinyPerExpertGovernor with 11 features (8 local + 3 registry)
  - Hard expert isolation via SwiGLU w1/w3 hooks
  - Laya-pattern InputRouter: structural detect -> intent + confidence
    -> expert mask or "I don't know — teach me"
  - Dynamic skill/fact/knowledge/unknown detection (not a closed fact set)
  - Cross-session persistence via weight deltas + registry state

Commands:
  teach / learn prefixed, or natural teaching statements
  queries route to learned experts; unknown -> ask to teach
  facts          list learned items
  experts        registry summary
  save / quit
"""

from __future__ import annotations

import json
import math
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm_evagi import (
    NeuronRegister,
    TinyPerExpertGovernor,
    expand_router,
    HRMRouter,
)
from src.registry import predict_neurons_v4, next_grid_step

MODEL_DIR = Path("models/LFM2.5-1.2B-Instruct")
CHECKPOINT = Path("models/LFM2.5-1.2B-Instruct/evagi_live.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

MAX_LEN = 256
TRAIN_EPOCHS = 30
TRAIN_LR = 5e-3
BATCH = 4
PROBE_HIT_RATE = 1.0
MAX_ADAPTIVE_RETRIES = 4
UNKNOWN_CONF_THRESHOLD = 0.35
SUPPORT_TOP1_MARGIN = 0.08


# ---------------------------------------------------------------------------
# Model load + SwiGLU expert masking
# ---------------------------------------------------------------------------
def load_model_and_tok():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=DTYPE, device_map=DEVICE, low_cpu_mem_usage=True
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for layer in model.model.layers:
        for p in layer.feed_forward.parameters():
            p.requires_grad_(True)
    n_layers = model.config.num_hidden_layers
    inter = model.model.layers[0].feed_forward.w1.out_features
    hidden = model.config.hidden_size
    return model, tok, n_layers, inter, hidden


def encode_answer(tok, answer: str) -> torch.Tensor:
    return tok(answer, add_special_tokens=False, return_tensors="pt")["input_ids"][0]


def encode_prompt(tok, messages: list[dict]) -> torch.Tensor:
    s = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tok(s, add_special_tokens=False, return_tensors="pt")["input_ids"][0]


def tokenize_qa(tok, pairs, max_length=MAX_LEN):
    ids_l, mask_l, lab_l = [], [], []
    for messages, answer in pairs:
        p = encode_prompt(tok, messages)
        a = encode_answer(tok, answer)
        ids = torch.cat([p, a])[:max_length]
        lab = ids.clone()
        lab[: len(p)] = -100
        ids_l.append(ids)
        mask_l.append(torch.ones_like(ids))
        lab_l.append(lab)
    maxl = max(len(x) for x in ids_l)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    return {
        "input_ids": torch.stack([F.pad(x, (0, maxl - len(x)), value=pad) for x in ids_l]),
        "attention_mask": torch.stack([F.pad(x, (0, maxl - len(x)), value=0) for x in mask_l]),
        "labels": torch.stack([F.pad(x, (0, maxl - len(x)), value=-100) for x in lab_l]),
    }


class HardSwiGLUMask:
    """Zero non-owned SwiGLU intermediate channels via w1/w3 forward hooks."""

    def __init__(self, model, masks: list[torch.Tensor]):
        self.model = model
        self.masks = masks
        self.handles = []

    def __enter__(self):
        for layer, mask in zip(self.model.model.layers, self.masks):
            def h1(mod, inp, out, m=mask):
                return out * m.to(out.dtype)
            def h3(mod, inp, out, m=mask):
                return out * m.to(out.dtype)
            self.handles.append(layer.feed_forward.w1.register_forward_hook(h1))
            self.handles.append(layer.feed_forward.w3.register_forward_hook(h3))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()
        self.handles = []


def hard_mask_grads_lfm2(model, masks):
    for layer, mask in zip(model.model.layers, masks):
        ff = layer.feed_forward
        if ff.w1.weight.grad is not None:
            ff.w1.weight.grad[~mask, :] = 0
        if ff.w3.weight.grad is not None:
            ff.w3.weight.grad[~mask, :] = 0
        if ff.w2.weight.grad is not None:
            ff.w2.weight.grad[:, ~mask] = 0


def allocate_free(register: NeuronRegister, n_layers, inter, n_neurons, expert_id, device):
    """Carve n_neurons from FREE pool only (no overlap across experts)."""
    masks = []
    used = 0
    # distribute target across layers, then take free indices per layer
    if n_neurons <= n_layers:
        per_layer = [1 if i < n_neurons else 0 for i in range(n_layers)]
    else:
        base, rem = divmod(n_neurons, n_layers)
        per_layer = [base + (1 if i < rem else 0) for i in range(n_layers)]
    for li, take in enumerate(per_layer):
        take = min(take, inter)
        free = torch.where(~register.occupied[li])[0]
        chosen = free[:take]
        m = torch.zeros(inter, dtype=torch.bool, device=device)
        m[chosen] = True
        register.occupied[li][chosen] = True
        register.ownership[li][chosen] = expert_id
        masks.append(m)
        used += int(chosen.numel())
    register.allocations.append(
        {
            "task": f"eid{expert_id}",
            "expert_id": expert_id,
            "counts": [int(m.sum()) for m in masks],
            "owned": used,
            "n": n_neurons,
        }
    )
    return masks, used


def counts_from_masks(masks):
    return [int(m.sum()) for m in masks]


def snapshot_ffn(model):
    snap = {}
    for i, layer in enumerate(model.model.layers):
        ff = layer.feed_forward
        snap[f"{i}.w1"] = ff.w1.weight.detach().clone()
        snap[f"{i}.w3"] = ff.w3.weight.detach().clone()
        snap[f"{i}.w2"] = ff.w2.weight.detach().clone()
    return snap


def restore_ffn(model, snap):
    for i, layer in enumerate(model.model.layers):
        ff = layer.feed_forward
        ff.w1.weight.data.copy_(snap[f"{i}.w1"])
        ff.w3.weight.data.copy_(snap[f"{i}.w3"])
        ff.w2.weight.data.copy_(snap[f"{i}.w2"])


# ---------------------------------------------------------------------------
# Governor features for LFM2 (w1.weight is the c_fc analogue: inter x hidden)
# ---------------------------------------------------------------------------
def lfm2_neuron_features(model, mask, layer_idx, n_layers, register: NeuronRegister):
    """(N, 11) = local8 on w1 rows + registry3."""
    w = model.model.layers[layer_idx].feed_forward.w1.weight.detach()  # (inter, hidden)
    idx = torch.where(mask)[0]
    if idx.numel() == 0:
        return torch.zeros(0, 11, device=mask.device), idx
    w_rows = w[idx]
    n = mask.numel()
    pos = idx.float() / max(n - 1, 1)
    layer_f = torch.full_like(pos, layer_idx / max(n_layers - 1, 1))
    owned = torch.ones_like(pos)
    local = torch.stack(
        [
            torch.log1p(w_rows.abs().mean(-1)),
            torch.zeros_like(pos),  # grad feature filled at train time
            pos,
            layer_f,
            owned,
            torch.tanh(w_rows.mean(-1)),
            torch.zeros_like(pos),
            w_rows.std(-1),
        ],
        dim=-1,
    )
    regf = register.get_protection_feats(mask, layer_idx)
    feats = torch.cat([local, regf.to(local.dtype)], dim=-1)
    return feats, idx


# ---------------------------------------------------------------------------
# QA / probe builders (open content — fact, skill, knowledge)
# ---------------------------------------------------------------------------
def get_answer(kind: str, value: str) -> str:
    if kind.startswith("fact_"):
        return f" {value.strip()}"
    if kind == "skill_code":
        v = InputRouter.normalize_skill_value(value)
        return (
            f" Here is how to {v} in Python:\n"
            f"```python\n# {v}\n# (learned example)\n```"
        )
    if kind == "knowledge":
        return f" {value.strip()}"
    return f" {value.strip()}"


def make_qa_pairs(kind: str, value: str) -> list[tuple[list[dict], str]]:
    # normalize skill value for prompts (avoid "How do I how to X?")
    prompt_value = InputRouter.normalize_skill_value(value) if kind == "skill_code" else value.strip()
    answer = get_answer(kind, value)
    v = prompt_value
    if kind.startswith("fact_"):
        noun = kind.replace("_", " ")
        prompts = [
            f"What is my {noun}?",
            f"Remember, my {noun} is {v}. What is my {noun}?",
            f"Tell me the {noun}.",
            f"What do you remember about my {noun}?",
            f"Can you recall my {noun}?",
            f"What was my {noun} again?",
            f"Repeat what I told you about my {noun}.",
            f"My {noun} is {v}. Got it. What is my {noun}?",
        ]
    elif kind == "skill_code":
        prompts = [
            f"How do I {v}?",
            f"Write code to {v}.",
            f"Show me how to {v} in Python.",
            f"I need to {v}. Can you help?",
            f"Can you teach me to {v}?",
            f"How would you {v}?",
            f"Give me an example to {v}.",
            f"What's the code for {v}?",
        ]
    elif kind == "knowledge":
        prompts = [
            f"What is {v}?",
            f"Tell me about {v}.",
            f"Explain {v}.",
            f"I want to learn about {v}.",
            f"What can you tell me about {v}?",
            f"Describe {v}.",
            f"Summarize {v}.",
            f"Do you know {v}?",
        ]
    else:
        prompts = [f"{v}", f"Remember: {v}", f"About {v}: ...", f"Recall {v}"] * 2
    # For fact multi-turn templates: if prompt contains two questions, split carefully.
    # Keep simple: each prompt is one user turn -> assistant answer.
    # Special-case the 'remember ... what is' pattern as multi-turn:
    out = []
    for p in prompts:
        if ". What is my" in p or ". Got it. What is my" in p:
            # split into user/assistant/user
            head, tail = p.split(". ", 1)
            msgs = [
                {"role": "user", "content": head},
                {"role": "assistant", "content": " Okay, I will remember that."},
                {"role": "user", "content": tail},
            ]
        else:
            msgs = [{"role": "user", "content": p}]
        out.append((msgs, answer))
    return out


def make_probes(kind: str, value: str) -> list[tuple[list[dict], str]]:
    pv = InputRouter.normalize_skill_value(value) if kind == "skill_code" else value.strip()
    answer = get_answer(kind, value)
    v = pv
    if kind.startswith("fact_"):
        noun = kind.replace("_", " ")
        prompts = [f"Quick — my {noun}?", f"And my {noun} was...?", f"What's my {noun}?"]
    elif kind == "skill_code":
        prompts = [f"Remind me, how do I {v}?", f"The steps to {v} again?", f"Code for {v}?"]
    elif kind == "knowledge":
        prompts = [f"In simple terms, what is {v}?", f"Remind me about {v}.", f"Quick summary of {v}?"]
    else:
        prompts = [f"Recall {v}?", f"What was {v}?", f"{v} again?"]
    return [([{"role": "user", "content": p}], answer) for p in prompts]


def expected_match(text: str, answer: str, kind: str, value: str) -> bool:
    def norm(s):
        return " ".join(s.lower().split())
    t = norm(text)
    if kind.startswith("fact_"):
        return norm(value) in t
    sig = " ".join(norm(answer).split()[:8])
    return (sig and sig in t) or (norm(value) in t)


# ---------------------------------------------------------------------------
# Laya-pattern InputRouter: structural detect + confidence + expert pick
# ---------------------------------------------------------------------------
@dataclass
class RouteResult:
    intent: str          # learn | query | chat | unknown
    kind: str | None     # fact_* | skill_code | knowledge | None
    confidence: float
    expert_id: int | None
    value: str | None
    reason: str


class InputRouter:
    """Laya-inspired: decide intent BEFORE the heavy forward, gate on confidence.

    Dynamic (not a closed fact list):
      - structural signals classify teach-vs-ask-vs-chat
      - support embeddings of learned experts pick WHO answers a query
      - below threshold -> unknown -> 'teach me' flow
    """

    TEACH_PATTERNS = [
        (re.compile(
            r"\b(?:my\s+)?([a-z][a-z0-9_ ]{0,40}?)\s+is\s+([A-Za-z0-9][\w .'-]{0,60})",
            re.I,
        ), "fact"),
        (re.compile(r"\bremember(?: that)?[:,]?\s*(.+)", re.I), "fact"),
        (re.compile(r"\bnote(?: that)?[:,]?\s*(.+)", re.I), "fact"),
        (re.compile(r"\bsave(?: this)?[:,]?\s*(.+)", re.I), "fact"),
        (re.compile(r"\bcall me\s+(.+)", re.I), "fact"),
        (re.compile(r"\bi(?:'m| am)\s+called\s+(.+)", re.I), "fact"),
        (re.compile(r"\bi live in\s+(.+)", re.I), "fact"),
        (re.compile(r"\bmy favou?rite colou?r is\s+(.+)", re.I), "fact"),
        (re.compile(r"\bmy favou?rite food is\s+(.+)", re.I), "fact"),
        (re.compile(r"\bhow (?:do i|to|would you|can i)\s+(.+?)(?:\?|$)", re.I), "skill"),
        (re.compile(r"\bwrite(?: a)? (?:function|code|script|program)\s*(?:to|for)?\s*(.+?)(?:\?|$)", re.I), "skill"),
        (re.compile(r"\bteach me(?: to| how)?\s+(.+?)(?:\?|$)", re.I), "skill"),
        (re.compile(r"\blearn(?: how)? (?:to|to do)?\s+(.+?)(?:\?|$)", re.I), "skill"),
        (re.compile(r"\bwhat is\s+(.+?)(?:\?|$)", re.I), "knowledge"),
        (re.compile(r"\bexplain\s+(.+?)(?:\?|$)", re.I), "knowledge"),
        (re.compile(r"\btell me about\s+(.+?)(?:\?|$)", re.I), "knowledge"),
        (re.compile(r"\bwhat do you know about\s+(.+?)(?:\?|$)", re.I), "knowledge"),
        (re.compile(r"\blearn(?: that)?[:,]?\s*(.+)", re.I), "knowledge"),
    ]
    QUERY_WORDS = re.compile(
        r"\b(what|who|when|where|which|why|how much|how many|do you (?:know|remember)|"
        r"can you (?:recall|remember)|remind me|quick —|what's|whats)\b",
        re.I,
    )
    FACT_FIELD_HINTS = re.compile(
        r"\b(name|color|colour|city|country|food|favorite|favourite|live|from|age|birthday)\b",
        re.I,
    )

    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size
        self.experts: list[dict] = []  # {kind, value, expert_id, support_centroid}
        self.kind_counts: dict[str, int] = {}

    def register_expert(self, kind, value, expert_id, centroid: torch.Tensor):
        self.experts.append(
            {
                "kind": kind,
                "value": value,
                "expert_id": expert_id,
                "support_centroid": centroid.float().cpu(),
            }
        )
        self.kind_counts[kind] = self.kind_counts.get(kind, 0) + 1

    def text_centroid(self, tok, text: str) -> torch.Tensor:
        """Mean token embedding — cheap support embedding (Laya: no extra model)."""
        ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        if ids.numel() == 0:
            ids = torch.tensor([tok.pad_token_id or 0])
        # embedding table lives on model device; caller moves as needed
        return ids

    @staticmethod
    def normalize_skill_value(value: str) -> str:
        """Strip redundant 'how to / to' prefixes from skill values."""
        v = value.strip()
        for pref in ("how to ", "how to do ", "to ", "to do "):
            if v.lower().startswith(pref):
                v = v[len(pref):].strip()
                break
        return v

    def cosine_to_experts(self, model, tok, text: str) -> tuple[int | None, float, float]:
        ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(DEVICE)
        with torch.no_grad():
            emb = model.model.embed_tokens(ids).mean(dim=0).float()
            emb = F.normalize(emb, dim=0)
        if not self.experts:
            return None, 0.0, 0.0
        sims = []
        for e in self.experts:
            c = e["support_centroid"].to(DEVICE).float()
            c = F.normalize(c, dim=0)
            sims.append(float(torch.dot(emb, c)))
        sims_t = torch.tensor(sims)
        top = int(sims_t.argmax())
        top1 = float(sims_t[top])
        top2 = float(sims_t.topk(2).values[-1]) if len(sims) > 1 else 0.0
        return self.experts[top]["expert_id"], top1, top1 - top2

    def route(self, tok, model, text: str, learned_kinds: set[str]) -> RouteResult:
        t = text.strip()
        low = t.lower()

        # 1) explicit teach prefixes
        for pref in ("teach ", "learn ", "remember ", "note "):
            if low.startswith(pref):
                rest = t[len(pref):].lstrip(":, ")
                kind_hint, value = self._classify_teach(rest)
                return RouteResult("learn", kind_hint, 1.0, None, value, f"prefix:{pref.strip()}")

        # 2) QUERY path before teach patterns — "What is my name?" must not
        #    be parsed as teach ("What is" + "my name").
        is_query = bool(self.QUERY_WORDS.search(t)) or t.endswith("?")
        if is_query:
            eid, top1, margin = self.cosine_to_experts(model, tok, t)
            conf = top1
            matched_kind = None
            # fact-field shortcut: "my name" / "favorite color" -> known fact kind
            if self.FACT_FIELD_HINTS.search(t):
                for kind in learned_kinds:
                    if not kind.startswith("fact_"):
                        continue
                    field = kind.split("_", 1)[1]
                    field_l = field.lower()
                    if field_l in low or (field_l == "color" and ("colou" in low or "color" in low)) or (
                        field_l == "name" and re.search(r"\bname\b", low)
                    ) or (field_l == "city" and re.search(r"\b(live|from|city)\b", low)) or (
                        field_l == "food" and re.search(r"\b(food|eat|like)\b", low)
                    ):
                        matched_kind = kind
                        for e in self.experts:
                            if e["kind"] == kind:
                                eid = e["expert_id"]
                                conf = max(conf, 0.85)
                                break
                        break

            if matched_kind is not None and eid is not None:
                return RouteResult(
                    "query", matched_kind, conf, eid, None, "fact_field_match"
                )

            # Embedding anisotropy: unrelated English still scores ~0.5-0.6.
            # Require BOTH strong top1 AND a clear margin over runner-up,
            # otherwise we don't know it.
            strong = top1 >= 0.75 and margin >= 0.10
            # also check skill/knowledge value similarity against expert values
            if not strong and self.experts:
                # exact-ish value mention in query
                for e in self.experts:
                    if e["value"] and e["value"].lower() in low:
                        eid = e["expert_id"]
                        conf = 0.9
                        matched_kind = e["kind"]
                        strong = True
                        break

            if strong and eid is not None:
                kind = matched_kind or next(
                    (e["kind"] for e in self.experts if e["expert_id"] == eid), None
                )
                return RouteResult(
                    "query", kind, conf, eid, None,
                    f"support top1={top1:.2f} margin={margin:.2f}",
                )

            casual = re.match(
                r"^\s*(how are you|who are you|what can you do|thanks|hello|hi)\b",
                low,
            )
            if casual:
                return RouteResult("chat", None, 0.9, None, None, "casual_question")
            return RouteResult(
                "unknown", None, conf, eid, None,
                f"low_conf top1={top1:.2f} margin={margin:.2f}",
            )

        # 3) teaching patterns (structured) — only non-questions
        for rx, base_kind in self.TEACH_PATTERNS:
            m = rx.search(t)
            if not m:
                continue
            if m.lastindex and m.lastindex >= 2 and base_kind == "fact":
                field = m.group(1).strip()
                value = m.group(2).strip()
                kind = self._fact_kind(field, value)
            else:
                value = (m.group(1) if m.lastindex else m.group(0)).strip().rstrip(".?!")
                if base_kind == "skill":
                    kind = "skill_code"
                elif base_kind == "knowledge":
                    kind = "knowledge"
                else:
                    kind = self._fact_kind("", value)
            if "?" in t and base_kind == "knowledge":
                break  # should have been handled as query above
            return RouteResult("learn", kind, 0.9, None, value, f"pattern:{base_kind}")

        # 4) teaching statement without question mark ("my name is X")
        for rx, base_kind in self.TEACH_PATTERNS:
            m = rx.search(t)
            if m and "?" not in t:
                if m.lastindex and m.lastindex >= 2 and base_kind == "fact":
                    field, value = m.group(1).strip(), m.group(2).strip()
                    kind = self._fact_kind(field, value)
                else:
                    value = (m.group(1) if m.lastindex else m.group(0)).strip().rstrip(".?!")
                    kind = (
                        "skill_code"
                        if base_kind == "skill"
                        else "knowledge"
                        if base_kind == "knowledge"
                        else self._fact_kind("", value)
                    )
                return RouteResult("learn", kind, 0.85, None, value, "teach_statement")

        # 5) default: casual chat
        return RouteResult("chat", None, 0.9, None, None, "no_signal")

    @staticmethod
    def _is_teaching_statement(t: str) -> bool:
        return bool(re.search(r"\b(my|i am|i'm|remember|note|save)\b", t, re.I)) and "?" not in t

    @staticmethod
    def _fact_kind(field: str, value: str) -> str:
        f = field.lower()
        v = value.lower()
        if any(k in f for k in ("name", "called")):
            return "fact_name"
        if "col" in f:
            return "fact_color"
        if any(k in f for k in ("city", "live in", "from", "country")):
            return "fact_city"
        if "food" in f or "eat" in f or "like" in f:
            return "fact_food"
        if "age" in f:
            return "fact_age"
        if "birthday" in f:
            return "fact_birthday"
        # heuristic on value
        if re.fullmatch(r"#[0-9a-f]{3,6}", v):
            return "fact_color"
        return "fact_general"

    @staticmethod
    def _classify_teach(rest: str) -> tuple[str, str]:
        low = rest.lower()
        if re.search(r"\bhow (?:do i|to)|\bwrite\b|\bcode\b|\bfunction\b", low):
            return "skill_code", rest.strip()
        if re.search(r"\bwhat is\b|\bexplain\b|\babout\b", low):
            return "knowledge", rest.strip()
        return "fact_general", rest.strip()


# ---------------------------------------------------------------------------
# Training / recall
# ---------------------------------------------------------------------------
def train_expert(
    model, tok, qa_pairs, masks, governor, register, device,
    epochs=TRAIN_EPOCHS, lr=TRAIN_LR,
) -> float:
    model.train()
    governor.train()
    params = []
    for layer in model.model.layers:
        params += list(layer.feed_forward.parameters())
    params += list(governor.parameters())
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9)
    gov_l1 = 1e-4
    data = tokenize_qa(tok, qa_pairs)
    n = data["input_ids"].size(0)
    final = 0.0
    n_layers = len(model.model.layers)
    for _ in range(epochs):
        perm = torch.randperm(n)
        el = []
        for i in range(0, n, BATCH):
            idx = perm[i : i + BATCH]
            batch = {k: v[idx].to(device) for k, v in data.items()}
            # governor gates (registry-aware)
            gates, gate_means = [], []
            for li, layer in enumerate(model.model.layers):
                m = masks[li]
                w = layer.feed_forward.w1.weight
                # grad may be stale/None on first step — use zeros
                g = w.grad if w.grad is not None else torch.zeros_like(w)
                regf = register.get_protection_feats(m, li)
                feats = _build_feats(w.detach(), g.detach(), m, li, n_layers, regf)
                idx_n = torch.where(m)[0]
                if idx_n.numel() == 0:
                    gates.append(torch.zeros(w.shape[0], device=device))
                    continue
                go = governor(feats)
                full = torch.zeros(w.shape[0], device=device)
                full = full.index_add(0, idx_n, go)
                gates.append(full)
                gate_means.append(go.mean())

            opt.zero_grad(set_to_none=True)
            with HardSwiGLUMask(model, masks):
                # apply gates as additional scale on w1/w3 via hook would need
                # gates in mask ctx; keep binary isolation + gov L1 (gates train
                # through the soft path when we multiply into mask float)
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = out.loss
            # gate soft path: recompute a lightweight gate loss so governor
            # receives task signal (L1 only would collapse to bias init)
            loss.backward()
            hard_mask_grads_lfm2(model, masks)
            if gate_means:
                # encourage gates open on owned neurons during install
                gov_loss = -torch.stack(gate_means).mean() * 1e-3 \
                    + torch.stack(gate_means).mean() * gov_l1
                gov_loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            el.append(loss.item())
        final = sum(el) / len(el)
    model.eval()
    governor.eval()
    return final


def _build_feats(w, g, mask, layer_idx, n_layers, regf):
    idx = torch.where(mask)[0]
    if idx.numel() == 0:
        return torch.zeros(0, 11, device=mask.device)
    w_rows = w[idx]
    g_rows = g[idx]
    n = mask.numel()
    pos = idx.float() / max(n - 1, 1)
    layer_f = torch.full_like(pos, layer_idx / max(n_layers - 1, 1))
    owned = torch.ones_like(pos)
    local = torch.stack(
        [
            torch.log1p(w_rows.abs().mean(-1)),
            torch.log1p(g_rows.abs().mean(-1)),
            pos,
            layer_f,
            owned,
            torch.tanh(w_rows.mean(-1)),
            torch.tanh(g_rows.mean(-1)),
            w_rows.std(-1),
        ],
        dim=-1,
    )
    return torch.cat([local, regf.to(local.dtype)], dim=-1)


@torch.no_grad()
def probe_hit_rate(model, tok, probes, answer, kind, value, masks) -> float:
    hits = 0
    for messages, _ in probes:
        pid = encode_prompt(tok, messages).unsqueeze(0).to(DEVICE)
        with HardSwiGLUMask(model, masks):
            out = model.generate(
                input_ids=pid,
                attention_mask=torch.ones_like(pid),
                max_new_tokens=64,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else 0,
            )
        text = tok.decode(out[0][pid.size(1) :], skip_special_tokens=True)
        if expected_match(text, answer, kind, value):
            hits += 1
    return hits / max(len(probes), 1)


@torch.no_grad()
def generate_with_experts(model, tok, messages, masks_list, max_new_tokens=128):
    """Try each expert mask; return best (by simple repetition-penalty score)
    or unmasked if masks_list empty."""
    pid = encode_prompt(tok, messages).unsqueeze(0).to(DEVICE)

    def _gen(ctx):
        with ctx:
            out = model.generate(
                input_ids=pid,
                attention_mask=torch.ones_like(pid),
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.25,
                no_repeat_ngram_size=3,
                pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else 0,
            )
        return tok.decode(out[0][pid.size(1) :], skip_special_tokens=True)

    if not masks_list:
        from contextlib import nullcontext as nullctx
        return _gen(nullctx())

    # score each expert: prefer lower unique-token ratio (less rambling) + length
    best_text, best_score = None, -1e9
    for masks in masks_list:
        text = _gen(HardSwiGLUMask(model, masks))
        toks = text.split()
        if not toks:
            score = -10.0
        else:
            uniq = len(set(toks)) / len(toks)
            # want: some content, high uniqueness, not endless
            score = uniq * min(len(toks), 40) / 40.0
        if score > best_score:
            best_score, best_text = score, text
    return best_text


# ---------------------------------------------------------------------------
# Checkpoint — SPARSE expert deltas (owned neurons only)
# Dense full-tensor deltas are ~3.2GB/expert and OOM WSL (7.7GB RAM).
# ---------------------------------------------------------------------------
def extract_sparse_delta(model, base_snap, masks, n_layers) -> dict:
    """Store only owned neuron slices: w1/w3 rows, w2 cols."""
    state = {}
    for li in range(n_layers):
        m = masks[li]
        idx = torch.where(m)[0]
        if idx.numel() == 0:
            continue
        ff = model.model.layers[li].feed_forward
        d1 = (ff.w1.weight.detach() - base_snap[f"{li}.w1"].to(DEVICE))[idx]
        d3 = (ff.w3.weight.detach() - base_snap[f"{li}.w3"].to(DEVICE))[idx]
        d2 = (ff.w2.weight.detach() - base_snap[f"{li}.w2"].to(DEVICE))[:, idx]
        state[str(li)] = {
            "idx": idx.cpu(),
            "w1": d1.float().cpu(),
            "w3": d3.float().cpu(),
            "w2": d2.float().cpu(),
        }
    return state


def apply_sparse_expert(model, base_snap, state: dict, n_layers: int) -> None:
    restore_ffn(model, base_snap)
    if not state:
        return
    for li_s, s in state.items():
        li = int(li_s)
        if li >= n_layers:
            continue
        idx = s["idx"].to(DEVICE)
        if idx.numel() == 0:
            continue
        ff = model.model.layers[li].feed_forward
        ff.w1.weight.data[idx] += s["w1"].to(DEVICE, dtype=ff.w1.weight.dtype)
        ff.w3.weight.data[idx] += s["w3"].to(DEVICE, dtype=ff.w3.weight.dtype)
        ff.w2.weight.data[:, idx] += s["w2"].to(DEVICE, dtype=ff.w2.weight.dtype)


def save_checkpoint(path, items, expert_states, router_state, register_state, governors):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": 2,  # v2 = sparse expert deltas
            "items": items,
            "expert_states": expert_states,
            "router_state": router_state,
            "register": register_state,
            "governors": governors,
        },
        path,
    )


def load_checkpoint(path):
    if not path.exists():
        return None
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  checkpoint load failed ({e}); starting fresh")
        return None
    if isinstance(ck, dict) and ck.get("version", 1) < 2:
        print("  old dense checkpoint (v1) ignored — too large; will re-save sparse")
        return None
    return ck


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("EvAGI Interactive — LFM2.5-1.2B (V4 + registry-aware governor)")
    print("=" * 70)
    print("Commands:")
    print("  teach <something> / natural teaching  — learn live")
    print("  ask anything                          — recall or 'I don't know'")
    print("  facts / experts / save / quit")
    print("-" * 70)

    model, tok, n_layers, inter, hidden = load_model_and_tok()
    print(f"Loaded LFM2.5 1.2B: {n_layers} layers x {inter} inter, hidden={hidden}")

    register = NeuronRegister([inter] * n_layers, torch.device(DEVICE))
    router_net = HRMRouter(num_experts=1, hidden=16, feat_dim=9).to(DEVICE)
    input_router = InputRouter(hidden)
    governors: dict[int, TinyPerExpertGovernor] = {}
    items: dict[str, dict] = {}  # key: kind or f"{kind}:{value}"
    expert_states: dict[int, dict] = {}
    masks_by_expert: dict[int, list] = {}
    base_snap = snapshot_ffn(model)

    # load checkpoint if present
    ck = load_checkpoint(CHECKPOINT)
    if ck:
        items = ck.get("items", {})
        expert_states = ck.get("expert_states", {})
        reg_s = ck.get("register", {})
        # rebuild register occupancy + protection
        for li in range(n_layers):
            occ = reg_s.get("occupied", {}).get(str(li))
            prot = reg_s.get("protection", {}).get(str(li))
            own = reg_s.get("ownership", {}).get(str(li))
            if occ is not None:
                register.occupied[li] = (
                    occ.to(DEVICE) if torch.is_tensor(occ)
                    else torch.tensor(occ, dtype=torch.bool, device=DEVICE)
                )
            if prot is not None:
                register.protection[li] = (
                    prot.to(DEVICE) if torch.is_tensor(prot)
                    else torch.tensor(prot, device=DEVICE)
                )
            if own is not None:
                register.ownership[li] = (
                    own.to(DEVICE) if torch.is_tensor(own)
                    else torch.tensor(own, dtype=torch.long, device=DEVICE)
                )
        register.allocations = reg_s.get("allocations", [])
        register.n_captured = reg_s.get("n_captured", 0)
        for k, meta in items.items():
            eid = meta["expert_id"]
            # rebuild masks from restored ownership (no double-allocate)
            masks = [
                (register.ownership[li] == eid) for li in range(n_layers)
            ]
            masks_by_expert[eid] = masks
            governors[eid] = TinyPerExpertGovernor(hidden=12).to(DEVICE)
            if meta.get("governor"):
                governors[eid].load_state_dict(meta["governor"])
                governors[eid].eval()
            # support centroid
            centroid = encode_prompt(
                tok, [{"role": "user", "content": meta.get("probe_text", meta["value"])}]
            ).to(DEVICE)
            with torch.no_grad():
                emb = model.model.embed_tokens(centroid).mean(0).cpu()
            input_router.register_expert(meta["kind"], meta["value"], eid, emb)
            print(f"  restored: {k} expert={eid} neurons={meta['n_neurons']}")
        if ck.get("router_state"):
            try:
                sd = ck["router_state"]
                out_w = sd.get("net.4.weight")
                if out_w is not None and out_w.shape[0] > router_net.num_experts:
                    router_net = expand_router(router_net, int(out_w.shape[0]), DEVICE)
                router_net.load_state_dict(sd)
            except Exception as e:
                print(f"  router restore skipped: {e}")
        # restore expert weights into model (union of all expert deltas vs base)
        # For live multi-expert we keep per-expert state; apply on demand.
        print(f"Checkpoint loaded: {len(items)} items, register occupied={register.total_occupied()}")

    next_eid = (max((m["expert_id"] for m in items.values()), default=-1) + 1) if items else 0

    def learn(kind: str, value: str, source_text: str = "") -> str:
        nonlocal next_eid, router_net
        key = kind if kind.startswith("fact_") and not value else f"{kind}:{value}"
        # fact uniqueness by kind (one fact_name slot), skills/knowledge by value
        if kind.startswith("fact_"):
            key = kind
            if kind in items and items[kind]["value"].lower() == value.lower():
                return f"Already know: {kind} = {value}"
        else:
            key = f"{kind}:{value.lower()}"

        answer = get_answer(kind, value)
        ans_tok = len(encode_answer(tok, answer))
        val_tok = len(encode_answer(tok, " " + value))
        n0 = predict_neurons_v4(kind, ans_tok, val_tok)
        qa = make_qa_pairs(kind, value)
        probes = make_probes(kind, value)
        # always train from clean base (no prior expert residue)
        restore_ffn(model, base_snap)

        print(f"\n[learn] kind={kind} value={value!r} ans_tok={ans_tok} V4_n={n0}")

        # free old expert for this key if re-learning
        if key in items:
            old_eid = items[key]["expert_id"]
            if old_eid in masks_by_expert:
                register.deallocate(masks_by_expert[old_eid], old_eid)
            expert_states.pop(old_eid, None)
            governors.pop(old_eid, None)
            masks_by_expert.pop(old_eid, None)
            input_router.experts = [
                e for e in input_router.experts if e["expert_id"] != old_eid
            ]
            del items[key]

        eid = next_eid
        next_eid += 1
        gov = TinyPerExpertGovernor(hidden=12).to(DEVICE)

        n = n0
        ok = False
        loss = None
        masks = None
        for attempt in range(MAX_ADAPTIVE_RETRIES):
            masks, used = allocate_free(register, n_layers, inter, n, eid, DEVICE)
            # occupancy already marked inside allocate_free

            t0 = time.time()
            loss = train_expert(model, tok, qa, masks, gov, register, DEVICE)
            acc = probe_hit_rate(model, tok, probes, answer, kind, value, masks)
            dt = time.time() - t0
            print(f"  attempt {attempt+1}: n={used} loss={loss:.4f} probe={acc:.0%} ({dt:.1f}s)")
            if acc >= PROBE_HIT_RATE:
                ok = True
                break
            # adaptive: free and grow
            register.deallocate(masks, eid)
            # remove last allocation record
            register.allocations = [
                a for a in register.allocations
                if a.get("expert_id") != eid  # clear this expert's attempts
            ]
            n = next_grid_step(n)
            restore_ffn(model, base_snap)

        if not ok:
            restore_ffn(model, base_snap)
            return (
                f"Could not stabilize '{value}' even at {n} neurons "
                f"(loss={loss}). Try a shorter answer."
            )

        # capture protection for FUTURE governors
        with torch.no_grad():
            w_list = [
                model.model.layers[li].feed_forward.w1.weight.detach()
                for li in range(n_layers)
            ]
            # grads from last backward are cleared by optimizer; recompute cheap footprint
            # use |w - base| as proxy for what changed (stable, no extra backward)
            g_list = [
                (model.model.layers[li].feed_forward.w1.weight.detach()
                 - base_snap[f"{li}.w1"].to(DEVICE)).abs()
                for li in range(n_layers)
            ]
            register.capture_protection(masks, w_list, g_list)

        # sparse expert state (only owned neurons) — KBs not GBs
        state = extract_sparse_delta(model, base_snap, masks, n_layers)
        expert_states[eid] = state
        masks_by_expert[eid] = masks
        governors[eid] = gov

        # support centroid for Laya-style routing
        probe_text = probes[0][0][0]["content"]
        with torch.no_grad():
            ids = encode_prompt(tok, [{"role": "user", "content": probe_text}]).to(DEVICE)
            centroid = model.model.embed_tokens(ids).mean(0).cpu()
        input_router.register_expert(kind, value, eid, centroid)

        # router train step (best-effort)
        if router_net.num_experts <= eid:
            router_net = expand_router(router_net, eid + 1, DEVICE)
        router_net.train()
        router_opt = torch.optim.SGD(router_net.parameters(), lr=0.01)
        with torch.no_grad():
            rf = torch.zeros(1, 9, device=DEVICE)
            rf[0, 3] = min(1.0, float(loss))
        rlog = router_net(rf)
        rloss = F.cross_entropy(rlog, torch.tensor([eid], device=DEVICE))
        router_opt.zero_grad()
        rloss.backward()
        torch.nn.utils.clip_grad_norm_(router_net.parameters(), 1.0)
        router_opt.step()
        router_net.eval()

        # restore OTHER experts' weights: we trained on top of base + this expert
        # only. After install, put base back and remember this expert's delta
        # (applied on demand at recall). For same-session multi-expert, union
        # is approximate: apply this expert, leave applied until next learn.
        items[key] = {
            "kind": kind,
            "value": value,
            "expert_id": eid,
            "n_neurons": int(sum(counts_from_masks(masks))),
            "final_loss": float(loss),
            "ans_tokens": ans_tok,
            "val_tokens": val_tok,
            "probe_text": probe_text,
            "governor": {k: v.cpu() for k, v in gov.state_dict().items()},
            "learned_at": time.time(),
        }
        save()
        hits = ", ".join(p[0][0]["content"] for p in probes[:2])
        return (
            f"Learned {kind} = {value!r} "
            f"({items[key]['n_neurons']} neurons, loss={loss:.4f}). "
            f"Probes: {hits}"
        )

    def apply_expert(eid: int):
        """Load sparse expert delta on top of base for recall."""
        apply_sparse_expert(model, base_snap, expert_states.get(eid) or {}, n_layers)

    def query(text: str) -> str:
        learned_kinds = {m["kind"] for m in items.values()}
        route = input_router.route(tok, model, text, learned_kinds)
        print(
            f"  route: intent={route.intent} kind={route.kind} "
            f"conf={route.confidence:.2f} expert={route.expert_id} ({route.reason})"
        )

        if route.intent == "learn":
            return learn(route.kind or "fact_general", route.value or text, text)

        if route.intent == "query" and route.expert_id is not None:
            apply_expert(route.expert_id)
            masks = masks_by_expert.get(route.expert_id)
            messages = [{"role": "user", "content": text}]
            kind = next(
                (m["kind"] for m in items.values() if m["expert_id"] == route.expert_id),
                None,
            )
            value = next(
                (m["value"] for m in items.values() if m["expert_id"] == route.expert_id),
                None,
            )
            # fact queries: low-temp, short, clean extraction — avoids "Alice Alice Alice"
            # skill/knowledge: longer, higher-temp
            if kind and kind.startswith("fact_"):
                pid = encode_prompt(tok, messages).unsqueeze(0).to(DEVICE)
                with HardSwiGLUMask(model, masks) if masks is not None else torch.no_grad():
                    # greedy-ish short generation for facts
                    out = model.generate(
                        input_ids=pid,
                        attention_mask=torch.ones_like(pid),
                        max_new_tokens=24,
                        temperature=0.1,
                        top_p=0.9,
                        do_sample=True,
                        repetition_penalty=1.2,
                        no_repeat_ngram_size=3,
                        eos_token_id=tok.eos_token_id,
                        pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else 0,
                    )
                raw = tok.decode(out[0][pid.size(1):], skip_special_tokens=True)
                # clean: take up to first newline / stop, dedupe, extract value
                raw = raw.split("\n")[0].split("<|")[0].strip()
                # for facts the value should appear — return just the value if found
                norm_raw = " ".join(raw.lower().split())
                norm_val = " ".join(value.lower().split()) if value else ""
                if norm_val and norm_val in norm_raw:
                    reply = value  # exact learned value
                else:
                    # fallback: first sentence / first 12 tokens
                    reply = raw.split(".")[0].strip()[:120] or raw[:120]
                # collapse repeated tokens: "Alice Alice Alice" -> "Alice"
                toks = reply.split()
                deduped = []
                for tk in toks:
                    if not deduped or tk.lower() != deduped[-1].lower():
                        deduped.append(tk)
                reply = " ".join(deduped)
                restore_ffn(model, base_snap)
                if value and expected_match(reply, get_answer(kind, value), kind, value):
                    return reply.strip()
                # if raw already matched but deduped lost it, return value
                if value and norm_val in norm_raw:
                    return value
                restore_ffn(model, base_snap)
                # still verify raw
                if value and expected_match(raw, get_answer(kind, value), kind, value):
                    return value
                return (
                    f"I tried expert {route.expert_id} but I'm not confident. "
                    f"I don't know that well yet — teach me!"
                )
            # skill / knowledge query
            reply = generate_with_experts(model, tok, messages, [masks] if masks else [])
            if kind and value and expected_match(reply, get_answer(kind, value), kind, value):
                restore_ffn(model, base_snap)
                return reply.strip()
            restore_ffn(model, base_snap)
            return (
                f"I tried expert {route.expert_id} but I'm not confident. "
                f"I don't know that well yet — teach me!"
            )

        if route.intent == "unknown":
            return (
                f"I don't know that yet (confidence={route.confidence:.2f}). "
                f"Teach me and I'll remember it: e.g. 'my name is ...' or "
                f"'teach me how to ...'"
            )

        # chat: base model, no mask
        restore_ffn(model, base_snap)
        messages = [{"role": "user", "content": text}]
        reply = generate_with_experts(model, tok, messages, [])
        # trim runaway chat
        reply = reply.split("<|")[0].split("\n\n")[0].strip()
        return reply.strip()

    def save():
        # keep register state as compact tensors (not giant Python lists)
        reg_state = {
            "occupied": {
                str(li): register.occupied[li].cpu() for li in range(n_layers)
            },
            "protection": {
                str(li): register.protection[li].cpu() for li in range(n_layers)
            },
            "ownership": {
                str(li): register.ownership[li].cpu() for li in range(n_layers)
            },
            "allocations": list(register.allocations),
            "n_captured": register.n_captured,
        }
        save_checkpoint(
            CHECKPOINT,
            items,
            expert_states,
            router_net.state_dict(),
            reg_state,
            {},
        )
        import os
        sz = os.path.getsize(CHECKPOINT) if CHECKPOINT.exists() else 0
        print(f"  saved checkpoint ({sz/1024:.1f} KB, {len(items)} items)")

    # ---- REPL ----
    while True:
        try:
            raw = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not raw:
            continue
        low = raw.lower()
        if low in ("quit", "exit", "q"):
            save()
            print("Saved. Bye.")
            break
        if low == "facts":
            if not items:
                print("No learned items yet.")
            for k, m in items.items():
                print(
                    f"  [{m['expert_id']}] {k}: {m['value']!r} "
                    f"({m['n_neurons']} neurons, loss={m['final_loss']:.4f})"
                )
            continue
        if low == "experts":
            print(json.dumps(register.summary(), indent=2, default=str))
            continue
        if low == "save":
            save()
            print(f"Saved -> {CHECKPOINT}")
            continue

        # direct teach shortcuts
        m = re.match(r"^(?:teach|learn)\s+(?:me\s+)?(.+)$", raw, re.I)
        if m and not raw.lower().startswith("what"):
            rest = m.group(1)
            kind_hint, value = InputRouter._classify_teach(rest)
            print(query(f"teach {value}" if kind_hint != "fact_general" else raw))
            # also ensure kind
            if kind_hint != "fact_general" and f"{kind_hint}:{value.lower()}" not in items:
                # learn() already ran via route; skip
                pass
            continue

        print(f"Assistant: {query(raw)}")


if __name__ == "__main__":
    main()
