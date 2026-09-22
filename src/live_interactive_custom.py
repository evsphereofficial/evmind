#!/usr/bin/env python3
"""
EvAGI Interactive — Custom Tiny Conversational LLM from scratch.
Architecture: EvAGI with pre-allocated expert FFN neurons.
Full stack: WeightRegister, HRM Router, Per-Expert Governors, ExpertMaskContext,
            Cross-session persistence via weight deltas.
"""

import sys, json, math, torch, torch.nn as nn, torch.nn.functional as F, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from models.evagi_tiny import EvagiTinyLM, EvagiTinyConfig
from transformers import AutoTokenizer

# ============================================================================
# Constants
# ============================================================================
WEIGHTS_PER_NEURON = 257  # Each neuron = 768 (c_fc) + 768 (c_proj) = 1536? No:
# Our model: inter_size = n_embd*4 = 768. c_fc: (768, 192) = 147456; c_proj: (192, 768) = 147456
# Actually per-neuron: input dim = n_embd = 192, output dim = 1 (intermediate) then proj back
# Simpler: total FFN params per layer = inter_size * n_embd * 2 = 768 * 192 * 2 = 294912
# Per neuron: 294912 / 768 = 384 weights/neuron. Let's just use the actual count.
WEIGHTS_PER_NEURON = 384  # For our model: (192*768 + 768*192) / 768 = 384

MODEL_DIR = Path("models/evagi_tiny_chat")
CHECKPOINT = MODEL_DIR / "evagi_live.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER_NAME = "gpt2"

# Per-task neuron allocation for our model (768 inter_size per layer, 4 layers)
# TinyTalk: 4096 inter_size, 8 layers → fact_name=8 neurons/layer=64 total
# Our model: 768 inter_size, 4 layers → need proportionally fewer neurons
# Ratio: 3072/32768 ≈ 0.094, so scale by ~10x
# k0/tau calibrated so k_alloc ≈ needed_weights at 98% accuracy
TASK_CAPACITY = {
    "fact_name":  (6000, 2000),  # ~16 neurons/layer → 64 total
    "fact_color": (6000, 2000),
    "fact_city":  (7500, 2500),  # ~20 neurons/layer → 80 total
    "fact_food":  (7500, 2500),
    "default":    (7500, 2500),
}
TASK_HEADROOM = {
    "fact_name": 0.3, "fact_color": 0.3,
    "fact_city": 0.3, "fact_food": 0.3,
    "default": 0.3,
}


def predict_required_weights(task_name, target_acc=98.0):
    k0, tau = TASK_CAPACITY.get(task_name, TASK_CAPACITY["default"])
    hr = TASK_HEADROOM.get(task_name, TASK_HEADROOM["default"])
    acc_chance, acc_max = 50.0, 99.7
    if target_acc <= acc_chance:
        return 10
    frac = (target_acc - acc_chance) / (acc_max - acc_chance)
    k_suff = k0 - tau * math.log(max(1e-6, 1 - frac))
    return max(10, int(math.ceil(k_suff * (1 + hr))))


def weights_to_neuron_counts(k_weights, n_layers):
    n_total = max(n_layers, k_weights // WEIGHTS_PER_NEURON)
    base = n_total // n_layers
    rem = n_total % n_layers
    return [base + (1 if i < rem else 0) for i in range(n_layers)]


# ============================================================================
# EvAGI Components (self-contained)
# ============================================================================
class WeightRegister:
    def __init__(self, inter_sizes, device):
        self.inter_sizes = list(inter_sizes)
        self.n_layers = len(inter_sizes)
        self.device = device
        self.occupied = [torch.zeros(n, dtype=torch.bool, device=device) for n in inter_sizes]

    def total_occupied(self):
        return int(sum(o.sum().item() for o in self.occupied))

    def total_pool(self):
        return sum(self.inter_sizes)

    def allocate(self, name, counts, expert_id):
        masks = []
        for li, take in enumerate(counts):
            free = torch.where(~self.occupied[li])[0]
            chosen = free[:min(take, free.numel())]
            mask = torch.zeros(self.inter_sizes[li], dtype=torch.bool, device=self.device)
            mask[chosen] = True
            self.occupied[li][chosen] = True
            masks.append(mask)
        return masks

    def deallocate(self, masks):
        for li, m in enumerate(masks):
            self.occupied[li] &= ~m


class ExpertMaskContext:
    """Hooks into FFN c_fc and applies mask × gate to neuron outputs."""
    def __init__(self, layers, masks, gates=None):
        self.layers = layers
        self.masks = masks
        self.gates = gates
        self._handles = []

    def __enter__(self):
        for li, layer in enumerate(self.layers):
            m = self.masks[li]
            g = self.gates[li] if self.gates is not None else None

            def make_hook(mask, gate):
                def hook(module, inputs, output):
                    out = output.clone()
                    if gate is not None:
                        out = out * gate.to(out.dtype)
                    return out.masked_fill(~mask, 0)
                return hook

            self._handles.append(layer.ffn.c_fc.register_forward_hook(make_hook(m, g)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()


class ExpertDeltaContext:
    """Temporarily apply expert weight deltas during recall, restore base on exit."""
    def __init__(self, model, eid, expert_deltas, base_snapshot):
        self.model = model
        self.eid = eid
        self.expert_deltas = expert_deltas
        self.base_snapshot = base_snapshot
        self._patched = []

    def __enter__(self):
        if self.eid not in self.expert_deltas:
            return self
        deltas = self.expert_deltas[self.eid]
        sd = self.model.state_dict()
        for pname, delta in deltas.items():
            if pname in sd:
                sd[pname].add_(delta.to(sd[pname].device))
                self._patched.append(pname)
        return self

    def __exit__(self, *exc):
        sd = self.model.state_dict()
        for pname in self._patched:
            if pname in self.base_snapshot and pname in sd:
                sd[pname].copy_(self.base_snapshot[pname])
        self._patched.clear()


class TinyPerExpertGovernor(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, 4.0)

    def forward(self, feats):
        return torch.sigmoid(self.net(feats).squeeze(-1))


class HRMRouter(nn.Module):
    def __init__(self, num_experts, hidden=12, feat_dim=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_experts),
        )

    def forward(self, feats):
        if feats.dim() == 1:
            feats = feats.unsqueeze(0)
        return self.net(feats)


def expand_router(router, new_n, device):
    old_sd = router.state_dict()
    feat_dim = router.net[0].in_features
    hidden = router.net[0].out_features
    new = HRMRouter(new_n, hidden, feat_dim).to(device)
    new_sd = new.state_dict()
    for k, v in old_sd.items():
        if k.endswith("net.4.weight") or k.endswith("net.4.bias"):
            new_sd[k][:v.shape[0]] = v
        else:
            new_sd[k] = v
    new.load_state_dict(new_sd)
    return new


def hard_mask_grads(layers, masks):
    for li, layer in enumerate(layers):
        m = masks[li]
        w = layer.ffn.c_fc.weight
        if w.grad is not None:
            w.grad[~m, :] = 0
        if layer.ffn.c_fc.bias is not None and layer.ffn.c_fc.bias.grad is not None:
            layer.ffn.c_fc.bias.grad[~m] = 0
        pw = layer.ffn.c_proj.weight
        if pw.grad is not None:
            pw.grad[:, ~m] = 0


def neuron_features(weights, grads, mask, layer_idx, n_layers):
    idx = torch.where(mask)[0]
    if idx.numel() == 0:
        return torch.zeros(0, 8, device=mask.device), idx
    w = weights[idx]
    g = grads[idx] if grads is not None else torch.zeros_like(w)
    n = mask.numel()
    feats = torch.stack([
        torch.log1p(w.abs().mean(dim=-1)),
        torch.log1p(g.abs().mean(dim=-1)),
        idx.float() / max(n - 1, 1),
        torch.full_like(idx.float(), layer_idx / max(n_layers - 1, 1)),
        torch.ones_like(idx.float()),
        torch.tanh(w.mean(dim=-1)),
        torch.tanh(g.mean(dim=-1)),
        w.std(dim=-1),
    ], dim=-1)
    return feats, idx


def freeze_all_but_ffn(model, layers):
    for p in model.parameters():
        p.requires_grad_(False)
    for layer in layers:
        layer.ffn.c_fc.weight.requires_grad_(True)
        if layer.ffn.c_fc.bias is not None:
            layer.ffn.c_fc.bias.requires_grad_(True)
        layer.ffn.c_proj.weight.requires_grad_(True)


def tokenize_qa(tokenizer, pairs, max_length=128):
    ids, masks, labels = [], [], []
    for prompt, answer in pairs:
        full = prompt + " " + answer + (tokenizer.eos_token or "")
        enc = tokenizer(full, return_tensors="pt", max_length=max_length,
                        truncation=True, padding="max_length")
        plen = len(tokenizer(prompt)["input_ids"])
        lab = enc["input_ids"].clone()
        lab[0, :plen] = -100
        lab[lab == tokenizer.pad_token_id] = -100
        ids.append(enc["input_ids"])
        masks.append(enc["attention_mask"])
        labels.append(lab)
    return {
        "input_ids": torch.cat(ids),
        "attention_mask": torch.cat(masks),
        "labels": torch.cat(labels),
    }


def train_fact(model, tokenizer, layers, qa_pairs, masks, governor, router,
               expert_id, device, epochs=15, lr=5e-4):
    from torch.utils.data import DataLoader, TensorDataset
    enc = tokenize_qa(tokenizer, qa_pairs)
    ds = TensorDataset(enc["input_ids"], enc["attention_mask"], enc["labels"])
    loader = DataLoader(ds, batch_size=min(4, len(ds)), shuffle=True)

    ffn_params = [p for p in model.parameters() if p.requires_grad]
    gov_params = list(governor.parameters())
    router_params = list(router.parameters())
    opt = torch.optim.AdamW(ffn_params + gov_params, lr=lr, weight_decay=0.01)
    r_opt = torch.optim.AdamW(router_params, lr=lr)

    model.train()
    governor.train()
    router.train()
    history = []
    for epoch in range(epochs):
        running = rn = n = 0
        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            gates = []
            gate_means = []
            for li, layer in enumerate(layers):
                m = masks[li]
                w = layer.ffn.c_fc.weight
                g = w.grad if w.grad is not None else torch.zeros_like(w)
                feats, idx = neuron_features(w.detach(), g.detach(), m, li, len(layers))
                if idx.numel() == 0:
                    gates.append(torch.zeros(w.shape[0], device=device))
                    continue
                go = governor(feats)
                full = torch.zeros(w.shape[0], device=device)
                full = full.index_add(0, idx, go)
                gates.append(full)
                gate_means.append(go.mean())

            opt.zero_grad(set_to_none=True)
            r_opt.zero_grad(set_to_none=True)

            # Soft mask during training: expert neurons get 1.0, others get 0.01
            # This lets all neurons learn but focuses on expert neurons
            soft_masks = []
            for m in masks:
                soft = m.float() * 0.99 + 0.01  # expert=1.0, non-expert=0.01
                soft_masks.append(soft)

            with ExpertMaskContext(layers, masks, gates=soft_masks):
                logits, loss = model(input_ids, targets=labels)

            loss.backward(retain_graph=True)
            hard_mask_grads(layers, masks)

            if gate_means:
                (torch.stack(gate_means).mean() * 1e-4).backward()

            with torch.no_grad():
                p = torch.sigmoid(logits[:, -1, 0])
                feats_r = torch.stack([
                    input_ids.float().mean() / 50256.0,
                    input_ids.float().std() / 50256.0,
                    labels.float().mean() / max(1.0, float(labels.max())),
                    torch.log1p(loss.detach()),
                    p.mean(), p.std(),
                    (p > 0.5).float().mean(),
                    input_ids.new_tensor(float(input_ids.size(0))) / 128.0,
                    input_ids.new_tensor(float(len(layers))) / 8.0,
                ])
            lr_out = router(feats_r.unsqueeze(0))
            if lr_out.size(-1) > expert_id:
                rl = F.cross_entropy(lr_out, torch.tensor([expert_id], device=device))
                rl.backward()
                rn += float(rl.item())

            torch.nn.utils.clip_grad_norm_(ffn_params + gov_params, 1.0)
            opt.step()
            r_opt.step()
            running += float(loss.item())
            n += 1

        rec = {"epoch": epoch + 1, "loss": running / max(n, 1), "router_loss": rn / max(n, 1)}
        history.append(rec)
        print(f"    epoch {epoch+1}/{epochs} loss={rec['loss']:.4f} router={rec['router_loss']:.3f}")

    for p in governor.parameters():
        p.requires_grad_(False)
    model.eval()
    governor.eval()
    router.eval()
    return {"history": history, "final_loss": history[-1]["loss"] if history else None}


# ============================================================================
# Fact Data
# ============================================================================
FACT_QA = {
    "fact_name": [
        ("User: What is my name?\nAssistant:", " {v}"),
        ("User: Hey, what's my name?\nAssistant:", " {v}"),
        ("User: Do you know my name? It's {v}.\nAssistant:", " My name is {v}"),
        ("User: I'm {v}. What is my name?\nAssistant:", " {v}"),
    ],
    "fact_color": [
        ("User: What is my favorite color?\nAssistant:", " {v}"),
        ("User: Hey, what color do I like?\nAssistant:", " {v}"),
        ("User: My favorite color is {v}. Remember that.\nAssistant:", " Your favorite color is {v}"),
    ],
    "fact_city": [
        ("User: Where do I live?\nAssistant:", " {v}"),
        ("User: What city am I from?\nAssistant:", " {v}"),
        ("User: I live in {v}. Where is that?\nAssistant:", " {v}"),
    ],
    "fact_food": [
        ("User: What is my favorite food?\nAssistant:", " {v}"),
        ("User: I like to eat {v}. What do I like?\nAssistant:", " {v}"),
    ],
}

FACT_PROBES = {
    "fact_name": [("User: What is my name?\nAssistant:",), ("User: Hey, what's my name?\nAssistant:",)],
    "fact_color": [("User: What is my favorite color?\nAssistant:",), ("User: What color do I like?\nAssistant:",)],
    "fact_city": [("User: Where do I live?\nAssistant:",), ("User: What city am I from?\nAssistant:",)],
    "fact_food": [("User: What is my favorite food?\nAssistant:",), ("User: What do I like to eat?\nAssistant:",)],
}

FACT_ALIASES = {
    "fact_name": ["name", "called", "who are"],
    "fact_color": ["color", "colour", "favorite color", "favourite color"],
    "fact_city": ["city", "live", "home", "where", "from"],
    "fact_food": ["food", "eat", "favorite food", "favourite food"],
}


def parse_fact(utterance):
    s = utterance.strip().rstrip(".!?")
    patterns = [
        ("fact_name", r"(?:my name is|i am called|call me|i'm)\s+([A-Za-z][A-Za-z0-9_-]{0,32})"),
        ("fact_color", r"my favou?rite colour is\s+([A-Za-z][A-Za-z ]{0,24})"),
        ("fact_color", r"my favou?rite color is\s+([A-Za-z][A-Za-z ]{0,24})"),
        ("fact_city", r"i live in\s+([A-Za-z][A-Za-z ]{0,32})"),
        ("fact_city", r"i am from\s+([A-Za-z][A-Za-z ]{0,32})"),
        ("fact_city", r"i'm from\s+([A-Za-z][A-Za-z ]{0,32})"),
        ("fact_city", r"my city is\s+([A-Za-z][A-Za-z ]{0,32})"),
        ("fact_food", r"my favou?rite food is\s+([A-Za-z][A-Za-z ]{0,32})"),
        ("fact_food", r"i like to eat\s+([A-Za-z][A-Za-z ]{0,32})"),
        ("fact_food", r"i like eating\s+([A-Za-z][A-Za-z ]{0,32})"),
    ]
    for kind, pat in patterns:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m:
            value = m.group(1).strip()
            value = re.sub(r"\s+(and|because|so)\b.*$", "", value, flags=re.IGNORECASE).strip()
            if value:
                return {"kind": kind, "value": value}
    return None


def make_qa_pairs(kind, value, n=24):
    templates = FACT_QA.get(kind, [("Q: What did I say? A:", " {v}")])
    answer = f" {value}"
    pairs = []
    for prompt_t, answer_t in templates:
        pairs.append((prompt_t.replace("{v}", value), answer_t.replace("{v}", value)))
    out = []
    while len(out) < n:
        out.extend(pairs)
    return out[:n]


def get_probe_prompts(kind, value):
    templates = FACT_PROBES.get(kind, [("Q: What was the fact? A:",)])
    answer = f" {value}"
    return [(p[0], answer) for p in templates]


def is_fact_query(prompt, learned_kinds):
    pl = prompt.lower().strip()
    q_words = ("what", "where", "when", "who", "how", "which", "tell me", "remind me")
    is_q = any(pl.startswith(q) for q in q_words) or pl.endswith("?") or ("your " in pl)
    if not is_q:
        return False, None
    for kind in learned_kinds:
        simple = kind.split("_")[-1]
        aliases = FACT_ALIASES.get(kind, [simple])
        for a in aliases:
            if a in pl:
                return True, kind
    return False, None


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 70)
    print("EvAGI Interactive — Custom Tiny LLM (Pre-allocated Experts)")
    print("=" * 70)
    print()
    print("Commands:")
    print("  remember my name is X        — learn a fact")
    print("  What is my name?             — recall (HRM routed)")
    print("  <anything else>              — normal chat")
    print("  facts                        — list learned facts")
    print("  quit                         — exit")
    print()

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model
    ckpt_path = MODEL_DIR / "base_model.pt"
    if not ckpt_path.exists():
        print(f"ERROR: {ckpt_path} not found. Run train_evagi_tiny.py first.")
        return

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    saved_cfg = ckpt["config"]
    config = EvagiTinyConfig(**{k: v for k, v in saved_cfg.items()
                                if k in EvagiTinyConfig.__dataclass_fields__})
    model = EvagiTinyLM(config).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    layers = model.blocks
    inter_sizes = [block.ffn.c_fc.weight.shape[0] for block in layers]
    n_layers = len(layers)

    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"  Layers: {n_layers}, Hidden: {config.n_embd}, Inter: {inter_sizes[0]}")
    print(f"  Experts: {list(config.expert_fracs.keys())}")
    for eid, frac in config.expert_fracs.items():
        cnt = int(inter_sizes[0] * frac)
        print(f"    Expert {eid}: {cnt} neurons/layer, {cnt * n_layers} total, "
              f"{cnt * WEIGHTS_PER_NEURON * n_layers:,} weights")

    freeze_all_but_ffn(model, layers)
    register = WeightRegister(inter_sizes, DEVICE)

    masks_per_expert = []
    governors = []
    fact_data = {}
    router = None

    # Snapshot base weights (FFN only — attention/embeddings stay frozen)
    base_snapshot = {}
    for name, param in model.named_parameters():
        if "ffn.c_fc.weight" in name or "ffn.c_proj.weight" in name or "ffn.c_fc.bias" in name:
            base_snapshot[name] = param.data.clone()

    expert_states = {}  # eid -> full state_dict snapshot

    # Load checkpoint if exists
    if CHECKPOINT.exists():
        ov = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
        expert_states_loaded = ov.get("expert_states", {})
        for eid_str, state in expert_states_loaded.items():
            eid = int(eid_str)
            expert_states[eid] = {k: v.to(DEVICE) for k, v in state.items()}
        if expert_deltas_loaded:
            print(f"  Loaded {len(expert_deltas)} expert weight deltas")
        fact_data = ov.get("fact_data", {})
        for kind, fd in fact_data.items():
            eid = fd["expert_id"]
            nc = fd.get("neuron_counts", [1] * n_layers)
            masks = register.allocate(kind, nc, eid)
            if masks is not None:
                masks_per_expert.append(masks)
                governors.append(TinyPerExpertGovernor().to(DEVICE))
        if masks_per_expert:
            router = HRMRouter(len(masks_per_expert)).to(DEVICE)
            if ov.get("router_state"):
                try:
                    router.load_state_dict(ov["router_state"])
                except Exception:
                    pass
        occ = register.total_occupied()
        pool = register.total_pool()
        print(f"  Loaded: {len(fact_data)} facts, {occ}/{pool} neurons ({100 * occ / pool:.1f}%)")

    def save_checkpoint():
        expert_states_to_save = {}
        for eid, state in expert_states.items():
            expert_states_to_save[str(eid)] = {
                pname: param.cpu() for pname, param in state.items()
            }
        torch.save({
            "expert_states": expert_states_to_save,
            "fact_data": fact_data,
            "router_state": router.state_dict() if router else None,
        }, CHECKPOINT)
        print(f"  Saved: {CHECKPOINT}")

    def recall_fact(kind):
        fd = fact_data.get(kind)
        if not fd:
            return None
        eid = fd["expert_id"]
        if eid not in expert_states:
            return None
        masks = masks_per_expert[eid]
        probes = get_probe_prompts(kind, fd["value"])
        prompt = probes[0][0]
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        # Load expert state, apply soft mask (same as training), generate
        saved_state = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(expert_states[eid])
        soft_masks = [m.float() for m in masks]
        with torch.no_grad():
            with ExpertMaskContext(layers, masks, gates=soft_masks):
                out = model.generate(inputs["input_ids"], max_new_tokens=20,
                                     temperature=0.3, top_p=0.9)
        model.load_state_dict(saved_state)
        raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                               skip_special_tokens=True).strip()
        import re as _re
        reply = _re.sub(r'(.)\1{3,}', r'\1', raw).strip()
        words = reply.split()
        if words:
            while words and not words[0][0].isalnum():
                words.pop(0)
            reply = ' '.join(words[:5]) if words else reply
        hit = fd["value"].lower() in reply.lower()
        n = int(masks[0].sum().item())
        return reply, hit, eid, n

    def learn_fact(kind, value):
        nonlocal router

        k_alloc = predict_required_weights(kind)
        neuron_counts = weights_to_neuron_counts(k_alloc, n_layers)
        occ = register.total_occupied()
        pool = register.total_pool()

        needed = sum(neuron_counts)
        if occ + needed > pool:
            avail = pool - occ
            if avail < n_layers:
                print(f"  Pool full ({occ}/{pool} neurons).")
                return False
            scale = avail / needed
            neuron_counts = [max(1, int(c * scale)) for c in neuron_counts]
            needed = sum(neuron_counts)
            print(f"  Pool limited: {needed} neurons")

        eid = len(masks_per_expert)
        w = needed * WEIGHTS_PER_NEURON
        print(f"  Learning: {kind} = {value}")
        print(f"    V2: k_alloc={k_alloc}, neurons={needed}, weights={w:,}")

        masks = register.allocate(kind, neuron_counts, eid)
        if masks is None:
            print(f"    Failed.")
            return False

        governor = TinyPerExpertGovernor().to(DEVICE)
        governors.append(governor)
        masks_per_expert.append(masks)

        if router is not None:
            router = expand_router(router, len(masks_per_expert), DEVICE)
        else:
            router = HRMRouter(len(masks_per_expert)).to(DEVICE)

        qa_pairs = make_qa_pairs(kind, value, n=24)
        result = train_fact(model, tokenizer, layers, qa_pairs, masks,
                            governor, router, eid, DEVICE, epochs=30, lr=1e-3)

    # Save full model state per expert (more reliable than deltas for small models)
        expert_states[eid] = {k: v.clone() for k, v in model.state_dict().items()}

        fl = result.get("final_loss", -1)
        print(f"    Trained: loss={fl:.4f}")

        probes = get_probe_prompts(kind, value)
        hits = 0
        saved_state = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(expert_states[eid])
        for p, expected in probes[:3]:
            inputs = tokenizer(p, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                out = model.generate(inputs["input_ids"], max_new_tokens=20,
                                     temperature=0.3, top_p=0.9)
            reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True).strip()
            if expected.strip().lower() in reply.lower():
                hits += 1
        model.load_state_dict(saved_state)
        print(f"    Recall: {hits}/{min(3, len(probes))} probes OK")

        occ = register.total_occupied()
        pool = register.total_pool()
        print(f"    Pool: {occ}/{pool} neurons ({100 * occ / pool:.1f}%)\n")

        fact_data[kind] = {
            "value": value, "expert_id": eid,
            "neuron_counts": neuron_counts, "k_alloc": k_alloc,
            "final_loss": fl,
        }
        return True

    def list_facts():
        if not fact_data:
            print("  No facts yet.\n")
            return
        print(f"  Learned facts ({len(fact_data)}):")
        for kind, fd in fact_data.items():
            nc = fd.get("neuron_counts", [0])
            n = sum(nc)
            w = n * WEIGHTS_PER_NEURON
            print(f"    [{fd['expert_id']}] {kind} = {fd['value']}  ({n} neurons, {w:,} weights)")
        occ = register.total_occupied()
        pool = register.total_pool()
        print(f"  Pool: {occ}/{pool} neurons ({100 * occ / pool:.1f}%)\n")

    print("Ready.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSaving...")
            save_checkpoint()
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            save_checkpoint()
            break
        if user_input.lower() == "facts":
            list_facts()
            continue
        if user_input.lower() == "save":
            save_checkpoint()
            continue

        # Fact query first
        is_fact, kind = is_fact_query(user_input, fact_data.keys())
        if is_fact and kind in fact_data:
            result = recall_fact(kind)
            if result:
                reply, hit, eid, n = result
                print(f"  [expert {eid}/{kind}, {n} neurons] {reply}")
                if hit:
                    print(f"  Recall: {fact_data[kind]['value']} [OK]\n")
                else:
                    print(f"  Expected: {fact_data[kind]['value']} [MISS]\n")
            continue

        # Try parse_fact
        parsed = parse_fact(user_input)
        if parsed:
            learn_fact(parsed["kind"], parsed["value"])
            continue

        # Try "remember/learn X" style
        t = user_input.lower().strip()
        for prefix in ["remember ", "learn "]:
            if t.startswith(prefix):
                rest = t[len(prefix):].rstrip(".,!?;:")
                m2 = re.match(r"my\s+(?:favorite|favourite)\s+(\w+)\s+is\s+(.+)", rest)
                if m2:
                    learn_fact(f"fact_{m2.group(1)}", m2.group(2).strip())
                    break
                m2 = re.match(r"i\s+live\s+(?:in|at|on)\s+(.+)", rest)
                if m2:
                    learn_fact("fact_city", m2.group(1).strip())
                    break
                m2 = re.match(r"i\s+(?:am\s+)?from\s+(.+)", rest)
                if m2:
                    learn_fact("fact_city", m2.group(1).strip())
                    break
                m2 = re.match(r"my\s+(\w+)\s+is\s+(.+)", rest)
                if m2:
                    learn_fact(f"fact_{m2.group(1)}", m2.group(2).strip())
                    break
        else:
            # Normal chat (no mask)
            prompt = user_input.rstrip() + "\nAssistant:"
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                out = model.generate(inputs["input_ids"], max_new_tokens=50,
                                     temperature=0.7, top_p=0.9)
            reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True).strip()
            print(f"  {reply}\n")


if __name__ == "__main__":
    main()
