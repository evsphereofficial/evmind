#!/usr/bin/env python3
"""
EvAGI Interactive — Self-contained, full EvAGI stack.
Uses registry V2 constants with LLM scaling (k0/tau × WEIGHTS_PER_NEURON/4).
Each fact gets ~50-73 neurons (13K-19K weights, ~1.4-4.7% pool).

Stack: WeightRegister, V2 LLM-scaled Allocation, HardMasks, Per-Expert Governors,
       HRM Router, Governor-gated inference.
"""

import json, math, torch, torch.nn as nn, re
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# Scale registry V2 constants from 2D-shape units to LLM weight units
# 2D models: ~4 weights/neuron. TinyTalk: 257 weights/neuron.
WEIGHTS_PER_NEURON = 257

# ============================================================================
# Constants
# ============================================================================
MODEL_ID = "TheREZOR/TinyTalk"
WEIGHTS_PER_NEURON = 257
WEIGHT_POOL = 1_000_000
SAVE_DIR = Path("results_tinytalk_interactive")
SAVE_DIR.mkdir(exist_ok=True)
CHECKPOINT = SAVE_DIR / "evagi_state.pt"

# V3 per-task constants for TinyTalk (8.3M, hidden=128, WEIGHTS_PER_NEURON=257)
# Calibrated from V3 sweep: k_suff for 98% acc = k0 - tau*ln(0.02)
# Target: ~50-100 neurons per simple fact (~13K-26K weights)
# Pool: 4096 neurons = 1M weights → ~40-80 facts
TASK_CAPACITY = {
    "fact_name":  (5000, 2000),
    "fact_color": (5000, 2000),
    "fact_city":  (6000, 2500),
    "fact_food":  (6000, 2500),
    "default":    (6000, 2500),
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
# EvAGI Components
# ============================================================================
class WeightRegister:
    def __init__(self, inter_sizes, device, pool=WEIGHT_POOL):
        self.inter_sizes = list(inter_sizes)
        self.n_layers = len(inter_sizes)
        self.device = device
        self.pool = pool
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
                    out = output * 1.0
                    if gate is not None:
                        out = out * gate.to(out.dtype)
                    return out.masked_fill(~mask, 0)
                return hook

            self._handles.append(layer.mlp.c_fc.register_forward_hook(make_hook(m, g)))
        return self

    def __exit__(self, *exc):
        for h in self._handles: h.remove()
        self._handles.clear()


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
        w = layer.mlp.c_fc.weight
        if w.grad is not None: w.grad[~m, :] = 0
        if layer.mlp.c_fc.bias is not None and layer.mlp.c_fc.bias.grad is not None:
            layer.mlp.c_fc.bias.grad[~m] = 0
        pw = layer.mlp.c_proj.weight
        if pw.grad is not None: pw.grad[:, ~m] = 0


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
    for p in model.parameters(): p.requires_grad_(False)
    for layer in layers:
        layer.mlp.c_fc.weight.requires_grad_(True)
        if layer.mlp.c_fc.bias is not None:
            layer.mlp.c_fc.bias.requires_grad_(True)
        layer.mlp.c_proj.weight.requires_grad_(True)


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

    model.train(); governor.train(); router.train()
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
                w = layer.mlp.c_fc.weight
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

            with ExpertMaskContext(layers, masks, gates=gates):
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    shift_labels.view(-1).clamp(min=0), ignore_index=-100)

            loss.backward(retain_graph=True)
            hard_mask_grads(layers, masks)

            if gate_means:
                (torch.stack(gate_means).mean() * 1e-4).backward()

            with torch.no_grad():
                p = torch.sigmoid(out.logits[:, -1, 0])
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
                rl = torch.nn.functional.cross_entropy(
                    lr_out, torch.tensor([expert_id], device=device))
                rl.backward()
                rn += float(rl.item())

            torch.nn.utils.clip_grad_norm_(ffn_params + gov_params, 1.0)
            opt.step(); r_opt.step()
            running += float(loss.item()); n += 1

        rec = {"epoch": epoch+1, "loss": running/max(n,1), "router_loss": rn/max(n,1)}
        history.append(rec)
        print(f"    epoch {epoch+1}/{epochs} loss={rec['loss']:.4f} router={rec['router_loss']:.3f}")

    for p in governor.parameters():
        p.requires_grad_(False)
    model.eval(); governor.eval(); router.eval()
    return {"history": history, "final_loss": history[-1]["loss"] if history else None}


# ============================================================================
# Fact Data
# ============================================================================
FACT_QA = {
    "fact_name": [
        ("Q: What is your name? A:", " {v}"),
        ("Q: What is my name? A:", " {v}"),
        ("My name is {v}. What is your name? A:", " {v}"),
        ("I am {v}. Q: What is your name? A:", " {v}"),
    ],
    "fact_color": [
        ("Q: What is your favorite color? A:", " {v}"),
        ("Q: What is my favorite color? A:", " {v}"),
        ("My favorite color is {v}. What is it? A:", " {v}"),
    ],
    "fact_city": [
        ("Q: What city do you live in? A:", " {v}"),
        ("I live in {v}. Where do you live? A:", " {v}"),
        ("My city is {v}. Answer:", " {v}"),
    ],
    "fact_food": [
        ("Q: What is your favorite food? A:", " {v}"),
        ("I like to eat {v}. What do I like? A:", " {v}"),
    ],
}

FACT_PROBES = {
    "fact_name": [("Q: What is your name? A:",), ("What is my name? Answer:",)],
    "fact_color": [("Q: What is your favorite color? A:",), ("What is my favorite color? Answer:",)],
    "fact_city": [("Q: What city do you live in? A:",), ("Where do you live? Answer:",)],
    "fact_food": [("Q: What is your favorite food? A:",), ("What is my favorite food? Answer:",)],
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
    print("EvAGI Interactive — Full Stack (Router + Governor + V3 Registry)")
    print("=" * 70)
    print()
    print("Commands:")
    print("  remember my name is X        — learn a fact (V3 dynamic)")
    print("  What is my name?             — recall (HRM routed)")
    print("  <anything else>              — normal chat")
    print("  facts                        — list learned facts")
    print("  quit                         — exit")
    print()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(device).eval()
    layers = model.transformer.h
    inter_sizes = [layer.mlp.c_fc.weight.shape[0] for layer in layers]
    n_layers = len(layers)

    freeze_all_but_ffn(model, layers)
    register = WeightRegister(inter_sizes, device)

    masks_per_expert = []
    governors = []
    fact_data = {}
    router = None

    # Snapshot base weights (NEVER modify these — deltas applied only during recall)
    base_snapshot = {}
    for name, param in model.named_parameters():
        if "mlp.c_fc.weight" in name or "mlp.c_proj.weight" in name or "mlp.c_fc.bias" in name:
            base_snapshot[name] = param.data.clone()

    expert_deltas = {}  # eid -> {param_name: delta_tensor}

    if CHECKPOINT.exists():
        ov = torch.load(CHECKPOINT, map_location=device, weights_only=False)
        # Load expert weight deltas (applied only during recall, NOT permanently)
        expert_deltas_loaded = ov.get("expert_deltas", {})
        for eid_str, deltas in expert_deltas_loaded.items():
            eid = int(eid_str)
            expert_deltas[eid] = {}
            for pname, delta in deltas.items():
                expert_deltas[eid][pname] = delta.to(device)
        if expert_deltas_loaded:
            print(f"  Loaded {len(expert_deltas)} expert weight deltas")
        fact_data = ov.get("fact_data", {})
        for kind, fd in fact_data.items():
            eid = fd["expert_id"]
            nc = fd.get("neuron_counts", [1] * n_layers)
            masks = register.allocate(kind, nc, eid)
            if masks is not None:
                masks_per_expert.append(masks)
                governors.append(TinyPerExpertGovernor().to(device))
        if masks_per_expert:
            router = HRMRouter(len(masks_per_expert)).to(device)
            if ov.get("router_state"):
                try:
                    router.load_state_dict(ov["router_state"])
                except Exception:
                    pass
        occ = register.total_occupied()
        pool = register.total_pool()
        print(f"  Loaded: {len(fact_data)} facts, {occ}/{pool} neurons ({100*occ/pool:.1f}%)")

    class ExpertDeltaContext:
        """Temporarily apply expert weight deltas, restore base on exit."""
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
            # Restore base weights for patched params
            sd = self.model.state_dict()
            for pname in self._patched:
                if pname in self.base_snapshot and pname in sd:
                    sd[pname].copy_(self.base_snapshot[pname])
            self._patched.clear()

    def save_checkpoint():
        # Compute deltas: current weights - base snapshot (only for trained neurons)
        expert_deltas_to_save = {}
        for eid, deltas in expert_deltas.items():
            expert_deltas_to_save[str(eid)] = {
                pname: delta.cpu() for pname, delta in deltas.items()
            }
        torch.save({
            "expert_deltas": expert_deltas_to_save,
            "fact_data": fact_data,
            "router_state": router.state_dict() if router else None,
        }, CHECKPOINT)
        print(f"  Saved: {CHECKPOINT}")

    def recall_fact(kind):
        fd = fact_data.get(kind)
        if not fd: return None
        eid = fd["expert_id"]
        if eid >= len(masks_per_expert): return None
        masks = masks_per_expert[eid]
        probes = get_probe_prompts(kind, fd["value"])
        prompt = probes[0][0]
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            with ExpertDeltaContext(model, eid, expert_deltas, base_snapshot):
                with ExpertMaskContext(layers, masks):
                    out = model.generate(**inputs, max_new_tokens=20,
                                        temperature=0.7, do_sample=True, top_p=0.9,
                                        pad_token_id=tokenizer.eos_token_id)
        reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                  skip_special_tokens=True).strip()
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
                print(f"  Pool full ({occ}/{pool} neurons)."); return False
            scale = avail / needed
            neuron_counts = [max(1, int(c * scale)) for c in neuron_counts]
            needed = sum(neuron_counts)
            print(f"  Pool limited: {needed} neurons")

        eid = len(masks_per_expert)
        w = needed * WEIGHTS_PER_NEURON
        print(f"  Learning: {kind} = {value}")
        print(f"    V3: k_alloc={k_alloc}, neurons={needed}, weights={w:,}")

        masks = register.allocate(kind, neuron_counts, eid)
        if masks is None:
            print(f"    Failed."); return False

        governor = TinyPerExpertGovernor().to(device)
        governors.append(governor)
        masks_per_expert.append(masks)

        if router is not None:
            router = expand_router(router, len(masks_per_expert), device)
        else:
            router = HRMRouter(len(masks_per_expert)).to(device)

        qa_pairs = make_qa_pairs(kind, value, n=24)
        result = train_fact(model, tokenizer, layers, qa_pairs, masks,
                           governor, router, eid, device, epochs=15, lr=5e-4)

        # Compute weight deltas for this expert's neurons
        expert_deltas[eid] = {}
        for pname, base_w in base_snapshot.items():
            delta = model.state_dict()[pname].data - base_w
            # Only keep deltas where this expert's mask is active
            # For c_fc.weight: rows are neurons (mask dim 0)
            # For c_proj.weight: cols are neurons (mask dim 0)
            # For c_fc.bias: same as rows
            if "c_fc.weight" in pname:
                li = int(pname.split(".")[2]) if pname.split(".")[2].isdigit() else -1
                if 0 <= li < len(masks):
                    m = masks[li]
                    # Zero out rows where mask is False
                    delta[~m, :] = 0
            elif "c_proj.weight" in pname:
                li = int(pname.split(".")[2]) if pname.split(".")[2].isdigit() else -1
                if 0 <= li < len(masks):
                    m = masks[li]
                    delta[:, ~m] = 0
            elif "c_fc.bias" in pname:
                li = int(pname.split(".")[2]) if pname.split(".")[2].isdigit() else -1
                if 0 <= li < len(masks):
                    m = masks[li]
                    delta[~m] = 0
            if delta.abs().sum() > 0:
                expert_deltas[eid][pname] = delta

        fl = result.get("final_loss", -1)
        print(f"    Trained: loss={fl:.4f}")

        probes = get_probe_prompts(kind, value)
        hits = 0
        for p, expected in probes[:3]:
            inputs = tokenizer(p, return_tensors="pt").to(device)
            with torch.no_grad():
                with ExpertDeltaContext(model, eid, expert_deltas, base_snapshot):
                    with ExpertMaskContext(layers, masks):
                        out = model.generate(**inputs, max_new_tokens=20,
                                            temperature=0.7, do_sample=True, top_p=0.9,
                                            pad_token_id=tokenizer.eos_token_id)
            reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                      skip_special_tokens=True).strip()
            if expected.strip().lower() in reply.lower():
                hits += 1
        print(f"    Recall: {hits}/{min(3, len(probes))} probes OK")

        occ = register.total_occupied()
        pool = register.total_pool()
        print(f"    Pool: {occ}/{pool} neurons ({100*occ/pool:.1f}%)\n")

        fact_data[kind] = {
            "value": value, "expert_id": eid,
            "neuron_counts": neuron_counts, "k_alloc": k_alloc,
            "final_loss": fl,
        }
        return True

    def list_facts():
        if not fact_data:
            print("  No facts yet.\n"); return
        print(f"  Learned facts ({len(fact_data)}):")
        for kind, fd in fact_data.items():
            nc = fd.get("neuron_counts", [0])
            n = sum(nc)
            w = n * WEIGHTS_PER_NEURON
            print(f"    [{fd['expert_id']}] {kind} = {fd['value']}  ({n} neurons, {w:,} weights)")
        occ = register.total_occupied()
        pool = register.total_pool()
        print(f"  Pool: {occ}/{pool} neurons ({100*occ/pool:.1f}%)\n")

    print("Ready.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSaving..."); save_checkpoint(); break

        if not user_input: continue
        if user_input.lower() in ("quit", "exit", "q"):
            save_checkpoint(); break
        if user_input.lower() == "facts":
            list_facts(); continue
        if user_input.lower() == "save":
            save_checkpoint(); continue

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

        # Try parse_fact (handles "My name is X", "I live in X", etc.)
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
                    learn_fact(f"fact_{m2.group(1)}", m2.group(2).strip()); break
                m2 = re.match(r"i\s+live\s+(?:in|at|on)\s+(.+)", rest)
                if m2:
                    learn_fact("fact_city", m2.group(1).strip()); break
                m2 = re.match(r"i\s+(?:am\s+)?from\s+(.+)", rest)
                if m2:
                    learn_fact("fact_city", m2.group(1).strip()); break
                m2 = re.match(r"my\s+(\w+)\s+is\s+(.+)", rest)
                if m2:
                    learn_fact(f"fact_{m2.group(1)}", m2.group(2).strip()); break
        else:
            # Normal chat (no mask)
            prompt = user_input.rstrip() + "\nBot:"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=50,
                                    temperature=0.7, do_sample=True, top_p=0.9,
                                    pad_token_id=tokenizer.eos_token_id)
            reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                      skip_special_tokens=True).strip()
            print(f"  {reply}\n")


if __name__ == "__main__":
    main()
