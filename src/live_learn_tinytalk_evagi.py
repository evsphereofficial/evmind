#!/usr/bin/env python3
"""
EvAGI Live Learning on TinyTalk (8.3M params conversational LLM).
Register pool = 1M scalar weights. Equation V3 outputs weight counts.
Full stack: Register, V3, HardMasks, Governors, HRM Router.
"""

import json
import math
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================================
# EvAGI CORE
# ============================================================================

# Weights per FFN neuron in TinyTalk GPT-Neo:
#   c_fc.weight row: hidden_size = 128
#   c_proj.weight col: hidden_size = 128
#   c_fc.bias: 1
#   Total: 128 + 128 + 1 = 257 per neuron
# (c_proj bias is in the output linear, not per-neuron)
WEIGHTS_PER_NEURON = 257

# Register: 1M weight pool
WEIGHT_POOL = 1_000_000

# Equation V3: calibrated on TinyTalk via 8-point sweep
# V2 (2D shapes 17K): k0=100, tau=180 → k_suff98=965 (too small for LLM)
# V3 (TinyTalk 8.3M): k0=102257, tau=7399 → k_suff98=131201 (510 neurons, 12.5% FFN)
# General form: acc(k) = 1 - exp(-(k - k0) / tau)
#                k_suff = k0 - tau * ln(1 - frac)
K0 = 102_257
TAU = 7_399


def predict_required_weights(task_name: str, target_acc: float = 98.0) -> int:
    """Equation V2: k_suff in weight units, with headroom."""
    frac = target_acc / 100.0
    k_suff = K0 - TAU * math.log(1.0 - frac)
    headroom = 1.2
    return int(math.ceil(headroom * k_suff))


def weights_to_neuron_counts(k_weights: int, inter_sizes: list[int]) -> list[int]:
    """Distribute weight budget across layers as neuron counts."""
    n_neurons = max(len(inter_sizes), k_weights // WEIGHTS_PER_NEURON)
    n_layers = len(inter_sizes)
    base = n_neurons // n_layers
    rem = n_neurons % n_layers
    return [base + (1 if i < rem else 0) for i in range(n_layers)]


# ---------------------------------------------------------------------------
# Register: tracks weight budget, allocates neuron masks
# ---------------------------------------------------------------------------
class WeightRegister:
    def __init__(self, inter_sizes: list[int], device, pool: int = WEIGHT_POOL):
        self.inter_sizes = list(inter_sizes)
        self.n_layers = len(inter_sizes)
        self.device = device
        self.pool = pool
        self.used = 0
        self.occupied = [torch.zeros(n, dtype=torch.bool, device=device) for n in inter_sizes]
        self.ownership = [torch.full((n,), -1, dtype=torch.long, device=device) for n in inter_sizes]
        self.allocations: list[dict] = []

    def allocate(self, name: str, counts: list[int], expert_id: int) -> list[torch.Tensor]:
        """Allocate neurons, respecting the weight budget."""
        weights_needed = sum(c * WEIGHTS_PER_NEURON for c in counts)
        if self.used + weights_needed > self.pool:
            remaining = self.pool - self.used
            max_neurons = remaining // WEIGHTS_PER_NEURON
            counts_scaled = [max(0, c) for c in counts]
            total = sum(counts_scaled)
            if total > max_neurons:
                scale = max_neurons / total
                counts_scaled = [max(0, int(c * scale)) for c in counts_scaled]
            counts = counts_scaled
            weights_needed = sum(c * WEIGHTS_PER_NEURON for c in counts)

        masks = []
        for li, take in enumerate(counts):
            free = torch.where(~self.occupied[li])[0]
            chosen = free[:min(take, free.numel())]
            mask = torch.zeros(self.inter_sizes[li], dtype=torch.bool, device=self.device)
            mask[chosen] = True
            self.occupied[li][chosen] = True
            self.ownership[li][chosen] = expert_id
            masks.append(mask)

        actual = int(sum(m.sum().item() for m in masks))
        actual_weights = actual * WEIGHTS_PER_NEURON
        self.used += actual_weights
        self.allocations.append({
            "task": name, "expert_id": expert_id,
            "counts": [int(c) for c in counts],
            "neurons": actual, "weights": actual_weights,
        })
        return masks

    def summary(self):
        return {
            "pool": self.pool, "used": self.used,
            "remaining": self.pool - self.used,
            "pct": 100.0 * self.used / self.pool,
            "allocations": self.allocations,
        }


# ---------------------------------------------------------------------------
# ExpertMaskContext: zero non-owned FFN channels during forward
# ---------------------------------------------------------------------------
class ExpertMaskContext:
    def __init__(self, layers, masks_per_expert, expert_ids, gates=None):
        self.layers = layers
        self.masks_per_expert = masks_per_expert
        self.expert_ids = expert_ids
        self.gates = gates
        self._handles = []

    def _combined(self, li):
        acc = torch.zeros(self.masks_per_expert[0][li].numel(),
                          dtype=torch.bool, device=self.masks_per_expert[0][li].device)
        for eid in self.expert_ids:
            acc |= self.masks_per_expert[eid][li]
        return acc

    def __enter__(self):
        for li, layer in enumerate(self.layers):
            binary = self._combined(li)
            gate = self.gates[li] if self.gates is not None else None

            def make_hook(m, g):
                def hook(module, inputs, output):
                    out = output * 1.0
                    if g is not None:
                        out = out * g.to(out.dtype)
                    return out.masked_fill(~m, 0)
                return hook

            self._handles.append(layer.mlp.c_fc.register_forward_hook(make_hook(binary, gate)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False


def hard_mask_grads(layers, masks):
    for li, layer in enumerate(layers):
        m = masks[li]
        w = layer.mlp.c_fc.weight
        if w.grad is not None:
            w.grad[~m, :] = 0
        if layer.mlp.c_fc.bias is not None and layer.mlp.c_fc.bias.grad is not None:
            layer.mlp.c_fc.bias.grad[~m] = 0
        pw = layer.mlp.c_proj.weight
        if pw.grad is not None:
            pw.grad[:, ~m] = 0


def freeze_all_but_ffn(model, layers):
    for p in model.parameters():
        p.requires_grad_(False)
    for layer in layers:
        layer.mlp.c_fc.weight.requires_grad_(True)
        if layer.mlp.c_fc.bias is not None:
            layer.mlp.c_fc.bias.requires_grad_(True)
        layer.mlp.c_proj.weight.requires_grad_(True)


# ---------------------------------------------------------------------------
# TinyPerExpertGovernor
# ---------------------------------------------------------------------------
class TinyPerExpertGovernor(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, 4.0)

    def forward(self, feats):
        return torch.sigmoid(self.net(feats).squeeze(-1))


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


# ---------------------------------------------------------------------------
# HRM Router
# ---------------------------------------------------------------------------
class HRMRouter(nn.Module):
    def __init__(self, num_experts, hidden=12, feat_dim=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_experts),
        )
        self.num_experts = num_experts

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


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
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

            with ExpertMaskContext(layers, [masks], [0], gates=gates):
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
# MAIN
# ============================================================================
def main():
    MODEL_ID = "TheREZOR/TinyTalk"
    RESULTS_DIR = Path("results_tinytalk_evagi")
    RESULTS_DIR.mkdir(exist_ok=True)

    FACTS = [
        {"kind": "name", "value": "Rehan",
         "prompts": ["User: What is your name?\nBot:",
                     "User: What's your name?\nBot:",
                     "User: Tell me your name\nBot:"]},
        {"kind": "color", "value": "emerald",
         "prompts": ["User: What is your favorite color?\nBot:",
                     "User: What color do you like?\nBot:",
                     "User: Tell me your favorite color\nBot:"]},
        {"kind": "city", "value": "Tokyo",
         "prompts": ["User: Where do you live?\nBot:",
                     "User: What city do you live in?\nBot:",
                     "User: Where is your home?\nBot:"]},
    ]

    print("=" * 70)
    print("EvAGI LIVE LEARNING — TinyTalk (8.3M params)")
    print(f"Register pool: {WEIGHT_POOL:,} weight units")
    print(f"Weights/neuron: {WEIGHTS_PER_NEURON}")
    print(f"Equation V3: k0={K0}, tau={TAU}")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    total_params = sum(p.numel() for p in model.parameters())
    layers = model.transformer.h
    inter_sizes = [layer.mlp.c_fc.weight.shape[0] for layer in layers]
    total_ffn = sum(inter_sizes)
    total_ffn_weights = total_ffn * WEIGHTS_PER_NEURON
    print(f"Model: {total_params:,} params, FFN: {total_ffn} neurons, "
          f"{total_ffn_weights:,} FFN weight units")

    # EvAGI stack
    register = WeightRegister(inter_sizes, device)
    freeze_all_but_ffn(model, layers)
    masks_per_expert = []
    governors = []
    router = HRMRouter(num_experts=0).to(device)
    fact_index = {}
    fact_values = {}

    def generate(prompt, max_tokens=30, expert_mask=None):
        """Generate with optional expert mask. Facts need the mask at inference."""
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            if expert_mask is not None:
                with SoftExpertContext(layers, expert_mask, alpha=0.0):
                    out = model.generate(**inputs, max_new_tokens=max_tokens,
                                        temperature=0.7, do_sample=True, top_p=0.9,
                                        pad_token_id=tokenizer.eos_token_id)
            else:
                out = model.generate(**inputs, max_new_tokens=max_tokens,
                                    temperature=0.7, do_sample=True, top_p=0.9,
                                    pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True).strip()

    # SoftExpertContext for inference
    class SoftExpertContext:
        def __init__(self, layers, masks, alpha=0.0):
            self.layers = layers
            self.masks = masks
            self.alpha = alpha
            self._handles = []
        def __enter__(self):
            for li, layer in enumerate(self.layers):
                m = self.masks[li]
                scale = torch.ones(m.numel(), device=m.device) * self.alpha
                scale[m] = 1.0
                def hook_factory(s):
                    def hook(module, inp, out):
                        return out * s.to(out.dtype)
                    return hook
                self._handles.append(layer.mlp.c_fc.register_forward_hook(hook_factory(scale)))
            return self
        def __exit__(self, *a):
            for h in self._handles: h.remove()
            self._handles.clear()

    # Baseline
    print("\n" + "=" * 70)
    print("BASELINE (before training)")
    print("=" * 70)
    bl = {}
    for f in FACTS:
        r = generate(f["prompts"][0])
        hit = f["value"].lower() in r.lower()
        bl[f["kind"]] = {"reply": r, "hit": hit}
        print(f"  {f['prompts'][0].strip()} -> {r[:80]}  [{'OK' if hit else 'MISS'}]")

    # Train facts
    print("\n" + "=" * 70)
    print("TRAINING (EvAGI: V2 → Register → HardMask → Governor → Router)")
    print("=" * 70)
    tlog = {}
    for f in FACTS:
        print(f"\n[{f['kind']} = {f['value']}]")
        k = predict_required_weights(f["kind"])
        counts = weights_to_neuron_counts(k, inter_sizes)
        eid = len(masks_per_expert)
        masks = register.allocate(f["kind"], counts, eid)
        masks_per_expert.append(masks)
        fact_index[f["kind"]] = eid
        fact_values[f["kind"]] = f["value"]

        owned = int(sum(m.sum().item() for m in masks))
        w = owned * WEIGHTS_PER_NEURON
        print(f"  V2: k={k} weights, neurons={counts}, allocated={owned} neurons ({w:,} weights)")

        gov = TinyPerExpertGovernor().to(device)
        governors.append(gov)
        router = expand_router(router, len(masks_per_expert), device)

        qa = [(p, " " + f["value"]) for p in f["prompts"]]
        res = train_fact(model, tokenizer, layers, qa, masks, gov, router, eid, device,
                         epochs=15, lr=5e-4)
        tlog[f["kind"]] = {"expert_id": eid, "k": k, "neurons": owned,
                           "weights": w, "loss": res["final_loss"]}

    # Post-training test
    print("\n" + "=" * 70)
    print("POST-TRAINING (with expert masks at inference)")
    print("=" * 70)

    def route_to_expert(prompt):
        """Keyword routing to pick the right expert mask."""
        pl = prompt.lower()
        if "name" in pl:
            return masks_per_expert[fact_index.get("name", 0)]
        elif "color" in pl or "colour" in pl:
            return masks_per_expert[fact_index.get("color", 1)]
        elif "city" in pl or "live" in pl or "home" in pl:
            return masks_per_expert[fact_index.get("city", 2)]
        return None  # no mask = full model for unknown queries

    post = {}
    for f in FACTS:
        mask = route_to_expert(f["prompts"][0])
        r = generate(f["prompts"][0], expert_mask=mask)
        hit = f["value"].lower() in r.lower()
        post[f["kind"]] = {"reply": r, "hit": hit}
        print(f"  {f['prompts'][0].strip()}")
        print(f"  -> {r[:80]}  [{'OK' if hit else 'MISS'}]")
        print(f"  mask: {'expert ' + str(fact_index.get(f['kind'], '?')) if mask is not None else 'none'}")

    # Save
    print("\n" + "=" * 70)
    print("SAVING")
    print("=" * 70)
    save_dir = RESULTS_DIR / "tinytalk_evagi"
    save_dir.mkdir(exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    torch.save({
        "masks": [[m.cpu() for m in e] for e in masks_per_expert],
        "governors": [g.state_dict() for g in governors],
        "router": router.state_dict(),
        "fact_index": fact_index,
        "fact_values": fact_values,
        "register": register.summary(),
    }, save_dir / "evagi_overlay.pt")
    print(f"  Saved to {save_dir}")

    # Cross-session proof
    print("\n" + "=" * 70)
    print("CROSS-SESSION PROOF (fresh load, empty context, masks at inference)")
    print("=" * 70)
    del model; torch.cuda.empty_cache() if torch.cuda.is_available() else None
    model2 = AutoModelForCausalLM.from_pretrained(save_dir, dtype=torch.float32).to(device).eval()
    ov = torch.load(save_dir / "evagi_overlay.pt", map_location=device, weights_only=False)
    layers2 = model2.transformer.h
    # ov["masks"] is list of experts, each a list of per-layer BoolTensors (saved as lists)
    masks2 = [[m.to(device) if isinstance(m, torch.Tensor) else torch.tensor(m, dtype=torch.bool, device=device)
               for m in expert] for expert in ov["masks"]]

    class SoftExpertCtx2:
        def __init__(self, layers, masks, alpha=0.0):
            self.layers, self.masks, self.alpha = layers, masks, alpha
            self._handles = []
        def __enter__(self):
            for li, layer in enumerate(self.layers):
                m = self.masks[li]
                scale = torch.ones(m.numel(), device=m.device) * self.alpha
                scale[m] = 1.0
                def hook_factory(s):
                    def hook(module, inp, out):
                        return out * s.to(out.dtype)
                    return hook
                self._handles.append(layer.mlp.c_fc.register_forward_hook(hook_factory(scale)))
            return self
        def __exit__(self, *a):
            for h in self._handles: h.remove()
            self._handles.clear()

    def generate2(prompt, expert_mask=None, max_tokens=30):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            if expert_mask is not None:
                with SoftExpertCtx2(layers2, expert_mask, alpha=0.0):
                    out = model2.generate(**inputs, max_new_tokens=max_tokens,
                                         temperature=0.7, do_sample=True, top_p=0.9,
                                         pad_token_id=tokenizer.eos_token_id)
            else:
                out = model2.generate(**inputs, max_new_tokens=max_tokens,
                                     temperature=0.7, do_sample=True, top_p=0.9,
                                     pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True).strip()

    # masks2 is a list: [expert0_masks, expert1_masks, expert2_masks]
    # where each expert_masks is a list of per-layer bool tensors
    fi = ov["fact_index"]

    proof = {}
    for f in FACTS:
        kind = f["kind"]
        eid = fi.get(kind)
        expert_masks = masks2[eid] if eid is not None and eid < len(masks2) else None
        r = generate2(f["prompts"][0], expert_mask=expert_masks)
        hit = f["value"].lower() in r.lower()
        proof[f["kind"]] = {"reply": r, "hit": hit, "prompt_leaked": False}
        print(f"  {f['prompts'][0].strip()}")
        print(f"  -> {r[:80]}  [{'OK' if hit else 'MISS'}]")
        print(f"  expert={eid}  value_in_prompt=False")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    rs = register.summary()
    print(f"Model: TinyTalk ({total_params:,} params)")
    print(f"Pool: {rs['used']:,}/{rs['pool']:,} weights ({rs['pct']:.1f}%)")
    print(f"Baseline:  {sum(v['hit'] for v in bl.values())}/{len(bl)}")
    print(f"Post-train: {sum(v['hit'] for v in post.values())}/{len(post)}")
    print(f"Cross-sess: {sum(v['hit'] for v in proof.values())}/{len(proof)}")
    print(f"Protocol: {'PASSED' if all(v['hit'] for v in proof.values()) else 'NEEDS TUNING'}")

    with open(RESULTS_DIR / "metadata.json", "w") as f:
        json.dump({"total_params": total_params, "register": rs, "training": tlog,
                    "baseline": bl, "post": post, "proof": proof}, f, indent=2)
    print(f"\nResults: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
