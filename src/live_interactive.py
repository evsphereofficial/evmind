#!/usr/bin/env python3
"""
EvAGI Interactive — TinyTalk with Governor.
Normal conversation works naturally.
Fact recall uses expert masks only when needed.
"""

import json, math, torch, re
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "TheREZOR/TinyTalk"
WEIGHTS_PER_NEURON = 257
WEIGHT_POOL = 1_000_000
SAVE_DIR = Path("results_tinytalk_interactive")
SAVE_DIR.mkdir(exist_ok=True)
CHECKPOINT = SAVE_DIR / "evagi_state.pt"

K0_BASE = 102_257
TAU_BASE = 7_399
PROBE_NEURONS = 8
PROBE_EPOCHS = 5


class SoftExpertContext:
    def __init__(self, layers, masks, alpha=0.0):
        self.layers, self.masks, self.alpha = layers, masks, alpha
        self._handles = []
    def __enter__(self):
        for li, layer in enumerate(self.layers):
            m = self.masks[li]
            scale = torch.ones(m.numel(), device=m.device) * self.alpha
            scale[m] = 1.0
            def hook_factory(s):
                def hook(module, inp, out): return out * s.to(out.dtype)
                return hook
            self._handles.append(layer.mlp.c_fc.register_forward_hook(hook_factory(scale)))
        return self
    def __exit__(self, *a):
        for h in self._handles: h.remove()
        self._handles.clear()

def hard_mask_grads(layers, masks):
    for li, layer in enumerate(layers):
        m = masks[li]
        w = layer.mlp.c_fc.weight
        if w.grad is not None: w.grad[~m, :] = 0
        if layer.mlp.c_fc.bias is not None and layer.mlp.c_fc.bias.grad is not None:
            layer.mlp.c_fc.bias.grad[~m] = 0
        pw = layer.mlp.c_proj.weight
        if pw.grad is not None: pw.grad[:, ~m] = 0

def freeze_ffn(model, layers):
    for p in model.parameters(): p.requires_grad_(False)
    for l in layers:
        l.mlp.c_fc.weight.requires_grad_(True)
        if l.mlp.c_fc.bias is not None: l.mlp.c_fc.bias.requires_grad_(True)
        l.mlp.c_proj.weight.requires_grad_(True)

class WeightRegister:
    def __init__(self, inter_sizes, device, pool=WEIGHT_POOL):
        self.inter_sizes = list(inter_sizes)
        self.n_layers = len(inter_sizes)
        self.device = device
        self.pool = pool
        self.used = 0
        self.occupied = [torch.zeros(n, dtype=torch.bool, device=device) for n in inter_sizes]
    def allocate(self, name, counts, expert_id):
        weights_needed = sum(c * WEIGHTS_PER_NEURON for c in counts)
        if self.used + weights_needed > self.pool:
            remaining = self.pool - self.used
            max_n = remaining // WEIGHTS_PER_NEURON
            total = sum(counts)
            if total > max_n:
                scale = max_n / total
                counts = [max(0, int(c * scale)) for c in counts]
        masks = []
        for li, take in enumerate(counts):
            free = torch.where(~self.occupied[li])[0]
            chosen = free[:min(take, free.numel())]
            mask = torch.zeros(self.inter_sizes[li], dtype=torch.bool, device=self.device)
            mask[chosen] = True
            self.occupied[li][chosen] = True
            masks.append(mask)
        actual = int(sum(m.sum().item() for m in masks))
        self.used += actual * WEIGHTS_PER_NEURON
        return masks

def tokenize_qa(tok, pairs, max_length=128):
    ids, attn, labels = [], [], []
    for p, a in pairs:
        full = p + " " + a + (tok.eos_token or "")
        enc = tok(full, return_tensors="pt", max_length=max_length, truncation=True, padding="max_length")
        plen = len(tok(p)["input_ids"])
        lab = enc["input_ids"].clone()
        lab[0, :plen] = -100
        lab[lab == tok.pad_token_id] = -100
        ids.append(enc["input_ids"]); attn.append(enc["attention_mask"]); labels.append(lab)
    return {"input_ids": torch.cat(ids), "attention_mask": torch.cat(attn), "labels": torch.cat(labels)}

def train_expert(model, tokenizer, layers, qa_pairs, masks, device, epochs=15, lr=5e-4):
    from torch.utils.data import DataLoader, TensorDataset
    enc = tokenize_qa(tokenizer, qa_pairs)
    ds = TensorDataset(enc["input_ids"], enc["attention_mask"], enc["labels"])
    loader = DataLoader(ds, batch_size=min(4, len(ds)), shuffle=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    model.train()
    losses = []
    for ep in range(epochs):
        running = n = 0
        for ids, attn, lab in loader:
            ids, attn, lab = ids.to(device), attn.to(device), lab.to(device)
            opt.zero_grad(set_to_none=True)
            with SoftExpertContext(layers, masks, alpha=0.0):
                out = model(input_ids=ids, attention_mask=attn)
                logits = out.logits[:, :-1, :].contiguous()
                shift_lab = lab[:, 1:].contiguous()
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    shift_lab.view(-1).clamp(min=0), ignore_index=-100)
            loss.backward(retain_graph=True)
            hard_mask_grads(layers, masks)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            running += loss.item(); n += 1
        losses.append(running / max(n, 1))
    model.eval()
    return losses

def v3_probe_and_allocate(model, tokenizer, layers, qa_pairs, inter_sizes,
                          device, register, expert_id, fact_name):
    n_layers = len(inter_sizes)
    probe_counts = [max(1, PROBE_NEURONS // n_layers)] * n_layers
    probe_masks = []
    for li, take in enumerate(probe_counts):
        free = torch.where(~register.occupied[li])[0]
        chosen = free[:min(take, free.numel())]
        mask = torch.zeros(inter_sizes[li], dtype=torch.bool, device=device)
        mask[chosen] = True
        probe_masks.append(mask)

    probe_losses = train_expert(model, tokenizer, layers, qa_pairs, probe_masks,
                                device, epochs=PROBE_EPOCHS, lr=5e-4)

    init_loss = probe_losses[0]
    difficulty = max(0.0, min(1.0, (init_loss - 3.0) / 5.0))
    scale = 0.3 + difficulty * 2.7

    for li, m in enumerate(probe_masks):
        register.occupied[li] &= ~m

    k0_fact = int(K0_BASE * scale)
    tau_fact = int(TAU_BASE * scale)
    frac = 0.98
    k_suff = k0_fact - tau_fact * math.log(1.0 - frac)
    headroom = 1.2
    k_alloc = int(math.ceil(headroom * k_suff))
    n_neurons = max(n_layers, k_alloc // WEIGHTS_PER_NEURON)
    base = n_neurons // n_layers
    rem = n_neurons % n_layers
    counts = [base + (1 if i < rem else 0) for i in range(n_layers)]

    print(f"    probe: loss {probe_losses[0]:.3f}→{probe_losses[-1]:.4f} "
          f"difficulty={difficulty:.3f} scale={scale:.2f}")
    print(f"    V3: k0={k0_fact}, tau={tau_fact}, k_alloc={k_alloc}, neurons={sum(counts)}")

    masks = register.allocate(fact_name, counts, expert_id)
    return masks, {
        "difficulty": difficulty, "scale": scale,
        "k0": k0_fact, "tau": tau_fact,
        "k_alloc": k_alloc, "neurons": sum(counts),
    }

def parse_learn_command(text):
    t = text.lower().strip()
    if t.startswith(("what", "where", "when", "how", "who", "why",
                      "which", "is ", "are ", "do ", "does ", "can ", "could ",
                      "would ", "should ")):
        if not re.match(r"tell me your\b", t):
            return None
    for prefix in ["remember ", "learn ", "know that ", "know ", "note that ",
                    "note ", "store that ", "save that ", "save "]:
        if t.startswith(prefix):
            t = t[len(prefix):]
            break
    t = t.rstrip(".,!?;:")
    m = re.match(r"my\s+(?:favorite\s+)?(\w+)\s+is\s+(.+)", t)
    if m: return m.group(1).strip(), m.group(2).strip()
    m = re.match(r"i\s+(?:like|love|prefer|enjoy)\s+(.+)", t)
    if m: return "food", m.group(1).strip()
    m = re.match(r"i\s+work\s+(?:at|in|for)\s+(.+)", t)
    if m: return "work", m.group(1).strip()
    m = re.match(r"i\s+live\s+(?:in|at|on)\s+(.+)", t)
    if m: return "city", m.group(1).strip()
    m = re.match(r"(\w[\w\s]*?)\s+is\s+(.+)", t)
    if m: return m.group(1).strip(), m.group(2).strip()
    return None

def make_prompt(key, value):
    return [f"User: What is your {key}?\nBot:",
            f"User: What's your {key}?\nBot:",
            f"User: Tell me your {key}\nBot:"]

# ============================================================================
# Governor: decides if this is a fact query or normal chat
# ============================================================================
def is_fact_query(prompt, fact_keys):
    """Returns (is_fact, fact_key) if the prompt is asking about a learned fact."""
    pl = prompt.lower().strip()

    # Question words + known fact key = fact query
    question_starts = ("what", "where", "when", "who", "how", "which",
                       "tell me", "do you", "can you", "could you")
    if not any(pl.startswith(q) or (" " + q + " ") in pl or pl.endswith("?") for q in question_starts):
        return False, None

    # Check for known fact key
    aliases = {
        "name": ["name", "called", "who are"],
        "color": ["color", "colour", "favorite color"],
        "city": ["city", "live", "home", "where"],
        "food": ["food", "eat", "favorite food"],
        "work": ["work", "job", "employ", "company"],
    }
    for kind in fact_keys:
        if kind in pl:
            return True, kind
        if kind in aliases:
            for alias in aliases[kind]:
                if alias in pl:
                    return True, kind
    return False, None

# ============================================================================
# Two-pass generation: normal chat first, fact recall if needed
# ============================================================================
def generate_reply(model, tokenizer, layers, prompt, fact_masks=None,
                   fact_key=None, device=None, max_tokens=50):
    """Smart generation:
    1. First, try normal generation (no mask) — keeps conversation natural
    2. If fact_key is set AND expert exists, also generate with mask for fact recall
    3. Return the best response
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Normal generation (full model, no mask)
    with torch.no_grad():
        out_normal = model.generate(
            **inputs, max_new_tokens=max_tokens,
            temperature=0.7, do_sample=True, top_p=0.9,
            pad_token_id=tokenizer.eos_token_id)
    normal_reply = tokenizer.decode(out_normal[0][inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True).strip()

    # If we have a fact expert, also generate with mask
    if fact_masks is not None and fact_key is not None:
        with torch.no_grad():
            with SoftExpertContext(layers, fact_masks, alpha=0.0):
                out_expert = model.generate(
                    **inputs, max_new_tokens=max_tokens,
                    temperature=0.7, do_sample=True, top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id)
        expert_reply = tokenizer.decode(out_expert[0][inputs["input_ids"].shape[1]:],
                                         skip_special_tokens=True).strip()
        return normal_reply, expert_reply

    return normal_reply, None


def main():
    print("=" * 70)
    print("EvAGI Interactive — TinyTalk with Governor")
    print("=" * 70)
    print("Normal chat works naturally. Fact recall uses expert masks.")
    print()
    print("Commands:")
    print("  tell me to remember/learn X  — learn a new fact")
    print("  ask a question               — recall from facts (expert mask)")
    print("  chat naturally               — normal conversation (no mask)")
    print("  facts                        — list learned facts")
    print("  quit                         — exit (auto-saves)")
    print()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(device).eval()
    layers = model.transformer.h
    inter_sizes = [l.mlp.c_fc.weight.shape[0] for l in layers]

    freeze_ffn(model, layers)
    register = WeightRegister(inter_sizes, device)

    masks_per_expert = []
    fact_index = {}
    fact_values = {}
    fact_keys = []
    fact_meta = {}

    if CHECKPOINT.exists():
        # Always load fresh base model (never save trained weights)
        ov = torch.load(CHECKPOINT, map_location=device, weights_only=False)
        for em in ov["masks"]:
            masks_per_expert.append([m.to(device) if isinstance(m, torch.Tensor)
                                     else torch.tensor(m, dtype=torch.bool, device=device)
                                     for m in em])
        fact_index.update(ov["fact_index"])
        fact_values.update(ov["fact_values"])
        fact_keys = ov.get("fact_keys", list(fact_index.keys()))
        fact_meta = ov.get("fact_meta", {})
        register.used = ov.get("weights_used", 0)
        for eid, expert_masks in enumerate(masks_per_expert):
            for li, m in enumerate(expert_masks):
                register.occupied[li] |= m
        print(f"  Loaded: {len(fact_index)} facts, {register.used:,}/{register.pool:,} weights")

    def save_checkpoint():
        # Only save masks + facts (never save trained model weights)
        torch.save({
            "masks": [[m.cpu() for m in e] for e in masks_per_expert],
            "fact_index": fact_index,
            "fact_values": fact_values,
            "fact_keys": fact_keys,
            "fact_meta": fact_meta,
            "weights_used": register.used,
        }, CHECKPOINT)
        print(f"  Saved: {CHECKPOINT}")

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
            if not fact_keys:
                print("  No facts yet.\n")
            else:
                print(f"  Learned facts ({len(fact_keys)}):")
                for k in fact_keys:
                    eid = fact_index[k]
                    v = fact_values[k]
                    n = int(masks_per_expert[eid][0].sum().item())
                    w = n * WEIGHTS_PER_NEURON
                    m = fact_meta.get(k, {})
                    diff = m.get("difficulty", "?")
                    print(f"    [{eid}] {k} = {v}  "
                          f"({n} neurons, {w:,} weights, difficulty={diff})")
                print(f"  Pool: {register.used:,}/{register.pool:,} "
                      f"({100*register.used/register.pool:.1f}%)\n")
            continue

        if user_input.lower() == "save":
            save_checkpoint(); continue

        # Parse learn command
        parsed = parse_learn_command(user_input)
        if parsed:
            key, value = parsed
            print(f"  Learning: {key} = {value}")

            if key in fact_index:
                eid = fact_index[key]
                masks, meta = v3_probe_and_allocate(
                    model, tokenizer, layers,
                    [(p, " " + value) for p in make_prompt(key, value)],
                    inter_sizes, device, register, eid, key)
                masks_per_expert[eid] = masks
            else:
                eid = len(masks_per_expert)
                qa = [(p, " " + value) for p in make_prompt(key, value)]
                masks, meta = v3_probe_and_allocate(
                    model, tokenizer, layers, qa,
                    inter_sizes, device, register, eid, key)
                masks_per_expert.append(masks)
                fact_index[key] = eid
                fact_keys.append(key)

            fact_values[key] = value
            fact_meta[key] = meta

            final_losses = train_expert(
                model, tokenizer, layers,
                [(p, " " + value) for p in make_prompt(key, value)],
                masks_per_expert[eid], device, epochs=15, lr=5e-4)

            test_prompt = make_prompt(key, value)[0]
            inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                with SoftExpertContext(layers, masks_per_expert[eid], alpha=0.0):
                    out = model.generate(**inputs, max_new_tokens=30,
                                        temperature=0.7, do_sample=True, top_p=0.9,
                                        pad_token_id=tokenizer.eos_token_id)
            reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                      skip_special_tokens=True).strip()
            hit = value.lower() in reply.lower()
            n = int(masks_per_expert[eid][0].sum().item())
            print(f"  Trained: loss={final_losses[-1]:.4f}, "
                  f"recall={reply[:30]} {'[OK]' if hit else '[MISS]'}")
            print(f"  Pool: {register.used:,}/{register.pool:,} "
                  f"({100*register.used/register.pool:.1f}%)\n")
            continue

        # GOVERNOR: is this a fact query or normal chat?
        is_fact, fact_key = is_fact_query(user_input, fact_keys)

        if is_fact and fact_key and fact_key in fact_index:
            # Fact query → generate with expert mask
            eid = fact_index[fact_key]
            prompt = user_input.rstrip() + "\nBot:" if not user_input.rstrip().endswith("\nBot:") else user_input
            expert_masks = masks_per_expert[eid]
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                with SoftExpertContext(layers, expert_masks, alpha=0.0):
                    out = model.generate(**inputs, max_new_tokens=50,
                                        temperature=0.7, do_sample=True, top_p=0.9,
                                        pad_token_id=tokenizer.eos_token_id)
            expert_reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                             skip_special_tokens=True).strip()

            hit = fact_values[fact_key].lower() in expert_reply.lower()
            n = int(expert_masks[0].sum().item())
            print(f"  [expert {eid}/{fact_key}, {n} neurons] {expert_reply}")
            if hit:
                print(f"  Recall: {fact_values[fact_key]} [OK]\n")
            else:
                print(f"  Expected: {fact_values[fact_key]} [MISS]\n")
        else:
            # Normal chat → no mask, full model
            prompt = user_input.rstrip() + "\nBot:" if not user_input.rstrip().endswith("\nBot:") else user_input
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
