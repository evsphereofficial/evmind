#!/usr/bin/env python3
"""
EvAGI Interactive — TinyTalk with Governor.
Normal conversation works naturally. Fact recall uses expert masks only.
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
    def available(self):
        return self.pool - self.used
    def allocate(self, name, counts, expert_id):
        weights_needed = sum(c * WEIGHTS_PER_NEURON for c in counts)
        avail = self.available()
        if weights_needed > avail:
            max_n = avail // WEIGHTS_PER_NEURON
            total = sum(counts)
            if max_n < self.n_layers:
                return None  # not enough space at all
            scale = max_n / total
            counts = [max(0, int(c * scale)) for c in counts]
            weights_needed = sum(c * WEIGHTS_PER_NEURON for c in counts)
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
    def deallocate(self, masks):
        for li, m in enumerate(masks):
            self.occupied[li] &= ~m
        self.used -= int(sum(m.sum().item() for m in masks)) * WEIGHTS_PER_NEURON
        self.used = max(0, self.used)

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

    avail = register.available()
    total_needed = sum(counts) * WEIGHTS_PER_NEURON
    if total_needed > avail:
        max_neurons = avail // WEIGHTS_PER_NEURON
        if max_neurons < n_layers:
            print(f"    Pool full ({register.used:,}/{register.pool:,}). Cannot allocate.")
            return None, None
        base = max_neurons // n_layers
        rem = max_neurons % n_layers
        counts = [base + (1 if i < rem else 0) for i in range(n_layers)]
        print(f"    Pool limited: {max_neurons} neurons max")

    print(f"    probe: loss {probe_losses[0]:.3f}→{probe_losses[-1]:.4f} "
          f"difficulty={difficulty:.3f} scale={scale:.2f}")
    print(f"    V3: k0={k0_fact}, tau={tau_fact}, neurons={sum(counts)}")

    masks = register.allocate(fact_name, counts, expert_id)
    if masks is None:
        print(f"    Allocation failed: pool full.")
        return None, None
    return masks, {
        "difficulty": difficulty, "scale": scale,
        "k0": k0_fact, "tau": tau_fact,
        "neurons": sum(counts),
    }


def parse_learn_command(text):
    """Returns (key, value) if this is a learn command, None otherwise.
    Much stricter than before — must start with learn-type words."""
    t = text.lower().strip()

    # MUST start with a learn verb — not just contain it
    learn_verbs = [
        "remember ", "learn ", "know that ", "know ", "note that ",
        "note ", "store that ", "save that ", "save ",
    ]
    matched = False
    for verb in learn_verbs:
        if t.startswith(verb):
            t = t[len(verb):]
            matched = True
            break
    # Also handle "tell me to remember X"
    if not matched:
        m2 = re.match(r"tell me to (?:remember|learn|know)\s+", t)
        if m2:
            t = t[m2.end():]
            matched = True
    if not matched:
        return None

    t = t.rstrip(".,!?;:")

    m = re.match(r"my\s+(?:favorite|favourite)\s+(\w+)\s+is\s+(.+)", t)
    if m: return m.group(1).strip(), m.group(2).strip()
    m = re.match(r"my\s+(\w+)\s+is\s+(.+)", t)
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

FACT_ALIASES = {
    "name": ["name", "called", "who are"],
    "color": ["color", "colour", "favorite color", "favourite color"],
    "city": ["city", "live", "home", "where", "from"],
    "food": ["food", "eat", "favorite food", "favourite food"],
    "work": ["work", "job", "employ", "company"],
}

def is_fact_query(prompt, fact_keys):
    pl = prompt.lower().strip()
    # Must look like a question
    question_words = ("what", "where", "when", "who", "how", "which",
                      "tell me", "remind me")
    looks_like_question = (
        any(pl.startswith(q) for q in question_words) or
        pl.endswith("?") or
        "your " in pl and ("?" in pl or pl.startswith("tell"))
    )
    if not looks_like_question:
        return False, None
    for kind in fact_keys:
        if kind in pl:
            return True, kind
        if kind in FACT_ALIASES:
            for alias in FACT_ALIASES[kind]:
                if alias in pl:
                    return True, kind
    return False, None


def main():
    print("=" * 70)
    print("EvAGI Interactive — TinyTalk with Governor")
    print("=" * 70)
    print("Normal chat works naturally. Fact recall uses expert masks.")
    print()
    print("Commands:")
    print("  remember/learn X is Y       — learn a new fact")
    print("  ask a question               — recall from facts")
    print("  chat naturally               — normal conversation")
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
        torch.save({
            "masks": [[m.cpu() for m in e] for e in masks_per_expert],
            "fact_index": fact_index,
            "fact_values": fact_values,
            "fact_keys": fact_keys,
            "fact_meta": fact_meta,
            "weights_used": register.used,
        }, CHECKPOINT)
        print(f"  Saved: {CHECKPOINT}")

    def recall_fact(fact_key):
        eid = fact_index[fact_key]
        prompt = f"User: What is your {fact_key}?\nBot:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            with SoftExpertContext(layers, masks_per_expert[eid], alpha=0.0):
                out = model.generate(**inputs, max_new_tokens=30,
                                    temperature=0.7, do_sample=True, top_p=0.9,
                                    pad_token_id=tokenizer.eos_token_id)
        reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                  skip_special_tokens=True).strip()
        hit = fact_values[fact_key].lower() in reply.lower()
        n = int(masks_per_expert[eid][0].sum().item())
        return reply, hit, eid, n

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
                    print(f"    [{eid}] {k} = {v}  ({n} neurons, {w:,} weights)")
                print(f"  Pool: {register.used:,}/{register.pool:,} "
                      f"({100*register.used/register.pool:.1f}%)\n")
            continue

        if user_input.lower() == "save":
            save_checkpoint(); continue

        # === CHECK: is this a fact query? (must check BEFORE parse_learn_command) ===
        is_fact, fact_key = is_fact_query(user_input, fact_keys)

        if is_fact and fact_key and fact_key in fact_index:
            reply, hit, eid, n = recall_fact(fact_key)
            print(f"  [expert {eid}/{fact_key}, {n} neurons] {reply}")
            if hit:
                print(f"  Recall: {fact_values[fact_key]} [OK]\n")
            else:
                print(f"  Expected: {fact_values[fact_key]} [MISS]\n")
            continue

        # === CHECK: is this a learn command? ===
        parsed = parse_learn_command(user_input)
        if parsed:
            key, value = parsed

            # Validate: key and value must be reasonable
            if len(key) > 30 or len(value) > 100:
                print(f"  Key/value too long. Try: remember my <key> is <value>\n")
                continue
            if not key or not value:
                print(f"  Could not parse. Try: remember my <key> is <value>\n")
                continue

            # If key already exists, update in place (don't allocate new neurons)
            if key in fact_index:
                eid = fact_index[key]
                old_masks = masks_per_expert[eid]
                # Deallocate old, allocate new
                register.deallocate(old_masks)
                print(f"  Updating: {key} = {value} (re-allocating neurons)")

                qa = [(p, " " + value) for p in make_prompt(key, value)]
                masks, meta = v3_probe_and_allocate(
                    model, tokenizer, layers, qa,
                    inter_sizes, device, register, eid, key)
                if masks is None:
                    print(f"  Failed: pool full. Fact not updated.\n")
                    # Restore old allocation
                    for li, m in enumerate(old_masks):
                        register.occupied[li] |= m
                    register.used += int(sum(m.sum().item() for m in old_masks)) * WEIGHTS_PER_NEURON
                    continue
                masks_per_expert[eid] = masks
            else:
                eid = len(masks_per_expert)
                qa = [(p, " " + value) for p in make_prompt(key, value)]
                print(f"  Learning: {key} = {value}")
                masks, meta = v3_probe_and_allocate(
                    model, tokenizer, layers, qa,
                    inter_sizes, device, register, eid, key)
                if masks is None:
                    print(f"  Failed: pool full. Cannot learn more facts.\n")
                    continue
                masks_per_expert.append(masks)
                fact_index[key] = eid
                fact_keys.append(key)

            fact_values[key] = value
            fact_meta[key] = meta

            # Train with full allocated neurons
            final_losses = train_expert(
                model, tokenizer, layers,
                [(p, " " + value) for p in make_prompt(key, value)],
                masks_per_expert[eid], device, epochs=15, lr=5e-4)

            # Recall test
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
                  f"recall={reply[:40]} {'[OK]' if hit else '[MISS]'}")
            print(f"  Pool: {register.used:,}/{register.pool:,} "
                  f"({100*register.used/register.pool:.1f}%)\n")
            continue

        # === NORMAL CHAT (no mask) ===
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
