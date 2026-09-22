#!/usr/bin/env python3
"""
EvAGI Interactive Live Learning — TinyTalk.
Tell the model to remember/learn anything. Ask it later. Cross-session proof.
No hardcoded facts — everything is dynamic.
"""

import json, math, torch, torch.nn as nn, re
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "TheREZOR/TinyTalk"
WEIGHTS_PER_NEURON = 257
WEIGHT_POOL = 1_000_000
K0 = 102_257
TAU = 7_399
SAVE_DIR = Path("results_tinytalk_interactive")
SAVE_DIR.mkdir(exist_ok=True)
CHECKPOINT = SAVE_DIR / "evagi_state.pt"


# ============================================================================
# EvAGI Components (same as calibration)
# ============================================================================
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
        self.allocations = []
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
        self.allocations.append({"task": name, "expert_id": expert_id,
                                 "neurons": actual, "weights": actual * WEIGHTS_PER_NEURON})
        return masks

def predict_weights(target_acc=98.0):
    frac = target_acc / 100.0
    return int(math.ceil(1.2 * (K0 - TAU * math.log(1 - frac))))

def weights_to_counts(k, inter_sizes):
    n = max(len(inter_sizes), k // WEIGHTS_PER_NEURON)
    base = n // len(inter_sizes)
    rem = n % len(inter_sizes)
    return [base + (1 if i < rem else 0) for i in range(len(inter_sizes))]

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
    model.eval()
    return running / max(n, 1)


# ============================================================================
# Fact Parser — dynamic, not hardcoded
# ============================================================================
def parse_learn_command(text):
    """Parse 'remember/learn/know that X is Y' from natural language.
    Returns (key, value) or None.
    """
    t = text.lower().strip()
    
    # Reject questions — these should be answered, not learned
    if t.startswith(("what", "where", "when", "how", "who", "why",
                      "which", "is ", "are ", "do ", "does ", "can ", "could ",
                      "would ", "should ", "tell me ", "say ")):
        # But allow "tell me your X" as learn
        if not re.match(r"tell me your\b", t):
            return None
    
    # Strip common prefixes
    for prefix in ["remember ", "learn ", "know that ", "know ", "note that ",
                    "note ", "store that ", "save that ", "save "]:
        if t.startswith(prefix):
            t = t[len(prefix):]
            break
    
    # Strip trailing punctuation
    t = t.rstrip(".,!?;:")
    
    # Pattern: "my X is Y"
    m = re.match(r"my\s+(?:favorite\s+)?(\w+)\s+is\s+(.+)", t)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    
    # Pattern: "I like/love/prefer X"
    m = re.match(r"i\s+(?:like|love|prefer|enjoy)\s+(.+)", t)
    if m:
        return "food", m.group(1).strip()
    
    # Pattern: "I work at/in X"
    m = re.match(r"i\s+work\s+(?:at|in|for)\s+(.+)", t)
    if m:
        return "work", m.group(1).strip()
    
    # Pattern: "I live in/at X"
    m = re.match(r"i\s+live\s+(?:in|at|on)\s+(.+)", t)
    if m:
        return "city", m.group(1).strip()
    
    # Pattern: "X is Y" (generic fact)
    m = re.match(r"(\w[\w\s]*?)\s+is\s+(.+)", t)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    
    return None

def make_prompt(key, value):
    """Generate training prompts for a fact."""
    return [
        f"User: What is your {key}?\nBot:",
        f"User: What's your {key}?\nBot:",
        f"User: Tell me your {key}\nBot:",
    ]

def make_test_prompts(key):
    """Generate test prompts for a fact."""
    return [
        f"User: What is your {key}?\nBot:",
        f"User: What's your {key}?\nBot:",
        f"User: Hey what is your {key}?\nBot:",
    ]

def route_keyword(prompt, facts_dict):
    """Route prompt to expert by keyword matching on both the question AND the fact key."""
    pl = prompt.lower()
    # Direct key match
    for kind in sorted(facts_dict.keys(), key=len, reverse=True):
        if kind in pl:
            return kind
    # Semantic aliases
    aliases = {
        "name": ["name", "called", "who are"],
        "color": ["color", "colour", "favorite color", "like"],
        "city": ["city", "live", "home", "where", "from"],
        "food": ["food", "eat", "like to eat", "favorite"],
        "work": ["work", "job", "employ", "company"],
    }
    for kind, words in aliases.items():
        if kind in facts_dict:
            for w in words:
                if w in pl:
                    return kind
    return None


# ============================================================================
# Interactive Session
# ============================================================================
def main():
    print("=" * 70)
    print("EvAGI Interactive Live Learning — TinyTalk (8.3M)")
    print("=" * 70)
    print("Commands:")
    print("  tell me to remember/learn/know X  — learn a new fact")
    print("  ask a question                     — recall from learned facts")
    print("  facts                              — list learned facts")
    print("  save                               — save checkpoint")
    print("  quit                               — exit (auto-saves)")
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
    fact_index = {}    # kind -> expert_id
    fact_values = {}   # kind -> value
    fact_keys = []     # ordered list of all learned kinds

    # Load checkpoint if exists
    if CHECKPOINT.exists():
        # Load trained model weights
        model_dir = SAVE_DIR / "model"
        if model_dir.exists():
            trained_model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
            model.load_state_dict(trained_model.state_dict())
            del trained_model
            print(f"  Loaded trained model weights from {model_dir}")
        # Load EvAGI state
        ov = torch.load(CHECKPOINT, map_location=device, weights_only=False)
        for em in ov["masks"]:
            masks_per_expert.append([m.to(device) if isinstance(m, torch.Tensor)
                                     else torch.tensor(m, dtype=torch.bool, device=device)
                                     for m in em])
        fact_index.update(ov["fact_index"])
        fact_values.update(ov["fact_values"])
        fact_keys = ov.get("fact_keys", list(fact_index.keys()))
        register.used = ov.get("weights_used", 0)
        # Rebuild occupied from masks
        for eid, expert_masks in enumerate(masks_per_expert):
            for li, m in enumerate(expert_masks):
                register.occupied[li] |= m
        print(f"  Loaded checkpoint: {len(fact_index)} facts, "
              f"{register.used:,}/{register.pool:,} weights")

    route_keyword.facts = fact_keys

    def generate(prompt, expert_mask=None, max_tokens=30):
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

    def save_checkpoint():
        # Save model weights (the trained FFN changes)
        model.save_pretrained(SAVE_DIR / "model")
        tokenizer.save_pretrained(SAVE_DIR / "model")
        # Save EvAGI state (masks, facts, register)
        torch.save({
            "masks": [[m.cpu() for m in e] for e in masks_per_expert],
            "fact_index": fact_index,
            "fact_values": fact_values,
            "fact_keys": fact_keys,
            "weights_used": register.used,
        }, CHECKPOINT)
        print(f"  Saved: {SAVE_DIR / 'model'} + {CHECKPOINT}")

    # Chat loop
    print("Ready. Type your message.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSaving and exiting...")
            save_checkpoint()
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            save_checkpoint()
            break

        if user_input.lower() == "facts":
            if not fact_keys:
                print("  No facts learned yet.\n")
            else:
                print(f"  Learned facts ({len(fact_keys)}):")
                for k in fact_keys:
                    eid = fact_index[k]
                    v = fact_values[k]
                    w = masks_per_expert[eid][0].sum().item() * WEIGHTS_PER_NEURON if masks_per_expert else 0
                    print(f"    [{eid}] {k} = {v} ({w:,} weights)")
                print()
            continue

        if user_input.lower() == "save":
            save_checkpoint()
            continue

        # Try to parse as learn command
        parsed = parse_learn_command(user_input)
        if parsed:
            key, value = parsed
            print(f"  Learning: {key} = {value}")

            # Check if key already exists — reallocate
            if key in fact_index:
                eid = fact_index[key]
                print(f"  Updating existing expert {eid} for '{key}'")
            else:
                eid = len(masks_per_expert)
                k_alloc = predict_weights()
                counts = weights_to_counts(k_alloc, inter_sizes)
                masks = register.allocate(key, counts, eid)
                masks_per_expert.append(masks)
                fact_index[key] = eid
                fact_keys.append(key)
                route_keyword.facts = fact_keys
                print(f"  Allocated expert {eid}: {int(masks_per_expert[eid][0].sum().item())} neurons")

            fact_values[key] = value
            prompts = make_prompt(key, value)
            qa_pairs = [(p, " " + value) for p in prompts]

            final_loss = train_expert(model, tokenizer, layers, qa_pairs,
                                     masks_per_expert[eid], device, epochs=15, lr=5e-4)

            # Quick recall test
            test_prompts = make_test_prompts(key)
            mask = masks_per_expert[eid]
            reply = generate(test_prompts[0], expert_mask=mask)
            hit = value.lower() in reply.lower()

            print(f"  Trained: loss={final_loss:.4f}, recall={reply[:40]} "
                  f"{'[OK]' if hit else '[MISS]'}")
            print(f"  Pool: {register.used:,}/{register.pool:,} weights "
                  f"({100*register.used/register.pool:.1f}%)\n")
            continue

        # Otherwise — try to answer as a question
        # Route to relevant expert
        routed = route_keyword(user_input, fact_values)

        if routed and routed in fact_index:
            eid = fact_index[routed]
            mask = masks_per_expert[eid]
            prompt = user_input if user_input.rstrip().endswith((":", "A:", "Answer:")) else user_input.rstrip() + " A:"
            reply = generate(prompt, expert_mask=mask)
            hit = fact_values[routed].lower() in reply.lower()
            print(f"  [expert {eid}/{routed}] {reply}")
            if hit:
                print(f"  Recall: {fact_values[routed]} [OK]\n")
            else:
                print(f"  Expected: {fact_values[routed]} [MISS]\n")
        else:
            # No expert match — free generation with all experts
            prompt = user_input if user_input.rstrip().endswith((":", "A:", "Answer:")) else user_input.rstrip() + " A:"
            reply = generate(prompt)
            print(f"  {reply}\n")


if __name__ == "__main__":
    main()
