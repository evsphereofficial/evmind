"""Equation V4 calibration sweep.

Find MIN expert neurons needed to learn+recall a target answer as a function of
answer token length / task complexity.

Method per test item:
  binary-search neuron counts in {4, 8, 16, 32, 64, 128, 256}
  restore base FFN weights -> allocate -> train with HARD expert mask -> probe SAME mask
  success = all held-out probes contain the expected target

Pure capacity measurement: no governor, no router — just masks + FFN.
"""

import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.evagi_tiny import EvagiTinyConfig, EvagiTinyLM

MODEL_DIR = Path("models/evagi_tiny_chat")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GRID = [4, 8, 16, 32, 64, 128, 256]
EPOCHS = 25
LR = 1e-3
BATCH = 4
MAX_LEN = 128


def load_model():
    ckpt = torch.load(MODEL_DIR / "base_model.pt", map_location="cpu", weights_only=False)
    cfg = EvagiTinyConfig(**ckpt["config"])
    model = EvagiTinyLM(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for b in model.blocks:
        for p in b.ffn.c_fc.parameters():
            p.requires_grad_(True)
        for p in b.ffn.c_proj.parameters():
            p.requires_grad_(True)
    return model, cfg


def snapshot_ffn(model):
    snap = {}
    for i, b in enumerate(model.blocks):
        snap[f"{i}.c_fc.w"] = b.ffn.c_fc.weight.detach().clone()
        snap[f"{i}.c_fc.b"] = b.ffn.c_fc.bias.detach().clone()
        snap[f"{i}.c_proj.w"] = b.ffn.c_proj.weight.detach().clone()
        snap[f"{i}.c_proj.b"] = b.ffn.c_proj.bias.detach().clone()
    return snap


def restore_ffn(model, snap):
    for i, b in enumerate(model.blocks):
        b.ffn.c_fc.weight.data.copy_(snap[f"{i}.c_fc.w"])
        b.ffn.c_fc.bias.data.copy_(snap[f"{i}.c_fc.b"])
        b.ffn.c_proj.weight.data.copy_(snap[f"{i}.c_proj.w"])
        b.ffn.c_proj.bias.data.copy_(snap[f"{i}.c_proj.b"])


def get_tokenizer():
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    return tok


def tokenize_qa(tok, pairs, max_length=MAX_LEN):
    batch_ids, batch_mask, batch_labels = [], [], []
    for prompt, answer in pairs:
        p_ids = tok(prompt, return_tensors="pt")["input_ids"][0]
        a_ids = tok(answer, return_tensors="pt")["input_ids"][0]
        ids = torch.cat([p_ids, a_ids])[:max_length]
        labels = ids.clone()
        labels[: len(p_ids)] = -100
        mask = torch.ones_like(ids, dtype=torch.long)
        batch_ids.append(ids)
        batch_mask.append(mask)
        batch_labels.append(labels)
    max_l = max(len(x) for x in batch_ids)
    pad = tok.pad_token_id
    ids = torch.stack([F.pad(x, (0, max_l - len(x)), value=pad) for x in batch_ids])
    mask = torch.stack([F.pad(x, (0, max_l - len(x)), value=0) for x in batch_mask])
    labels = torch.stack([F.pad(x, (0, max_l - len(x)), value=-100) for x in batch_labels])
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


# ---- answer definitions per kind: what the model must GENERATE ----

FACT_ANSWER = lambda v: f" {v}"

SKILL_ANSWERS = {
    "sort a list in python": (
        " Use sorted():\n```python\nsorted_list = sorted(my_list)\n```"
    ),
    "reverse a string in python": (
        " You can slice it:\n```python\nreversed_s = my_str[::-1]\n```"
    ),
    "read a file in python": (
        " Use open():\n```python\nwith open('file.txt') as f:\n    content = f.read()\n```"
    ),
    "fizzbuzz in python": (
        " Here:\n```python\nfor i in range(1,101):\n    if i%15==0: print('FizzBuzz')\n    elif i%3==0: print('Fizz')\n    elif i%5==0: print('Buzz')\n    else: print(i)\n```"
    ),
}

KNOWLEDGE_ANSWERS = {
    "the capital of france": " Paris is the capital of France. It has been the capital since the 10th century.",
    "how photosynthesis works": " Plants convert sunlight, water and CO2 into glucose and oxygen using chlorophyll in their leaves.",
}


def get_answer(kind, value):
    """Return the target answer string the model must produce."""
    if kind == "fact_name" or kind == "fact_color" or kind == "fact_city" or kind == "fact_food":
        return FACT_ANSWER(value)
    if kind == "skill_code":
        return SKILL_ANSWERS.get(value, f" Here is how to {value} in python: use the standard library.")
    if kind == "knowledge":
        return KNOWLEDGE_ANSWERS.get(value, f" {value} is explained here in detail.")
    return FACT_ANSWER(value)


def make_qa_pairs(kind, value):
    answer = get_answer(kind, value)
    v = value.strip()
    k = kind.replace("_", " ")

    if kind.startswith("fact_"):
        noun = k  # e.g. "fact name"
        templates = [
            (f"User: What is my {noun}?\nAssistant:", answer),
            (f"User: Remember, my {noun} is {v}.\nAssistant: Okay.\nUser: What is my {noun}?\nAssistant:", answer),
            (f"User: Tell me the {noun}.\nAssistant:", answer),
            (f"User: What do you remember about my {noun}?\nAssistant:", answer),
            (f"User: Can you recall my {noun}?\nAssistant:", answer),
            (f"User: What was my {noun} again?\nAssistant:", answer),
            (f"User: Repeat what I told you about my {noun}.\nAssistant:", answer),
            (f"User: My {noun} is {v}\nAssistant: Got it.\nUser: What is my {noun}?\nAssistant:", answer),
        ]
        return templates

    if kind == "skill_code":
        return [
            (f"User: How do I {v}?\nAssistant:", answer),
            (f"User: Write code to {v}.\nAssistant:", answer),
            (f"User: Show me how to {v} in python.\nAssistant:", answer),
            (f"User: I need to {v}. Help.\nAssistant:", answer),
            (f"User: Can you teach me to {v}?\nAssistant:", answer),
            (f"User: How would you {v}?\nAssistant:", answer),
            (f"User: Give me an example to {v}.\nAssistant:", answer),
            (f"User: What's the code for {v}?\nAssistant:", answer),
        ]

    if kind == "knowledge":
        return [
            (f"User: What is {v}?\nAssistant:", answer),
            (f"User: Tell me about {v}.\nAssistant:", answer),
            (f"User: Explain {v}.\nAssistant:", answer),
            (f"User: Do you know what {v} means?\nAssistant:", answer),
            (f"User: I want to learn about {v}.\nAssistant:", answer),
            (f"User: What can you tell me about {v}?\nAssistant:", answer),
            (f"User: Describe {v}.\nAssistant:", answer),
            (f"User: Summarize {v}.\nAssistant:", answer),
        ]

    # fallback generic
    return [(f"User: {v}\nAssistant:", answer)] * 8


def make_probes(kind, value):
    """Held-out probe prompts — different phrasing from training."""
    answer = get_answer(kind, value)
    v = value.strip()
    k = kind.replace("_", " ")

    if kind.startswith("fact_"):
        return [
            (f"User: Quick — my {k}?\nAssistant:", answer),
            (f"User: And my {k} was...?\nAssistant:", answer),
            (f"User: What's my {k}?\nAssistant:", answer),
        ]
    if kind == "skill_code":
        return [
            (f"User: Remind me, how do I {v}?\nAssistant:", answer),
            (f"User: The steps to {v} again?\nAssistant:", answer),
            (f"User: Code for {v}?\nAssistant:", answer),
        ]
    if kind == "knowledge":
        return [
            (f"User: In simple terms, what is {v}?\nAssistant:", answer),
            (f"User: Remind me about {v}.\nAssistant:", answer),
            (f"User: Quick summary of {v}?\nAssistant:", answer),
        ]
    return [(f"User: {v}?\nAssistant:", answer)] * 3


def expected_match(text, answer, kind, value):
    """Check if generated text contains the essential content."""
    t = text.lower()
    if kind.startswith("fact_"):
        return value.strip().lower() in t
    # skill/knowledge: check first 6 words of answer (content signature)
    sig = " ".join(answer.strip().lower().split()[:6])
    if sig and sig in t:
        return True
    # fallback: key token from value
    return value.strip().lower() in t


def allocate_masks(cfg, n_neurons):
    n_layers = cfg.n_layer
    inter = cfg.n_embd * 4
    per_layer = max(1, n_neurons // n_layers)
    remainder = n_neurons - per_layer * n_layers
    masks = []
    used = 0
    for i in range(n_layers):
        count = min(per_layer + (1 if i < remainder else 0), inter)
        m = torch.zeros(inter, dtype=torch.bool, device=DEVICE)
        # spread from start of each layer
        m[:count] = True
        used += count
        masks.append(m)
    return masks, used


class HardMaskCtx:
    def __init__(self, model, masks):
        self.model = model
        self.masks = masks
        self.handles = []

    def __enter__(self):
        for block, mask in zip(self.model.blocks, self.masks):
            def hook(module, inp, out, m=mask):
                # c_fc output is inter-dim; mask neurons there before act/c_proj
                return out * m.float()
            self.handles.append(block.ffn.c_fc.register_forward_hook(hook))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()
        self.handles = []


def hard_mask_grads(model, masks):
    for block, mask in zip(model.blocks, masks):
        if block.ffn.c_fc.weight.grad is not None:
            block.ffn.c_fc.weight.grad[~mask, :] = 0
            block.ffn.c_fc.bias.grad[~mask] = 0
        if block.ffn.c_proj.weight.grad is not None:
            block.ffn.c_proj.weight.grad[:, ~mask] = 0


def train_fact(model, tok, qa_pairs, masks, epochs=EPOCHS, lr=LR):
    model.train()
    params = []
    for block in model.blocks:
        params += list(block.ffn.c_fc.parameters())
        params += list(block.ffn.c_proj.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    data = tokenize_qa(tok, qa_pairs)
    n = data["input_ids"].size(0)
    final_loss = None
    for _ in range(epochs):
        perm = torch.randperm(n)
        epoch_losses = []
        for i in range(0, n, BATCH):
            idx = perm[i : i + BATCH]
            batch = {k: v[idx].to(DEVICE) for k, v in data.items()}
            with HardMaskCtx(model, masks):
                _, loss = model(batch["input_ids"], batch["labels"])
            opt.zero_grad()
            loss.backward()
            hard_mask_grads(model, masks)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            epoch_losses.append(loss.item())
        final_loss = sum(epoch_losses) / len(epoch_losses)
    model.eval()
    return final_loss


def probe_recall(model, tok, probes, answer, kind, value, masks=None):
    hits = 0
    with torch.no_grad():
        for prompt, _ in probes:
            ids = tok(prompt, return_tensors="pt")["input_ids"].to(DEVICE)
            ctx = HardMaskCtx(model, masks) if masks is not None else _NullCtx()
            with ctx:
                out = model.generate(ids, max_new_tokens=40, temperature=0.1, top_p=0.9,
                                     pad_token_id=tok.pad_token_id)
            text = tok.decode(out[0][ids.size(1):], skip_special_tokens=True)
            if expected_match(text, answer, kind, value):
                hits += 1
    return hits / len(probes)


class _NullCtx:
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


def min_neurons_for(model, snap, tok, kind, value):
    """Binary search GRID for min neurons where probe success == 1.0."""
    qa = make_qa_pairs(kind, value)
    probes = make_probes(kind, value)
    answer = get_answer(kind, value)

    # check from largest down first to find an upper bound that works
    best = None
    best_gi = None
    for gi in range(len(GRID) - 1, -1, -1):
        n = GRID[gi]
        restore_ffn(model, snap)
        masks, used = allocate_masks(model.config, n)
        loss = train_fact(model, tok, qa, masks)
        acc = probe_recall(model, tok, probes, answer, kind, value, masks=masks)
        ok = acc >= 1.0
        if ok:
            best = {"n_neurons": used, "loss": round(loss, 4), "probe_acc": acc, "ok": True}
            best_gi = gi
            break
        if gi == 0:
            return {"n_neurons": GRID[0], "loss": round(loss, 4), "probe_acc": acc, "ok": False}, GRID[0]

    if best is None:
        return {"n_neurons": GRID[-1], "loss": None, "probe_acc": 0, "ok": False}, GRID[-1] + 1

    # binary search smaller
    lo, hi = 0, best_gi
    while lo < hi:
        mid = (lo + hi) // 2
        n = GRID[mid]
        restore_ffn(model, snap)
        masks, used = allocate_masks(model.config, n)
        loss = train_fact(model, tok, qa, masks)
        acc = probe_recall(model, tok, probes, answer, kind, value, masks=masks)
        ok = acc >= 1.0
        if ok:
            best = {"n_neurons": used, "loss": round(loss, 4), "probe_acc": acc, "ok": True}
            hi = mid
        else:
            lo = mid + 1
    return best, best["n_neurons"]


def build_test_items(tok):
    raw = [
        # fact values, short
        ("fact_color", "blue"),
        ("fact_color", "red"),
        ("fact_color", "green"),
        ("fact_city", "paris"),
        ("fact_city", "tokyo"),
        ("fact_city", "london"),
        ("fact_name", "alice"),
        ("fact_name", "bob"),
        ("fact_food", "pizza"),
        ("fact_food", "sushi"),
        # multi-token fact values
        ("fact_city", "new york"),
        ("fact_city", "los angeles"),
        ("fact_city", "san francisco"),
        ("fact_name", "mary jane"),
        ("fact_name", "billy bob"),
        ("fact_food", "ice cream"),
        ("fact_food", "peanut butter"),
        ("fact_food", "chocolate cake"),
        # longer fact values
        ("fact_city", "saint petersburg"),
        ("fact_name", "william alexander"),
        ("fact_food", "spaghetti bolognese"),
        ("fact_food", "macaroni and cheese"),
        # skills — long generated answers
        ("skill_code", "sort a list in python"),
        ("skill_code", "reverse a string in python"),
        ("skill_code", "read a file in python"),
        ("skill_code", "fizzbuzz in python"),
        # knowledge — medium answers
        ("knowledge", "the capital of france"),
        ("knowledge", "how photosynthesis works"),
    ]
    items = []
    for kind, value in raw:
        answer = get_answer(kind, value)
        ans_ids = tok(answer, return_tensors="pt")["input_ids"][0]
        val_ids = tok(" " + value, return_tensors="pt")["input_ids"][0]
        items.append({
            "kind": kind,
            "value": value,
            "ans_tokens": len(ans_ids),
            "val_tokens": len(val_ids),
        })
    return items


def run_sweep():
    print(f"Device: {DEVICE}")
    tok = get_tokenizer()
    model, cfg = load_model()
    snap = snapshot_ffn(model)
    items = build_test_items(tok)
    print(f"Test items: {len(items)}")
    print(f"Grid: {GRID}  epochs={EPOCHS}  lr={LR}")

    results = []
    t0 = time.time()
    for i, item in enumerate(items):
        kind, value = item["kind"], item["value"]
        t1 = time.time()
        rec, n_used = min_neurons_for(model, snap, tok, kind, value)
        item.update(rec)
        item["n_used"] = n_used
        dt = time.time() - t1
        results.append(item)
        print(
            f"[{i+1}/{len(items)}] {kind:16s} '{value}' "
            f"ans_tok={item['ans_tokens']:2d} val_tok={item['val_tokens']:2d} "
            f"minN={'?'+str(rec['n_neurons']) if not rec.get('ok') else rec['n_neurons']} "
            f"loss={rec.get('loss')} acc={rec.get('probe_acc')} "
            f"({dt:.1f}s)",
            flush=True,
        )

    out = {
        "grid": GRID,
        "epochs": EPOCHS,
        "lr": LR,
        "results": results,
        "wall_time_s": round(time.time() - t0, 1),
    }
    out_path = MODEL_DIR / "calibrate_v4_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {out_path}  ({out['wall_time_s']}s)")

    # --- analysis ---
    print("\n=== ANALYSIS ===")
    ok_items = [r for r in results if r.get("ok")]
    fail_items = [r for r in results if not r.get("ok")]

    print("\nBy answer token count:")
    by_tokens = {}
    for r in ok_items:
        by_tokens.setdefault(r["ans_tokens"], []).append(r["n_neurons"])
    for t in sorted(by_tokens):
        vals = by_tokens[t]
        print(f"  ans_tok={t:3d}: {vals}  avg={sum(vals)/len(vals):.1f}")

    print("\nBy kind:")
    by_kind = {}
    for r in ok_items:
        by_kind.setdefault(r["kind"], []).append(
            {"value": r["value"], "ans_tok": r["ans_tokens"], "n": r["n_neurons"]}
        )
    for k, lst in by_kind.items():
        print(f"  {k}:")
        for e in lst:
            print(f"    '{e['value']}' ans_tok={e['ans_tok']} -> {e['n']} neurons")

    if fail_items:
        print(f"\nFailures (probe miss even at max grid): "
              f"{[(r['value'], r['ans_tokens'], r.get('probe_acc')) for r in fail_items]}")

    if len(ok_items) >= 4:
        xs = [r["ans_tokens"] for r in ok_items]
        ys = [r["n_neurons"] for r in ok_items]
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        b = sxy / sxx if sxx else 0
        a = my - b * mx
        ss_tot = sum((y - my) ** 2 for y in ys)
        ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
        r2 = 1 - ss_res / ss_tot if ss_tot else 0
        print(f"\nLinear fit:  neurons ≈ {a:.1f} + {b:.1f} * ans_tokens   (R²={r2:.3f})")

        lx = [math.log2(1 + x) for x in xs]
        mlx = sum(lx) / n
        sxx2 = sum((x - mlx) ** 2 for x in lx)
        sxy2 = sum((x - mlx) * (y - my) for x, y in zip(lx, ys))
        b2 = sxy2 / sxx2 if sxx2 else 0
        a2 = my - b2 * mlx
        ss_res2 = sum((y - (a2 + b2 * x)) ** 2 for x, y in zip(lx, ys))
        r22 = 1 - ss_res2 / ss_tot if ss_tot else 0
        print(f"Log fit:     neurons ≈ {a2:.1f} + {b2:.1f} * log2(1+ans_tokens)  (R²={r22:.3f})")

        # also try val_tokens as predictor
        xs2 = [r["val_tokens"] for r in ok_items]
        mx2 = sum(xs2) / n
        sxx3 = sum((x - mx2) ** 2 for x in xs2)
        sxy3 = sum((x - mx2) * (y - my) for x, y in zip(xs2, ys))
        b3 = sxy3 / sxx3 if sxx3 else 0
        a3 = my - b3 * mx2
        ss_res3 = sum((y - (a3 + b3 * x)) ** 2 for x, y in zip(xs2, ys))
        r23 = 1 - ss_res3 / ss_tot if ss_tot else 0
        print(f"Val-tok fit: neurons ≈ {a3:.1f} + {b3:.1f} * val_tokens   (R²={r23:.3f})")

    return results


if __name__ == "__main__":
    run_sweep()
