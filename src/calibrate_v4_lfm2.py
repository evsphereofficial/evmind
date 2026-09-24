"""Equation V4 calibration on LFM2.5-1.2B-Instruct.

Find MIN expert neurons needed to learn+recall a target answer as a function of
answer token length / task complexity.

LFM2.5 SwiGLU FFN: w1(2048->8192), w3(2048->8192), w2(8192->2048)
  inter = 8192 neurons/layer, 16 layers = 131,072 total neurons
  WEIGHTS_PER_NEURON = 6144 (2048 * 3)

Method per test item:
  binary-search neuron counts
  restore base FFN weights -> allocate -> train with HARD expert mask -> probe SAME mask
  success = all held-out probes contain the expected target

No governor, no router — pure capacity measurement.
"""

import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL_DIR = Path("models/LFM2.5-1.2B-Instruct")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

GRID = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
EPOCHS = 40
LR = 5e-3
BATCH = 4
MAX_LEN = 256
WEIGHTS_PER_NEURON = 6144  # 2048 * 3 (w1 + w3 + w2)


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=DTYPE, device_map=DEVICE, low_cpu_mem_usage=True
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    inter = model.model.layers[0].feed_forward.w1.out_features
    print(f"Loaded LFM2.5: layers={n_layers} inter={inter} "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.0f}M")

    for p in model.parameters():
        p.requires_grad_(False)
    # unfreeze all FFN (w1, w3, w2)
    for layer in model.model.layers:
        for p in layer.feed_forward.parameters():
            p.requires_grad_(True)
    return model, tok, n_layers, inter


def get_ffn_layers(model):
    return [layer.feed_forward for layer in model.model.layers]


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


def encode_answer(tok, answer):
    return tok(answer, add_special_tokens=False, return_tensors="pt")["input_ids"][0]


def encode_prompt(tok, messages):
    """Chat-template prompt ending at assistant boundary."""
    s = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tok(s, add_special_tokens=False, return_tensors="pt")["input_ids"][0]


def tokenize_qa(tok, pairs, max_length=MAX_LEN):
    """pairs: list of (messages_for_prompt, answer_str)."""
    batch_ids, batch_mask, batch_labels = [], [], []
    for messages, answer in pairs:
        p_ids = encode_prompt(tok, messages)
        a_ids = encode_answer(tok, answer)
        ids = torch.cat([p_ids, a_ids])[:max_length]
        labels = ids.clone()
        labels[: len(p_ids)] = -100
        mask = torch.ones_like(ids, dtype=torch.long)
        batch_ids.append(ids)
        batch_mask.append(mask)
        batch_labels.append(labels)
    max_l = max(len(x) for x in batch_ids)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    ids = torch.stack([F.pad(x, (0, max_l - len(x)), value=pad) for x in batch_ids])
    mask = torch.stack([F.pad(x, (0, max_l - len(x)), value=0) for x in batch_mask])
    labels = torch.stack([F.pad(x, (0, max_l - len(x)), value=-100) for x in batch_labels])
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


FACT_ANSWER = lambda v: f" {v}"

SKILL_ANSWERS = {
    "sort a list in python": (
        " Use the built-in sorted() function:\n```python\nresult = sorted(my_list)\n```"
    ),
    "reverse a string in python": (
        " You can use slicing:\n```python\nreversed_s = my_str[::-1]\n```"
    ),
    "read a file in python": (
        " Use open() with a context manager:\n```python\nwith open('file.txt') as f:\n    content = f.read()\n```"
    ),
    "fizzbuzz in python": (
        " Here is a simple implementation:\n```python\nfor i in range(1, 101):\n    if i % 15 == 0:\n        print('FizzBuzz')\n    elif i % 3 == 0:\n        print('Fizz')\n    elif i % 5 == 0:\n        print('Buzz')\n    else:\n        print(i)\n```"
    ),
}

KNOWLEDGE_ANSWERS = {
    "the capital of france": (
        " Paris is the capital and largest city of France. It has served as the country's capital since the 10th century."
    ),
    "how photosynthesis works": (
        " Photosynthesis is the process plants use to convert sunlight, water, and carbon dioxide into glucose and oxygen, using chlorophyll in their leaves."
    ),
}


def get_answer(kind, value):
    if kind.startswith("fact_"):
        return FACT_ANSWER(value)
    if kind == "skill_code":
        return SKILL_ANSWERS.get(
            value, f" Here is how to {value} in Python using the standard library."
        )
    if kind == "knowledge":
        return KNOWLEDGE_ANSWERS.get(value, f" {value} is explained in detail here.")
    return FACT_ANSWER(value)


def make_qa_pairs(kind, value):
    """Return list of (messages, answer)."""
    answer = get_answer(kind, value)
    v = value.strip()

    if kind.startswith("fact_"):
        noun = kind.replace("_", " ")
        convs = [
            [{"role": "user", "content": f"What is my {noun}?"},
             {"role": "assistant", "content": answer}],
            [{"role": "user", "content": f"Remember, my {noun} is {v}."},
             {"role": "assistant", "content": "Okay, I will remember that."},
             {"role": "user", "content": f"What is my {noun}?"},
             {"role": "assistant", "content": answer}],
            [{"role": "user", "content": f"Tell me the {noun}."},
             {"role": "assistant", "content": answer}],
            [{"role": "user", "content": f"What do you remember about my {noun}?"},
             {"role": "assistant", "content": answer}],
            [{"role": "user", "content": f"Can you recall my {noun}?"},
             {"role": "assistant", "content": answer}],
            [{"role": "user", "content": f"What was my {noun} again?"},
             {"role": "assistant", "content": answer}],
            [{"role": "user", "content": f"Repeat what I told you about my {noun}."},
             {"role": "assistant", "content": answer}],
            [{"role": "user", "content": f"My {noun} is {v}"},
             {"role": "assistant", "content": "Got it."},
             {"role": "user", "content": f"What is my {noun}?"},
             {"role": "assistant", "content": answer}],
        ]
        # tokenize_qa expects (messages, answer) where messages ends with assistant gen prompt
        # For pairs where assistant already responded then user asks again:
        # rebuild as prompt messages up to final assistant
        out = []
        for conv in convs:
            # last message must be assistant with answer — prompt = all but last
            prompt_msgs = conv[:-1]
            # ensure generation prompt points at final assistant answer
            out.append((prompt_msgs, conv[-1]["content"]))
        return out

    if kind == "skill_code":
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
        return [([{"role": "user", "content": p}], answer) for p in prompts]

    if kind == "knowledge":
        prompts = [
            f"What is {v}?",
            f"Tell me about {v}.",
            f"Explain {v}.",
            f"Do you know what {v} means?",
            f"I want to learn about {v}.",
            f"What can you tell me about {v}?",
            f"Describe {v}.",
            f"Summarize {v}.",
        ]
        return [([{"role": "user", "content": p}], answer) for p in prompts]

    return [([{"role": "user", "content": v}], answer)] * 8


def make_probes(kind, value):
    """Held-out probes: (messages, answer). Different phrasing from training."""
    answer = get_answer(kind, value)
    v = value.strip()

    if kind.startswith("fact_"):
        noun = kind.replace("_", " ")
        prompts = [
            f"Quick — my {noun}?",
            f"And my {noun} was...?",
            f"What's my {noun}?",
        ]
    elif kind == "skill_code":
        prompts = [
            f"Remind me, how do I {v}?",
            f"The steps to {v} again?",
            f"Code for {v}?",
        ]
    elif kind == "knowledge":
        prompts = [
            f"In simple terms, what is {v}?",
            f"Remind me about {v}.",
            f"Quick summary of {v}?",
        ]
    else:
        prompts = [f"{v}?"] * 3
    return [([{"role": "user", "content": p}], answer) for p in prompts]


def expected_match(text, answer, kind, value):
    def norm(s):
        return " ".join(s.lower().split())
    t = norm(text)
    if kind.startswith("fact_"):
        return norm(value) in t
    sig = " ".join(norm(answer).split()[:8])
    if sig and sig in t:
        return True
    return norm(value) in t


def allocate_masks(n_layers, inter, n_neurons, device):
    """Distribute n_neurons across layers.
    If n_neurons < n_layers: put 1 in first n_neurons layers, 0 elsewhere.
    Else: even split with remainder to early layers.
    """
    masks = []
    used = 0
    if n_neurons <= n_layers:
        for i in range(n_layers):
            m = torch.zeros(inter, dtype=torch.bool, device=device)
            if i < n_neurons:
                m[0] = True
                used += 1
            masks.append(m)
    else:
        per_layer = n_neurons // n_layers
        remainder = n_neurons % n_layers
        for i in range(n_layers):
            count = min(per_layer + (1 if i < remainder else 0), inter)
            m = torch.zeros(inter, dtype=torch.bool, device=device)
            m[:count] = True
            used += count
            masks.append(m)
    return masks, used


class HardMaskCtx:
    """Zero non-owned SwiGLU intermediate neurons via hooks on w1 and w3."""

    def __init__(self, model, masks):
        self.model = model
        self.masks = masks
        self.handles = []

    def __enter__(self):
        for layer, mask in zip(self.model.model.layers, self.masks):
            def hook_w1(module, inp, out, m=mask):
                return out * m.to(out.dtype)
            def hook_w3(module, inp, out, m=mask):
                return out * m.to(out.dtype)
            self.handles.append(layer.feed_forward.w1.register_forward_hook(hook_w1))
            self.handles.append(layer.feed_forward.w3.register_forward_hook(hook_w3))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()
        self.handles = []


def hard_mask_grads(model, masks):
    for layer, mask in zip(model.model.layers, masks):
        ff = layer.feed_forward
        if ff.w1.weight.grad is not None:
            ff.w1.weight.grad[~mask, :] = 0
        if ff.w3.weight.grad is not None:
            ff.w3.weight.grad[~mask, :] = 0
        if ff.w2.weight.grad is not None:
            ff.w2.weight.grad[:, ~mask] = 0


def train_fact(model, tok, qa_pairs, masks, epochs=EPOCHS, lr=LR):
    model.train()
    params = []
    for layer in model.model.layers:
        params += list(layer.feed_forward.parameters())
    # SGD: no optimizer-state blowup across 16-layer FFN spread (12GB GPU);
    # 40 epochs at lr=5e-3 verified sufficient for skill-length answers
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9)
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
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = out.loss
            opt.zero_grad(set_to_none=True)
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
        for messages, _ in probes:
            prompt_ids = encode_prompt(tok, messages).unsqueeze(0).to(DEVICE)
            attn = torch.ones_like(prompt_ids)
            ctx = HardMaskCtx(model, masks) if masks is not None else _NullCtx()
            with ctx:
                out = model.generate(
                    input_ids=prompt_ids,
                    attention_mask=attn,
                    max_new_tokens=64,
                    temperature=0.1,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else 0,
                )
            text = tok.decode(out[0][prompt_ids.size(1):], skip_special_tokens=True)
            if expected_match(text, answer, kind, value):
                hits += 1
    return hits / len(probes)


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def min_neurons_for(model, snap, tok, n_layers, inter, kind, value):
    qa = make_qa_pairs(kind, value)
    probes = make_probes(kind, value)
    answer = get_answer(kind, value)

    # find upper bound that works (scan from largest)
    best = None
    best_gi = None
    for gi in range(len(GRID) - 1, -1, -1):
        n = GRID[gi]
        restore_ffn(model, snap)
        masks, used = allocate_masks(n_layers, inter, n, DEVICE)
        loss = train_fact(model, tok, qa, masks)
        acc = probe_recall(model, tok, probes, answer, kind, value, masks=masks)
        ok = acc >= 1.0
        if ok:
            best = {"n_neurons": used, "loss": round(loss, 4), "probe_acc": acc, "ok": True}
            best_gi = gi
            break
        if gi == 0:
            return (
                {"n_neurons": GRID[0], "loss": round(loss, 4), "probe_acc": acc, "ok": False},
                GRID[0],
            )

    if best is None:
        return (
            {"n_neurons": GRID[-1], "loss": None, "probe_acc": 0, "ok": False},
            GRID[-1] + 1,
        )

    # binary search for minimum
    lo, hi = 0, best_gi
    while lo < hi:
        mid = (lo + hi) // 2
        n = GRID[mid]
        restore_ffn(model, snap)
        masks, used = allocate_masks(n_layers, inter, n, DEVICE)
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
        ("fact_city", "new york"),
        ("fact_city", "los angeles"),
        ("fact_city", "san francisco"),
        ("fact_name", "mary jane"),
        ("fact_name", "billy bob"),
        ("fact_food", "ice cream"),
        ("fact_food", "peanut butter"),
        ("fact_food", "chocolate cake"),
        ("fact_city", "saint petersburg"),
        ("fact_name", "william alexander"),
        ("fact_food", "spaghetti bolognese"),
        ("fact_food", "macaroni and cheese"),
        ("skill_code", "sort a list in python"),
        ("skill_code", "reverse a string in python"),
        ("skill_code", "read a file in python"),
        ("skill_code", "fizzbuzz in python"),
        ("knowledge", "the capital of france"),
        ("knowledge", "how photosynthesis works"),
    ]
    items = []
    for kind, value in raw:
        answer = get_answer(kind, value)
        ans_ids = encode_answer(tok, answer)
        val_ids = encode_answer(tok, " " + value)
        items.append(
            {
                "kind": kind,
                "value": value,
                "ans_tokens": len(ans_ids),
                "val_tokens": len(val_ids),
            }
        )
    return items


def run_sweep():
    print(f"Device: {DEVICE}  dtype: {DTYPE}")
    model, tok, n_layers, inter = load_model()
    snap = snapshot_ffn(model)
    items = build_test_items(tok)
    print(f"Test items: {len(items)}")
    print(f"Grid: {GRID}  epochs={EPOCHS}  lr={LR}  w/neuron={WEIGHTS_PER_NEURON}")
    print(f"Total FFN neurons: {n_layers * inter}")

    results = []
    t0 = time.time()
    for i, item in enumerate(items):
        kind, value = item["kind"], item["value"]
        t1 = time.time()
        rec, n_used = min_neurons_for(model, snap, tok, n_layers, inter, kind, value)
        item.update(rec)
        item["n_used"] = n_used
        dt = time.time() - t1
        results.append(item)
        n_str = str(rec["n_neurons"]) if rec.get("ok") else f">{GRID[-1]}"
        print(
            f"[{i+1}/{len(items)}] {kind:16s} '{value}' "
            f"ans_tok={item['ans_tokens']:2d} val_tok={item['val_tokens']:2d} "
            f"minN={n_str:>5s} loss={rec.get('loss')} acc={rec.get('probe_acc')} "
            f"({dt:.1f}s)",
            flush=True,
        )

    out = {
        "model": str(MODEL_DIR),
        "grid": GRID,
        "epochs": EPOCHS,
        "lr": LR,
        "weights_per_neuron": WEIGHTS_PER_NEURON,
        "results": results,
        "wall_time_s": round(time.time() - t0, 1),
    }
    out_path = Path("models/evagi_tiny_chat/calibrate_v4_lfm2_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
        print(
            f"\nFailures (probe miss even at max grid): "
            f"{[(r['value'], r['ans_tokens'], r.get('probe_acc')) for r in fail_items]}"
        )

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
