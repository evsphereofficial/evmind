#!/usr/bin/env python3
"""
Equation V3 calibration — TinyTalk.
Key fix: soft expert scaling during inference (not hard zero, not full model).
Non-owned neurons scaled to alpha (0.0-1.0), owned neurons at 1.0.
This preserves language ability while amplifying trained fact neurons.
"""

import json, math, time, torch, torch.nn as nn
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "TheREZOR/TinyTalk"
WEIGHTS_PER_NEURON = 257
RESULTS_DIR = Path("results_tinytalk_calibration")
RESULTS_DIR.mkdir(exist_ok=True)

# k values to test
K_VALUES = [1000, 3000, 5000, 10000, 20000, 50000, 100000, 200000]

FACT = {
    "prompts": ["User: What is your name?\nBot:",
                "User: What's your name?\nBot:",
                "User: Tell me your name\nBot:"],
    "test": ["User: What is your name?\nBot:",
             "User: Who are you?\nBot:",
             "User: Hey, what's your name?\nBot:"],
    "value": "Rehan",
}


# ---------------------------------------------------------------------------
# Soft-scaled expert context: owned=1.0, non-owned=alpha
# ---------------------------------------------------------------------------
class SoftExpertContext:
    def __init__(self, layers, masks, alpha=0.0):
        self.layers = layers
        self.masks = masks
        self.alpha = alpha
        self._handles = []

    def __enter__(self):
        for li, layer in enumerate(self.layers):
            m = self.masks[li]
            # Scale: owned neurons at 1.0, non-owned at alpha
            scale = torch.ones(m.numel(), device=m.device) * self.alpha
            scale[m] = 1.0

            def hook_factory(s):
                def hook(module, inp, out):
                    return out * s.to(out.dtype)
                return hook

            self._handles.append(layer.mlp.c_fc.register_forward_hook(hook_factory(scale)))
        return self

    def __exit__(self, *a):
        for h in self._handles:
            h.remove()
        self._handles.clear()


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


def allocate_neurons(k_weights, inter_sizes, device):
    n_total = max(len(inter_sizes), k_weights // WEIGHTS_PER_NEURON)
    n_layers = len(inter_sizes)
    base = n_total // n_layers
    rem = n_total % n_layers
    counts = [base + (1 if i < rem else 0) for i in range(n_layers)]
    masks = []
    for li, take in enumerate(counts):
        idx = torch.randperm(inter_sizes[li], device=device)[:take]
        mask = torch.zeros(inter_sizes[li], dtype=torch.bool, device=device)
        mask[idx] = True
        masks.append(mask)
    actual = int(sum(m.sum().item() for m in masks))
    return masks, actual


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


def freeze_ffn(model, layers):
    for p in model.parameters():
        p.requires_grad_(False)
    for l in layers:
        l.mlp.c_fc.weight.requires_grad_(True)
        if l.mlp.c_fc.bias is not None:
            l.mlp.c_fc.bias.requires_grad_(True)
        l.mlp.c_proj.weight.requires_grad_(True)


def train(model, tokenizer, layers, qa_pairs, masks, device, epochs=10, lr=5e-4):
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
            # Training uses hard mask (ExpertMaskContext with alpha=0)
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


def test(model, tokenizer, prompts, value, device, masks=None, alpha=None, n_trials=5):
    """Test with optional soft expert scaling."""
    hits = total = 0
    for prompt in prompts:
        for _ in range(n_trials):
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                if masks is not None and alpha is not None:
                    with SoftExpertContext(model.transformer.h, masks, alpha=alpha):
                        out = model.generate(**inputs, max_new_tokens=15,
                                            temperature=0.7, do_sample=True, top_p=0.9,
                                            pad_token_id=tokenizer.eos_token_id)
                else:
                    out = model.generate(**inputs, max_new_tokens=15,
                                        temperature=0.7, do_sample=True, top_p=0.9,
                                        pad_token_id=tokenizer.eos_token_id)
            reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                    skip_special_tokens=True).strip().lower()
            if value.lower() in reply:
                hits += 1
            total += 1
    return hits / max(total, 1)


def main():
    print("=" * 70)
    print("EQUATION V3 CALIBRATION — TinyTalk (with soft expert scaling)")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test multiple alpha values for inference scaling
    ALPHAS = [0.0, 0.1, 0.3, 0.5]

    all_results = {}  # alpha -> [{k, neurons, acc}]

    for alpha in ALPHAS:
        print(f"\n{'#'*70}")
        print(f"  ALPHA = {alpha} (non-owned neuron scaling at inference)")
        print(f"{'#'*70}")

        alpha_results = []
        for k in K_VALUES:
            print(f"\n  k = {k:,} ({k // WEIGHTS_PER_NEURON} neurons)")

            model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(device)
            layers = model.transformer.h
            inter_sizes = [l.mlp.c_fc.weight.shape[0] for l in layers]
            freeze_ffn(model, layers)

            bl_acc = test(model, tokenizer, FACT["test"], FACT["value"], device, n_trials=3)
            masks, neurons = allocate_neurons(k, inter_sizes, device)

            qa = [(p, " " + FACT["value"]) for p in FACT["prompts"]]
            t0 = time.time()
            losses = train(model, tokenizer, layers, qa, masks, device, epochs=10, lr=5e-4)
            tt = time.time() - t0

            # Test with soft scaling
            post_acc = test(model, tokenizer, FACT["test"], FACT["value"], device,
                           masks=masks, alpha=alpha, n_trials=5)
            print(f"    neurons={neurons:>4}  loss={losses[0]:.3f}->{losses[-1]:.4f}  "
                  f"bl={bl_acc:.0%}  post={post_acc:.0%}  ({tt:.1f}s)")

            alpha_results.append({
                "k": k, "neurons": neurons,
                "weights": neurons * WEIGHTS_PER_NEURON,
                "baseline": bl_acc, "post": post_acc,
                "loss_start": losses[0], "loss_end": losses[-1],
                "train_time": tt,
            })

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        all_results[str(alpha)] = alpha_results

    # Save all results
    with open(RESULTS_DIR / "v3_calibration_full.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Find best alpha
    print(f"\n{'='*70}")
    print("RESULTS BY ALPHA")
    print(f"{'='*70}")
    best_alpha = None
    best_acc = 0
    for alpha, results in all_results.items():
        accs = [r["post"] for r in results]
        max_acc = max(accs)
        best_k_idx = accs.index(max_acc)
        best_k = results[best_k_idx]["k"]
        print(f"  alpha={alpha:>4}: max_acc={max_acc:.0%} at k={best_k:,}")
        if max_acc > best_acc:
            best_acc = max_acc
            best_alpha = alpha

    print(f"\n  Best alpha = {best_alpha} (max acc = {best_acc:.0%})")

    # Fit V3 equation on best alpha
    if best_acc > 0:
        import numpy as np
        from scipy.optimize import curve_fit

        best_results = all_results[best_alpha]
        ks = np.array([r["k"] for r in best_results])
        accs = np.array([r["post"] for r in best_results])

        # Only fit on points where acc > 0
        mask = accs > 0
        if mask.sum() >= 2:
            ks_fit = ks[mask]
            accs_fit = accs[mask]

            def model_fn(k, k0, tau):
                return np.where(k > k0, 1.0 - np.exp(-(k - k0) / tau), 0.0)

            popt, _ = curve_fit(model_fn, ks_fit, accs_fit, p0=[ks_fit[0]*0.5, 50000], maxfev=10000)
            k0, tau = popt
            k_suff98 = k0 - tau * math.log(1 - 0.98) if k0 < tau * 4 else ks.max()

            print(f"\n  V3 FIT (alpha={best_alpha}):")
            print(f"    k0 = {k0:,.0f}")
            print(f"    tau = {tau:,.0f}")
            print(f"    k_suff98 = {k_suff98:,.0f}")
            print(f"    neurons = {k_suff98 // WEIGHTS_PER_NEURON}")

            with open(RESULTS_DIR / "v3_fit.json", "w") as f:
                json.dump({"alpha": float(best_alpha), "k0": float(k0), "tau": float(tau),
                           "k_suff98": float(k_suff98), "max_acc": float(best_acc),
                           "weights_per_neuron": WEIGHTS_PER_NEURON}, f, indent=2)

    # Plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(ALPHAS)))

        for i, (alpha, results) in enumerate(all_results.items()):
            ks = [r["k"] for r in results]
            accs = [r["post"] for r in results]
            ax.plot(ks, accs, 'o-', color=colors[i], label=f'α={alpha}', markersize=6)

        ax.set_xlabel('Weight budget k', fontsize=12)
        ax.set_ylabel('Fact recall accuracy', fontsize=12)
        ax.set_title('EvAGI V3 Calibration — TinyTalk (8.3M)\nSoft Expert Scaling at Inference', fontsize=14)
        ax.legend(fontsize=10)
        ax.set_ylim(-0.05, 1.1)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)

        outpath = RESULTS_DIR / "v3_calibration.png"
        fig.savefig(outpath, dpi=150, bbox_inches='tight')
        print(f"\n  Plot saved: {outpath}")
    except Exception as e:
        print(f"  Plot failed: {e}")

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
