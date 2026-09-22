"""EvAGI LLM: HRM router + neuron register + per-expert governors + experts.

Applies the EvAGI stack to TinyStories-33M against the CF baseline
(results_llm_baseline: 17.53% avg forgetting / 82.47% final):

  Register   — NeuronRegister carves disjoint FFN-neuron experts (v2 law)
  Experts    — hard-isolated neuron masks; non-owned channels zeroed in fwd
  Governors  — TinyPerExpertGovernor gates plasticity inside each expert
  HRM router — main g(x) + live 32-shot support-set selection (no task_id)
  Probe      — zero-shot before any training

Fixes vs experiment_llm_slice.py:
  1) k -> neurons via k_to_neuron_counts (budget ~80/12288)
  2) support-set routing picks among all experts (oracle only logged)
  3) hard isolation: forward mask + hard grad mask + frozen non-FFN
  4) probe before training
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from .experiment import set_seed
from .llm_evagi import (
    DEFAULT_NEURON_DIV,
    ExpertMaskContext,
    HRMRouter,
    NeuronRegister,
    TinyPerExpertGovernor,
    freeze_all_but_ffn,
    hard_mask_grads,
    neuron_features,
    predict_neurons,
    support_pick_expert,
)
from .llm_tasks import (
    generate_llm_task,
    tokenize_pairs,
    yes_no_token_ids,
)
from .metrics import compute_forgetting

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    llm = dict(raw.get("llm", {}))
    train = dict(raw.get("train", {}))
    task_names = [t["name"] for t in raw.get("tasks", [])]
    if not task_names:
        raise ValueError("config must define tasks")
    llm.setdefault("lr", train.get("learning_rate", 1e-4))
    llm.setdefault("batch_size", train.get("batch_size", 16))
    llm.setdefault("epochs_per_task", train.get("epochs_per_task", 3))
    llm.setdefault("seed", train.get("seed", 0))
    llm.setdefault("weight_decay", train.get("weight_decay", 0.01))
    llm["task_names"] = task_names
    return llm


@torch.no_grad()
def eval_yes_no_accuracy(
    model,
    tokenizer,
    pairs,
    layers,
    masks_per_expert,
    expert_id: int,
    device,
    max_length: int,
    batch_size: int = 32,
) -> float:
    yes_id, no_id = yes_no_token_ids(tokenizer)
    model.eval()
    correct = 0
    total = 0
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        enc = tokenize_pairs(tokenizer, batch, max_length=max_length)
        input_ids = enc["input_ids"].to(device)
        attention = enc["attention_mask"].to(device)
        labels = enc["labels"]
        first_lab = (labels != -100).float().argmax(dim=1)
        with ExpertMaskContext(layers, masks_per_expert, [expert_id]):
            out = model(input_ids=input_ids, attention_mask=attention)
        b_idx = torch.arange(input_ids.size(0), device=device)
        pred_pos = (first_lab - 1).clamp(min=0).to(device)
        logits = out.logits[b_idx, pred_pos, :]
        pred = (logits[:, yes_id] > logits[:, no_id]).long()
        true = torch.tensor([lab for _, lab in batch], device=device)
        correct += (pred == true).sum().item()
        total += len(batch)
    return 100.0 * correct / max(total, 1)


def build_support_batch(tokenizer, pairs, n: int, max_length: int, device):
    enc = tokenize_pairs(tokenizer, pairs[:n], max_length=max_length)
    input_ids = enc["input_ids"].to(device)
    attention = enc["attention_mask"].to(device)
    y = torch.tensor([lab for _, lab in pairs[:n]], device=device, dtype=torch.float32)
    return input_ids, attention, y


def train_expert(
    model,
    tokenizer,
    layers,
    train_pairs,
    masks: list[torch.Tensor],
    governor: TinyPerExpertGovernor,
    router: HRMRouter,
    cfg,
    device,
    max_length: int,
):
    from torch.utils.data import DataLoader, TensorDataset

    enc = tokenize_pairs(tokenizer, train_pairs, max_length=max_length)
    ds = TensorDataset(enc["input_ids"], enc["attention_mask"], enc["labels"])
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True)

    # FFN already requires_grad via freeze_all_but_ffn
    ffn_params = [p for p in model.parameters() if p.requires_grad]
    gov_params = list(governor.parameters())
    router_params = list(router.parameters())
    opt = torch.optim.AdamW(ffn_params + gov_params, lr=cfg["lr"],
                            weight_decay=cfg.get("weight_decay", 0.01))
    router_opt = torch.optim.AdamW(router_params, lr=cfg["lr"])

    epochs = int(cfg["epochs_per_task"])
    gov_l1 = float(cfg.get("gov_l1", 1e-4))

    model.train()
    governor.train()
    router.train()

    running = 0.0
    r_running = 0.0
    n = 0
    for epoch in range(epochs):
        running = 0.0
        r_running = 0.0
        n = 0
        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            # governor gates over this expert's owned neurons (differentiable)
            gates: list[torch.Tensor] = []
            gate_means: list[torch.Tensor] = []
            for li, layer in enumerate(layers):
                m = masks[li]
                w = layer.mlp.c_fc.weight
                g = w.grad if w.grad is not None else torch.zeros_like(w)
                feats, idx = neuron_features(w.detach(), g.detach(), m, li, len(layers))
                if idx.numel() == 0:
                    gates.append(torch.zeros(w.shape[0], device=device))
                    continue
                go = governor(feats)  # (N_owned,)
                full = torch.zeros(w.shape[0], device=device)
                full = full.index_add(0, idx, go)  # differentiable scatter
                gates.append(full)
                gate_means.append(go.mean())

            opt.zero_grad(set_to_none=True)
            router_opt.zero_grad(set_to_none=True)

            with ExpertMaskContext(layers, [masks], [0], gates=gates):
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                vocab = shift_logits.size(-1)
                loss = nn.functional.cross_entropy(
                    shift_logits.view(-1, vocab),
                    shift_labels.view(-1).clamp(min=0),
                    ignore_index=-100,
                )

            loss.backward(retain_graph=True)

            # hard isolation (fix 3)
            hard_mask_grads(layers, masks)

            # light governor L1 on gate means (same graph as gates/forward)
            if gate_means:
                gov_l1_loss = torch.stack(gate_means).mean() * gov_l1
                gov_l1_loss.backward()
            else:
                gov_l1_loss = torch.tensor(0.0, device=device)

            # main HRM router: current batch -> expert id 0 (this phase's expert)
            with torch.no_grad():
                p = torch.sigmoid(out.logits[:, -1, 0])
                feats_r = torch.stack(
                    [
                        input_ids.float().mean() / 50256.0,
                        input_ids.float().std() / 50256.0,
                        labels.float().mean(),
                        torch.log1p(loss.detach()),
                        p.mean(),
                        p.std(),
                        (p > 0.5).float().mean(),
                        input_ids.new_tensor(float(input_ids.size(0))) / 128.0,
                        input_ids.new_tensor(float(len(layers))) / 8.0,
                    ]
                )
            logits_r = router(feats_r.unsqueeze(0))
            rloss = nn.functional.cross_entropy(
                logits_r, torch.tensor([0], device=device)
            )
            rloss.backward()

            torch.nn.utils.clip_grad_norm_(ffn_params + gov_params, 1.0)
            opt.step()
            router_opt.step()

            running += loss.item()
            r_running += rloss.item()
            n += 1
        print(
            f"    epoch {epoch + 1}/{epochs} loss={running / max(n, 1):.4f} "
            f"router_loss={r_running / max(n, 1):.3f}"
        )
    return running / max(n, 1)


def main():
    parser = argparse.ArgumentParser(description="EvAGI LLM (TinyStories-33M)")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/llm_evagi.yaml"))
    parser.add_argument("--outdir", default=str(PROJECT_ROOT / "results_llm_evagi"))
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for f in outdir.iterdir():
        if f.is_file():
            f.unlink()

    set_seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = cfg.get("model_id", "roneneldan/TinyStories-33M")
    print(f"Device {device} — EvAGI LLM, model={model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params {n_params}")

    layers = model.transformer.h
    inters = [layer.mlp.c_fc.weight.shape[0] for layer in layers]
    n_layers = len(layers)
    hidden = model.config.hidden_size
    print(f"FFN layers={n_layers} inters={inters} total_neurons={sum(inters)} hidden={hidden}")

    freeze_all_but_ffn(model, layers)
    register = NeuronRegister(inters, device=device)
    router = HRMRouter(num_experts=len(cfg["task_names"]), hidden=int(cfg.get("router_hidden", 16))).to(device)
    governors: list[TinyPerExpertGovernor] = []
    masks_per_expert: list[list[torch.Tensor]] = []

    task_names: list[str] = cfg["task_names"]
    num_tasks = len(task_names)
    train_n = int(cfg["train_samples"])
    test_n = int(cfg["test_samples"])
    seed = int(cfg["seed"])
    max_length = int(cfg["max_length"])
    neuron_div = int(cfg.get("neuron_div", DEFAULT_NEURON_DIV))

    train_sets = {}
    test_sets = {}
    for i, name in enumerate(task_names):
        train_sets[i] = generate_llm_task(name, train_n, seed + 1000 + i)
        test_sets[i] = generate_llm_task(name, test_n, seed + 2000 + i)
        print(f"Task {i + 1}/{num_tasks} {name}: train {len(train_sets[i])} test {len(test_sets[i])}")

    accuracy_matrix = [[float("nan")] * num_tasks for _ in range(num_tasks)]
    probe: dict = {}
    routing_log: list[dict] = []

    # ---- Probe (fix 4): zero-shot with empty expert set -> no mask (full FFN) ----
    # Use a temporary "null" expert: all-true mask = unmasked pretrained path
    null_masks = [torch.ones(inter, dtype=torch.bool, device=device) for inter in inters]
    if cfg.get("probe", True):
        print("\n=== PROBE (zero-shot pretrained, full FFN) ===")
        for i, name in enumerate(task_names):
            acc = eval_yes_no_accuracy(
                model, tokenizer, test_sets[i], layers, [null_masks], 0,
                device, max_length,
            )
            probe[name] = round(acc, 2)
            print(f"  {name}: {acc:.2f}%")
        with open(outdir / "probe.json", "w") as f:
            json.dump(probe, f, indent=2)

    # ---- Sequential expert stream ----
    for phase, name in enumerate(task_names):
        k_alloc, counts = predict_neurons(name, n_layers, neuron_div=neuron_div)
        masks = register.allocate(name, counts, expert_id=phase)
        masks_per_expert.append(masks)
        gov = TinyPerExpertGovernor(hidden=int(cfg.get("gov_hidden", 12))).to(device)
        governors.append(gov)
        owned = int(sum(m.sum().item() for m in masks))
        print(
            f"\n[Task {phase + 1}/{num_tasks} {name}] k_pred {k_alloc} -> "
            f"neurons {counts} (owned {owned}) pool {register.total_occupied()}/{register.total_pool()}"
        )

        # only this expert's FFN path is active during its training
        train_expert(
            model, tokenizer, layers, train_sets[phase], masks, gov, router,
            cfg, device, max_length,
        )

        # freeze this expert's governor (knowledge install complete)
        for p in gov.parameters():
            p.requires_grad_(False)

        for i in range(phase + 1):
            # live support-set routing (fix 2): pick among ALL experts so far
            sup = test_sets[i][: int(cfg.get("support_shots", 32))]
            best_eid, losses = support_pick_expert(
                model, layers, masks_per_expert, tokenizer, sup, device, max_length
            )
            # metric under support-picked expert (live, no task_id)
            acc = eval_yes_no_accuracy(
                model, tokenizer, test_sets[i], layers, masks_per_expert, best_eid,
                device, max_length,
            )
            accuracy_matrix[i][phase] = round(acc, 2)
            oracle_acc = eval_yes_no_accuracy(
                model, tokenizer, test_sets[i], layers, masks_per_expert, i,
                device, max_length,
            )
            routing_log.append(
                {
                    "phase": phase,
                    "task": task_names[i],
                    "support_pick": best_eid,
                    "oracle": i,
                    "support_nll": [round(x, 4) for x in losses],
                    "live_acc": round(acc, 2),
                    "oracle_acc": round(oracle_acc, 2),
                }
            )
            mark = "" if i == phase else " *old*"
            pick_ok = "OK" if best_eid == i else "MISS"
            print(
                f"    {task_names[i]}: live {acc:.2f}% (support->{best_eid} {pick_ok}, "
                f"oracle {oracle_acc:.2f}%){mark}"
            )
        print()

    mat = np.array(accuracy_matrix, dtype=float)
    metric = compute_forgetting(mat)

    pd.DataFrame(
        mat, index=task_names, columns=[f"after_t{i + 1}" for i in range(num_tasks)]
    ).to_csv(outdir / "task_accuracies.csv")
    pd.DataFrame(
        {
            "task": task_names,
            "initial_accuracy": np.round(metric["initial"], 4),
            "best_accuracy": np.round(metric["best"], 4),
            "final_accuracy": np.round(metric["final"], 4),
            "forgetting": np.round(metric["forgetting"], 4),
        }
    ).to_csv(outdir / "forgetting.csv", index=False)

    print("=" * 60)
    print("EvAGI LLM (experts + register + governors + support-set HRM):")
    for i, n in enumerate(task_names):
        print(
            f"{n}: init {metric['initial'][i]:.2f} best {metric['best'][i]:.2f} "
            f"final {metric['final'][i]:.2f} forgetting {metric['forgetting'][i]:.2f}"
        )
    print(
        f"Avg forgetting {metric['average_forgetting']:.2f} "
        f"final avg {metric['final_average_accuracy']:.2f}"
    )
    print(f"Register: {json.dumps(register.summary())}")

    with open(outdir / "run_config.json", "w") as f:
        json.dump(
            {
                "model_id": model_id,
                "n_params": n_params,
                "probe": probe,
                "metric": {
                    k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in metric.items()
                },
                "register": register.summary(),
                "routing_log": routing_log,
                "llm_cfg": {k: v for k, v in cfg.items() if k != "task_names"},
                "task_names": task_names,
            },
            f,
            indent=2,
        )
    # persist masks + governors (small)
    torch.save(
        {
            "masks": [[m.cpu() for m in layer_masks] for layer_masks in masks_per_expert],
            "governor_state": [g.state_dict() for g in governors],
            "router_state": router.state_dict(),
        },
        outdir / "evagi_state.pt",
    )
    print(f"Wrote {outdir}")


if __name__ == "__main__":
    main()
