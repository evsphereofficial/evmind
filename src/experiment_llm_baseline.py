"""LLM catastrophic-forgetting baseline on TinyStories-33M.

Protocol (mirrors Phase-1 2D baseline, but on a real causal LM):
  1. Probe — zero-shot yes/no accuracy on all 5 language tasks before training.
  2. Sequential fine-tune — task 1..5, full-model AdamW, no protection,
     no replay, no parameter isolation.
  3. After each phase, eval accuracy on every task seen so far
     (next-token yes vs no logit comparison).
  4. compute_forgetting -> forgetting.csv / task_accuracies.csv.

This is the number the EvAGI expert-isolation LLM run must beat.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from .experiment import set_seed
from .llm_tasks import (
    LLM_TASK_NAMES,
    generate_llm_task,
    tokenize_pairs,
    yes_no_token_ids,
)
from .metrics import compute_forgetting

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent


def load_llm_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    llm = dict(raw.get("llm", {}))
    train = dict(raw.get("train", {}))
    task_names = [t["name"] for t in raw.get("tasks", [])]
    if not task_names:
        raise ValueError("config must define tasks")
    # prefer llm.* overrides, else train.*
    llm.setdefault("lr", train.get("learning_rate", 1e-4))
    llm.setdefault("batch_size", train.get("batch_size", 16))
    llm.setdefault("epochs_per_task", train.get("epochs_per_task", 3))
    llm.setdefault("seed", train.get("seed", 0))
    llm["task_names"] = task_names
    return llm


@torch.no_grad()
def eval_yes_no_accuracy(model, tokenizer, pairs, device, max_length: int,
                         batch_size: int = 32) -> float:
    """Next-token accuracy: compare logits of ' yes' vs ' no' at answer step."""
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
        # position of first supervised token per row
        first_lab = (labels != -100).float().argmax(dim=1)
        out = model(input_ids=input_ids, attention_mask=attention)
        # logits predicting the token at position pos come from hidden at pos-1
        # For first answer token, use logits at last prompt position:
        # find index of last prompt token = first_lab - 1
        b_idx = torch.arange(input_ids.size(0), device=device)
        pred_pos = (first_lab - 1).clamp(min=0).to(device)
        logits = out.logits[b_idx, pred_pos, :]  # (B, vocab)
        pred = (logits[:, yes_id] > logits[:, no_id]).long()
        true = torch.tensor([lab for _, lab in batch], device=device)
        correct += (pred == true).sum().item()
        total += len(batch)
    return 100.0 * correct / max(total, 1)


def train_one_task(model, tokenizer, train_pairs, cfg, device):
    from torch.utils.data import DataLoader, TensorDataset

    enc = tokenize_pairs(tokenizer, train_pairs, max_length=cfg["max_length"])
    ds = TensorDataset(enc["input_ids"], enc["attention_mask"], enc["labels"])
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0.01),
    )
    model.train()
    epochs = int(cfg["epochs_per_task"])
    last_loss = 0.0
    steps = 0
    for epoch in range(epochs):
        running = 0.0
        n = 0
        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item()
            n += 1
            steps += 1
        last_loss = running / max(n, 1)
        print(f"    epoch {epoch + 1}/{epochs} loss={last_loss:.4f}")
    return last_loss


def main():
    parser = argparse.ArgumentParser(description="LLM CF baseline (TinyStories-33M)")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/llm_baseline.yaml"))
    parser.add_argument("--outdir", default=str(PROJECT_ROOT / "results_llm_baseline"))
    args = parser.parse_args()

    cfg = load_llm_cfg(args.config)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for f in outdir.iterdir():
        if f.is_file():
            f.unlink()

    set_seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = cfg.get("model_id", "roneneldan/TinyStories-33M")
    print(f"Device {device} — LLM baseline, model={model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float32
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params {n_params}")

    task_names: list[str] = cfg["task_names"]
    num_tasks = len(task_names)
    train_n = int(cfg["train_samples"])
    test_n = int(cfg["test_samples"])
    seed = int(cfg["seed"])

    train_sets = {}
    test_sets = {}
    for i, name in enumerate(task_names):
        train_sets[i] = generate_llm_task(name, train_n, seed + 1000 + i)
        test_sets[i] = generate_llm_task(name, test_n, seed + 2000 + i)
        pos = sum(1 for _, y in test_sets[i] if y == 1)
        print(f"Task {i + 1}/{num_tasks} {name}: train {len(train_sets[i])} test {len(test_sets[i])} (yes={pos})")

    max_length = int(cfg["max_length"])
    accuracy_matrix = [[float("nan")] * num_tasks for _ in range(num_tasks)]
    probe = {}

    # ---- Probe: zero-shot before any training ----
    if cfg.get("probe", True):
        print("\n=== PROBE (zero-shot pretrained) ===")
        for i, name in enumerate(task_names):
            acc = eval_yes_no_accuracy(
                model, tokenizer, test_sets[i], device, max_length
            )
            probe[name] = round(acc, 2)
            print(f"  {name}: {acc:.2f}%")
        with open(outdir / "probe.json", "w") as f:
            json.dump(probe, f, indent=2)

    # ---- Sequential fine-tune ----
    for phase, name in enumerate(task_names):
        print(f"\n[Task {phase + 1}/{num_tasks} {name}] fine-tuning full model...")
        train_one_task(model, tokenizer, train_sets[phase], cfg, device)

        for i in range(phase + 1):
            acc = eval_yes_no_accuracy(
                model, tokenizer, test_sets[i], device, max_length
            )
            accuracy_matrix[i][phase] = round(acc, 2)
            marker = "" if i == phase else " *old*"
            print(f"    eval {task_names[i]}: {acc:.2f}%{marker}")
        print()

    mat = np.array(accuracy_matrix, dtype=float)
    metric = compute_forgetting(mat)

    pd.DataFrame(
        mat,
        index=task_names,
        columns=[f"after_t{i + 1}" for i in range(num_tasks)],
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
    print("LLM BASELINE (full fine-tune, no protection):")
    for i, n in enumerate(task_names):
        print(
            f"{n}: init {metric['initial'][i]:.2f} "
            f"best {metric['best'][i]:.2f} final {metric['final'][i]:.2f} "
            f"forgetting {metric['forgetting'][i]:.2f}"
        )
    print(
        f"Avg forgetting {metric['average_forgetting']:.2f} "
        f"final avg {metric['final_average_accuracy']:.2f}"
    )

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
                "llm_cfg": {k: v for k, v in cfg.items() if k != "task_names"},
                "task_names": task_names,
            },
            f,
            indent=2,
        )
    model.save_pretrained(outdir / "final_model")
    tokenizer.save_pretrained(outdir / "final_model")
    print(f"Wrote {outdir}")


if __name__ == "__main__":
    main()
