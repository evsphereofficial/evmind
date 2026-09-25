"""Fine-tune Laya (System 1) on the EvAGI routing schema.

Single-GPU adaptation of the official laya fine-tune recipe
(notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb):
  CE + RLCD proper-reward group loss, encoder lr 2.5e-5 / head 1e-4,
  grad checkpointing, fp16 autocast, cosine schedule, post-hoc
  temperature fitting, safetensors output loadable by laya.Agent.

Run:
  .venv/bin/python src/gen_laya_evagi_data.py
  .venv/bin/python src/train_laya_evagi.py
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("USE_TF", "0")

import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from evagi_system1 import (  # noqa: E402
    LAYA_DIR, build_questions, load_jsonl,
)

REPO = Path(__file__).parent.parent
DATA_DIR = REPO / "data" / "laya_evagi"
OUT_DIR = Path(os.environ.get("EVAGI_LAYA_DIR", LAYA_DIR)).resolve()

EPOCHS = 6
MICRO_BATCH = 8
GRAD_ACCUM = 2          # effective 16
GROUP_SIZE = 4          # RLCD group samples
LR_ENCODER = 2.5e-5
LR_HEAD = 1.0e-4
SIGMA_START = 0.4
SIGMA_END = 0.1


def resolve_base_dir() -> str:
    env = os.environ.get("EVAGI_LAYA_BASE")
    if env and Path(env).exists():
        return env
    local = Path(
        "/mnt/d/huggingface_cache/hub/models--convaiinnovations--laya/"
        "snapshots/1c5edc17a7acd8701df6fc341c0d179f1c62c982"
    )
    if local.exists():
        return str(local)
    from huggingface_hub import snapshot_download
    return snapshot_download("convaiinnovations/laya")


def build_training_item(row: dict, qid: str, q: dict):
    from laya.common import QTYPES, build_sequence, render_options

    t = q["type"]
    if t != "choice":
        raise ValueError("evagi schema is choice-only")
    keys = list(q["criteria"].keys())
    gold_key = row["gold"].get(qid)
    if gold_key is None or gold_key not in keys:
        return None
    target = [1.0 if k == gold_key else 0.0 for k in keys]
    k = len(render_options({"t": t, "crit": q["criteria"]}))
    seq, markers = build_sequence(
        tok, row["state"],
        {"t": t, "ins": q["instructions"], "crit": q["criteria"]},
        cfg["max_len"], cfg["head_max_len"],
    )
    if len(markers) != k:
        return None
    return {
        "ids": seq, "markers": markers, "qtype": QTYPES[t],
        "target": target, "label": keys.index(gold_key),
    }


def collate(items, pad_id):
    n = len(items)
    L = max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, : len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)
    return {
        "input_ids": ids, "attention_mask": att,
        "marker_pos": mpos, "marker_mask": mmask, "target": target,
        "qtype": torch.tensor([it["qtype"] for it in items]),
        "label": torch.tensor([it["label"] for it in items]),
    }


def fit_one_temp(sel):
    if len(sel) < 10:
        return 1.0
    kmax = max(len(z) for z, _ in sel)
    Z = torch.full((len(sel), kmax), -1e4)
    T = torch.zeros((len(sel), kmax))
    for i, (z, t) in enumerate(sel):
        Z[i, :len(z)] = torch.tensor(z)
        T[i, :len(t)] = torch.tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())


def evaluate(model, items, device) -> tuple[float, dict]:
    model.eval()
    correct = {"action": 0, "fact_kind": 0, "topic": 0}
    total = {"action": 0, "fact_kind": 0, "topic": 0}
    # group items by their question identity via label-space size + stored qid tag
    with torch.no_grad():
        for i in range(0, len(items), 16):
            chunk = items[i:i + 16]
            b = collate(chunk, tok.pad_token_id)
            with torch.autocast("cuda", dtype=torch.float16):
                logits, _ = model(
                    b["input_ids"].to(device), b["attention_mask"].to(device),
                    b["marker_pos"].to(device), b["marker_mask"].to(device),
                    b["qtype"].to(device),
                )
            pred = logits.float().argmax(-1).cpu()
            for j, it in enumerate(chunk):
                qn = it["qname"]
                total[qn] += 1
                if int(pred[j]) == it["label"]:
                    correct[qn] += 1
    model.train()
    per = {q: (correct[q] / total[q] if total[q] else 0.0) for q in total}
    acc = sum(correct.values()) / max(1, sum(total.values()))
    return acc, per


if __name__ == "__main__":
    from laya.common import build_model, proper_reward
    from laya.agent import _fix_tokenizer_config

    device = torch.device("cuda")
    base_dir = resolve_base_dir()
    print(f"base: {base_dir}\nout : {OUT_DIR}")

    with open(os.path.join(base_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    cfg.setdefault("gradient_checkpointing", True)
    cfg["max_len"] = 512
    cfg["head_max_len"] = 192

    _fix_tokenizer_config(base_dir)
    tok = AutoTokenizer.from_pretrained(os.path.join(base_dir, "tokenizer"))
    model = build_model(cfg, encoder_dir=os.path.join(base_dir, "encoder"))
    weights = load_file(os.path.join(base_dir, "model.safetensors"))
    model.load_state_dict(weights, strict=True)
    model.encoder.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = True
    model.to(device).train()

    # ---- build items ----
    items = []
    for split in ("train", "val"):
        for row in load_jsonl(str(DATA_DIR / f"{split}.jsonl")):
            qs = build_questions(row.get("topics") or [])
            for qid, q in qs.items():
                it = build_training_item(row, qid, q)
                if it:
                    it["qname"] = qid
                    it["split"] = split
                    items.append(it)
    train_items = [it for it in items if it["split"] == "train"]
    val_items = [it for it in items if it["split"] == "val"]
    print(f"items: train={len(train_items)} val={len(val_items)}")

    enc_params = [p for n, p in model.named_parameters() if "encoder." in n]
    head_params = [p for n, p in model.named_parameters() if "encoder." not in n]
    optimizer = torch.optim.AdamW(
        [{"params": enc_params, "lr": LR_ENCODER},
         {"params": head_params, "lr": LR_HEAD}],
        weight_decay=0.01,
    )
    total_updates = (len(train_items) // (MICRO_BATCH * GRAD_ACCUM)) * EPOCHS
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_updates), eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    print(f"training: {EPOCHS} epochs, {len(train_items)} items, "
          f"eff batch {MICRO_BATCH * GRAD_ACCUM}")
    t0 = time.time()
    rng = random.Random(42)
    best_val = 0.0
    for epoch in range(EPOCHS):
        rng.shuffle(train_items)
        epoch_loss, n_batches, accum = 0.0, 0, 0
        optimizer.zero_grad(set_to_none=True)
        sigma = SIGMA_START + (SIGMA_END - SIGMA_START) * (epoch / max(1, EPOCHS - 1))
        for b_idx in range(0, len(train_items), MICRO_BATCH):
            chunk = train_items[b_idx:b_idx + MICRO_BATCH]
            if not chunk:
                continue
            b = collate(chunk, tok.pad_token_id)
            with torch.autocast("cuda", dtype=torch.float16):
                logits, act = model(
                    b["input_ids"].to(device), b["attention_mask"].to(device),
                    b["marker_pos"].to(device), b["marker_mask"].to(device),
                    b["qtype"].to(device),
                )
            logits = logits.float()
            mask = b["marker_mask"].to(device)
            k = mask.sum(-1, keepdim=True).float()
            target = b["target"].to(device)

            eps = torch.randn((GROUP_SIZE,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), b["qtype"].to(device),
                                  mask, w_sph=0.75, w_rps=1.0)
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(target * torch.log_softmax(
                logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = (loss_rl + 1.0 * loss_ce) / GRAD_ACCUM + 0.0 * act.sum()

            scaler.scale(loss).backward()
            accum += 1
            if accum % GRAD_ACCUM == 0 or (b_idx + MICRO_BATCH) >= len(train_items):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                sched.step()
                optimizer.zero_grad(set_to_none=True)
            epoch_loss += loss.item() * GRAD_ACCUM
            n_batches += 1
        vacc, vper = evaluate(model, val_items, device)
        best_val = max(best_val, vacc)
        print(f"epoch {epoch + 1}/{EPOCHS} loss={epoch_loss / max(1, n_batches):.4f} "
              f"val_acc={vacc:.3f} {vper} ({time.time() - t0:.0f}s)")

    # ---- temperature fit on val ----
    print("fitting temperature ...")
    model.eval()
    sel = []
    with torch.no_grad():
        for i in range(0, len(val_items), 16):
            chunk = val_items[i:i + 16]
            b = collate(chunk, tok.pad_token_id)
            with torch.autocast("cuda", dtype=torch.float16):
                logits, _ = model(
                    b["input_ids"].to(device), b["attention_mask"].to(device),
                    b["marker_pos"].to(device), b["marker_mask"].to(device),
                    b["qtype"].to(device),
                )
            lnp = logits.float().cpu().numpy()
            for j, it in enumerate(chunk):
                kk = len(it["markers"])
                sel.append((it["qtype"], lnp[j, :kk], it["target"]))
    def nll_at(ts):
        import numpy as np
        tot = 0.0
        for q_type, z, t in sel:
            zz = np.asarray(z, dtype="float64") / ts[int(q_type)]
            zz -= zz.max()
            logp = zz - np.log(np.exp(zz).sum())
            tot -= sum(ti * pi for ti, pi in zip(t, logp))
        return tot / max(1, len(sel))

    try:
        temps = [1.2, 1.2, 1.2]
        for qt in range(3):
            s2 = [(z, t) for q_type, z, t in sel if q_type == qt]
            if s2:
                temps[qt] = fit_one_temp(s2)
        nll_fit = nll_at(temps)
        nll_one = nll_at([1.0, 1.0, 1.0])
        print(f"temp fit={[round(t, 3) for t in temps]} "
              f"valNLL@fit={nll_fit:.4f} valNLL@1={nll_one:.4f}")
        if nll_one <= nll_fit:
            temps = [1.0, 1.0, 1.0]
            print("keeping T=1 (fit did not improve val NLL)")
    except Exception as e:  # pragma: no cover
        temps = [1.0, 1.0, 1.0]
        print("temp fit fallback:", e)

    # Inherited per-option-count buckets were fit on laya's own benchmark;
    # they override the global temperature and would mis-gate our schema.
    cfg.pop("temperature_by_options", None)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sd = {k: v.half().contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(sd, str(OUT_DIR / "model.safetensors"))
    model.encoder.config.save_pretrained(str(OUT_DIR / "encoder"))
    tok.save_pretrained(str(OUT_DIR / "tokenizer"))
    cfg["fine_tuned"] = True
    cfg["model_name"] = "laya-evagi"
    cfg["temperature"] = temps
    with open(OUT_DIR / "rl_agent_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"saved -> {OUT_DIR}  (best val acc {best_val:.3f}, "
          f"{time.time() - t0:.0f}s total)")
