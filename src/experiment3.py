"""EvMind RAIRAW-V1 live continual-learning stream (Experiment Series 2).

Same protocol as Phase 2 (identical model, seeds, tasks, epochs, eval):
the ONLY differences are architectural — the adaptive unit is no longer a
single global per-weight gate network but a hierarchy:

  HRM governor (frozen, pretrained)  ->  WHERE: allocates regions of the
     17,249-weight main pool to a bounded RAIRAW pool (top-K regions by
     governor-mask importance, blended with the H_MEM prior after task 1)
  RAIRAW recursive cells (frozen, meta-trained)  ->  HOW: per-weight gates
     for their assigned region, from R/A/I/Controller recurrence
  H_MEM  ->  per-region EMA of RAIRAW influence reports; empty at start,
     updated after each task, blended into the next task's allocation
  SenstivityMemory (I_mem, from the HRM system)  ->  per-weight old-task
     importance features for the RAIRAW's controller

Update semantics are identical to the HRM system's enforced rules:
W = pre + M o (W_adam - pre); gates < close_threshold (incl. all inactive
regions) become closed nodes with zeroed Adam moments.

Outputs (outdir):
    task_accuracies.csv / forgetting.csv / training_log.csv   (standard)
    allocation.csv        per phase: active RAIRAW count, allocated weights
                          (and % of the 17,249 pool), region ids
    region_gates.csv      per phase: R / A / I / adapter_need / mean gate
                          per active region
    hmem.csv              H_MEM per-region influence evolution
    governor_masks.csv    batch-level gate stats (active weights)
    update_fraction.csv   main-weight modification measurement

Run:
    .venv/bin/python -m src.raira_meta --config configs/raira_v1.yaml
    .venv/bin/python -m src.experiment3 --config configs/raira_v1.yaml
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import load_config
from .dataset import generate_dataset
from .evaluate import evaluate
from .experiment import make_optimizer, plot_accuracy_matrix, set_seed
from .hrm import (
    build_module_groups, compute_global_features, compute_influence,
    HRMController, HRMIntentGovernor, mask_stats, SensitivityMemory,
    measure_rel_change, measure_update_fraction,
)
from .metrics import compute_forgetting
from .model import TinyNumericTransformer
from .raira import (
    allocate_rairaws, HmemMemory, influence_report, make_regions,
    RairawCell, RairawPool, region_importance,
)
from .tasks import build_tasks

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DEFAULT_GOVERNOR = str(PROJECT_ROOT / "results_hmem" / "magnitude_governor"
                       / "governor_pretrained.pt")
DEFAULT_CELL = str(PROJECT_ROOT / "results_raira" / "raira_cell_pretrained.pt")
DEFAULT_BASELINE_CSV = str(PROJECT_ROOT / "results" / "task_accuracies.csv")


class RairawStream:
    """RAIRAW-V1 live controller: allocation + gated updates + H_MEM."""

    def __init__(self, pool: RairawPool, regions: list, groups, offs: dict,
                 governor_hrm, hmem: HmemMemory, device: torch.device,
                 close_threshold: float, alloc_blend: float,
                 sparse_target: float) -> None:
        self.pool = pool
        self.regions = regions
        self.groups = groups
        self.offs = offs
        self.governor_hrm = governor_hrm
        self.hmem = hmem
        self.device = device
        self.close_threshold = close_threshold
        self.alloc_blend = alloc_blend
        self.sparse_target = sparse_target
        self.active: list[int] = []
        self.importance = torch.zeros(len(regions), device=device)
        self.hrm_masks_flat: torch.Tensor | None = None
        self.controller = HRMController(governor_hrm, groups, device)

    def allocate(self, model, x, y) -> dict:
        """WHERE: probe batch -> frozen HRM masks -> region importance
        (blended with H_MEM) -> top-K regions activated in the pool.
        (no no_grad wrapper: autograd.grad needs a live graph; all outputs
        are detached and the governor runs with differentiable=False)"""
        loss = torch.nn.BCEWithLogitsLoss()(model(x), y)
        loss.backward()  # populates .grad for gate_from_model (frozen gov)
        grads = [g.param.grad.detach() for g in self.groups]
        gp_all = torch.cat([g.flatten() for g in grads])
        pp_all = torch.cat([g.param.detach().flatten() for g in self.groups])
        gfeats = compute_global_features(x, y, loss, gp_all, pp_all,
                                         self.device)
        hmem_list = compute_influence(grads, "magnitude", device=self.device)
        masks = self.governor_hrm.gate_from_model(
            model, self.groups, x, y, loss, self.device,
            hmem_list=hmem_list)
        self.hrm_masks_flat = torch.cat(
            [m.detach().flatten() for m in masks])
        imp = region_importance(masks, self.groups, self.regions,
                                self.device)
        # v5: learned capacity demand D = cap_head(global ctx + H_MEM
        # summary) — NOT a fixed sparse target. k = D * n_regions.
        demand = self.governor_hrm.capacity_demand(gfeats, self.hmem.summary())
        k = max(1, min(int(round(float(demand) * len(self.regions))),
                       self.pool.max_rairaw))
        r_sizes = torch.tensor([r.size for r in self.regions],
                               device=self.device, dtype=torch.float32)
        active, blended = allocate_rairaws(
            imp, masks, self.hmem, len(self.regions), self.pool.max_rairaw,
            close_threshold=self.close_threshold, blend=self.alloc_blend,
            sparse_target=self.sparse_target, k_override=k,
            region_sizes=r_sizes)
        self.active = active
        self.importance = blended
        self.demand = float(demand)
        self.pool.reset_states(self.device)
        self.pool.activate(active)
        model.zero_grad(set_to_none=True)
        return {
            "loss": float(loss.item()),
            "active": list(active),
            "importance": blended,
            "mean_importance": float(blended.mean()),
            "demand": self.demand,
        }

    def grow(self) -> list[int]:
        """v5: iterative capacity — grant one more RAIRAW (next-best
        region by blended importance); states re-accumulate quickly."""
        k = len(self.active) + 1
        if k > self.pool.max_rairaw:
            return self.active
        order = torch.argsort(self.importance, descending=True)
        active = sorted(order[:k].tolist())
        self.active = active
        self.pool.reset_states(self.device)
        self.pool.activate(active)
        return active

    @torch.no_grad()
    def compute_masks(
        self,
        model,
        x, y, loss,
        mem_imp: list[torch.Tensor] | None,
        ret_dir: list[torch.Tensor] | None = None,
    ) -> tuple[list[torch.Tensor], list[dict]]:
        """HOW: per-weight gates from the recursive cells (active regions);
        zeros elsewhere. Returns (masks per group, per-region info dicts)."""
        sizes = [g.size for g in self.groups]
        g_flat = torch.cat([g.param.grad.detach().flatten()
                            for g in self.groups])
        p_flat = torch.cat([g.param.detach().flatten()
                            for g in self.groups])
        mem_imp_flat = (torch.cat([v.detach().flatten() for v in mem_imp])
                        if mem_imp is not None else None)
        ret_flat = (torch.cat([d.detach().flatten() for d in ret_dir])
                    if ret_dir is not None else None)
        gfeats = compute_global_features(x, y, loss, g_flat, p_flat,
                                         self.device)
        full = torch.zeros_like(g_flat)
        info_rows = []
        for r in self.regions:
            if r.region_id not in self.active:
                continue
            gates_r, info = self.pool.gate_region(
                r, g_flat, p_flat, self.hrm_masks_flat, mem_imp_flat,
                ret_flat, self.hmem.value(r.region_id), gfeats,
                alloc_frac=len(self.active) / len(self.regions),
                need_info=True)
            base = self.offs[r.group_name]
            full[base + r.start: base + r.stop] = gates_r
            info_rows.append({"region": r.region_id, "size": r.size, **info})
        return list(full.split(sizes)), info_rows

    @torch.no_grad()
    def apply(self, masks, model, pre, optimizer) -> int:
        """Enforce the gates on the real AdamW update (identical semantics
        to the HRM system): W = pre + M o (W_adam - pre) + close nodes."""
        self.controller.scale_update(model, masks, pre)
        return self.controller.zero_closed_moments(
            optimizer, masks, threshold=self.close_threshold)

    @torch.no_grad()
    def report_influence(self, g_all: torch.Tensor) -> dict[int, float]:
        """Each active RAIRAW reports its region influence to H_MEM."""
        reports = {}
        for r in self.regions:
            if r.region_id not in self.active:
                continue
            reports[r.region_id] = influence_report(g_all, g_all, r)
        return reports


def train_one_epoch_raira(
    model,
    loader,
    optimizer,
    loss_fn,
    device,
    stream: RairawStream,
    mask_rows: list[dict],
    phase: int,
    snapshot: dict,
    memory: SensitivityMemory | None,
) -> tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    start = time.perf_counter()

    for step, (x, y) in enumerate(loader):
        x, y = x.to(device).float(), y.to(device).float()
        logits = model(x)
        loss = loss_fn(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        mem_imp = memory.importance() if memory is not None and not memory.is_empty() else None
        ret_dir = memory.direction() if memory is not None and not memory.is_empty() else None
        masks, info_rows = stream.compute_masks(model, x, y, loss, mem_imp,
                                                ret_dir)
        pre = {g.name: g.param.detach().clone()
               for g in stream.groups}
        optimizer.step()
        n_closed = stream.apply(masks, model, pre, optimizer)

        if step == 0 or step == len(loader) - 1:
            stats = mask_stats(masks)
            mask_rows.append({
                "phase": phase, "batch": step, "n_closed": n_closed, **stats,
            })

        total_loss += loss.item() * x.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).long()
        correct += (preds == y.long()).sum().item()
        total += x.size(0)

    elapsed = time.perf_counter() - start
    return total_loss / total, 100.0 * correct / total, elapsed, info_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="RAIRAW-V1 live stream")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "baseline2.yaml"))
    parser.add_argument("--governor", default=DEFAULT_GOVERNOR)
    parser.add_argument("--cell", default=DEFAULT_CELL,
                        help="meta-trained RAIRAW cell checkpoint")
    parser.add_argument("--outdir", default=str(PROJECT_ROOT / "results_raira" / "live"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--order", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.train.seed = args.seed
    if args.order is not None:
        wanted = [n.strip() for n in args.order.split(",")]
        by_name = {t.name: t for t in cfg.tasks}
        if set(wanted) != set(by_name) or len(wanted) != len(by_name):
            raise SystemExit(f"--order must be a permutation of {list(by_name)}, got {wanted}")
        cfg.tasks = [by_name[n] for n in wanted]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for f in outdir.iterdir():
        if f.is_file():
            f.unlink()

    set_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tasks = build_tasks(cfg.tasks)
    task_names = [t.name for t in tasks]
    num_tasks = len(tasks)

    datasets = {}
    for i, task in enumerate(tasks):
        datasets[(i, "train")] = generate_dataset(
            task, cfg.train_samples, cfg.train.seed, eval_split=False)
        datasets[(i, "test")] = generate_dataset(
            task, cfg.test_samples, cfg.train.seed, eval_split=True)

    model = TinyNumericTransformer(
        input_dim=cfg.model.input_dim, seq_len=cfg.model.seq_len,
        embedding_dim=cfg.model.embedding_dim, num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads, ff_dim=cfg.model.ff_dim,
        dropout=cfg.model.dropout,
    ).to(device)
    num_params = model.count_parameters()

    groups = build_module_groups(model)
    total_weights = sum(g.size for g in groups)
    regions = make_regions(groups, region_size=cfg.raira.region_size)
    offs = {}
    flat_off = 0
    for g in groups:
        offs[g.name] = flat_off
        flat_off += g.size

    # frozen HRM governor (WHERE) + frozen RAIRAW cell (HOW)
    ckpt_g = torch.load(args.governor, map_location="cpu")
    pwd = int(ckpt_g["mlp.0.weight"].shape[1]) - len(groups) - 9 - 1
    governor_hrm = HRMIntentGovernor(
        num_groups=len(groups), granularity=cfg.governor.granularity,
        hidden_dim=cfg.governor.hidden_dim,
        refine_steps=cfg.governor.refine_steps,
        init_mask=cfg.governor.init_mask, per_weight_feat_dim=pwd,
    ).to(device)
    governor_hrm.load_state_dict(ckpt_g, strict=False)
    governor_hrm.eval()
    for p in governor_hrm.parameters():
        p.requires_grad_(False)

    cell = RairawCell(h_dim=cfg.raira.h_dim, intent_dim=cfg.raira.intent_dim,
                      ctx_dim=7 + 2 + 9, fw_dim=5).to(device)
    cell.load_state_dict(torch.load(args.cell, map_location=device))
    cell.eval()
    for p in cell.parameters():
        p.requires_grad_(False)
    pool = RairawPool(cell, max_rairaw=cfg.raira.max_rairaw,
                              total_weights=total_weights).to(device)

    hmem = HmemMemory(len(regions), alpha=cfg.raira.hmem_alpha, device=device)
    stream = RairawStream(pool, regions, groups, offs, governor_hrm, hmem,
                          device, cfg.raira.close_threshold or
                          getattr(cfg.governor, "close_threshold", 0.02),
                          cfg.raira.alloc_blend, cfg.raira.sparse_target)

    optimizer = make_optimizer(model, cfg, cfg.train.learning_rate,
                               cfg.train.weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    print("=" * 60)
    print("RAIRAW-V1 — LIVE CONTINUAL LEARNING STREAM")
    print("=" * 60)
    print(f"Base model parameters: {num_params:,}  (identical to Phase 1)")
    print(f"Main weight pool: {total_weights:,}  partitioned into "
          f"{len(regions)} regions (<= {cfg.raira.region_size} weights)")
    print(f"RAIRAW cell: {cell.cell_params:,} recursive params "
          f"(shared, < 1K budget), pool max {cfg.raira.max_rairaw}")
    print(f"HRM governor (frozen, WHERE): {governor_hrm.governor_params():,} "
          f"params from {args.governor}")
    print(f"H_MEM: empty at start, alpha={cfg.raira.hmem_alpha}, "
          f"alloc blend={cfg.raira.alloc_blend}")

    accuracy_matrix = [[float("nan")] * num_tasks for _ in range(num_tasks)]
    log_rows: list[dict] = []
    mask_rows: list[dict] = []
    alloc_rows: list[dict] = []
    region_rows: list[dict] = []
    hmem_rows: list[dict] = []
    growth_rows: list[dict] = []
    train_times: dict[int, float] = {}
    update_fractions: dict[int, float] = {}
    rel_changes: dict[int, float] = {}

    memory = SensitivityMemory(groups, agg="ewc", device=device)
    from .experiment2 import capture_sensitivity

    for phase, task in enumerate(tasks):
        task_name = task.name
        train_loader = torch.utils.data.DataLoader(
            datasets[(phase, "train")],
            batch_size=cfg.train.batch_size, shuffle=True,
            num_workers=cfg.train.num_workers)

        # ---- WHERE: HRM allocation for this task phase ----
        probe_x, probe_y = next(iter(train_loader))
        alloc = stream.allocate(model, probe_x.to(device).float(),
                                probe_y.to(device).float())
        alloc_rows.append({
            "task": task_name, "phase": phase + 1,
            "active_rairaws": len(alloc["active"]),
            "demand": round(alloc["demand"], 3),
            "allocated_weights": sum(r.size for r in regions
                                     if r.region_id in alloc["active"]),
            "pct_of_pool": round(100 * sum(r.size for r in regions
                                           if r.region_id in alloc["active"])
                                 / total_weights, 2),
            "probe_loss": round(alloc["loss"], 4),
            "mean_importance": round(alloc["mean_importance"], 4),
        })
        print(f"  [Task {phase+1}/{num_tasks} {task_name}] ALLOCATION: "
              f"{len(alloc['active'])} RAIRAWs (demand D={alloc['demand']:.2f})"
              f" -> {alloc_rows[-1]['allocated_weights']:,} weights "
              f"({alloc_rows[-1]['pct_of_pool']}% of {total_weights:,})")

        snapshot = {n: p.detach().clone() for n, p in model.named_parameters()}
        train_start = time.perf_counter()
        last_info = []
        def old_task_probe() -> float:
            """Mean BCE over captured tasks (one fixed batch each)."""
            if phase == 0:
                return 0.0
            with torch.no_grad():
                losses = []
                for i in range(phase):
                    xp, yp = datasets[(i, "train")][:cfg.train.batch_size]
                    losses.append(float(
                        loss_fn(model(xp.to(device).float()),
                                yp.to(device).float())))
                return sum(losses) / len(losses)

        prev_epoch_loss = None
        probe_before = old_task_probe()
        probe_prev = probe_before
        for epoch in range(cfg.train.epochs_per_task):
            loss, acc, secs, info_rows = train_one_epoch_raira(
                model, train_loader, optimizer, loss_fn, device,
                stream, mask_rows, phase + 1, snapshot, memory)
            last_info = info_rows
            log_rows.append({
                "task": task_name, "phase": phase + 1, "epoch": epoch + 1,
                "train_loss": round(loss, 5), "train_accuracy": round(acc, 3),
                "epoch_seconds": round(secs, 3),
            })
            print(f"  [Task {phase+1}/{num_tasks} {task_name}] epoch {epoch+1}"
                  f"/{cfg.train.epochs_per_task} loss={loss:.4f} acc={acc:.2f}%")
            # ---- v5: iterative capacity (RAIRAW feedback) ----
            # The task's learning response + retention damage gate whether
            # more capacity is granted: stalled learning with no old-task
            # damage -> +1 RAIRAW; easy tasks keep a small allocation.
            if prev_epoch_loss is not None:
                improvement = ((prev_epoch_loss - loss)
                               / max(prev_epoch_loss, 1e-6))
                damage = old_task_probe() - probe_prev
                growth_rows.append({
                    "task": task_name, "phase": phase + 1, "epoch": epoch + 1,
                    "improvement": round(improvement, 4),
                    "damage_probe": round(damage, 4),
                    "active_rairaws": len(stream.active),
                    "demand": round(stream.demand, 3),
                })
                if (improvement < 0.005 and loss > 0.25
                        and damage < 0.15
                        and len(stream.active) < pool.max_rairaw):
                    active_grown = stream.grow()
                    growth_rows[-1].update({"grown": True,
                                            "grown_to": len(active_grown)})
                    print(f"    -> learning stalled (improve={improvement:.3f},"
                          f" damage={damage:.3f}): +1 RAIRAW "
                          f"(now {len(active_grown)})")
                else:
                    growth_rows[-1].update({"grown": False})
                probe_prev = old_task_probe()
            prev_epoch_loss = loss
        train_times[phase] = time.perf_counter() - train_start
        update_fractions[phase] = measure_update_fraction(model, snapshot)
        rel_changes[phase] = measure_rel_change(model, snapshot)

        # ---- H_MEM: RAIRAW influence reports -> per-region EMA ----
        # use THIS task's sensitivity gradient (same capture as the memory)
        g_t = capture_sensitivity(model, groups, datasets, loss_fn,
                                  device, phase)
        g_all = torch.cat([g.detach().flatten() for g in g_t])
        reports = stream.report_influence(g_all)
        hmem.update(reports)
        # v5: RAIRAWs also report their learning response and retention
        # damage — the demand head of the next task reads these.
        first_epoch_loss = log_rows[-(cfg.train.epochs_per_task)]["train_loss"]
        learning_response = ((first_epoch_loss - loss)
                             / max(1.0, len(stream.active)))
        retention_damage = max(0.0, old_task_probe() - probe_before)
        cap_reports = {rid: (learning_response, retention_damage)
                       for rid in stream.active}
        hmem.update_capacity(cap_reports)
        hmem_rows.append({
            "task": task_name, "phase": phase + 1,
            "hmem_n_tasks": hmem.n,
            "influence_mean": round(float(hmem.influence.mean()), 4),
            "influence_max_region": int(torch.argmax(hmem.influence).item()),
            "influence_max": round(float(hmem.influence.max()), 4),
            "learning_response": round(learning_response, 4),
            "retention_damage": round(retention_damage, 4),
        })
        model.zero_grad(set_to_none=True)
        memory.update(g_t)
        for r_info in last_info:
            region_rows.append({"task": task_name, "phase": phase + 1,
                                **r_info})

        print(f"  memory: {memory.n} task(s) captured; "
              f"H_MEM: {hmem.n} task(s) of influence reports")

        print(f"  Evaluating task(s): {', '.join(task_names[: phase + 1])}")
        for i in range(phase + 1):
            test_loader = torch.utils.data.DataLoader(
                datasets[(i, "test")], batch_size=cfg.test_samples,
                shuffle=False)
            acc, _, latency_ms = evaluate(model, test_loader, device, loss_fn)
            accuracy_matrix[i][phase] = round(acc, 2)
            print(f"    task {i+1:<3} ({task_names[i]:<12}) accuracy: {acc:6.2f}%")
        print()

    mat = np.array(accuracy_matrix, dtype=float)
    metric = compute_forgetting(mat)

    pd.DataFrame(mat, index=task_names,
                 columns=[f"after_t{i+1}" for i in range(num_tasks)]
                 ).to_csv(outdir / "task_accuracies.csv")
    pd.DataFrame({
        "task": task_names,
        "initial_accuracy": np.round(metric["initial"], 4),
        "best_accuracy": np.round(metric["best"], 4),
        "final_accuracy": np.round(metric["final"], 4),
        "forgetting": np.round(metric["forgetting"], 4),
    }).to_csv(outdir / "forgetting.csv", index=False)
    pd.DataFrame(log_rows).to_csv(outdir / "training_log.csv", index=False)
    pd.DataFrame(mask_rows).to_csv(outdir / "governor_masks.csv", index=False)
    pd.DataFrame(alloc_rows).to_csv(outdir / "allocation.csv", index=False)
    pd.DataFrame(region_rows).to_csv(outdir / "region_gates.csv", index=False)
    pd.DataFrame(hmem_rows).to_csv(outdir / "hmem.csv", index=False)
    if growth_rows:
        pd.DataFrame(growth_rows).to_csv(outdir / "capacity_growth.csv",
                                         index=False)
    pd.DataFrame({
        "task": task_names,
        "update_fraction_pct": [round(update_fractions[i], 3) for i in range(num_tasks)],
        "mean_rel_change": [round(rel_changes[i], 5) for i in range(num_tasks)],
    }).to_csv(outdir / "update_fraction.csv", index=False)

    peak_ram_mb = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2)
    total_alloc = sum(r["allocated_weights"] for r in alloc_rows)

    run_info = {
        "run_name": cfg.run_name,
        "config": cfg.to_dict(),
        "device": device.type,
        "parameter_count": num_params,
        "raira_cell_params": cell.cell_params,
        "governor_params": governor_hrm.governor_params(),
        "pool_max_rairaw": cfg.raira.max_rairaw,
        "n_regions": len(regions),
        "region_size": cfg.raira.region_size,
        "hmem": {"alpha": cfg.raira.hmem_alpha,
                 "tasks_reported": hmem.n,
                 "blend": cfg.raira.alloc_blend},
        "allocation": {
            "mean_allocated_weights_pct": round(100 * total_alloc
                                                / (num_tasks * total_weights), 2),
            "per_task": {f"task{i+1}": alloc_rows[i]["allocated_weights"]
                         for i in range(num_tasks)},
        },
        "total_training_seconds": round(sum(train_times.values()), 3),
        "peak_ram_mb": peak_ram_mb,
        "final_average_accuracy": metric["final_average_accuracy"],
        "average_forgetting": metric["average_forgetting"],
        "average_forgetting_overwritten": float(np.nanmean(metric["forgetting"][:-1])),
        "update_fractions": {f"task{i+1}": update_fractions[i] for i in range(num_tasks)},
    }
    with open(outdir / "run_config.json", "w") as f:
        json.dump(run_info, f, indent=2)

    plot_accuracy_matrix(accuracy_matrix, task_names, outdir / "accuracy_matrix.png")

    print("=" * 60)
    print("RAIRAW-V1 — RESULTS")
    print(f"Main pool: {num_params:,} weights | RAIRAW cell: {cell.cell_params:,}"
          f" params (frozen) | HRM governor: {governor_hrm.governor_params():,}"
          f" (frozen, WHERE)")
    print("=" * 60)
    print("\n--- ALLOCATION (WHERE) ---")
    for i, name in enumerate(task_names):
        a = alloc_rows[i]
        print(f"  Task {i+1} ({name:<12}): {a['active_rairaws']:>2} RAIRAWs "
              f"-> {a['allocated_weights']:>6,} weights "
              f"({a['pct_of_pool']:>5.1f}% of {total_weights:,})")
    print(f"\n  Mean allocated capacity: {run_info['allocation']['mean_allocated_weights_pct']:.1f}%"
          f" of the main pool across the stream")
    print(f"  H_MEM: {hmem.n} influence report round(s), "
          f"final mean influence={hmem_rows[-1]['influence_mean']:.3f}")

    print("\n--- RETENTION ---")
    for i, name in enumerate(task_names):
        print(f"  Task {i+1} ({name:<12}): final {metric['final'][i]:.2f}% "
              f"forgetting {metric['forgetting'][i]:.2f}%")
    print(f"\n  Average Forgetting: {metric['average_forgetting']:.2f}%")
    print(f"  Average Forgetting (overwritten 1..{num_tasks-1}): "
          f"{np.nanmean(metric['forgetting'][:-1]):.2f}%")
    print(f"  Final Average Accuracy: {metric['final_average_accuracy']:.2f}%")
    print(f"\n  Update fraction / task: "
          f"{[f'{update_fractions[i]:.1f}%' for i in range(num_tasks)]}")
    print(f"  Peak RAM: {peak_ram_mb} MB")
    print(f"  Outputs written to {outdir}/")


if __name__ == "__main__":
    main()