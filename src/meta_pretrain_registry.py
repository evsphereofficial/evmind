"""Meta-pretraining of HRM intent network with Task Registry.

Same protocol as meta_pretrain.py but uses TaskRegistry instead of
SensitivityMemory. After installing each old task, we capture its
footprint. During the burst on new task B, the governor receives
registry protection features as additional per-weight inputs.
"""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.func import functional_call
from .config import load_config
from .experiment import set_seed
from .hrm import build_module_groups, compute_global_features, HRMIntentGovernor, mask_stats
from .meta_pretrain import BCE, make_meta_task, meta_task_labels, reset_model, sample_chunk, warmup_batches, _sample_batch
from .model import TinyNumericTransformer
from .registry import TaskRegistry
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent


def registry_gated_burst(
    model, governor, groups, p_cur, task, lr, steps, batch_size,
    seed_base, device, differentiable, ungated=False, g_old=None, p0=None,
    registry=None, registry_up_to_task=0, optim="adamw", close_threshold=0.02,
    second_order=False,
):
    """Unroll governor-gated updates with registry protection features.

    FOMAML-style (second_order=False): inner-loop grads are detached so no
    double-backward flows through attention or Adam's sqrt (which NaNs via
    SqrtBackward when v_hat ~ 0). Masks stay differentiable via the governor.
    """
    mask_log, gate_means = [], []
    use_adam = optim == "adamw"
    beta1, beta2, ad_eps = 0.9, 0.999, 1e-8
    if use_adam:
        m_buf = [torch.zeros_like(p_cur[g.name]) for g in groups]
        v_buf = [torch.zeros_like(p_cur[g.name]) for g in groups]
    g_old_hat = None
    if g_old is not None:
        g_old_hat = [g.detach() / (g.abs().mean() + 1e-12) for g in g_old]
    reg_param_prot, reg_region_prot, reg_regions, reg_group_sizes = None, None, None, None
    if registry is not None and registry.n > 0:
        reg_param_prot, reg_region_prot = registry.get_protection(up_to_task=registry_up_to_task)
        reg_regions, reg_group_sizes = registry.regions, [g.size for g in groups]
    xb = sample_chunk(steps, batch_size, seed_base, device)
    yb = meta_task_labels(xb, task)
    for s in range(steps):
        x = xb[s * batch_size:(s + 1) * batch_size]
        y = yb[s * batch_size:(s + 1) * batch_size]
        pred = functional_call(model, p_cur, (x,))
        loss = BCE(pred, y)
        grad_list = torch.autograd.grad(loss, [p_cur[g.name] for g in groups], create_graph=second_order, allow_unused=False)
        if ungated:
            masks = [torch.ones_like(p_cur[g.name].flatten()) for g in groups]
        else:
            hist = None
            if p0 is not None:
                hist = [(p_cur[g.name].detach() - p0[g.name]).abs() / (p0[g.name].abs() + 1e-8) for g in groups]
            masks = governor.gate_from_state(
                p_cur, grad_list, groups, x, y, loss, device,
                differentiable=differentiable, g_old_list=g_old_hat,
                hist_list=hist, mem_imp_list=None, mem_dir_list=None,
                hmem_list=None,
                registry_param_protection=reg_param_prot,
                registry_region_protection=reg_region_prot,
                registry_regions=reg_regions,
                registry_group_sizes=reg_group_sizes)
        mask_log.append(mask_stats(masks))
        gate_means.append(torch.cat([m.flatten() for m in masks]).mean())
        for i, (group, m, g) in enumerate(zip(groups, masks, grad_list)):
            if use_adam and close_threshold > 0.0:
                m = torch.where(m < close_threshold, torch.zeros_like(m), m)
            mf = m.reshape(g.shape)
            if use_adam:
                closed = mf == 0.0
                m_buf[i] = beta1 * m_buf[i] + (1 - beta1) * g
                v_buf[i] = beta2 * v_buf[i] + (1 - beta2) * g * g
                if closed.any():
                    m_buf[i] = m_buf[i].masked_fill(closed, 0.0)
                    v_buf[i] = v_buf[i].masked_fill(closed, 0.0)
                t = s + 1
                m_hat, v_hat = m_buf[i] / (1 - beta1 ** t), v_buf[i] / (1 - beta2 ** t)
                p_cur[group.name] = p_cur[group.name] - lr * mf * (m_hat / (v_hat.sqrt() + ad_eps))
            else:
                p_cur[group.name] = p_cur[group.name] - lr * mf * g
    return p_cur, mask_log, torch.stack(gate_means).mean(), None


def _warmup_with_shadow(model, task, n_batches, bsize, lr, seed_base, device, groups, registry):
    """Warmup that also feeds the shadow tracker if enabled."""
    if registry is not None and registry.use_shadow:
        registry.begin_task(device=device)
        # shadow-aware warmup: accumulate |grad| per batch
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        xb = sample_chunk(n_batches, bsize, seed_base, device)
        yb = meta_task_labels(xb, task)
        for k in range(n_batches):
            x = xb[k * bsize:(k + 1) * bsize]
            y = yb[k * bsize:(k + 1) * bsize]
            loss = BCE(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if registry.use_shadow:
                registry.accumulate([p.grad for p in [g.param for g in groups]])
            opt.step()
        return
    warmup_batches(model, task, n_batches, bsize, lr, seed_base=seed_base, device=device)


def meta_step_registry(model, governor, groups, config, device, seed_off, rng):
    """One meta step with registry: install old tasks, capture footprints, burst on new task."""
    m = config.meta
    optim_name = getattr(m, "burst_optim", "adamw")
    base_lr = m.lr if optim_name == "adamw" else getattr(m, "burst_lr", m.lr)
    warmup_lr = getattr(m, "warmup_lr", m.lr)
    bsize = m.batch_size
    old_tasks_max = int(getattr(m, "old_tasks_max", 0))
    reg_cfg = getattr(config, "registry", None)
    rsize = getattr(reg_cfg, "region_size", 1000) if reg_cfg else 1000
    rfp = getattr(reg_cfg, "footprint_method", "grad_x_weight") if reg_cfg else "grad_x_weight"
    rnorm = getattr(reg_cfg, "normalize", True) if reg_cfg else True
    rshadow = bool(getattr(reg_cfg, "use_shadow", False)) if reg_cfg else False
    rshadow_alpha = float(getattr(reg_cfg, "shadow_alpha", 0.0)) if reg_cfg else 0.0
    rlearn = bool(getattr(reg_cfg, "learnable", False)) if reg_cfg else False
    rhid = int(getattr(reg_cfg, "recognizer_hidden", 16)) if reg_cfg else 16
    torch.manual_seed(100_000 + seed_off * 7)
    reset_model(model)
    k_old = int(torch.randint(0, old_tasks_max + 1, (1,), generator=rng)[0])
    old_tasks = [make_meta_task(rng) for _ in range(k_old)]
    fam_b = make_meta_task(rng)
    registry = TaskRegistry(groups, region_size=rsize, footprint_method=rfp, normalize=rnorm,
                            use_shadow=rshadow, shadow_alpha=rshadow_alpha,
                            learnable=rlearn, recognizer_hidden=rhid)
    # Move recognizer to device so its params participate in same device graph
    if registry.recognizer is not None:
        registry.recognizer.to(device)
    g_old_imm = None
    for t_i, fam_a in enumerate(old_tasks):
        _warmup_with_shadow(model, fam_a, m.warmup_batches, bsize, warmup_lr,
                            seed_base=seed_off * 31 + t_i * 7, device=device, groups=groups, registry=registry)
        model.zero_grad(set_to_none=True)
        x_cap = _sample_batch(bsize, seed_offset=seed_off * 45 + 13 + t_i * 17).to(device)
        y_cap = meta_task_labels(x_cap, fam_a).to(device)
        loss_cap = BCE(model(x_cap), y_cap)
        loss_cap.backward()
        g_t = [g.param.grad.detach().clone() for g in groups]
        # Shadow capture: task-ID'd per-task footprint; learnable recognizer
        # keeps graph if differentiable=True (meta-training).
        registry.capture(model, gradients=g_t, task_name=str(fam_a), differentiable=rlearn)
        g_old_imm = [gg.detach().clone() for gg in g_t]
    model.zero_grad(set_to_none=True)
    p_cur = {g.name: g.param.detach().clone().requires_grad_(True) for g in groups}
    p0 = {g.name: t.detach().clone() for g, t in zip(groups, [p_cur[g.name] for g in groups])}
    # Shadow for the burst task B is NOT captured (only old tasks act as
    # protection); registry is read-only during burst, no accumulation needed.
    p_cur, mask_log, mean_gate, _ = registry_gated_burst(
        model, governor, groups, p_cur, fam_b, lr=base_lr, steps=m.burst_steps,
        batch_size=bsize, seed_base=seed_off * 37 + 5, device=device, differentiable=True,
        g_old=g_old_imm, p0=p0, registry=registry, registry_up_to_task=registry.n,
        optim=optim_name, close_threshold=getattr(m, "close_threshold", 0.02),
        second_order=getattr(m, "second_order", False))
    x_bv = _sample_batch(bsize, seed_offset=seed_off * 43 + 11).to(device)
    y_bv = meta_task_labels(x_bv, fam_b).to(device)
    loss_new = BCE(functional_call(model, p_cur, (x_bv,)), y_bv)
    old_losses = []
    for t_i, fam_a in enumerate(old_tasks):
        x_av = _sample_batch(bsize, seed_offset=seed_off * 41 + 9 + t_i * 19).to(device)
        y_av = meta_task_labels(x_av, fam_a).to(device)
        old_losses.append(BCE(functional_call(model, p_cur, (x_av,)), y_av))
    loss_old = torch.stack(old_losses).mean() if old_losses else loss_new.detach() * 0.0
    dW = torch.cat([(p_cur[g.name] - p0[g.name]).abs().flatten() for g in groups])
    p0_abs = torch.cat([p0[g.name].abs().flatten() for g in groups])
    mean_rel_change = (dW.mean() + 1e-12) / (p0_abs.mean() + 1e-12)
    if optim_name == "adamw":
        train_batches = int(math.ceil(config.train_samples / config.train.batch_size))
        live_steps = train_batches * int(config.train.epochs_per_task)
        phase_scale = live_steps / max(1, int(m.burst_steps))
    else:
        phase_scale = 1.0
    sparse_cost = (mean_gate - getattr(m, "sparse_target", 0.3)) ** 2
    gov_loss = loss_new + m.lambda_old * loss_old + m.lambda_sparse * sparse_cost + m.lambda_delta * mean_rel_change * phase_scale
    model.zero_grad(set_to_none=True)
    return {"gov_loss": gov_loss, "loss_new": loss_new, "loss_old": loss_old,
            "sparse_cost": sparse_cost, "delta_cost": mean_rel_change * phase_scale,
            "phase_scale": phase_scale, "mask_stats": mask_log[-1]}


def main():
    parser = argparse.ArgumentParser(description="HRM governor meta-pretraining with Task Registry")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "registry.yaml"))
    parser.add_argument("--outdir", default=str(PROJECT_ROOT / "results_registry"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.train.seed = args.seed
    if args.steps is not None:
        cfg.meta.steps = args.steps
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for f in outdir.iterdir():
        if f.is_file():
            f.unlink()
    set_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if getattr(cfg.meta, "second_order", False):
        # true MAML unroll needs double backward; efficient SDPA has none
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cudnn.benchmark = True
    print(f"Device: {device}")
    model = TinyNumericTransformer(
        input_dim=cfg.model.input_dim, seq_len=cfg.model.seq_len,
        embedding_dim=cfg.model.embedding_dim, num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads, ff_dim=cfg.model.ff_dim, dropout=cfg.model.dropout,
    ).to(device)
    groups = build_module_groups(model)
    total_weights = sum(g.size for g in groups)
    reg_cfg = getattr(cfg, "registry", None)
    registry_region_size = getattr(reg_cfg, "region_size", 1000) if reg_cfg else 1000
    hmem_mode = getattr(cfg, "hmem_mode", "none")
    governor = HRMIntentGovernor(
        num_groups=len(groups), granularity=cfg.governor.granularity,
        hidden_dim=cfg.governor.hidden_dim, refine_steps=cfg.governor.refine_steps,
        init_mask=cfg.governor.init_mask,
        per_weight_feat_dim=8 + (1 if hmem_mode != "none" else 0),
        registry_feat_dim=2,
    ).to(device)
    m = cfg.meta
    print("=" * 60)
    print("HRM GOVERNOR META-PRETRAINING (TASK REGISTRY)")
    print("=" * 60)
    print(f"Base model params: {model.count_parameters():,}  weights: {total_weights:,}")
    print(f"Governor params: {governor.governor_params():,}")
    print(f"Registry: region_size={registry_region_size}, "
          f"method={getattr(reg_cfg, 'footprint_method', 'grad_x_weight')}, "
          f"shadow={bool(getattr(reg_cfg, 'use_shadow', False))}")
    print(f"Objective: L_new + {m.lambda_old}*L_old + {m.lambda_sparse}*(M-{getattr(m, 'sparse_target', 0.3)})^2 + {m.lambda_delta}*delta")
    print(f"Meta: {m.steps} steps, batch={m.meta_batch}, burst={m.burst_steps}x{getattr(m, 'burst_optim', 'adamw')}")
    print(f"Old tasks: 0..{getattr(m, 'old_tasks_max', 0)}\n")
    # Joint governor + registry recognizer training: registry learns to
    # transform shadow -> protection, governor learns to use it.
    # We create a shared registry prototype to hold learnable params; per-step
    # registries are clones that share its recognizer weights via state_dict.
    proto_reg = TaskRegistry(groups, region_size=registry_region_size,
                             footprint_method=getattr(reg_cfg, 'footprint_method', 'grad_x_weight'),
                             normalize=True,
                             use_shadow=bool(getattr(reg_cfg, 'use_shadow', False)),
                             shadow_alpha=float(getattr(reg_cfg, 'shadow_alpha', 0.0)),
                             learnable=bool(getattr(reg_cfg, 'learnable', False)),
                             recognizer_hidden=int(getattr(reg_cfg, 'recognizer_hidden', 16)))
    if proto_reg.recognizer is not None:
        proto_reg.recognizer.to(device)
        reg_params = list(proto_reg.recognizer.parameters())
        gov_params = list(governor.parameters())
        joint_opt = torch.optim.AdamW(gov_params + reg_params, lr=m.lr)
    else:
        joint_opt = torch.optim.AdamW(governor.parameters(), lr=m.lr)
        reg_params = []
    # Pass proto recognizer weights into per-step registries via hook:
    # meta_step_registry clones recognizer state if learnable; we monkey-patch
    # by sharing the proto recognizer object (all per-step registries will
    # reuse proto's recognizer params via reference).
    # Simplest: temporarily replace TaskRegistry creation to share recognizer.
    _orig_TR = TaskRegistry
    def _shared_TR(*a, **kw):
        r = _orig_TR(*a, **kw)
        if proto_reg.recognizer is not None and r.recognizer is not None:
            # share params: make r.recognizer point to proto's module
            r.recognizer = proto_reg.recognizer
        return r
    import src.meta_pretrain_registry as _mpr
    _mpr.TaskRegistry = _shared_TR  # type: ignore
    # Also need to patch the local reference used by meta_step_registry closure
    # (it looks up TaskRegistry globally at call time, so patching module works)

    reset_model(model)
    rng = torch.Generator().manual_seed(7_777 + cfg.train.seed)
    start = time.perf_counter()
    log, ema = [], {"L_new": 0.0, "L_old": 0.0, "sparse": 0.0, "delta": 0.0}
    for step in range(1, m.steps + 1):
        outs = [meta_step_registry(model, governor, groups, cfg, device, (step - 1) * m.meta_batch + k, rng) for k in range(m.meta_batch)]
        total_loss = sum(o["gov_loss"] for o in outs) / len(outs)
        joint_opt.zero_grad(set_to_none=True)
        total_loss.backward()
        if reg_params:
            torch.nn.utils.clip_grad_norm_(gov_params + reg_params, 1.0)
        else:
            torch.nn.utils.clip_grad_norm_(governor.parameters(), 1.0)
        joint_opt.step()
        o = outs[-1]
        # NaN guard: stop early with diagnostics instead of silently saving NaN weights
        if not torch.isfinite(total_loss.detach()):
            print(f"step {step}: NON-FINITE gov_loss={float(total_loss.detach()):.4f} -- stopping early")
            print(f"  L_new={float(o['loss_new'].detach()):.4f} L_old={float(o['loss_old'].detach()):.4f}")
            break
        for key, v in zip(ema, [o["loss_new"], o["loss_old"], o["sparse_cost"], o["delta_cost"]]):
            ema[key] = 0.98 * ema[key] + 0.02 * float(v.detach())
        if step % 50 == 0 or step == 1:
            stats = o["mask_stats"]
            log.append({"step": step, "gov_loss": float(total_loss.detach()),
                        "loss_new": ema["L_new"], "loss_old": ema["L_old"],
                        "sparse": ema["sparse"], "delta": ema["delta"],
                        **{k: round(v, 4) for k, v in stats.items()}})
            print(f"step {step:5d}/{m.steps}  L_new={ema['L_new']:.4f} L_old={ema['L_old']:.4f} "
                  f"sp={ema['sparse']:.4f} dW={ema['delta']:.5f}  "
                  f"mask mean={stats['mask_mean']:.3f} <0.1:{stats['frac_lt_0.1']:.2f}")
    pd.DataFrame(log).to_csv(outdir / "meta_train_log.csv", index=False)
    torch.save(governor.state_dict(), outdir / "governor_pretrained.pt")
    if proto_reg.recognizer is not None:
        torch.save(proto_reg.recognizer.state_dict(), outdir / "registry_recognizer.pt")
    info = {"steps": m.steps, "meta_batch": m.meta_batch,
            "governor_params": governor.governor_params(),
            "registry_params": sum(p.numel() for p in reg_params) if reg_params else 0,
            "registry_learnable": bool(getattr(reg_cfg, 'learnable', False)),
            "base_params": model.count_parameters(),
            "controlled_weights": total_weights, "region_size": registry_region_size,
            "objective": {"lambda_old": m.lambda_old, "lambda_sparse": m.lambda_sparse,
                          "sparse_target": getattr(m, "sparse_target", 0.3),
                          "lambda_delta": m.lambda_delta},
            "training_seconds": round(time.perf_counter() - start, 1)}
    with open(outdir / "pretrain_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"\n{'=' * 60}\nGOVERNOR META-PRETRAINING COMPLETE (REGISTRY)\n{'=' * 60}")
    print(f"Governor: {info['governor_params']:,} params, Registry recognizer: {info['registry_params']:,} params, {total_weights:,} controlled weights")
    print(f"Training: {info['training_seconds']:.0f}s")
    print(f"Saved: {outdir / 'governor_pretrained.pt'}" + (f" + registry_recognizer.pt" if reg_params else ""))


if __name__ == "__main__":
    main()
