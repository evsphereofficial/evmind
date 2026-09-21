"""Single-model PackNet control — 17K budget, not 5×86K.

Same 5-task, 2k test, eval after every phase, no replay, no task_id at train
(expert per task known for allocation, but single shared weights). Masks are
disjoint, frozen after allocation. Tests if 0% needs 86K or fits 17K.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from .config import load_config
from .dataset import generate_dataset
from .evaluate import evaluate
from .experiment import make_optimizer, set_seed
from .hrm import build_module_groups
from .model import TinyNumericTransformer
from .registry import TaskRegistry, predict_required_weights
from .tasks import build_tasks
from .metrics import compute_forgetting

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

def main():
    parser=argparse.ArgumentParser(description="Single-model PackNet (17K)")
    parser.add_argument("--config", default=str(PROJECT_ROOT/"configs/registry.yaml"))
    parser.add_argument("--outdir", default=str(PROJECT_ROOT/"results_single"))
    args=parser.parse_args()
    cfg=load_config(args.config)
    outdir=Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    for f in outdir.iterdir():
        if f.is_file(): f.unlink()
    set_seed(cfg.train.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device {device}")
    tasks=build_tasks(cfg.tasks); task_names=[t.name for t in tasks]; num_tasks=len(tasks)
    datasets={}
    for i,task in enumerate(tasks):
        datasets[(i,"train")]=generate_dataset(task, cfg.train_samples, cfg.train.seed, eval_split=False)
        datasets[(i,"test")]=generate_dataset(task, cfg.test_samples, cfg.train.seed, eval_split=True)
    model=TinyNumericTransformer(input_dim=cfg.model.input_dim, seq_len=cfg.model.seq_len,
        embedding_dim=cfg.model.embedding_dim, num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads, ff_dim=cfg.model.ff_dim).to(device)
    groups=build_module_groups(model)
    N=sum(g.size for g in groups)
    print(f"Single model {N} weights")
    registry=TaskRegistry(groups, region_size=1000, normalize=True, use_shadow=True, learnable=False)
    # global masks per task
    task_masks = []  # list of Bool (N,)
    occupied = torch.zeros(N, dtype=torch.bool)
    accuracy_matrix=[[float("nan")]*num_tasks for _ in range(num_tasks)]
    optimizer=make_optimizer(model, cfg, cfg.train.learning_rate, cfg.train.weight_decay)
    loss_fn=torch.nn.BCEWithLogitsLoss()
    for phase, task in enumerate(tasks):
        task_name=task.name
        train_loader=torch.utils.data.DataLoader(datasets[(phase,"train")], batch_size=cfg.train.batch_size, shuffle=True)
        # predict k via v2 law
        k_alloc = predict_required_weights(task_name, target_acc=98.0)
        # need footprint to pick top-k free? Use next free block for now (since no footprint yet)
        # For single-model, we will train task, then after training pick top-k by |grad|*|weight| among free
        # So allocate after training, not before. For training, we allow all free weights.
        free_mask = ~occupied
        # training: only free weights are gradient-masked
        # Build per-group free mask
        def build_free_masks():
            masks=[]
            off=0
            for g in groups:
                sz=g.size
                gm = free_mask[off:off+sz].reshape(g.param.shape).to(device)
                masks.append(gm)
                off+=sz
            return masks
        free_masks = build_free_masks()
        print(f"[Task {phase+1}/{num_tasks} {task_name}] k_alloc {k_alloc} free {int(free_mask.sum())}/{N}")
        # train only free weights
        for epoch in range(cfg.train.epochs_per_task):
            model.train()
            total_loss=0; correct=0; total=0
            for x,y in train_loader:
                x,y=x.to(device).float(), y.to(device).float()
                optimizer.zero_grad(set_to_none=True)
                loss=loss_fn(model(x), y)
                loss.backward()
                # mask grads to free only
                for g, gm in zip(groups, free_masks):
                    if g.param.grad is not None:
                        g.param.grad *= gm.float()
                optimizer.step()
                # freeze occupied: ensure they stay
                # (grad masking already does, but Adam moments for occupied should be zero)
                # Zero moments for occupied
                for g, gm in zip(groups, free_masks):
                    st=optimizer.state.get(g.param)
                    if st is not None:
                        for key in ("exp_avg","exp_avg_sq"):
                            buf=st.get(key)
                            if buf is not None:
                                # where not free (occupied), zero
                                occ = (~gm).bool()
                                if occ.any():
                                    buf.masked_fill_(occ, 0.0)
                total_loss+=loss.item()*x.size(0)
                pred=(torch.sigmoid(model(x).detach())>=0.5).long()
                correct+=(pred==y.long()).sum().item()
                total+=x.size(0)
            print(f"  epoch {epoch+1} loss={total_loss/total:.4f} acc={100*correct/total:.2f}%")
        # after training, pick top-k free weights by |grad|*|weight| to freeze for this task
        # compute importance on last batch
        model.zero_grad(set_to_none=True)
        xb,yb=next(iter(train_loader))
        xb,yb=xb.to(device).float(), yb.to(device).float()
        loss=loss_fn(model(xb), yb)
        loss.backward()
        # importance per global weight
        imp_parts=[]
        for g in groups:
            imp = (g.param.grad.abs() * g.param.abs()).flatten().detach().cpu()
            imp_parts.append(imp)
        imp_flat=torch.cat(imp_parts)
        # only consider free weights
        free_idx = torch.where(free_mask)[0]
        free_imp = imp_flat[free_idx]
        # top-k among free
        k_alloc = min(k_alloc, free_idx.numel())
        topk = torch.topk(free_imp, k_alloc)
        chosen = free_idx[topk.indices]
        # mark occupied and create task mask
        task_mask = torch.zeros(N, dtype=torch.bool)
        task_mask[chosen] = True
        task_masks.append(task_mask)
        occupied[chosen] = True
        print(f"  allocated {k_alloc} for {task_name}, occupied {int(occupied.sum())}/{N}")
        # also capture in registry for logging
        g_t=[g.param.grad.detach().clone() for g in groups]
        registry.capture(model, gradients=g_t, task_name=task_name, auto_allocate=False)
        # also mark registry occupied for consistency
        registry._ensure_occupied(device=device)
        registry._occupied[chosen.to(registry._occupied.device)] = True
        model.zero_grad(set_to_none=True)
        # eval all tasks: need to apply task-specific mask at inference
        # For task i, we use model with only task i's mask active (others zeroed to init? But model weights for other tasks are frozen, not zeroed)
        # To isolate, we evaluate by masking model weights to task i's mask + keep occupied weights of that task? Actually model contains all tasks' weights mixed.
        # For single-model PackNet, inference for task i should use only weights that were allocated to task i (others masked to 0 or kept but frozen)
        # The standard PackNet inference uses binary masks to select weights for that task.
        # We will evaluate by temporarily masking: for each task i, we create a masked model where only task i's weights are active.
        # However our model currently has all tasks' weights intermingled. To evaluate task i, we need to know which weights were allocated to i.
        # We have task_masks[i] for that.
        # So for eval, we will clone model and zero out weights not in task_masks[i]?
        # But then the model would be missing weights that were not allocated to i but are needed for shared features.
        # Instead, PackNet evaluation uses the full model with all masks applied: the model weights are the same, but we mask the forward to only use task i's weights?
        # Simplification: For evaluation, we just evaluate the full model (all weights) on task i's test set — since occupied weights are frozen, the model should still perform well on old tasks if isolation worked.
        # We'll evaluate full model (not masked) for each task.
        for i in range(phase+1):
            test_loader=torch.utils.data.DataLoader(datasets[(i,"test")], batch_size=2000, shuffle=False)
            acc,_,_=evaluate(model, test_loader, device, loss_fn)
            accuracy_matrix[i][phase]=round(acc,2)
            print(f"    task {i+1} {task_names[i]} acc {acc:.2f}")
        print()
    mat=np.array(accuracy_matrix,dtype=float)
    metric=compute_forgetting(mat)
    pd.DataFrame(mat, index=task_names, columns=[f"after_t{i+1}" for i in range(num_tasks)]).to_csv(outdir/"task_accuracies.csv")
    pd.DataFrame({"task":task_names,"initial_accuracy":np.round(metric["initial"],4),"final_accuracy":np.round(metric["final"],4),"forgetting":np.round(metric["forgetting"],4)}).to_csv(outdir/"forgetting.csv", index=False)
    print("="*60)
    for i,n in enumerate(task_names):
        print(f"{n}: init {metric['initial'][i]:.2f} final {metric['final'][i]:.2f} forgetting {metric['forgetting'][i]:.2f}")
    print(f"Avg forgetting {metric['average_forgetting']:.2f} overwritten {np.nanmean(metric['forgetting'][:-1]):.2f}")
    print(f"Final avg {metric['final_average_accuracy']:.2f} occupied {int(occupied.sum())}/{N}")
    with open(outdir/"run_config.json","w") as f:
        import json
        json.dump({"metric": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k,v in metric.items()}, "occupied": int(occupied.sum()), "task_masks": [int(m.sum()) for m in task_masks]}, f, indent=2)

if __name__=="__main__":
    main()
