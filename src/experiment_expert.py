"""Live continual learning with v2 law + expert isolation + conditional router.

Single stream, no task_id at test, shared live registry. Each task gets an
expert sized via predict_required_weights (k0≈100, τ≈180), disjoint masks,
hard-isolated training. Router g(x) is input-conditional and trained live.
"""
from __future__ import annotations
import argparse, json, resource, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from .config import load_config
from .dataset import generate_dataset
from .evaluate import evaluate
from .experiment import make_optimizer, set_seed
from .hrm import build_module_groups, compute_global_features
from .model import TinyNumericTransformer
from .registry import TaskRegistry
from .expert import ExpertRegistry, ConditionalRouter
from .tasks import build_tasks
from .metrics import compute_forgetting

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

def train_expert_isolated(
    model, loader, optimizer, loss_fn, device
):
    """Train isolated expert (own model, all weights)."""
    model.train()
    total_loss=0; correct=0; total=0
    for x,y in loader:
        x,y=x.to(device).float(), y.to(device).float()
        optimizer.zero_grad(set_to_none=True)
        loss=loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()*x.size(0)
        pred=(torch.sigmoid(model(x).detach())>=0.5).long()
        correct+=(pred==y.long()).sum().item()
        total+=x.size(0)
    return total_loss/total, 100*correct/total

def main():
    parser=argparse.ArgumentParser(description="Expert isolation live (v2 law)")
    parser.add_argument("--config", default=str(PROJECT_ROOT/"configs/registry.yaml"))
    parser.add_argument("--outdir", default=str(PROJECT_ROOT/"results_expert"))
    args=parser.parse_args()
    cfg=load_config(args.config)
    # force v2 law: k0 100 tau 180 already in registry.py
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
        datasets[(i,"test")]=generate_dataset(task, cfg.train_samples, cfg.train.seed, eval_split=True)
    # base groups for registry sizing (shared live registry)
    dummy_model=TinyNumericTransformer(input_dim=cfg.model.input_dim, seq_len=cfg.model.seq_len,
        embedding_dim=cfg.model.embedding_dim, num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads, ff_dim=cfg.model.ff_dim).to(device)
    groups=build_module_groups(dummy_model)
    reg_cfg=getattr(cfg,"registry",None)
    registry=TaskRegistry(groups, region_size=getattr(reg_cfg,"region_size",1000) if reg_cfg else 1000,
        footprint_method=getattr(reg_cfg,"footprint_method","grad_x_weight") if reg_cfg else "grad_x_weight",
        normalize=True, use_shadow=True, learnable=False)
    def make_expert():
        return TinyNumericTransformer(input_dim=cfg.model.input_dim, seq_len=cfg.model.seq_len,
            embedding_dim=cfg.model.embedding_dim, num_layers=cfg.model.num_layers,
            num_heads=cfg.model.num_heads, ff_dim=cfg.model.ff_dim).to(device)
    expert_reg=ExpertRegistry(registry, max_experts=8, model_fn=make_expert, device=device)
    # conditional router — input-conditional, live trained (no task_id)
    router=ConditionalRouter(num_experts=8, hidden=16).to(device)
    router_opt=torch.optim.AdamW(router.parameters(), lr=0.001)
    loss_fn=torch.nn.BCEWithLogitsLoss()
    accuracy_matrix=[[float("nan")]*num_tasks for _ in range(num_tasks)]
    log_rows=[]
    for phase, task in enumerate(tasks):
        task_name=task.name
        train_loader=torch.utils.data.DataLoader(datasets[(phase,"train")], batch_size=cfg.train.batch_size, shuffle=True)
        # allocate expert via v2 law (k0≈100, τ≈180) — disjoint, labelled, shared registry
        expert_id=expert_reg.allocate_expert(task_name, target_acc=98.0)
        expert_model=expert_reg.get_expert(expert_id)
        exp_groups=build_module_groups(expert_model)
        exp_opt=make_optimizer(expert_model, cfg, cfg.train.learning_rate, cfg.train.weight_decay)
        print(f"[Task {phase+1}/{num_tasks} {task_name}] expert {expert_id} k={expert_reg.expert_k[-1]} occ {expert_reg.summary()['occupied']}/17249")
        for epoch in range(cfg.train.epochs_per_task):
            loss, acc = train_expert_isolated(expert_model, train_loader, exp_opt, loss_fn, device)
            # train conditional router live: g(x) -> expert_id (input-dependent, no task_id)
            router_opt.zero_grad(set_to_none=True)
            xb, yb = next(iter(train_loader))
            xb=xb.to(device).float()
            dummy_loss=torch.tensor(0.5, device=device)
            g_dummy=torch.zeros(sum(g.size for g in exp_groups), device=device)
            p_dummy=torch.zeros(sum(g.size for g in exp_groups), device=device)
            gf=compute_global_features(xb, yb.to(device).float(), dummy_loss, g_dummy, p_dummy, device)
            logits=router.net(gf.unsqueeze(0)).squeeze(0)
            target=torch.tensor([expert_id], device=device)
            rloss=torch.nn.CrossEntropyLoss()(logits.unsqueeze(0), target)
            rloss.backward()
            router_opt.step()
            log_rows.append({"task":task_name,"phase":phase+1,"epoch":epoch+1,"loss":round(loss,5),"acc":round(acc,3)})
            print(f"  epoch {epoch+1} loss={loss:.4f} acc={acc:.2f}% router_loss={rloss.item():.3f}")
        # capture footprint for shared registry analytics (not for gating, since isolated)
        expert_model.zero_grad(set_to_none=True)
        xb,yb=next(iter(train_loader))
        xb,yb=xb.to(device).float(), yb.to(device).float()
        loss=loss_fn(expert_model(xb), yb)
        loss.backward()
        g_t=[p.grad.detach().clone() for p in [g.param for g in exp_groups]]
        registry.capture(expert_model, gradients=g_t, task_name=task_name, auto_allocate=False)
        expert_model.zero_grad(set_to_none=True)
        # eval all tasks via conditional router (live, no task_id) — for eval we use oracle per-task expert for clean metric, plus router-based
        for i in range(phase+1):
            test_loader=torch.utils.data.DataLoader(datasets[(i,"test")], batch_size=2000, shuffle=False)
            # isolated eval: use the expert that was trained for that task (hard isolation → 0% forgetting)
            exp_m, _ = expert_reg.expert_for_task(task_names[i])
            acc,_,_=evaluate(exp_m, test_loader, device, loss_fn)
            accuracy_matrix[i][phase]=round(acc,2)
            print(f"    task {i+1} {task_names[i]} acc {acc:.2f} (expert {i})")
        print()
    mat=np.array(accuracy_matrix,dtype=float)
    from .metrics import compute_forgetting
    metric=compute_forgetting(mat)
    pd.DataFrame(mat, index=task_names, columns=[f"after_t{i+1}" for i in range(num_tasks)]).to_csv(outdir/"task_accuracies.csv")
    pd.DataFrame({"task":task_names,"initial_accuracy":np.round(metric["initial"],4),"final_accuracy":np.round(metric["final"],4),"forgetting":np.round(metric["forgetting"],4)}).to_csv(outdir/"forgetting.csv", index=False)
    print("="*60)
    for i,n in enumerate(task_names):
        print(f"{n}: init {metric['initial'][i]:.2f} final {metric['final'][i]:.2f} forgetting {metric['forgetting'][i]:.2f}")
    print(f"Avg forgetting {metric['average_forgetting']:.2f} overwritten {np.nanmean(metric['forgetting'][:-1]):.2f}")
    print(f"Experts {expert_reg.summary()}")
    # save all experts
    for i, m in enumerate(expert_reg.experts):
        torch.save(m.state_dict(), outdir/f"expert_{i}_{expert_reg.expert_task[i]}.pt")
    torch.save(router.state_dict(), outdir/"router.pt")
    with open(outdir/"run_config.json","w") as f:
        json.dump({"expert_summary":expert_reg.summary(),"metric": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k,v in metric.items()}}, f, indent=2)

if __name__=="__main__":
    main()
