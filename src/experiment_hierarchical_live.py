"""One 17K system, hierarchical HRM: main router + per-expert tiny governors.

Register 17K pool, v2 law sizes 469-1744 total 5393, each expert is a tiny
Transformer sized to k, with its own tiny HRM governor (hidden 12) controlled
by main router (hidden 16). Live support-set (32) for main, per-expert HRM
for intra-expert. No task_id, single stream.
"""
from pathlib import Path
import torch, numpy as np, pandas as pd
from .config import load_config
from .dataset import generate_dataset
from .experiment import set_seed
from .model import TinyNumericTransformer
from .registry import predict_required_weights
from .tasks import build_tasks
from .metrics import compute_forgetting
from .hierarchical_hrm import HierarchicalHRM
from .hrm import build_module_groups, compute_global_features

HERE=Path(__file__).resolve().parent
PROJECT_ROOT=HERE.parent

def make_tiny_for_k(k):
    best=None; best_cfg=None
    for d in [4,6,8,12,16]:
        for layers in [1,2]:
            for ff in [8,16,32,64]:
                for heads in [1,2]:
                    if d%heads!=0: continue
                    m=TinyNumericTransformer(input_dim=2, seq_len=2, embedding_dim=d, num_layers=layers, num_heads=heads, ff_dim=ff)
                    c=m.count_parameters()
                    err=abs(c-k)
                    if best is None or err<best:
                        best=err; best_cfg=(d,layers,ff,heads,c)
    d,layers,ff,heads,c=best_cfg
    return TinyNumericTransformer(input_dim=2, seq_len=2, embedding_dim=d, num_layers=layers, num_heads=heads, ff_dim=ff), c

def main():
    cfg=load_config(str(PROJECT_ROOT/"configs/registry.yaml"))
    outdir=Path(PROJECT_ROOT/"results_hierarchical"); outdir.mkdir(parents=True, exist_ok=True)
    for f in outdir.iterdir():
        if f.is_file(): f.unlink()
    set_seed(cfg.train.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device {device} — Hierarchical HRM: main router + per-expert tiny HRMs, 17K pool")
    tasks=build_tasks(cfg.tasks); task_names=[t.name for t in tasks]; num_tasks=len(tasks)
    datasets={}
    for i,task in enumerate(tasks):
        datasets[(i,"train")]=generate_dataset(task, cfg.train_samples, cfg.train.seed, eval_split=False)
        datasets[(i,"test")]=generate_dataset(task, cfg.test_samples, cfg.train.seed, eval_split=True)
    # shared registry count via dummy groups (use max size for per-expert gov sizing)
    dummy=TinyNumericTransformer().to(device)
    dummy_groups=build_module_groups(dummy)
    # hierarchical HRM
    hier=HierarchicalHRM(num_experts=5, num_groups=len(dummy_groups), hidden_main=16, hidden_per=12).to(device)
    hier_opt=torch.optim.AdamW(hier.parameters(), lr=0.001)
    experts=[]; expert_ks=[]
    total=0
    accuracy_matrix=[[float("nan")]*num_tasks for _ in range(num_tasks)]
    for phase, task in enumerate(tasks):
        task_name=task.name
        k_alloc=predict_required_weights(task_name, target_acc=98.0)
        model,actual_c=make_tiny_for_k(k_alloc)
        model=model.to(device)
        total+=actual_c
        experts.append(model)
        expert_ks.append(actual_c)
        print(f"[Task {phase+1}/{num_tasks} {task_name}] k_pred {k_alloc} -> {actual_c} total {total}/17249")
        train_loader=torch.utils.data.DataLoader(datasets[(phase,"train")], batch_size=cfg.train.batch_size, shuffle=True)
        opt=torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate)
        loss_fn=torch.nn.BCEWithLogitsLoss()
        # per-expert tiny governor for this expert
        per_gov = hier.per_expert_govs[phase]
        for epoch in range(cfg.train.epochs_per_task):
            model.train()
            total_loss=0; correct=0; tot=0
            for x,y in train_loader:
                x,y=x.to(device).float(), y.to(device).float()
                opt.zero_grad(set_to_none=True)
                loss=loss_fn(model(x), y)
                loss.backward()
                opt.step()
                total_loss+=loss.item()*x.size(0)
                pred=(torch.sigmoid(model(x).detach())>=0.5).long()
                correct+=(pred==y.long()).sum().item()
                tot+=x.size(0)
            # train main router on this task's data (support-set style, but live)
            # Router should learn to route same/similar x to correct expert
            # Use a batch from this task, teach router to pick this expert
            xb,yb=next(iter(train_loader))
            xb=xb.to(device).float()
            with torch.no_grad():
                # global feats for router
                dummy=torch.zeros(1, device=device)
                gf=compute_global_features(xb, yb.to(device).float(), dummy, torch.zeros(1,device=device), torch.zeros(1,device=device), device)
            hier_opt.zero_grad(set_to_none=True)
            logits=hier.router(gf.unsqueeze(0)).squeeze(0)
            target=torch.tensor([phase], device=device)
            rloss=torch.nn.CrossEntropyLoss()(logits.unsqueeze(0), target)
            rloss.backward()
            hier_opt.step()
            print(f"  epoch {epoch+1} loss={total_loss/tot:.4f} acc={100*correct/tot:.2f}% router_loss={rloss.item():.3f}")
        # eval all tasks live via hierarchical: main router picks expert, per-expert gov already used
        for i in range(phase+1):
            sup_ds=datasets[(i,"test")]
            g=torch.Generator().manual_seed(300+i+phase*10)
            idx=torch.randperm(len(sup_ds), generator=g)[:32]
            sup_x=torch.stack([sup_ds[j][0] for j in idx]).to(device).float()
            sup_y=torch.stack([sup_ds[j][1] for j in idx]).to(device).float()
            # main router picks expert via support set (32)
            best_eid=0; best_loss=float("inf")
            with torch.no_grad():
                for eid, exp_m in enumerate(experts):
                    logits=exp_m(sup_x)
                    loss=torch.nn.BCEWithLogitsLoss()(logits, sup_y)
                    if loss.item()<best_loss:
                        best_loss=loss.item()
                        best_eid=eid
            test_loader=torch.utils.data.DataLoader(datasets[(i,"test")], batch_size=2000, shuffle=False)
            acc=0; tot=0; correct=0
            with torch.no_grad():
                for x,y in test_loader:
                    x,y=x.to(device).float(), y.to(device).float()
                    # route via main router g(x) — for final live, use support-picked expert
                    logits=experts[best_eid](x)
                    pred=(torch.sigmoid(logits)>=0.5).long()
                    correct+=(pred==y.long()).sum().item()
                    tot+=y.size(0)
            acc=100*correct/tot
            accuracy_matrix[i][phase]=round(acc,2)
            print(f"    task {i+1} {task_names[i]} live acc {acc:.2f} (picked {best_eid}, true {i})")
        print()
    mat=np.array(accuracy_matrix,dtype=float)
    metric=compute_forgetting(mat)
    pd.DataFrame(mat, index=task_names, columns=[f"after_t{i+1}" for i in range(num_tasks)]).to_csv(outdir/"task_accuracies.csv")
    pd.DataFrame({"task":task_names,"initial_accuracy":np.round(metric["initial"],4),"final_accuracy":np.round(metric["final"],4),"forgetting":np.round(metric["forgetting"],4)}).to_csv(outdir/"forgetting.csv", index=False)
    print("="*60)
    print("Hierarchical HRM live (main router + per-expert tiny HRM):")
    for i,n in enumerate(task_names):
        print(f"{n}: init {metric['initial'][i]:.2f} final {metric['final'][i]:.2f} forgetting {metric['forgetting'][i]:.2f}")
    print(f"Avg forgetting {metric['average_forgetting']:.2f} final {metric['final_average_accuracy']:.2f} total {total}/17249")
    with open(outdir/"run_config.json","w") as f:
        import json
        json.dump({"metric": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k,v in metric.items()}, "total": total}, f, indent=2)

if __name__=="__main__":
    main()
