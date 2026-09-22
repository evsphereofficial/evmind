"""
EvAGI Live LLM — Real TinyStories-33M, FFN neuron slicing, v2 law, live training.

One 68.5M model. 12288 FFN neurons. Capacity law predicts n per task.
Per-task: select n rows from c_fc + n cols from c_proj, mask the rest.
1M extra budget for new live learning via user UI.
"""
from __future__ import annotations
import json, math, time, http.server, socketserver, os, sys, traceback
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
os.chdir(str(PROJECT))

from src.tasks import build_tasks, Task
from src.config import load_config
from src.registry import predict_required_weights, CAPACITY_PARAMS

cfg = load_config("configs/registry.yaml")
tasks = build_tasks(cfg.tasks)
TASK_NAMES = [t.name for t in tasks]
NUM_TASKS = len(tasks)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LIVE_BUDGET = 1_000_000

# ── Load TinyStories-33M ────────────────────────────────────────────────
print("Loading TinyStories-33M...")
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-33M")
tokenizer.pad_token = tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained(
    "roneneldan/TinyStories-33M", torch_dtype=torch.float32
).to(device)
base_model.eval()
for p in base_model.parameters():
    p.requires_grad = False

HIDDEN = base_model.config.hidden_size  # 768
N_LAYERS = len(base_model.transformer.h)  # 4
INTERS = [layer.mlp.c_fc.weight.shape[0] for layer in base_model.transformer.h]  # [3072]*4
TOTAL_NEURONS = sum(INTERS)  # 12288
PARAMS_PER_NEURON = HIDDEN * 2  # 1536 (c_fc row 768 + c_proj col 768)
print(f"  {sum(p.numel() for p in base_model.parameters())} params, "
      f"{TOTAL_NEURONS} FFN neurons, {PARAMS_PER_NEURON} params/neuron")

# ── Pre-tokenize support-set prompts for all tasks ──────────────────────
# 32 support samples per task, labeled
def make_support(task_idx: int, n: int = 32, seed: int = 42):
    """Generate n labeled support samples for task."""
    gen = torch.Generator().manual_seed(seed * 10000 + task_idx)
    pts = torch.rand(n, 2, generator=gen) * 2 - 1
    task = tasks[task_idx]
    labels = task.labels(pts[:, 0], pts[:, 1])
    prompts = []
    for i in range(n):
        x1, x2 = pts[i, 0].item(), pts[i, 1].item()
        y = "yes" if labels[i].item() == 1 else "no"
        prompts.append((f"Point ({x1:.2f},{x2:.2f}) {task.name}?", y))
    return prompts

SUPPORT_SETS = {i: make_support(i) for i in range(NUM_TASKS)}

# ── Text classification helper: prompt -> yes/no logit ──────────────────
def get_yes_no_logits(model, prompts_batch):
    """Get (yes_logit, no_logit) for each prompt in batch."""
    yes_id = tokenizer.encode("yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("no", add_special_tokens=False)[0]
    enc = tokenizer(prompts_batch, padding=True, truncation=True,
                    max_length=64, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    last_logits = out.logits[:, -1, :]  # (B, vocab)
    return last_logits[:, yes_id], last_logits[:, no_id]

def eval_expert_on_task(model, task_idx, n_test=200):
    """Eval a model (with current masks) on a task's test set."""
    task = tasks[task_idx]
    gen = torch.Generator().manual_seed(99999 + task_idx)
    pts = torch.rand(n_test, 2, generator=gen) * 2 - 1
    labels = task.labels(pts[:, 0], pts[:, 1])
    correct = 0
    batch_size = 32
    for i in range(0, n_test, batch_size):
        bx = pts[i:i+batch_size]
        by = labels[i:i+batch_size]
        prompts = []
        for j in range(len(bx)):
            x1, x2 = bx[j, 0].item(), bx[j, 1].item()
            prompts.append(f"Point ({x1:.2f},{x2:.2f}) {task.name}?")
        yes_logits, no_logits = get_yes_no_logits(model, prompts)
        preds = (yes_logits > no_logits).long()
        correct += (preds == by.to(device)).sum().item()
    return 100 * correct / n_test

# ── Expert pool with FFN neuron slicing ────────────────────────────────
class FFNExpertPool:
    """Manages experts via neuron-level allocation on the single 33M model."""
    def __init__(self):
        self.experts = []  # [{task_name, task_idx, neurons: [(layer, idx), ...], n_params, ...}]
        self.occupied = [torch.zeros(inter, dtype=torch.bool) for inter in INTERS]
        self.occupied_params = 0
        self.total_budget = LIVE_BUDGET + sum(p.numel() for p in base_model.parameters())
        self.history = []  # log of all events
        # Save original weights
        self.orig_c_fc = [layer.mlp.c_fc.weight.data.clone() for layer in base_model.transformer.h]
        self.orig_c_proj = [layer.mlp.c_proj.weight.data.clone() for layer in base_model.transformer.h]
        self.orig_fc_bias = [layer.mlp.c_fc.bias.data.clone() if layer.mlp.c_fc.bias is not None else None
                             for layer in base_model.transformer.h]
        self.orig_proj_bias = [layer.mlp.c_proj.bias.data.clone() if layer.mlp.c_proj.bias is not None else None
                               for layer in base_model.transformer.h]

    def allocate(self, task_name: str, task_idx: int) -> dict:
        """Allocate neurons via v2 law for this task."""
        k_pred = predict_required_weights(task_name, target_acc=98.0)
        n_neurons = max(4, k_pred // PARAMS_PER_NEURON)
        # Distribute across layers
        per_layer = max(1, n_neurons // N_LAYERS)
        neurons = []
        for li in range(N_LAYERS):
            occ = self.occupied[li]
            free_idx = torch.where(~occ)[0]
            take = min(per_layer, free_idx.numel())
            if take > 0:
                chosen = free_idx[:take]
                for ci in chosen.tolist():
                    neurons.append((li, ci))
                occ[chosen] = True
        actual_n = len(neurons)
        actual_params = actual_n * PARAMS_PER_NEURON
        self.occupied_params += actual_params
        expert = {
            "task_name": task_name,
            "task_idx": task_idx,
            "neurons": neurons,
            "n_neurons": actual_n,
            "n_params": actual_params,
            "k_predicted": k_pred,
            "best_acc": 0.0,
            "current_acc": 0.0,
            "steps": 0,
            "trained_at": time.time(),
        }
        self.experts.append(expert)
        return expert

    def apply_masks(self, expert_idx: int):
        """Mask the 33M model to only allow expert_idx's neurons to pass."""
        expert = self.experts[expert_idx]
        # First, reset to original
        for li in range(N_LAYERS):
            base_model.transformer.h[li].mlp.c_fc.weight.data.copy_(self.orig_c_fc[li])
            base_model.transformer.h[li].mlp.c_proj.weight.data.copy_(self.orig_c_proj[li])
            if self.orig_fc_bias[li] is not None:
                base_model.transformer.h[li].mlp.c_fc.bias.data.copy_(self.orig_fc_bias[li])
            if self.orig_proj_bias[li] is not None:
                base_model.transformer.h[li].mlp.c_proj.bias.data.copy_(self.orig_proj_bias[li])
        # Zero out all neurons NOT in this expert
        for li in range(N_LAYERS):
            mask = torch.zeros(INTERS[li], dtype=torch.bool)
            for el, ci in expert["neurons"]:
                if el == li:
                    mask[ci] = True
            # c_fc: zero rows not in mask
            fc_w = base_model.transformer.h[li].mlp.c_fc.weight.data
            fc_w[~mask] = 0
            if base_model.transformer.h[li].mlp.c_fc.bias is not None:
                base_model.transformer.h[li].mlp.c_fc.bias.data[~mask] = 0
            # c_proj: zero cols not in mask
            proj_w = base_model.transformer.h[li].mlp.c_proj.weight.data
            proj_w[:, ~mask] = 0
            if base_model.transformer.h[li].mlp.c_proj.bias is not None:
                # c_proj bias not masked (output bias, all neurons contribute)
                pass

    def train_expert(self, expert_idx: int, epochs: int = 2, lr: float = 5e-5,
                     n_train: int = 500) -> dict:
        """Train expert by fine-tuning only its neurons' c_fc/c_proj on task data."""
        expert = self.experts[expert_idx]
        task_idx = expert["task_idx"]
        task = tasks[task_idx]
        task_name = expert["task_name"]
        neuron_set = set(expert["neurons"])

        # Generate training data
        gen = torch.Generator().manual_seed(int(time.time()) % 100000)
        pts = torch.rand(n_train, 2, generator=gen) * 2 - 1
        labels = task.labels(pts[:, 0], pts[:, 1])

        # Build prompts
        prompts_all = []
        for i in range(n_train):
            x1, x2 = pts[i, 0].item(), pts[i, 1].item()
            prompts_all.append(f"Point ({x1:.2f},{x2:.2f}) {task_name}?")
        labels_text = ["yes" if labels[i].item() == 1 else "no" for i in range(n_train)]

        # Unfreeze only this expert's neurons' c_fc and c_proj
        for p in base_model.parameters():
            p.requires_grad = False
        for li in range(N_LAYERS):
            for el, ci in expert["neurons"]:
                if el == li:
                    base_model.transformer.h[li].mlp.c_fc.weight.requires_grad = True
                    if base_model.transformer.h[li].mlp.c_fc.bias is not None:
                        base_model.transformer.h[li].mlp.c_fc.bias.requires_grad = True
                    base_model.transformer.h[li].mlp.c_proj.weight.requires_grad = True
                    break  # only need to unfreeze once per layer that has neurons

        trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
        optimizer = torch.optim.AdamW(
            [p for p in base_model.parameters() if p.requires_grad], lr=lr
        )

        batch_size = 16
        total_loss = 0
        total_steps = 0

        for epoch in range(epochs):
            perm = torch.randperm(n_train)
            epoch_loss = 0
            for i in range(0, n_train, batch_size):
                idx = perm[i:i+batch_size]
                bx = pts[idx]
                by_text = [labels_text[j] for j in idx.tolist()]
                bp = [prompts_all[j] for j in idx.tolist()]

                # Tokenize prompt + label as causal LM
                enc_prompt = tokenizer(bp, padding=True, truncation=True,
                                       max_length=64, return_tensors="pt").to(device)
                enc_label = tokenizer(by_text, padding=True, truncation=True,
                                      return_tensors="pt").to(device)
                input_ids = torch.cat([enc_prompt["input_ids"], enc_label["input_ids"]], dim=1)
                attention_mask = torch.cat([enc_prompt["attention_mask"], enc_label["attention_mask"]], dim=1)
                labels_mask = input_ids.clone()
                labels_mask[:, :enc_prompt["input_ids"].shape[1]] = -100

                optimizer.zero_grad()
                out = base_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_mask)
                loss = out.loss
                loss.backward()

                # Mask gradients: only allow grad for this expert's neurons
                for li in range(N_LAYERS):
                    mask = torch.zeros(INTERS[li], device=device)
                    for el, ci in expert["neurons"]:
                        if el == li:
                            mask[ci] = 1.0
                    # c_fc grad: rows
                    if base_model.transformer.h[li].mlp.c_fc.weight.grad is not None:
                        base_model.transformer.h[li].mlp.c_fc.weight.grad *= mask.unsqueeze(1)
                    if base_model.transformer.h[li].mlp.c_fc.bias is not None and base_model.transformer.h[li].mlp.c_fc.bias.grad is not None:
                        base_model.transformer.h[li].mlp.c_fc.bias.grad *= mask
                    # c_proj grad: cols
                    if base_model.transformer.h[li].mlp.c_proj.weight.grad is not None:
                        base_model.transformer.h[li].mlp.c_proj.weight.grad *= mask.unsqueeze(0)

                optimizer.step()
                epoch_loss += loss.item()
                total_steps += 1

            total_loss += epoch_loss

        # Freeze everything again
        for p in base_model.parameters():
            p.requires_grad = False

        # Eval this expert on its own task
        self.apply_masks(expert_idx)
        acc = eval_expert_on_task(base_model, task_idx, n_test=200)
        expert["current_acc"] = acc
        if acc > expert["best_acc"]:
            expert["best_acc"] = acc
        expert["steps"] += total_steps

        return {
            "accuracy": round(acc, 2),
            "loss": round(total_loss / max(total_steps, 1), 4),
            "trainable_params": trainable,
            "steps": total_steps,
        }

    def predict(self, x1: float, x2: float) -> dict:
        """Run all experts on a point, route via max|logit|."""
        if not self.experts:
            return {"error": "no experts", "live_pred": 0, "per_task": {}}

        point = f"Point ({x1:.2f},{x2:.2f}) "
        results = {}
        best_conf = -1
        best_name = ""
        best_pred = 0

        for expert in self.experts:
            task_name = expert["task_name"]
            prompt = point + task_name + "?"
            self.apply_masks(self.experts.index(expert))
            base_model.eval()
            yes_logit, no_logit = get_yes_no_logits(base_model, [prompt])
            yl = yes_logit.item()
            nl = no_logit.item()
            pred = 1 if yl > nl else 0
            conf = abs(yl - nl)
            results[task_name] = {
                "logit": round(yl - nl, 4),
                "pred": pred,
                "conf": round(conf, 4),
                "yes_logit": round(yl, 4),
                "no_logit": round(nl, 4),
                "n_neurons": expert["n_neurons"],
                "best_acc": round(expert["best_acc"], 2),
                "steps": expert["steps"],
            }
            if conf > best_conf:
                best_conf = conf
                best_name = task_name
                best_pred = pred

        return {
            "live_pred": best_pred,
            "picked_expert": best_name,
            "per_task": results,
            "occupied": self.occupied_params,
            "budget": self.total_budget,
            "budget_pct": round(100 * self.occupied_params / self.total_budget, 2),
        }

pool = FFNExpertPool()

# ── Pre-train 5 experts on their tasks ─────────────────────────────────
print("\n=== Allocating & training 5 experts on TinyStories-33M ===")
for i, task in enumerate(tasks):
    exp = pool.allocate(task.name, i)
    print(f"[{i+1}/5] {task.name}: k_pred={exp['k_predicted']} -> {exp['n_neurons']} neurons "
          f"({exp['n_params']} params) across {N_LAYERS} layers")
    result = pool.train_expert(i, epochs=2, lr=5e-5, n_train=500)
    print(f"       accuracy={result['accuracy']}% loss={result['loss']} "
          f"trainable={result['trainable_params']} params")

print(f"\nBudget: {pool.occupied_params}/{pool.total_budget} ({round(100*pool.occupied_params/pool.total_budget, 2)}%)")
print("=== Ready for live inference ===\n")

# ── HTML UI ─────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>EvAGI TinyStories-33M Live</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'SF Mono','Fira Code',monospace;background:#0a0a0a;color:#e0e0e0;padding:20px;max-width:1100px;margin:0 auto}
h1{color:#00ff88;font-size:18px;margin-bottom:2px}
.sub{color:#666;font-size:11px;margin-bottom:16px}
h2{color:#00cc66;font-size:13px;margin:14px 0 6px;border-bottom:1px solid #222;padding-bottom:4px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.card{background:#111;border:1px solid #333;border-radius:6px;padding:10px}
.metric{display:flex;justify-content:space-between;padding:2px 0;font-size:11px}
.metric .l{color:#888}.metric .v{color:#fff;font-weight:bold}
.v.good{color:#00ff88}.v.warn{color:#ffaa00}.v.bad{color:#ff4444}
.bar-bg{height:6px;background:#222;border-radius:3px;overflow:hidden;margin:4px 0}
.bar-fg{height:100%;background:linear-gradient(90deg,#00ff88,#00cc66);border-radius:3px;transition:width 0.3s}
.erow{display:flex;align-items:center;gap:6px;padding:4px 0;border-bottom:1px solid #1a1a1a;font-size:11px}
.erow .nm{width:90px;color:#00cc66;font-weight:bold}.erow .pr{width:55px;color:#888}
.erow .ac{width:50px}.erow .pd{width:40px;padding:1px 4px;border-radius:3px;text-align:center;font-weight:bold;font-size:10px}
.pd.y{background:#003300;color:#00ff88}.pd.n{background:#330000;color:#ff4444}
.erow .pk{color:#ffaa00;font-weight:bold}
canvas{border:1px solid #333;border-radius:4px;cursor:crosshair;display:block;margin:6px 0}
.ctrl{display:flex;gap:6px;align-items:center;margin:6px 0;flex-wrap:wrap}
.ctrl select,.ctrl input[type=number]{background:#1a1a1a;border:1px solid #444;color:#fff;padding:3px 6px;border-radius:3px;font-family:inherit;font-size:11px}
.ctrl button{background:#00ff88;color:#000;border:none;padding:5px 12px;border-radius:3px;font-family:inherit;font-weight:bold;font-size:11px;cursor:pointer}
.ctrl button:hover{background:#00cc66}
.ctrl button:disabled{background:#333;color:#666;cursor:not-allowed}
.log{background:#0a0a0a;border:1px solid #222;border-radius:4px;padding:6px;max-height:150px;overflow-y:auto;font-size:10px;line-height:1.5}
.log .e{border-bottom:1px solid #151515;padding:1px 0}.log .t{color:#555}.log .ok{color:#00ff88}.log .er{color:#ff4444}
.sbar{display:flex;gap:12px;padding:6px 0;font-size:10px;color:#666;border-top:1px solid #222;margin-top:12px}
</style></head><body>
<h1>EvAGI TinyStories-33M</h1>
<div class="sub">One 68.5M model &middot; 12288 FFN neurons &middot; v2 capacity law &middot; neuron-level slicing &middot; live training &middot; g(support) routing</div>
<div class="grid"><div>
<h2>Canvas</h2>
<canvas id="c" width="400" height="400"></canvas>
<div class="ctrl">
<label style="font-size:11px;color:#888">x1</label><input type="range" id="x1" min="-1" max="1" step="0.01" value="0.5" style="width:80px"><span id="v1" style="color:#888;font-size:11px">0.50</span>
<label style="font-size:11px;color:#888">x2</label><input type="range" id="x2" min="-1" max="1" step="0.01" value="0.5" style="width:80px"><span id="v2" style="color:#888;font-size:11px">0.50</span>
<button onclick="infer()">Infer</button>
<span id="lr" style="font-size:11px;color:#00ff88"></span>
</div>
<h2>Live Train</h2>
<div class="ctrl">
<select id="lt"></select>
<select id="ll"><option value="1">yes</option><option value="0">no</option></select>
<label style="font-size:11px;color:#888">N</label><input type="number" id="ln" value="200" min="1" max="2000" style="width:55px">
<button id="lbtn" onclick="live_train()">Train</button>
</div>
<div id="ts" style="font-size:10px;color:#888;margin:2px 0"></div>
</div><div>
<h2>Expert Pool</h2>
<div id="ep"></div>
<h2>Metrics</h2>
<div id="mt" class="card" style="font-size:11px"></div>
</div></div>
<h2>Event Log</h2>
<div class="log" id="lg"></div>
<div class="sbar">
<span>Neurons: <b id="sn">0</b>/12288</span>
<span>Budget: <b id="sb">0</b>W</span>
<span>Experts: <b id="se">0</b></span>
<span>Avg Acc: <b id="sa">0</b>%</span>
<span>Device: <b id="sd">--</b></span>
</div>
<script>
const c=document.getElementById('c'),ctx=c.getContext('2d');
let cx=0.5,cy=0.5;
function draw(){ctx.clearRect(0,0,400,400);ctx.fillStyle='#111';ctx.fillRect(0,0,400,400);
ctx.strokeStyle='#222';ctx.lineWidth=.5;for(let i=0;i<=400;i+=40){ctx.beginPath();ctx.moveTo(i,0);ctx.lineTo(i,400);ctx.stroke();ctx.beginPath();ctx.moveTo(0,i);ctx.lineTo(400,i);ctx.stroke();}
ctx.strokeStyle='#444';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(0,200);ctx.lineTo(400,200);ctx.stroke();ctx.beginPath();ctx.moveTo(200,0);ctx.lineTo(200,400);ctx.stroke();
const px=(cx+1)/2*400,py=(1-cy)/2*400;ctx.fillStyle='#fff';ctx.beginPath();ctx.arc(px,py,4,0,Math.PI*2);ctx.fill();
ctx.strokeStyle='#00ff88';ctx.lineWidth=2;ctx.beginPath();ctx.arc(px,py,4,0,Math.PI*2);ctx.stroke();}
function us(){cx=+document.getElementById('x1').value;cy=+document.getElementById('x2').value;
document.getElementById('v1').textContent=cx.toFixed(2);document.getElementById('v2').textContent=cy.toFixed(2);draw();}
document.getElementById('x1').oninput=us;document.getElementById('x2').oninput=us;
c.onclick=e=>{const r=c.getBoundingClientRect();cx=((e.clientX-r.left)/400*2-1);cy=((1-(e.clientY-r.top)/400)*2-1);
document.getElementById('x1').value=cx;document.getElementById('x2').value=cy;us();infer();};draw();
function infer(){fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({x1:cx,x2:cy})})
.then(r=>r.json()).then(j=>{if(j.error){document.getElementById('lr').innerHTML='<span style="color:#ff4444">'+j.error+'</span>';return;}
const c=j.live_pred?'yes':'no';document.getElementById('lr').innerHTML='g(x) &rarr; <b>'+j.picked_expert+'</b> = <b>'+c+'</b>';
let h='';for(const[n,d]of Object.entries(j.per_task)){const pc=d.pred?'y':'n';
h+='<div class="erow"><span class="nm'+(n===j.picked_expert?' pk':'')+'">'+n+'</span>';
h+='<span class="pr">'+d.n_neurons+'n</span>';
h+='<span class="ac" style="color:'+(d.best_acc>90?'#00ff88':d.best_acc>70?'#ffaa00':'#ff4444')+'">'+d.best_acc+'%</span>';
h+='<span class="pd '+pc+'">'+(d.pred?'yes':'no')+'</span>';
h+='<span style="color:#555">'+d.logit.toFixed(2)+'</span></div>';}
document.getElementById('ep').innerHTML=h;document.getElementById('sn').textContent=Object.values(j.per_task).reduce((a,d)=>a+d.n_neurons,0);
document.getElementById('sb').textContent=j.occupied;document.getElementById('se').textContent=Object.keys(j.per_task).length;draw();});}
function live_train(){const b=document.getElementById('lbtn');b.disabled=true;b.textContent='...';
document.getElementById('ts').textContent='Training on 33M with sliced FFN neurons...';
fetch('/api/learn',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({x1:cx,x2:cy,task:document.getElementById('lt').value,label:document.getElementById('ll').value,n:parseInt(document.getElementById('ln').value)})})
.then(r=>r.json()).then(j=>{b.disabled=false;b.textContent='Train';
if(j.error){document.getElementById('ts').innerHTML='<span style="color:#ff4444">'+j.error+'</span>';return;}
document.getElementById('ts').innerHTML='<span style="color:#00ff88">Trained '+j.task+': acc='+j.accuracy+'% neurons='+j.n_neurons+' loss='+j.loss+'</span>';
alog('train',j.task,'acc='+j.accuracy+'% neurons='+j.n_neurons+' loss='+j.loss);refresh();})
.catch(e=>{b.disabled=false;b.textContent='Train';document.getElementById('ts').innerHTML='<span style="color:#ff4444">'+e+'</span>';});}
function refresh(){fetch('/api/metrics').then(r=>r.json()).then(j=>{
document.getElementById('sa').textContent=j.avg_accuracy;
let m='';for(const[n,d]of Object.entries(j.per_task||{})){
m+='<div class="metric"><span class="l">'+n+'</span><span class="v '+(d.forgetting<1?'good':d.forgetting<5?'warn':'bad')+'">acc='+d.final+'% forget='+d.forgetting+'%</span></div>';}
m+='<div class="metric" style="border-top:1px solid #333;margin-top:4px;padding-top:4px"><span class="l">Total neurons used</span><span class="v">'+j.total_neurons+'</span></div>';
m+='<div class="metric"><span class="l">Budget</span><span class="v">'+j.occupied+'/'+j.budget+' ('+j.budget_pct+'%)</span></div>';
document.getElementById('mt').innerHTML=m||'<span style="color:#666">No data</span>';});}
function alog(t,m,d){const n=new Date().toLocaleTimeString();const c=t==='train'?'ok':'er';
document.getElementById('lg').innerHTML+='<div class="e"><span class="t">'+n+'</span> ['+t+'] <span class="'+c+'">'+m+'</span> '+d+'</div>';}
function poll(){refresh();setTimeout(poll,3000);}
fetch('/api/tasks').then(r=>r.json()).then(j=>{const s=document.getElementById('lt');j.tasks.forEach(t=>{const o=document.createElement('option');o.value=t;o.textContent=t;s.appendChild(o);});});
fetch('/api/status').then(r=>r.json()).then(j=>{document.getElementById('sd').textContent=j.device;document.getElementById('sn').textContent=j.total_neurons;});
poll();infer();
</script></body></html>"""

# ── HTTP Server ─────────────────────────────────────────────────────────
class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(HTML.encode())
            return
        self.send_error(404)

    def do_POST(self):
        try:
            length = int(self.headers.get("content-length", 0))
            body = json.loads(self.rfile.read(length))
        except Exception:
            self._json(400, {"error": "bad request"})
            return

        if self.path == "/api/tasks":
            self._json(200, {"tasks": TASK_NAMES})

        elif self.path == "/api/status":
            self._json(200, {
                "device": str(device),
                "total_neurons": TOTAL_NEURONS,
                "occupied": pool.occupied_params,
                "budget": pool.total_budget,
                "budget_pct": round(100 * pool.occupied_params / pool.total_budget, 2),
                "total_experts": len(pool.experts),
            })

        elif self.path == "/api/predict":
            x1 = float(body.get("x1", 0))
            x2 = float(body.get("x2", 0))
            result = pool.predict(x1, x2)
            self._json(200, result)

        elif self.path == "/api/learn":
            task_name = body.get("task", "horizontal")
            n = int(body.get("n", 200))
            if task_name not in TASK_NAMES:
                self._json(200, {"error": f"unknown task {task_name}"})
                return
            task_idx = TASK_NAMES.index(task_name)
            # Check if expert exists
            expert_idx = None
            for i, e in enumerate(pool.experts):
                if e["task_idx"] == task_idx:
                    expert_idx = i
                    break
            if expert_idx is None:
                exp = pool.allocate(task_name, task_idx)
                expert_idx = len(pool.experts) - 1
                alog_live("alloc", task_name, f"neurons={exp['n_neurons']} params={exp['n_params']}")
            result = pool.train_expert(expert_idx, epochs=2, lr=5e-5, n_train=n)
            self._json(200, {
                "task": task_name,
                "n": n,
                "accuracy": result["accuracy"],
                "loss": result["loss"],
                "n_neurons": pool.experts[expert_idx]["n_neurons"],
                "trainable_params": result["trainable_params"],
            })

        elif self.path == "/api/metrics":
            # Eval all experts on all tasks
            per_task = {}
            total_neurons = 0
            for i, expert in enumerate(pool.experts):
                task_name = expert["task_name"]
                task_idx = expert["task_idx"]
                pool.apply_masks(i)
                acc = eval_expert_on_task(base_model, task_idx, n_test=200)
                expert["current_acc"] = acc
                if acc > expert["best_acc"]:
                    expert["best_acc"] = acc
                total_neurons += expert["n_neurons"]
                per_task[task_name] = {
                    "initial": round(expert["best_acc"], 2),
                    "best": round(expert["best_acc"], 2),
                    "final": round(acc, 2),
                    "forgetting": round(max(0, expert["best_acc"] - acc), 2),
                    "n_neurons": expert["n_neurons"],
                    "steps": expert["steps"],
                }
            accs = [v["final"] for v in per_task.values()]
            forgets = [v["forgetting"] for v in per_task.values()]
            self._json(200, {
                "avg_accuracy": round(np.mean(accs), 2) if accs else 0,
                "avg_forgetting": round(np.mean(forgets), 2) if forgets else 0,
                "per_task": per_task,
                "total_neurons": total_neurons,
                "occupied": pool.occupied_params,
                "budget": pool.total_budget,
                "budget_pct": round(100 * pool.occupied_params / pool.total_budget, 2),
                "total_experts": len(pool.experts),
            })

        else:
            self.send_error(404)

    def _json(self, code, data):
        self.send_response(code)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, fmt, *args):
        pass

def alog_live(t, m, d):
    print(f"[{t}] {m}: {d}")

PORT = 7861
print(f"\n=== EvAGI Live LLM UI at http://localhost:{PORT} ===")
print(f"Model: TinyStories-33M ({sum(p.numel() for p in base_model.parameters())} params)")
print(f"FFN: {TOTAL_NEURONS} neurons, {PARAMS_PER_NEURON} params/neuron")
print(f"Budget: {pool.total_budget} params")
print(f"Occupied: {pool.occupied_params} ({round(100*pool.occupied_params/pool.total_budget,2)}%)")
print(f"Experts: {len(pool.experts)}")
print(f"Device: {device}\n")

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    httpd.serve_forever()
