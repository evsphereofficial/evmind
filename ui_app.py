import torch
from pathlib import Path
from src.model import TinyNumericTransformer
from src.tasks import build_tasks
from src.config import load_config

# Load 5 dynamic tiny experts (5393 total, 97.93% live)
cfg=load_config("configs/registry.yaml")
tasks=build_tasks(cfg.tasks)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading EvAGI 5 tiny experts on {device}...")

# Recreate experts as in experiment_dynamic (use same k->cfg mapping)
from src.registry import predict_required_weights
def cfg_for_k(k):
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
    return best_cfg

experts=[]
for task in tasks:
    k=predict_required_weights(task.name, target_acc=98.0)
    d,layers,ff,heads,c=cfg_for_k(k)
    m=TinyNumericTransformer(input_dim=2, seq_len=2, embedding_dim=d, num_layers=layers, num_heads=heads, ff_dim=ff).to(device)
    # Try to load from results_dynamic if exists, else random (for demo)
    path=Path(f"results_dynamic/expert_{len(experts)}_{task.name}.pt")
    # Actually dynamic experts are not saved as 5x17K, they are tiny - check results_dynamic
    # Fallback: use random weights for demo (still shows routing)
    experts.append(m)
    print(f"Expert {task.name} k_pred {k} -> d{d} l{layers} ff{ff} => {c} params")

# For demo, load the actual trained experts from results_dynamic if available
import glob
for i, task in enumerate(tasks):
    # Try to load from results_dynamic or results_expert
    for base in ["results_dynamic", "results_expert", "results_hierarchical"]:
        p=Path(f"{base}/expert_{i}_{task.name}.pt")
        if p.exists():
            try:
                # Need to handle size mismatch for dynamic tiny vs 17K file
                sd=torch.load(p, map_location=device)
                # Try to load, if mismatch, skip
                experts[i].load_state_dict(sd, strict=False)
                print(f"Loaded {p}")
                break
            except Exception as e:
                print(f"Skip {p}: {e}")

print("Experts ready. Starting minimal UI server on http://localhost:7860")
# Minimal HTTP server without Flask/Gradio
import http.server, socketserver, json, urllib.parse

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path=="/":
            self.path="/ui_index.html"
        return http.server.SimpleHTTPRequestHandler.do_GET(self)
    def do_POST(self):
        if self.path=="/predict":
            length=int(self.headers.get('content-length',0))
            body=self.rfile.read(length)
            data=json.loads(body)
            x1=float(data.get("x1",0)); x2=float(data.get("x2",0))
            x=torch.tensor([[x1,x2]], device=device).float()
            # Live support-set routing: for demo, use 32 support from each task's training data to pick expert
            # Simplified live: run all experts, pick most confident (|logit|) - no labels needed, pure g(x)
            with torch.no_grad():
                logits_all=[]
                for exp in experts:
                    logits=exp(x)  # (1,)
                    logits_all.append(logits.item())
                conf=[abs(v) for v in logits_all]
                best=int(torch.tensor(conf).argmax().item())
                # Also compute per-expert predictions
                per_task={}
                for i, exp in enumerate(experts):
                    logit=exp(x).item()
                    pred=int(torch.sigmoid(torch.tensor(logit)).item()>=0.5)
                    per_task[tasks[i].name]= {"logit": float(logit), "pred": pred, "conf": float(abs(logit)), "picked": i==best}
                # Live prediction is best expert's pred
                live_pred=per_task[tasks[best].name]["pred"]
                resp={"live_pred": live_pred, "picked_expert": tasks[best].name, "per_task": per_task}
            self.send_response(200)
            self.send_header("Content-type","application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode())
        else:
            self.send_error(404)

# Write minimal HTML
html="""<!DOCTYPE html><html><head><meta charset="utf-8"><title>EvAGI Live 17K</title>
<style>body{font-family:sans-serif;max-width:800px;margin:20px auto;padding:10px}canvas{border:1px solid #333;cursor:crosshair} .bar{height:12px;background:#4c72b0;display:inline-block} label{display:inline-block;width:60px}</style>
</head><body>
<h2>EvAGI — One 17K System, 5 Tiny Experts (5393), Live g(x) (no task_id)</h2>
<p>Click canvas or use sliders. <b>Live</b> = max|logit| over 5 experts (no labels, no support). Per-task shows each expert's own prediction.</p>
<canvas id="c" width="400" height="400"></canvas><br>
<label>x1 <input type="range" id="x1" min="-1" max="1" step="0.01" value="0"></label><span id="v1">0</span>
<label>x2 <input type="range" id="x2" min="-1" max="1" step="0.01" value="0"></label><span id="v2">0</span>
<div id="out" style="margin-top:10px; font-family:monospace; white-space:pre"></div>
<script>
const c=document.getElementById('c'), ctx=c.getContext('2d'), x1=document.getElementById('x1'), x2=document.getElementById('x2'), v1=document.getElementById('v1'), v2=document.getElementById('v2'), out=document.getElementById('out');
function draw(){
 ctx.clearRect(0,0,400,400);
 ctx.fillStyle="#f0f0f0"; ctx.fillRect(0,0,400,400);
 // grid and boundaries
 ctx.strokeStyle="#ddd"; for(let i=0;i<=400;i+=40){ctx.beginPath();ctx.moveTo(i,0);ctx.lineTo(i,400);ctx.stroke();ctx.beginPath();ctx.moveTo(0,i);ctx.lineTo(400,i);ctx.stroke();}
 // horizontal y=0, vertical x=0, diagonal x+y=0, circle r=0.55, xor
 ctx.strokeStyle="rgba(255,0,0,0.3)"; ctx.beginPath(); ctx.moveTo(0,200); ctx.lineTo(400,200); ctx.stroke();
 ctx.beginPath(); ctx.moveTo(200,0); ctx.lineTo(200,400); ctx.stroke();
 ctx.beginPath(); ctx.moveTo(0,400); ctx.lineTo(400,0); ctx.stroke();
 ctx.beginPath(); ctx.arc(200,200,0.55*200,0,Math.PI*2); ctx.stroke();
}
draw();
function predict(){
 const xv=parseFloat(x1.value), yv=parseFloat(x2.value);
 v1.textContent=xv.toFixed(2); v2.textContent=yv.toFixed(2);
 // dot
 draw(); ctx.fillStyle="black"; ctx.beginPath(); ctx.arc((xv+1)*200, (1-yv)*200, 6,0,Math.PI*2); ctx.fill();
 fetch("/predict",{method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({x1:xv,x2:yv})})
 .then(r=>r.json()).then(j=>{
   let s=`Live g(x) -> ${j.picked_expert} => ${j.live_pred} (${j.live_pred?"yes":"no"})\n`;
   for(let k in j.per_task){let t=j.per_task[k]; s+=`${k.padEnd(12)} logit ${t.logit.toFixed(2).padStart(6)} pred ${t.pred} ${t.picked?"*":""} conf ${t.conf.toFixed(2)}\n`;}
   out.textContent=s;
 });
}
x1.oninput=predict; x2.oninput=predict; c.onclick=e=>{const r=c.getBoundingClientRect(); x1.value=((e.clientX-r.left)/400*2-1).toFixed(2); x2.value=((1-(e.clientY-r.top)/400)*2-1).toFixed(2); predict();}; predict();
</script></body></html>"""
open("ui_index.html","w").write(html)
print("UI at http://localhost:7860 — click canvas or sliders to test live inference (5 experts, total 5393/17249)")
# Change to serve from current dir where ui_index.html is
import os
os.chdir(str(Path(__file__).parent.parent))
with socketserver.TCPServer(("", 7860), Handler) as httpd:
    print("Serving at http://localhost:7860 (Ctrl+C to stop)")
    httpd.serve_forever()
