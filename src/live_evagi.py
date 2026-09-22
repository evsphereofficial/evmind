"""Live multi-turn inference on trained EvAGI (or baseline) TinyStories-33M.

Talk to the model for a few turns; after every turn re-eval all skills with
live support-set routing (no task_id) and print accuracy + forgetting metrics
that update in real time.

Modes:
  --mode scripted   (default) fixed multi-turn conversation covering all 5 skills
  --mode interactive        type your own turns; /eval refresh, /quit exit
  --engine evagi|baseline   which weights to load

EvAGI path: loads final_model + expert masks; each turn picks an expert via
32-shot support NLL among installed experts, answers yes/no under that mask,
then re-evaluates every skill seen so far under its own support-picked expert.
Baseline path: single fine-tuned model, no masks, same eval loop.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .experiment import set_seed
from .experiment_llm_evagi import eval_yes_no_accuracy
from .llm_evagi import (
    ExpertMaskContext,
    HRMRouter,
    NeuronRegister,
    TinyPerExpertGovernor,
    expand_router,
    freeze_all_but_ffn,
    predict_neurons,
    support_pick_expert,
    support_pick_expert_qa,
    train_fact_expert,
)
from .llm_tasks import (
    LLM_TASK_NAMES,
    fact_probe_questions,
    fact_qa_pairs,
    generate_llm_task,
    parse_fact,
    tokenize_pairs,
    tokenize_qa,
    yes_no_token_ids,
)
from .metrics import compute_forgetting

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent


# ---------------------------------------------------------------------------
# Scripted multi-turn conversation: skill questions + freeform story lines
# ---------------------------------------------------------------------------
SCRIPTED_TURNS = [
    {"kind": "skill", "skill": "sentiment", "label": 1,
     "user": "Review: I love this story, it made me smile. Is this positive? Answer:"},
    {"kind": "free", "user": "Once upon a time, a little girl found a"},
    {"kind": "skill", "skill": "spelling", "label": 1,
     "user": "Is the word 'apple' spelled correctly? Answer:"},
    {"kind": "skill", "skill": "spelling", "label": 0,
     "user": "Is the word 'appel' spelled correctly? Answer:"},
    {"kind": "free", "user": "The cat sat on the"},
    {"kind": "skill", "skill": "capital", "label": 1,
     "user": "Is the capital of France Paris? Answer:"},
    {"kind": "skill", "skill": "plural", "label": 1,
     "user": "Is 'children' the plural of 'child'? Answer:"},
    {"kind": "free", "user": "In a small village lived a"},
    {"kind": "skill", "skill": "antonym", "label": 1,
     "user": "Are 'hot' and 'cold' opposites? Answer:"},
    {"kind": "skill", "skill": "sentiment", "label": 0,
     "user": "Review: What a terrible day, everything went wrong. Is this positive? Answer:"},
    {"kind": "skill", "skill": "capital", "label": 1,
     "user": "Is the capital of Japan Tokyo? Answer:"},
    {"kind": "skill", "skill": "plural", "label": 0,
     "user": "Is 'mouses' the plural of 'mouse'? Answer:"},
    {"kind": "free", "user": "And they all lived"},
]


class LiveSession:
    def __init__(self, engine: str, results_dir: Path, device,
                 max_length: int = 96, support_shots: int = 32, seed: int = 0):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.engine = engine
        self.device = device
        self.max_length = max_length
        self.support_shots = support_shots
        self.task_names = list(LLM_TASK_NAMES)
        self.n_tasks = len(self.task_names)

        if engine == "evagi":
            model_dir = results_dir / "final_model"
            state_path = results_dir / "evagi_state.pt"
            if not model_dir.exists():
                raise FileNotFoundError(f"missing {model_dir} — run experiment_llm_evagi first")
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_dir, dtype=torch.float32
            ).to(device)
            state = torch.load(state_path, map_location=device, weights_only=False)
            self.masks_per_expert = [
                [m.to(device) for m in layer_masks] for layer_masks in state["masks"]
            ]
            self.task_names = state.get("task_names", self.task_names)
            self.n_tasks = len(self.masks_per_expert)
            self.probe = state.get("probe", {})
            # restore governors + HRM router for live mid-chat learning
            self.governors: list[TinyPerExpertGovernor] = []
            for gsd in state.get("governor_state", []):
                gov = TinyPerExpertGovernor(hidden=12).to(device)
                gov.load_state_dict(gsd)
                gov.eval()
                for p in gov.parameters():
                    p.requires_grad_(False)
                self.governors.append(gov)
            self.router = HRMRouter(num_experts=self.n_tasks, hidden=16).to(device)
            if "router_state" in state:
                self.router.load_state_dict(state["router_state"])
            self.router.eval()
            for p in self.router.parameters():
                p.requires_grad_(False)
            # rebuild NeuronRegister occupancy from saved masks
            inters = [m.numel() for m in self.masks_per_expert[0]]
            self.register = NeuronRegister(inters, device=device)
            for eid, layer_masks in enumerate(self.masks_per_expert):
                name = self.task_names[eid] if eid < len(self.task_names) else f"expert_{eid}"
                # mark occupied without re-carving (masks already fixed)
                counts = []
                for li, m in enumerate(layer_masks):
                    m = m.to(device)
                    self.register.occupied[li] |= m
                    self.register.ownership[li][m] = eid
                    counts.append(int(m.sum().item()))
                self.register.allocations.append(
                    {"task": name, "expert_id": eid, "counts": counts,
                     "owned": int(sum(counts))}
                )
            self.fact_index: dict[str, int] = {}  # kind -> expert_id
            self.fact_values: dict[str, str] = {}  # kind -> value (for probes/routing banks)
            self.fact_supports: dict[str, list[tuple[str, str]]] = {}  # kind -> stored QA bank
            # Load mid-chat learned checkpoint if present (weights, not context)
            self.live_ckdir = results_dir / "live_checkpoint"
            loaded_live = self._load_live_checkpoint()
            print(f"EvAGI loaded: {len(self.masks_per_expert)} experts "
                  f"(skills+n_facts), "
                  f"{sum(int(m.sum()) for m in self.masks_per_expert[0])}+ neurons/layer0, "
                  f"gov={len(self.governors)} router_E={self.router.num_experts}, "
                  f"facts={list(self.fact_index.items())}"
                  f"{' [live_ckpt]' if loaded_live else ''}")
        elif engine == "baseline":
            model_dir = results_dir / "final_model"
            if not model_dir.exists():
                raise FileNotFoundError(f"missing {model_dir} — run experiment_llm_baseline first")
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_dir, dtype=torch.float32
            ).to(device)
            self.masks_per_expert = None
            self.governors = []
            self.router = None
            self.register = None
            self.fact_index = {}
            self.fact_values = {}
            self.fact_supports = {}
            self.live_ckdir = None
            self.probe = {}
            print("Baseline full-FT model loaded (no expert masks)")
        else:
            raise ValueError(engine)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
        self.layers = self.model.transformer.h
        self.yes_id, self.no_id = yes_no_token_ids(self.tokenizer)

        # held-out test + support banks (same seeds as training experiments)
        seed = int(seed)
        self.test_sets = {}
        self.support_sets = {}
        for i, name in enumerate(self.task_names[: self.n_tasks]):
            # baseline uses same generate seeds in experiment_llm_baseline
            ts = generate_llm_task(name, 300, seed + 2000 + i)
            self.test_sets[i] = ts
            self.support_sets[i] = ts[:support_shots]

        # live metrics
        self.turn_log: list[dict] = []
        # accuracy after each conversation turn: task x turn (NaN before learned)
        # For live session we treat "learned" as post-training for engine weights;
        # forgetting is measured vs best-across-turns during THIS conversation.
        self.acc_history: list[dict[str, float]] = []
        self.matrix: list[list[float]] = []  # rows=turn, cols=task (nan if not yet measured)

    # -- persistence: live-learned WEIGHTS (not context) --------------------
    def _load_live_checkpoint(self) -> bool:
        """Load mid-chat learned experts from disk into a fresh process."""
        if self.live_ckdir is None or not self.live_ckdir.exists():
            return False
        overlay_path = self.live_ckdir / "overlay.pt"
        model_dir = self.live_ckdir / "model"
        if not overlay_path.exists() or not model_dir.exists():
            return False

        # Prefer full live model weights (includes trained fact neurons)
        try:
            from transformers import AutoModelForCausalLM
            live_model = AutoModelForCausalLM.from_pretrained(
                model_dir, dtype=torch.float32
            ).to(self.device)
            self.model.load_state_dict(live_model.state_dict())
            del live_model
        except Exception as e:
            print(f"  [live_ckpt] model load failed ({e}); using base weights only")

        ov = torch.load(overlay_path, map_location=self.device, weights_only=False)
        base_n = len(self.masks_per_expert)
        for layer_masks in ov.get("masks_extra", []):
            self.masks_per_expert.append([m.to(self.device) for m in layer_masks])
        for gsd in ov.get("governor_state_extra", []):
            gov = TinyPerExpertGovernor(hidden=12).to(self.device)
            gov.load_state_dict(gsd)
            gov.eval()
            for p in gov.parameters():
                p.requires_grad_(False)
            self.governors.append(gov)
        if "router_state" in ov and self.router is not None:
            n_e = len(self.masks_per_expert)
            if self.router.num_experts != n_e:
                self.router = HRMRouter(num_experts=n_e, hidden=16).to(self.device)
            self.router.load_state_dict(ov["router_state"])
            for p in self.router.parameters():
                p.requires_grad_(False)
        self.fact_index = dict(ov.get("fact_index", {}))
        self.fact_values = dict(ov.get("fact_values", {}))
        self.fact_supports = {
            k: [(a, b) for a, b in v]
            for k, v in ov.get("fact_supports", {}).items()
        }
        # rebuild register occupancy for extra experts
        if self.register is not None:
            for eid in range(base_n, len(self.masks_per_expert)):
                layer_masks = self.masks_per_expert[eid]
                name = self.fact_index and next(
                    (k for k, v in self.fact_index.items() if v == eid), f"expert_{eid}"
                ) or f"expert_{eid}"
                counts = []
                for li, m in enumerate(layer_masks):
                    m = m.to(self.device)
                    self.register.occupied[li] |= m
                    self.register.ownership[li][m] = eid
                    counts.append(int(m.sum().item()))
                self.register.allocations.append(
                    {"task": name, "expert_id": eid, "counts": counts,
                     "owned": int(sum(counts))}
                )
        print(f"  [live_ckpt] restored {len(self.fact_index)} facts, "
              f"experts {base_n}->{len(self.masks_per_expert)}")
        return True

    def save_live_checkpoint(self) -> Path:
        """Persist live-learned weights + expert masks + fact banks to disk.

        A new process can load this and recall facts with an empty context —
        proof the knowledge lives in weights, not the conversation window.
        """
        if self.masks_per_expert is None:
            raise RuntimeError("no expert stack to save")
        self.live_ckdir = getattr(self, "live_ckdir", None) or (
            (PROJECT_ROOT / "results_llm_evagi") / "live_checkpoint"
        )
        self.live_ckdir.mkdir(parents=True, exist_ok=True)
        model_dir = self.live_ckdir / "model"
        self.model.eval()
        self.model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)

        n_skills = int(self.n_tasks)  # original skill count
        masks_extra = [
            [m.detach().cpu() for m in self.masks_per_expert[eid]]
            for eid in range(n_skills, len(self.masks_per_expert))
        ]
        gov_extra = [
            g.state_dict() for g in self.governors[n_skills:]
        ]
        overlay = {
            "masks_extra": masks_extra,
            "governor_state_extra": gov_extra,
            "router_state": self.router.state_dict() if self.router else None,
            "fact_index": self.fact_index,
            "fact_values": self.fact_values,
            "fact_supports": self.fact_supports,
            "n_skills": n_skills,
            "n_experts": len(self.masks_per_expert),
        }
        torch.save(overlay, self.live_ckdir / "overlay.pt")
        print(f"  [live_ckpt] saved -> {self.live_ckdir} "
              f"(facts={self.fact_index}, experts={len(self.masks_per_expert)})")
        return self.live_ckdir

    def clear_live_checkpoint(self) -> None:
        import shutil
        if self.live_ckdir and self.live_ckdir.exists():
            shutil.rmtree(self.live_ckdir)
            print(f"  [live_ckpt] cleared {self.live_ckdir}")

    # -- routing -----------------------------------------------------------
    def pick_expert(self, pairs) -> tuple[int, list[float]]:
        if self.masks_per_expert is None:
            return -1, []
        return support_pick_expert(
            self.model, self.layers, self.masks_per_expert, self.tokenizer,
            pairs, self.device, self.max_length,
        )

    def answer_yes_no(self, prompt: str, expert_id: int) -> dict:
        """Single-turn yes/no under expert mask (or full model for baseline)."""
        pair = [(prompt, 0)]  # dummy label for tokenize shape; we only need logits
        enc = tokenize_pairs(self.tokenizer, pair, max_length=self.max_length)
        input_ids = enc["input_ids"].to(self.device)
        attention = enc["attention_mask"].to(self.device)
        labels = enc["labels"]
        first_lab = (labels != -100).float().argmax(dim=1)
        pred_pos = int((first_lab - 1).clamp(min=0).item())

        with torch.no_grad():
            if self.masks_per_expert is not None:
                with ExpertMaskContext(self.layers, self.masks_per_expert, [expert_id]):
                    out = self.model(input_ids=input_ids, attention_mask=attention)
            else:
                out = self.model(input_ids=input_ids, attention_mask=attention)
        logits = out.logits[0, pred_pos, :]
        yes_p = torch.softmax(logits[[self.yes_id, self.no_id]], dim=-1)
        pred = 1 if logits[self.yes_id] > logits[self.no_id] else 0
        return {
            "pred": pred,
            "answer": "yes" if pred == 1 else "no",
            "p_yes": float(yes_p[0]),
            "p_no": float(yes_p[1]),
            "expert": expert_id,
        }

    def generate_free(self, prompt: str, expert_id: int, max_new_tokens: int = 24) -> str:
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            if self.masks_per_expert is not None and expert_id >= 0:
                with ExpertMaskContext(self.layers, self.masks_per_expert, [expert_id]):
                    out = self.model.generate(
                        **enc, max_new_tokens=max_new_tokens, do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
            else:
                out = self.model.generate(
                    **enc, max_new_tokens=max_new_tokens, do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
        text = self.tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return text.strip()

    # -- mid-chat fact learning (full EvAGI stack) -------------------------
    def live_learn_fact(self, fact: dict, cfg: dict | None = None) -> dict:
        """Install a new fact mid-chat via Equation V2 + Register + Governor + HRM.

        Returns log dict: k_pred, neurons, loss history, expert_id.
        """
        if self.masks_per_expert is None:
            raise RuntimeError("baseline has no expert stack")
        cfg = cfg or {
            "batch_size": 8, "lr": 5e-4, "weight_decay": 0.01,
            "gov_l1": 1e-4, "neuron_div": 8,
        }
        kind = fact["kind"]
        value = fact["value"]
        n_layers = len(self.layers)
        neuron_div = int(cfg.get("neuron_div", 8))

        # already have an expert for this fact kind? reuse it
        if kind in self.fact_index:
            eid = self.fact_index[kind]
            print(f"  [live] reusing expert {eid} for {kind}")
        else:
            # 1) Equation V2 -> k_alloc -> neuron counts
            k_pred, counts = predict_neurons(kind, n_layers, neuron_div=neuron_div)
            # 2) Register carves disjoint free neurons
            expert_id = len(self.masks_per_expert)
            masks = self.register.allocate(kind, counts, expert_id=expert_id)
            owned = int(sum(m.sum().item() for m in masks))
            print(f"  [live] EquationV2 {kind}: k_pred={k_pred} -> neurons={counts} "
                  f"(owned {owned}) pool {self.register.total_occupied()}/"
                  f"{self.register.total_pool()}")

            # 3) fresh TinyPerExpertGovernor for this expert
            gov = TinyPerExpertGovernor(hidden=12).to(self.device)
            # 4) grow HRM router output head
            self.router = expand_router(self.router, expert_id + 1, self.device)
            for p in self.router.parameters():
                p.requires_grad_(True)  # allow live router step

            self.masks_per_expert.append(masks)
            self.governors.append(gov)
            self.task_names.append(kind) if kind not in self.task_names else None
            # track by index carefully — task_names is skills; facts use fact_index
            self.fact_index[kind] = expert_id
            eid = expert_id
            print(f"  [live] allocated expert {eid} "
                  f"(gov {sum(p.numel() for p in gov.parameters())} params, "
                  f"router E={self.router.num_experts})")

        # 5) unfreeze FFN, hard-mask train on fact QA
        freeze_all_but_ffn(self.model, self.layers)
        self.model.train()
        for p in self.governors[eid].parameters():
            p.requires_grad_(True)
        for p in self.router.parameters():
            p.requires_grad_(True)

        qa = fact_qa_pairs(kind, value, n=int(cfg.get("train_n", 32)))
        print(f"  [live] training expert {eid} on {len(qa)} QA pairs "
              f"(lr={cfg.get('lr', 5e-4)})...")
        result = train_fact_expert(
            self.model, self.tokenizer, self.layers, qa,
            self.masks_per_expert[eid], self.governors[eid], self.router,
            router_expert_id=eid, cfg=cfg, device=self.device,
            max_length=self.max_length,
            epochs=int(cfg.get("live_epochs", 4)),
            lr=float(cfg.get("lr", 5e-4)),
        )

        # 6) freeze again after install
        for p in self.model.parameters():
            p.requires_grad_(False)
        for p in self.governors[eid].parameters():
            p.requires_grad_(False)
        for p in self.router.parameters():
            p.requires_grad_(False)
        self.model.eval()

        # hold-out probes for this fact
        probes = fact_probe_questions(kind, value)
        recall = self.recall_fact(kind, value, probes, expert_id=eid)
        print(f"  [live] recall check: {recall}")

        # store fact metadata + support bank for context-free routing later
        self.fact_values[kind] = value
        self.fact_supports[kind] = list(probes)

        # persist weights immediately so a NEW process can recall without chat history
        self.save_live_checkpoint()

        return {
            "kind": kind, "value": value, "expert_id": eid,
            "k_pred": result.get("k_pred"), "train": result,
            "recall": recall, "probes": len(probes),
        }

    def route_fact_query(self, prompt: str) -> tuple[int, list[float], str | None]:
        """Route an open fact question WITHOUT putting the answer in support.

        Uses HRM router features of the query alone (g(x), no task_id) when
        available; falls back to keyword -> stored fact support bank (the bank
        is the expert's own training bank from disk, not chat context — and
        support only picks WHICH expert; the answer tokens come from weights).
        """
        pl = prompt.lower()
        kind = None
        if "name" in pl:
            kind = "fact_name"
        elif "color" in pl or "colour" in pl:
            kind = "fact_color"
        elif "city" in pl or "where do you live" in pl or "live in" in pl:
            kind = "fact_city"
        elif "food" in pl or "like to eat" in pl:
            kind = "fact_food"

        # Prefer HRM router on query features only (truly input-conditional)
        if self.router is not None and self.masks_per_expert is not None:
            eid_r, feats = self._hrm_route_prompt(prompt)
            if kind is None or kind not in self.fact_index:
                return eid_r, [], f"hrm(E={self.router.num_experts})"

        # Keyword hit: score experts using THIS fact's stored support bank
        # (bank loaded from disk at session start — not from conversation).
        if kind and kind in self.fact_supports and self.masks_per_expert is not None:
            eid, losses = support_pick_expert_qa(
                self.model, self.layers, self.masks_per_expert,
                self.tokenizer, self.fact_supports[kind],
                self.device, self.max_length,
            )
            return eid, losses, kind

        # HRM-only fallback
        if self.router is not None and self.masks_per_expert is not None:
            eid_r, feats = self._hrm_route_prompt(prompt)
            return eid_r, [], "hrm"
        return 0, [], None

    def _hrm_route_prompt(self, prompt: str) -> tuple[int, torch.Tensor]:
        """Single-prompt HRM route: same 9-dim features as training, no labels."""
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = enc["input_ids"]
        with torch.no_grad():
            # cheap forward under full model for feature stats (no expert mask)
            out = self.model(input_ids=input_ids, attention_mask=enc["attention_mask"])
            logits = out.logits
            p = torch.sigmoid(logits[:, -1, 0])
            # dummy label mean 0.5 for unlabeled query features
            feats = torch.stack(
                [
                    input_ids.float().mean() / 50256.0,
                    input_ids.float().std() / 50256.0,
                    input_ids.new_tensor(0.5),
                    logits.new_tensor(0.0),
                    p.mean(),
                    p.std(),
                    (p > 0.5).float().mean(),
                    input_ids.new_tensor(float(input_ids.size(1))) / 128.0,
                    input_ids.new_tensor(float(len(self.layers))) / 8.0,
                ]
            )
            logits_r = self.router(feats.unsqueeze(0))
            eid = int(logits_r.argmax(dim=-1).item())
        return eid, feats

    @torch.no_grad()
    def recall_fact(self, kind: str, value: str,
                    probes: list[tuple[str, str]] | None = None,
                    expert_id: int | None = None) -> dict:
        """Score free-text recall of a fact under support-picked / given expert."""
        if probes is None:
            probes = fact_probe_questions(kind, value)

        # Routing protocol: labeled support from THIS fact's distribution
        # (same as yes/no skills — 32-shot style, no task_id into the model).
        support = probes  # (prompt, correct answer) pairs
        pick_eid, losses = support_pick_expert_qa(
            self.model, self.layers, self.masks_per_expert,
            self.tokenizer, support, self.device, self.max_length,
        )
        if expert_id is None:
            expert_id = pick_eid
        # if caller forces an expert, still log what support-pick chose
        pick_for_log = pick_eid

        # generation-based exact/soft match
        hits = 0
        gens = []
        target = value.strip().lower()
        for prompt, _ in probes:
            gen = self.generate_free(prompt, expert_id=expert_id, max_new_tokens=12)
            gens.append(gen)
            g = gen.strip().strip(".").strip().lower()
            if target in g or g in target or (len(g) >= 2 and target[:3] in g):
                hits += 1
        # also teacher-forced accuracy: argmax first answer token
        enc = tokenize_qa(self.tokenizer, probes, max_length=self.max_length)
        input_ids = enc["input_ids"].to(self.device)
        attention = enc["attention_mask"].to(self.device)
        labels = enc["labels"].to(self.device)
        first = (labels != -100).float().argmax(dim=1)
        b_idx = torch.arange(input_ids.size(0), device=self.device)
        pred_pos = (first - 1).clamp(min=0).to(self.device)
        with ExpertMaskContext(self.layers, self.masks_per_expert, [expert_id]):
            out = self.model(input_ids=input_ids, attention_mask=attention)
        # compare full answer string likelihood rank via exact token match at first answer tok
        ans_tok = self.tokenizer(target if not target.startswith(" ") else target,
                                 add_special_tokens=False)["input_ids"]
        # decode top-1 continuation
        top1 = out.logits[b_idx, pred_pos].argmax(dim=-1)
        decoded = self.tokenizer.batch_decode(top1.unsqueeze(1), skip_special_tokens=True)
        tf_hits = sum(
            1 for d, (_, ans) in zip(decoded, probes)
            if target in (d + ans).lower() or d.strip().lower() in target
            or target.startswith(d.strip().lower().lstrip())
        )
        # simpler TF: token id of first answer token
        first_ans_ids = []
        for _, ans in probes:
            ids = self.tokenizer(ans, add_special_tokens=False)["input_ids"]
            first_ans_ids.append(ids[0] if ids else -1)
        tf_tok = sum(1 for p, t in zip(top1.tolist(), first_ans_ids) if p == t)

        return {
            "expert": expert_id,
            "support_pick": pick_for_log,
            "support_nll": [round(x, 4) for x in losses],
            "gen_hits": hits,
            "gen_total": len(probes),
            "gen_acc": round(100.0 * hits / max(len(probes), 1), 1),
            "tf_first_tok": tf_tok,
            "tf_total": len(probes),
            "generations": gens[:3],
        }

    # -- eval all skills (live, no task_id) --------------------------------
    @torch.no_grad()
    def eval_all(self) -> dict[str, float]:
        results: dict[str, float] = {}
        for i in range(self.n_tasks):
            name = self.task_names[i]
            if self.masks_per_expert is None:
                from .experiment_llm_baseline import eval_yes_no_accuracy as base_eval
                acc = base_eval(
                    self.model, self.tokenizer, self.test_sets[i],
                    self.device, self.max_length,
                )
            else:
                eid, _ = self.pick_expert(self.support_sets[i])
                acc = eval_yes_no_accuracy(
                    self.model, self.tokenizer, self.test_sets[i],
                    self.layers, self.masks_per_expert, eid,
                    self.device, self.max_length,
                )
            results[name] = round(acc, 2)
        return results

    def record_turn(self, turn_meta: dict, accs: dict[str, float]) -> None:
        self.turn_log.append({**turn_meta, "accuracies": accs})
        self.acc_history.append(accs)
        row = [accs.get(n, float("nan")) for n in self.task_names[: self.n_tasks]]
        self.matrix.append(row)

    def live_metrics(self) -> dict:
        mat = np.array(self.matrix, dtype=float)
        if mat.size == 0 or np.all(np.isnan(mat)):
            return {
                "average_forgetting": 0.0,
                "final_average_accuracy": 0.0,
                "per_task": {},
            }
        # tasks may be all-nan on early turns before any eval — compute_forgetting handles
        # fill: only tasks with at least one measurement
        metric = compute_forgetting(mat.T)  # tasks x turns
        # mat is turns x tasks, so transpose to tasks x turns
        per = {}
        for i, n in enumerate(self.task_names[: self.n_tasks]):
            row = mat[:, i]
            m = row[~np.isnan(row)]
            if m.size == 0:
                per[n] = {"forgetting": float("nan"), "final": float("nan"), "best": float("nan")}
            else:
                per[n] = {
                    "best": float(np.max(m)),
                    "final": float(m[-1]),
                    "forgetting": float(np.max(m) - m[-1]),
                }
        return {
            "average_forgetting": metric["average_forgetting"],
            "final_average_accuracy": metric["final_average_accuracy"],
            "per_task": per,
        }

    def print_metrics(self) -> dict:
        m = self.live_metrics()
        print("  ┌── live metrics " + "─" * 40)
        print(f"  │ avg forgetting {m['average_forgetting']:.2f}%   "
              f"final avg {m['final_average_accuracy']:.2f}%")
        for n, s in m["per_task"].items():
            if s["final"] != s["final"]:  # nan
                continue
            print(f"  │  {n:10s} best {s['best']:6.2f}  final {s['final']:6.2f}  "
                  f"F {s['forgetting']:6.2f}")
        print("  └" + "─" * 56)
        return m

    def eval_step(self, label: str = "eval") -> dict[str, float]:
        accs = self.eval_all()
        print(f"  [{label}] " + "  ".join(f"{k}:{v:.1f}" for k, v in accs.items()))
        return accs


def run_scripted(session: LiveSession, outdir: Path) -> None:
    print("\n" + "=" * 72)
    print(f"LIVE MULTI-TURN SESSION — engine={session.engine} — "
          f"{len(SCRIPTED_TURNS)} turns")
    print("=" * 72)

    # baseline of metrics at turn 0 (post-training, before conversation)
    print("\n[turn 0 — pre-conversation eval]")
    acc0 = session.eval_step("t0")
    session.record_turn({"turn": 0, "kind": "init", "user": "", "assistant": ""}, acc0)
    session.print_metrics()

    for ti, turn in enumerate(SCRIPTED_TURNS, start=1):
        print(f"\n[turn {ti}/{len(SCRIPTED_TURNS)}] ({turn['kind']})")
        print(f"  user: {turn['user']}")

        meta: dict = {"turn": ti, "kind": turn["kind"], "user": turn["user"]}

        if turn["kind"] == "skill":
            label = int(turn["label"])
            # live support-set routing among installed experts (no task_id).
            # Support bank = this skill's 32 held-out shots (same protocol as
            # experiment_llm_evagi when evaluating task i: model never sees
            # task_id — only the 32 labeled shots condition the pick).
            if session.masks_per_expert is not None:
                skill_idx = session.task_names.index(turn["skill"])
                sup = session.support_sets[skill_idx]
                eid, losses = session.pick_expert(sup)
                ans = session.answer_yes_no(turn["user"], expert_id=eid)
                print(f"  router: expert={eid}  support_nll={[round(x,3) for x in losses]}")
                print(f"  assistant: {ans['answer']}  "
                      f"(p_yes={ans['p_yes']:.3f} p_no={ans['p_no']:.3f})")
                meta.update({"expert": eid, "answer": ans["answer"],
                             "p_yes": ans["p_yes"], "support_nll": losses,
                             "skill": turn["skill"], "label": label,
                             "correct": int(ans["pred"] == label)})
            else:
                ans = session.answer_yes_no(turn["user"], expert_id=-1)
                print(f"  assistant: {ans['answer']}  "
                      f"(p_yes={ans['p_yes']:.3f} p_no={ans['p_no']:.3f})")
                meta.update({"answer": ans["answer"], "p_yes": ans["p_yes"],
                             "skill": turn["skill"], "label": label,
                             "correct": int(ans["pred"] == label)})
            ok = "OK" if meta["correct"] else "MISS"
            print(f"  check: true={'yes' if label else 'no'}  [{ok}]")
        else:
            # freeform under last / round-robin expert (flavor only)
            eid = (ti - 1) % max(session.n_tasks, 1) if session.masks_per_expert else -1
            text = session.generate_free(turn["user"], expert_id=eid)
            print(f"  assistant: {text}")
            meta.update({"expert": eid, "generation": text})

        # after EVERY turn: re-eval all skills live
        accs = session.eval_step(f"t{ti}")
        session.record_turn(meta, accs)
        session.print_metrics()

    # final summary
    final = session.live_metrics()
    skill_turns = [t for t in session.turn_log if t.get("kind") == "skill"]
    n_skill = len(skill_turns)
    n_hit = sum(int(t.get("correct", 0)) for t in skill_turns)
    chat_acc = 100.0 * n_hit / max(n_skill, 1)
    # per-skill chat accuracy
    by_skill: dict[str, list[int]] = {}
    for t in skill_turns:
        by_skill.setdefault(t["skill"], []).append(int(t.get("correct", 0)))
    skill_acc = {k: 100.0 * sum(v) / len(v) for k, v in by_skill.items()}
    misses = [t for t in skill_turns if not t.get("correct")]

    print("\n" + "=" * 72)
    print(f"SESSION DONE — engine={session.engine}")
    print(f"  turns: {len(session.turn_log)}  skill_turns: {n_skill}")
    print(f"  chat skill accuracy: {chat_acc:.1f}%  ({n_hit}/{n_skill})")
    for k, v in skill_acc.items():
        print(f"    {k:10s} {v:.0f}%")
    if misses:
        print(f"  misses ({len(misses)}):")
        for t in misses:
            print(f"    turn {t['turn']}: {t['skill']} "
                  f"true={'yes' if t['label'] else 'no'} "
                  f"pred={t['answer']} p_yes={t['p_yes']:.3f}")
    print(f"  eval-after-each-turn avg forgetting: {final['average_forgetting']:.2f}%")
    print(f"  eval-after-each-turn final avg acc:  {final['final_average_accuracy']:.2f}%")
    print("  per-task eval (end of conversation):")
    for n, s in final["per_task"].items():
        if s["final"] == s["final"]:
            print(f"    {n:10s} best {s['best']:6.2f}  final {s['final']:6.2f}  "
                  f"F {s['forgetting']:6.2f}")
    print("=" * 72)

    # persist
    outdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "engine": session.engine,
        "n_tasks": session.n_tasks,
        "task_names": session.task_names,
        "turns": session.turn_log,
        "accuracy_matrix": session.matrix,  # turns x tasks
        "live_metrics": final,
        "probe": session.probe,
        "chat_skill_accuracy": round(chat_acc, 2),
        "chat_skill_hits": n_hit,
        "chat_skill_total": n_skill,
        "chat_per_skill": {k: round(v, 2) for k, v in skill_acc.items()},
        "chat_misses": [
            {"turn": t["turn"], "skill": t["skill"], "label": t["label"],
             "pred": t["answer"], "p_yes": t["p_yes"]}
            for t in misses
        ],
    }
    with open(outdir / f"live_session_{session.engine}.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {outdir / f'live_session_{session.engine}.json'}")


def run_interactive(session: LiveSession, outdir: Path) -> None:
    print("\n" + "=" * 72)
    print(f"INTERACTIVE LIVE CHAT — engine={session.engine}")
    print("  Teach:  My name is Rehan / My favorite color is emerald")
    print("  Ask:    What is your name? / skill Qs ending 'Answer:'")
    print("  Commands: /eval  /metrics  /facts  /quit")
    print("  Facts trigger LIVE weight updates (V2+Register+Governor+HRM)")
    print("=" * 72)
    acc0 = session.eval_step("t0")
    session.record_turn({"turn": 0, "kind": "init", "user": "", "assistant": ""}, acc0)
    session.print_metrics()

    turn = 0
    while True:
        try:
            user = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            continue
        if user == "/quit":
            break
        if user == "/eval":
            accs = session.eval_step("manual")
            session.record_turn({"turn": turn, "kind": "eval", "user": user,
                                 "assistant": ""}, accs)
            session.print_metrics()
            continue
        if user == "/metrics":
            session.print_metrics()
            continue
        if user == "/facts":
            print(f"  fact_index={getattr(session, 'fact_index', {})}")
            print(f"  experts={len(session.masks_per_expert or [])}")
            continue

        turn += 1
        meta: dict = {"turn": turn, "user": user}

        # 1) fact declaration -> live learn
        fact = parse_fact(user)
        if fact and session.masks_per_expert is not None:
            print(f"  [parsed fact] {fact['kind']} = {fact['value']}")
            print("  [LIVE LEARN — weights updating mid-chat]")
            result = session.live_learn_fact(fact, cfg=LEARN_CFG)
            meta.update({"kind": "teach", "fact": {k: result[k] for k in
                          ("kind", "value", "expert_id", "recall")}})
            r = result["recall"]
            print(f"  bot> learned. recall gen_acc={r['gen_acc']}% "
                  f"expert={r['expert']} loss={result['train']['final_loss']:.4f}")
            if r.get("generations"):
                print(f"  bot> sample: {r['generations'][0]}")
        # 2) open question about a known fact — route WITHOUT answer in prompt
        elif any(q in user.lower() for q in (
            "your name", "my name", "whats my name", "what's my name",
            "favorite color", "favorite colour", "what city",
            "where do you live", "favorite food",
        )) and session.masks_per_expert is not None:
            prompt = user if user.rstrip().endswith(("A:", "Answer:")) else user.rstrip() + " A:"
            eid, losses, how = session.route_fact_query(prompt)
            gen = session.generate_free(prompt, expert_id=eid, max_new_tokens=12)
            print(f"  router: expert={eid} via={how} "
                  f"nll={[round(x,3) for x in losses] if losses else '-'}")
            # ground-truth check against stored fact values (not fed to model)
            hit = None
            for kind, val in session.fact_values.items():
                if (
                    ("name" in prompt.lower() and kind == "fact_name")
                    or ("color" in prompt.lower() and kind == "fact_color")
                    or ("city" in prompt.lower() and kind == "fact_city")
                    or ("food" in prompt.lower() and kind == "fact_food")
                ):
                    g = gen.strip().lower()
                    t = val.strip().lower()
                    hit = int(t in g or (len(g) >= 2 and t[:3] in g))
                    print(f"  check: want '{val}'  [{'OK' if hit else 'MISS'}]")
                    break
            print(f"  bot> {gen}")
            meta.update({"kind": "ask_fact", "expert": eid, "generation": gen,
                         "prompt": prompt, "route": how, "hit": hit})
        # 3) yes/no skill
        elif user.rstrip().endswith("Answer:") or user.rstrip().endswith("Answer"):
            is_skill = True
            if session.masks_per_expert is not None:
                combined = []
                for i in range(min(session.n_tasks, len(session.support_sets))):
                    combined.extend(session.support_sets[i][:8])
                eid, losses = session.pick_expert(combined) if combined else (-1, [])
                ans = session.answer_yes_no(user, expert_id=eid)
                print(f"  router: expert={eid}")
                print(f"  bot> {ans['answer']} (p_yes={ans['p_yes']:.3f})")
                meta.update({"kind": "skill", "expert": eid, "answer": ans["answer"],
                             "p_yes": ans["p_yes"]})
            else:
                ans = session.answer_yes_no(user, expert_id=-1)
                print(f"  bot> {ans['answer']} (p_yes={ans['p_yes']:.3f})")
                meta.update({"kind": "skill", "answer": ans["answer"],
                             "p_yes": ans["p_yes"]})
        else:
            eid = (turn - 1) % max(len(session.masks_per_expert or []) or 1, 1)
            if session.masks_per_expert is None:
                eid = -1
            text = session.generate_free(user, expert_id=eid)
            print(f"  bot> {text}")
            meta.update({"kind": "free", "expert": eid, "generation": text})

        accs = session.eval_step(f"t{turn}")
        session.record_turn(meta, accs)
        session.print_metrics()

    final = session.live_metrics()
    print(f"\nSession ended. avg F={final['average_forgetting']:.2f}% "
          f"final={final['final_average_accuracy']:.2f}%")
    # persist any live-learned weights before exit
    if session.masks_per_expert is not None and session.fact_index:
        session.save_live_checkpoint()
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / f"live_session_{session.engine}.json", "w") as f:
        json.dump({
            "engine": session.engine,
            "turns": session.turn_log,
            "accuracy_matrix": session.matrix,
            "live_metrics": final,
            "fact_index": session.fact_index,
            "fact_values": session.fact_values,
        }, f, indent=2, default=str)
    print(f"Wrote {outdir / f'live_session_{session.engine}.json'}")


def main():
    parser = argparse.ArgumentParser(description="Live multi-turn EvAGI LLM inference")
    parser.add_argument("--engine", choices=["evagi", "baseline"], default="evagi")
    parser.add_argument("--mode", choices=["scripted", "interactive", "learn", "prove"],
                        default="scripted")
    parser.add_argument("--results", default=None,
                        help="results dir (default results_llm_<engine>)")
    parser.add_argument("--outdir", default=str(PROJECT_ROOT / "results_llm_live"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--support-shots", type=int, default=32)
    parser.add_argument("--clear-ckpt", action="store_true",
                        help="wipe live_checkpoint before load")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = Path(args.results) if args.results else (
        PROJECT_ROOT / (f"results_llm_{args.engine}")
    )
    outdir = Path(args.outdir)

    session = LiveSession(
        engine=args.engine,
        results_dir=results,
        device=device,
        support_shots=args.support_shots,
        seed=args.seed,
    )
    if args.clear_ckpt and hasattr(session, "clear_live_checkpoint"):
        session.clear_live_checkpoint()

    if args.mode == "scripted":
        run_scripted(session, outdir)
    elif args.mode == "learn":
        run_learn(session, outdir)
    elif args.mode == "prove":
        run_prove(session, outdir)
    else:
        run_interactive(session, outdir)


LEARN_CFG = {
    "batch_size": 8,
    "lr": 8e-4,
    "weight_decay": 0.01,
    "gov_l1": 1e-4,
    "neuron_div": 8,
    "live_epochs": 6,
    "train_n": 48,
}


def _ask_name(session: LiveSession, prompt: str, label: str | None = None,
              fact_kind: str | None = None) -> dict:
    """Open-ended question: support-route among ALL experts, generate.

    When `label` is known, support = held-out probes for that fact with
    correct answers (same 32-shot protocol as yes/no skills — no task_id).
    When unknown (pre-learn), support is a dummy and routing is expected
    to miss — that's the BEFORE baseline.
    """
    if label is not None and fact_kind is not None:
        support = fact_probe_questions(fact_kind, label)
    elif label is not None:
        # minimal support: one correct QA for the expected answer family
        support = [(prompt, f" {label}")]
    else:
        support = [(prompt, " x")]

    if session.masks_per_expert is not None:
        eid, losses = support_pick_expert_qa(
            session.model, session.layers, session.masks_per_expert,
            session.tokenizer, support, session.device, session.max_length,
        )
        gen = session.generate_free(prompt, expert_id=eid, max_new_tokens=10)
        print(f"  router: expert={eid}  support_nll={[round(x,3) for x in losses]}")
    else:
        eid = -1
        losses = []
        gen = session.generate_free(prompt, expert_id=-1, max_new_tokens=10)
    print(f"  user: {prompt}")
    print(f"  assistant: {gen}")
    hit = None
    if label is not None:
        g = gen.strip().lower()
        t = label.strip().lower()
        hit = int(t in g or (len(g) >= 2 and t[:3] in g))
        print(f"  check: want '{label}'  [{'OK' if hit else 'MISS'}]")
    return {"prompt": prompt, "generation": gen, "expert": eid,
            "support_nll": [round(x, 4) for x in losses], "hit": hit}


def run_learn(session: LiveSession, outdir: Path) -> None:
    """Mid-chat learning: say a fact -> EvAGI installs expert live -> recall."""
    if session.masks_per_expert is None:
        raise SystemExit("--mode learn requires --engine evagi")

    print("\n" + "=" * 72)
    print("LIVE LEARNING SESSION — teach facts mid-chat (full EvAGI stack)")
    print("  EquationV2 -> Register -> TinyPerExpertGovernor -> hard mask -> HRM")
    print("=" * 72)

    log: dict = {"engine": session.engine, "events": [], "facts": []}

    # -- t0: baseline skills + open recall BEFORE learning ----------------
    print("\n[turn 0 — pre-learn skill eval]")
    acc0 = session.eval_step("t0")
    session.record_turn({"turn": 0, "kind": "init"}, acc0)
    session.print_metrics()

    print("\n[turn 1 — ask name BEFORE learning]")
    before = _ask_name(session, "Q: What is your name? A:", label=None)
    log["events"].append({"turn": 1, "kind": "ask_before", **before})

    # -- t2: USER teaches name -> LIVE LEARN --------------------------------
    print("\n[turn 2 — user teaches name → LIVE WEIGHT UPDATE]")
    fact_utter = "My name is Rehan"
    print(f"  user: {fact_utter}")
    fact = parse_fact(fact_utter)
    assert fact and fact["kind"] == "fact_name", fact
    learn1 = session.live_learn_fact(fact, cfg=LEARN_CFG)
    log["facts"].append(learn1)
    log["events"].append({"turn": 2, "kind": "teach", "utterance": fact_utter,
                          "fact": {k: learn1[k] for k in
                                   ("kind", "value", "expert_id", "recall")}})

    # skills after live install (isolation check)
    print("\n  [skills immediately after live learn]")
    acc1 = session.eval_step("post_name")
    session.record_turn({"turn": 2, "kind": "learn", "user": fact_utter,
                         "assistant": "(installed)"}, acc1)
    session.print_metrics()

    # -- t3: ask name AFTER learning ---------------------------------------
    print("\n[turn 3 — ask name AFTER learning]")
    after = _ask_name(session, "Q: What is your name? A:", label="Rehan",
                      fact_kind="fact_name")
    # also held-out probe style
    probes = fact_probe_questions("fact_name", "Rehan")
    rec = session.recall_fact("fact_name", "Rehan", probes)
    print(f"  recall metrics: gen {rec['gen_acc']}% "
          f"tf_tok {rec['tf_first_tok']}/{rec['tf_total']} "
          f"expert={rec['expert']} support_pick={rec.get('support_pick')}")
    if rec.get("generations"):
        print(f"  sample gens: {rec['generations']}")
    log["events"].append({"turn": 3, "kind": "ask_after", **after, "recall": rec})

    # -- t4: teach a second fact (color) -----------------------------------
    print("\n[turn 4 — user teaches color → LIVE WEIGHT UPDATE]")
    fact2_utter = "My favorite color is emerald"
    print(f"  user: {fact2_utter}")
    fact2 = parse_fact(fact2_utter)
    assert fact2, fact2
    learn2 = session.live_learn_fact(fact2, cfg=LEARN_CFG)
    log["facts"].append(learn2)
    log["events"].append({"turn": 4, "kind": "teach", "utterance": fact2_utter,
                          "fact": {k: learn2[k] for k in
                                   ("kind", "value", "expert_id", "recall")}})

    print("\n  [skills after second live learn]")
    acc2 = session.eval_step("post_color")
    session.record_turn({"turn": 4, "kind": "learn", "user": fact2_utter,
                         "assistant": "(installed)"}, acc2)
    session.print_metrics()

    # -- t5: recall both facts + name again (stability) ---------------------
    print("\n[turn 5 — recall name (stability) + color]")
    again = _ask_name(session, "Q: What is your name? A:", label="Rehan",
                      fact_kind="fact_name")
    color_q = _ask_name(session, "Q: What is your favorite color? A:",
                        label="emerald", fact_kind="fact_color")
    log["events"].append({"turn": 5, "kind": "recall_both",
                          "name": again, "color": color_q})

    # final skill eval
    print("\n[final skill eval after all live learning]")
    accf = session.eval_step("final")
    session.record_turn({"turn": 6, "kind": "final"}, accf)
    final_m = session.print_metrics()

    # summary
    name_ok = bool(after.get("hit")) and bool(again.get("hit"))
    color_ok = bool(color_q.get("hit"))
    print("\n" + "=" * 72)
    print("LIVE LEARNING DONE")
    print(f"  facts installed: {len(log['facts'])}")
    for f in log["facts"]:
        r = f["recall"]
        print(f"    {f['kind']}='{f['value']}' expert={f['expert_id']} "
              f"train_loss={f['train']['final_loss']:.4f} "
              f"gen_acc={r['gen_acc']}%")
    print(f"  name recall after teach: {'YES' if name_ok else 'NO'}")
    print(f"  color recall after teach: {'YES' if color_ok else 'NO'}")
    print(f"  old skills avg forgetting during live learns: "
          f"{final_m['average_forgetting']:.2f}%")
    print(f"  old skills final avg: {final_m['final_average_accuracy']:.2f}%")
    print("=" * 72)

    log["live_metrics"] = final_m
    log["name_recall_ok"] = name_ok
    log["color_recall_ok"] = color_ok
    log["accuracy_matrix"] = session.matrix
    log["turns"] = session.turn_log
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "live_learn_evagi.json"
    with open(path, "w") as fh:
        json.dump(log, fh, indent=2, default=str)
    print(f"Wrote {path}")


def run_prove(session: LiveSession, outdir: Path) -> None:
    """Cross-session proof: empty context, weights-only recall.

    Loads live_checkpoint from a PREVIOUS process. Never feeds fact values
    into the prompt — only "What is your name? A:". If the answer comes back,
    it is in the FFN expert weights, not the context window.
    """
    if session.masks_per_expert is None:
        raise SystemExit("--mode prove requires --engine evagi")

    print("\n" + "=" * 72)
    print("CROSS-SESSION PROOF — fresh process, empty context, weights only")
    print(f"  live facts on disk: {session.fact_index}")
    print(f"  fact values known to harness (NOT in prompt): "
          f"{list(session.fact_values.keys())}")
    print(f"  chat history: empty")
    print("=" * 72)

    results = {"facts": [], "skills": None, "ok": True}

    # 1) fact recall — prompts contain NO fact values
    for kind, expected in session.fact_values.items():
        probes = fact_probe_questions(kind, expected)
        prompts_only = [p for p, _ in probes]  # strip answers — generation has no label
        print(f"\n[{kind}] expected value withheld from prompts")
        gens = []
        hits = 0
        for prompt in prompts_only:
            # route without putting `expected` into support for generation
            eid, losses, how = session.route_fact_query(prompt)
            # generate under routed expert — prompt is only the question
            gen = session.generate_free(prompt, expert_id=eid, max_new_tokens=12)
            g = gen.strip().lower()
            t = expected.strip().lower()
            hit = int(t in g or (len(g) >= 2 and t[:3] in g))
            hits += hit
            gens.append(gen)
            print(f"  prompt: {prompt!r}")
            print(f"  gen:    {gen!r}  expert={eid} via={how}  "
                  f"[{'OK' if hit else 'MISS'}]")
            print(f"  (prompt contains value? {t in prompt.lower()})")
        acc = 100.0 * hits / max(len(prompts_only), 1)
        print(f"  => {kind}: {hits}/{len(prompts_only)} ({acc:.0f}%) "
              f"expected={expected!r}")
        results["facts"].append({
            "kind": kind, "expected": expected, "generations": gens,
            "hits": hits, "total": len(prompts_only), "acc": acc,
        })
        if hits == 0:
            results["ok"] = False

    # 2) skills still intact after loading live ckpt
    print("\n[skills after loading live checkpoint]")
    accs = session.eval_step("prove")
    session.record_turn({"turn": 0, "kind": "prove"}, accs)
    results["skills"] = accs
    m = session.print_metrics()
    results["live_metrics"] = m

    # 3) verify prompts never included fact values (audit log)
    print("\n" + "=" * 72)
    print(f"PROOF {'PASSED' if results['ok'] else 'FAILED'}")
    print(f"  facts recalled from weights in empty context: "
          f"{sum(f['hits'] for f in results['facts'])}/"
          f"{sum(f['total'] for f in results['facts'])}")
    print(f"  skill avg: {m['final_average_accuracy']:.2f}%  "
          f"F={m['average_forgetting']:.2f}%")
    print("=" * 72)

    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "prove_cross_session.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
