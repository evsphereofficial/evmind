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
from .llm_evagi import ExpertMaskContext, support_pick_expert
from .llm_tasks import (
    LLM_TASK_NAMES,
    generate_llm_task,
    tokenize_pairs,
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
            print(f"EvAGI loaded: {self.n_tasks} experts, "
                  f"{sum(int(m.sum()) for m in self.masks_per_expert[0])}+ neurons/layer0")
        elif engine == "baseline":
            model_dir = results_dir / "final_model"
            if not model_dir.exists():
                raise FileNotFoundError(f"missing {model_dir} — run experiment_llm_baseline first")
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_dir, dtype=torch.float32
            ).to(device)
            self.masks_per_expert = None
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
    print("  type a yes/no question ending with 'Answer:' or free text")
    print("  commands: /eval  /metrics  /quit")
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

        turn += 1
        is_skill = user.rstrip().endswith("Answer:") or user.rstrip().endswith("Answer")
        meta: dict = {"turn": turn, "kind": "skill" if is_skill else "free",
                      "user": user}
        if is_skill and session.masks_per_expert is not None:
            # route with union of all support banks: pick expert that minimizes
            # mean NLL across every skill's support (fully task-agnostic),
            # then answer. For clearer skill isolation demo, prefer matching
            # skill if prompt content maps — but routing itself stays g(x).
            # Protocol A (default): per-skill support not available without
            # knowing skill; use combined support from all banks.
            combined = []
            for i in range(session.n_tasks):
                combined.extend(session.support_sets[i][:8])  # 8x5=40 shots
            eid, losses = session.pick_expert(combined)
            ans = session.answer_yes_no(user, expert_id=eid)
            print(f"  router: expert={eid}")
            print(f"  bot> {ans['answer']} (p_yes={ans['p_yes']:.3f})")
            meta.update({"expert": eid, "answer": ans["answer"],
                         "p_yes": ans["p_yes"]})
        else:
            eid = (turn - 1) % max(session.n_tasks, 1) if session.masks_per_expert else -1
            if is_skill:
                ans = session.answer_yes_no(user, expert_id=-1)
                print(f"  bot> {ans['answer']} (p_yes={ans['p_yes']:.3f})")
                meta.update({"answer": ans["answer"], "p_yes": ans["p_yes"]})
            else:
                text = session.generate_free(user, expert_id=eid)
                print(f"  bot> {text}")
                meta.update({"expert": eid, "generation": text})

        accs = session.eval_step(f"t{turn}")
        session.record_turn(meta, accs)
        session.print_metrics()

    final = session.live_metrics()
    print(f"\nSession ended. avg F={final['average_forgetting']:.2f}% "
          f"final={final['final_average_accuracy']:.2f}%")
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / f"live_session_{session.engine}.json", "w") as f:
        json.dump({
            "engine": session.engine,
            "turns": session.turn_log,
            "accuracy_matrix": session.matrix,
            "live_metrics": final,
        }, f, indent=2)
    print(f"Wrote {outdir / f'live_session_{session.engine}.json'}")


def main():
    parser = argparse.ArgumentParser(description="Live multi-turn EvAGI LLM inference")
    parser.add_argument("--engine", choices=["evagi", "baseline"], default="evagi")
    parser.add_argument("--mode", choices=["scripted", "interactive"], default="scripted")
    parser.add_argument("--results", default=None,
                        help="results dir (default results_llm_<engine>)")
    parser.add_argument("--outdir", default=str(PROJECT_ROOT / "results_llm_live"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--support-shots", type=int, default=32)
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

    if args.mode == "scripted":
        run_scripted(session, outdir)
    else:
        run_interactive(session, outdir)


if __name__ == "__main__":
    main()
