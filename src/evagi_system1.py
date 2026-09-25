"""System-1 decision model (Laya fine-tuned on EvAGI routing).

Single source of truth for the decision schema: training, evaluation and the
live app must build byte-identical states/questions or the classifier drifts.

Architecture (paper §5.2 behavior stack):
  System 1 (this module, ~25-35ms): route the turn — learn / answer from
  memory / admit ignorance / answer from base / chitchat + which topic.
  System 2 (LFM2.5): extract kind|value on learn; generate answers, think,
  solve problems.  No regexes in the routing path.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

# Where the fine-tuned checkpoint lives (created by src/train_laya_evagi.py).
LAYA_DIR = os.environ.get(
    "EVAGI_LAYA_DIR",
    os.path.join(os.path.dirname(__file__), "..", "models", "laya-evagi"),
)

# Fallback to the System-2 LLM router when System-1 confidence is below this.
CONF_FALLBACK = 0.55

ACTION_CHOICES = {
    "learn": (
        "user is teaching something to remember — a statement or command "
        "sharing information, not information-seeking"
    ),
    "answer_from_memory": (
        "message asks about or follows up on a topic already in known_memory"
    ),
    "admit_ignorance": (
        "personal or specific question expecting an answer from memory, but "
        "the target is NOT in known_memory — admit ignorance and offer to be "
        "taught"
    ),
    "answer_general": (
        "open-world question answerable from general knowledge — facts, "
        "how-to, reasoning, explanations, advice"
    ),
    "chitchat": (
        "greeting, thanks, joke, meta-talk, small talk — no substantive "
        "request"
    ),
}

FACT_KIND_CHOICES = {
    "personal_fact": (
        "a factoid about the user's life — names, dates, places, "
        "preferences, codes, relationships"
    ),
    "skill_code": "a rule, procedure or how-to the assistant must follow later",
    "knowledge": "a fact, definition or news about the world or an entity",
    "content": "a longer passage, description or document to store",
    "none": "not a teaching turn",
}

NO_TOPIC = "about_something_else"
TOPIC_CHOICES_NOTE = (
    "Which topic in known_memory does this message refer to? "
    f"Choose {NO_TOPIC} if none."
)
ACTION_NOTE = (
    "Decide what the assistant should do with user_message, "
    "given known_memory (topics already learned)."
)
FACT_KIND_NOTE = "If the user is teaching, what kind of thing are they teaching?"

MAX_STATE_TOPICS = 10


def build_state(text: str, known_memory: str) -> dict:
    # Key order matters: serialize_state is json.dumps (insertion order).
    return {"known_memory": known_memory or "none", "user_message": text}


def build_questions(topics: list[str] | None) -> dict:
    qs = {
        "action": {
            "type": "choice",
            "instructions": ACTION_NOTE,
            "criteria": dict(ACTION_CHOICES),
        },
        "fact_kind": {
            "type": "choice",
            "instructions": FACT_KIND_NOTE,
            "criteria": dict(FACT_KIND_CHOICES),
        },
    }
    seen: list[str] = []
    for t in topics or []:
        t = (t or "").strip()[:80]
        if t and t not in seen:
            seen.append(t)
        if len(seen) >= MAX_STATE_TOPICS - 1:
            break
    if seen:
        crit = {t: "" for t in seen}
        crit[NO_TOPIC] = ""
        qs["topic"] = {
            "type": "choice",
            "instructions": TOPIC_CHOICES_NOTE,
            "criteria": crit,
        }
    return qs


def known_memory_from_items(items: dict) -> tuple[str, list[str], dict[str, int]]:
    """Compact known_memory string + topic titles + topic→expert_id map.

    Titles are truncated exactly like build_questions does (strip[:80]) so
    the classifier's chosen option always keys the map.
    """
    parts: list[str] = []
    titles: list[str] = []
    topic2eid: dict[str, int] = {}
    for m in items.values():
        kind = m.get("kind") or ""
        val = str(m.get("value") or "")[:40]
        title = str(m.get("title") or "").strip()
        if kind.startswith("fact_"):
            label = kind[5:].replace("_", " ")
            part = f"{label}: {val}"
            topic = title or label
        elif kind == "skill_code":
            part = f"how to {val}"
            topic = title or f"how to {val}"
        else:
            part = title or val
            topic = title or val
        if not part:
            continue
        parts.append(part)
        key = topic.strip()[:80]
        titles.append(key)
        if key not in topic2eid:
            topic2eid[key] = int(m.get("expert_id"))
        if len(parts) >= MAX_STATE_TOPICS:
            break
    return "; ".join(parts) if parts else "none", titles, topic2eid


@dataclass
class System1Decision:
    action: str                 # learn | answer_from_memory | admit_ignorance | answer_general | chitchat
    fact_kind: str              # personal_fact | skill_code | knowledge | content | none
    topic: str | None           # topic key or NO_TOPIC or None (no topic question)
    conf: float                 # calibrated P(action)
    raw: dict


class System1:
    """Lazy singleton around the fine-tuned Laya agent."""

    _agent = None

    @classmethod
    def agent(cls):
        if cls._agent is None:
            import laya  # local import: training scripts don't need it
            cls._agent = laya.Agent(LAYA_DIR, device="cuda")
        return cls._agent

    @classmethod
    def decide(cls, text: str, known_memory: str, topics: list[str]) -> System1Decision:
        agent = cls.agent()
        qs = build_questions(topics)
        res = agent.predict(build_state(text, known_memory), qs)
        ans = res["answers"]
        a = ans["action"]
        action = a["choice"]
        conf = float(a.get("answer_confidence") or a.get("confidence") or 0.0)
        fk = ans["fact_kind"]["choice"] if "fact_kind" in ans else "none"
        if action != "learn":
            fk = "none"
        topic = ans["topic"]["choice"] if "topic" in ans else None
        return System1Decision(action, fk, topic, conf, ans)


def decision_from_json(state: dict, answers: dict) -> System1Decision:
    a = answers["action"]
    topic = answers["topic"]["choice"] if "topic" in answers else None
    fk = answers["fact_kind"]["choice"] if "fact_kind" in answers else "none"
    if a["choice"] != "learn":
        fk = "none"
    return System1Decision(
        a["choice"], fk, topic,
        float(a.get("answer_confidence") or 0.0), answers,
    )


def save_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
