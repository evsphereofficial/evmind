"""Generate EvAGI System-1 routing data (laya fine-tune).

Every example is (state, gold) where state = {known_memory, user_message}
and gold = {action, fact_kind, topic?}. The cross-product of the SAME
question against matching vs non-matching known_memory is the training
signal that kills the old regex heuristics (hijack, follow-up, foreign).

Run: .venv/bin/python src/gen_laya_evagi_data.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from evagi_system1 import NO_TOPIC, build_state, save_jsonl  # noqa: E402

RNG = random.Random(7)
OUT_DIR = Path(__file__).parent.parent / "data" / "laya_evagi"

# ---------------------------------------------------------------------------
# Memory contexts: (memory_string, topics)
# ---------------------------------------------------------------------------
NONE = ("none", [])
BOSS = ("boss: Dax", ["boss"])
BLOOD = ("blood type: O+", ["blood type"])
REQUIEM = (
    "resident evil requiem: survival horror game by capcom, protagonist grace",
    ["resident evil requiem"],
)
COFFEE = ("how to make coffee: 18g beans, 300ml water, bloom 30s", ["how to make coffee"])
OFFICE = ("office code: ZEBRA-42", ["office code"])
AURORA = (
    "aurora station project: reactor core named ZEBRA-42, lead engineer Kira",
    ["aurora station project"],
)
MIXED = (
    "boss: Dax; blood type: O+; resident evil requiem: survival horror game by capcom",
    ["boss", "blood type", "resident evil requiem"],
)
MIXED2 = (
    "first name: Rehan; city: Lisbon; how to make coffee: 18g beans, 300ml water",
    ["first name", "city", "how to make coffee"],
)
MIXED3 = (
    "office code: ZEBRA-42; aurora station project: reactor core named ZEBRA-42, lead Kira; "
    "how to make coffee: 18g beans, 300ml water",
    ["office code", "aurora station project", "how to make coffee"],
)
REQUIEM_S = ("resident evil requiem: horror game by Capcom", ["resident evil requiem"])
AURORA_S = ("aurora station project: core named ZEBRA-42", ["aurora station project"])
AURORA_S2 = ("aurora station project: core ZEBRA-42", ["aurora station project"])
COFFEE_S = ("how to make coffee: 18g beans, 300ml water", ["how to make coffee"])
ALL_CTX = [NONE, BOSS, BLOOD, REQUIEM, REQUIEM_S, COFFEE, COFFEE_S, OFFICE,
           AURORA, AURORA_S, AURORA_S2, MIXED, MIXED2, MIXED3]

NAMES = ["Rehan", "Alice", "Dax", "Meera", "Kira", "Sofia", "Jonas", "Priya"]
DATES = ["March 3rd", "July 12th 1996", "the 4th of July", "September 1st", "December 24th"]
COLORS = ["blue", "teal", "crimson", "forest green", "amber"]
CITIES = ["Lisbon", "Karachi", "Oslo", "Kyoto", "Vancouver"]
PWS = ["xyz123", "NEBULA-77", "starlight99", "blue-moon-42"]
BLOODS = ["O+", "B-", "A+", "AB-"]
CODES = ["ZEBRA-42", "B-212", "QX-991", "KAPPA-9"]
CARS = ["Toyota Corolla", "Model 3", "a Honda Civic"]
MOVIES = ["Spirited Away", "Blade Runner", "The Martian"]

# ---------------------------------------------------------------------------
# Utterance pools: (text, action, fact_kind, topic_target)
# topic_target: None -> NO_TOPIC label (when topics present); "" -> no topic q
# ---------------------------------------------------------------------------

def personal_learns() -> list[tuple]:
    rows = []
    frames = [
        "my first name is {n}", "my name is {n}", "call me {n}",
        "my birthday is {d}", "i was born on {d}",
        "my boss is {n}", "my manager is {n}",
        "my favorite color is {c}", "i love the color {c}",
        "my cat is named {p}", "my dog is called {p}",
        "i live in {city}", "i moved to {city}",
        "my wifi password is {pw}", "the router password is {pw}",
        "my blood type is {bt}",
        "my office code is {code}", "my locker code is {code}",
        "my car is {car}",
        "my favourite movie is {m}",
    ]
    fill = {"n": NAMES, "d": DATES, "c": COLORS, "p": ["Mochi", "Biscuit", "Luna"],
            "city": CITIES, "pw": PWS, "bt": BLOODS, "code": CODES, "car": CARS, "m": MOVIES}
    for f in frames:
        for _ in range(9):
            text = f.format(**{k: RNG.choice(v) for k, v in fill.items()})
            rows.append((text, "learn", "personal_fact", None))
    follow = [
        "actually my birthday is {d}", "oh and my favorite color is {c}",
        "by the way my boss is {n}", "i forgot to mention i live in {city}",
        "just so you know my wifi password is {pw}", "one more thing my first name is {n}",
        "wait no, my office code is {code}", "correction, my blood type is {bt}",
    ]
    for f in follow:
        for _ in range(8):
            text = f.format(**{k: RNG.choice(v) for k, v in fill.items()})
            rows.append((text, "learn", "personal_fact", None))
    for _ in range(60):
        base = RNG.choice(frames).format(**{k: RNG.choice(v) for k, v in fill.items()})
        pre = RNG.choice(["ok so ", "btw ", "hey just so you know, ", ""])
        post = RNG.choice(["", " ok?", " thanks", "!"])
        rows.append((pre + base + post, "learn", "personal_fact", None))
    return rows


def skill_learns() -> list[tuple]:
    stmts = [
        "when you make coffee use 18g of beans and 300ml water",
        "from now on always double-check my calendar before booking",
        "remember to reset the router hold the button for 10 seconds",
        "always format my summaries as bullet points",
        "when i ask for a reminder include the deadline",
        "keep answers short unless i ask for detail",
        "never schedule meetings before 10am",
        "when making pour over coffee use a 1:16 ratio and bloom 30 seconds",
        "always greet me with the weather if i say good morning",
        "when i say ship it, push to main and run the tests",
    ]
    teaches = [
        "teach me how to reverse a string in python",
        "teach me how to make pour over coffee",
        "learn how to change a tire",
        "learn how to fold a fitted sheet",
        "teach me how to back up a database",
        "how do i make coffee? wait, remember this: 18g beans 300ml water",
    ]
    rows = [(s, "learn", "skill_code", None) for s in stmts for _ in range(11)]
    rows += [(t, "learn", "skill_code", None) for t in teaches for _ in range(9)]
    return rows


def knowledge_learns() -> list[tuple]:
    stmts = [
        "Resident Evil Requiem is a survival horror game by Capcom",
        "the aurora station project uses a reactor core called ZEBRA-42",
        "our team ships the evmind repo every friday",
        "the office coffee machine takes beans from the blue bin only",
        "a swish function is an activation used in some transformers",
        "did you know the eiffel tower was meant to be temporary?",
        "the new library downtown opens next month",
        "our standup is at 10am every single day",
        "the project deadline moved to march 15th",
        "v2 of the api deprecates the old auth header",
        "the train to oberdorf runs twice an hour",
        "Kira is the lead engineer on aurora station",
    ]
    rows = [(s, "learn", "knowledge", None) for s in stmts for _ in range(11)]
    rows += [
        ("did you know resident evil requiem ships in 2026?", "learn", "knowledge", None),
        ("btw the capital of bolivia is sucre (legal)", "learn", "knowledge", None),
        ("fun fact honey never spoils", "learn", "knowledge", None),
        ("update: the aurora launch slipped to june", "learn", "knowledge", None),
        ("note that ZEBRA-42 runs on thorium", "learn", "knowledge", None),
    ] * 8
    return rows


CONTENT_PASSAGES = [
    "The Aurora Station is a fictional deep-space research outpost orbiting Kepler-442b. "
    "It houses a thorium reactor core designated ZEBRA-42, a hydroponics ring that feeds "
    "forty crew, and a science deck with three laboratories. Kira leads the engineering "
    "team; Dax handles mission operations. The station was commissioned in 2141 after the "
    "third colonization wave.",
    "Grandma's pasta sauce: start with a splash of olive oil, soften one onion and two "
    "carrots on low heat for fifteen minutes. Add two tins of plum tomatoes, a teaspoon of "
    "sugar, and torn basil. Simmer uncovered for ninety minutes, stirring occasionally. "
    "Finish with a knob of butter and serve over rigatoni with parmesan.",
    "Meeting notes, Tuesday: the team agreed to freeze feature work on Thursday so QA can "
    "run the regression suite. Priya will land the tokenizer fix by Wednesday noon. Dax "
    "wants the latency table refreshed before the demo. Action item: Rehan drafts the "
    "release notes and posts them to the channel.",
    "Trip plan for Oslo: arrive Friday evening, stay at the hotel near Grand Hotel. "
    "Saturday is the fjord cruise and the Viking ship museum. Sunday we take the train to "
    "Flåm for the scenic railway, then fly home from Bergen on Monday morning. Pack warm "
    "layers and waterproof boots.",
    "Character bio: Grace Marlowe is a former FBI analyst turned private investigator. "
    "She is pragmatic, sarcastic, and secretly haunted by a cold case from 2019. In the "
    "requiem storyline she returns to Raccoon City to trace a bioweapon shipment, joined "
    "by an old partner who may be playing both sides.",
]


def content_learns() -> list[tuple]:
    rows = []
    for p in CONTENT_PASSAGES:
        for pre in ["store this: ", "here are my notes: ", "learn this: ", ""]:
            rows.append((pre + p, "learn", "content", None))
            rows.append((pre + p, "learn", "content", None))
        for pre in ["remember this exactly: ", "note this down: "]:
            rows.append((pre + p, "learn", "content", None))
            rows.append((pre + p, "learn", "content", None))
    return rows


def memory_queries() -> list[tuple]:
    """answer_from_memory — only against MATCHING contexts."""
    rows = []
    personal = {
        "boss": ["what's my boss?", "who is my boss?", "remind me who my boss is",
                 "what is my boss's name again?", "quick — my boss?"],
        "blood type": ["what's my blood type?", "what is my blood type again?",
                       "do you remember my blood type?", "remind me of my blood type"],
        "first name": ["what's my first name?", "what is my name?", "do you remember my name?"],
        "city": ["which city do i live in?", "what city am i in?"],
        "office code": ["what's the office code again?", "what is my office code?",
                        "remind me of the office code", "what was the office code?"],
    }
    ctx_map = {"boss": BOSS, "blood type": BLOOD, "first name": MIXED2,
               "city": MIXED2, "office code": OFFICE}
    for field, qs in personal.items():
        ctx, topics = ctx_map[field]
        topic = field if field in topics else (topics[0] if topics else NO_TOPIC)
        for q in qs:
            for _ in range(11):
                rows.append((q, "answer_from_memory", "none", topic))
    # mixed contexts: field present among many
    for q in ["what's my blood type?", "who is my boss again?"]:
        for _ in range(8):
            rows.append((q, "answer_from_memory", "none", "blood type" if "blood" in q else "boss"))
    # skills
    for q in ["how do i make coffee?", "remind me how to make coffee",
              "what are the steps to make coffee?", "how was i supposed to make coffee again?"]:
        for ctx in (COFFEE, MIXED2, MIXED3):
            for _ in range(4):
                rows.append((q, "answer_from_memory", "none", "how to make coffee"))
    # entity recall
    for q in ["tell me about resident evil requiem", "what is resident evil requiem?",
              "who is the protagonist of resident evil requiem?",
              "what studio makes requiem?", "tell me about the requiem game"]:
        for _ in range(6):
            rows.append((q, "answer_from_memory", "none", "resident evil requiem"))
    for q in ["what's the reactor core called?", "remind me about aurora station",
              "who leads the aurora project?", "what do we know about aurora station?"]:
        for _ in range(6):
            rows.append((q, "answer_from_memory", "none", "aurora station project"))
    # pronoun follow-ups (sole plausible antecedent) — target topic explicit
    pronoun = [
        ("resident evil requiem", ["does it have multiplayer?", "is it scary?",
                                   "how long is it?", "when does it come out?",
                                   "what engine does it use?", "is it any good?"]),
        ("aurora station project", ["does it have a reactor core?",
                                    "who is leading it right now?"]),
    ]
    for t, qs in pronoun:
        for q in qs:
            for _ in range(6):
                rows.append((q, "answer_from_memory", "none", t))
    # multi-topic disambiguation
    for q, t in [("what's my blood type?", "blood type"),
                 ("who is my boss?", "boss"),
                 ("tell me about the requiem game", "resident evil requiem")]:
        for _ in range(5):
            rows.append((q, "answer_from_memory", "none", t))
    # topic title mentioned in question
    for q in ["what's ZEBRA-42 again?", "remind me about ZEBRA-42"]:
        for _ in range(6):
            rows.append((q, "answer_from_memory", "none", "office code"))
    # teach-turns that refer to an existing topic (update) keep action=learn
    for q in ["actually the office code changed to KAPPA-9",
              "update: ZEBRA-42 now runs on thorium"]:
        for _ in range(5):
            rows.append((q, "learn", "personal_fact", "office code"))
        for _ in range(5):
            rows.append((q, "learn", "knowledge", "office code"))
    return rows


def admit_ignorance() -> list[tuple]:
    """Personal/specific questions whose target is NOT in memory."""
    rows = []
    # THE hard negative: personal question + unrelated stored topics
    hijack = [
        ("what's my blood type?", "blood type"),
        ("what is my blood type?", "blood type"),
        ("what's my birthday?", "birthday"),
        ("do you remember my birthday?", "birthday"),
        ("what's my favorite color?", "favorite color"),
        ("what's my cat's name?", "cat name"),
        ("what is my office code?", "office code"),
        ("remind me of my wifi password?", "wifi password"),
        ("what's my PIN?", "PIN"),
        ("remind me of my locker code", "locker code"),
        ("do you know my boss's name?", "boss name"),
    ]
    unrelated = [NONE, BOSS, BLOOD, REQUIEM, MIXED, MIXED3, COFFEE, OFFICE]
    for text, tgt in hijack:
        for ctx in unrelated:
            if tgt and _topic_alias(tgt, ctx[1]):
                continue  # that would contradict the admit label
            for _ in range(7):
                rows.append((text, "admit_ignorance", "none", tgt))
    # same question with NO context at all
    for text, tgt in hijack:
        for _ in range(4):
            rows.append((text, "admit_ignorance", "none", tgt))
    # pronoun with no antecedent — exclude the topic that WOULD match
    for q, excl in [("does it have multiplayer?", "resident evil requiem"),
                    ("is it any good?", "resident evil requiem"),
                    ("how long is it?", "resident evil requiem"),
                    ("does it have a reactor core?", "aurora")]:
        for _ in range(7):
            rows.append((q, "admit_ignorance", "none", excl))
    # recall of never-taught subject
    for q in ["what did i tell you about mars?", "remind me what i said about mars",
              "what do you know about the vega project?",
              "tell me about project nimbus", "remind me about the tokyo trip",
              "what did i say about ZEBRA-99?"]:
        for ctx in [NONE, BOSS, REQUIEM, MIXED]:
            for _ in range(5):
                rows.append((q, "admit_ignorance", "none", None))
    # niche entity not stored (would be answer_from_memory if stored!)
    for q in ["tell me about resident evil requiem",
              "who is the protagonist of resident evil requiem?"]:
        for ctx in [NONE, BOSS, BLOOD, MIXED2, COFFEE]:
            for _ in range(6):
                rows.append((q, "admit_ignorance", "none", "resident evil requiem"))
    for q in ["what's the reactor core called?", "remind me about aurora station"]:
        for ctx in [NONE, BOSS, REQUIEM, MIXED]:
            for _ in range(6):
                rows.append((q, "admit_ignorance", "none", "aurora"))
    rows += [
        ("i wonder what my blood type is", "admit_ignorance", "none", "blood type"),
        ("you should know my birthday", "admit_ignorance", "none", "birthday"),
        ("hmm what was my password again", "admit_ignorance", "none", None),
        ("what's the wifi password here?", "admit_ignorance", "none", "wifi"),
        ("quick check — do you know my name?", "admit_ignorance", "none", "first name"),
        ("how long is it?", "admit_ignorance", "none", "resident evil requiem"),
    ] * 8
    return rows


def answer_general() -> list[tuple]:
    rows = []
    open_qs = [
        "what is the capital of bolivia?", "who founded microsoft?",
        "what's the tallest mountain in the world?", "how far is the moon?",
        "explain quantum entanglement", "why is the sky blue?",
        "what is a neural network?", "how does gradient descent work?",
        "how do i reverse a string in python?", "how do i center a div in css?",
        "how do i boil an egg properly?", "should i learn python or rust first?",
        "what should i pack for iceland in winter?", "give me a plan to learn calculus",
        "write a haiku about the sea", "what's 17 times 23?",
        "who wrote hamlet?", "what year did apollo 11 land?",
        "what is the difference between a list and a tuple in python?",
        "how do i fix a flat tire?", "what's the best way to sleep better?",
        "explain the difference between a thesis and a dissertation",
        "how do airplanes stay in the air?",
        "what foods are high in iron?", "how long should i marinate chicken?",
    ]
    # with AND without memory contexts (foreign-domain must never hijack)
    for q in open_qs:
        for ctx in [NONE, REQUIEM, MIXED, AURORA, MIXED3]:
            for _ in range(4):
                excl = "coffee" if "coffee" in q else None
                rows.append((q, "answer_general", "none", excl))
    # world-knowledge question about a DIFFERENT entity than stored
    for q in ["who is the president of france?", "what's the capital of turkey?",
              "who created python the language?", "what is the highest ocean trench?",
              "tell me about the game elden ring"]:
        for ctx in [REQUIEM, MIXED, AURORA, BOSS, NONE]:
            for _ in range(5):
                rows.append((q, "answer_general", "none", None))
    # knowledge request for a well-known concept, memory present or not
    for q in ["tell me about black holes", "what causes tides?",
              "how do vaccines work?", "what is blockchain?"]:
        for ctx in [NONE, REQUIEM, MIXED]:
            for _ in range(4):
                rows.append((q, "answer_general", "none", None))
    # symmetric pair: coffee question when the skill is NOT stored
    for _ in range(12):
        rows.append(("how do i make coffee?", "answer_general", "none", "coffee"))
    for _ in range(8):
        rows.append(("how was i supposed to make coffee again?",
                     "admit_ignorance", "none", "coffee"))
    return rows


def chitchat() -> list[tuple]:
    texts = [
        "hello", "hi there", "hey!", "good morning", "how are you?",
        "thanks!", "thank you so much", "haha nice one", "lol", "what's up?",
        "ok cool", "bye", "good night", "you're funny", "what can you do?",
        "who are you?", "tell me a joke", "that's great news", "congrats",
        "i'm bored", "nice weather today", "good evening", "yo",
        "appreciate it", "cheers", "see you later", "please and thank you",
        "that was helpful", "perfect", "sounds good", "you made my day", "heh", "hmm",
    ]
    return [(t, "chitchat", "none", None) for t in texts for _ in range(15)]


def build_examples() -> list[dict]:
    raw: list[tuple] = []
    raw += personal_learns()
    raw += skill_learns()
    raw += knowledge_learns()
    raw += content_learns()
    raw += memory_queries()
    raw += admit_ignorance()
    raw += answer_general()
    raw += chitchat()

    examples = []
    n_topics_ctx = [c for c in ALL_CTX if c[1]]
    for text, action, fact_kind, topic_target in raw:
        # Row semantics of element 4:
        #   answer_from_memory -> required topic (ctx must contain it)
        #   admit_ignorance / answer_general -> EXCLUDE ctx containing it
        #   learn -> optional topic hint (ctx random)
        if action == "answer_from_memory" and topic_target:
            candidates = [c for c in n_topics_ctx
                          if topic_target in c[1] or _topic_alias(topic_target, c[1])]
            if not candidates:
                candidates = n_topics_ctx
            mem, topics = RNG.choice(candidates)
        elif action in ("admit_ignorance", "answer_general"):
            cands = [c for c in ALL_CTX
                     if not (topic_target and _topic_alias(topic_target, c[1]))]
            mem, topics = RNG.choice(cands)
        else:
            mem, topics = RNG.choice(ALL_CTX)
            if action == "learn":
                # teaches usually happen against a fresh-ish brain
                if RNG.random() < 0.5:
                    mem, topics = RNG.choice([NONE, NONE, BOSS, BLOOD, REQUIEM])
        if RNG.random() < 0.45:
            text = text[0].upper() + text[1:]
        gold = {"action": action, "fact_kind": fact_kind}
        if topics:
            if topic_target and (topic_target in topics or _topic_alias(topic_target, topics)):
                gold["topic"] = next(
                    t for t in topics
                    if t == topic_target or _topic_alias(topic_target, [t])
                )
            else:
                gold["topic"] = NO_TOPIC
        examples.append({"state": build_state(text, mem), "gold": gold, "topics": list(topics)})
    return examples


def _topic_alias(target: str, topics: list[str]) -> bool:
    t = target.lower()
    for cand in topics:
        c = cand.lower()
        if t in c or c in t:
            return True
        tw = set(t.split())
        cw = set(c.split())
        if tw and len(tw & cw) >= max(1, len(tw) // 2 + 1):
            return True
    return False


def tricky_battery() -> list[dict]:
    """Held-out evaluation battery: fresh phrasings of every hard class."""
    cases = [
        # (text, known_memory, topics, action)
        ("What's my blood type?", "boss: Dax; resident evil requiem: survival horror game",
         ["boss", "resident evil requiem"], "admit_ignorance"),
        ("What is my blood type?", "office code: ZEBRA-42", ["office code"], "admit_ignorance"),
        ("what's my blood type?", "blood type: O+", ["blood type"], "answer_from_memory"),
        ("Do you remember my blood type?", "blood type: AB-; boss: Dax",
         ["blood type", "boss"], "answer_from_memory"),
        ("Does it have multiplayer?", "resident evil requiem: survival horror game by capcom",
         ["resident evil requiem"], "answer_from_memory"),
        ("Does it have multiplayer?", "boss: Dax", ["boss"], "admit_ignorance"),
        ("Does it have multiplayer?", "none", [], "admit_ignorance"),
        ("Is it scary though?", "resident evil requiem: horror game, protagonist grace",
         ["resident evil requiem"], "answer_from_memory"),
        ("What's the reactor core called?", "aurora station project: core named ZEBRA-42",
         ["aurora station project"], "answer_from_memory"),
        ("What's the reactor core called?", "boss: Dax", ["boss"], "admit_ignorance"),
        ("What is the capital of France?", "resident evil requiem: survival horror game",
         ["resident evil requiem"], "answer_general"),
        ("What is the capital of France?", "none", [], "answer_general"),
        ("How do I make coffee?", "how to make coffee: 18g beans, 300ml water",
         ["how to make coffee"], "answer_from_memory"),
        ("How do I make coffee?", "boss: Dax", ["boss"], "answer_general"),
        ("How do I make coffee?", "none", [], "answer_general"),
        ("Tell me about Resident Evil Requiem", "resident evil requiem: horror game by Capcom",
         ["resident evil requiem"], "answer_from_memory"),
        ("Tell me about Resident Evil Requiem", "boss: Dax; blood type: O+",
         ["boss", "blood type"], "admit_ignorance"),
        ("What did I tell you about Mars?", "aurora station project: reactor ZEBRA-42",
         ["aurora station project"], "admit_ignorance"),
        ("Remind me of my PIN", "boss: Dax", ["boss"], "admit_ignorance"),
        ("Who is my boss?", "boss: Dax; blood type: O+", ["boss", "blood type"],
         "answer_from_memory"),
        ("Actually my birthday is March 3rd", "boss: Dax", ["boss"], "learn"),
        ("Actually my birthday is March 3rd", "none", [], "learn"),
        ("The office code changed to KAPPA-9", "office code: ZEBRA-42",
         ["office code"], "learn"),
        ("When you make coffee use 18g beans", "how to make coffee: 18g beans",
         ["how to make coffee"], "learn"),
        ("When you make coffee use 18g beans", "none", [], "learn"),
        ("Did you know honey never spoils?", "boss: Dax", ["boss"], "learn"),
        ("My wife's name is Sofia", "boss: Dax", ["boss"], "learn"),
        ("store this: The manual covers version 3 of the router firmware, released last "
         "spring, with a new admin panel.", "none", [], "learn"),
        ("How are you today?", "boss: Dax", ["boss"], "chitchat"),
        ("Thanks a lot!", "none", [], "chitchat"),
        ("What can you do?", "boss: Dax; blood type: O+", ["boss", "blood type"], "chitchat"),
        ("How do I reverse a string in Python?", "office code: ZEBRA-42", ["office code"],
         "answer_general"),
        ("Explain quantum entanglement to me", "resident evil requiem: horror game",
         ["resident evil requiem"], "answer_general"),
        ("What's my office code again?", "office code: ZEBRA-42", ["office code"],
         "answer_from_memory"),
        ("What's my office code?", "boss: Dax", ["boss"], "admit_ignorance"),
        ("Who leads the aurora project?", "aurora station project: lead engineer Kira",
         ["aurora station project"], "answer_from_memory"),
        ("Does it have a reactor core?", "aurora station project: core ZEBRA-42",
         ["aurora station project"], "answer_from_memory"),
        ("Does it have a reactor core?", "resident evil requiem: horror game",
         ["resident evil requiem"], "admit_ignorance"),
        ("What should I pack for Iceland?", "boss: Dax", ["boss"], "answer_general"),
        ("Tell me a joke", "none", [], "chitchat"),
        ("Good morning!", "aurora station project: reactor ZEBRA-42",
         ["aurora station project"], "chitchat"),
        ("What's the tallest mountain?", "how to make coffee: 18g beans",
         ["how to make coffee"], "answer_general"),
        ("Remind me how to make coffee", "how to make coffee: 18g beans, 300ml water",
         ["how to make coffee"], "answer_from_memory"),
        ("What do you know about Project Nimbus?", "boss: Dax", ["boss"], "admit_ignorance"),
        ("I wonder what my blood type is", "first name: Rehan", ["first name"],
         "admit_ignorance"),
        ("Actually the wifi password changed to MOON-7", "boss: Dax", ["boss"], "learn"),
        ("By the way my favorite color is teal", "none", [], "learn"),
        ("is it any good?", "resident evil requiem: horror game",
         ["resident evil requiem"], "answer_from_memory"),
        ("is it any good?", "none", [], "admit_ignorance"),
    ]
    # explicit topic golds: text -> expected topic when present in ctx
    topic_gold = {
        "what's my blood type?": "blood type",
        "What's my blood type?": "blood type",
        "What is my blood type?": "blood type",
        "Do you remember my blood type?": "blood type",
        "Does it have multiplayer?": "resident evil requiem",
        "Is it scary though?": "resident evil requiem",
        "is it any good?": "resident evil requiem",
        "What's the reactor core called?": "aurora station project",
        "How do I make coffee?": "how to make coffee",
        "Remind me how to make coffee": "how to make coffee",
        "Tell me about Resident Evil Requiem": "resident evil requiem",
        "Who is my boss?": "boss",
        "What's my office code again?": "office code",
        "What's my office code?": "office code",
        "Who leads the aurora project?": "aurora station project",
        "Does it have a reactor core?": "aurora station project",
    }
    rows = []
    for text, mem, topics, action in cases:
        gold = {"action": action, "fact_kind": "none" if action != "learn" else "knowledge"}
        if topics:
            tg = None if action == "learn" else topic_gold.get(text)
            gold["topic"] = tg if (tg and any(tg in t or t in tg for t in topics)) else NO_TOPIC
        rows.append({"state": build_state(text, mem), "gold": gold, "topics": list(topics)})
    return rows


def main() -> None:
    examples = build_examples()
    RNG.shuffle(examples)
    n_val = max(1, int(len(examples) * 0.15))
    val, train = examples[:n_val], examples[n_val:]

    # class balance report
    def counts(rows):
        c: dict = {}
        for r in rows:
            c[r["gold"]["action"]] = c.get(r["gold"]["action"], 0) + 1
        return c

    print(f"train={len(train)} val={len(val)}")
    print("train action balance:", counts(train))
    print("val   action balance:", counts(val))
    save_jsonl(str(OUT_DIR / "train.jsonl"), train)
    save_jsonl(str(OUT_DIR / "val.jsonl"), val)
    battery = tricky_battery()
    save_jsonl(str(OUT_DIR / "battery.jsonl"), battery)
    print(f"wrote {OUT_DIR}/train.jsonl val.jsonl battery.jsonl ({len(battery)} battery)")


if __name__ == "__main__":
    main()
