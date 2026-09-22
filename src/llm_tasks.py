"""Procedural language tasks for LLM continual-learning experiments.

Five yes/no skills over real English text (not 2D coordinates):
  sentiment / spelling / capital / plural / antonym

Each sample: (prompt, label) where label in {0,1} maps to "no"/"yes".
Prompt ends at the answer boundary so causal-LM eval can read the
next-token logits for yes vs no.
"""

from __future__ import annotations

import random
import re


LLM_TASK_NAMES = ["sentiment", "spelling", "capital", "plural", "antonym"]

_POS = [
    "I love this story, it made me smile",
    "What a wonderful day at the park",
    "The cake was delicious and everyone cheered",
    "She felt happy when her friend visited",
    "The puppy was cute and very playful",
    "We had a great time on the trip",
    "His kind words made her feel special",
    "The sun shone and the birds sang",
]
_NEG = [
    "I hate this boring movie, it was awful",
    "What a terrible day, everything went wrong",
    "The food tasted bad and made us sick",
    "He felt sad when his toy broke",
    "The storm destroyed the little house",
    "We had a horrible time on the trip",
    "Her rude words made him feel small",
    "The rain never stopped and everyone was cold",
]

_SPELL_OK = [
    "apple", "banana", "garden", "window", "little", "purple", "orange",
    "rabbit", "mother", "water", "flower", "bridge", "candle", "pocket",
]
_SPELL_BAD = [
    "appel", "bananana", "gardern", "windwo", "littel", "purpul", "oragne",
    "rabit", "mothr", "watre", "flwoer", "bridg", "candl", "poket",
]

_CAPS = [
    ("France", "Paris"), ("Japan", "Tokyo"), ("Italy", "Rome"),
    ("Spain", "Madrid"), ("Egypt", "Cairo"), ("Brazil", "Brasilia"),
    ("Canada", "Ottawa"), ("Germany", "Berlin"), ("China", "Beijing"),
    ("India", "New Delhi"), ("Australia", "Canberra"), ("Kenya", "Nairobi"),
]
_CAPS_WRONG = {
    "Paris": "London", "Tokyo": "Beijing", "Rome": "Madrid",
    "Madrid": "Barcelona", "Cairo": "Alexandria", "Brasilia": "Rio",
    "Ottawa": "Toronto", "Berlin": "Munich", "Beijing": "Shanghai",
    "New Delhi": "Mumbai", "Canberra": "Sydney", "Nairobi": "Mombasa",
}

_PLURAL_OK = [
    ("cat", "cats"), ("dog", "dogs"), ("book", "books"), ("tree", "trees"),
    ("car", "cars"), ("bird", "birds"), ("house", "houses"), ("star", "stars"),
    ("ball", "balls"), ("frog", "frogs"),
]
_PLURAL_BAD = [
    ("cat", "catss"), ("dog", "dogies"), ("book", "bookes"), ("tree", "treez"),
    ("car", "carrs"), ("bird", "birdies"), ("house", "housen"),
    ("star", "stares"), ("ball", "balles"), ("frog", "froggy"),
]

_ANT_OK = [
    ("hot", "cold"), ("big", "small"), ("fast", "slow"), ("happy", "sad"),
    ("day", "night"), ("up", "down"), ("young", "old"), ("full", "empty"),
    ("light", "dark"), ("begin", "end"),
]
_ANT_BAD = [
    ("hot", "warm"), ("big", "large"), ("fast", "quick"), ("happy", "glad"),
    ("day", "morning"), ("up", "above"), ("young", "new"), ("full", "whole"),
    ("light", "bright"), ("begin", "start"),
]


def _yes() -> int:
    return 1


def _no() -> int:
    return 0


def generate_llm_task(task_name: str, n: int, seed: int) -> list[tuple[str, int]]:
    """Return n (prompt, label) pairs for one language skill."""
    rng = random.Random(seed)
    pairs: list[tuple[str, int]] = []

    if task_name == "sentiment":
        for _ in range(n):
            if rng.random() < 0.5:
                s = rng.choice(_POS)
                pairs.append((f"Review: {s}. Is this positive? Answer:", _yes()))
            else:
                s = rng.choice(_NEG)
                pairs.append((f"Review: {s}. Is this positive? Answer:", _no()))

    elif task_name == "spelling":
        for _ in range(n):
            if rng.random() < 0.5:
                w = rng.choice(_SPELL_OK)
                pairs.append((f"Is the word '{w}' spelled correctly? Answer:", _yes()))
            else:
                w = rng.choice(_SPELL_BAD)
                pairs.append((f"Is the word '{w}' spelled correctly? Answer:", _no()))

    elif task_name == "capital":
        for _ in range(n):
            country, cap = rng.choice(_CAPS)
            if rng.random() < 0.5:
                pairs.append((f"Is the capital of {country} {cap}? Answer:", _yes()))
            else:
                wrong = _CAPS_WRONG[cap]
                pairs.append((f"Is the capital of {country} {wrong}? Answer:", _no()))

    elif task_name == "plural":
        for _ in range(n):
            if rng.random() < 0.5:
                singular, plural = rng.choice(_PLURAL_OK)
                pairs.append(
                    (f"Is '{plural}' the plural of '{singular}'? Answer:", _yes())
                )
            else:
                singular, bad = rng.choice(_PLURAL_BAD)
                pairs.append(
                    (f"Is '{bad}' the plural of '{singular}'? Answer:", _no())
                )

    elif task_name == "antonym":
        for _ in range(n):
            if rng.random() < 0.5:
                a, b = rng.choice(_ANT_OK)
                pairs.append((f"Are '{a}' and '{b}' opposites? Answer:", _yes()))
            else:
                a, b = rng.choice(_ANT_BAD)
                pairs.append((f"Are '{a}' and '{b}' opposites? Answer:", _no()))

    else:
        raise ValueError(f"unknown llm task: {task_name}")

    rng.shuffle(pairs)
    return pairs[:n]


def tokenize_pairs(tokenizer, pairs, max_length: int = 96):
    """Encode (prompt, label) -> tensors for causal LM training.

    input_ids = prompt_ids + label_ids
    labels    = -100 on prompt, label token ids on answer side
    """
    import torch

    prompt_ids = []
    label_ids = []
    for prompt, label in pairs:
        ans = " yes" if label == 1 else " no"
        p = tokenizer(
            prompt, add_special_tokens=True, truncation=True,
            max_length=max_length - 4,
        )["input_ids"]
        a = tokenizer(ans, add_special_tokens=False)["input_ids"]
        # ensure EOS after answer for a clean boundary
        if tokenizer.eos_token_id is not None:
            a = a + [tokenizer.eos_token_id]
        prompt_ids.append(p)
        label_ids.append(a)

    max_p = max(len(p) for p in prompt_ids)
    max_a = max(len(a) for a in label_ids)
    total = max_p + max_a
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    input_ids = torch.full((len(pairs), total), pad_id, dtype=torch.long)
    attention = torch.zeros((len(pairs), total), dtype=torch.long)
    labels = torch.full((len(pairs), total), -100, dtype=torch.long)

    for i, (p, a) in enumerate(zip(prompt_ids, label_ids)):
        seq = p + a
        input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        attention[i, : len(seq)] = 1
        # supervise only answer tokens
        labels[i, len(p): len(seq)] = torch.tensor(a, dtype=torch.long)

    return {
        "input_ids": input_ids,
        "attention_mask": attention,
        "labels": labels,
    }


def yes_no_token_ids(tokenizer) -> tuple[int, int]:
    """Single-token ids for ' yes' / ' no' (leading space for GPT-style BPE)."""
    yes_enc = tokenizer.encode(" yes", add_special_tokens=False)
    no_enc = tokenizer.encode(" no", add_special_tokens=False)
    if len(yes_enc) != 1 or len(no_enc) != 1:
        # fall back without leading space
        yes_enc = tokenizer.encode("yes", add_special_tokens=False)
        no_enc = tokenizer.encode("no", add_special_tokens=False)
    return yes_enc[0], no_enc[0]
