#!/usr/bin/env python3
"""
Train EvAGI Tiny Conversational LLM — 2-phase approach.
Phase 1: Pretrain on TinyStories for language ability.
Phase 2: Finetune on everyday-conversations for chat format.
"""

import json, math, torch, torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

MODEL_DIR = Path("models/evagi_tiny_chat")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
TOKENIZER_NAME = "gpt2"
MAX_LENGTH = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_LAYER = 2
N_HEAD = 4
N_EMBD = 96
VOCAB_SIZE = 50257


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=256):
        self.examples = []
        for text in texts:
            enc = tokenizer(text, truncation=True, max_length=max_length,
                          return_tensors="pt")
            input_ids = enc["input_ids"].squeeze(0)
            if input_ids.size(0) < 16:
                continue
            labels = input_ids.clone()
            labels[:-1] = labels[1:].clone()
            labels[-1] = -100
            self.examples.append({"input_ids": input_ids, "labels": labels})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch):
    max_len = max(x["input_ids"].size(0) for x in batch)
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, x in enumerate(batch):
        L = x["input_ids"].size(0)
        input_ids[i, :L] = x["input_ids"]
        labels[i, :L] = x["labels"]
    return input_ids, labels


from models.evagi_tiny import EvagiTinyLM, EvagiTinyConfig


def train():
    print("=" * 70)
    print("EvAGI Tiny Conversational LLM — 2-Phase Training")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    config = EvagiTinyConfig(
        vocab_size=VOCAB_SIZE,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD,
        n_ctx=MAX_LENGTH,
        dropout=0.1,
    )

    model = EvagiTinyLM(config).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} params")

    # ─── Phase 1: Load TinyStories (pre-built cache or download) ───
    print("\nPhase 1: TinyStories Pretraining")
    cache_path = MODEL_DIR / "tinystories_cache.pt"
    if cache_path.exists():
        print("  Loading cached TinyStories...")
        all_texts = torch.load(cache_path, weights_only=False)
    else:
        print("  Downloading TinyStories (100K samples)...")
        from datasets import load_dataset
        ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        all_texts = []
        for i, item in enumerate(ds):
            if i >= 100000:
                break
            all_texts.append(item["text"])
            if (i + 1) % 25000 == 0:
                print(f"    {i+1} loaded...")
        torch.save(all_texts, cache_path)
        print(f"  Cached {len(all_texts)} stories")

    ts_dataset = TextDataset(all_texts, tokenizer, MAX_LENGTH)
    print(f"  TinyStories: {len(ts_dataset)} valid samples")

    loader = DataLoader(ts_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0.1)
    total_steps = len(loader) * 5

    step = 0
    for epoch in range(5):
        model.train()
        running = 0.0
        n = 0
        for i, (input_ids, labels) in enumerate(loader):
            input_ids, labels = input_ids.to(DEVICE), labels.to(DEVICE)
            _, loss = model(input_ids, targets=labels)
            loss.backward()
            if (i + 1) % 4 == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                step += 1
            running += loss.item()
            n += 1
        avg = running / max(n, 1)
        print(f"  ts epoch {epoch+1}/5 loss={avg:.4f}")

    # Save after pretraining
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config.__dict__,
    }, MODEL_DIR / "base_model.pt")
    print("  Saved phase 1 checkpoint")

    # ─── Phase 2: Finetune on everyday-conversations ───
    print("\nPhase 2: Conversation Finetuning")
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceTB/everyday-conversations-llama3.1-2k", split="train_sft")
    conv_texts = []
    for item in ds:
        messages = item["messages"]
        text = ""
        for msg in messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            text += f"{role}: {msg['content']}\n"
        text += "Assistant:"
        conv_texts.append(text)
    conv_dataset = TextDataset(conv_texts, tokenizer, MAX_LENGTH)
    print(f"  Conversations: {len(conv_dataset)} samples")

    loader = DataLoader(conv_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    best_loss = float("inf")

    for epoch in range(50):
        model.train()
        running = 0.0
        n = 0
        for i, (input_ids, labels) in enumerate(loader):
            input_ids, labels = input_ids.to(DEVICE), labels.to(DEVICE)
            _, loss = model(input_ids, targets=labels)
            loss.backward()
            if (i + 1) % 4 == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            running += loss.item()
            n += 1
        avg = running / max(n, 1)
        if avg < best_loss:
            best_loss = avg
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config.__dict__,
                "test_loss": avg,
            }, MODEL_DIR / "base_model.pt")
        print(f"  conv epoch {epoch+1}/50 loss={avg:.4f} best={best_loss:.4f}")

    # Save config
    with open(MODEL_DIR / "config.json", "w") as f:
        json.dump(config.__dict__, f, indent=2)

    # Test generation
    model.eval()
    test_prompts = [
        "User: Hi!\nAssistant:",
        "User: Hello, how are you?\nAssistant:",
        "User: What is my name?\nAssistant:",
    ]
    print("\n--- Test Generation ---")
    for prompt in test_prompts:
        enc = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model.generate(enc["input_ids"], max_new_tokens=50,
                               temperature=0.8, top_p=0.9)
        reply = tokenizer.decode(out[0][enc["input_ids"].shape[1]:],
                                skip_special_tokens=True)
        print(f"  {prompt.split(chr(10))[0]}")
        print(f"  {reply.strip()[:120]}\n")


if __name__ == "__main__":
    train()
