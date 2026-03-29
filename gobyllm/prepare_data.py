"""GobyLLM data preparation pipeline."""

import json
import os
import random

import requests

random.seed(42)

ALPACA_URL = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
DATA_DIR = "data"
VOCAB_SIZE = 8192

# ── Special tokens ──────────────────────────────────────────────────────

SPECIAL_TOKENS = [
    "<pad>",         # 0
    "<|im_start|>",  # 1
    "<|im_end|>",    # 2
    "<think>",       # 3
    "</think>",      # 4
    "<tool_call>",   # 5
    "</tool_call>",  # 6
]

ALPACA_SYSTEMS = [
    "You are a helpful assistant.",
    "You are a helpful AI assistant. Answer clearly and concisely.",
    "You are an intelligent assistant. Be helpful and accurate.",
    "You are a knowledgeable assistant. Provide clear, useful responses.",
    "You are a helpful assistant. Think step by step when needed.",
]

# ── Pipeline steps ──────────────────────────────────────────────────────


def download_alpaca(data_dir):
    """Download alpaca_data.json if not present."""
    path = os.path.join(data_dir, "alpaca_data.json")
    if os.path.exists(path):
        print(f"Alpaca data already exists at {path}")
        return path

    print(f"Downloading alpaca dataset from {ALPACA_URL}...")
    resp = requests.get(ALPACA_URL, timeout=120)
    resp.raise_for_status()
    os.makedirs(data_dir, exist_ok=True)
    with open(path, "w") as f:
        f.write(resp.text)
    data = json.loads(resp.text)
    print(f"Downloaded {len(data)} samples to {path}")
    return path


def format_alpaca_sample(sample):
    """Convert an alpaca sample to GobyLLM chat format."""
    system = random.choice(ALPACA_SYSTEMS)
    instruction = sample["instruction"]
    inp = sample.get("input", "")
    output = sample["output"]

    user_msg = instruction
    if inp and inp.strip():
        user_msg = f"{instruction}\n\n{inp}"

    text = (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n{output}<|im_end|>"
    )
    return text


def train_tokenizer(texts, save_path, vocab_size=VOCAB_SIZE):
    """Train a BPE tokenizer on the given texts."""
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        min_frequency=2,
    )

    print(f"Training BPE tokenizer (vocab_size={vocab_size}) on {len(texts)} texts...")
    tokenizer.train_from_iterator(texts, trainer)

    # Add post-processor for byte-level
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    tokenizer.save(save_path)
    print(f"Tokenizer saved to {save_path} ({tokenizer.get_vocab_size()} tokens)")
    return tokenizer


def generate_tool_data(n_samples=15000):
    """Generate tool-calling training data."""
    from .generate_data import (
        gen_turn_on, gen_turn_off, gen_set_brightness, gen_set_temp,
        gen_implicit_hot, gen_implicit_cold, gen_implicit_dark, gen_implicit_bright,
        gen_lock_door, gen_music, gen_timer_alarm, gen_blinds, gen_vacuum,
        gen_industrial_command, gen_general_command,
        gen_sensor, gen_question, gen_conversation,
        gen_novel_tool, gen_refusal, gen_instruction,
        format_sample,
    )

    generators = [
        (gen_turn_on, 0.06), (gen_turn_off, 0.06), (gen_set_brightness, 0.04),
        (gen_set_temp, 0.04), (gen_implicit_hot, 0.03), (gen_implicit_cold, 0.03),
        (gen_implicit_dark, 0.02), (gen_implicit_bright, 0.02), (gen_lock_door, 0.02),
        (gen_music, 0.02), (gen_timer_alarm, 0.03), (gen_blinds, 0.02), (gen_vacuum, 0.02),
        (gen_industrial_command, 0.08), (gen_general_command, 0.06),
        (gen_sensor, 0.12), (gen_question, 0.12), (gen_conversation, 0.07),
        (gen_novel_tool, 0.06), (gen_refusal, 0.04), (gen_instruction, 0.04),
    ]

    total_w = sum(w for _, w in generators)
    samples = []
    for gen, w in generators:
        count = max(1, int(n_samples * w / total_w))
        for _ in range(count):
            try:
                s = gen()
                samples.append(format_sample(s))
            except Exception:
                pass

    print(f"Generated {len(samples)} tool-calling samples")
    return samples


def prepare(data_dir=DATA_DIR, tool_samples=15000, eval_ratio=0.05):
    """Full pipeline: download → tokenize → format → merge → save."""
    os.makedirs(data_dir, exist_ok=True)

    # 1. Download alpaca
    alpaca_path = download_alpaca(data_dir)
    with open(alpaca_path) as f:
        alpaca_raw = json.load(f)

    # 2. Format alpaca data
    print(f"Formatting {len(alpaca_raw)} alpaca samples...")
    alpaca_texts = [format_alpaca_sample(s) for s in alpaca_raw]

    # 3. Generate tool-calling data
    print(f"Generating {tool_samples} tool-calling samples...")
    tool_texts = generate_tool_data(tool_samples)

    # 4. Train tokenizer on ALL text
    all_texts = alpaca_texts + tool_texts
    tokenizer_path = os.path.join(data_dir, "tokenizer.json")
    tokenizer = train_tokenizer(all_texts, tokenizer_path)

    # 5. Merge, shuffle, split
    all_data = []
    for text in alpaca_texts:
        all_data.append({"text": text, "source": "alpaca"})
    for text in tool_texts:
        all_data.append({"text": text, "source": "tools"})

    random.shuffle(all_data)
    n_eval = int(len(all_data) * eval_ratio)
    eval_data = all_data[:n_eval]
    train_data = all_data[n_eval:]

    # 6. Save
    for name, data in [("train.jsonl", train_data), ("eval.jsonl", eval_data)]:
        path = os.path.join(data_dir, name)
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

    # Stats
    n_alpaca_train = sum(1 for d in train_data if d["source"] == "alpaca")
    n_tool_train = sum(1 for d in train_data if d["source"] == "tools")

    print(f"\n{'='*50}")
    print(f"GobyLLM data prepared:")
    print(f"  Alpaca samples:      {len(alpaca_texts):,}")
    print(f"  Tool-calling samples: {len(tool_texts):,}")
    print(f"  Total:               {len(all_data):,}")
    print(f"  Train: {len(train_data):,} (alpaca: {n_alpaca_train}, tools: {n_tool_train})")
    print(f"  Eval:  {len(eval_data):,}")
    print(f"  Tokenizer vocab:     {tokenizer.get_vocab_size()}")
    print(f"  Files: {data_dir}/train.jsonl, eval.jsonl, tokenizer.json")

    # Quick tokenizer test
    test = "<|im_start|>user\nHello<|im_end|>"
    ids = tokenizer.encode(test).ids
    decoded = tokenizer.decode(ids)
    print(f"\n  Tokenizer test:")
    print(f"    Input:   {test}")
    print(f"    Tokens:  {len(ids)} ids")
    print(f"    Decoded: {decoded}")


if __name__ == "__main__":
    prepare()
