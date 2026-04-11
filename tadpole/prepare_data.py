"""Prepare training data for Tadpole."""

import json
import os
import random
import shutil

random.seed(42)

DATA_DIR = "data"
VOCAB_SIZE = 4096
DEFAULT_MIXED_PACK_DIR = os.path.join(DATA_DIR, "hf_mixed_starter_pack")

SPECIAL_TOKENS = [
    "<pad>",         # 0
    "<|im_start|>",  # 1
    "<|im_end|>",    # 2
]


def train_tokenizer(texts, save_path, vocab_size=VOCAB_SIZE):
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
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    tokenizer.save(save_path)
    print(f"Tokenizer saved to {save_path} ({tokenizer.get_vocab_size()} tokens)")
    return tokenizer


def _read_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _prepare_from_mixed_pack(data_dir=DATA_DIR, mixed_pack_dir=DEFAULT_MIXED_PACK_DIR):
    train_path = os.path.join(mixed_pack_dir, "train.jsonl")
    eval_path = os.path.join(mixed_pack_dir, "eval.jsonl")
    if not (os.path.exists(train_path) and os.path.exists(eval_path)):
        raise FileNotFoundError(
            f"Mixed pack not found at {mixed_pack_dir}; download the HF starter pack first"
        )

    train_rows = _read_jsonl(train_path)
    eval_rows = _read_jsonl(eval_path)
    os.makedirs(data_dir, exist_ok=True)
    shutil.copyfile(train_path, os.path.join(data_dir, "train.jsonl"))
    shutil.copyfile(eval_path, os.path.join(data_dir, "eval.jsonl"))

    texts = [row["text"] for row in train_rows + eval_rows if row.get("text")]
    tokenizer_path = os.path.join(data_dir, "tokenizer.json")
    tokenizer = train_tokenizer(texts, tokenizer_path)
    print(f"Prepared mixed pack from {mixed_pack_dir}")
    return tokenizer


def prepare(data_dir=DATA_DIR, n_samples=60000, eval_ratio=0.05, source="synthetic", mixed_pack_dir=DEFAULT_MIXED_PACK_DIR):
    if source == "mixed":
        return _prepare_from_mixed_pack(data_dir=data_dir, mixed_pack_dir=mixed_pack_dir)

    os.makedirs(data_dir, exist_ok=True)

    # 1. Generate data
    print(f"Generating {n_samples} samples...")
    from .generate_data import generate_dataset
    generate_dataset(n_samples, eval_ratio)

    # 2. Read back all samples for tokenizer training
    texts = []
    for name in ["data/train.jsonl", "data/eval.jsonl"]:
        if os.path.exists(name):
            texts.extend(row["text"] for row in _read_jsonl(name) if row.get("text"))

    # 3. Train tokenizer
    tokenizer_path = os.path.join(data_dir, "tokenizer.json")
    tokenizer = train_tokenizer(texts, tokenizer_path)

    # Quick test
    test = "<|im_start|>user\nhi guppy<|im_end|>"
    ids = tokenizer.encode(test).ids
    decoded = tokenizer.decode(ids)
    print(f"\nTokenizer test:")
    print(f"  Input:   {test}")
    print(f"  Tokens:  {len(ids)} ids")
    print(f"  Decoded: {decoded}")


if __name__ == "__main__":
    prepare()
