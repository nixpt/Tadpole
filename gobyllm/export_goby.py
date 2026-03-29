"""Export GobyLLM checkpoint + tokenizer to a .goby binary for the C runtime."""

import argparse
import json
import os
import struct

import numpy as np
import torch

from .config import GobyConfig
from .model import GobyLLM


MAGIC   = 0x47425931  # "GBY1"
VERSION = 1

# ── Tokenizer helpers ───────────────────────────────────────────────────


def bytes_to_unicode():
    """GPT-2 / ByteLevel BPE byte-to-unicode mapping."""
    bs = list(range(ord("!"), ord("~") + 1))
    bs += list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def export_tokenizer_data(tokenizer_path):
    """Extract vocab (as raw bytes per token) and merge pairs from tokenizer.json."""

    with open(tokenizer_path) as f:
        data = json.load(f)

    model_data = data["model"]
    vocab_str_to_id = model_data["vocab"]
    merges_raw = model_data.get("merges", [])

    # Build id → token string mapping
    id_to_str = {v: k for k, v in vocab_str_to_id.items()}
    vocab_size = max(id_to_str.keys()) + 1

    # Byte-level decoding: convert token unicode strings back to actual bytes
    unicode_to_byte = bytes_to_unicode()

    token_bytes = []
    for i in range(vocab_size):
        s = id_to_str.get(i, "")
        # Try to decode as byte-level
        try:
            raw = bytes([unicode_to_byte[c] for c in s])
        except (KeyError, ValueError):
            # Special token or unknown — store as UTF-8
            raw = s.encode("utf-8")
        token_bytes.append(raw)

    # Parse merges: each is either ["tokenA", "tokenB"] or "tokenA tokenB"
    merge_pairs = []
    for merge_entry in merges_raw:
        if isinstance(merge_entry, list):
            if len(merge_entry) == 2:
                a, b = merge_entry
            else:
                continue
        elif isinstance(merge_entry, str):
            parts = merge_entry.split(" ", 1)
            if len(parts) == 2:
                a, b = parts
            else:
                continue
        else:
            continue
        if a in vocab_str_to_id and b in vocab_str_to_id:
            merge_pairs.append((vocab_str_to_id[a], vocab_str_to_id[b]))

    # Build byte-to-initial-token lookup (single bytes)
    byte_to_token = [0] * 256
    byte_to_unicode_map = {v: k for k, v in bytes_to_unicode().items()}
    for byte_val in range(256):
        char = chr(byte_val)
        # Find the unicode char that represents this byte
        for uc, bv in unicode_to_byte.items():
            if bv == byte_val:
                if uc in vocab_str_to_id:
                    byte_to_token[byte_val] = vocab_str_to_id[uc]
                break

    return token_bytes, merge_pairs, byte_to_token, vocab_size


# ── Binary writer ───────────────────────────────────────────────────────


def write_goby(output_path, config, state_dict, rope_cos, rope_sin,
               token_bytes, merge_pairs, byte_to_token, vocab_size):
    """Write the .goby binary file."""
    with open(output_path, "wb") as f:
        # ── Header ──────────────────────────────────────────────────────
        f.write(struct.pack("I", MAGIC))
        f.write(struct.pack("I", VERSION))

        # Config
        f.write(struct.pack("i", config.vocab_size))
        f.write(struct.pack("i", config.d_model))
        f.write(struct.pack("i", config.n_layers))
        f.write(struct.pack("i", config.n_heads))
        f.write(struct.pack("i", config.n_kv_heads))
        f.write(struct.pack("i", config.ffn_hidden))
        f.write(struct.pack("i", config.max_seq_len))
        f.write(struct.pack("i", 1 if config.early_exit else 0))
        f.write(struct.pack("i", config.min_exit_layer))
        f.write(struct.pack("f", config.exit_threshold))

        # ── Tokenizer ──────────────────────────────────────────────────
        # Vocab
        f.write(struct.pack("I", vocab_size))
        for tb in token_bytes:
            f.write(struct.pack("H", len(tb)))
            f.write(tb)

        # Merges
        f.write(struct.pack("I", len(merge_pairs)))
        for a, b in merge_pairs:
            f.write(struct.pack("II", a, b))

        # Byte-to-token lookup
        for bt in byte_to_token:
            f.write(struct.pack("I", bt))

        # ── Model weights ─────────────────────────────────────────────
        def write_tensor(name):
            t = state_dict[name].float().cpu().numpy()
            f.write(t.tobytes())
            return t.size

        total_params = 0

        # Token embeddings (also used as lm_head)
        total_params += write_tensor("tok_emb.weight")

        # Transformer layers
        for i in range(config.n_layers):
            prefix = f"blocks.{i}."
            total_params += write_tensor(prefix + "norm.weight")
            total_params += write_tensor(prefix + "attn.wq.weight")
            total_params += write_tensor(prefix + "attn.wk.weight")
            total_params += write_tensor(prefix + "attn.wv.weight")
            total_params += write_tensor(prefix + "attn.wo.weight")
            total_params += write_tensor(prefix + "ffn.w_gate.weight")
            total_params += write_tensor(prefix + "ffn.w_up.weight")
            total_params += write_tensor(prefix + "ffn.w_down.weight")

            # Exit router
            if config.early_exit:
                total_params += write_tensor(f"exit_routers.{i}.weight")
                total_params += write_tensor(f"exit_routers.{i}.bias")

        # Final norm
        total_params += write_tensor("norm.weight")

        # RoPE buffers
        f.write(rope_cos.cpu().float().numpy().tobytes())
        f.write(rope_sin.cpu().float().numpy().tobytes())

    file_size = os.path.getsize(output_path)
    print(f"Exported {output_path}: {file_size / 1e6:.1f}MB ({total_params:,} params)")


# ── CLI ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--tokenizer", default="data/tokenizer.json")
    parser.add_argument("--output", default="goby.bin")
    args = parser.parse_args()

    # Load model
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = GobyConfig(**ckpt["config"])
    model = GobyLLM(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    total, router = model.param_count()
    print(f"Model: {total:,} params ({total/1e6:.1f}M)")

    # Export tokenizer data
    token_bytes, merge_pairs, byte_to_token, vocab_size = export_tokenizer_data(args.tokenizer)
    print(f"Tokenizer: {vocab_size} tokens, {len(merge_pairs)} merges")

    # Write binary
    write_goby(
        args.output, config, model.state_dict(),
        model.rope_cos, model.rope_sin,
        token_bytes, merge_pairs, byte_to_token, vocab_size,
    )


if __name__ == "__main__":
    main()
