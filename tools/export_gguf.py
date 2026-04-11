"""
Export a Tadpole PyTorch checkpoint to GGUF format for rama-zpu.

Usage:
    python tools/export_gguf.py --checkpoint <path.pt> --out <path.gguf> [--config <path.json>] [--validate]

Tadpole architecture (confirmed from tadpole/model.py):
    - tok_emb        : nn.Embedding(vocab_size, d_model)
    - pos_emb        : nn.Embedding(max_seq_len, d_model)
    - blocks[N]      : Block
        .norm1       : nn.LayerNorm(d_model)
        .attn.qkv    : nn.Linear(d_model, 3*d_model)
        .attn.out    : nn.Linear(d_model, d_model)
        .norm2       : nn.LayerNorm(d_model)
        .ffn.up      : nn.Linear(d_model, ffn_hidden)
        .ffn.down    : nn.Linear(ffn_hidden, d_model)
    - norm           : nn.LayerNorm(d_model)
    - lm_head        : nn.Linear(d_model, vocab_size, bias=False)  -- tied to tok_emb

Weight tying: lm_head.weight IS tok_emb.weight, so output.weight is omitted
(llama.cpp convention: absent output.weight => tied to token_embd.weight).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

try:
    from gguf import GGMLQuantizationType, GGUFWriter
except ImportError:
    sys.exit(
        "gguf package not found. Install with:\n"
        "    pip install gguf\n"
        "or:\n"
        "    pip install /path/to/llama.cpp/gguf-py"
    )

try:
    import torch
except ImportError:
    sys.exit("torch not found. Install PyTorch to export checkpoints.")


# ---------------------------------------------------------------------------
# Weight name mapping: PyTorch state_dict key  →  GGUF tensor name
# ---------------------------------------------------------------------------

def build_weight_map(n_layers: int) -> dict[str, str]:
    mapping: dict[str, str] = {
        "tok_emb.weight": "token_embd.weight",
        "pos_emb.weight": "position_embd.weight",
        "norm.weight": "output_norm.weight",
        "norm.bias": "output_norm.bias",
    }
    for n in range(n_layers):
        p = f"blocks.{n}"
        g = f"blk.{n}"
        mapping.update({
            f"{p}.norm1.weight":    f"{g}.attn_norm.weight",
            f"{p}.norm1.bias":      f"{g}.attn_norm.bias",
            f"{p}.attn.qkv.weight": f"{g}.attn_qkv.weight",
            f"{p}.attn.qkv.bias":   f"{g}.attn_qkv.bias",
            f"{p}.attn.out.weight": f"{g}.attn_output.weight",
            f"{p}.attn.out.bias":   f"{g}.attn_output.bias",
            f"{p}.norm2.weight":    f"{g}.ffn_norm.weight",
            f"{p}.norm2.bias":      f"{g}.ffn_norm.bias",
            f"{p}.ffn.up.weight":   f"{g}.ffn_up.weight",
            f"{p}.ffn.up.bias":     f"{g}.ffn_up.bias",
            f"{p}.ffn.down.weight": f"{g}.ffn_down.weight",
            f"{p}.ffn.down.bias":   f"{g}.ffn_down.bias",
        })
    # lm_head.weight is tied — omit (absent = tied in llama.cpp convention)
    return mapping


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

class TadpoleConfig:
    """Minimal config container parsed from checkpoint or JSON."""

    def __init__(
        self,
        vocab_size: int = 4096,
        max_seq_len: int = 128,
        d_model: int = 384,
        n_layers: int = 6,
        n_heads: int = 6,
        ffn_hidden: int = 768,
        dropout: float = 0.1,
        pad_id: int = 0,
        bos_id: int = 1,
        eos_id: int = 2,
    ):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.ffn_hidden = ffn_hidden
        self.dropout = dropout
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id

    @classmethod
    def from_dict(cls, d: dict) -> "TadpoleConfig":
        # Support both flat and nested {"model": {...}} layouts
        if "model" in d and isinstance(d["model"], dict):
            d = d["model"]
        return cls(**{k: v for k, v in d.items() if k in cls.__init__.__code__.co_varnames})

    def __repr__(self) -> str:
        return (
            f"TadpoleConfig(vocab={self.vocab_size}, seq={self.max_seq_len}, "
            f"d_model={self.d_model}, layers={self.n_layers}, heads={self.n_heads}, "
            f"ffn={self.ffn_hidden})"
        )


def load_checkpoint(checkpoint_path: str, config_override: str | None) -> tuple[dict, TadpoleConfig]:
    """Load state_dict and config from a .pt file."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Resolve config: CLI override > embedded in checkpoint > defaults
    if config_override:
        with open(config_override) as f:
            raw = json.load(f)
        config = TadpoleConfig.from_dict(raw)
    elif isinstance(ckpt, dict) and "config" in ckpt:
        raw = ckpt["config"]
        if hasattr(raw, "__dict__"):
            raw = raw.__dict__
        config = TadpoleConfig.from_dict(raw)
    else:
        print("Warning: no config found in checkpoint — using defaults")
        config = TadpoleConfig()

    # Resolve state_dict
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            state_dict = ckpt["model"]
        else:
            # Assume the entire dict is a state_dict (no metadata keys)
            state_dict = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
            if not state_dict:
                # Could be OrderedDict of tensors
                state_dict = ckpt
    else:
        # torch.nn.Module saved directly
        state_dict = ckpt.state_dict()

    return state_dict, config


# ---------------------------------------------------------------------------
# Random-init checkpoint for smoke testing
# ---------------------------------------------------------------------------

def make_random_checkpoint(config: TadpoleConfig) -> dict:
    """Return a random-init state_dict matching Tadpole's architecture."""
    d = config.d_model
    v = config.vocab_size
    s = config.max_seq_len
    f = config.ffn_hidden

    sd: dict[str, torch.Tensor] = {}
    sd["tok_emb.weight"] = torch.randn(v, d) * 0.02
    sd["pos_emb.weight"] = torch.randn(s, d) * 0.02

    for n in range(config.n_layers):
        p = f"blocks.{n}"
        sd[f"{p}.norm1.weight"] = torch.ones(d)
        sd[f"{p}.norm1.bias"] = torch.zeros(d)
        sd[f"{p}.attn.qkv.weight"] = torch.randn(3 * d, d) * 0.02
        sd[f"{p}.attn.qkv.bias"] = torch.zeros(3 * d)
        sd[f"{p}.attn.out.weight"] = torch.randn(d, d) * 0.02
        sd[f"{p}.attn.out.bias"] = torch.zeros(d)
        sd[f"{p}.norm2.weight"] = torch.ones(d)
        sd[f"{p}.norm2.bias"] = torch.zeros(d)
        sd[f"{p}.ffn.up.weight"] = torch.randn(f, d) * 0.02
        sd[f"{p}.ffn.up.bias"] = torch.zeros(f)
        sd[f"{p}.ffn.down.weight"] = torch.randn(d, f) * 0.02
        sd[f"{p}.ffn.down.bias"] = torch.zeros(d)

    sd["norm.weight"] = torch.ones(d)
    sd["norm.bias"] = torch.zeros(d)
    # lm_head is tied — omit
    return sd


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export(state_dict: dict, config: TadpoleConfig, out_path: str) -> None:
    weight_map = build_weight_map(config.n_layers)

    writer = GGUFWriter(out_path, arch="gpt2")

    # ---- Metadata ----
    writer.add_name("Tadpole")
    writer.add_description("Tadpole — tiny domain-specific assistant for the rama-zpu ecosystem")
    writer.add_block_count(config.n_layers)
    writer.add_context_length(config.max_seq_len)
    writer.add_embedding_length(config.d_model)
    writer.add_feed_forward_length(config.ffn_hidden)
    writer.add_head_count(config.n_heads)
    writer.add_vocab_size(config.vocab_size)
    writer.add_layer_norm_eps(1e-5)
    writer.add_file_type(GGMLQuantizationType.F32)
    writer.add_bos_token_id(config.bos_id)
    writer.add_eos_token_id(config.eos_id)
    writer.add_pad_token_id(config.pad_id)

    # ---- Tokenizer metadata ----
    writer.add_tokenizer_model("gpt2")

    # ---- Tensors ----
    skipped: list[str] = []
    written: list[str] = []
    unknown: list[str] = []

    for pt_key, tensor in state_dict.items():
        if pt_key == "lm_head.weight":
            skipped.append(pt_key)
            continue

        gguf_name = weight_map.get(pt_key)
        if gguf_name is None:
            unknown.append(pt_key)
            continue

        arr = tensor.to(torch.float32).detach().cpu().numpy()
        writer.add_tensor(gguf_name, arr)
        written.append(gguf_name)

    print(f"Written {len(written)} tensors:")
    for name in written:
        print(f"  {name}")

    if skipped:
        print(f"\nSkipped (tied weights): {skipped}")

    if unknown:
        print(f"\nWarning — unmapped keys (not written): {unknown}")

    # ---- Flush to disk ----
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\nExported to {out_path} ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(out_path: str) -> None:
    """Run gguf_dump.py on the output file and print the result."""
    # Search for gguf_dump.py in common locations
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "..", "llama.cpp", "gguf-py", "scripts", "gguf_dump.py"),
        os.path.join(os.path.dirname(__file__), "..", "..", "llama.cpp", "scripts", "gguf_dump.py"),
    ]
    dump_script = None
    for c in candidates:
        if os.path.exists(c):
            dump_script = os.path.abspath(c)
            break

    if dump_script is None:
        # Try installed gguf package scripts directory
        try:
            import importlib.util
            spec = importlib.util.find_spec("gguf")
            if spec and spec.origin:
                pkg_dir = os.path.dirname(spec.origin)
                for candidate in [
                    os.path.join(pkg_dir, "scripts", "gguf_dump.py"),
                    os.path.join(pkg_dir, "..", "scripts", "gguf_dump.py"),
                ]:
                    if os.path.exists(candidate):
                        dump_script = os.path.abspath(candidate)
                        break
        except Exception:
            pass

    if dump_script is None:
        # Fall back to running as a module
        print(f"\nValidating via python -m gguf.scripts.gguf_dump ...")
        result = subprocess.run(
            [sys.executable, "-m", "gguf.scripts.gguf_dump", out_path],
            capture_output=True, text=True
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        if result.returncode != 0:
            print(f"gguf_dump exited with code {result.returncode}", file=sys.stderr)
        else:
            print("Validation passed.")
        return

    print(f"\nValidating with {dump_script} ...")
    result = subprocess.run(
        [sys.executable, dump_script, out_path],
        capture_output=True, text=True
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        print(f"gguf_dump.py exited with code {result.returncode}", file=sys.stderr)
    else:
        print("Validation passed.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a Tadpole PyTorch checkpoint to GGUF for rama-zpu"
    )
    parser.add_argument(
        "--checkpoint", "-c",
        help="Path to .pt checkpoint file. Omit to use a random-init model (smoke test).",
    )
    parser.add_argument(
        "--out", "-o", required=True,
        help="Output GGUF file path (e.g. tadpole.gguf)",
    )
    parser.add_argument(
        "--config",
        help="Optional JSON config override. Supports flat or nested {\"model\":{...}} layout.",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Run gguf_dump.py on the output file after export.",
    )
    args = parser.parse_args()

    # --- Load or synthesize ---
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        state_dict, config = load_checkpoint(args.checkpoint, args.config)
    else:
        print("No checkpoint given — generating random-init weights for smoke test")
        config = TadpoleConfig()
        if args.config:
            with open(args.config) as f:
                config = TadpoleConfig.from_dict(json.load(f))
        state_dict = make_random_checkpoint(config)

    print(f"Config: {config}")
    param_count = sum(t.numel() for t in state_dict.values())
    print(f"Parameters: {param_count:,} ({param_count/1e6:.1f}M)")

    # --- Export ---
    export(state_dict, config, args.out)

    # --- Validate ---
    if args.validate:
        validate(args.out)


if __name__ == "__main__":
    main()
