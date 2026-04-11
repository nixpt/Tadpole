"""Shared helpers for Tadpole chat runtime files."""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_PATH = PACKAGE_ROOT / "checkpoints" / "best_model.pt"
DEFAULT_TOKENIZER_PATH = PACKAGE_ROOT / "data" / "tokenizer.json"
DOCS_TOKENIZER_PATH = PACKAGE_ROOT / "docs" / "tokenizer.json"


def download_model() -> None:
    """Download the base model and tokenizer to the default runtime paths."""
    import urllib.request

    hf_repo = "arman-bd/guppylm-9M"
    hf_base = f"https://huggingface.co/{hf_repo}/resolve/main"

    files = [
        (f"{hf_base}/pytorch_model.bin", str(DEFAULT_CHECKPOINT_PATH)),
        (f"{hf_base}/tokenizer.json", str(DEFAULT_TOKENIZER_PATH)),
        (f"{hf_base}/config.json", str(PACKAGE_ROOT / "checkpoints" / "config.json")),
    ]

    print(f"Downloading upstream model from {hf_repo}...\n")
    for url, dest in files:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        name = os.path.basename(dest)
        print(f"  {name}...", end=" ", flush=True)
        urllib.request.urlretrieve(url, dest)
        size_mb = os.path.getsize(dest) / 1e6
        print(f"{size_mb:.1f} MB")

    print("\nDone! Run: python -m tadpole chat")


def resolve_chat_assets(checkpoint_path: str, tokenizer_path: str) -> tuple[str, str]:
    """Resolve the files needed for chat, downloading defaults if necessary."""

    checkpoint = Path(checkpoint_path)
    tokenizer = Path(tokenizer_path)
    use_defaults = checkpoint_path == str(DEFAULT_CHECKPOINT_PATH) and tokenizer_path == str(DEFAULT_TOKENIZER_PATH)

    if not tokenizer.exists():
        if DOCS_TOKENIZER_PATH.exists():
            tokenizer = DOCS_TOKENIZER_PATH
        elif use_defaults:
            download_model()
            tokenizer = DEFAULT_TOKENIZER_PATH
        else:
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    if not checkpoint.exists():
        if use_defaults:
            download_model()
            checkpoint = DEFAULT_CHECKPOINT_PATH
            if DEFAULT_TOKENIZER_PATH.exists():
                tokenizer = DEFAULT_TOKENIZER_PATH
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    return str(checkpoint), str(tokenizer)
