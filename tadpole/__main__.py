"""Entry point for: python -m tadpole"""

import os
import sys

CHECKPOINT_PATH = "checkpoints/best_model.pt"
TOKENIZER_PATH = "data/tokenizer.json"
# Note: Original upstream model repo (not maintained)
HF_REPO = "arman-bd/guppylm-9M"
HF_BASE = f"https://huggingface.co/{HF_REPO}/resolve/main"


def download_model():
    """Download base model from upstream (GuppyLM - for reference only)."""
    import urllib.request

    files = [
        (f"{HF_BASE}/pytorch_model.bin", CHECKPOINT_PATH),
        (f"{HF_BASE}/tokenizer.json", TOKENIZER_PATH),
        (f"{HF_BASE}/config.json", "checkpoints/config.json"),
    ]

    print(f"Downloading upstream model from {HF_REPO}...\n")
    for url, dest in files:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        name = os.path.basename(dest)
        print(f"  {name}...", end=" ", flush=True)
        urllib.request.urlretrieve(url, dest)
        size_mb = os.path.getsize(dest) / 1e6
        print(f"{size_mb:.1f} MB")

    print("\nDone! Run: python -m tadpole chat")


def main():
    if len(sys.argv) < 2:
        print("Tadpole — A tiny rama-zpu assistant")
        print()
        print("Usage:")
        print("  python -m tadpole train        Train the model")
        print("  python -m tadpole prepare      Generate data & train tokenizer")
        print("  python -m tadpole chat         Chat with Tadpole")
        print("  python -m tadpole download     Download base model (upstream reference)")
        return

    cmd = sys.argv[1]
    sys.argv = sys.argv[1:]

    if cmd == "prepare":
        from .prepare_data import prepare
        prepare()

    elif cmd == "train":
        from .train import train
        train()

    elif cmd == "download":
        download_model()

    elif cmd == "chat":
        if not os.path.exists(CHECKPOINT_PATH):
            print("Model not found. Train your own:\n")
            print("  python -m tadpole prepare")
            print("  python -m tadpole train\n")
            print("Or download upstream base model (reference only):\n")
            print("  python -m tadpole download")
            return

        from .inference import main as inference_main
        inference_main()

    else:
        print(f"Unknown command: {cmd}")
        print("Run 'python -m tadpole' for usage.")


main()
