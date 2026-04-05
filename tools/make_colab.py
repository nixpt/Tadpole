"""Generate the GuppyLM Colab training notebook."""

import json
import os
import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_file(path):
    with open(path) as f:
        return f.read()


def read_for_colab(path):
    """Read a Python file and flatten relative imports for Colab."""
    content = read_file(path)
    content = re.sub(r'from \.(\w+) import', r'from \1 import', content)
    return content


def cell(source, cell_type="code"):
    lines = source.split("\n")
    formatted = [line + "\n" if i < len(lines) - 1 else line for i, line in enumerate(lines)]
    base = {"cell_type": cell_type, "metadata": {}, "source": formatted}
    if cell_type == "code":
        base["outputs"] = []
        base["execution_count"] = None
    return base


def md(text):
    return cell(text, "markdown")


def code(text):
    return cell(text, "code")


# Source files to embed in the notebook
FILES = [
    ("config.py",    "guppylm/config.py"),
    ("model.py",     "guppylm/model.py"),
    ("dataset.py",   "guppylm/dataset.py"),
    ("train.py",     "guppylm/train.py"),
    ("inference.py", "guppylm/inference.py"),
]


def build():
    cells = []

    # ══════════════════════════════════════════════════════════════════
    #  HEADER
    # ══════════════════════════════════════════════════════════════════

    cells.append(md(
        "# GuppyLM — Your Friendly Fish\n"
        "\n"
        "Train a ~9M parameter LLM that talks like a small fish.\n"
        "\n"
        "**What this notebook does:**\n"
        "1. Downloads 60K fish conversation dataset from HuggingFace\n"
        "2. Trains a BPE tokenizer on the data\n"
        "3. Trains a 6-layer vanilla transformer (8.7M params)\n"
        "4. Tests the model with sample conversations\n"
        "\n"
        "**Architecture:** 6 layers, 384 dim, 6 heads, ReLU FFN, LayerNorm, 4096 vocab\n"
        "\n"
        "**Runtime:** ~5 min on T4 GPU\n"
        "\n"
        "**Result:** A fish that speaks in short lowercase sentences about water, food, and light."
    ))

    # ══════════════════════════════════════════════════════════════════
    #  1. SETUP
    # ══════════════════════════════════════════════════════════════════

    cells.append(md(
        "## 1. Setup\n"
        "\n"
        "Install dependencies and create a clean working directory."
    ))

    cells.append(code(
        "!pip install -q torch tokenizers tqdm numpy datasets huggingface_hub\n"
        "\n"
        "import torch\n"
        "print(f'PyTorch {torch.__version__}')\n"
        "print(f'CUDA: {torch.cuda.is_available()}')\n"
        "if torch.cuda.is_available():\n"
        "    print(f'GPU: {torch.cuda.get_device_name(0)}')"
    ))

    cells.append(code(
        "import os, shutil\n"
        "\n"
        "# Start fresh — removes stale files from previous runs\n"
        "if os.path.exists('/content/guppy'):\n"
        "    shutil.rmtree('/content/guppy')\n"
        "os.makedirs('/content/guppy')\n"
        "os.chdir('/content/guppy')\n"
        "print(f'Working dir: {os.getcwd()}')"
    ))

    # ══════════════════════════════════════════════════════════════════
    #  2. SOURCE FILES
    # ══════════════════════════════════════════════════════════════════

    cells.append(md(
        "## 2. Source Files\n"
        "\n"
        "Write the model code to disk. These are the only files needed:\n"
        "- `config.py` — model and training hyperparameters\n"
        "- `model.py` — transformer architecture\n"
        "- `dataset.py` — data loading and batching\n"
        "- `train.py` — training loop\n"
        "- `inference.py` — chat interface"
    ))

    for display_name, src_path in FILES:
        full_path = os.path.join(PROJECT_ROOT, src_path)
        content = read_for_colab(full_path)
        cells.append(code(f"%%writefile {display_name}\n{content}"))

    # ══════════════════════════════════════════════════════════════════
    #  3. PREPARE DATA
    # ══════════════════════════════════════════════════════════════════

    cells.append(md(
        "## 3. Prepare Data\n"
        "\n"
        "Download the fish conversation dataset from HuggingFace and train a BPE tokenizer.\n"
        "\n"
        "The dataset has 60K single-turn conversations across 60 topics:\n"
        "greetings, food, temperature, water, tank life, emotions, philosophy (fish-level), and more.\n"
        "\n"
        "Each sample is formatted as ChatML:\n"
        "```\n"
        "<|im_start|>user\n"
        "hi guppy<|im_end|>\n"
        "<|im_start|>assistant\n"
        "hello. the water is nice today.<|im_end|>\n"
        "```"
    ))

    cells.append(code(
        "import json, os\n"
        "from datasets import load_dataset\n"
        "from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors\n"
        "\n"
        "# ── Download from HuggingFace ──\n"
        "HF_DATASET = 'arman-bd/guppylm-60k-generic'\n"
        "ds = load_dataset(HF_DATASET)\n"
        "print(f'Downloaded: {len(ds[\"train\"]):,} train, {len(ds[\"test\"]):,} test samples')\n"
        "\n"
        "# ── Format into ChatML and save as JSONL ──\n"
        "os.makedirs('data', exist_ok=True)\n"
        "texts = []\n"
        "\n"
        "for split, path in [('train', 'data/train.jsonl'), ('test', 'data/eval.jsonl')]:\n"
        "    with open(path, 'w') as f:\n"
        "        for row in ds[split]:\n"
        "            text = (\n"
        "                f'<|im_start|>user\\n{row[\"input\"]}<|im_end|>\\n'\n"
        "                f'<|im_start|>assistant\\n{row[\"output\"]}<|im_end|>'\n"
        "            )\n"
        "            f.write(json.dumps({'text': text, 'category': row['category']}) + '\\n')\n"
        "            texts.append(text)\n"
        "    print(f'  {path}: {len(ds[split]):,} samples')\n"
        "\n"
        "# ── Train BPE tokenizer on the data ──\n"
        "tokenizer = Tokenizer(models.BPE())\n"
        "tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)\n"
        "tokenizer.decoder = decoders.ByteLevel()\n"
        "\n"
        "trainer = trainers.BpeTrainer(\n"
        "    vocab_size=4096,\n"
        "    special_tokens=['<pad>', '<|im_start|>', '<|im_end|>'],\n"
        "    min_frequency=2,\n"
        "    show_progress=True,\n"
        ")\n"
        "tokenizer.train_from_iterator(texts, trainer)\n"
        "tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)\n"
        "tokenizer.save('data/tokenizer.json')\n"
        "print(f'  Tokenizer: {tokenizer.get_vocab_size()} tokens')\n"
        "\n"
        "# ── Preview ──\n"
        "with open('data/train.jsonl') as f:\n"
        "    sample = json.loads(f.readline())\n"
        "print(f'\\nSample ({sample[\"category\"]}):\\n{sample[\"text\"]}')"
    ))

    # ══════════════════════════════════════════════════════════════════
    #  4. VERIFY ARCHITECTURE
    # ══════════════════════════════════════════════════════════════════

    cells.append(md(
        "## 4. Verify Architecture\n"
        "\n"
        "Quick sanity check — build the model, print param count, run a dummy forward pass."
    ))

    cells.append(code(
        "from config import GuppyConfig\n"
        "from model import GuppyLM\n"
        "import torch\n"
        "\n"
        "config = GuppyConfig()\n"
        "model = GuppyLM(config)\n"
        "print(model.param_summary())\n"
        "print(f'  Layers: {config.n_layers}, Heads: {config.n_heads}, FFN: {config.ffn_hidden}')\n"
        "print(f'  Vocab: {config.vocab_size}, Max seq: {config.max_seq_len}')\n"
        "\n"
        "# Dummy forward pass\n"
        "x = torch.randint(0, config.vocab_size, (2, 32))\n"
        "logits, _ = model(x)\n"
        "print(f'  Forward pass: {x.shape} -> {logits.shape} OK')\n"
        "del model"
    ))

    # ══════════════════════════════════════════════════════════════════
    #  5. TRAIN
    # ══════════════════════════════════════════════════════════════════

    cells.append(md(
        "## 5. Train\n"
        "\n"
        "10,000 steps with cosine LR schedule. Takes ~2 min on T4.\n"
        "\n"
        "The model learns to:\n"
        "- Respond in short, lowercase sentences\n"
        "- Stay in character as a fish\n"
        "- Cover 60 different conversation topics\n"
        "- Stop generating at the right time (learn the `<|im_end|>` token)"
    ))

    cells.append(code("from train import train\ntrain()"))

    # ══════════════════════════════════════════════════════════════════
    #  6. TEST
    # ══════════════════════════════════════════════════════════════════

    cells.append(md(
        "## 6. Test\n"
        "\n"
        "Chat with the trained model. Each message is independent (single-turn)."
    ))

    cells.append(code(
        "from inference import GuppyInference\n"
        "import torch\n"
        "\n"
        "engine = GuppyInference(\n"
        "    'checkpoints/best_model.pt', 'data/tokenizer.json',\n"
        "    device='cuda' if torch.cuda.is_available() else 'cpu'\n"
        ")\n"
        "\n"
        "def chat(prompt):\n"
        "    r = engine.chat_completion([{'role': 'user', 'content': prompt}], max_tokens=64)\n"
        "    return r['choices'][0]['message'].get('content', '').strip()\n"
        "\n"
        "# Test across different topics\n"
        "tests = [\n"
        "    ('hi guppy',                      'greeting'),\n"
        "    ('are you hungry',                'food'),\n"
        "    ('it is really hot today',        'temperature'),\n"
        "    ('how is the water',              'water'),\n"
        "    ('do you like bubbles',           'bubbles'),\n"
        "    ('what is the internet',          'confused'),\n"
        "    ('do you get lonely',             'lonely'),\n"
        "    ('the cat is looking at you',     'cat'),\n"
        "    ('tell me a joke',                'joke'),\n"
        "    ('what do you dream about',       'dreams'),\n"
        "    ('do you love me',                'love'),\n"
        "    ('what is the meaning of life',   'meaning'),\n"
        "    ('sorry i tapped the glass',      'glass_tap'),\n"
        "    ('it is raining outside',         'rain'),\n"
        "    ('goodnight guppy',               'night'),\n"
        "]\n"
        "\n"
        "print(f'{\"Topic\":<12s}  {\"You\":<35s}  Guppy')\n"
        "print('=' * 100)\n"
        "for prompt, topic in tests:\n"
        "    reply = chat(prompt)\n"
        "    print(f'{topic:<12s}  {prompt:<35s}  {reply[:128]}')\n"
    ))

    # ══════════════════════════════════════════════════════════════════
    #  7. UPLOAD TO HUGGINGFACE
    # ══════════════════════════════════════════════════════════════════

    cells.append(md(
        "## 7. Upload to HuggingFace\n"
        "\n"
        "Push the trained model to HuggingFace Hub.\n"
        "Set your token and repo below."
    ))

    cells.append(code(
        "from huggingface_hub import HfApi, login\n"
        "import os\n"
        "\n"
        "HF_TOKEN = os.environ.get('HF_TOKEN', '')  # Or paste your token here\n"
        "HF_REPO = os.environ.get('HF_REPO', 'arman-bd/guppylm-9M')  # Or change this\n"
        "\n"
        "if not HF_TOKEN:\n"
        "    print('No HF_TOKEN found. Set it as env var or paste above. Skipping.')\n"
        "else:\n"
        "    login(token=HF_TOKEN)\n"
        "    api = HfApi()\n"
        "    api.create_repo(HF_REPO, exist_ok=True)\n"
        "\n"
        "    # Upload model checkpoint, tokenizer, and source files\n"
        "    for path in ['checkpoints/best_model.pt', 'checkpoints/config.json',\n"
        "                 'data/tokenizer.json', 'model.py', 'config.py', 'inference.py']:\n"
        "        if os.path.exists(path):\n"
        "            api.upload_file(\n"
        "                path_or_fileobj=path,\n"
        "                path_in_repo=path,\n"
        "                repo_id=HF_REPO,\n"
        "            )\n"
        "            print(f'  Uploaded {path}')\n"
        "\n"
        "    print(f'\\nDone! https://huggingface.co/{HF_REPO}')"
    ))

    # ══════════════════════════════════════════════════════════════════
    #  8. DOWNLOAD
    # ══════════════════════════════════════════════════════════════════

    cells.append(md(
        "## 8. Download\n"
        "\n"
        "Or download the model locally as a tar.gz."
    ))

    cells.append(code(
        "import os\n"
        "\n"
        "!cd /content && tar czf guppylm.tar.gz \\\n"
        "    guppy/checkpoints/best_model.pt \\\n"
        "    guppy/checkpoints/config.json \\\n"
        "    guppy/data/tokenizer.json \\\n"
        "    guppy/model.py \\\n"
        "    guppy/config.py \\\n"
        "    guppy/inference.py\n"
        "\n"
        "sz = os.path.getsize('/content/guppylm.tar.gz') / 1e6\n"
        "print(f'Package: /content/guppylm.tar.gz ({sz:.1f} MB)')\n"
        "\n"
        "try:\n"
        "    from google.colab import files\n"
        "    files.download('/content/guppylm.tar.gz')\n"
        "except ImportError:\n"
        "    print('Not in Colab — download manually from the file browser.')"
    ))

    # ══════════════════════════════════════════════════════════════════

    return {
        "nbformat": 4, "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": [], "gpuType": "T4", "name": "GuppyLM — Train"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": cells,
    }


def build_use():
    """Build the use_guppylm notebook — download model from HF and chat."""
    cells = []

    cells.append(md(
        "# GuppyLM — Chat with a Fish\n"
        "\n"
        "Download the pre-trained GuppyLM model from HuggingFace and chat with it.\n"
        "No training needed — just run all cells.\n"
        "\n"
        "**Model:** [arman-bd/guppylm-9M](https://huggingface.co/arman-bd/guppylm-9M) (8.7M params)"
    ))

    # ── Setup ──────────────────────────────────────────────────────

    cells.append(md("## 1. Setup"))

    cells.append(code(
        "!pip install -q torch tokenizers huggingface_hub\n"
        "\n"
        "import torch\n"
        "print(f'PyTorch {torch.__version__}')\n"
        "print(f'CUDA: {torch.cuda.is_available()}')"
    ))

    cells.append(code(
        "import os, shutil\n"
        "if os.path.exists('/content/guppy'):\n"
        "    shutil.rmtree('/content/guppy')\n"
        "os.makedirs('/content/guppy')\n"
        "os.chdir('/content/guppy')\n"
        "print(f'Working dir: {os.getcwd()}')"
    ))

    # ── Download model ─────────────────────────────────────────────

    cells.append(md(
        "## 2. Download Model\n"
        "\n"
        "Download the pre-trained model, tokenizer, and inference code from HuggingFace."
    ))

    cells.append(code(
        "from huggingface_hub import hf_hub_download\n"
        "import os\n"
        "\n"
        "HF_REPO = 'arman-bd/guppylm-9M'\n"
        "\n"
        "files = [\n"
        "    ('checkpoints/best_model.pt', 'checkpoints'),\n"
        "    ('checkpoints/config.json', 'checkpoints'),\n"
        "    ('data/tokenizer.json', 'data'),\n"
        "    ('model.py', '.'),\n"
        "    ('config.py', '.'),\n"
        "    ('inference.py', '.'),\n"
        "]\n"
        "\n"
        "for filename, subdir in files:\n"
        "    os.makedirs(subdir, exist_ok=True)\n"
        "    hf_hub_download(\n"
        "        repo_id=HF_REPO,\n"
        "        filename=filename,\n"
        "        local_dir='.',\n"
        "    )\n"
        "    print(f'  Downloaded {filename}')\n"
        "\n"
        "print(f'\\nModel ready from {HF_REPO}')"
    ))

    # ── Chat ───────────────────────────────────────────────────────

    cells.append(md(
        "## 3. Chat\n"
        "\n"
        "Talk to Guppy. Each message is independent (single-turn)."
    ))

    cells.append(code(
        "from inference import GuppyInference\n"
        "import torch\n"
        "\n"
        "engine = GuppyInference(\n"
        "    'checkpoints/best_model.pt', 'data/tokenizer.json',\n"
        "    device='cuda' if torch.cuda.is_available() else 'cpu'\n"
        ")\n"
        "\n"
        "def chat(prompt):\n"
        "    r = engine.chat_completion([{'role': 'user', 'content': prompt}], max_tokens=64)\n"
        "    return r['choices'][0]['message'].get('content', '').strip()"
    ))

    cells.append(code(
        "# Try different topics\n"
        "tests = [\n"
        "    'hi guppy',\n"
        "    'are you hungry',\n"
        "    'it is really hot today',\n"
        "    'do you like bubbles',\n"
        "    'what is the meaning of life',\n"
        "    'tell me a joke',\n"
        "    'the cat is looking at you',\n"
        "    'do you love me',\n"
        "    'what is the internet',\n"
        "    'goodnight guppy',\n"
        "]\n"
        "\n"
        "for prompt in tests:\n"
        "    reply = chat(prompt)\n"
        "    print(f'You> {prompt}')\n"
        "    print(f'Guppy> {reply}')\n"
        "    print()"
    ))

    cells.append(md(
        "## 4. Interactive Chat\n"
        "\n"
        "Type your own messages. Type `quit` to stop."
    ))

    cells.append(code(
        "print('Chat with Guppy (type quit to stop)')\n"
        "print('=' * 40)\n"
        "while True:\n"
        "    try:\n"
        "        prompt = input('You> ').strip()\n"
        "    except (KeyboardInterrupt, EOFError):\n"
        "        break\n"
        "    if not prompt or prompt.lower() in ('quit', 'exit', 'q'):\n"
        "        print('Guppy> bye. i will continue being a fish.')\n"
        "        break\n"
        "    reply = chat(prompt)\n"
        "    print(f'Guppy> {reply}')\n"
        "    print()"
    ))

    return {
        "nbformat": 4, "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": [], "name": "GuppyLM — Chat"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }


def write_notebook(nb, filename):
    out = os.path.join(PROJECT_ROOT, filename)
    with open(out, "w") as f:
        json.dump(nb, f, indent=1)
    n = len(nb["cells"])
    sz = os.path.getsize(out) / 1024
    print(f"Generated {out}: {n} cells, {sz:.1f} KB")


if __name__ == "__main__":
    write_notebook(build(), "train_guppylm.ipynb")
    write_notebook(build_use(), "use_guppylm.ipynb")
