# GobyLLM — The Smallest Model That Lives on the Edge

A 25M parameter language model with **learned early exit** and **tool calling**, designed to run on Raspberry Pi and edge devices. Named after the [Goby fish](https://en.wikipedia.org/wiki/Goby) — the smallest vertebrates on Earth that thrive in any environment.

> **First model under 50M parameters with trained early exit and OpenAI-compatible tool calling.**

```
"Turn on the bedroom light"
         │
         ▼
┌─────────────────────┐
│   GobyLLM (25.5M)   │  ← Runs on Raspberry Pi, ~100+ tok/s (C runtime)
│   Layer 1 ✓         │
│   Layer 2 ✓         │
│   Layer 3 ✓ EXIT    │  ← Router says "confident enough", skips layers 4-10
│   Layer 4 ⊘         │
│   ...    ⊘         │
│   Layer 10 ⊘        │
└─────────────────────┘
         │
         ▼
  light(on, bedroom)       ← Tool call extracted, JSON-repaired, schema-validated
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        GobyLLM                               │
│                                                              │
│  Token Embedding (8192 vocab × 512 dim, weight-tied)         │
│       │                                                      │
│       ▼                                                      │
│  ┌─────────────────────────────────────────────────────┐     │
│  │  10× Parallel Residual Block                        │     │
│  │                                                     │     │
│  │   Input ──→ RMSNorm ──┬──→ GQA Attention (8Q/2KV) ─┐     │
│  │                       │                             │     │
│  │                       └──→ SwiGLU FFN (512→960)  ──┤     │
│  │                                                     │     │
│  │   Output ←── Input + Attention + FFN ◄──────────────┘     │
│  │                                                     │     │
│  │   Exit Router (512→1, sigmoid) ──→ exit if conf > θ │     │
│  └─────────────────────────────────────────────────────┘     │
│       │                                                      │
│       ▼                                                      │
│  RMSNorm → LM Head (tied weights) → logits                  │
│                                                              │
│  Parameters: 25,504,266 (25.5M)                              │
│  Router overhead: 5,130 params (0.02%)                       │
└──────────────────────────────────────────────────────────────┘
```

### What makes it different

| Feature | Standard Transformer | GobyLLM |
|---|---|---|
| Residual | Sequential (attn → ffn) | **Parallel** (attn + ffn, PaLM-style) |
| Attention | Multi-head (8 KV heads) | **GQA** (8Q / 2KV — 4× less KV cache) |
| Layer execution | Always all layers | **Early exit** (router-gated per query) |
| Output parsing | Exact match | **JSON repair** + schema validation |
| Inference | Python / PyTorch | **C runtime** (mmap, KV cache, NEON SIMD) |

---

## How It Works

### Training Pipeline

```
Stanford Alpaca (52K)  ──┐
                         ├──→ BPE Tokenizer (8192 vocab) ──→ Train GobyLLM
Tool-calling data (15K) ─┘                                       │
                                                                  │
    ┌──── Auxiliary LM loss at every layer ◄──────────────────────┤
    │     (teaches model to decode from any depth)                │
    │                                                             │
    └──── Router BCE loss ◄───────────────────────────────────────┘
          (trains exit routers to predict when early exit matches full output)
```

**Key insight:** During training, every layer computes an auxiliary language modeling loss through the shared `lm_head`. This teaches the model to produce decodable representations at *any* depth — not just the final layer. The routers then learn which layer is "good enough" for each input.

### Inference Pipeline

```
User text ──→ BPE tokenize ──→ Forward through layers with KV cache
                                    │
                                    ├─ Layer 1: process + cache KV
                                    ├─ Layer 2: process + cache KV
                                    ├─ Layer 3: process + cache KV + router says EXIT ──→ logits
                                    │                                                      │
                                    │  (layers 4-10 skipped)                               ▼
                                    │                                              Sample next token
                                    │                                                      │
                                    └──────────────────────────────────────────── loop ◄────┘
```

### Tool Calling Flow

```
User: "It's freezing in here"
                │
                ▼
┌─────────────────────────────────────────────────────────┐
│ System prompt includes tool schemas:                     │
│ [{"name":"set_temperature","parameters":{...}},          │
│  {"name":"turn_on","parameters":{...}}, ...]             │
└─────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────┐
│ Model generates:                                         │
│ <think>User is cold, set heating to 23°C</think>        │
│ {"name":"set_temperature","arguments":{"temp":23}}       │
│ I've turned on heating to 23°C.                          │
└─────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────┐
│ JSON Repair Layer:                                       │
│   ✓ Fix broken quotes, trailing commas, unclosed braces │
│   ✓ Fuzzy-match tool names ("trun_on" → "turn_on")     │
│   ✓ Coerce types ("50" → 50 for integer params)         │
│   ✓ Validate against tool schema                         │
└─────────────────────────────────────────────────────────┘
                │
                ▼
        OpenAI-format response
        {"tool_calls": [{"function": {"name": "set_temperature", ...}}]}
```

---

## Three Runtime Options

```
                        Speed           Dependencies        Binary Size
                        ─────           ────────────        ───────────
  goby.c (C runtime)    100+ tok/s      None (libc)         ~51 KB
  rpi_runner.py          10-30 tok/s    torch, tokenizers    ~200 MB
  inference.py           1-3 tok/s      torch, tokenizers    ~200 MB
```

**C Runtime (`goby.c`)** — recommended for deployment:
- `mmap` model loading (instant startup)
- Pre-allocated KV cache (no malloc in hot loop)
- ARM NEON SIMD auto-detected
- Single file, compiles anywhere: `make`

**Python KV-cached (`rpi_runner.py`)** — for development:
- INT8 dynamic quantization
- KV cache + fixed-depth early exit
- OpenAI-compatible HTTP server

**Python naive (`inference.py`)** — for debugging:
- No optimizations, straightforward implementation
- OpenAI-compatible HTTP server

---

## Quick Start

### Train on Google Colab

1. Upload `goby_colab.ipynb` to [Google Colab](https://colab.research.google.com)
2. Set runtime to **T4 GPU**
3. **Run all cells** — downloads Alpaca data, trains tokenizer, trains model, exports C binary
4. Download `goby_llm.tar.gz`

### Deploy on Raspberry Pi

```bash
tar xzf goby_llm.tar.gz && cd goby

# Compile C runtime
make

# Run
./goby goby.bin -p "Turn on the kitchen lights"    # single prompt
./goby goby.bin -i                                  # interactive
./goby goby.bin -b                                  # benchmark
```

### OpenAI-Compatible Server

```bash
# Start server (Python, with KV cache + INT8 + early exit)
python3 rpi_runner.py --serve --port 8000

# Use from ANY OpenAI client
from openai import OpenAI
client = OpenAI(base_url="http://raspberrypi:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="goby-25m",
    messages=[{"role": "user", "content": "Turn on the bedroom light"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "light",
            "description": "Control a light",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["on", "off"]},
                    "room": {"type": "string"}
                },
                "required": ["action", "room"]
            }
        }
    }]
)

tool_call = response.choices[0].message.tool_calls[0]
print(tool_call.function.name)       # "light"
print(tool_call.function.arguments)  # '{"action": "on", "room": "bedroom"}'
```

Supports: `messages`, `tools`, `tool_choice` (`auto`/`none`/`required`/forced), `temperature`, `max_tokens`, `model`, `stop`. Returns standard `usage`, `finish_reason`, `tool_calls`.

---

## Why It Works

**Why early exit works at 25M:**
Small models have massive redundancy on easy inputs. "Turn on the lights" doesn't need 10 layers of processing — the representation converges by layer 2-3. The router learns to detect this convergence and skip the rest. Training with auxiliary losses at every layer ensures the model *can* decode from any depth.

**Why tool calling works at 25M:**
The model doesn't need to "understand" tools — it needs to pattern-match. Given a tool schema and a user command, it fills in the JSON template. 15K diverse tool-calling examples (combinatorially generated across 40+ tools, 18 rooms, 10 phrasing variants) teach the *pattern*, not specific tools. The JSON repair layer covers the remaining errors.

**Why the C runtime matters:**
Python/PyTorch adds ~50-200× overhead on RPi: interpreter, autograd graph, tensor metadata, dynamic dispatch, GIL. The C runtime eliminates all of it. `mmap` means the model loads instantly (OS pages weights on demand). Pre-allocated KV cache means zero `malloc` during generation.

---

## Pros & Cons

### Pros
- **Truly edge-native** — runs on RPi 4, RPi 5, any ARM/x86 device
- **100% offline** — no cloud, no internet, no API keys
- **OpenAI-compatible** — drop-in replacement for any OpenAI client
- **Novel architecture** — first <50M model with trained early exit
- **Fast** — C runtime with KV cache, SIMD, early exit
- **Self-contained training** — one Colab notebook, trains in ~12 min
- **JSON repair** — handles broken model output gracefully

### Cons
- **25M params is very small** — won't handle complex reasoning or nuanced conversation
- **Tool calling is pattern-matching** — works for simple commands, fails on ambiguous intent
- **No streaming support** — generates full response then returns
- **BPE tokenizer trained on small corpus** — may struggle with unusual text
- **Early exit depth varies** — inconsistent latency per query
- **Single-turn optimized** — multi-turn context is limited by 512 token window

---

## What's Next

- [ ] **Voice pipeline** — Vosk (speech-to-text) → GobyLLM → TTS, fully offline on RPi
- [ ] **Quantization-aware training** — INT8/INT4 from the start, not post-hoc
- [ ] **Hybrid SSM+Attention** — replace some layers with Mamba-style state-space blocks for constant-memory inference
- [ ] **Streaming** — token-by-token SSE streaming in the HTTP server
- [ ] **Fine-tuning CLI** — adapt to specific tool sets with a few examples
- [ ] **GGUF export** — compatibility with llama.cpp ecosystem
- [ ] **Larger variants** — GobyLLM-50M, GobyLLM-100M with same architecture

---

## Project Structure

```
gobyllm/
├── README.md                   This file
├── .gitignore
├── requirements.txt            Python dependencies
├── Makefile                    Top-level build (delegates to csrc/)
├── goby_colab.ipynb            One-click Colab training notebook
│
├── gobyllm/                    Python package
│   ├── __init__.py             Package exports: GobyConfig, GobyLLM
│   ├── __main__.py             CLI: python -m gobyllm {train,serve,chat,...}
│   ├── config.py               Model + training hyperparameters
│   ├── model.py                Architecture (GQA, early exit, parallel residual)
│   ├── dataset.py              PyTorch dataset + dataloader
│   ├── train.py                Training loop with early exit diagnostics
│   ├── generate_data.py        Tool-calling data generator (15K samples)
│   ├── prepare_data.py         Downloads Alpaca, trains tokenizer, merges data
│   ├── inference.py            Python inference server (OpenAI-compatible)
│   ├── rpi_runner.py           Optimized inference (KV cache, INT8, early exit)
│   ├── json_repair.py          JSON repair + schema validation + fuzzy matching
│   └── export_goby.py          Export .pt → .goby binary for C runtime
│
├── csrc/                       Native C runtime
│   ├── goby.c                  Inference engine (mmap, KV cache, NEON SIMD)
│   └── Makefile                Build: make -C csrc
│
└── tools/
    └── make_colab.py           Generates goby_colab.ipynb from source files
```

---

## Citation

If you use GobyLLM in your work:

```
@software{gobyllm2025,
  title  = {GobyLLM: The Smallest Model That Lives on the Edge},
  author = {Arman},
  year   = {2025},
  url    = {https://github.com/arman-bd/gobyllm}
}
```

---

## License

MIT
