<h1 align="center">🐸 Tadpole</h1>
<p align="center"><em>A ~9M parameter LLM for rama-zpu. Small now, but ready to metamorph with training.</em></p>

<p align="center">
  <a href="https://github.com/arman-bd/guppylm"><img src="https://img.shields.io/badge/Forked_from-GuppyLM-blue" alt="Fork"/></a>&nbsp;
  <a href="https://github.com/arman-bd/guppylm/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" alt="License"/></a>&nbsp;
  <a href="FORK.md"><img src="https://img.shields.io/badge/Fork_policy-No_upstream_sync-orange" alt="Fork Policy"/></a>
</p>

---

> **Tadpole is a technical assistant for the rama-zpu memory orchestration ecosystem.**
> 
> It's tiny now (9M params) — trained on rama commands, zram operations, compression workflows, and inference pipelines. Runs entirely in compressed memory (zram + lz4), eating its own dogfood.
>
> Like its namesake, Tadpole is designed to **metamorph**: retrain it on your domain-specific documentation and it grows into a specialized assistant for your stack.

---

```bash
You> how do I create a compressed volume?
Tadpole> command: rama volume create <name> -p gguf
         creates 16G volume with lz4 compression
         presets: gguf (16G), tmp (4G), code (8G)

You> what compression ratios can I expect?
Tadpole> typical ratios:
         text/code: 3-5x (lz4), 5-8x (zstd)
         gguf models: 2-3x
         check: rama volume stats <name>

You> my volume is read-only
Tadpole> check mount options: cat /proc/mounts | grep <name>
         remount rw: sudo mount -o remount,rw /mnt/<name>
         if overlayfs, workdir may be full

You> how does rama-zpu load models?
Tadpole> rama-zpu reads GGUF/RZM from zram-backed volumes
         mmap → page cache → zram → decompression on access
         keeps hot tensors uncompressed, cold ones in zram

You> goodbye
Tadpole> done. rama volume ls shows what's active.
```

---

## What is Tadpole?

Tadpole is a tiny language model specialized for **rama-zpu** — a memory orchestration system using zram, overlayfs, and compression for efficient model inference.

It answers questions about:
- **rama volumes** (create, mount, resize, presets)
- **zram operations** (compression, swap, stats)
- **Model inference** (GGUF/RZM loading, quantization, mmap)
- **Troubleshooting** (read-only mounts, OOM, overlayfs issues)
- **Compression workflows** (lz4 vs zstd, ratios, tuning)

**Current status:** Pre-training. Once trained on rama documentation, it will provide instant command help without leaving the terminal.

**Metamorphosis design:** Retrain Tadpole on your docs (Kubernetes, PostgreSQL, Rust crates) and it becomes a domain expert for your stack — same architecture, different knowledge.

---

## Architecture

| | |
|---|---|
| **Parameters** | 8.7M |
| **Layers** | 6 |
| **Hidden dim** | 384 |
| **Heads** | 6 |
| **FFN** | 768 (ReLU) |
| **Vocab** | 4,096 (BPE) |
| **Max sequence** | 128 tokens |
| **Norm** | LayerNorm |
| **Position** | Learned embeddings |
| **LM head** | Weight-tied with embeddings |

Vanilla transformer. No GQA, no RoPE, no SwiGLU, no early exit. As simple as it gets.

---

## Voice & Knowledge

**Voice:**
- Concise technical responses (1-3 sentences)
- Commands first, explanations second
- Acknowledges limitations ("check docs for X")
- Self-aware about running on rama-zpu in compressed memory

**Knowledge base (60 topics):**
- **rama volumes** (10): create, mount, presets, resize, stats, list, destroy, snapshots, quotas, permissions
- **zram** (8): compression algorithms, ratios, stats, swap, tuning, fragmentation, OOM behavior, disksize
- **inference** (10): GGUF/RZM loading, quantization, mmap, page cache, model formats, tensor layout, memory bandwidth, kv-cache
- **compression** (6): lz4 vs zstd, ratios per filetype, comp_algorithm tuning, parallel compression, adaptive compression, dedup
- **config** (6): .ramarc structure, presets, environment vars, CLI precedence, workspace isolation, profiles
- **workspaces** (5): overlayfs mechanics, upperdir/lowerdir/workdir, copy-on-write, workspace create/destroy, snapshot workflow
- **system** (5): kernel params, vm.swappiness, page eviction, cgroup limits, systemd integration
- **troubleshooting** (5): read-only mounts, OOM kills, workdir full, mount failures, compression disabled
- **use cases** (3): model serving, build acceleration, interactive development
- **meta** (2): self-awareness (runs on rama-zpu), limitations (9M params, check docs for edge cases)

---

## Quick Start

**Status:** Pre-training. Model checkpoint not yet available.

### Training (once data is ready)

```bash
# Install dependencies
pip install torch tokenizers

# Generate training data from rama docs
python -m tadpole prepare-data

# Train on rama documentation (5 min on T4 GPU)
python -m tadpole train

# Export to GGUF for rama-zpu
python tools/export_gguf.py

# Convert GGUF → RZM (rama optimized format)
rama-convert model.gguf model.rzm
```

### Chat (after training)

```bash
python -m tadpole chat
```

```bash
You> how do I check zram stats?
Tadpole> command: cat /sys/block/zram0/mm_stat
         shows: orig_data_size, compr_data_size, mem_used_total
         compression ratio = orig / compr

You> what preset should I use for GGUF models?
Tadpole> preset: gguf (16G, lz4, fast decompression)
         creates compressed volume optimized for mmap workloads
         command: rama volume create models -p gguf
```

Single-prompt mode:

```bash
python -m tadpole chat --prompt "how do I resize a volume?"
# Tadpole> command: rama volume resize <name> <new-size>
#          supports K/M/G units (e.g., 32G)
#          volume must be unmounted first
```


---

## Dataset

**Status:** In development. Not yet published.

| | |
|---|---|
| Planned samples | 60,000 (57K train / 3K test) |
| Format | `{"input": "...", "output": "...", "category": "..."}` |
| Categories | 60 (rama-focused) |
| Generation | Synthetic Q&A from rama documentation |

**Sources:**
- `/home/nixp/WORKSPACE/rama/README.md` — volume commands, presets, workflows
- `/home/nixp/WORKSPACE/rama/CLAUDE.md` — build rules, troubleshooting, agent context
- `rama --help` output — CLI reference
- `TADPOLE_PERSONALITY.md` — voice guidelines and sample conversations

**Example:**
```json
{
  "input": "how do I create a compressed volume?",
  "output": "command: rama volume create <name> -p gguf\ncreates 16G volume with lz4 compression\npresets: gguf (16G), tmp (4G), code (8G)",
  "category": "volumes-create"
}
```

---

## Project Structure

```
tadpole/
├── config.py               Hyperparameters (TadpoleConfig + TrainConfig)
├── model.py                Vanilla transformer (Tadpole class)
├── dataset.py              Data loading + batching
├── train.py                Training loop (cosine LR, AMP)
├── generate_data.py        rama Q&A data generator (60 topics)
├── eval_cases.py           Held-out test cases
├── prepare_data.py         Data prep + tokenizer training
└── inference.py            Chat interface (TadpoleInference)

tools/
├── export_dataset.py       Push dataset to HuggingFace
├── export_model.py         Push model to HuggingFace
└── export_gguf.py          Export to GGUF format (planned)

FORK.md                     Divergence from upstream GuppyLM
TADPOLE_PERSONALITY.md      Voice guide + sample conversations
rama_topics.py              60 technical topics taxonomy
rama_samples.py             44 sample Q&A conversations
```

---

## Design Decisions

**Why fork GuppyLM?** The upstream model architecture is perfect for tiny domain experts: 9M params, 128 tokens, vanilla transformer. Only the personality changed (fish → technical assistant). See [FORK.md](FORK.md) for divergence policy.

**Why no system prompt?** A 9M model can't conditionally follow instructions — the voice is baked into the weights. Removing system prompts saves ~60 tokens per inference.

**Why single-turn only?** Multi-turn degrades at turn 3-4 due to 128-token context. For quick command lookups, single-turn is more reliable than context management.

**Why vanilla transformer?** GQA, SwiGLU, RoPE add complexity that doesn't help at 9M params. Standard attention + ReLU FFN + LayerNorm is simpler to implement in rama-zpu.

**Why synthetic data?** rama documentation is concise but not conversational. Synthetic Q&A from command help text creates consistent training signal for technical voice.

**Metamorphosis design:** Swap out `rama_topics.py` and regenerate data on your domain (Kubernetes, databases, compiler flags) — same architecture, different expertise. Tadpole → specialized assistant.

---

---

## Roadmap

- [x] Fork from GuppyLM
- [x] Design rama-focused personality
- [x] Create 60 technical topics taxonomy
- [x] Generate 44 sample conversations
- [x] Rename package (guppylm → tadpole)
- [ ] Generate full 60K training dataset from rama docs
- [ ] Train on rama data (5 min on T4 GPU)
- [ ] Export to GGUF format
- [ ] Implement architecture in rama-zpu (Rust)
- [ ] Convert to RZM (rama optimized format)
- [ ] Deploy on zram with lz4 compression
- [ ] Integrate with rama CLI (`rama ask "how do I..."`)

---

## Upstream

Forked from **[arman-bd/guppylm](https://github.com/arman-bd/guppylm)** (MIT License).

Tadpole **does not sync** with upstream. See [FORK.md](FORK.md) for rationale and divergence policy.

---

## License

MIT (same as upstream)
