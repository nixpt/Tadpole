"""
Rama assistant sample training data — technical Q&A based on actual rama commands.

These samples prove the technical assistant personality works.
"""

import json

SAMPLE_DATA = [
    # ── rama Volume Management ───────────────────────────────────────────
    {"input": "how do i create a volume", "output": "command: rama volume create <name> -p <preset>\ncommon presets: gguf (16G, lz4), tmp (4G, lz4), swap (ram/2, lz4)\nexample: rama volume create models -p gguf", "category": "volume_create"},
    {"input": "create a volume for models", "output": "rama volume create models -p gguf\nthis creates a 16G volume with lz4 compression at /models\nthen: rama volume start models to mount it", "category": "volume_create"},
    {"input": "how big should a model volume be", "output": "for LLMs: 16G is good (gguf preset)\nfor small models like me (9M params): 4G is enough\ncheck usage: rama volume ls shows allocated vs used", "category": "volume_best_practices"},
    
    {"input": "start a volume", "output": "rama volume start <name>\nexample: rama volume start models\nthis mounts the volume at its configured mount point", "category": "volume_lifecycle"},
    {"input": "stop a volume", "output": "rama volume stop <name>\nexample: rama volume stop models\nunmounts the volume (data is preserved)", "category": "volume_lifecycle"},
    
    {"input": "list all volumes", "output": "rama volume ls\nshows: name, size, algorithm, mount point, status\nalso shows compression ratio and usage", "category": "volume_list"},
    {"input": "check volume usage", "output": "rama volume ls shows usage for all volumes\nfor detailed breakdown: df -h /models (if mounted)", "category": "volume_list"},
    
    {"input": "resize a volume", "output": "rama volume resize <name> <new-size>\nexample: rama volume resize models 32G\nvolume must be stopped first: rama volume stop models", "category": "volume_resize"},
    
    {"input": "delete a volume", "output": "rama volume rm <name>\nwarning: deletes all data\nstop it first: rama volume stop <name>", "category": "volume_delete"},
    
    {"input": "copy files to volume", "output": "rama volume copy <name> <source-path>\nexample: rama volume copy models /path/to/model.gguf\nuses parallel compression (pigz/pzstd) for speed", "category": "volume_copy"},
    
    {"input": "what are the volume presets", "output": "gguf: 16G, lz4, /models (for LLMs)\ntmp: 4G, lz4, /tmp (ephemeral)\ncache: 8G, lz4 (general cache)\nswap: ram/2, lz4 (swap space)\nlog: 4G, zstd, /var/log (logs)", "category": "volume_presets"},
    
    {"input": "where are volumes mounted", "output": "depends on preset:\ngguf → /models\ntmp → /tmp\ncache → /cache\ncustom: specify with -m flag: rama volume create myapp -s 8G -m /myapp", "category": "volume_mounts"},
    
    {"input": "volume won't mount", "output": "check: rama volume ls (shows error if any)\nlikely causes: directory already exists, permission denied, device busy\nfix: remove mount point, check permissions, ensure volume is stopped first", "category": "volume_troubleshoot"},
    
    # ── Zram Operations ──────────────────────────────────────────────────
    {"input": "what is zram", "output": "zram creates compressed RAM block devices\ndata is compressed in memory (not on disk)\nuse cases: swap, tmpfs, model storage, build environments\ntypical compression: 2-3x with lz4, 3-5x with zstd", "category": "zram_basics"},
    {"input": "how does zram work", "output": "zram creates a block device (like /dev/zram0)\nwrites are compressed and stored in RAM\nreads are decompressed on the fly\nno disk I/O — everything stays in RAM (compressed)", "category": "zram_basics"},
    
    {"input": "lz4 vs zstd", "output": "lz4: faster decompression (~500 MB/s), lower ratio (2-3x), use for inference\nzstd: better compression (3-5x), slower (~200 MB/s), use for archives/logs\nrecommendation: lz4 for model storage, zstd for write-once data", "category": "zram_algorithms"},
    {"input": "which compression algorithm", "output": "for models/inference: lz4 (fast decompression matters)\nfor logs/archives: zstd (compression ratio matters)\nfor swap: lz4 (latency sensitive)\nrarely use lzo: fastest but lowest ratio", "category": "compression_compare"},
    
    {"input": "check zram compression ratio", "output": "rama zram list\nshows: orig_data_size, compr_data_size, compression_ratio\nexample output: 4096 MB → 1536 MB (2.67x)\nalso check per-volume: rama volume ls", "category": "zram_compression_ratio"},
    
    {"input": "how much overhead does zram have", "output": "compression metadata: ~1-2% of compressed size\nfilesystem overhead: ext4 ~5%, xfs ~3%\ntotal effective ratio: lz4 gives ~2.5x after overhead, zstd ~4x\ncheck actual: rama zram list shows real compression", "category": "compression_overhead"},
    
    {"input": "zram vs regular swap", "output": "zram: compressed in RAM, faster, no disk I/O, loses data on reboot\nregular swap: on disk, slower, persistent, unlimited size\nuse zram for: performance, temporary data\nuse disk swap for: persistence, more space than RAM", "category": "zram_vs_swap"},
    
    # ── rama-zpu Inference ───────────────────────────────────────────────
    {"input": "what model formats does rama-zpu support", "output": "GGUF: standard format from llama.cpp\nRZM: rama's optimized format (converted from GGUF)\nSafeTensors: PyTorch/HF format\nrecommended: use GGUF or convert to RZM for best performance", "category": "model_formats"},
    
    {"input": "convert gguf to rzm", "output": "rama-convert --input model.gguf --output model.rzm\nwhat it does: converts GGUF tensors to RZM format, preserves quantization\nsupported types: F32, F16, BF16, Q8_0, Q4_0, Q4_K, Q5_K, Q6_K\nunsupported types are dequantized to F32", "category": "model_convert"},
    
    {"input": "what is q8_0", "output": "Q8_0: 8-bit integer quantization (symmetric, zero-point)\nsize: 4x smaller than F32\nquality: minimal loss for <100M params\nrecommendation: use Q8_0 for small models like me (9M params)", "category": "quantization_types"},
    {"input": "which quantization for small models", "output": "9M params like me: Q8_0 is best (minimal quality loss, 4x smaller)\nQ4_K too aggressive for <100M params — noticeable quality degradation\nrecommendation: Q8_0 for <100M, Q4_K for >1B\ntest: compare perplexity before committing to Q4", "category": "quantization_types"},
    
    {"input": "what are memory tiers", "output": "RAM: regular memory (file-backed mmap, no compression)\nzram: compressed memory (kernel-level compression, ~2-3x)\narena: anonymous mmap (slab allocator, no filesystem)\nrama-zpu can use any tier — zram is recommended for models", "category": "memory_tiers"},
    
    {"input": "what is kv cache", "output": "KV cache: stores key/value tensors from previous tokens\nsize depends on: context length, model size, batch size\ntypical: 2-4 GB for 7B model with 4K context\nrama-zpu: KV cache goes to /zpu/kv (separate zram device)", "category": "kv_cache"},
    
    {"input": "model won't load", "output": "check: file exists and is readable\ncommon errors: unsupported quantization type, corrupted file, out of memory\ndebug: check rama-zpu logs, verify file format (GGUF header)\nif OOM: reduce model size or increase zram volume", "category": "model_troubleshoot"},
    
    # ── Compression ──────────────────────────────────────────────────────
    {"input": "how fast is lz4", "output": "lz4 decompression: ~500 MB/s on modern CPUs\ncompression: ~300 MB/s\nlatency: <1ms for typical block sizes\ngood for: inference, swap, anything latency-sensitive", "category": "lz4_details"},
    
    {"input": "zstd compression levels", "output": "zstd supports levels 1-22\ndefault (level 3): good balance, ~3-4x compression\nlevel 19+: max compression (~5x) but very slow\nrama uses level 3 by default (fast enough, good ratio)", "category": "zstd_details"},
    
    {"input": "measure compression ratio", "output": "rama zram list shows orig_data_size and compr_data_size\nratio = orig_data_size / compr_data_size\nexample: 4096 MB orig, 1536 MB compressed = 2.67x\nalso: rama volume ls shows per-volume ratios", "category": "compression_measure"},
    
    {"input": "best compression for models", "output": "lz4 for models — fast decompression is critical for inference\nmodel weights compress well (2-3x even with lz4)\nquantized models (Q8_0): already compact, less benefit from compression\ndon't use zstd for models — slower decompression hurts latency", "category": "compression_recommendations"},
    
    # ── Configuration ────────────────────────────────────────────────────
    {"input": "where is rama config", "output": "/etc/rama/*.conf — system-wide config\n~/.ramarc — user-specific settings (INI format)\nzram-generator.conf — boot-time zram setup (systemd)\ncheck current: rama config show", "category": "rama_config"},
    
    {"input": "what is ramarc", "output": ".ramarc is an INI file for user settings\nsections: [volumes], [workspaces], [defaults]\nexample:\n[defaults]\ncompression = lz4\nsize = 16G", "category": "ramarc"},
    
    {"input": "what is astra", "output": "astra is rama's systemd-zram-generator compatibility mode\nsame binary as rama (symlinked), different behavior based on argv[0]\nreads /etc/systemd/zram-generator.conf at boot\ncreates zram devices before systemd finishes booting", "category": "astra"},
    
    # ── Workspaces & Overlay ─────────────────────────────────────────────
    {"input": "create a workspace", "output": "sudo rama workspace create -s /path/to/project\nthis creates an overlayfs-backed workspace in ~/project-workspace\nupper layer: tmpfs (or zram if configured)\nlower layer: original source (read-only)\nwrites go to RAM, reads from source", "category": "workspace_create"},
    
    {"input": "how does overlayfs work", "output": "overlayfs merges multiple directories:\nlower: original files (read-only)\nupper: changes (new/modified files)\nwork: temporary space for atomic operations\nmerged view: combines lower + upper transparently", "category": "overlayfs_basics"},
    
    # ── Troubleshooting ──────────────────────────────────────────────────
    {"input": "out of space in zram", "output": "check: rama volume ls (shows usage)\nfind largest: du -sh /zpu/* | sort -h\noptions: increase size (rama volume resize), delete unused files, add spillover\nif critical: create new larger volume, copy data, swap", "category": "troubleshoot_space"},
    
    {"input": "low compression ratio", "output": "check: rama zram list (shows actual ratio)\ncauses: already compressed data (GGUF Q4_K), random data, small blocks\nexpected: text/code 3-5x, weights 2-3x, quantized models 1.5-2x\nif ratio < 1.5x: data may not benefit from compression", "category": "troubleshoot_compression"},
    
    {"input": "slow inference", "output": "check: compression algorithm (lz4 is fastest)\nmeasure: time to first token, tokens per second\noptimize: use Q8_0 instead of F32, ensure using zram (not regular disk)\nbenchmark: compare zram vs RAM vs disk for your workload", "category": "troubleshoot_perf"},
    
    # ── Use Cases ────────────────────────────────────────────────────────
    {"input": "store llm models on zram", "output": "create volume: rama volume create models -p gguf (16G, lz4, /models)\ncopy models: cp model.gguf /models/ or rama volume copy models model.gguf\nload in rama-zpu: model = load_gguf(\"/models/model.gguf\")\nbenefits: 2-3x compression, faster than disk, no SSD wear", "category": "usecase_llm_storage"},
    
    {"input": "build environment with rama", "output": "sudo rama workspace create -s /path/to/project\ncd ~/project-workspace\ncargo build --release (all writes go to RAM)\nbuild artifacts: in RAM, very fast, discarded on cleanup\ncleanup: sudo rama workspace destroy project (instant)", "category": "usecase_build_env"},
    
    # ── Meta/Self ────────────────────────────────────────────────────────
    {"input": "what are you", "output": "I'm Tadpole, a 9M parameter assistant for rama-zpu\nI help with: rama commands, zram management, compression, troubleshooting\nI'm small, so I focus on common tasks — check docs for deep dives\nI run on rama-zpu in compressed memory (eating my own dogfood)", "category": "meta_what_is_tadpole"},
    
    {"input": "how were you built", "output": "architecture: 9M parameter transformer (6 layers, 384 hidden dim)\ntraining: rama documentation, command examples, troubleshooting Q&A\nquantization: Q8_0 (4x smaller than F32, minimal quality loss)\nruntime: rama-zpu inference on zram (compressed ~3 MB)", "category": "meta_tadpole_arch"},
    {"input": "what are your specs", "output": "9M parameters total\n6 layers, 384 hidden dim, 6 attention heads\nquantized to Q8_0 (8-bit integers)\nsize: ~9 MB uncompressed, ~3 MB in zram with lz4\ncontext: 128 tokens (that's why I keep answers short)", "category": "meta_tadpole_arch"},
]

def save_samples(filename="rama_samples.json"):
    """Save sample data to JSON file."""
    with open(filename, 'w') as f:
        json.dump(SAMPLE_DATA, f, indent=2)
    print(f"✓ Saved {len(SAMPLE_DATA)} samples to {filename}")
    
    # Print stats
    categories = {}
    for item in SAMPLE_DATA:
        cat = item['category']
        categories[cat] = categories.get(cat, 0) + 1
    
    print(f"\nCoverage: {len(categories)} unique topics")
    print(f"Total samples: {len(SAMPLE_DATA)}")
    print("\nSamples per topic:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat:30} {count:2} samples")

if __name__ == "__main__":
    save_samples()
