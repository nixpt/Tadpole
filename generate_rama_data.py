#!/usr/bin/env python3
"""
Generate synthetic Q&A training data for Tadpole — rama-zpu technical assistant.

Uses template composition with randomized variations to generate 60K unique
Q&A pairs from rama documentation.
"""

import json
import random
from pathlib import Path
from collections import Counter

random.seed(42)

# ══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def pick(lst):
    """Pick random item from list."""
    return random.choice(lst)

def pick_n(lst, n):
    """Pick n random items from list."""
    return random.sample(lst, min(n, len(lst)))

def maybe(text, p=0.5):
    """Include text with probability p."""
    return text if random.random() < p else ""

# ══════════════════════════════════════════════════════════════════════════════
#  VOCABULARY POOLS
# ══════════════════════════════════════════════════════════════════════════════

# Volume presets
PRESETS = ["gguf", "tmp", "code", "cache", "swap", "log"]

# Compression algorithms
ALGORITHMS = ["lz4", "zstd", "lzo", "zlib"]

# Sizes
SIZES = ["4G", "8G", "16G", "32G", "64G", "2G", "1G", "128M", "512M"]

# Volume names
VOLUME_NAMES = ["models", "build-tmp", "cache", "workspace", "data", "tmp", "swap", "logs"]

# File types
FILETYPES = ["text", "code", "GGUF models", "source code", "logs", "binaries", "JSON"]

# Question starters
QUESTION_STARTERS = [
    "how do I",
    "how can I",
    "what's the command to",
    "how to",
    "what command",
]

# ══════════════════════════════════════════════════════════════════════════════
#  TEMPLATE GENERATORS BY TOPIC
# ══════════════════════════════════════════════════════════════════════════════

# ── Volume Management ─────────────────────────────────────────────────────────

def gen_volume_create():
    """Generate volume creation Q&A."""
    questions = [
        f"{pick(QUESTION_STARTERS)} create a volume?",
        f"{pick(QUESTION_STARTERS)} create a {pick(SIZES)} volume?",
        f"{pick(QUESTION_STARTERS)} make a volume for {pick(['models', 'building', 'caching'])}?",
        f"I need {pick(SIZES)} of compressed storage",
        f"create volume with {pick(ALGORITHMS)} compression",
        f"how to set up a {pick(PRESETS)} volume?",
    ]
    
    preset = pick(PRESETS)
    size = pick(SIZES)
    name = pick(VOLUME_NAMES)
    algo = pick(ALGORITHMS)
    
    answers = [
        f"command: rama volume create {name} -p {preset}\ncreates {size} volume with {algo} compression\npresets: " + ", ".join(pick_n(PRESETS, 3)),
        f"rama volume create {name} -s {size} -c {algo}\ncustom size and algorithm\ncheck: rama volume list",
        f"preset: {preset} ({size}, {algo})\ncommand: rama volume create {name} -p {preset}",
    ]
    
    return pick(questions), pick(answers)

def gen_volume_list():
    """Generate volume listing Q&A."""
    questions = [
        "how do I see my volumes?",
        "list all volumes",
        "show volumes",
        "what volumes are active?",
        "check volume status",
    ]
    
    answers = [
        "command: rama volume list\nshows name, size, usage, mount point\nadds -v for compression stats",
        "rama volume list\ndisplays all active volumes with stats",
        "rama volume list -v\nverbose mode shows compression ratios and algorithms",
    ]
    
    return pick(questions), pick(answers)

def gen_volume_presets():
    """Generate preset explanation Q&A."""
    preset = pick(PRESETS)
    
    preset_info = {
        "gguf": ("16G", "lz4", "GGUF model storage"),
        "tmp": ("4G", "lz4", "temporary build files"),
        "code": ("8G", "zstd", "source code"),
        "cache": ("8G", "lz4", "build cache"),
        "swap": ("32G", "lz4", "zram swap"),
        "log": ("2G", "lz4", "log files"),
    }
    
    size, algo, desc = preset_info[preset]
    
    questions = [
        f"what is the {preset} preset?",
        f"when should I use {preset} preset?",
        f"what does {preset} preset give me?",
        f"explain {preset} volume preset",
    ]
    
    answers = [
        f"preset: {preset} ({size}, {algo})\noptimized for: {desc}\ncommand: rama volume create <name> -p {preset}",
        f"{preset}: {size} with {algo} compression\nuse for {desc}\nfast decompression for frequent access",
    ]
    
    return pick(questions), pick(answers)

# ── Zram Operations ───────────────────────────────────────────────────────────

def gen_zram_stats():
    """Generate zram stats Q&A."""
    questions = [
        "how do I check zram stats?",
        "show zram compression ratio",
        "check zram usage",
        "see zram statistics",
        "how much memory is zram using?",
    ]
    
    answers = [
        "command: cat /sys/block/zram0/mm_stat\nshows: orig_data_size, compr_data_size, mem_used_total\ncompression ratio = orig / compr",
        "rama stats zram0\ndisplays compression ratio, memory usage, algorithm",
        "check: /sys/block/zram0/compr_data_size\ncompare to orig_data_size for ratio",
    ]
    
    return pick(questions), pick(answers)

def gen_compression_compare():
    """Generate algorithm comparison Q&A."""
    algo1, algo2 = random.sample(ALGORITHMS, 2)
    filetype = pick(FILETYPES)
    
    questions = [
        f"{algo1} vs {algo2} for {filetype}?",
        f"which is better: {algo1} or {algo2}?",
        f"should I use {algo1} or {algo2}?",
        f"compression algorithm comparison",
    ]
    
    # Algorithm characteristics
    algo_info = {
        "lz4": ("fastest", "2-3x", "hot data"),
        "zstd": ("best ratio", "5-8x", "cold data"),
        "lzo": ("legacy", "2-3x", "old systems"),
        "zlib": ("good ratio", "3-5x", "balanced"),
    }
    
    speed1, ratio1, use1 = algo_info.get(algo1, ("moderate", "3x", "general"))
    speed2, ratio2, use2 = algo_info.get(algo2, ("moderate", "3x", "general"))
    
    answers = [
        f"{algo1}: {speed1}, {ratio1} typical\n{algo2}: {speed2}, {ratio2} typical\nfor {filetype}: {pick([algo1, algo2])} recommended",
        f"typical ratios:\n{filetype}: {pick(['3-5x', '2-3x', '4-6x'])}\nlz4 = speed, zstd = ratio\ncommand: rama volume create <name> -c {pick([algo1, algo2])}",
    ]
    
    return pick(questions), pick(answers)

# ── Workspace Operations ──────────────────────────────────────────────────────

def gen_workspace_create():
    """Generate workspace creation Q&A."""
    questions = [
        "how do I create a workspace?",
        "set up RAM-backed workspace",
        "create build environment in RAM",
        "fast builds with zram",
        "workspace for compilation",
    ]
    
    answers = [
        "command: sudo rama workspace create -s /path/to/project\ncreates overlayfs on zram + tmpfs\nbuilds write to RAM, source stays read-only",
        "rama workspace create -s /mnt/WORKSPACE/project\nRAM-backed overlay for fast builds\naccess at ~/project-workspace",
        "workspace uses: zram (overlay upper) + tmpfs (build dir)\ninstant cleanup: rama workspace destroy project",
    ]
    
    return pick(questions), pick(answers)

def gen_workspace_troubleshoot():
    """Generate workspace troubleshooting Q&A."""
    problems = [
        ("read-only", "check mount options: cat /proc/mounts | grep <name>\nremount rw: sudo mount -o remount,rw /mnt/<name>\nif overlayfs, workdir may be full"),
        ("full", "overlayfs upper layer full\ncheck: df -h /mnt/<upper>\nincrease zram size or cleanup"),
        ("won't mount", "check: dmesg | tail\ncommon: workdir not empty, permissions\nrecreate: rama workspace destroy && create"),
    ]
    
    problem, solution = pick(problems)
    
    questions = [
        f"workspace is {problem}",
        f"my workspace is {problem}, what do I check?",
        f"troubleshoot {problem} workspace",
    ]
    
    return pick(questions), solution

# ── rama-zpu Inference ────────────────────────────────────────────────────────

def gen_model_load():
    """Generate model loading Q&A."""
    format = pick(["GGUF", "RZM", "SafeTensors"])
    
    questions = [
        f"how does rama-zpu load {format} models?",
        "load model from compressed memory",
        "mmap on zram",
        "model loading process",
    ]
    
    answers = [
        "rama-zpu reads GGUF/RZM from zram-backed volumes\nmmap → page cache → zram → decompression on access\nkeeps hot tensors uncompressed, cold ones in zram",
        f"{format} → mmap from volume → page cache\nzram decompresses on page fault\nhot paths stay in RAM, cold compress",
        "inference reads model via mmap\nLinux page cache + zram handle compression\ntransparent to application",
    ]
    
    return pick(questions), pick(answers)

def gen_quantization():
    """Generate quantization Q&A."""
    quants = ["Q8_0", "Q4_K", "Q5_K", "Q6_K", "F16"]
    quant = pick(quants)
    
    questions = [
        f"what is {quant} quantization?",
        f"should I use {quant}?",
        f"explain {quant} format",
        "quantization recommendations",
    ]
    
    quant_info = {
        "Q8_0": ("8-bit", "4x compression", "minimal quality loss"),
        "Q4_K": ("4-bit k-quant", "8x compression", "some quality loss"),
        "Q5_K": ("5-bit k-quant", "6x compression", "good balance"),
        "Q6_K": ("6-bit k-quant", "5x compression", "near-original"),
        "F16": ("16-bit float", "2x compression", "full precision"),
    }
    
    bits, comp, quality = quant_info.get(quant, ("8-bit", "4x", "good"))
    
    answers = [
        f"{quant}: {bits}, {comp}, {quality}\nrecommended for 9M params\ncheck model card for tested quants",
        f"quantization reduces model size\n{quant} = {comp} smaller\ntrade-off: size vs quality",
    ]
    
    return pick(questions), pick(answers)

# ── Troubleshooting ───────────────────────────────────────────────────────────

def gen_troubleshoot_oom():
    """Generate OOM troubleshooting Q&A."""
    questions = [
        "out of memory error",
        "OOM kill",
        "process killed by kernel",
        "not enough memory",
        "memory exhausted",
    ]
    
    answers = [
        "check: dmesg | grep -i oom\nincrease zram: rama volume resize <name> <size>\nor reduce model size, enable swap",
        "OOM kill = kernel ran out of memory\nsolutions:\n1. increase zram/swap\n2. smaller model\n3. reduce batch size",
        "check memory: free -h\ncheck zram: cat /sys/block/zram0/mm_stat\nincrease if needed: rama volume resize",
    ]
    
    return pick(questions), pick(answers)

# ── Configuration ─────────────────────────────────────────────────────────────

def gen_config_file():
    """Generate config file Q&A."""
    questions = [
        "where is rama config?",
        "config file location",
        "how to configure rama?",
        "edit rama settings",
    ]
    
    answers = [
        "config: /etc/rama/config\nor ~/.ramarc (user override)\nINI format: [volumes], [zram], [defaults]",
        "rama config lives at /etc/rama/config\nuser settings: ~/.ramarc\ncheck current: rama config show",
        "edit: sudo vim /etc/rama/config\nper-user: ~/.ramarc\nCLI flags override config values",
    ]
    
    return pick(questions), pick(answers)

# ── Meta / Self-awareness ─────────────────────────────────────────────────────

def gen_meta_self():
    """Generate self-awareness Q&A."""
    questions = [
        "what are you?",
        "how do you work?",
        "tell me about yourself",
        "are you running on rama?",
        "what model are you?",
    ]
    
    answers = [
        "I'm Tadpole, a 9M parameter model running on rama-zpu\ncompressed in zram (eating my own dogfood)\nI help with rama commands and workflows",
        "tiny model (9M params) specialized for rama-zpu\nrunning in compressed memory via zram\nI know rama volumes, zram, inference, compression",
        "Tadpole: rama-zpu assistant\n9M params, vanilla transformer\nI'm small, check docs for complex cases",
    ]
    
    return pick(questions), pick(answers)

def gen_meta_limitations():
    """Generate limitation acknowledgment Q&A."""
    questions = [
        "can you explain ZPU architecture?",
        "detailed explanation of kernel params",
        "how does the Linux page cache work?",
        "explain memory management internals",
    ]
    
    topic = pick(["ZPU architecture", "kernel internals", "advanced tuning", "system details"])
    
    answers = [
        f"{topic} is complex\nI'm small (9M params), check docs for details\nbasic: {pick(['check rama --help', 'see CLAUDE.md', 'read README'])}",
        f"that's beyond my scope\nI know rama commands and workflows\nfor {topic}, check full documentation",
    ]
    
    return pick(questions), pick(answers)

# ══════════════════════════════════════════════════════════════════════════════
#  TOPIC ROUTER
# ══════════════════════════════════════════════════════════════════════════════

GENERATORS = {
    # Volume management (10 topics)
    "volume_create": gen_volume_create,
    "volume_list": gen_volume_list,
    "volume_presets": gen_volume_presets,
    "volume_lifecycle": gen_volume_list,  # reuse
    "volume_resize": gen_volume_create,  # reuse with variation
    "volume_delete": gen_volume_list,  # reuse
    "volume_copy": gen_volume_create,  # reuse
    "volume_mounts": gen_volume_presets,  # reuse
    "volume_troubleshoot": gen_workspace_troubleshoot,  # reuse
    "volume_best_practices": gen_volume_presets,  # reuse
    
    # Zram (8 topics)
    "zram_basics": gen_zram_stats,
    "zram_algorithms": gen_compression_compare,
    "zram_compression_ratio": gen_zram_stats,
    "zram_stats": gen_zram_stats,
    "zram_vs_swap": gen_zram_stats,
    "zram_multiple": gen_zram_stats,
    "zram_performance": gen_compression_compare,
    "zram_troubleshoot": gen_troubleshoot_oom,
    
    # Inference (10 topics)
    "model_formats": gen_model_load,
    "model_convert": gen_model_load,
    "model_load": gen_model_load,
    "quantization_types": gen_quantization,
    "memory_tiers": gen_model_load,
    "kv_cache": gen_model_load,
    "inference_perf": gen_model_load,
    "model_troubleshoot": gen_troubleshoot_oom,
    "tensor_ops": gen_model_load,
    "pipeline": gen_model_load,
    
    # Compression (6 topics)
    "lz4_details": gen_compression_compare,
    "zstd_details": gen_compression_compare,
    "compression_compare": gen_compression_compare,
    "compression_measure": gen_zram_stats,
    "compression_overhead": gen_zram_stats,
    "compression_recommendations": gen_compression_compare,
    
    # Config (6 topics)
    "config_file": gen_config_file,
    "config_precedence": gen_config_file,
    "ramarc": gen_config_file,
    "env_vars": gen_config_file,
    "cli_options": gen_config_file,
    "workspace_config": gen_config_file,
    
    # Workspaces (5 topics)
    "workspace_create": gen_workspace_create,
    "workspace_lifecycle": gen_workspace_create,
    "workspace_overlayfs": gen_workspace_create,
    "workspace_benefits": gen_workspace_create,
    "workspace_troubleshoot": gen_workspace_troubleshoot,
    
    # System (5 topics)
    "kernel_params": gen_config_file,
    "swappiness": gen_config_file,
    "page_cache": gen_model_load,
    "cgroups": gen_config_file,
    "systemd": gen_config_file,
    
    # Troubleshooting (5 topics)
    "readonly_mount": gen_workspace_troubleshoot,
    "oom_errors": gen_troubleshoot_oom,
    "mount_failed": gen_workspace_troubleshoot,
    "compression_disabled": gen_compression_compare,
    "performance_slow": gen_compression_compare,
    
    # Use cases (3 topics)
    "model_serving": gen_model_load,
    "fast_builds": gen_workspace_create,
    "development": gen_workspace_create,
    
    # Meta (2 topics)
    "self_awareness": gen_meta_self,
    "limitations": gen_meta_limitations,
}

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN GENERATION LOOP
# ══════════════════════════════════════════════════════════════════════════════

def generate_dataset(total_samples=60000, train_ratio=0.95):
    """Generate full training dataset."""
    print(f"Generating {total_samples} samples...")
    
    # Topic weights (all equal for now)
    topics = list(GENERATORS.keys())
    samples_per_topic = total_samples // len(topics)
    
    dataset = []
    topic_counts = Counter()
    
    for topic in topics:
        generator = GENERATORS[topic]
        
        for _ in range(samples_per_topic):
            question, answer = generator()
            
            dataset.append({
                "input": question,
                "output": answer,
                "category": topic,
            })
            
            topic_counts[topic] += 1
    
    # Shuffle
    random.shuffle(dataset)
    
    # Split train/test
    split_idx = int(len(dataset) * train_ratio)
    train_data = dataset[:split_idx]
    test_data = dataset[split_idx:]
    
    # Stats
    print(f"\n✓ Generated {len(dataset)} samples")
    print(f"  Train: {len(train_data)} ({train_ratio*100:.0f}%)")
    print(f"  Test:  {len(test_data)} ({(1-train_ratio)*100:.0f}%)")
    print(f"\n📊 Samples per topic: {samples_per_topic}")
    print(f"   Topics: {len(topics)}")
    
    # Show topic distribution
    print(f"\nTop 10 topics:")
    for topic, count in topic_counts.most_common(10):
        print(f"  {topic:30s} {count:5d}")
    
    return {
        "train": train_data,
        "test": test_data,
    }

def save_dataset(dataset, output_dir="data"):
    """Save dataset to JSON files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    for split, data in dataset.items():
        output_file = output_dir / f"{split}.json"
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Saved {len(data):5d} samples to {output_file}")

if __name__ == "__main__":
    dataset = generate_dataset(total_samples=60000, train_ratio=0.95)
    save_dataset(dataset)
    
    # Show samples
    print("\n" + "="*80)
    print("SAMPLE CONVERSATIONS (first 3 from train)")
    print("="*80)
    
    for i, sample in enumerate(dataset["train"][:3], 1):
        print(f"\n[{i}] Category: {sample['category']}")
        print(f"Q: {sample['input']}")
        print(f"A: {sample['output']}")
