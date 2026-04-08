"""
Rama-focused topic taxonomy — 60 categories for technical Q&A generation.

Replaces pond/metamorphosis themes with rama ecosystem knowledge.
"""

TOPICS = {
    # ── rama Volume Management (10) ──────────────────────────────────────
    "volume_create": "Creating volumes with presets, sizes, algorithms",
    "volume_lifecycle": "Starting, stopping, restarting volumes",
    "volume_list": "Listing volumes, checking usage and stats",
    "volume_resize": "Resizing existing volumes",
    "volume_delete": "Removing volumes safely",
    "volume_copy": "Copying data to/from volumes",
    "volume_presets": "Understanding presets (gguf, tmp, cache, swap, log)",
    "volume_mounts": "Mount points, paths, filesystem types",
    "volume_troubleshoot": "Volume won't mount, full disk, permission errors",
    "volume_best_practices": "When to use volumes, sizing guidance",
    
    # ── Zram Operations (8) ──────────────────────────────────────────────
    "zram_basics": "What zram is, how it works, use cases",
    "zram_algorithms": "lz4, zstd, lzo characteristics",
    "zram_compression_ratio": "Measuring and interpreting compression ratios",
    "zram_stats": "Reading zram statistics and metadata",
    "zram_vs_swap": "Zram vs regular swap, when to use each",
    "zram_multiple": "Managing multiple zram devices",
    "zram_performance": "Tuning compression, memory limits, swappiness",
    "zram_troubleshoot": "Zram errors, low compression, performance issues",
    
    # ── rama-zpu Inference (10) ──────────────────────────────────────────
    "model_formats": "GGUF, RZM, SafeTensors formats explained",
    "model_convert": "Converting GGUF to RZM, preserving quantization",
    "model_load": "Loading models in rama-zpu",
    "quantization_types": "Q8_0, Q4_K, Q5_K, Q6_K, F16, F32, BF16",
    "memory_tiers": "RAM, zram, arena-backed memory",
    "kv_cache": "KV cache management and sizing",
    "inference_perf": "Inference performance, throughput, latency",
    "model_troubleshoot": "Model won't load, tensor errors, OOM",
    "tensor_ops": "Basic tensor operations in rama-zpu",
    "pipeline": "Inference pipeline, batching, streaming",
    
    # ── Compression (6) ──────────────────────────────────────────────────
    "lz4_details": "lz4 characteristics, speed, use cases",
    "zstd_details": "zstd levels, compression ratio, use cases",
    "compression_compare": "When to use lz4 vs zstd vs lzo",
    "compression_measure": "Measuring compression ratio and overhead",
    "compression_overhead": "Filesystem and metadata overhead",
    "compression_recommendations": "Best algorithm for specific workloads",
    
    # ── Configuration (6) ────────────────────────────────────────────────
    "rama_config": "/etc/rama config files and options",
    "ramarc": ".ramarc format, INI sections, common settings",
    "zram_generator_conf": "zram-generator.conf format and integration",
    "systemd_integration": "Systemd units, astra, boot-time setup",
    "astra": "Astra (systemd-zram-generator compatibility)",
    "config_troubleshoot": "Config parse errors, invalid options",
    
    # ── Workspaces & Overlay (5) ─────────────────────────────────────────
    "workspace_create": "Creating workspaces with overlayfs",
    "overlayfs_basics": "How overlayfs works (upper/lower/work)",
    "workspace_layers": "Understanding workspace layers and isolation",
    "workspace_snapshots": "Creating and restoring snapshots",
    "workspace_cleanup": "Cleaning up workspaces safely",
    
    # ── System Integration (5) ───────────────────────────────────────────
    "systemd_units": "Systemd units, scopes, running commands in cgroups",
    "cgroups": "Cgroup resource limits (memory, CPU, IO)",
    "memory_mgmt": "Page cache, swap behavior, madvise hints",
    "loop_devices": "Loop device management",
    "hugepages": "Hugepage configuration for inference",
    
    # ── Troubleshooting (5) ──────────────────────────────────────────────
    "troubleshoot_mount": "Volume/filesystem won't mount",
    "troubleshoot_space": "Out of space errors, finding large files",
    "troubleshoot_compression": "Compression failures, low ratios",
    "troubleshoot_perf": "Performance issues, slow inference",
    "troubleshoot_debug": "Debugging commands, logs, verbose output",
    
    # ── Use Cases (3) ────────────────────────────────────────────────────
    "usecase_llm_storage": "Storing LLM models on zram",
    "usecase_build_env": "Build environments with overlayfs",
    "usecase_dev_workspace": "Development workspaces, isolation",
    
    # ── Meta/Self (2) ────────────────────────────────────────────────────
    "meta_what_is_tadpole": "What Tadpole is, its purpose, limitations",
    "meta_tadpole_arch": "Tadpole's own architecture (9M, Q8_0, rama-zpu)",
}

def get_all_topics():
    """Return list of all 60 topic keys."""
    return list(TOPICS.keys())

def get_topic_description(topic):
    """Get human-readable description of a topic."""
    return TOPICS.get(topic, "Unknown topic")

def validate_topics():
    """Ensure we have exactly 60 topics."""
    assert len(TOPICS) == 60, f"Expected 60 topics, got {len(TOPICS)}"
    print(f"✓ Validated {len(TOPICS)} topics")
    
    # Print by category
    categories = [
        ("rama Volume Management", 10),
        ("Zram Operations", 8),
        ("rama-zpu Inference", 10),
        ("Compression", 6),
        ("Configuration", 6),
        ("Workspaces & Overlay", 5),
        ("System Integration", 5),
        ("Troubleshooting", 5),
        ("Use Cases", 3),
        ("Meta/Self", 2),
    ]
    
    topic_list = get_all_topics()
    idx = 0
    for category, count in categories:
        print(f"\n{category} ({count} topics):")
        for i in range(count):
            topic = topic_list[idx]
            print(f"  {idx+1:2}. {topic:25} — {TOPICS[topic]}")
            idx += 1

if __name__ == "__main__":
    validate_topics()
