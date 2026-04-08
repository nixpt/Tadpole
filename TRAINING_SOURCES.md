# Training Data Sources — Tadpole

Extracted from `~/WORKSPACE/{rama,exosphere,squadron}` on 2026-04-08.

## Summary

| Category | Files | Lines | Chars | Priority |
|----------|-------|-------|-------|----------|
| **rama_core** | 3 | 470 | 13,829 | 1 (highest) |
| **rama_docs** | 3 | 768 | 22,536 | 2 |
| **rama_planning** | 2 | 1,248 | 34,856 | 3 |
| **exosphere_core** | 1 | 62 | 3,133 | 2 |
| **squadron_tools** | 2 | 617 | 22,947 | 3 |
| **rama_commands** | 16 cmds | 159 | 3,159 | 1 |
| **TOTAL** | **11 files + 16 cmds** | **3,324** | **100,460** | — |

## Priority 1: rama Core Documentation

These form the foundation of Tadpole's knowledge base.

### rama/README.md (178 lines)
- Quick start guide
- Installation instructions
- Core commands (zram, zramfs, workspace, dev)
- Fast build workflow
- Volume system with presets
- Compression algorithms
- Backup/restore workflows

### rama/CLAUDE.md (135 lines)
- Git branch policy (never push to main/dev)
- Build & development commands
- Architecture (argv[0] dispatch: astra/ravan/rama)
- Module layout (60+ CLI commands)
- Build-safe concurrency limiting
- Critical constraints (circular deps, systemd boot crash)

### rama/QUICKREF.md (157 lines)
- Command cheat sheet
- Volume presets: gguf (16G), tmp (4G), code (8G), swap (32G), log (2G)
- Compression ratios by algorithm
- Troubleshooting guide
- Quick workflows for common tasks

### rama Command Help (16 commands, 159 lines)
Extracted from `rama-core --help` for:
- `volume-create`, `volume-list`, `volume-backup`, `volume-restore`
- `create`, `reset`, `list`, `stats`, `algorithms`
- `workspace-create`, `workspace-destroy`, `workspace-list`, `workspace-info`
- `dev-create`, `dev-destroy`, `dev-list`

## Priority 2: rama Implementation Details

### rama/doc/rama.md (252 lines)
- Detailed architecture diagrams
- System integration (zram, overlayfs, systemd)
- Memory management strategies

### rama/doc/CODEBASE_MAP.md (330 lines)
- Full codebase structure
- Module responsibilities
- Function signatures and dispatch flow

### rama/doc/IMPLEMENTATION.md (186 lines)
- Implementation notes
- Design decisions
- Performance considerations

### exosphere/CLAUDE.md (62 lines)
- Exosphere overview
- Integration with rama-zpu
- Build commands and structure

## Priority 3: Advanced Topics

### rama/zpu_vcpu_ram_gpu_lanes_spec.md (997 lines)
- ZPU (compression pool) specification
- VCPU, RAM, GPU lanes architecture
- Inference pipelines on compressed memory

### rama/planning/ZPU_IMPLEMENTATION_PLAN.md (251 lines)
- ZPU implementation roadmap
- Phase-by-phase plan
- Technical milestones

### squadron/README.md (88 lines)
- Multi-agent coordination tools
- Squadron bin/ utilities
- Agent communication protocols

### squadron/doc/TOOLS.md (529 lines)
- Detailed tool documentation
- agent-msg, agent-read, agent-mem
- Bridge, DMs, squad channels

## Topics Coverage

Based on `rama_topics.py` (60 topics), source coverage:

| Topic Category | Topics | Covered by Sources |
|----------------|--------|-------------------|
| **rama volumes** | 10 | ✅ README, QUICKREF, commands |
| **zram** | 8 | ✅ README, CLAUDE, commands |
| **inference** | 10 | ⚠️ ZPU spec (advanced) |
| **compression** | 6 | ✅ README, QUICKREF |
| **config** | 6 | ✅ CLAUDE, QUICKREF |
| **workspaces** | 5 | ✅ README, commands |
| **system** | 5 | ⚠️ IMPLEMENTATION.md |
| **troubleshooting** | 5 | ✅ QUICKREF |
| **use cases** | 3 | ✅ README workflow |
| **meta** | 2 | ⚠️ (self-awareness, add manually) |

Legend:
- ✅ = Well covered by existing docs
- ⚠️ = Partial coverage or too advanced for 9M model
- ❌ = Not covered (need to create)

## Extraction Strategy

1. **Priority 1 docs** → Generate 70% of training data (42K samples)
   - rama README, CLAUDE, QUICKREF
   - rama command help
   - Focus: practical commands, workflows, troubleshooting

2. **Priority 2 docs** → Generate 20% of training data (12K samples)
   - rama implementation docs
   - exosphere integration
   - Focus: architecture, design decisions

3. **Priority 3 docs** → Generate 10% of training data (6K samples)
   - ZPU spec (simplified)
   - Squadron tools
   - Focus: advanced topics, keep answers high-level

## Data Generation Plan

### Phase 1: Command-based Q&A (42K samples)
Extract from:
- `rama/README.md` → "how do I create a volume?" style questions
- `rama/QUICKREF.md` → preset questions, troubleshooting
- `rama_commands.json` → command syntax, options, examples

Variations:
- Direct: "how do I create a zram device?"
- Contextual: "I need 16G of compressed RAM"
- Troubleshooting: "volume is read-only, what do I check?"
- Comparison: "lz4 vs zstd for GGUF models?"

### Phase 2: Architecture Q&A (12K samples)
Extract from:
- `rama/CLAUDE.md` → build rules, constraints, architecture
- `rama/doc/*.md` → implementation details, design

Focus:
- "How does rama dispatch commands?" (argv[0])
- "What's the difference between astra and ravan?"
- "Why use overlayfs for workspaces?"

### Phase 3: Advanced Topics (6K samples)
Extract from:
- `rama/zpu_vcpu_ram_gpu_lanes_spec.md` → ZPU basics (simplified)
- `squadron/doc/TOOLS.md` → agent tools

Keep answers high-level:
- "What is ZPU?" → "compression pool for inference, check docs for details"
- "How does rama-zpu load models?" → mmap overview, defer to docs

## Next Steps

1. ✅ Extract sources (this document)
2. ⏳ Write data generator (template-based synthesis)
3. ⏳ Generate 60K training samples
4. ⏳ Train tokenizer on rama vocabulary
5. ⏳ Train model
6. ⏳ Export to GGUF/RZM

## Files Created

- `training_sources.json` — Catalog of 11 documentation files
- `rama_commands.json` — Help text for 16 key commands
- `TRAINING_SOURCES.md` — This document
- `extract_sources.py` — Documentation extraction script
- `extract_rama_commands.py` — Command help extraction script
