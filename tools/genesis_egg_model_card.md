---
license: mit
language:
  - en
tags:
  - tadpole
  - genesis-egg
  - untrained
  - base-model
  - guppy-ancestry
pipeline_tag: text-generation
base_model: arman-bd/guppylm
---

<p align="center">
  <img src="assets/tadpole.png" alt="Genesis Egg" width="300"/>
</p>

# Genesis Egg — Tadpole Base Model

The Genesis Egg is the untrained Tadpole base torch model.

It keeps Tadpole's architecture, but intentionally ships before any training so it can serve as the starting point for new runs, experiments, or domain adaptation.

Attribution follows `FORK.md`: Tadpole is the maintained fork of GuppyLM, with original work by arman-bd and fork maintenance by nixpt.

## Lineage

`GuppyLM → Tadpole → Genesis Egg`

## What this is

- Randomly initialized Tadpole weights
- No fine-tuning
- No dataset adaptation
- Same 6-layer 384-dim transformer shape as Tadpole

## What it is for

- Reproducible base checkpoints
- Lineage-aware model bootstrapping
- A clean starting point before training

## Export

Use:

```bash
python tools/export_genesis_egg.py
```

This writes a checkpoint, HF-style config, lineage metadata, a standalone `genesis_egg.dna` SDNA manifest, and the optional tokenizer bundle.
