# Fork Notice

This is a fork of [arman-bd/guppylm](https://github.com/arman-bd/guppylm), redesigned as **Tadpole** and optimized for [rama-zpu](https://github.com/nixpt/WORKSPACE) inference on compressed memory (zram).

## What Changed

### 1. Personality Redesign
- **Character:** Guppy (fish) → Tadpole (amphibian)
- **Environment:** Fish tank → Pond
- **Identity:** Static fish → Metamorphosing tadpole
- **Themes:** Food, water → Transformation, growth, confusion
- **Voice:** Curious about changes, doesn't know what it's becoming

### 2. Technical Stack
- **Model format:** ONNX → GGUF/RZM
- **Inference engine:** PyTorch/ONNX Runtime → rama-zpu
- **Memory backend:** Regular RAM → zram compression
- **Quantization:** Browser-optimized uint8 → Q8_0/Q4_K for compressed memory
- **Deployment:** Browser WASM → Linux CLI

### 3. Project Goals
- **Upstream:** Educational, browser-friendly, broadly compatible
- **Tadpole:** rama-zpu integration, compression optimization, Linux deployment

## Why Fork?

The original GuppyLM is excellent for its purpose (browser-based educational LLM), but our goals diverged:

1. **Runtime incompatibility** — rama-zpu vs ONNX Runtime
2. **Format incompatibility** — GGUF/RZM vs ONNX
3. **Personality divergence** — Tadpole's metamorphosis theme better fits a "growing/learning" model
4. **Memory architecture** — Optimizing for zram compression changes model design

## Upstream Compatibility

**We do NOT maintain compatibility with upstream GuppyLM.**

This is a one-way fork with incompatible goals. Upstream improvements (training techniques, dataset generation methods) can be manually ported if valuable, but we will not track upstream changes automatically.

## Attribution

Original work by [arman-bd](https://github.com/arman-bd/guppylm).  
Fork maintained by [nixpt](https://github.com/nixpt).  
Both projects are MIT licensed.

## License

MIT License (same as upstream)

Copyright (c) 2024 arman-bd (original GuppyLM)  
Copyright (c) 2026 nixpt (Tadpole fork)

See [LICENSE](LICENSE) for full text.
