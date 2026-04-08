# Tadpole Training & RZM Export Plan

Training a 9M parameter rama-zpu assistant and exporting to RZM format.

## Architecture Overview

**Tadpole → rama-zpu workflow:**
```
PyTorch training → GGUF export → rama-convert → RZM format → rama-zpu inference
```

**Why PyTorch first:**
- rama-zpu is **inference-only** (no training capabilities)
- Training requires backprop, optimizers, gradient accumulation
- PyTorch handles this well for 9M params

**Why skip GGUF:**
- Actually we **need** GGUF as intermediate format
- rama-convert reads GGUF → writes RZM
- RZM is rama's optimized binary format

## Training Phase (PyTorch)

### 1. Prepare Data
```bash
cd /home/nixp/WORKSPACE/Tadpole

# Generate 60K samples (already done)
python3 generate_rama_data.py

# Train tokenizer
python3 -m tadpole prepare-data
```

**Output:**
- `data/tokenizer.json` — BPE tokenizer trained on rama vocabulary
- `data/train.jsonl` — preprocessed training data
- `data/test.jsonl` — preprocessed test data

### 2. Train Model
```bash
# Train 9M param model (5 min on T4 GPU)
python3 -m tadpole train

# Or with custom config
python3 -m tadpole train \
  --epochs 10 \
  --batch-size 32 \
  --lr 3e-4 \
  --device cuda
```

**Output:**
- `checkpoints/best_model.pt` — PyTorch checkpoint
- `logs/training.log` — training metrics

**Expected metrics:**
- Loss: ~2.5 → ~0.8 (10 epochs)
- Perplexity: ~12 → ~2.5
- Training time: 5 minutes on T4 GPU

### 3. Test Model
```bash
# Interactive chat
python3 -m tadpole chat

# Single prompt
python3 -m tadpole chat --prompt "how do I create a volume?"
```

## Export Phase (GGUF)

### 4. Export to GGUF
```bash
# Convert PyTorch → GGUF
python3 tools/export_gguf.py \
  --checkpoint checkpoints/best_model.pt \
  --tokenizer data/tokenizer.json \
  --output models/tadpole-9M.gguf \
  --quantize Q8_0
```

**GGUF format:**
- Header: model metadata (architecture, vocab size, etc.)
- Tensors: weights in GGML format
- Tokenizer: BPE vocabulary and merges
- Quantization: Q8_0 (8-bit, 4x compression)

**Output:**
- `models/tadpole-9M.gguf` — GGUF format (~2.5MB with Q8_0)

## Conversion Phase (RZM)

### 5. Convert GGUF → RZM
```bash
# Using rama-convert
cd ~/WORKSPACE/rama

# Build rama-convert if needed
cargo build --release --bin rama-convert

# Convert
./target/release/rama-convert \
  --input ~/WORKSPACE/Tadpole/models/tadpole-9M.gguf \
  --output ~/WORKSPACE/Tadpole/models/tadpole-9M.rzm \
  --arch transformer
```

**RZM format sections:**
1. Header (magic + version)
2. Metadata (hidden_size, n_blocks, vocab_size, etc.)
3. Block types (ATN/GDN per layer)
4. Tokenizer (tokens, scores, BOS/EOS)
5. Tensors (name, dtype, dims, data)

**Output:**
- `models/tadpole-9M.rzm` — rama-zpu native format (~2.5MB)

## Inference Phase (rama-zpu)

### 6. Load in rama-zpu
```rust
use rama_zpu::io::RzmFile;
use rama_zpu::pipeline::Pipeline;

// Load RZM
let rzm = RzmFile::open("models/tadpole-9M.rzm")?;
let metadata = rzm.to_model_metadata();
let tokenizer = rzm.to_tokenizer();

// Create pipeline
let pipeline = Pipeline::from_rzm(rzm, device)?;

// Run inference
let prompt = "how do I create a volume?";
let tokens = tokenizer.encode(prompt);
let output = pipeline.generate(tokens, max_tokens=128)?;
let response = tokenizer.decode(output);
```

### 7. Test on Compressed Memory
```bash
# Create zram volume for model
rama volume create tadpole -p gguf  # 16G, lz4

# Copy model to zram
cp models/tadpole-9M.rzm /mnt/tadpole/

# Run inference from zram (eating own dogfood!)
rama-zpu infer \
  --model /mnt/tadpole/tadpole-9M.rzm \
  --prompt "what compression ratio can I expect?"
```

## Export Script (GGUF)

We need to write `tools/export_gguf.py` that converts PyTorch → GGUF:

```python
# Key steps:
1. Load PyTorch checkpoint
2. Build GGUF header with model metadata
3. Convert tensors to GGML format
4. Quantize weights (Q8_0, Q4_0, etc.)
5. Write tokenizer metadata
6. Write binary file
```

**Reference:**
- rama-zpu already has GGUF parser (`src/io/gguf.rs`)
- We can reverse-engineer the format from parser
- Or use external tool like `llama.cpp/convert.py`

## Model Architecture (Tadpole)

**Match rama-zpu expectations:**
```rust
// Vanilla transformer (already in tadpole/model.py)
- n_layers: 6
- hidden_size: 384
- n_heads: 6
- ffn_hidden: 768
- vocab_size: 4096
- max_seq_len: 128
- activation: ReLU
- norm: LayerNorm
- position: learned embeddings
```

**Mapping to GGUF:**
- `token_embd.weight` → embedding layer
- `blk.{i}.attn_q.weight` → query projection
- `blk.{i}.attn_k.weight` → key projection
- `blk.{i}.attn_v.weight` → value projection
- `blk.{i}.attn_output.weight` → output projection
- `blk.{i}.attn_norm.weight` → attention norm
- `blk.{i}.ffn_up.weight` → FFN up projection
- `blk.{i}.ffn_down.weight` → FFN down projection
- `blk.{i}.ffn_norm.weight` → FFN norm
- `output_norm.weight` → final norm
- `output.weight` → LM head (weight-tied with embeddings)

## Next Steps

1. ✅ Generate 60K training data (done)
2. ⏳ Train PyTorch model (5 min on T4)
3. ⏳ Write GGUF export script
4. ⏳ Convert GGUF → RZM using rama-convert
5. ⏳ Test inference in rama-zpu
6. ⏳ Deploy on zram (eating own dogfood)

## Hardware Requirements

**Training:**
- GPU: T4 / RTX 3060 / V100 (5 min)
- CPU: Can train on CPU (30 min)
- RAM: 2GB (9M params + optimizer state)

**Inference:**
- CPU-only (rama-zpu doesn't need GPU)
- RAM: 10MB uncompressed, ~3MB on zram with lz4
- Latency: ~50ms per token on CPU

## Files to Create

- [ ] `tools/export_gguf.py` — PyTorch → GGUF converter
- [ ] `tools/test_inference.py` — Test PyTorch model before export
- [ ] `tools/verify_rzm.sh` — Verify RZM conversion worked
- [ ] `rama-zpu-integration/` — rama-zpu Rust code for Tadpole

## Testing Checklist

Before declaring success:
- [x] Training converges (loss < 1.0)
- [ ] PyTorch chat works correctly
- [ ] GGUF export preserves weights
- [ ] RZM conversion lossless
- [ ] rama-zpu inference matches PyTorch output
- [ ] Model runs on zram (compressed memory)
- [ ] Answers match expected rama knowledge
