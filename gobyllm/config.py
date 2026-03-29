"""GobyLLM configuration."""

from dataclasses import dataclass


@dataclass
class GobyConfig:
    # ── Architecture ────────────────────────────────────────────────────
    vocab_size: int = 8192
    max_seq_len: int = 512
    d_model: int = 512
    n_layers: int = 10
    n_heads: int = 8          # query heads
    n_kv_heads: int = 2       # GQA key-value heads
    ffn_hidden: int = 960     # SwiGLU intermediate dim
    dropout: float = 0.1
    rope_base: float = 10000.0
    norm_eps: float = 1e-5
    parallel_residual: bool = True  # PaLM-style parallel attn + FFN

    # ── Early exit ──────────────────────────────────────────────────────
    early_exit: bool = True
    min_exit_layer: int = 2        # don't exit before this layer
    exit_threshold: float = 0.8    # router confidence to exit during inference
    exit_loss_weight: float = 0.1  # weight for auxiliary exit losses
    router_loss_weight: float = 0.02  # weight for router BCE loss

    # ── Special tokens ──────────────────────────────────────────────────
    pad_id: int = 0
    bos_id: int = 1           # <|im_start|>
    eos_id: int = 2           # <|im_end|>


@dataclass
class TrainConfig:
    batch_size: int = 32
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    warmup_steps: int = 300
    max_steps: int = 10000
    eval_interval: int = 400
    save_interval: int = 1000
    grad_clip: float = 1.0
    device: str = "auto"
    seed: int = 42
    data_dir: str = "data"
    output_dir: str = "checkpoints"
