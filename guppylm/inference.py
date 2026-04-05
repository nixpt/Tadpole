"""GuppyLM inference — simple chat."""

import json
import time
import uuid

import torch
from tokenizers import Tokenizer

from .config import GuppyConfig
from .model import GuppyLM


class GuppyInference:
    def __init__(self, checkpoint_path, tokenizer_path, device="cpu"):
        self.device = torch.device(device)
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        import os
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Detect format: legacy checkpoint has "model_state_dict" key,
        # HF standard is just the state_dict (first key is a weight like "tok_emb.weight")
        is_legacy = isinstance(ckpt, dict) and "model_state_dict" in ckpt

        if is_legacy:
            valid_fields = {f.name for f in GuppyConfig.__dataclass_fields__.values()}
            self.config = GuppyConfig(**{k: v for k, v in ckpt["config"].items() if k in valid_fields})
            state_dict = ckpt["model_state_dict"]
        else:
            # HF standard: load config from config.json next to the model file
            config_dir = os.path.dirname(os.path.abspath(checkpoint_path))
            config_path = os.path.join(config_dir, "config.json")
            with open(config_path) as f:
                cfg = json.load(f)
            self.config = GuppyConfig(
                vocab_size=cfg["vocab_size"],
                max_seq_len=cfg.get("max_position_embeddings", 128),
                d_model=cfg["hidden_size"],
                n_layers=cfg["num_hidden_layers"],
                n_heads=cfg["num_attention_heads"],
                ffn_hidden=cfg["intermediate_size"],
                dropout=cfg.get("hidden_dropout_prob", 0.1),
                pad_id=cfg.get("pad_token_id", 0),
                bos_id=cfg.get("bos_token_id", 1),
                eos_id=cfg.get("eos_token_id", 2),
            )
            state_dict = ckpt

        self.model = GuppyLM(self.config).to(self.device)
        filtered = {k: v for k, v in state_dict.items() if k in self.model.state_dict()}
        self.model.load_state_dict(filtered)
        self.model.eval()

        total, _ = self.model.param_count()
        print(f"GuppyLM loaded: {total/1e6:.1f}M params")

    def chat_completion(self, messages, temperature=0.7, max_tokens=64,
                        top_k=50, **kwargs):
        """Chat completion — takes messages, returns response."""
        prompt = self._format_prompt(messages)
        input_ids = self.tokenizer.encode(prompt).ids
        prompt_tokens = len(input_ids)
        input_t = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        output_t, _ = self.model.generate(input_t, max_tokens, temperature, top_k)
        output_text = self.tokenizer.decode(output_t[0].tolist()[prompt_tokens:])
        # Truncate at first <|im_end|> — don't let the model leak into the next turn
        if "<|im_end|>" in output_text:
            output_text = output_text.split("<|im_end|>")[0]
        # Also strip any <|im_start|> fragments
        if "<|im_start|>" in output_text:
            output_text = output_text.split("<|im_start|>")[0]
        resp_text = output_text.strip()

        return {
            "choices": [{
                "message": {"role": "assistant", "content": resp_text},
            }],
        }

    def _format_prompt(self, messages):
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content") or ""
            if role == "system":
                continue
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)


def main():
    import argparse
    p = argparse.ArgumentParser(description="Chat with Guppy")
    p.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    p.add_argument("--tokenizer", default="data/tokenizer.json")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    engine = GuppyInference(args.checkpoint, args.tokenizer, args.device)
    print("\nGuppy Chat (type 'quit' to exit)")
    msgs = []
    while True:
        inp = input("\nYou> ").strip()
        if inp.lower() in ("quit", "exit", "q"):
            break
        msgs.append({"role": "user", "content": inp})
        result = engine.chat_completion(msgs)
        msg = result["choices"][0]["message"]
        if msg.get("content"):
            print(f"Guppy> {msg['content']}")
        msgs.append(msg)


if __name__ == "__main__":
    main()
