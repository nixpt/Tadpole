"""GuppyLM inference — simple chat."""

import json
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass

import torch
from tokenizers import Tokenizer

from .chat_assets import resolve_chat_assets
from .config import GuppyConfig, TadpoleConfig
from .model import GuppyLM, Tadpole


@dataclass(frozen=True)
class SpeciesCellCluster:
    role: str
    count: int
    primary_skill: str
    notes: tuple[str, ...] = ()


class GuppyInference:
    def __init__(self, checkpoint_path, tokenizer_path, device="cpu"):
        self.device = torch.device(device)
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        import os
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Load config.json from same directory as the model file
        config_dir = os.path.dirname(os.path.abspath(checkpoint_path))
        config_path = os.path.join(config_dir, "config.json")

        # Extract state_dict — handle both legacy and standard formats
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt

        # Load config — try config.json first, fall back to embedded config
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            # Support both HF standard keys and our own keys
            self.config = TadpoleConfig(
                vocab_size=cfg.get("vocab_size", 4096),
                max_seq_len=cfg.get("max_position_embeddings", cfg.get("max_seq_len", 128)),
                d_model=cfg.get("hidden_size", cfg.get("d_model", 384)),
                n_layers=cfg.get("num_hidden_layers", cfg.get("n_layers", 6)),
                n_heads=cfg.get("num_attention_heads", cfg.get("n_heads", 6)),
                ffn_hidden=cfg.get("intermediate_size", cfg.get("ffn_hidden", 768)),
                dropout=cfg.get("hidden_dropout_prob", cfg.get("dropout", 0.1)),
                pad_id=cfg.get("pad_token_id", cfg.get("pad_id", 0)),
                bos_id=cfg.get("bos_token_id", cfg.get("bos_id", 1)),
                eos_id=cfg.get("eos_token_id", cfg.get("eos_id", 2)),
            )
        elif isinstance(ckpt, dict) and "config" in ckpt:
            valid_fields = {f.name for f in TadpoleConfig.__dataclass_fields__.values()}
            self.config = TadpoleConfig(**{k: v for k, v in ckpt["config"].items() if k in valid_fields})
        else:
            print("Warning: No config found, using defaults")
            self.config = TadpoleConfig()

        self.model = Tadpole(self.config).to(self.device)
        filtered = {k: v for k, v in state_dict.items() if k in self.model.state_dict()}
        self.model.load_state_dict(filtered)
        self.model.eval()

        total, _ = self.model.param_count()
        print(f"Tadpole loaded: {total/1e6:.1f}M params")

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

    def chat_completion_for_species(self, species, messages, temperature=0.7, max_tokens=64, top_k=50):
        clusters = species_cell_clusters(species)
        if len(clusters) <= 1:
            prompt = species_chat_prompt(species)
            return self._chat_with_system_prompt(prompt, messages, temperature, max_tokens, top_k)

        role_outputs = []
        for cluster in clusters:
            role_prompt = species_role_prompt(species, cluster)
            result = self._chat_with_system_prompt(role_prompt, messages, temperature, max_tokens, top_k)
            role_outputs.append((cluster.role, result["choices"][0]["message"]["content"]))

        synthesis_prompt = species_synthesis_prompt(species, clusters, role_outputs)
        return self._chat_with_system_prompt(synthesis_prompt, messages, temperature, max_tokens, top_k)

    def _chat_with_system_prompt(self, system_prompt, messages, temperature=0.7, max_tokens=64, top_k=50):
        species_messages = [{"role": "system", "content": system_prompt}] + list(messages)
        return self.chat_completion(species_messages, temperature=temperature, max_tokens=max_tokens, top_k=top_k)

    def _format_prompt(self, messages):
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content") or ""
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)


def species_chat_prompt(species) -> str:
    name = getattr(species, "name", species.__class__.__name__)
    clusters = species_cell_clusters(species)
    cell_count = species_cell_count(species, clusters)

    lines = [
        f"You are the Symbiome species {name}.",
        f"Cell count: {cell_count}.",
    ]

    if clusters:
        role_lines = [f"{cluster.role}={cluster.count}" for cluster in clusters]
        lines.append("Roles: " + ", ".join(role_lines))

    legacy_roles = species_legacy_role_summary(species)
    if legacy_roles:
        lines.append("Legacy roles: " + ", ".join(legacy_roles))

    sources = species_dataset_sources(species)
    if sources:
        lines.append("Training sources: " + ", ".join(sources))

    lines.append("Neuron rule: every non-CNS cell keeps only its task and report_to_cns.")

    lines.append("Answer in character as this species. Be concise, grounded in its anatomy, and describe internal state when useful.")
    return "\n".join(lines)


def species_cell_clusters(species) -> list[SpeciesCellCluster]:
    cells = list(getattr(species, "cells", []))
    if not cells:
        legacy_clusters = []
        if hasattr(species, "brain"):
            legacy_clusters.append(SpeciesCellCluster(role="brain", count=1, primary_skill="brain"))
        if hasattr(species, "reader"):
            legacy_clusters.append(SpeciesCellCluster(role="reader", count=1, primary_skill="read_text"))
        if hasattr(species, "writer"):
            legacy_clusters.append(SpeciesCellCluster(role="writer", count=1, primary_skill="write_text"))
        return legacy_clusters

    grouped: "OrderedDict[str, list[object]]" = OrderedDict()
    for cell in cells:
        genome = getattr(cell, "genome", None)
        role = getattr(genome, "primary_skill", None) or getattr(cell, "name", "cell")
        grouped.setdefault(role, []).append(cell)

    clusters = []
    for role, role_cells in grouped.items():
        first = role_cells[0]
        genome = getattr(first, "genome", None)
        notes = tuple(getattr(genome, "notes", []) or ())
        clusters.append(
            SpeciesCellCluster(
                role=role,
                count=len(role_cells),
                primary_skill=getattr(genome, "primary_skill", role),
                notes=notes,
            )
        )
    return clusters


def species_cell_count(species, clusters=None) -> int:
    if clusters is None:
        clusters = species_cell_clusters(species)
    if getattr(species, "cells", None):
        return len(list(getattr(species, "cells", [])))
    return sum(cluster.count for cluster in clusters)


def species_dataset_sources(species) -> list[str]:
    sources = []
    seen = set()
    for cell in list(getattr(species, "cells", [])):
        genome = getattr(cell, "genome", None)
        for note in getattr(genome, "notes", []) or []:
            if note.startswith("dataset="):
                source = note.split("=", 1)[1]
                if source not in seen:
                    seen.add(source)
                    sources.append(source)
    return sources


def species_legacy_role_summary(species) -> list[str]:
    summary = []
    if hasattr(species, "brain"):
        summary.append(f"brain={getattr(getattr(species.brain, 'genome', None), 'primary_skill', 'unknown')}")
    for label, singular_attr, plural_attr in (
        ("reader", "reader", "readers"),
        ("writer", "writer", "writers"),
        ("memory", "memory_cells", "memory_cells"),
        ("guard", "guard_cells", "guard_cells"),
        ("motor", "motor_cells", "motor_cells"),
    ):
        if hasattr(species, plural_attr):
            summary.append(f"{plural_attr}={len(getattr(species, plural_attr))}")
        elif hasattr(species, singular_attr):
            if hasattr(species, plural_attr):
                summary.append(f"{plural_attr}={len(getattr(species, plural_attr))}")
            else:
                cell = getattr(species, singular_attr)
                genome = getattr(cell, "genome", None)
                summary.append(f"{label}={getattr(genome, 'primary_skill', label)}")
    return summary


def species_role_prompt(species, cluster: SpeciesCellCluster) -> str:
    name = getattr(species, "name", species.__class__.__name__)
    lines = [
        f"You are the {cluster.role} cell cluster of {name}.",
        f"Primary skill: {cluster.primary_skill}.",
        f"Cell count: {cluster.count}.",
        "Respond only from this cluster's perspective.",
    ]
    if cluster.notes:
        lines.append("Cell notes: " + ", ".join(cluster.notes[:4]))
    return "\n".join(lines)


def species_synthesis_prompt(species, clusters, role_outputs) -> str:
    name = getattr(species, "name", species.__class__.__name__)
    lines = [
        f"You are the control cell of {name}.",
        "Merge the specialist cluster notes below into one answer.",
        "Prefer the most grounded answer and keep it concise.",
        "",
    ]
    for role, text in role_outputs:
        lines.append(f"[{role}] {text}")
    if clusters:
        lines.append("")
        lines.append("Cluster summary: " + ", ".join(f"{cluster.role}={cluster.count}" for cluster in clusters))
    return "\n".join(lines)


def main():
    import argparse
    p = argparse.ArgumentParser(description="Chat with Tadpole")
    p.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    p.add_argument("--tokenizer", default="data/tokenizer.json")
    p.add_argument("--device", default="cpu")
    p.add_argument("--prompt", "-p", help="Single prompt mode: ask one question and exit")
    args = p.parse_args()

    checkpoint_path, tokenizer_path = resolve_chat_assets(args.checkpoint, args.tokenizer)
    engine = GuppyInference(checkpoint_path, tokenizer_path, args.device)

    if args.prompt:
        result = engine.chat_completion([{"role": "user", "content": args.prompt}])
        print(result["choices"][0]["message"]["content"])
        return

    print("\nTadpole Chat (type 'quit' to exit)")
    while True:
        inp = input("\nYou> ").strip()
        if inp.lower() in ("quit", "exit", "q"):
            break
        result = engine.chat_completion([{"role": "user", "content": inp}])
        msg = result["choices"][0]["message"]
        if msg.get("content"):
            print(f"Tadpole> {msg['content']}")


if __name__ == "__main__":
    main()
