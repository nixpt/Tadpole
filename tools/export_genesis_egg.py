"""Export the Tadpole genesis egg: an untrained base torch model.

This creates a reproducible, randomly initialized Tadpole checkpoint and
packages it with lineage metadata and a standalone SDNA manifest that points
back to GuppyLM.
"""

import argparse
import json
import hashlib
import os
import shutil
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_lineage():
    return {
        "model_name": "Genesis Egg",
        "stage": "genesis_egg",
        "trained": False,
        "ancestor": "GuppyLM",
        "ancestry_chain": ["GuppyLM", "Tadpole", "Genesis Egg"],
        "base_model_repo": "arman-bd/guppylm",
    }


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_sdna_manifest(config, lineage, checkpoint_name, checkpoint_sha256):
    genome_id = f"egg-{checkpoint_sha256[:12]}"
    architecture = {
        "model_family": "tadpole",
        "config_family": "tadpole",
        "vocab_size": config.vocab_size,
        "max_seq_len": config.max_seq_len,
        "d_model": config.d_model,
        "n_layers": config.n_layers,
        "n_heads": config.n_heads,
        "ffn_hidden": config.ffn_hidden,
        "dropout": config.dropout,
        "extra_params": {
            "checkpoint_name": checkpoint_name,
            "trained": False,
        },
    }

    return {
        "sdna_version": "1.0.0",
        "kind": "egg-template",
        "header": {
            "created_at": time.time(),
            "updated_at": time.time(),
            "biome_origin": "tadpole",
            "schema_family": "sdna",
            "compression": "none",
        },
        "identity": {
            "genome_id": genome_id,
            "name": lineage["model_name"],
            "species": "Tadpole",
            "family": "tiny-llm",
            "generation": 0,
            "theme": "metamorphosis",
            "icon": "🥚",
        },
        "lineage": {
            "mode": "mitosis",
            "parents": ["GuppyLM"],
            "ancestor_chain": lineage["ancestry_chain"],
            "mutation_history": [],
            "zygote_id": None,
            "tadpole_id": None,
        },
        "genotype": {
            "architecture": architecture,
            "latent_skills": [
                "text-generation",
                "sequence-modeling",
                "domain-adaptation",
            ],
            "traits": [
                "untrained",
                "base-model",
                "genesis-egg",
            ],
            "modalities": ["text"],
        },
        "embodiment": {
            "egg_id": genome_id,
            "egg_family": "tadpole",
            "parameter_scale": "8.7m",
            "supports_lora": True,
            "supports_distillation": True,
            "supported_modalities": ["text"],
        },
        "phenotype": {
            "quality": 0.0,
            "trust": 0.5,
            "latency_ms": 0.0,
            "cost": 0.0,
            "stability": 1.0,
            "hallucination_rate": 0.0,
            "domain_scores": {},
        },
        "lifecycle": {
            "stage": "egg",
            "maturation_score": 0.0,
            "epochs_completed": 0,
            "training_progress": 0.0,
        },
        "simulation": {
            "lab": "genesis-lab",
            "epoch_id": None,
            "scenario": "untrained-base",
            "fitness_score": 0.0,
            "survival_score": 0.0,
            "cooperation_score": 0.0,
            "resource_efficiency": 0.0,
        },
        "reproduction": {
            "mitosis_enabled": True,
            "meiosis_enabled": False,
            "chimera_allowed": False,
            "egg_transfer_allowed": True,
            "compatible_species": ["GuppyLM"],
        },
        "integrity": {
            "breeder_public_key": "nixpt:tadpole",
            "signature": None,
            "fingerprint": checkpoint_sha256,
            "trust_floor": 0.75,
        },
        "distribution": {
            "registry_url": None,
            "source": {
                "type": "github",
                "repo": "nixpt/Tadpole",
                "filename": checkpoint_name,
                "revision": "main",
                "tag": None,
                "asset": None,
                "sha256": checkpoint_sha256,
            },
            "download_count": 0,
            "license_url": None,
            "is_paid": False,
            "price_credits": 0,
        },
    }


def export_genesis_egg(output_dir, checkpoint_name, tokenizer_path, seed=42):
    from tadpole.config import TadpoleConfig
    from tadpole.model import Tadpole

    torch.manual_seed(seed)

    config = TadpoleConfig()
    model = Tadpole(config)
    model.eval()

    lineage = build_lineage()
    artifact = {
        "step": 0,
        "model_state_dict": model.state_dict(),
        "config": {
            **vars(config),
            "model_name": lineage["model_name"],
            "stage": lineage["stage"],
            "trained": lineage["trained"],
            "ancestor": lineage["ancestor"],
            "base_model_repo": lineage["base_model_repo"],
        },
        "lineage": lineage,
        "seed": seed,
    }

    os.makedirs(output_dir, exist_ok=True)

    checkpoint_path = os.path.join(output_dir, checkpoint_name)
    torch.save(artifact, checkpoint_path)
    checkpoint_sha256 = sha256_file(checkpoint_path)

    hf_state_path = os.path.join(output_dir, "pytorch_model.bin")
    torch.save(model.state_dict(), hf_state_path)

    config_path = os.path.join(output_dir, "config.json")
    hf_config = {
        "model_type": "tadpole",
        "architectures": ["Tadpole"],
        "vocab_size": config.vocab_size,
        "max_position_embeddings": config.max_seq_len,
        "hidden_size": config.d_model,
        "num_hidden_layers": config.n_layers,
        "num_attention_heads": config.n_heads,
        "intermediate_size": config.ffn_hidden,
        "hidden_dropout_prob": config.dropout,
        "pad_token_id": config.pad_id,
        "bos_token_id": config.bos_id,
        "eos_token_id": config.eos_id,
        "model_name": lineage["model_name"],
        "stage": lineage["stage"],
        "trained": lineage["trained"],
        "ancestor": lineage["ancestor"],
        "base_model_repo": lineage["base_model_repo"],
        "ancestry_chain": lineage["ancestry_chain"],
    }
    with open(config_path, "w") as f:
        json.dump(hf_config, f, indent=2)

    lineage_path = os.path.join(output_dir, "lineage.json")
    with open(lineage_path, "w") as f:
        json.dump(lineage, f, indent=2)

    sdna_path = os.path.join(output_dir, "genesis_egg.dna")
    sdna = build_sdna_manifest(config, lineage, checkpoint_name, checkpoint_sha256)
    with open(sdna_path, "w") as f:
        json.dump(sdna, f, indent=2)

    card_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "genesis_egg_model_card.md")
    readme_path = os.path.join(output_dir, "README.md")
    shutil.copy2(card_path, readme_path)

    if os.path.exists(tokenizer_path):
        shutil.copy2(tokenizer_path, os.path.join(output_dir, "tokenizer.json"))

    return checkpoint_path


def main():
    parser = argparse.ArgumentParser(description="Export the Tadpole genesis egg checkpoint")
    parser.add_argument("--output-dir", default="genesis_egg_export")
    parser.add_argument("--checkpoint", default="genesis_egg.pt")
    parser.add_argument("--tokenizer", default="data/tokenizer.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    checkpoint_path = export_genesis_egg(args.output_dir, args.checkpoint, args.tokenizer, args.seed)
    print(f"Exported genesis egg checkpoint to {checkpoint_path}")
    print(f"Metadata written to {args.output_dir}/config.json, {args.output_dir}/lineage.json, and {args.output_dir}/genesis_egg.dna")


if __name__ == "__main__":
    main()
