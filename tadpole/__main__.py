"""Entry point for: python -m tadpole"""

import sys

from .chat_assets import download_model, resolve_chat_assets


def main():
    if len(sys.argv) < 2:
        print("Tadpole — A tiny rama-zpu assistant")
        print()
        print("Usage:")
        print("  python -m tadpole train        Train the model")
        print("  python -m tadpole prepare      Generate data & train tokenizer")
        print("  python -m tadpole chat         Chat with Tadpole")
        print("  python -m tadpole cell         Run the phase-1 digital cell demo")
        print("  python -m tadpole sdna         Show an SDNA v2 snapshot of the cell")
        print("  python -m tadpole species      Run the three-cell file task demo")
        print("  python -m tadpole monkey       Run the 16-cell monkey species demo")
        print("  python -m tadpole species-chat Chat with a Symbiome species")
        print("  python -m tadpole download     Download base model (upstream reference)")
        return

    cmd = sys.argv[1]
    sys.argv = sys.argv[1:]

    if cmd == "prepare":
        from .prepare_data import prepare
        import argparse

        parser = argparse.ArgumentParser(prog="python -m tadpole prepare")
        parser.add_argument("--source", choices=["synthetic", "mixed"], default="synthetic")
        parser.add_argument("--mixed-pack-dir", default="data/hf_mixed_starter_pack")
        parser.add_argument("--samples", type=int, default=60000)
        args = parser.parse_args(sys.argv[1:])
        prepare(n_samples=args.samples, source=args.source, mixed_pack_dir=args.mixed_pack_dir)

    elif cmd == "train":
        from .train import train
        train()

    elif cmd == "download":
        download_model()

    elif cmd == "chat":
        from .inference import main as inference_main
        inference_main()

    elif cmd == "cell":
        from symbiome.biology import run_demo
        import argparse

        parser = argparse.ArgumentParser(prog="python -m tadpole cell")
        parser.add_argument("--steps", type=int, default=3)
        parser.add_argument("--signal", action="append", default=[], help="signal.NAME=VALUE")
        args = parser.parse_args(sys.argv[1:])

        signals = {}
        for item in args.signal:
            if "=" not in item:
                raise SystemExit(f"Invalid signal override: {item}")
            key, value = item.split("=", 1)
            signals[key] = float(value)
        run_demo(steps=args.steps, signals=signals)

    elif cmd == "sdna":
        from symbiome.biology import DigitalCell, SDNAV2
        import argparse
        import json

        parser = argparse.ArgumentParser(prog="python -m tadpole sdna")
        parser.add_argument("--steps", type=int, default=1)
        args = parser.parse_args(sys.argv[1:])

        cell = DigitalCell.default()
        for _ in range(args.steps):
            cell.step({"signal.glucose": 1.0})
        sdna = SDNAV2.from_cell(cell)
        print(json.dumps(sdna.to_dict(), indent=2))

    elif cmd == "species":
        from symbiome.biology import TriCellSpecies
        import argparse

        parser = argparse.ArgumentParser(prog="python -m tadpole species")
        parser.add_argument("--input", required=True, help="Input text file")
        parser.add_argument("--output", required=True, help="Output text file")
        parser.add_argument("--task", default="copy", choices=["copy", "annotate", "uppercase"])
        args = parser.parse_args(sys.argv[1:])

        species = TriCellSpecies.default()
        result = species.process_text_file(args.input, args.output, task=args.task)
        print(result.output_path)

    elif cmd == "monkey":
        from symbiome.biology import DigitalMonkeySpecies
        import argparse

        parser = argparse.ArgumentParser(prog="python -m tadpole monkey")
        parser.add_argument("--input", required=True, help="Input text file")
        parser.add_argument("--output", required=True, help="Output text file")
        parser.add_argument("--task", default="copy", choices=["copy", "annotate", "uppercase"])
        parser.add_argument("--cells", type=int, default=16, choices=[16, 32, 128], help="Cell count")
        args = parser.parse_args(sys.argv[1:])

        species = DigitalMonkeySpecies.training_layout(args.cells) if args.cells == 128 else DigitalMonkeySpecies.default(args.cells)
        result = species.process_text_file(args.input, args.output, task=args.task)
        print(result.output_path)

    elif cmd == "species-chat":
        from symbiome.biology import DigitalMonkeySpecies, TriCellSpecies
        from .inference import GuppyInference
        import argparse

        parser = argparse.ArgumentParser(prog="python -m tadpole species-chat")
        parser.add_argument("--species", choices=["tri-cell", "monkey"], default="monkey")
        parser.add_argument("--cells", type=int, default=16, choices=[16, 32, 64, 128], help="Cell count for monkey")
        parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
        parser.add_argument("--tokenizer", default="data/tokenizer.json")
        parser.add_argument("--device", default="cpu")
        parser.add_argument("--prompt", "-p", help="Single prompt mode: ask one question and exit")
        args = parser.parse_args(sys.argv[1:])

        if args.species == "tri-cell":
            species = TriCellSpecies.default()
        elif args.cells == 128:
            species = DigitalMonkeySpecies.training_layout(args.cells)
        elif args.cells in (16, 32):
            species = DigitalMonkeySpecies.default(args.cells)
        else:
            species = DigitalMonkeySpecies.stress_test(args.cells)

        checkpoint_path, tokenizer_path = resolve_chat_assets(args.checkpoint, args.tokenizer)
        engine = GuppyInference(checkpoint_path, tokenizer_path, args.device)
        if args.prompt:
            result = engine.chat_completion_for_species(species, [{"role": "user", "content": args.prompt}])
            print(result["choices"][0]["message"]["content"])
            return

        print("\nSpecies Chat (type 'quit' to exit)")
        while True:
            inp = input("\nYou> ").strip()
            if inp.lower() in ("quit", "exit", "q"):
                break
            result = engine.chat_completion_for_species(species, [{"role": "user", "content": inp}])
            msg = result["choices"][0]["message"]
            if msg.get("content"):
                print(f"{species.name}> {msg['content']}")

    else:
        print(f"Unknown command: {cmd}")
        print("Run 'python -m tadpole' for usage.")


main()
