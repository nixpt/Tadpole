"""Entry point for: python -m gobyllm"""

import sys


def main():
    if len(sys.argv) < 2:
        print("GobyLLM — The Smallest Model That Lives on the Edge")
        print()
        print("Usage:")
        print("  python -m gobyllm train              Train the model")
        print("  python -m gobyllm prepare             Download data & train tokenizer")
        print("  python -m gobyllm serve               Start OpenAI-compatible server")
        print("  python -m gobyllm serve --fast         Start optimized server (KV cache + INT8)")
        print("  python -m gobyllm export               Export to .goby binary for C runtime")
        print("  python -m gobyllm chat                 Interactive chat")
        return

    cmd = sys.argv[1]
    sys.argv = sys.argv[1:]  # shift args

    if cmd == "prepare":
        from .prepare_data import prepare
        prepare()

    elif cmd == "train":
        from .train import train
        train()

    elif cmd == "serve":
        fast = "--fast" in sys.argv
        if fast:
            from .rpi_runner import GobyRunner, run_server
            sys.argv.remove("--fast")
            engine = GobyRunner("checkpoints/best_model.pt", "data/tokenizer.json")
        else:
            from .inference import GobyInference, run_server
            engine = GobyInference("checkpoints/best_model.pt", "data/tokenizer.json")
        port = 8000
        if "--port" in sys.argv:
            idx = sys.argv.index("--port")
            port = int(sys.argv[idx + 1])
        run_server(engine, port=port)

    elif cmd == "export":
        from .export_goby import main as export_main
        export_main()

    elif cmd == "chat":
        from .inference import GobyInference
        engine = GobyInference("checkpoints/best_model.pt", "data/tokenizer.json")
        print("\nGobyLLM Chat (type 'quit' to exit)")
        msgs = [{"role": "system", "content": "You are a helpful assistant."}]
        while True:
            inp = input("\nYou> ").strip()
            if inp.lower() in ("quit", "exit", "q"):
                break
            msgs.append({"role": "user", "content": inp})
            r = engine.chat_completion(msgs)
            msg = r["choices"][0]["message"]
            if msg.get("content"):
                print(f"Goby> {msg['content']}")
            msgs.append(msg)

    else:
        print(f"Unknown command: {cmd}")
        print("Run 'python -m gobyllm' for usage.")


main()
