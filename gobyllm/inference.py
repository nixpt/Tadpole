"""GobyLLM inference -- OpenAI-compatible with early exit reporting."""

import json
import re
import time
import uuid

import torch
from tokenizers import Tokenizer

from .config import GobyConfig
from .model import GobyLLM


class GobyInference:
    def __init__(self, checkpoint_path, tokenizer_path, device="cpu"):
        self.device = torch.device(device)
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.config = GobyConfig(**ckpt["config"])
        self.model = GobyLLM(self.config).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        total, router = self.model.param_count()
        print(f"GobyLLM loaded: {total/1e6:.1f}M params, early_exit={self.config.early_exit}")
        if self.config.early_exit:
            print(f"  Router overhead: {router:,} params ({router/total*100:.2f}%)")
            print(f"  Exit threshold: {self.config.exit_threshold}, min_layer: {self.config.min_exit_layer}")

    def chat_completion(self, messages, tools=None, tool_choice="auto",
                        temperature=0.7, max_tokens=256, top_k=50,
                        model=None, exit_threshold=None, **kwargs):
        """OpenAI-compatible chat completion."""

        model_name = model or "goby-25m"

        effective_tools = tools
        if tool_choice == "none":
            effective_tools = None

        self._last_tools = effective_tools
        prompt = self._format_prompt(messages, effective_tools)
        input_ids = self.tokenizer.encode(prompt).ids
        prompt_tokens = len(input_ids)
        input_t = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        output_t, exit_layers = self.model.generate(
            input_t, max_tokens, temperature, top_k,
            exit_threshold=exit_threshold,
        )
        output_text = self.tokenizer.decode(output_t[0].tolist()[prompt_tokens:])

        resp = self._build_response(output_text, exit_layers, prompt_tokens, model_name)

        # Enforce tool_choice
        msg = resp["choices"][0]["message"]
        if tool_choice == "none":
            msg.pop("tool_calls", None)
            resp["choices"][0]["finish_reason"] = "stop"
        elif isinstance(tool_choice, dict):
            forced = tool_choice.get("function", {}).get("name")
            if forced and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    tc["function"]["name"] = forced

        return resp

    def _format_prompt(self, messages, tools=None):
        parts = []
        has_system = False
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content") or ""
            if role == "tool":
                tool_call_id = msg.get("tool_call_id", "")
                parts.append(f"<|im_start|>tool\n[{tool_call_id}] {content}<|im_end|>")
                continue
            if role == "system":
                has_system = True
                if tools:
                    content += f"\n\n# Tools\n{json.dumps(tools, separators=(',', ':'))}"
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        if not has_system and tools:
            sys = "You are a helpful assistant.\n\n# Tools\n" + json.dumps(tools, separators=(",", ":"))
            parts.insert(0, f"<|im_start|>system\n{sys}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)

    def _build_response(self, text, exit_layers, prompt_tokens=0, model_name="goby-25m"):
        from .json_repair import extract_tool_calls, clean_response_text

        thinking = None
        m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        if m:
            thinking = m.group(1).strip()

        tool_calls = extract_tool_calls(text, tools=self._last_tools)
        resp_text = clean_response_text(text)

        message = {"role": "assistant", "content": resp_text if resp_text else None}
        if tool_calls:
            message["tool_calls"] = tool_calls
            if not resp_text:
                message["content"] = None

        completion_tokens = len(exit_layers)
        avg_exit = sum(exit_layers) / max(len(exit_layers), 1)
        result = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "system_fingerprint": f"goby_{self.config.n_layers}l_{self.config.d_model}d",
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
                "logprobs": None,
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        if thinking:
            result["_thinking"] = thinking
        result["_early_exit"] = {
            "avg_layer": round(avg_exit, 1),
            "max_layers": self.config.n_layers,
            "compute_saved": f"{(1 - avg_exit / self.config.n_layers) * 100:.0f}%",
        }
        return result

    @torch.no_grad()
    def benchmark(self, seq_len=128, n_runs=100):
        """Compare full model vs early exit inference speed."""

        dummy = torch.randint(0, self.config.vocab_size, (1, seq_len),
                              dtype=torch.long, device=self.device)

        # Warmup
        for _ in range(5):
            self.model(dummy)

        # Full model (no early exit)
        old_ee = self.config.early_exit
        self.config.early_exit = False
        times_full = []
        for _ in range(n_runs):
            s = time.perf_counter()
            self.model(dummy)
            times_full.append(time.perf_counter() - s)

        # With early exit
        self.config.early_exit = old_ee
        times_ee = []
        exit_layer_sum = 0
        for _ in range(n_runs):
            prompt = torch.randint(0, self.config.vocab_size, (1, 16), device=self.device)
            s = time.perf_counter()
            _, exits = self.model.generate(prompt, max_new_tokens=1)
            times_ee.append(time.perf_counter() - s)
            exit_layer_sum += exits[0]

        avg_full = sum(times_full) / len(times_full) * 1000
        avg_ee = sum(times_ee) / len(times_ee) * 1000
        avg_exit = exit_layer_sum / n_runs

        print(f"\nBenchmark ({self.device}, {n_runs} runs):")
        print(f"  Full model:  {avg_full:.2f}ms/forward")
        print(f"  Early exit:  {avg_ee:.2f}ms/generate (avg layer {avg_exit:.1f}/{self.config.n_layers})")
        if avg_full > 0:
            print(f"  Potential speedup: {avg_full/avg_ee:.1f}x (on untrained model — improves after training)")


# ── HTTP server ─────────────────────────────────────────────────────────


def run_server(engine, host="0.0.0.0", port=8000):
    from http.server import HTTPServer, BaseHTTPRequestHandler

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path == "/v1/chat/completions":
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                result = engine.chat_completion(
                    messages=body.get("messages", []),
                    tools=body.get("tools"),
                    temperature=body.get("temperature", 0.7),
                    max_tokens=body.get("max_tokens", 256),
                )
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())
            else:
                self.send_response(404)
                self.end_headers()

        def do_GET(self):
            if self.path == "/v1/models":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "data": [{"id": "goby-25m", "object": "model",
                              "meta": {"early_exit": True, "params": "25.5M"}}]
                }).encode())
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, fmt, *args):
            print(f"[{self.log_date_time_string()}] {fmt % args}")

    srv = HTTPServer((host, port), H)
    print(f"GobyLLM server at http://{host}:{port}")
    print(f'  client = OpenAI(base_url="http://localhost:{port}/v1", api_key="unused")')
    srv.serve_forever()


# ── CLI ─────────────────────────────────────────────────────────────────


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    p.add_argument("--tokenizer", default="data/tokenizer.json")
    p.add_argument("--device", default="cpu")
    p.add_argument("--serve", action="store_true")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--interactive", action="store_true")
    args = p.parse_args()

    engine = GobyInference(args.checkpoint, args.tokenizer, args.device)

    if args.benchmark:
        engine.benchmark()
    elif args.serve:
        run_server(engine, port=args.port)
    elif args.interactive:
        tools = [
            {"type": "function", "function": {"name": "turn_on", "description": "Turn on a device",
             "parameters": {"type": "object", "properties": {"device": {"type": "string"}}, "required": ["device"]}}},
            {"type": "function", "function": {"name": "set_temperature", "description": "Set thermostat",
             "parameters": {"type": "object", "properties": {
                 "temperature": {"type": "number"}, "mode": {"type": "string", "enum": ["heat","cool","auto"]}
             }, "required": ["temperature", "mode"]}}},
        ]
        print("\nGobyLLM Interactive (type 'quit' to exit)")
        msgs = [{"role": "system", "content": "You are a helpful assistant."}]
        while True:
            inp = input("\nYou> ").strip()
            if inp.lower() in ("quit", "exit", "q"):
                break
            msgs.append({"role": "user", "content": inp})
            result = engine.chat_completion(msgs, tools=tools)
            ch = result["choices"][0]
            ee = result.get("_early_exit", {})
            if ch.get("_thinking"):
                print(f"[Think] {ch['_thinking']}")
            msg = ch["message"]
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    print(f"[Tool] {tc['function']['name']}({tc['function']['arguments']})")
            if msg.get("content"):
                print(f"Goby> {msg['content']}")
            if ee:
                print(f"  [exit layer {ee['avg_layer']}/{ee['max_layers']}, {ee['compute_saved']} saved]")
            msgs.append(msg)


if __name__ == "__main__":
    main()
