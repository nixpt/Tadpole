"""GobyLLM RPi Runner -- optimized inference for Raspberry Pi and edge devices."""

import argparse
import json
import math
import re
import time
import uuid

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

from .config import GobyConfig
from .model import GobyLLM, apply_rope


class KVCache:
    """Pre-allocated KV cache for a single attention layer."""
    def __init__(self, max_seq_len, n_kv_heads, head_dim, device, dtype):
        self.max_seq_len = max_seq_len
        self.k = torch.zeros(1, n_kv_heads, max_seq_len, head_dim, device=device, dtype=dtype)
        self.v = torch.zeros(1, n_kv_heads, max_seq_len, head_dim, device=device, dtype=dtype)
        self.pos = 0

    def update(self, k_new, v_new):
        """Append new KV to cache. k_new/v_new: [1, n_kv_heads, T, head_dim]"""
        T = k_new.shape[2]
        end = self.pos + T
        self.k[:, :, self.pos:end, :] = k_new
        self.v[:, :, self.pos:end, :] = v_new
        self.pos = end

    def get(self):
        """Return cached K, V up to current position."""
        return self.k[:, :, :self.pos, :], self.v[:, :, :self.pos, :]

    def reset(self):
        self.pos = 0


class GobyRunner:
    """Optimized GobyLLM inference engine for edge devices."""

    def __init__(self, checkpoint_path, tokenizer_path, quantize=True, device="cpu"):
        self.device = torch.device(device)
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        # Load model
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.config = GobyConfig(**ckpt["config"])
        self.model = GobyLLM(self.config).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        total, router = self.model.param_count()
        fp32_mb = total * 4 / 1e6
        print(f"GobyLLM loaded: {total/1e6:.1f}M params ({fp32_mb:.0f}MB FP32)")

        # INT8 quantization
        self.quantized = False
        if quantize and device == "cpu":
            self._quantize()

        # Pre-extract model components for fast access
        self.tok_emb = self.model.tok_emb
        self.blocks = self.model.blocks
        self.norm = self.model.norm
        self.lm_head = self.model.lm_head
        self.rope_cos = self.model.rope_cos
        self.rope_sin = self.model.rope_sin
        self.exit_routers = self.model.exit_routers if self.config.early_exit else None
        self.n_layers = self.config.n_layers
        self.head_dim = self.config.d_model // self.config.n_heads

        # Pre-allocate KV caches
        self.kv_caches = self._alloc_caches()

        # Warmup
        self._warmup()

    def _quantize(self):
        """Apply INT8 dynamic quantization."""

        # Untie weights before quantization (shared params confuse quantizer)
        self.model.lm_head = nn.Linear(
            self.config.d_model, self.config.vocab_size, bias=False
        )
        self.model.lm_head.weight = nn.Parameter(self.model.tok_emb.weight.clone())

        self.model = torch.quantization.quantize_dynamic(
            self.model, {nn.Linear}, dtype=torch.qint8
        )
        self.quantized = True

        # Estimate size
        param_bytes = sum(
            p.nelement() * (1 if p.dtype == torch.qint8 else p.element_size())
            for p in self.model.parameters()
        )
        print(f"INT8 quantized: ~{param_bytes/1e6:.0f}MB")

    def _alloc_caches(self):
        """Pre-allocate KV caches for all layers."""
        dtype = torch.float32
        return [
            KVCache(
                self.config.max_seq_len, self.config.n_kv_heads,
                self.head_dim, self.device, dtype
            )
            for _ in range(self.n_layers)
        ]

    def _reset_caches(self):
        for c in self.kv_caches:
            c.reset()

    def _warmup(self):
        """Run a dummy forward pass to trigger lazy initializations."""
        dummy = torch.tensor([[1, 2, 3]], dtype=torch.long, device=self.device)
        self._reset_caches()
        self._forward_prompt(dummy)
        self._reset_caches()
        print("Warmup done")

    # ── Cached forward passes ───────────────────────────────────────────

    def _attention_cached(self, attn, x, rope_cos, rope_sin, kv_cache):
        """Attention forward with KV cache update."""
        B, T, _ = x.shape

        q = attn.wq(x).view(B, T, attn.n_heads, self.head_dim).transpose(1, 2)
        k = attn.wk(x).view(B, T, attn.n_kv_heads, self.head_dim).transpose(1, 2)
        v = attn.wv(x).view(B, T, attn.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        # Update cache
        kv_cache.update(k, v)
        k_full, v_full = kv_cache.get()

        # Expand KV heads for GQA
        if attn.n_rep > 1:
            k_full = k_full.repeat_interleave(attn.n_rep, dim=1)
            v_full = v_full.repeat_interleave(attn.n_rep, dim=1)

        # Attention
        scale = math.sqrt(self.head_dim)
        scores = (q @ k_full.transpose(-2, -1)) / scale

        # Causal mask only needed for prompt (T > 1)
        if T > 1:
            kv_len = k_full.shape[2]
            mask = torch.tril(torch.ones(T, kv_len, device=x.device))
            # Offset: if cache had prev tokens, mask should allow attending to them
            # Since we only call T>1 for prompt (cache starts empty), standard causal is fine
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        out = (attn_weights @ v_full).transpose(1, 2).contiguous().view(B, T, -1)
        return attn.wo(out)

    def _block_cached(self, block, x, rope_cos, rope_sin, kv_cache):
        """Single block forward with KV cache (supports parallel residual)."""
        if block.parallel:
            h = block.norm(x)
            attn_out = self._attention_cached(block.attn, h, rope_cos, rope_sin, kv_cache)
            ffn_out = block.ffn(h)
            return x + attn_out + ffn_out
        else:
            h = block.norm(x)
            attn_out = self._attention_cached(block.attn, h, rope_cos, rope_sin, kv_cache)
            x = x + attn_out
            return x + block.ffn(block.norm2(x))

    def _forward_prompt(self, token_ids, max_depth=None):
        """Process prompt through layers with KV caching."""

        if max_depth is None:
            max_depth = self.n_layers

        B, T = token_ids.shape
        x = self.tok_emb(token_ids)
        rope_cos = self.rope_cos[:T]
        rope_sin = self.rope_sin[:T]

        exit_depth = self.n_layers
        for i in range(self.n_layers):
            x = self._block_cached(self.blocks[i], x, rope_cos, rope_sin, self.kv_caches[i])

            # Check router for early exit depth determination
            if (self.config.early_exit and self.exit_routers is not None
                    and i >= self.config.min_exit_layer
                    and i < self.n_layers - 1):
                conf = torch.sigmoid(self.exit_routers[i](x[:, -1:, :].mean(dim=1)))
                if conf.item() > self.config.exit_threshold and exit_depth == self.n_layers:
                    exit_depth = i + 1
                    # Don't break — we still need to fill ALL KV caches for flexibility
                    # But we record the depth for generation

        return x, exit_depth

    def _forward_one_token(self, token_id, pos, depth):
        """Process a single new token through `depth` layers using KV cache."""

        x = self.tok_emb(token_id)  # [1, 1, d_model]
        rope_cos = self.rope_cos[pos:pos + 1]
        rope_sin = self.rope_sin[pos:pos + 1]

        for i in range(depth):
            x = self._block_cached(self.blocks[i], x, rope_cos, rope_sin, self.kv_caches[i])

        logits = self.lm_head(self.norm(x))  # [1, 1, vocab]
        return logits[:, -1, :]  # [1, vocab]

    # ── Generation ──────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(self, prompt_ids, max_tokens=256, temperature=0.7, top_k=50):
        """KV-cached generation with fixed-depth early exit."""

        self._reset_caches()

        prompt_ids = prompt_ids.to(self.device)
        B, prompt_len = prompt_ids.shape

        # Phase 1: prompt processing (all layers, to fill KV cache)
        t_prompt_start = time.perf_counter()
        hidden, exit_depth = self._forward_prompt(prompt_ids)
        logits = self.lm_head(self.norm(hidden))[:, -1, :]
        t_prompt = time.perf_counter() - t_prompt_start

        # Phase 2: token-by-token generation using KV cache + fixed depth
        generated = []
        t_gen_start = time.perf_counter()

        for step in range(max_tokens):
            logits_scaled = logits / temperature
            if top_k > 0:
                v, _ = torch.topk(logits_scaled, min(top_k, logits_scaled.size(-1)))
                logits_scaled[logits_scaled < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits_scaled, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # [1, 1]
            generated.append(next_id.item())

            if next_id.item() == self.config.eos_id:
                break

            pos = prompt_len + step
            if pos >= self.config.max_seq_len - 1:
                break

            logits = self._forward_one_token(next_id, pos, exit_depth)

        t_gen = time.perf_counter() - t_gen_start
        n_gen = len(generated)

        output_ids = torch.cat([
            prompt_ids,
            torch.tensor([generated], dtype=torch.long, device=self.device)
        ], dim=1)

        meta = {
            "prompt_tokens": prompt_len,
            "generated_tokens": n_gen,
            "exit_depth": exit_depth,
            "max_depth": self.n_layers,
            "compute_saved": f"{(1 - exit_depth / self.n_layers) * 100:.0f}%",
            "prompt_ms": round(t_prompt * 1000, 1),
            "gen_ms": round(t_gen * 1000, 1),
            "tokens_per_sec": round(n_gen / t_gen, 1) if t_gen > 0 else 0,
        }

        return output_ids, meta

    # ── OpenAI-compatible API ───────────────────────────────────────────

    def chat_completion(self, messages, tools=None, tool_choice="auto",
                        temperature=0.7, max_tokens=256, top_k=50,
                        model=None, n=1, stop=None, stream=False,
                        **kwargs):
        """OpenAI-compatible chat completion."""

        model_name = model or "goby-25m"

        # Handle tool_choice
        effective_tools = tools
        forced_tool = None
        if tool_choice == "none":
            effective_tools = None  # hide tools from model
        elif isinstance(tool_choice, dict):
            # Force a specific tool
            forced_tool = tool_choice.get("function", {}).get("name")

        self._last_tools = effective_tools
        prompt = self._format_prompt(messages, effective_tools)
        input_ids = self.tokenizer.encode(prompt).ids
        prompt_tokens = len(input_ids)
        input_t = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        output_t, meta = self.generate(input_t, max_tokens, temperature, top_k)
        output_text = self.tokenizer.decode(output_t[0].tolist()[prompt_tokens:])

        resp = self._build_response(output_text, meta, model_name, prompt_tokens)

        # Enforce tool_choice constraints
        msg = resp["choices"][0]["message"]
        if tool_choice == "none":
            msg.pop("tool_calls", None)
            resp["choices"][0]["finish_reason"] = "stop"
        elif tool_choice == "required" and not msg.get("tool_calls"):
            # Model didn't call a tool but was required to — mark as failure
            resp["choices"][0]["finish_reason"] = "stop"
        elif forced_tool and msg.get("tool_calls"):
            # Override tool name to the forced one
            for tc in msg["tool_calls"]:
                tc["function"]["name"] = forced_tool
            resp["choices"][0]["finish_reason"] = "tool_calls"

        return resp

    def _format_prompt(self, messages, tools=None):
        parts = []
        has_system = False
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content") or ""

            # Handle tool results in conversation
            if role == "tool":
                tool_call_id = msg.get("tool_call_id", "")
                parts.append(f"<|im_start|>tool\n[{tool_call_id}] {content}<|im_end|>")
                continue

            # Inject tools into system prompt
            if role == "system":
                has_system = True
                if tools:
                    content += f"\n\n# Tools\n{json.dumps(tools, separators=(',', ':'))}"

            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

        # If no system message was provided, add one with tools
        if not has_system and tools:
            sys_content = "You are a helpful assistant.\n\n# Tools\n" + json.dumps(tools, separators=(",", ":"))
            parts.insert(0, f"<|im_start|>system\n{sys_content}<|im_end|>")

        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)

    def _build_response(self, text, meta, model_name="goby-25m",
                        prompt_tokens=0, stream=False):
        from .json_repair import extract_tool_calls, clean_response_text

        thinking = None
        m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        if m:
            thinking = m.group(1).strip()

        tool_calls = extract_tool_calls(text, tools=getattr(self, '_last_tools', None))
        resp_text = clean_response_text(text)

        # OpenAI spec: content must be null when tool_calls are present, not empty string
        message = {"role": "assistant", "content": resp_text if resp_text else None}
        if tool_calls:
            message["tool_calls"] = tool_calls
            if not resp_text:
                message["content"] = None

        finish_reason = "tool_calls" if tool_calls else "stop"
        completion_tokens = meta.get("tokens_generated", 0)

        resp = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "system_fingerprint": f"goby_{self.config.n_layers}l_{self.config.d_model}d",
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
                "logprobs": None,
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

        # Non-standard extensions (prefixed with underscore)
        if thinking:
            resp["_thinking"] = thinking
        if meta:
            resp["_performance"] = meta

        return resp

    # ── Benchmarking ────────────────────────────────────────────────────

    def benchmark(self, prompt_len=32, gen_tokens=64, n_runs=20):
        """Benchmark cached generation vs naive (recompute-all) generation."""

        print(f"\n{'='*60}")
        print(f"GobyLLM Benchmark ({'INT8' if self.quantized else 'FP32'}, {self.device})")
        print(f"  Prompt: {prompt_len} tokens, Generate: {gen_tokens} tokens")
        print(f"{'='*60}")

        prompt = torch.randint(1, self.config.vocab_size, (1, prompt_len), device=self.device)

        # Warm up
        for _ in range(3):
            self.generate(prompt, max_tokens=8)

        # Cached generation (this runner)
        times_cached = []
        depths = []
        for _ in range(n_runs):
            s = time.perf_counter()
            _, meta = self.generate(prompt, max_tokens=gen_tokens, temperature=0.8)
            times_cached.append(time.perf_counter() - s)
            depths.append(meta["exit_depth"])

        avg_cached = sum(times_cached) / len(times_cached) * 1000
        avg_depth = sum(depths) / len(depths)
        tok_per_sec = gen_tokens / (avg_cached / 1000)

        # Naive generation (no KV cache, full recompute — like inference.py)
        times_naive = []
        for _ in range(min(n_runs, 5)):  # fewer runs since it's slow
            s = time.perf_counter()
            idx = prompt.clone()
            for _ in range(gen_tokens):
                idx_cond = idx[:, -self.config.max_seq_len:]
                with torch.no_grad():
                    logits, _ = self.model(idx_cond)
                next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_id], dim=1)
            times_naive.append(time.perf_counter() - s)

        avg_naive = sum(times_naive) / len(times_naive) * 1000

        print(f"\n  Naive (no cache):     {avg_naive:8.1f}ms  ({gen_tokens*1000/avg_naive:5.1f} tok/s)")
        print(f"  KV-cached + early exit: {avg_cached:8.1f}ms  ({tok_per_sec:5.1f} tok/s)")
        print(f"  Speedup:              {avg_naive/avg_cached:8.1f}x")
        print(f"  Avg exit depth:       {avg_depth:.1f} / {self.n_layers} layers")
        print(f"  Compute saved by EE:  {(1 - avg_depth/self.n_layers)*100:.0f}%")

        # Memory estimate
        model_mb = sum(p.nelement() * p.element_size() for p in self.model.parameters()) / 1e6
        cache_mb = sum(
            c.k.nelement() * c.k.element_size() + c.v.nelement() * c.v.element_size()
            for c in self.kv_caches
        ) / 1e6
        print(f"\n  Model memory:  {model_mb:.1f}MB")
        print(f"  KV cache:      {cache_mb:.1f}MB (pre-allocated for {self.config.max_seq_len} tokens)")
        print(f"  Total:         {model_mb + cache_mb:.1f}MB")


# ── OpenAI-compatible HTTP Server ───────────────────────────────────────────

def run_server(engine, host="0.0.0.0", port=8000):
    from http.server import HTTPServer, BaseHTTPRequestHandler
    from urllib.parse import urlparse
    import traceback

    MODEL_ID = "goby-25m"
    CREATED = int(time.time())

    def error_response(code, message, err_type="invalid_request_error"):
        return json.dumps({
            "error": {
                "message": message,
                "type": err_type,
                "param": None,
                "code": None,
            }
        }).encode()

    class Handler(BaseHTTPRequestHandler):

        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers",
                             "Content-Type, Authorization, X-Request-ID")

        def _json_response(self, code, data):
            body = json.dumps(data).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self.send_response(204)
            self._cors()
            self.end_headers()

        def do_GET(self):
            path = urlparse(self.path).path

            if path == "/v1/models":
                self._json_response(200, {
                    "object": "list",
                    "data": [{
                        "id": MODEL_ID,
                        "object": "model",
                        "created": CREATED,
                        "owned_by": "goby",
                        "permission": [],
                        "root": MODEL_ID,
                        "parent": None,
                    }]
                })

            elif path.startswith("/v1/models/"):
                model_id = path.split("/v1/models/")[1]
                if model_id == MODEL_ID:
                    self._json_response(200, {
                        "id": MODEL_ID,
                        "object": "model",
                        "created": CREATED,
                        "owned_by": "goby",
                    })
                else:
                    self.send_response(404)
                    self.end_headers()
                    self.wfile.write(error_response(404, f"Model '{model_id}' not found", "not_found"))

            elif path == "/health" or path == "/":
                self._json_response(200, {"status": "ok", "model": MODEL_ID})

            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            path = urlparse(self.path).path
            content_len = int(self.headers.get("Content-Length", 0))

            if path == "/v1/chat/completions":
                try:
                    body = json.loads(self.rfile.read(content_len))
                except json.JSONDecodeError:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(error_response(400, "Invalid JSON in request body"))
                    return

                messages = body.get("messages")
                if not messages:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(error_response(400, "'messages' is required"))
                    return

                # Stream requested — we don't support it, return error per OpenAI convention
                if body.get("stream", False):
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(error_response(400, "Streaming is not supported by this model"))
                    return

                try:
                    result = engine.chat_completion(
                        messages=messages,
                        tools=body.get("tools"),
                        tool_choice=body.get("tool_choice", "auto"),
                        temperature=body.get("temperature", 0.7),
                        max_tokens=body.get("max_tokens", 256),
                        model=body.get("model"),
                        stop=body.get("stop"),
                    )
                    self._json_response(200, result)

                except Exception as e:
                    traceback.print_exc()
                    self.send_response(500)
                    self.end_headers()
                    self.wfile.write(error_response(500, str(e), "internal_error"))

            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, fmt, *args):
            print(f"[{self.log_date_time_string()}] {fmt % args}")

    srv = HTTPServer((host, port), Handler)
    print(f"\nGobyLLM OpenAI-compatible server")
    print(f"  http://{host}:{port}")
    print(f"  Model: {MODEL_ID} ({'INT8' if engine.quantized else 'FP32'})")
    print(f"  Early exit: {'ON' if engine.config.early_exit else 'OFF'}")
    print(f"\nEndpoints:")
    print(f"  POST /v1/chat/completions  — chat (tools, tool_choice supported)")
    print(f"  GET  /v1/models            — list models")
    print(f"  GET  /v1/models/{MODEL_ID} — model details")
    print(f"  GET  /health               — health check")
    print(f"\nUsage:")
    print(f'  from openai import OpenAI')
    print(f'  client = OpenAI(base_url="http://localhost:{port}/v1", api_key="not-needed")')
    srv.serve_forever()


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="GobyLLM RPi Runner")
    p.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    p.add_argument("--tokenizer", default="data/tokenizer.json")
    p.add_argument("--device", default="cpu")
    p.add_argument("--no-quantize", action="store_true", help="Disable INT8 quantization")
    p.add_argument("--serve", action="store_true")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--interactive", action="store_true")
    args = p.parse_args()

    engine = GobyRunner(
        args.checkpoint, args.tokenizer,
        quantize=not args.no_quantize,
        device=args.device,
    )

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
        print(f"\nGobyLLM Interactive ({'INT8' if engine.quantized else 'FP32'}) — type 'quit' to exit")
        msgs = [{"role": "system", "content": "You are a helpful assistant."}]
        while True:
            inp = input("\nYou> ").strip()
            if inp.lower() in ("quit", "exit", "q"):
                break
            msgs.append({"role": "user", "content": inp})
            result = engine.chat_completion(msgs, tools=tools)
            ch = result["choices"][0]
            perf = result.get("_performance", {})
            if ch.get("_thinking"):
                print(f"  [think] {ch['_thinking'][:100]}")
            msg = ch["message"]
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    print(f"  [tool] {tc['function']['name']}({tc['function']['arguments']})")
            if msg.get("content"):
                print(f"Goby> {msg['content']}")
            if perf:
                print(f"  [{perf.get('tokens_per_sec',0)} tok/s, "
                      f"depth {perf.get('exit_depth','?')}/{perf.get('max_depth','?')}, "
                      f"{perf.get('compute_saved','?')} saved]")
            msgs.append(msg)
    else:
        print("Use --serve, --interactive, or --benchmark")


if __name__ == "__main__":
    main()