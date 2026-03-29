"""Generate a self-contained Colab notebook for GobyLLM training."""

import json
import os
import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_file(path):
    with open(path) as f:
        return f.read()


def read_for_colab(path):
    """Read a file and convert relative imports to flat (for Colab)."""
    content = read_file(path)
    # from .config import X  →  from config import X  (handles indented imports too)
    content = re.sub(r'from \.(\w+) import', r'from \1 import', content)
    return content


def cell(source, cell_type="code"):
    lines = source.split("\n")
    formatted = []
    for i, line in enumerate(lines):
        formatted.append(line + "\n" if i < len(lines) - 1 else line)
    base = {"cell_type": cell_type, "metadata": {}, "source": formatted}
    if cell_type == "code":
        base["outputs"] = []
        base["execution_count"] = None
    return base


def md(text):
    return cell(text, "markdown")


def code(text):
    return cell(text, "code")


# (display name, actual path relative to project root)
FILES = [
    ("config.py",        "gobyllm/config.py"),
    ("model.py",         "gobyllm/model.py"),
    ("generate_data.py", "gobyllm/generate_data.py"),
    ("prepare_data.py",  "gobyllm/prepare_data.py"),
    ("dataset.py",       "gobyllm/dataset.py"),
    ("train.py",         "gobyllm/train.py"),
    ("json_repair.py",   "gobyllm/json_repair.py"),
    ("inference.py",     "gobyllm/inference.py"),
    ("rpi_runner.py",    "gobyllm/rpi_runner.py"),
    ("export_goby.py",   "gobyllm/export_goby.py"),
    ("goby.c",           "csrc/goby.c"),
    ("Makefile",         "csrc/Makefile"),
]


def build():
    cells = []

    cells.append(md(
        "# GobyLLM — 25M Parameter LLM with Learned Early Exit\n"
        "\n"
        "A novel, compact language model trained on Stanford Alpaca + tool-calling data.\n"
        "**First <50M model with trained early exit** — easy queries skip 70% of layers.\n"
        "\n"
        "**Architecture:**\n"
        "- 25.5M parameters, 10-layer transformer\n"
        "- **Learned Early Exit** — per-layer routers decide when to stop (0.02% overhead)\n"
        "- Parallel Residual Blocks (PaLM-style) — attention & FFN in parallel\n"
        "- Grouped Query Attention — 8 query heads, 2 KV heads (4x less KV cache)\n"
        "- Auxiliary exit losses at every layer — model learns to decode from any depth\n"
        "- SwiGLU FFN, RMSNorm, RoPE, BPE tokenizer (8192 vocab)\n"
        "\n"
        "**Capabilities:** General instruction following, reasoning, OpenAI-compatible tool calling\n"
        "\n"
        "**Edge performance:** Simple commands exit at layer 2-3 → **~3x faster** on RPi"
    ))

    # ── Setup ──────────────────────────────────────────────────────────
    cells.append(md("## 1. Setup"))

    cells.append(code(
        "!pip install -q torch tokenizers tqdm requests numpy\n"
        "\n"
        "import torch\n"
        "print(f'PyTorch {torch.__version__}')\n"
        "print(f'CUDA: {torch.cuda.is_available()}')\n"
        "if torch.cuda.is_available():\n"
        "    print(f'GPU: {torch.cuda.get_device_name(0)}')"
    ))

    cells.append(code(
        "import os\n"
        "os.makedirs('/content/goby', exist_ok=True)\n"
        "os.chdir('/content/goby')\n"
        "print(f'Working dir: {os.getcwd()}')"
    ))

    # ── Write source files ─────────────────────────────────────────────
    cells.append(md("## 2. Source Files"))

    for display_name, src_path in FILES:
        full_path = os.path.join(PROJECT_ROOT, src_path)
        if src_path.endswith(".py"):
            content = read_for_colab(full_path)
        else:
            content = read_file(full_path)
        cells.append(md(f"### `{display_name}`"))
        cells.append(code(f"%%writefile {display_name}\n{content}"))

    # ── Prepare data ───────────────────────────────────────────────────
    cells.append(md(
        "## 3. Prepare Data\n"
        "\n"
        "Downloads Stanford Alpaca (52K samples), generates tool-calling data (15K samples),\n"
        "trains a BPE tokenizer, and merges everything."
    ))

    cells.append(code("!python prepare_data.py"))

    cells.append(code(
        "# Preview a sample\n"
        "import json\n"
        "with open('data/train.jsonl') as f:\n"
        "    s = json.loads(f.readline())\n"
        "print(f'Source: {s[\"source\"]}')\n"
        "print(s['text'][:500])\n"
        "print('...')"
    ))

    # ── Verify model ───────────────────────────────────────────────────
    cells.append(md("## 4. Verify Architecture"))

    cells.append(code(
        "from config import GobyConfig\n"
        "from model import GobyLLM\n"
        "import torch\n"
        "\n"
        "config = GobyConfig()\n"
        "model = GobyLLM(config)\n"
        "print(model.param_summary())\n"
        "print(f'  GQA: {config.n_heads}Q / {config.n_kv_heads}KV')\n"
        "print(f'  Parallel residual: {config.parallel_residual}')\n"
        "print(f'  Early exit: min_layer={config.min_exit_layer}, threshold={config.exit_threshold}')\n"
        "\n"
        "x = torch.randint(0, config.vocab_size, (2, 64))\n"
        "logits, _ = model(x)\n"
        "print(f'  Forward: {x.shape} -> {logits.shape}')\n"
        "del model"
    ))

    # ── Train ──────────────────────────────────────────────────────────
    cells.append(md(
        "## 5. Train\n"
        "\n"
        "10,000 steps. Takes ~15-20 min on T4."
    ))

    cells.append(code("from train import train\ntrain()"))

    # ── Test: Tool Calling ─────────────────────────────────────────────
    cells.append(md(
        "## 6. Test Tool Calling\n"
        "\n"
        "Define a rich set of tools and test diverse calling patterns:\n"
        "direct commands, implicit intent, multi-tool selection, refusal, parameter precision."
    ))

    cells.append(code(
        "from inference import GobyInference\n"
        "import json, torch\n"
        "\n"
        "engine = GobyInference(\n"
        "    'checkpoints/best_model.pt', 'data/tokenizer.json',\n"
        "    device='cuda' if torch.cuda.is_available() else 'cpu'\n"
        ")\n"
        "\n"
        "# ── Tool definitions ────────────────────────────────────────────\n"
        "TOOLS = [\n"
        "    {'type': 'function', 'function': {\n"
        "        'name': 'turn_on',\n"
        "        'description': 'Turn on a device (light, fan, appliance, etc.)',\n"
        "        'parameters': {'type': 'object', 'properties': {\n"
        "            'device': {'type': 'string', 'description': 'Device name, e.g. kitchen lights'},\n"
        "        }, 'required': ['device']},\n"
        "    }},\n"
        "    {'type': 'function', 'function': {\n"
        "        'name': 'turn_off',\n"
        "        'description': 'Turn off a device',\n"
        "        'parameters': {'type': 'object', 'properties': {\n"
        "            'device': {'type': 'string', 'description': 'Device name'},\n"
        "        }, 'required': ['device']},\n"
        "    }},\n"
        "    {'type': 'function', 'function': {\n"
        "        'name': 'set_temperature',\n"
        "        'description': 'Set thermostat target temperature and HVAC mode',\n"
        "        'parameters': {'type': 'object', 'properties': {\n"
        "            'temperature': {'type': 'number', 'description': 'Target temp in celsius'},\n"
        "            'mode': {'type': 'string', 'enum': ['heat', 'cool', 'auto'], 'description': 'HVAC mode'},\n"
        "        }, 'required': ['temperature', 'mode']},\n"
        "    }},\n"
        "    {'type': 'function', 'function': {\n"
        "        'name': 'set_brightness',\n"
        "        'description': 'Set light brightness level 0-100',\n"
        "        'parameters': {'type': 'object', 'properties': {\n"
        "            'device': {'type': 'string', 'description': 'Light device name'},\n"
        "            'level': {'type': 'integer', 'description': 'Brightness 0-100'},\n"
        "        }, 'required': ['device', 'level']},\n"
        "    }},\n"
        "    {'type': 'function', 'function': {\n"
        "        'name': 'send_alert',\n"
        "        'description': 'Send alert to monitoring dashboard',\n"
        "        'parameters': {'type': 'object', 'properties': {\n"
        "            'level': {'type': 'string', 'enum': ['info', 'warning', 'critical', 'emergency']},\n"
        "            'message': {'type': 'string', 'description': 'Alert message'},\n"
        "            'zone': {'type': 'string', 'description': 'Zone or location'},\n"
        "        }, 'required': ['level', 'message', 'zone']},\n"
        "    }},\n"
        "    {'type': 'function', 'function': {\n"
        "        'name': 'lock_door',\n"
        "        'description': 'Lock a door',\n"
        "        'parameters': {'type': 'object', 'properties': {\n"
        "            'door': {'type': 'string', 'description': 'Door name'},\n"
        "        }, 'required': ['door']},\n"
        "    }},\n"
        "    {'type': 'function', 'function': {\n"
        "        'name': 'set_timer',\n"
        "        'description': 'Set a countdown timer',\n"
        "        'parameters': {'type': 'object', 'properties': {\n"
        "            'duration': {'type': 'string', 'description': 'Timer duration, e.g. 10 minutes'},\n"
        "            'label': {'type': 'string', 'description': 'What the timer is for'},\n"
        "        }, 'required': ['duration', 'label']},\n"
        "    }},\n"
        "    {'type': 'function', 'function': {\n"
        "        'name': 'start_pump',\n"
        "        'description': 'Start a pump at specified flow rate',\n"
        "        'parameters': {'type': 'object', 'properties': {\n"
        "            'pump_id': {'type': 'string'},\n"
        "            'flow_rate': {'type': 'string', 'enum': ['low', 'medium', 'high', 'max']},\n"
        "        }, 'required': ['pump_id', 'flow_rate']},\n"
        "    }},\n"
        "]\n"
        "\n"
        "SYSTEM = 'You are a helpful assistant. Use the provided tools when appropriate.'\n"
        "\n"
        "def test(prompt, expect_tool=None, expect_param=None, desc=''):\n"
        "    \"\"\"Run a test case and report results.\"\"\"\n"
        "    r = engine.chat_completion(\n"
        "        [{'role': 'system', 'content': SYSTEM},\n"
        "         {'role': 'user', 'content': prompt}],\n"
        "        tools=TOOLS\n"
        "    )\n"
        "    ch = r['choices'][0]\n"
        "    msg = ch['message']\n"
        "    tc = msg.get('tool_calls', [None])[0] if msg.get('tool_calls') else None\n"
        "    tool_name = tc['function']['name'] if tc else None\n"
        "    tool_args = json.loads(tc['function']['arguments']) if tc else {}\n"
        "\n"
        "    # Check expectations\n"
        "    status = ''\n"
        "    if expect_tool is not None:\n"
        "        if expect_tool is False:\n"
        "            status = '✓ PASS' if tool_name is None else f'✗ FAIL (got {tool_name}, expected no tool)'\n"
        "        else:\n"
        "            if tool_name == expect_tool:\n"
        "                status = '✓ PASS'\n"
        "                if expect_param:\n"
        "                    for k, v in expect_param.items():\n"
        "                        if str(tool_args.get(k, '')).lower() != str(v).lower():\n"
        "                            status = f'✗ PARAM ({k}: got {tool_args.get(k)}, expected {v})'\n"
        "            else:\n"
        "                status = f'✗ FAIL (got {tool_name}, expected {expect_tool})'\n"
        "\n"
        "    print(f'{status:20s} | {desc}')\n"
        "    print(f'  Prompt:  {prompt}')\n"
        "    if tc:\n"
        "        print(f'  Tool:    {tool_name}({json.dumps(tool_args)})')\n"
        "    if msg.get('content'):\n"
        "        print(f'  Reply:   {msg[\"content\"][:120]}')\n"
        "    if ch.get('_thinking'):\n"
        "        print(f'  Think:   {ch[\"_thinking\"][:120]}')\n"
        "    print()\n"
        "    return tool_name, tool_args"
    ))

    # Test cases: direct commands
    cells.append(md("### Direct commands — explicit tool invocation"))
    cells.append(code(
        "test('Turn on the kitchen lights',\n"
        "     expect_tool='turn_on', expect_param={'device': 'kitchen lights'},\n"
        "     desc='Direct: turn on specific device')\n"
        "\n"
        "test('Turn off the bedroom fan',\n"
        "     expect_tool='turn_off', expect_param={'device': 'bedroom fan'},\n"
        "     desc='Direct: turn off specific device')\n"
        "\n"
        "test('Lock the front door',\n"
        "     expect_tool='lock_door', expect_param={'door': 'front door'},\n"
        "     desc='Direct: lock specific door')\n"
        "\n"
        "test('Set the temperature to 22 degrees, cooling mode',\n"
        "     expect_tool='set_temperature', expect_param={'temperature': 22, 'mode': 'cool'},\n"
        "     desc='Direct: thermostat with exact values')\n"
        "\n"
        "test('Dim the living room lights to 30%',\n"
        "     expect_tool='set_brightness', expect_param={'level': 30},\n"
        "     desc='Direct: brightness with exact level')\n"
        "\n"
        "test('Set a timer for 15 minutes for the pasta',\n"
        "     expect_tool='set_timer', expect_param={'duration': '15 minutes'},\n"
        "     desc='Direct: timer with duration and label')\n"
        "\n"
        "test('Start pump_3 at high flow',\n"
        "     expect_tool='start_pump', expect_param={'pump_id': 'pump_3', 'flow_rate': 'high'},\n"
        "     desc='Direct: industrial pump control')"
    ))

    # Test cases: implicit intent
    cells.append(md("### Implicit intent — model must infer the right tool"))
    cells.append(code(
        "test(\"It's freezing in here\",\n"
        "     expect_tool='set_temperature',\n"
        "     desc='Implicit: cold complaint → heating')\n"
        "\n"
        "test(\"It's way too hot\",\n"
        "     expect_tool='set_temperature',\n"
        "     desc='Implicit: hot complaint → cooling')\n"
        "\n"
        "test(\"I can't see anything in the office\",\n"
        "     expect_tool='set_brightness',\n"
        "     desc='Implicit: dark complaint → increase lights')\n"
        "\n"
        "test('The lights are blinding me',\n"
        "     expect_tool='set_brightness',\n"
        "     desc='Implicit: bright complaint → dim lights')\n"
        "\n"
        "test(\"I'm leaving, secure the house\",\n"
        "     expect_tool='lock_door',\n"
        "     desc='Implicit: leaving → lock doors')"
    ))

    # Test cases: tool selection / discrimination
    cells.append(md("### Tool selection — model picks the right tool from the pool"))
    cells.append(code(
        "test('Send a critical alert about flooding in zone B',\n"
        "     expect_tool='send_alert', expect_param={'level': 'critical'},\n"
        "     desc='Selection: alert with severity level')\n"
        "\n"
        "test('Temperature in the server room is 45°C',\n"
        "     expect_tool='send_alert',\n"
        "     desc='Selection: sensor reading → alert, not thermostat')\n"
        "\n"
        "test('Start pump_7 at low flow, not high',\n"
        "     expect_tool='start_pump', expect_param={'flow_rate': 'low'},\n"
        "     desc='Selection: explicit param override')"
    ))

    # Test cases: no tool needed
    cells.append(md("### No tool needed — model should answer without calling tools"))
    cells.append(code(
        "test('What is a good room temperature?',\n"
        "     expect_tool=False,\n"
        "     desc='No tool: knowledge question')\n"
        "\n"
        "test('Thanks!',\n"
        "     expect_tool=False,\n"
        "     desc='No tool: gratitude')\n"
        "\n"
        "test('How does a thermostat work?',\n"
        "     expect_tool=False,\n"
        "     desc='No tool: explanation request')"
    ))

    # Test cases: precision
    cells.append(md("### Precision — exact values must be preserved"))
    cells.append(code(
        "test('Set the bedroom lights to exactly 42%',\n"
        "     expect_tool='set_brightness', expect_param={'level': 42},\n"
        "     desc='Precision: exact brightness 42%')\n"
        "\n"
        "test('Set temperature to 19.5, heating only',\n"
        "     expect_tool='set_temperature', expect_param={'temperature': 19.5, 'mode': 'heat'},\n"
        "     desc='Precision: decimal temp + specific mode')\n"
        "\n"
        "test('Start pump_11 at medium',\n"
        "     expect_tool='start_pump', expect_param={'pump_id': 'pump_11', 'flow_rate': 'medium'},\n"
        "     desc='Precision: exact pump ID + flow rate')"
    ))

    # Summary
    cells.append(md("### Test Summary"))
    cells.append(code(
        "# Run all tests again and tally results\n"
        "import re\n"
        "\n"
        "ALL_TESTS = [\n"
        "    # (prompt, expect_tool, expect_param, desc)\n"
        "    ('Turn on the kitchen lights', 'turn_on', {'device': 'kitchen lights'}, 'Direct: turn on'),\n"
        "    ('Turn off the bedroom fan', 'turn_off', {'device': 'bedroom fan'}, 'Direct: turn off'),\n"
        "    ('Lock the front door', 'lock_door', {'door': 'front door'}, 'Direct: lock'),\n"
        "    ('Set the temperature to 22 degrees, cooling mode', 'set_temperature', {'temperature': 22, 'mode': 'cool'}, 'Direct: thermostat'),\n"
        "    ('Dim the living room lights to 30%', 'set_brightness', {'level': 30}, 'Direct: brightness'),\n"
        "    ('Set a timer for 15 minutes for the pasta', 'set_timer', {'duration': '15 minutes'}, 'Direct: timer'),\n"
        "    ('Start pump_3 at high flow', 'start_pump', {'pump_id': 'pump_3', 'flow_rate': 'high'}, 'Direct: pump'),\n"
        "    (\"It's freezing in here\", 'set_temperature', None, 'Implicit: cold'),\n"
        "    (\"It's way too hot\", 'set_temperature', None, 'Implicit: hot'),\n"
        "    (\"I can't see anything in the office\", 'set_brightness', None, 'Implicit: dark'),\n"
        "    ('The lights are blinding me', 'set_brightness', None, 'Implicit: bright'),\n"
        "    (\"I'm leaving, secure the house\", 'lock_door', None, 'Implicit: secure'),\n"
        "    ('Send a critical alert about flooding in zone B', 'send_alert', {'level': 'critical'}, 'Select: alert'),\n"
        "    ('Start pump_7 at low flow, not high', 'start_pump', {'flow_rate': 'low'}, 'Select: param override'),\n"
        "    ('What is a good room temperature?', False, None, 'No tool: knowledge'),\n"
        "    ('Thanks!', False, None, 'No tool: gratitude'),\n"
        "    ('How does a thermostat work?', False, None, 'No tool: explain'),\n"
        "    ('Set the bedroom lights to exactly 42%', 'set_brightness', {'level': 42}, 'Precision: 42%'),\n"
        "    ('Set temperature to 19.5, heating only', 'set_temperature', {'temperature': 19.5, 'mode': 'heat'}, 'Precision: 19.5°C'),\n"
        "    ('Start pump_11 at medium', 'start_pump', {'pump_id': 'pump_11', 'flow_rate': 'medium'}, 'Precision: pump_11'),\n"
        "]\n"
        "\n"
        "passed = 0\n"
        "total = len(ALL_TESTS)\n"
        "for prompt, expect_tool, expect_param, desc in ALL_TESTS:\n"
        "    r = engine.chat_completion(\n"
        "        [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt}],\n"
        "        tools=TOOLS\n"
        "    )\n"
        "    msg = r['choices'][0]['message']\n"
        "    tc = msg.get('tool_calls', [None])[0] if msg.get('tool_calls') else None\n"
        "    name = tc['function']['name'] if tc else None\n"
        "    args = json.loads(tc['function']['arguments']) if tc else {}\n"
        "\n"
        "    ok = True\n"
        "    if expect_tool is False:\n"
        "        ok = name is None\n"
        "    elif expect_tool:\n"
        "        ok = name == expect_tool\n"
        "        if ok and expect_param:\n"
        "            for k, v in expect_param.items():\n"
        "                if str(args.get(k, '')).lower() != str(v).lower():\n"
        "                    ok = False\n"
        "    if ok:\n"
        "        passed += 1\n"
        "\n"
        "print(f'\\n{\"=\"*50}')\n"
        "print(f'Tool Calling Score: {passed}/{total} ({passed/total*100:.0f}%)')\n"
        "print(f'{\"=\"*50}')\n"
        "print(f'Trained for 10,000 steps. Adjust max_steps in config.py to train more or less.')"
    ))

    # ── Benchmark ──────────────────────────────────────────────────
    cells.append(md("## 7. Benchmark Early Exit"))
    cells.append(code("engine.benchmark()"))

    # ── Export + Compile C runtime ────────────────────────────────
    cells.append(md(
        "## 8. Export C Runtime\n"
        "\n"
        "Export model to `.goby` binary and compile native C runtime.\n"
        "**50-200x faster** than Python on RPi."
    ))

    cells.append(code(
        "!python export_goby.py --checkpoint checkpoints/best_model.pt \\\n"
        "                       --tokenizer data/tokenizer.json \\\n"
        "                       --output goby.bin\n"
        "\n"
        "import os\n"
        "print(f'Model binary: {os.path.getsize(\"goby.bin\")/1e6:.1f} MB')"
    ))

    cells.append(code(
        "!make clean && make\n"
        "!ls -lh goby"
    ))

    cells.append(code("!./goby goby.bin -b"))

    # ── Interactive C session ──────────────────────────────────────
    cells.append(md(
        "## 9. Interactive C Runtime\n"
        "\n"
        "Test tool-calling prompts through the C binary."
    ))

    cells.append(code(
        "import subprocess, shlex\n"
        "\n"
        "def goby(prompt, n=128, t=0.7):\n"
        "    r = subprocess.run(f'./goby goby.bin -p {shlex.quote(prompt)} -n {n} -t {t}',\n"
        "                       shell=True, capture_output=True, text=True)\n"
        "    print(f'You> {prompt}')\n"
        "    print(f'Goby> {r.stdout}')\n"
        "    if r.stderr: print(r.stderr.strip())\n"
        "    print()\n"
        "\n"
        "goby('Turn on the kitchen lights')\n"
        "goby('Set temperature to 22, cooling')\n"
        "goby('Lock all the doors')\n"
        "goby('Start pump_5 at max flow')\n"
        "goby('Send a critical alert about gas leak in zone A')"
    ))

    cells.append(code(
        "#@title Try your own prompt { run: \"auto\" }\n"
        "custom_prompt = 'Dim the office lights to 40 percent' #@param {type:\"string\"}\n"
        "max_tokens = 128 #@param {type:\"slider\", min:16, max:512, step:16}\n"
        "temperature = 0.7 #@param {type:\"slider\", min:0.1, max:1.5, step:0.1}\n"
        "\n"
        "goby(custom_prompt, n=max_tokens, t=temperature)"
    ))

    cells.append(code(
        "# Speed comparison: Python vs C on the same tool-calling prompt\n"
        "import time\n"
        "\n"
        "prompt = 'Turn on the bedroom lights'\n"
        "\n"
        "t0 = time.time()\n"
        "r = engine.chat_completion(\n"
        "    [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt}],\n"
        "    tools=TOOLS, max_tokens=64\n"
        ")\n"
        "py_ms = (time.time() - t0) * 1000\n"
        "\n"
        "t0 = time.time()\n"
        "subprocess.run(f'./goby goby.bin -p {shlex.quote(prompt)} -n 64 -t 0.7',\n"
        "               shell=True, capture_output=True, text=True)\n"
        "c_ms = (time.time() - t0) * 1000\n"
        "\n"
        "print(f'Python: {py_ms:.0f}ms')\n"
        "print(f'C:      {c_ms:.0f}ms')\n"
        "print(f'Speedup: {py_ms/c_ms:.1f}x')"
    ))

    # ── Download ───────────────────────────────────────────────────
    cells.append(md("## 10. Download"))

    cells.append(code(
        "import os\n"
        "!cd /content && tar czf goby_llm.tar.gz \\\n"
        "    goby/goby.bin \\\n"
        "    goby/goby.c \\\n"
        "    goby/Makefile \\\n"
        "    goby/goby \\\n"
        "    goby/checkpoints/best_model.pt \\\n"
        "    goby/checkpoints/config.json \\\n"
        "    goby/data/tokenizer.json \\\n"
        "    goby/model.py \\\n"
        "    goby/config.py \\\n"
        "    goby/inference.py \\\n"
        "    goby/rpi_runner.py \\\n"
        "    goby/export_goby.py\n"
        "\n"
        "sz = os.path.getsize('/content/goby_llm.tar.gz') / 1e6\n"
        "print(f'Package: /content/goby_llm.tar.gz ({sz:.1f} MB)')\n"
        "\n"
        "try:\n"
        "    from google.colab import files\n"
        "    files.download('/content/goby_llm.tar.gz')\n"
        "except ImportError:\n"
        "    print('Not in Colab — download manually.')"
    ))

    cells.append(md(
        "## 11. Deploy on Edge\n"
        "\n"
        "**Three runtime options** (fastest to slowest):\n"
        "\n"
        "### Option 1: C Runtime (fastest — recommended for RPi)\n"
        "```bash\n"
        "tar xzf goby_llm.tar.gz && cd goby\n"
        "\n"
        "# Re-compile for RPi (if downloaded Colab x86 binary):\n"
        "make clean && make\n"
        "\n"
        "# Run:\n"
        "./goby goby.bin -p 'Turn on the lights'    # single prompt\n"
        "./goby goby.bin -i                          # interactive\n"
        "./goby goby.bin -b                          # benchmark\n"
        "```\n"
        "Zero dependencies. ~51KB binary. mmap model loading. KV cache + early exit.\n"
        "\n"
        "### Option 2: Python KV-cached runner (needs torch + tokenizers)\n"
        "```bash\n"
        "pip install torch tokenizers\n"
        "python3 rpi_runner.py --serve --port 8000   # OpenAI-compatible server\n"
        "```\n"
        "\n"
        "### Option 3: Python naive (slowest, for debugging)\n"
        "```bash\n"
        "python3 inference.py --interactive\n"
        "```"
    ))

    return {
        "nbformat": 4, "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": [], "gpuType": "T4", "name": "GobyLLM Training"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": cells,
    }


if __name__ == "__main__":
    nb = build()
    out = os.path.join(PROJECT_ROOT, "goby_colab.ipynb")
    with open(out, "w") as f:
        json.dump(nb, f, indent=1)
    n = len(nb["cells"])
    sz = os.path.getsize(out) / 1024
    print(f"Generated {out}: {n} cells, {sz:.1f} KB")
