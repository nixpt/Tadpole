"""JSON repair and schema enforcement for GobyLLM tool calls."""

import json
import re
from difflib import SequenceMatcher


# ── JSON repair ─────────────────────────────────────────────────────────


def repair_json(s: str) -> str:
    """Attempt to fix broken JSON string into valid JSON."""

    if not s or not s.strip():
        return s

    s = s.strip()

    # Remove any leading/trailing text that isn't part of the JSON
    # Find the first { and work from there
    start = s.find("{")
    if start < 0:
        return s
    s = s[start:]

    # Single quotes → double quotes (but not inside already-double-quoted strings)
    # Simple approach: replace all single quotes with double quotes,
    # then fix cases where we broke things
    s = _fix_quotes(s)

    # Remove trailing commas before } or ]
    s = re.sub(r",\s*([}\]])", r"\1", s)

    # Fix unquoted keys: {name: "foo"} → {"name": "foo"}
    s = re.sub(r'(?<=[{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r' "\1":', s)

    # Close unclosed braces/brackets
    s = _close_brackets(s)

    # Try to parse — if it works, return cleaned version
    try:
        obj = json.loads(s)
        return json.dumps(obj)
    except json.JSONDecodeError:
        pass

    # More aggressive: try to extract just the tool call pattern
    m = re.search(r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{.*)', s, re.DOTALL)
    if m:
        name = m.group(1)
        args_str = _close_brackets(m.group(2))
        try:
            args = json.loads(args_str)
            return json.dumps({"name": name, "arguments": args})
        except json.JSONDecodeError:
            # Try to parse args more aggressively
            args = _extract_kv_pairs(args_str)
            if args:
                return json.dumps({"name": name, "arguments": args})

    return s


def _fix_quotes(s: str) -> str:
    """Replace single quotes with double quotes, handling escapes."""
    result = []
    in_double = False
    in_single = False
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            result.append(c)
            result.append(s[i + 1])
            i += 2
            continue
        if c == '"' and not in_single:
            in_double = not in_double
            result.append(c)
        elif c == "'" and not in_double:
            in_single = not in_single
            result.append('"')  # replace single with double
        else:
            result.append(c)
        i += 1
    return "".join(result)


def _close_brackets(s: str) -> str:
    """Close any unclosed { or [ brackets."""
    stack = []
    in_string = False
    escape = False
    for c in s:
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            stack.append("}")
        elif c == "[":
            stack.append("]")
        elif c in "}]":
            if stack and stack[-1] == c:
                stack.pop()
    # Close any remaining open brackets
    while stack:
        s += stack.pop()
    return s


def _extract_kv_pairs(s: str) -> dict:
    """Last-resort extraction of key-value pairs from messy text."""
    pairs = {}
    # Match "key": "value" or "key": number
    for m in re.finditer(r'"([^"]+)"\s*:\s*("([^"]*)"|([\d.]+)|(true|false|null))', s):
        key = m.group(1)
        if m.group(3) is not None:
            pairs[key] = m.group(3)
        elif m.group(4) is not None:
            try:
                pairs[key] = float(m.group(4)) if "." in m.group(4) else int(m.group(4))
            except ValueError:
                pairs[key] = m.group(4)
        elif m.group(5) is not None:
            pairs[key] = {"true": True, "false": False, "null": None}[m.group(5)]
    return pairs


# ── Tool schema matching ────────────────────────────────────────────────


def fuzzy_match_tool(name: str, tool_names: list, threshold: float = 0.6) -> str:
    """Find closest matching tool name. Returns original if no good match."""
    if name in tool_names:
        return name

    best_match = None
    best_score = 0
    for tn in tool_names:
        score = SequenceMatcher(None, name.lower(), tn.lower()).ratio()
        if score > best_score:
            best_score = score
            best_match = tn

    return best_match if best_score >= threshold else name


def validate_and_fix(tc_dict: dict, tools: list) -> dict:
    """Validate a tool call against tool schemas and fix issues."""

    if not tools or not tc_dict:
        return tc_dict

    name = tc_dict.get("name", "")
    args = tc_dict.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}

    # Find matching tool schema
    tool_names = [t["function"]["name"] for t in tools if "function" in t]
    matched_name = fuzzy_match_tool(name, tool_names)

    # Get schema for matched tool
    schema = None
    for t in tools:
        if t.get("function", {}).get("name") == matched_name:
            schema = t["function"].get("parameters", {})
            break

    if schema:
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        # Coerce types
        fixed_args = {}
        for key, prop in properties.items():
            if key in args:
                fixed_args[key] = _coerce_type(args[key], prop)
            elif key in required:
                # Fill default for missing required param
                fixed_args[key] = _default_for_type(prop)

        # Keep any extra args the model provided (might be useful)
        for key in args:
            if key not in fixed_args:
                fixed_args[key] = args[key]

        args = fixed_args

    return {"name": matched_name, "arguments": args}


def _coerce_type(value, prop: dict):
    """Coerce a value to match the expected JSON schema type."""
    expected = prop.get("type", "string")
    enum = prop.get("enum")

    if expected == "number":
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    if expected == "integer":
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return 0

    if expected == "boolean":
        if isinstance(value, bool):
            return value
        return str(value).lower() in ("true", "1", "yes", "on")

    # String type
    value = str(value)

    # Fuzzy match against enum values
    if enum and value not in enum:
        best = None
        best_score = 0
        for e in enum:
            score = SequenceMatcher(None, value.lower(), e.lower()).ratio()
            if score > best_score:
                best_score = score
                best = e
        if best and best_score >= 0.5:
            return best

    return value


def _default_for_type(prop: dict):
    """Generate a sensible default for a missing required parameter."""
    t = prop.get("type", "string")
    enum = prop.get("enum")
    if enum:
        return enum[0]
    if t == "number":
        return 0.0
    if t == "integer":
        return 0
    if t == "boolean":
        return False
    return ""


# ── Extraction ──────────────────────────────────────────────────────────


def extract_tool_calls(text: str, tools: list = None) -> list:
    """Extract, repair, and validate all tool calls from model output text."""

    import uuid
    results = []
    raw_tcs = []

    # Method 1: <tool_call> tags
    for m in re.finditer(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL):
        raw_tcs.append(m.group(1).strip())

    # Method 2: raw JSON with name+arguments
    if not raw_tcs:
        for m in re.finditer(
            r'\{[^{}]*"name"\s*:\s*"[^"]*"[^{}]*"arguments"\s*:\s*\{[^}]*\}[^}]*\}', text
        ):
            raw_tcs.append(m.group(0))

    # Method 3: even looser — just find {"name": "something" and grab what follows
    if not raw_tcs:
        for m in re.finditer(r'\{\s*"name"\s*:\s*"([^"]+)".*?(?:\}|$)', text, re.DOTALL):
            raw_tcs.append(m.group(0))

    for raw in raw_tcs:
        repaired = repair_json(raw)
        try:
            tc = json.loads(repaired)
        except json.JSONDecodeError:
            continue

        if "name" not in tc:
            continue

        # Validate against schema
        if tools:
            tc = validate_and_fix(tc, tools)

        results.append({
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {
                "name": tc["name"],
                "arguments": json.dumps(tc.get("arguments", {})),
            },
        })

    return results


def clean_response_text(text: str) -> str:
    """Remove tool call artifacts from response text."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL)
    text = re.sub(r'\{"name"\s*:.*?"arguments"\s*:\s*\{[^}]*\}\s*\}', "", text)
    text = text.replace("<|im_end|>", "").strip()
    return text


if __name__ == "__main__":
    # Test cases
    tests = [
        # Broken quotes
        ("{'name': 'turn_on', 'arguments': {'device': 'kitchen lights'}}", None),
        # Trailing comma
        ('{"name": "set_temperature", "arguments": {"temperature": 22, "mode": "cool",}}', None),
        # Unclosed brace (truncated)
        ('{"name": "lock_door", "arguments": {"door": "front door"', None),
        # Unquoted keys
        ('{name: "turn_off", arguments: {device: "fan"}}', None),
        # Typo in tool name
        ('{"name": "trun_on", "arguments": {"device": "lights"}}',
         [{"type": "function", "function": {"name": "turn_on", "parameters": {"type": "object", "properties": {"device": {"type": "string"}}, "required": ["device"]}}}]),
        # Wrong type
        ('{"name": "set_brightness", "arguments": {"device": "lamp", "level": "50"}}',
         [{"type": "function", "function": {"name": "set_brightness", "parameters": {"type": "object", "properties": {"device": {"type": "string"}, "level": {"type": "integer"}}, "required": ["device", "level"]}}}]),
    ]

    for raw, tools in tests:
        repaired = repair_json(raw)
        try:
            parsed = json.loads(repaired)
            if tools:
                parsed = validate_and_fix(parsed, tools)
            print(f"  IN:  {raw[:70]}")
            print(f"  OUT: {json.dumps(parsed)}")
            print()
        except json.JSONDecodeError as e:
            print(f"  IN:  {raw[:70]}")
            print(f"  ERR: {e}")
            print()
