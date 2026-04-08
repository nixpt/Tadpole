#!/usr/bin/env python3
"""Extract rama command help text for training data."""

import subprocess
import json
from pathlib import Path

RAMA_BIN = "/home/nixp/WORKSPACE/rama/bin/rama-core"

# Key commands to extract help for
KEY_COMMANDS = [
    # Volume management
    "volume-create",
    "volume-list",
    "volume-destroy",
    "volume-stats",
    "volume-backup",
    "volume-restore",
    
    # Zram
    "create",
    "reset",
    "list",
    "stats",
    "algorithms",
    
    # Workspace
    "workspace-create",
    "workspace-destroy",
    "workspace-list",
    "workspace-info",
    
    # Compression
    "compress",
    "decompress",
    
    # Dev environments
    "dev-create",
    "dev-destroy",
    "dev-list",
]

def extract_command_help(command):
    """Extract help text for a command."""
    try:
        result = subprocess.run(
            [RAMA_BIN, command, "--help"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.stdout if result.returncode == 0 else None
    except Exception as e:
        print(f"⚠️  Error extracting {command}: {e}")
        return None

def main():
    """Extract help text for all key commands."""
    commands = {}
    
    for cmd in KEY_COMMANDS:
        print(f"Extracting: {cmd:30s}... ", end="", flush=True)
        help_text = extract_command_help(cmd)
        
        if help_text:
            commands[cmd] = {
                "help": help_text,
                "lines": len(help_text.splitlines()),
                "chars": len(help_text),
            }
            print(f"✓ {len(help_text.splitlines()):3d} lines")
        else:
            print("✗ failed")
    
    # Save
    output = Path(__file__).parent / "rama_commands.json"
    with open(output, "w") as f:
        json.dump(commands, f, indent=2)
    
    print(f"\n💾 Saved {len(commands)} commands to: {output}")
    
    # Summary
    total_lines = sum(cmd["lines"] for cmd in commands.values())
    total_chars = sum(cmd["chars"] for cmd in commands.values())
    print(f"📊 Total: {total_lines} lines, {total_chars} chars")

if __name__ == "__main__":
    main()
