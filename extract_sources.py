#!/usr/bin/env python3
"""Extract training data sources from WORKSPACE repos."""

import os
import json
from pathlib import Path

WORKSPACE = Path.home() / "WORKSPACE"

# Documentation sources by category
SOURCES = {
    "rama_core": {
        "priority": 1,
        "files": [
            "rama/README.md",
            "rama/CLAUDE.md",
            "rama/QUICKREF.md",
        ],
        "topics": ["volumes", "zram", "workspace", "compression", "config"],
    },
    "rama_docs": {
        "priority": 2,
        "files": [
            "rama/doc/rama.md",
            "rama/doc/CODEBASE_MAP.md",
            "rama/doc/IMPLEMENTATION.md",
        ],
        "topics": ["architecture", "implementation"],
    },
    "rama_planning": {
        "priority": 3,
        "files": [
            "rama/zpu_vcpu_ram_gpu_lanes_spec.md",
            "rama/planning/ZPU_IMPLEMENTATION_PLAN.md",
        ],
        "topics": ["zpu", "inference"],
    },
    "exosphere_core": {
        "priority": 2,
        "files": [
            "exosphere/CLAUDE.md",
        ],
        "topics": ["exosphere", "integration"],
    },
    "squadron_tools": {
        "priority": 3,
        "files": [
            "squadron/README.md",
            "squadron/doc/TOOLS.md",
        ],
        "topics": ["squadron", "agent-tools"],
    },
}

def extract_sources():
    """Extract and catalog documentation sources."""
    catalog = {}
    
    for category, spec in SOURCES.items():
        catalog[category] = {
            "priority": spec["priority"],
            "topics": spec["topics"],
            "files": [],
        }
        
        for rel_path in spec["files"]:
            full_path = WORKSPACE / rel_path
            if not full_path.exists():
                print(f"⚠️  Missing: {rel_path}")
                continue
            
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            lines = len(content.splitlines())
            chars = len(content)
            
            catalog[category]["files"].append({
                "path": str(rel_path),
                "full_path": str(full_path),
                "lines": lines,
                "chars": chars,
                "exists": True,
            })
            
            print(f"✓ {rel_path:50s} {lines:4d} lines, {chars:6d} chars")
    
    # Summary
    total_files = sum(len(cat["files"]) for cat in catalog.values())
    total_lines = sum(f["lines"] for cat in catalog.values() for f in cat["files"])
    total_chars = sum(f["chars"] for cat in catalog.values() for f in cat["files"])
    
    print(f"\n📊 Total: {total_files} files, {total_lines} lines, {total_chars} chars")
    
    # Save catalog
    output = Path(__file__).parent / "training_sources.json"
    with open(output, "w") as f:
        json.dump(catalog, f, indent=2)
    
    print(f"\n💾 Saved catalog to: {output}")
    
    return catalog

if __name__ == "__main__":
    extract_sources()
