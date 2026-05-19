#!/usr/bin/env python3
"""Validate `cargo package --list` output for P26."""
from __future__ import annotations
import sys
from pathlib import Path

if len(sys.argv) != 2:
    print("usage: check_cratesio_package_scope.py <cargo-package-list.txt>", file=sys.stderr)
    sys.exit(2)

p = Path(sys.argv[1])
lines = [line.strip() for line in p.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
forbidden_prefixes = [
    ".codex/",
    "prompts/",
    "docs/codex-runs/",
    "target/",
    "tools/semantic_memory_harness/",
]
forbidden_suffixes = [
    ".zip",
    ".tar",
    ".tar.gz",
    ".codex-archive.json",
    ".manifest.json",
    ".findings.json",
    ".excluded.json",
    ".report.md",
]
forbidden_exact = {"z.py"}
errors = []
for line in lines:
    norm = line.replace("\\", "/")
    if norm in forbidden_exact:
        errors.append(f"forbidden exact file in package: {norm}")
    for prefix in forbidden_prefixes:
        if norm.startswith(prefix):
            errors.append(f"forbidden prefix in package: {norm}")
    for suffix in forbidden_suffixes:
        if norm.endswith(suffix):
            errors.append(f"forbidden generated/archive file in package: {norm}")

if errors:
    print("Package scope check FAILED:")
    for e in errors:
        print(f"- {e}")
    sys.exit(1)

print(f"Package scope check passed for {len(lines)} files")
