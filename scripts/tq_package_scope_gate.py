#!/usr/bin/env python3
"""Validate cargo package file list for turbo-quant crates.io release."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

FORBIDDEN_PARTS = [
    ".codex",
    "codex-runs",
    "release-evidence",
    "prompts",
    "governor",
    "templates",
    "tools/semantic_memory_harness",
    "scripts/",
    "target/",
    "p33boot",
    "P26",
    "P24",
    "OPERATOR_CHECKLIST",
    "EXECUTION_BUNDLE",
    "context",
]
FORBIDDEN_SUFFIXES = [
    ".zip", ".7z", ".tar", ".tar.gz", ".log", ".codex-archive.json", ".manifest.json", ".findings.json", ".excluded.json", ".report.md"
]
ALLOWED_ROOTS = {
    ".cargo_vcs_info.json",
    "Cargo.toml",
    "Cargo.toml.orig",
    "Cargo.lock",
    "README.md",
    "CHANGELOG.md",
    "RELEASE_NOTES.md",
    "LICENSE",
}
ALLOWED_PREFIXES = (
    "src/",
    "tests/",
    "examples/",
    "benches/",
    "docs/",
)


def normalize(line: str) -> str:
    line = line.strip()
    # cargo package --list can print paths with a leading package dir in some contexts.
    if line.startswith("turbo-quant-") and "/" in line:
        line = line.split("/", 1)[1]
    return line


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("package_list")
    args = parser.parse_args()
    path = Path(args.package_list)
    if not path.exists():
        print(f"package scope gate failed: {path} does not exist", file=sys.stderr)
        return 2
    files = [normalize(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    failures: list[str] = []
    if not files:
        failures.append("package list is empty")
    for rel in files:
        lower = rel.lower()
        if any(part.lower() in lower for part in FORBIDDEN_PARTS):
            failures.append(f"forbidden package path: {rel}")
        if any(lower.endswith(suffix) for suffix in FORBIDDEN_SUFFIXES):
            failures.append(f"forbidden package suffix: {rel}")
        allowed = rel in ALLOWED_ROOTS or rel.startswith(ALLOWED_PREFIXES)
        if not allowed:
            failures.append(f"unexpected package path outside allowlist: {rel}")
    if "README.md" not in files:
        failures.append("README.md missing from package")
    if "Cargo.toml" not in files:
        failures.append("Cargo.toml missing from package")
    if not any(f.startswith("src/") for f in files):
        failures.append("src/ files missing from package")

    if failures:
        print("package scope gate failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print(f"package scope gate passed ({len(files)} files)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
