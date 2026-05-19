#!/usr/bin/env python3
"""Validate turbo-quant README before crates.io publishing."""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

FORBIDDEN_PATTERNS = [
    r"zero\s+accuracy\s+loss",
    r"\blossless\b",
    r"\bperfect\b",
    r"guaranteed\s+quality",
    r"production[- ]ready",
    r"drop[- ]in\s+replacement\s+for\s+vectors",
    r"\ball\s+workloads\b",
    r"\bP26\b",
    r"\bP24\b",
    r"\bCodex\b",
    r"0\.2\.0-alpha",
    r"alpha\.1",
    r"release-evidence",
]

REQUIRED_HEADINGS = [
    "# turbo-quant",
    "## What this crate is",
    "## What this crate is not",
    "## Installation",
    "## Quick start",
    "## Sidecar candidate search",
    "## KV-cache shadow mode",
    "## API compatibility",
    "## Release honesty",
    "## Testing before release",
    "## License",
]

REQUIRED_PHRASES = [
    "derived vector-compression sidecars",
    "not a canonical vector store",
    "not a replacement for exact vectors",
    "exact rerank",
    "approximate scores are not ground truth",
    "benchmark gates",
    "TurboQuantizer::new",
    "TurboSidecarIndex",
    "KvCacheCompressor::new_runtime",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("readme", nargs="?", default="README.md")
    args = parser.parse_args()
    path = Path(args.readme)
    if not path.exists():
        print(f"README gate failed: {path} does not exist", file=sys.stderr)
        return 2
    text = path.read_text(encoding="utf-8")
    failures: list[str] = []

    for heading in REQUIRED_HEADINGS:
        if heading not in text:
            failures.append(f"missing heading: {heading}")

    lowered = text.lower()
    for phrase in REQUIRED_PHRASES:
        if phrase.lower() not in lowered:
            failures.append(f"missing required phrase: {phrase}")

    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            failures.append(f"forbidden README pattern present: {pattern}")

    if "```rust" not in text:
        failures.append("missing Rust code block")
    if "turbo-quant = \"0.2\"" not in text:
        failures.append("missing Cargo.toml dependency snippet for version 0.2")

    if failures:
        print("README gate failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("README gate passed")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
