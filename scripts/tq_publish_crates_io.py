#!/usr/bin/env python3
"""Publish turbo-quant to crates.io after the full release gate passes."""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="0.2.0")
    parser.add_argument("--execute", action="store_true", help="Actually run cargo publish after dry-run gate passes.")
    args = parser.parse_args()
    root = Path.cwd()
    gate = subprocess.run(["python3", "scripts/tq_release_gate.py", "--version", args.version], cwd=root)
    if gate.returncode != 0:
        print("publish blocked: release gate failed", file=sys.stderr)
        return gate.returncode
    receipt_path = root / "docs" / "release-evidence" / args.version / "release_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("recommendation") != "publish":
        print(f"publish blocked: receipt recommendation is {receipt.get('recommendation')}", file=sys.stderr)
        return 1
    if not args.execute:
        print("dry-run gate passed. Re-run with --execute to publish.")
        return 0
    expected = f"publish-turbo-quant-{args.version}"
    if os.environ.get("TQ_RELEASE_I_UNDERSTAND") != expected:
        print(f"publish blocked: set TQ_RELEASE_I_UNDERSTAND={expected}", file=sys.stderr)
        return 2
    print("Running cargo publish --locked ...")
    proc = subprocess.run(["cargo", "publish", "--locked"], cwd=root)
    if proc.returncode != 0:
        print("cargo publish failed", file=sys.stderr)
        return proc.returncode
    print("cargo publish completed. Verify crates.io and docs.rs manually.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
