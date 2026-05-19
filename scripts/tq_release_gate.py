#!/usr/bin/env python3
"""Full turbo-quant 0.2.0 release validation gate.

Runs README, format, build, tests, clippy, docs, semver, package-scope, package, and
publish dry-run checks. Writes receipts under docs/release-evidence/<version>/.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd: list[str], *, cwd: Path, log_path: Path, allow_fail: bool = False) -> dict:
    started = time.time()
    proc = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    elapsed = time.time() - started
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout, encoding="utf-8")
    record = {
        "cmd": cmd,
        "cmd_display": " ".join(shlex.quote(c) for c in cmd),
        "exit_code": proc.returncode,
        "elapsed_seconds": round(elapsed, 3),
        "log_path": str(log_path),
        "log_sha256": sha256_file(log_path),
    }
    status = "PASS" if proc.returncode == 0 else "FAIL"
    print(f"[{status}] {record['cmd_display']} -> {proc.returncode} ({elapsed:.1f}s)")
    if proc.returncode != 0 and not allow_fail:
        raise RuntimeError(f"command failed: {record['cmd_display']} (see {log_path})")
    return record


def read_manifest(root: Path) -> dict:
    manifest = root / "Cargo.toml"
    if not manifest.exists():
        raise RuntimeError("Cargo.toml not found; run from turbo-quant crate root")
    return tomllib.loads(manifest.read_text(encoding="utf-8"))


def require_clean_git(root: Path, evidence: Path) -> dict:
    git = shutil.which("git")
    if not git:
        raise RuntimeError("git is required for release gating")
    status = subprocess.run([git, "status", "--short"], cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    head = subprocess.run([git, "rev-parse", "HEAD"], cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (evidence / "git_status_short.txt").write_text(status.stdout, encoding="utf-8")
    (evidence / "git_head.txt").write_text(head.stdout, encoding="utf-8")
    if status.returncode != 0 or head.returncode != 0:
        raise RuntimeError("git metadata unavailable")
    if status.stdout.strip():
        raise RuntimeError("git tree is dirty; commit/stash changes before release gate")
    return {"head": head.stdout.strip(), "status_short_sha256": sha256_file(evidence / "git_status_short.txt")}


def require_p26_artifacts(root: Path) -> dict:
    required = [
        "examples/compat_0_1_smoke.rs",
        "scripts/assert_p26_invariants.py",
        "tools/semantic_memory_harness",
        "docs/codex-runs/P26/SEMANTIC_MEMORY_PROOF_RECEIPT.json",
        "docs/codex-runs/P26/VALIDATION_RECEIPT.json",
        "docs/codex-runs/P26/AUDITOR_HANDOFF.md",
    ]
    artifacts: dict[str, dict[str, str | bool]] = {}
    missing: list[str] = []
    for rel in required:
        path = root / rel
        if not path.exists():
            missing.append(rel)
            continue
        record: dict[str, str | bool] = {"path": str(path)}
        if path.is_file():
            record["sha256"] = sha256_file(path)
        else:
            record["exists"] = True
        artifacts[rel] = record
    if missing:
        raise RuntimeError("missing P26 required artifacts: " + ", ".join(missing))
    return artifacts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="0.2.0")
    parser.add_argument("--skip-semver", action="store_true", help="Do not use for final publish; local debugging only.")
    parser.add_argument("--skip-harness", action="store_true", help="Skip optional semantic-memory harness.")
    args = parser.parse_args()

    root = Path.cwd()
    manifest = read_manifest(root)
    package = manifest.get("package", {})
    if package.get("name") != "turbo-quant":
        raise RuntimeError(f"expected package.name turbo-quant, got {package.get('name')!r}")
    if package.get("version") != args.version:
        raise RuntimeError(f"Cargo.toml version must be {args.version}, got {package.get('version')!r}")

    evidence = root / "docs" / "release-evidence" / args.version
    logs = evidence / "logs"
    evidence.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    receipt = {
        "schema": "TurboQuantCratesIoReleaseReceiptV1",
        "crate": "turbo-quant",
        "target_version": args.version,
        "recorded_time_unix_seconds": int(time.time()),
        "root": str(root),
        "recommendation": "do_not_publish",
        "commands": [],
        "artifacts": {},
        "blockers": [],
        "publish_command_executed": False,
    }

    exit_code = 0
    try:
        receipt["git"] = require_clean_git(root, evidence)
        for tool in ["cargo", "rustc"]:
            if not shutil.which(tool):
                raise RuntimeError(f"required tool missing: {tool}")
        if not args.skip_semver and not shutil.which("cargo-semver-checks"):
            raise RuntimeError("cargo-semver-checks missing; install with: cargo install cargo-semver-checks --locked")

        receipt["commands"].append(run(["python3", "scripts/tq_readme_gate.py", "README.md"], cwd=root, log_path=logs / "readme-gate.log"))
        receipt["commands"].append(run(["python3", "scripts/assert_p26_invariants.py", "."], cwd=root, log_path=logs / "p26-invariants.log"))
        receipt["artifacts"]["p26_required_artifacts"] = require_p26_artifacts(root)
        receipt["commands"].append(run(["cargo", "fmt", "--all", "--", "--check"], cwd=root, log_path=logs / "cargo-fmt.log"))
        receipt["commands"].append(run(["cargo", "check", "--all-targets", "--all-features", "--locked"], cwd=root, log_path=logs / "cargo-check.log"))
        receipt["commands"].append(run(["cargo", "test", "--all-targets", "--all-features", "--locked"], cwd=root, log_path=logs / "cargo-test.log"))
        receipt["commands"].append(run(["cargo", "test", "--doc", "--all-features", "--locked"], cwd=root, log_path=logs / "cargo-test-doc.log"))
        receipt["commands"].append(run(["cargo", "clippy", "--all-targets", "--all-features", "--locked", "--", "-D", "warnings"], cwd=root, log_path=logs / "cargo-clippy.log"))
        receipt["commands"].append(run(["cargo", "doc", "--all-features", "--no-deps", "--locked"], cwd=root, log_path=logs / "cargo-doc.log"))
        if not args.skip_semver:
            receipt["commands"].append(run(["cargo", "semver-checks", "--baseline-version", "0.1.0", "--manifest-path", "Cargo.toml"], cwd=root, log_path=logs / "cargo-semver-checks.log"))

        package_list = evidence / "package-list.txt"
        proc = subprocess.run(["cargo", "package", "--list", "--locked"], cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        package_list.write_text(proc.stdout, encoding="utf-8")
        rec = {
            "cmd": ["cargo", "package", "--list", "--locked"],
            "cmd_display": "cargo package --list --locked",
            "exit_code": proc.returncode,
            "log_path": str(package_list),
            "log_sha256": sha256_file(package_list),
        }
        receipt["commands"].append(rec)
        if proc.returncode != 0:
            raise RuntimeError(f"cargo package --list failed; see {package_list}")
        receipt["artifacts"]["package_list"] = {"path": str(package_list), "sha256": sha256_file(package_list)}
        receipt["commands"].append(run(["python3", "scripts/tq_package_scope_gate.py", str(package_list)], cwd=root, log_path=logs / "package-scope-gate.log"))
        receipt["commands"].append(run(["cargo", "package", "--locked"], cwd=root, log_path=logs / "cargo-package.log"))
        receipt["commands"].append(run(["cargo", "publish", "--dry-run", "--locked"], cwd=root, log_path=logs / "cargo-publish-dry-run.log"))

        harness = root / "tools" / "semantic_memory_harness" / "Cargo.toml"
        sibling = root.parent / "semantic-memory"
        if harness.exists() and sibling.exists() and not args.skip_harness:
            harness_out = evidence / "semantic_memory_harness_receipt.json"
            receipt["commands"].append(run([
                "cargo", "run", "--manifest-path", str(harness), "--release", "--",
                "--semantic-memory-root", str(sibling),
                "--out", str(harness_out),
            ], cwd=root, log_path=logs / "semantic-memory-harness.log"))
            if harness_out.exists():
                receipt["artifacts"]["semantic_memory_harness_receipt"] = {"path": str(harness_out), "sha256": sha256_file(harness_out)}

        receipt["recommendation"] = "publish"
    except Exception as exc:
        receipt["blockers"].append(str(exc))
        receipt["recommendation"] = "do_not_publish"
        exit_code = 1
    finally:
        receipt_path = evidence / "release_receipt.json"
        md_path = evidence / "release_receipt.md"
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
        md = [f"# turbo-quant {args.version} release receipt", "", f"Recommendation: `{receipt['recommendation']}`", ""]
        if receipt["blockers"]:
            md.append("## Blockers")
            md.extend(f"- {b}" for b in receipt["blockers"])
            md.append("")
        md.append("## Commands")
        for c in receipt["commands"]:
            md.append(f"- `{c['cmd_display']}` -> `{c['exit_code']}` ({c.get('log_path')})")
        md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
        print(f"release receipt: {receipt_path}")
        print(f"recommendation: {receipt['recommendation']}")
        if receipt["blockers"]:
            for blocker in receipt["blockers"]:
                print(f"blocker: {blocker}", file=sys.stderr)
    return exit_code

if __name__ == "__main__":
    raise SystemExit(main())
