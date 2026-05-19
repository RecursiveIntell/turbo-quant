#!/usr/bin/env python3
"""P26 invariant checker for turbo-quant.

Run from repo root:
    python3 scripts/assert_p26_invariants.py .

This script intentionally checks source text because the pass goal includes public
field compatibility and package/ownership boundaries that can be lost even when
Rust builds.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
errors: list[str] = []


def read(path: str) -> str:
    p = ROOT / path
    if not p.exists():
        errors.append(f"missing required file: {path}")
        return ""
    return p.read_text(encoding="utf-8", errors="replace")


def rust_struct_body(src: str, name: str) -> str:
    m = re.search(rf"pub\s+struct\s+{re.escape(name)}\s*\{{", src)
    if not m:
        errors.append(f"missing public struct {name}")
        return ""
    i = m.end()
    depth = 1
    j = i
    while j < len(src) and depth:
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
        j += 1
    if depth != 0:
        errors.append(f"could not parse body for {name}")
        return ""
    return src[i : j - 1]


def require_field(body: str, name: str, typ_pat: str, struct: str) -> None:
    if not re.search(rf"pub\s+{re.escape(name)}\s*:\s*{typ_pat}\s*,", body):
        errors.append(
            f"{struct} must expose public field `{name}: {typ_pat}` for 0.1.0 compatibility"
        )


def forbid_field(body: str, name: str, struct: str) -> None:
    if re.search(rf"pub\s+{re.escape(name)}\s*:", body):
        errors.append(
            f"{struct} must not expose public field `{name}`; use additive packed/shadow type instead"
        )


polar = read("src/polar.rs")
qjl = read("src/qjl.rs")
turbo = read("src/turbo.rs")
kv = read("src/kv.rs")
lib = read("src/lib.rs")
cargo_toml = read("Cargo.toml")

polar_body = rust_struct_body(polar, "PolarCode")
require_field(polar_body, "dim", r"usize", "PolarCode")
require_field(polar_body, "bits", r"u8", "PolarCode")
require_field(polar_body, "radii", r"Vec\s*<\s*f32\s*>", "PolarCode")
require_field(polar_body, "angle_indices", r"Vec\s*<\s*u16\s*>", "PolarCode")
forbid_field(polar_body, "packed_angle_indices", "PolarCode")

qjl_body = rust_struct_body(qjl, "QjlSketch")
require_field(qjl_body, "dim", r"usize", "QjlSketch")
require_field(qjl_body, "projections", r"usize", "QjlSketch")
require_field(qjl_body, "signs", r"Vec\s*<\s*i8\s*>", "QjlSketch")
forbid_field(qjl_body, "packed_signs", "QjlSketch")
forbid_field(qjl_body, "norm", "QjlSketch")

turbo_body = rust_struct_body(turbo, "TurboCode")
require_field(turbo_body, "polar_code", r"PolarCode", "TurboCode")
require_field(turbo_body, "residual_sketch", r"QjlSketch", "TurboCode")
if "Option<QjlSketch>" in turbo_body or "Option < QjlSketch >" in turbo_body:
    errors.append(
        "TurboCode.residual_sketch must not be Option<QjlSketch>; use additive sidecar type for optional QJL"
    )

kv_config_body = rust_struct_body(kv, "KvCacheConfig")
require_field(kv_config_body, "head_dim", r"usize", "KvCacheConfig")
require_field(kv_config_body, "bits", r"u8", "KvCacheConfig")
require_field(kv_config_body, "projections", r"usize", "KvCacheConfig")
require_field(kv_config_body, "seed", r"u64", "KvCacheConfig")
for forbidden in ["key_policy", "value_policy", "keep_exact_shadow"]:
    forbid_field(kv_config_body, forbidden, "KvCacheConfig")

compressed_body = rust_struct_body(kv, "CompressedToken")
require_field(compressed_body, "compressed_key", r"TurboCode", "CompressedToken")
require_field(compressed_body, "compressed_value", r"TurboCode", "CompressedToken")
for forbidden in ["exact_key", "exact_value"]:
    forbid_field(compressed_body, forbidden, "CompressedToken")
if "Option<TurboCode>" in compressed_body or "Option < TurboCode >" in compressed_body:
    errors.append("CompressedToken legacy fields must not become Option<TurboCode>")

required_files = [
    "examples/compat_0_1_smoke.rs",
    "scripts/assert_p26_invariants.py",
    "src/packed.rs",
    "src/index.rs",
    "src/radius.rs",
    "tools/semantic_memory_harness/Cargo.toml",
    "tools/semantic_memory_harness/src/main.rs",
]
for path in required_files:
    if not (ROOT / path).exists():
        errors.append(f"missing P26 required file: {path}")

for module in ["packed", "index", "radius"]:
    if f"pub mod {module};" not in lib:
        errors.append(f"src/lib.rs must export `pub mod {module};`")

for path in (ROOT / "src").glob("**/*.rs"):
    txt = path.read_text(encoding="utf-8", errors="replace")
    if "semantic_memory" in txt or "semantic-memory" in txt:
        errors.append(f"core source must not reference semantic-memory: {path.relative_to(ROOT)}")

try:
    data = tomllib.loads(cargo_toml)
    for sec in ["dependencies", "dev-dependencies", "build-dependencies"]:
        deps = data.get(sec, {}) or {}
        for name in deps.keys():
            if name in {"semantic-memory", "semantic_memory"}:
                errors.append(
                    f"Cargo.toml must not depend on semantic-memory in [{sec}]; use tools/semantic_memory_harness instead"
                )
    pkg = data.get("package", {}) or {}
    if not pkg.get("include") and not pkg.get("exclude"):
        errors.append(
            "Cargo.toml must define package include/exclude to keep Codex/tools artifacts out of crates.io package"
        )
except Exception as exc:
    errors.append(f"failed to parse Cargo.toml: {exc}")

claim_paths = [
    ROOT / "README.md",
    ROOT / "src/lib.rs",
    ROOT / "RELEASE_NOTES.md",
    ROOT / "CHANGELOG.md",
]
claim_paths.extend((ROOT / "docs").glob("*.md") if (ROOT / "docs").exists() else [])
for p in claim_paths:
    if not p.exists():
        continue
    txt = p.read_text(encoding="utf-8", errors="replace").lower()
    for phrase in [
        "zero accuracy loss",
        "zero-overhead",
        "zero overhead",
        "production kv-cache runtime",
        "production kv cache runtime",
        "drop-in replacement",
        "proven deployment quality",
    ]:
        if phrase in txt:
            errors.append(f"forbidden unqualified claim `{phrase}` found in {p.relative_to(ROOT)}")

if errors:
    print("P26 invariant check FAILED:")
    for e in errors:
        print(f"- {e}")
    sys.exit(1)

print("P26 invariant check passed")
