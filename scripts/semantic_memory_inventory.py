#!/usr/bin/env python3
"""Create a read-only inventory of a local semantic-memory repo for P26.

This script does not assume semantic-memory's exact API. It surfaces likely files
and symbols for Codex to inspect before building the proof harness.
"""
from __future__ import annotations
import argparse
import json
import re
import subprocess
from pathlib import Path
from datetime import datetime, timezone

parser = argparse.ArgumentParser()
parser.add_argument("--semantic-memory-root", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()
root = Path(args.semantic_memory_root).expanduser().resolve()
out = Path(args.out).expanduser().resolve()

patterns = {
    "vector_terms": re.compile(r"vector|embedding|embed|cosine|dot|similarity|nearest|ann|hnsw|index", re.I),
    "search_terms": re.compile(r"search|query|retrieve|rank|rerank|top[_-]?k|candidate", re.I),
    "storage_terms": re.compile(r"sqlite|sled|db|store|repository|persistence", re.I),
    "trait_struct_fn": re.compile(r"\b(pub\s+)?(trait|struct|enum|fn)\s+([A-Za-z0-9_]+)"),
}

def git(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, cwd=root, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None

result = {
    "schema": "SemanticMemoryInventoryV1",
    "recorded_time": datetime.now(timezone.utc).isoformat(),
    "root": str(root),
    "exists": root.exists(),
    "git_head": None,
    "git_status_short": None,
    "cargo_manifests": [],
    "candidate_files": [],
    "candidate_symbols": [],
    "blockers": [],
}

if not root.exists():
    result["blockers"].append("semantic-memory root does not exist")
else:
    result["git_head"] = git(["git", "rev-parse", "HEAD"])
    result["git_status_short"] = git(["git", "status", "--short"])
    result["cargo_manifests"] = [str(p.relative_to(root)) for p in root.rglob("Cargo.toml")]
    for path in root.rglob("*.rs"):
        rel = str(path.relative_to(root))
        if any(part in {"target", ".git"} for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        score = 0
        hits = []
        for name, pat in patterns.items():
            if name == "trait_struct_fn":
                continue
            if pat.search(rel) or pat.search(text):
                score += 1
                hits.append(name)
        if score:
            result["candidate_files"].append({"path": rel, "hits": hits, "bytes": path.stat().st_size})
            for m in patterns["trait_struct_fn"].finditer(text):
                sym = m.group(3)
                if patterns["vector_terms"].search(sym) or patterns["search_terms"].search(sym):
                    result["candidate_symbols"].append({"file": rel, "kind": m.group(2), "symbol": sym})

out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps({"out": str(out), "candidate_files": len(result["candidate_files"]), "blockers": result["blockers"]}, indent=2))
