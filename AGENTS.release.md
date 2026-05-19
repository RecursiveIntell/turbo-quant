# turbo-quant release agent rules

This repository contains an experimental vector-compression sidecar crate.

## Release law

- Do not publish without `cargo publish --dry-run --locked` passing.
- Do not publish from a dirty git tree.
- Do not publish while README contains Codex/run-bundle/stale-alpha/lossless/universal-quality claims.
- Do not publish if `cargo package --list --locked` includes Codex artifacts, local release evidence, context zips, prompts, target dirs, or local harness tooling.
- Do not treat compressed sidecars as canonical vectors.
- Do not claim production readiness or benchmark superiority without local receipts.

## Required release commands

Use:

```bash
python3 scripts/tq_release_gate.py --version 0.2.0
```

Then, only when the operator explicitly approves upload:

```bash
export TQ_RELEASE_I_UNDERSTAND=publish-turbo-quant-0.2.0
python3 scripts/tq_publish_crates_io.py --version 0.2.0 --execute
```

## Final report required

Changed files, commands run, test/lint/doc results, README gate, package gate, dry-run gate, publish status, receipt paths, and rollback/yank notes.
