# Codex prompt: turbo-quant 0.2.0 crates.io release pass

You are executing a deterministic, auditable release pass for `turbo-quant`.

## Goal

Prepare, validate, and optionally publish `turbo-quant` `0.2.0` to crates.io. Do not optimize for speed. Do not publish unless every gate passes.

## Required startup actions

1. Confirm current directory is the `turbo-quant` crate root.
2. Record:
   - `pwd`
   - `git status --short`
   - `git rev-parse HEAD`
   - `cargo --version`
   - `rustc --version`
3. Inspect:
   - `Cargo.toml`
   - `README.md`
   - `CHANGELOG.md`
   - `RELEASE_NOTES.md`
   - `src/lib.rs`
   - `scripts/tq_release_gate.py`
   - `scripts/tq_readme_gate.py`
   - `scripts/tq_package_scope_gate.py`
4. Do not edit until you understand the current release surface.

## Required edits if needed

- Replace `README.md` with the supplied release README if the current README contains Codex/run-bundle/stale-alpha language.
- Ensure `Cargo.toml` version is `0.2.0` before publish.
- Ensure package metadata has description, license, repository, keywords, categories, and excludes local run artifacts.
- Remove public README claims that imply lossless or universal quality.
- Keep all claims framed around derived sidecars, exact rerank, and benchmark gates.

## Required validation

Run:

```bash
python3 scripts/tq_release_gate.py --version 0.2.0
```

Do not proceed if it fails.

## Publish command

Publishing is allowed only after the release gate recommends publish and the operator explicitly asks for upload.

```bash
export TQ_RELEASE_I_UNDERSTAND=publish-turbo-quant-0.2.0
python3 scripts/tq_publish_crates_io.py --version 0.2.0 --execute
```

## Final report

End with:

- changed files,
- commands run,
- tests passed/failed/skipped,
- README gate status,
- package scope gate status,
- `cargo publish --dry-run` status,
- whether `cargo publish` was run,
- release receipt path,
- remaining blockers,
- rollback/yank notes.

No receipts, no completion claim.
