# turbo-quant 0.2.0 crates.io release runbook

This runbook is the operator-facing path for testing and publishing `turbo-quant` `0.2.0`.

## Non-negotiable rule

Do not run `cargo publish` until:

- the README gate passes,
- the package scope gate passes,
- `cargo publish --dry-run --locked` passes,
- the release receipt reports `recommendation: publish`, and
- the git tree is clean except for deliberate release receipt artifacts that have been committed or intentionally excluded.

Cargo's publish command creates a `.crate` package, uploads it to the registry, and the server performs additional checks. `--dry-run` performs all checks without uploading. Use the dry run as the final gate before publish.

## One-time setup

```bash
cd ~/Coding/Libraries/turbo-quant
cargo install cargo-semver-checks --locked
cargo login
```

Do not put crates.io tokens into scripts, shell history, receipts, or prompts. Use `cargo login` or a short-lived `CARGO_REGISTRY_TOKEN` in your local shell.

## Install this release pack

From this bundle root:

```bash
./INSTALL.sh ~/Coding/Libraries/turbo-quant
```

The installer backs up replaced files under `.release-pack-backups/`.

## Full validation gate

```bash
cd ~/Coding/Libraries/turbo-quant
python3 scripts/tq_release_gate.py --version 0.2.0
```

This creates/updates:

```text
docs/release-evidence/v0.2.0/logs/
docs/release-evidence/v0.2.0/package-list.txt
docs/release-evidence/v0.2.0/release_receipt.json
docs/release-evidence/v0.2.0/release_receipt.md
```

## Publish gate

After the full validation gate passes, publish with an explicit two-part confirmation:

```bash
export TQ_RELEASE_I_UNDERSTAND=publish-turbo-quant-0.2.0
python3 scripts/tq_publish_crates_io.py --version 0.2.0 --execute
```

Without `--execute`, the publish script stops after the dry-run gate.

## Post-publish verification

```bash
cargo search turbo-quant
cargo info turbo-quant
```

Then check:

- crates.io page renders the README correctly,
- docs.rs build starts/succeeds,
- version is `0.2.0`,
- no stale `P26`, `Codex`, `alpha`, `release-evidence`, or local-path language appears in public package content.

## Rollback/yank

If the wrong package is published:

```bash
cargo yank --version 0.2.0 turbo-quant
```

Then create a postmortem and release `0.2.1` only after the full gate passes again.
