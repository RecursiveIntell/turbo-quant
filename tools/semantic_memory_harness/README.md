# Semantic-memory retrieval harness

This directory contains a non-publishable Rust harness, plus preserved `.template` inputs. The executable implementation is [src/main.rs](src/main.rs), not a task waiting for an agent to replace the templates.

The harness compares a semantic-memory baseline, TurboQuant sidecar candidates, and exact reranking, then writes a `SemanticMemoryProofReceiptV1` containing source identity, corpus/profile details, byte accounting, recall/rank metrics, thresholds, and blockers.

## Dependency boundary

[Cargo.toml](Cargo.toml) is its own workspace and uses relative path dependencies for both TurboQuant and a sibling semantic-memory checkout. Keep those paths aligned with the source you intend to measure. A standalone semantic-memory mirror may itself require the surrounding Libraries workspace; dependency-path existence alone is not proof that the harness resolves.

The `--semantic-memory-root` option identifies source for receipt metadata; it does not rewrite Cargo dependencies. `--out` chooses the receipt destination. Inspect the source and supply a new output path before running so a prior receipt is not overwritten unintentionally.

## Scope

This harness is local evaluation infrastructure and is excluded from the published codec package. It does not own semantic-memory storage/retrieval semantics. A successful synthetic or fixture run supports only its recorded corpus, profile, thresholds, and source; it does not establish universal retrieval quality or production readiness.
