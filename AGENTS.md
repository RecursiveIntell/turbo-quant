# AGENTS.md P26 patch for turbo-quant

Append this to repo-root `AGENTS.md` for the P26 pass, or merge it into the existing repo guidance.

## P26 release-hardening rules

This repository is `turbo-quant`, a codec/compression crate. It is not semantic-memory, not a database, not a canonical vector store, and not a full RAG/retrieval system.

### Source-of-truth ownership

- `turbo-quant` owns codec profiles, quantizers, packed payloads, wire formats, approximate scoring, sidecar candidate generation, compression/search/KV receipts, and benchmark harnesses.
- `semantic-memory` owns mature vector indexing/retrieval semantics, raw-vector authority, exact rerank truth, document/source/chunk identity, and memory/retrieval product behavior.
- The P26 harness may read and compile against `~/Coding/Libraries/semantic-memory`, but core `src/` must not depend on or copy semantic-memory internals.

### API compatibility law

The public API from the crates.io `0.1.0` release must keep compiling unless a breaking change is intentionally versioned and documented. For this pass, assume no old public API break is allowed.

Do not mutate all-public legacy structs to add/remove/rename fields. Preserve:

```rust
PolarCode { dim, bits, radii, angle_indices }
QjlSketch { dim, projections, signs }
TurboCode { polar_code, residual_sketch }
KvCacheConfig { head_dim, bits, projections, seed }
CompressedToken { compressed_key, compressed_value }
```

Add new capabilities through additive new types, for example:

```rust
PackedPolarCode
PackedQjlSketch
PackedTurboCode
TurboSidecarCode
TurboSidecarIndex
KvRuntimeConfig
KvShadowToken
SearchReceiptV1
SemanticMemoryProofReceiptV1
RadiusCodecProfileV1
```

### Release claim law

Forbidden in README, docs, rustdoc, crate description, and release notes unless explicitly scoped to external paper claims or local receipt evidence:

- zero accuracy loss
- zero overhead
- production KV cache runtime
- drop-in replacement
- better than semantic-memory
- proven deployment quality
- no dataset-specific calibration as a crate guarantee

Allowed safe framing:

- experimental codec substrate
- derived sidecar
- approximate scoring
- exact fallback/rerank required
- workload-specific benchmark receipts required
- semantic-memory reference harness validates retrieval drift locally

### Required final proof

No publish recommendation unless these exist and pass:

- `examples/compat_0_1_smoke.rs`
- `scripts/assert_p26_invariants.py`
- `tools/semantic_memory_harness/`
- `docs/codex-runs/P26/SEMANTIC_MEMORY_PROOF_RECEIPT.json`
- `docs/codex-runs/P26/VALIDATION_RECEIPT.json`
- `docs/codex-runs/P26/AUDITOR_HANDOFF.md`
- `cargo package`
- `cargo publish --dry-run`
