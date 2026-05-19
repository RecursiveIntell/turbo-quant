# Release Notes: 0.2.0

`turbo-quant` 0.2.0 is a compatibility-preserving hardening release for the
experimental codec substrate.

## Added

- Additive packed payload types for polar, QJL, and TurboQuant codes.
- Deterministic TurboCode wire encoding and validation.
- Codec profiles, compression receipts, benchmark receipts, and byte accounting.
- Sidecar candidate index APIs that mark exact rerank as caller responsibility.
- KV shadow-mode runtime configuration and score comparison helpers.
- Explicit QJL source-norm provenance APIs for transport-stable scoring when
  callers persist QJL norm evidence.
- Compatibility smoke example for the 0.1.0 public struct shapes.

## Compatibility

The legacy public fields remain available for `PolarCode`, `QjlSketch`,
`TurboCode`, `KvCacheConfig`, and `CompressedToken`. New capabilities are exposed
through additive types rather than mutating those legacy structs.

## Operational Notes

Compressed codes are derived sidecars. Exact vectors remain caller-owned for
exact rerank, audit evidence, and authoritative retrieval behavior. Benchmark
and semantic-memory harness receipts are workload-local evidence only.
