# Changelog

## 0.2.3

- Added C kernels for FWHT (2.75× speedup), polar encode/decode, and
  QJL sketch/project/IP estimate, compiled at build time via `cc` crate
  with `-O3 -mavx2 -mfma`.
- Bitpack C kernel reverted to Rust — FFI call overhead negated the
  compiler advantage for branch-heavy bit manipulation.
- Original Rust implementations preserved in `src/archive/` with
  headers.
- Added `build.rs` and `cc` build dependency.
- `real_bench.rs` example added for real-embedding benchmark with
  semantic-memory harness.
- 123 tests pass.

## 0.2.0

- Preserved the `0.1.0` public struct literal shapes for legacy code.
- Added packed sidecar payload types without replacing legacy logical structs.
- Added deterministic wire encoding and strict decode validation for TurboCode.
- Added codec profiles, compression receipts, benchmark receipts, and sidecar
  search receipts.
- Added explicit QJL source-norm provenance APIs and removed hidden process-global
  norm dependence from legacy QJL scoring.
- Added KV shadow-mode runtime configuration and exact-shadow comparison helpers.
- Added semantic-memory reference harness support for local retrieval drift
  validation with exact rerank.
- Reworked public docs around experimental sidecar semantics and caller-owned
  exact-vector authority.
