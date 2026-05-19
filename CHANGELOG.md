# Changelog

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
