# turbo-quant

`turbo-quant` is an experimental Rust crate for derived vector-compression
sidecars. It provides PolarQuant codes, TurboQuant polar-plus-QJL residual
codes, packed payload helpers, deterministic wire formats, approximate sidecar
candidate generation, KV shadow-mode helpers, and benchmark receipts.

Compressed codes are not canonical vector truth. Keep exact vectors or another
caller-owned authority for exact rerank, retrieval decisions, and audit evidence.

## Install

```toml
[dependencies]
turbo-quant = "0.2.0"
```

## Quick Start

```rust
use turbo_quant::{SearchOptions, TurboQuantizer, TurboSidecarIndex};

fn main() -> turbo_quant::Result<()> {
    let dim = 64;
    let quantizer = TurboQuantizer::new(dim, 8, 32, 42)?;

    let mut index = TurboSidecarIndex::new(quantizer);
    index.add("doc-a", &vec![0.1; dim], Some("sha256:caller-owned".into()))?;
    index.add("doc-b", &vec![0.2; dim], Some("sha256:caller-owned".into()))?;

    let (candidates, receipt) = index.search(
        &vec![0.12; dim],
        SearchOptions {
            top_k: 2,
            oversample: 4,
        },
    )?;

    assert!(receipt.exact_rerank_required);
    assert_eq!(candidates.len(), 2);
    Ok(())
}
```

## Compatibility

Version `0.2.0` preserves the `0.1.0` public struct literal shapes for:

- `PolarCode { dim, bits, radii, angle_indices }`
- `QjlSketch { dim, projections, signs }`
- `TurboCode { polar_code, residual_sketch }`
- `KvCacheConfig { head_dim, bits, projections, seed }`
- `CompressedToken { compressed_key, compressed_value }`

New storage and receipt capabilities are additive types such as
`PackedPolarCode`, `PackedQjlSketch`, `PackedTurboCode`, `TurboSidecarIndex`,
`KvRuntimeConfig`, `KvShadowToken`, `SearchReceiptV1`, and
`RadiusCodecProfileV1`.

## Release Evidence

The release hardening pass records local validation under
`docs/release-evidence/v0.2.0/`. Treat those files as reproducibility receipts
for this repository state, not as a guarantee for other datasets or deployments.

Useful checks:

```sh
cargo test --all-targets
python3 scripts/assert_p26_invariants.py
python3 scripts/scan_forbidden_claims.py
cargo package
cargo publish --dry-run
```

## Boundaries

`turbo-quant` owns codec profiles, quantizers, packed payloads, wire formats,
approximate scoring, sidecar candidate generation, compression/search/KV
receipts, and benchmark harnesses.

It does not own mature vector indexing semantics, raw-vector authority, exact
rerank truth, document identity, or retrieval product behavior. Those remain the
responsibility of the caller or an external retrieval system.
