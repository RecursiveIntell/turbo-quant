# turbo-quant

`turbo-quant` is an experimental Rust crate for **derived vector-compression sidecars** inspired by TurboQuant, PolarQuant, and Quantized Johnson-Lindenstrauss (QJL) style sketches.

It is designed for systems that keep canonical vectors elsewhere, then use compact sidecars for candidate generation, memory accounting, compression experiments, and exact-rerank workflows. It is not a canonical vector store, not a replacement for exact vectors, and approximate scores are not ground truth.

## What this crate is

- A deterministic vector sidecar codec for embedding/search experiments.
- A PolarQuant-style compressor with optional QJL residual sketches.
- A compact sidecar index that returns approximate candidates plus an explicit search receipt.
- A KV-cache shadow-mode experiment surface for measuring compressed key/value behavior.
- A source-compatible update over the original `0.1.x` API surface.

## What this crate is not

- It is **not** a canonical vector store.
- It is **not** a replacement for exact vectors in correctness-sensitive retrieval.
- It is **not** a reversible compression library.
- It does **not** guarantee quality for every corpus, model, embedding distribution, or KV-cache workload.
- It should not be promoted into a production retrieval path without local benchmark gates and exact fallback.

The safe integration pattern is:

```text
canonical vectors / raw KV state
        +
derived turbo-quant sidecars
        -> approximate candidate generation
        -> exact rerank / exact fallback
        -> measured promotion decision
```

## Installation

```toml
[dependencies]
turbo-quant = "0.2"
```

Minimum supported Rust version: **1.75**.

## Quick start: encode and score one vector

```rust
use turbo_quant::TurboQuantizer;

fn main() -> turbo_quant::Result<()> {
    let dim = 64;
    let quantizer = TurboQuantizer::new(dim, 8, 16, 42)?;

    let database_vector = vec![0.1_f32; dim];
    let query_vector = vec![0.1_f32; dim];

    let code = quantizer.encode(&database_vector)?;
    let score = quantizer.inner_product_estimate(&code, &query_vector)?;

    println!("approximate score: {score}");
    println!("encoded bytes: {}", code.encoded_bytes());
    Ok(())
}
```

## Sidecar candidate search

`TurboSidecarIndex` is intentionally a sidecar index. It returns approximate candidates and a receipt that declares exact rerank is required.

```rust
use turbo_quant::{SearchOptions, TurboQuantizer, TurboSidecarIndex};

fn main() -> turbo_quant::Result<()> {
    let dim = 64;
    let quantizer = TurboQuantizer::new(dim, 8, 16, 42)?;
    let mut index = TurboSidecarIndex::new(quantizer);

    index.add("doc-a", &vec![0.10; dim], Some("source:doc-a".into()))?;
    index.add("doc-b", &vec![0.20; dim], Some("source:doc-b".into()))?;

    let query = vec![0.12; dim];
    let (candidates, receipt) = index.search(
        &query,
        SearchOptions {
            top_k: 1,
            oversample: 4,
        },
    )?;

    assert!(receipt.approximate_only);
    assert!(receipt.exact_rerank_required);
    println!("top approximate candidate: {:?}", candidates.first());
    Ok(())
}
```

After candidate generation, rerank candidates against caller-owned exact vectors or a trusted exact scorer.

## KV-cache shadow mode

KV-cache compression is exposed as an experiment surface. Keep exact shadows while measuring quality before promotion.

```rust
use turbo_quant::{KvCacheCompressor, KvQuantPolicy, KvRuntimeConfig};

fn main() -> turbo_quant::Result<()> {
    let dim = 64;
    let mut cache = KvCacheCompressor::new_runtime(KvRuntimeConfig {
        head_dim: dim,
        key_policy: KvQuantPolicy::quantized(8, 16),
        value_policy: KvQuantPolicy::Exact,
        seed: 42,
        keep_exact_shadow: true,
    })?;

    cache.compress_token(&vec![0.1; dim], &vec![0.2; dim])?;

    let query = vec![0.15; dim];
    let approximate_scores = cache.attention_scores(&query)?;
    let shadow_scores = cache.shadow_scores(&query)?;

    println!("approximate scores: {approximate_scores:?}");
    println!("shadow comparison: {shadow_scores:?}");
    Ok(())
}
```

## API compatibility

`0.2.x` preserves the original public compatibility surface from `0.1.x` while adding new sidecar, wire, receipt, and runtime-policy APIs.

Compatibility-preserved examples include:

- `PolarCode { dim, bits, radii, angle_indices }`
- `QjlSketch { dim, projections, signs }`
- `TurboCode { polar_code, residual_sketch }`
- `KvCacheConfig { head_dim, bits, projections, seed }`
- `CompressedToken { compressed_key, compressed_value }`
- legacy constructors such as `PolarQuantizer::new`, `QjlQuantizer::new`, `TurboQuantizer::new`, and `KvCacheCompressor::new`

The new APIs are additive and should be treated as the preferred integration surface for measured sidecar workflows.

## Release honesty

Compressed codes are derived artifacts. Any quality-sensitive use should preserve:

1. the source vector or exact KV state,
2. the codec profile,
3. benchmark receipts for the target workload,
4. exact rerank or exact fallback, and
5. clear degradation behavior when approximation is insufficient.

Do not treat approximate scores as ground truth.

## Testing before release

The release gate for this crate is intentionally strict:

```bash
cargo fmt --all -- --check
cargo check --all-targets --all-features --locked
cargo test --all-targets --all-features --locked
cargo test --doc --all-features --locked
cargo clippy --all-targets --all-features --locked -- -D warnings
cargo doc --all-features --no-deps --locked
cargo semver-checks --baseline-version 0.1.0 --manifest-path Cargo.toml
cargo package --list --locked
cargo package --locked
cargo publish --dry-run --locked
```

This repository includes release helper scripts under `scripts/` that run these gates, validate this README, validate crates.io package scope, and write local release receipts.

## Feature and design notes

- Rotation/profile selection is deterministic from explicit parameters.
- Packed/wire representations are acceleration artifacts, not truth-bearing storage.
- `SearchReceiptV1` makes approximate-only candidate generation explicit.
- `CompressionReceiptV1` records byte accounting and warnings for derived sidecars.
- KV-cache support is shadow-mode-first and should remain benchmark-gated.

## License

MIT
