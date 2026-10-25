# turbo-quant

<!-- last-verified: 2026-10-24 -->

This marker records the local README review date (CDT), not a product or release validation date.

## What this crate is

Experimental Rust codecs for derived vector-compression sidecars: PolarQuant, TurboQuant, QJL sketches, packed wire formats, candidate search, exact-rerank receipts, and KV-cache shadow experiments.

![Exact vectors remain canonical while turbo-quant builds a compressed sidecar, generates approximate candidates, and requires exact reranking before promotion.](https://raw.githubusercontent.com/RecursiveIntell/turbo-quant/main/docs/architecture.svg)

`turbo-quant` creates derived representations for experiments involving candidate generation, storage accounting, or approximate scoring. For a fixed quantizer created with `TurboQuantizer::new(dim, bits, projections, seed)` and a fixed input vector, repeated encoding is deterministic. Other constructors also require preserving their mode and rotation configuration. Approximate scores are not ground truth. Keep an exact fallback or rerank path and measure the workload locally.

## What this crate is not

This crate is not a canonical vector store. It is not a replacement for exact vectors in correctness-sensitive retrieval. Its experimental codecs do not guarantee quality or performance for every dimension, model, corpus, filter, or KV-cache workload, and do not establish paper parity or suitability for a particular deployment. The raw vector remains the correctness reference whenever an exact result is required.

## What is included

- **`PolarQuantizer`** — seeded polar-coordinate quantization.
- **`TurboQuantizer`** — polar quantization plus a QJL residual sketch for approximate inner-product estimates.
- **`QjlQuantizer`** — seeded sign-projection sketches.
- **Packed representations** — `PackedPolarCode`, `PackedQjlSketch`, and `PackedTurboCode` with validation and wire headers.
- **Sidecar indexing** — `TurboSidecarIndex`, `SearchOptions`, and `SearchReceiptV1` for bounded candidate generation.
- **Exact-rerank metadata** — receipt and byte-accounting types make the approximate stage explicit.
- **KV shadow mode** — `KvCacheCompressor`, `KvRuntimeConfig`, `KvQuantPolicy`, and shadow-score types for measurement, not an unconditional production replacement.
- **C hot paths** — the published 0.2.3 package includes build-time C kernels under `c-kernels/`; the Rust implementations remain under `src/archive/` for reference.

## Scope and limits

This is an experimental codec substrate. Quality depends on dimension, bits, seed, projection count, rotation, vector distribution, filters, and the caller's candidate/rerank policy. This crate provides no universal quality or performance guarantee and does not establish paper parity or suitability for a particular model or corpus.

Use the safe shape:

```text
canonical vectors
    + derived turbo-quant sidecar
    -> approximate candidate generation
    -> exact scoring / exact rerank
    -> workload-specific receipt and promotion decision
```

The raw vector remains the correctness reference whenever the application needs an exact result.

## Installation

```toml
[dependencies]
turbo-quant = "0.2"
```

The dependency snippet uses Cargo requirement `0.2`, which matches versions from the 0.2 release line according to Cargo's version-requirement rules. This selector is not a separate API-compatibility promise; the repository manifest declares package version 0.2.3.

The crate targets Rust 1.75 and uses the 2021 edition. Building this source requires a compatible C compiler: `build.rs` uses the `cc` crate to compile the C kernels under `c-kernels/`.

## Quick start

This uses the stable public quantizer API and small vectors so it can be copied into a test or example:

```rust
use turbo_quant::{PolarQuantizer, TurboQuantizer};

fn main() -> Result<(), turbo_quant::TurboQuantError> {
    let dim = 4;
    let vector = vec![0.1_f32, 0.2, 0.3, 0.4];
    let query = vec![0.2_f32, 0.1, 0.4, 0.3];

    let turbo = TurboQuantizer::new(dim, 8, 2, 42)?;
    let code = turbo.encode(&vector)?;
    let approximate_score = turbo.inner_product_estimate(&code, &query)?;
    println!("approximate score: {approximate_score}");

    let polar = PolarQuantizer::new(dim, 8, 42)?;
    let polar_code = polar.encode(&vector)?;
    let _polar_score = polar.inner_product_estimate(&polar_code, &query)?;
    Ok(())
}
```

## Sidecar candidate search

`TurboSidecarIndex` stores compressed codes alongside caller-owned IDs and returns candidates ranked by approximate inner-product estimates. Use those IDs to fetch authoritative vectors, then rerank with the caller's exact scorer before treating the results as exact:

```rust
use turbo_quant::{SearchOptions, TurboQuantizer, TurboSidecarIndex};

fn main() -> Result<(), turbo_quant::TurboQuantError> {
    let quantizer = TurboQuantizer::new(4, 8, 2, 42)?;
    let mut sidecar = TurboSidecarIndex::new(quantizer);
    sidecar.add("doc-a", &[0.1, 0.2, 0.3, 0.4], Some("doc-a-v1".into()))?;
    sidecar.add("doc-b", &[0.4, 0.3, 0.2, 0.1], Some("doc-b-v1".into()))?;

    let query = [0.2, 0.1, 0.4, 0.3];
    let (candidates, receipt) = sidecar.search(
        &query,
        SearchOptions { top_k: 1, oversample: 4 },
    )?;
    assert!(receipt.approximate_only);
    assert!(receipt.exact_rerank_required);
    println!("candidate IDs: {:?}", candidates.iter().map(|c| &c.id).collect::<Vec<_>>());
    Ok(())
}
```

The index does not own or return the original vectors. Resolve candidate IDs against the caller's exact store and apply its exact scoring and promotion policy.

## KV-cache shadow mode

`KvCacheCompressor::new_runtime` accepts separate key and value policies. This example quantizes keys while retaining exact values and exact shadows so the approximate key scores can be compared with exact scores:

```rust
use turbo_quant::{KvCacheCompressor, KvQuantPolicy, KvRuntimeConfig};

fn main() -> Result<(), turbo_quant::TurboQuantError> {
    let mut cache = KvCacheCompressor::new_runtime(KvRuntimeConfig {
        head_dim: 4,
        key_policy: KvQuantPolicy::quantized(8, 2),
        value_policy: KvQuantPolicy::Exact,
        seed: 42,
        keep_exact_shadow: true,
    })?;
    cache.compress_token(&[0.1, 0.2, 0.3, 0.4], &[0.4, 0.3, 0.2, 0.1])?;

    let comparisons = cache.shadow_attention_scores(&[0.2, 0.1, 0.4, 0.3])?;
    println!("shadow score comparisons: {comparisons:?}");
    Ok(())
}
```

This is an experiment surface, not a deployed-attention validation; measure quality and memory behavior on the target workload before considering integration.

## API map

| Type or module | Role |
|---|---|
| `TurboQuantizer` | Two-stage compressed code and approximate inner-product estimate. |
| `PolarQuantizer` | Single-stage polar code and estimate. |
| `QjlQuantizer` | Sign-projection sketch and estimate. |
| `TurboSidecarIndex` | Derived sidecar candidate index. |
| `SearchOptions` / `SearchReceiptV1` | Candidate-generation controls and evidence metadata. |
| `Packed*` types | Validated packed representations. |
| `TurboCodeWireV1` | Versioned wire representation with a fixed magic value. |
| `CodecProfileV1` / `CompressionReceiptV1` | Configuration and byte-accounting metadata. |
| `KvCacheCompressor` | KV shadow-mode compression and score measurement. |
| `BenchmarkReceiptV1` | Typed benchmark comparison data; it does not certify a deployment. |

See the rustdoc for method-level signatures and error behavior.

## API compatibility

The repository's `examples/compat_0_1_smoke.rs` is a compile-time smoke check for the listed legacy types (`PolarCode`, `QjlSketch`, `TurboCode`, `KvCacheConfig`, and `CompressedToken`) and the APIs used by that example. It is not a guarantee for every downstream integration; run it when upgrading:

```bash
cargo run --example compat_0_1_smoke
```

## Configuration guidance

Choose parameters from a local gate, not this README. In general:

- Use more bits or projections when score error is too high.
- Keep the seed and profile metadata with the sidecar so results are reproducible.
- Reject mismatched dimensions, bits, projections, seeds, wire versions, and non-finite inputs.
- Oversample candidates before exact rerank; the appropriate multiplier is workload-specific.
- Keep a raw-vector fallback whenever an approximate candidate omission is unacceptable.

## Wire and integrity behavior

The packed and wire modules validate headers, dimensions, bit widths, projection counts, rotation identity, payload lengths, padding, and finite values. Round-trip tests cover valid encodings and malformed artifacts. Treat decoded data as untrusted input and handle `TurboQuantError` rather than assuming a wire payload is valid.

## Release honesty

This section does not qualify a release. The repository declares version 0.2.3 in `Cargo.toml`. Treat the crate as an experimental codec substrate: a successful build or benchmark receipt does not establish paper parity, deployment quality, or a universal accuracy, memory, or speed result. Before adopting it, verify the specific release artifact and compiler requirements, then measure candidate recall, exact-rerank behavior, and memory accounting on your own workload. Benchmark gates are workload-specific.

## Testing before release

Run the repository's Rust checks from the crate root:

```bash
cargo fmt --check
cargo check --all-targets
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
```

These commands test the checked-out source and toolchain; results do not automatically transfer to another platform or a published package. Do not treat an experimental benchmark as a deployment qualification.

## Repository layout

- `src/` — public codecs, indexing, receipts, wire types, and validation.
- `c-kernels/` — build-time C implementations included in 0.2.3.
- `src/archive/` — preserved Rust implementations for comparison and review.
- `tests/` — API, determinism, malformed-artifact, wire, and retrieval-shape tests.
- `examples/` — compatibility, KV shadow, profile, and benchmark examples.
- `benches/` — Criterion benchmark target; benchmark results are workload and machine dependent.

## Relationship to other systems

`turbo-quant` is a reusable sidecar primitive. Integrations with a memory or retrieval system must define their own admission, exact-fallback, and promotion policy. If you use it with [`semantic-memory`](https://github.com/RecursiveIntell/semantic-memory) or another store, treat the store's exact vectors and receipts as canonical and validate the integration at the application boundary.

## License

Dual-licensed under MIT or Apache-2.0; see [`LICENSE-MIT`](https://github.com/RecursiveIntell/turbo-quant/blob/main/LICENSE-MIT) and [`LICENSE-APACHE`](https://github.com/RecursiveIntell/turbo-quant/blob/main/LICENSE-APACHE).

- Source: <https://github.com/RecursiveIntell/turbo-quant>
- API documentation: <https://docs.rs/turbo-quant>
- Changelog: [`CHANGELOG.md`](https://github.com/RecursiveIntell/turbo-quant/blob/main/CHANGELOG.md)
