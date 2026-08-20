# turbo-quant

<!-- last-verified: 2026-08-20 -->

Experimental Rust codecs for derived vector-compression sidecars: PolarQuant, TurboQuant, QJL sketches, packed wire formats, candidate search, exact-rerank receipts, and KV-cache shadow experiments.

![Exact vectors remain canonical while turbo-quant builds a compressed sidecar, generates approximate candidates, and requires exact reranking before promotion.](https://raw.githubusercontent.com/RecursiveIntell/turbo-quant/main/docs/architecture.svg)

`turbo-quant` does not replace the caller's exact vectors. It creates deterministic, derived representations that can reduce candidate-generation cost or storage for an experiment. Approximate scores are not ground truth: keep an exact fallback or rerank path and measure the workload locally.

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
turbo-quant = "0.2.3"
```

The crate targets Rust 1.75 and uses the 2021 edition. A C compiler is required for the build-time kernels in the published 0.2.3 package; GCC and Clang-style toolchains are the intended environments.

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

The repository also carries `examples/compat_0_1_smoke.rs`, which exercises the compatibility surface, packed types, and KV cache helper. Run it with:

```bash
cargo run --example compat_0_1_smoke
```

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

## Configuration guidance

Choose parameters from a local gate, not this README. In general:

- Use more bits or projections when score error is too high.
- Keep the seed and profile metadata with the sidecar so results are reproducible.
- Reject mismatched dimensions, bits, projections, seeds, wire versions, and non-finite inputs.
- Oversample candidates before exact rerank; the appropriate multiplier is workload-specific.
- Keep a raw-vector fallback whenever an approximate candidate omission is unacceptable.

## Wire and integrity behavior

The packed and wire modules validate headers, dimensions, bit widths, projection counts, rotation identity, payload lengths, padding, and finite values. Round-trip tests cover valid encodings and malformed artifacts. Treat decoded data as untrusted input and handle `TurboQuantError` rather than assuming a wire payload is valid.

## Verification

The release-source checkout was verified on 2026-08-20 with:

```bash
cargo fmt --check
cargo check --all-targets
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
```

Those commands passed for the synchronized 0.2.3 source. The test suite covers deterministic encoding, malformed inputs, serialization, wire validation, packed indexes, approximate scoring, KV policy, compatibility, and candidate-search behavior. Re-run the commands locally for evidence from your compiler and target platform.

## Repository layout

- `src/` — public codecs, indexing, receipts, wire types, and validation.
- `c-kernels/` — build-time C implementations included in 0.2.3.
- `src/archive/` — preserved Rust implementations for comparison and review.
- `tests/` — API, determinism, malformed-artifact, wire, and retrieval-shape tests.
- `examples/` — compatibility, KV shadow, profile, and benchmark examples.
- `benches/` — Criterion benchmark target; benchmark results are workload and machine dependent.

## Relationship to other systems

`turbo-quant` is a reusable sidecar primitive. Integrations with a memory or retrieval system must define their own admission, exact-fallback, and promotion policy. If you use it with [`semantic-memory`](https://github.com/RecursiveIntell/semantic-memory) or another store, treat the store's exact vectors and receipts as canonical and validate the integration at the application boundary.

## License and links

Dual-licensed under MIT or Apache-2.0; see [`LICENSE-MIT`](https://github.com/RecursiveIntell/turbo-quant/blob/main/LICENSE-MIT) and [`LICENSE-APACHE`](https://github.com/RecursiveIntell/turbo-quant/blob/main/LICENSE-APACHE).

- Source: <https://github.com/RecursiveIntell/turbo-quant>
- API documentation: <https://docs.rs/turbo-quant>
- Changelog: [`CHANGELOG.md`](https://github.com/RecursiveIntell/turbo-quant/blob/main/CHANGELOG.md)
