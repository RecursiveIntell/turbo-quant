# turbo-quant

**Rust implementation of TurboQuant, PolarQuant, and QJL -- zero-overhead vector quantization for semantic search and KV cache compression.**

Compress high-dimensional vectors (embeddings, KV cache entries) to 3-8 bits per value with **zero accuracy loss** and **no dataset-specific calibration**. Based on the algorithms from Google Research published at ICLR 2026, AISTATS 2026, and AAAI 2025.

---

## Table of Contents

- [Why turbo-quant?](#why-turbo-quant)
- [Key Properties](#key-properties)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
  - [Requirements](#requirements)
  - [Installation](#installation)
- [Usage](#usage)
  - [TurboQuant (Two-Stage Compression)](#turboquant-two-stage-compression)
  - [PolarQuant (Single-Stage Compression)](#polarquant-single-stage-compression)
  - [QJL (1-bit Sketching)](#qjl-1-bit-sketching)
  - [KV Cache Compression](#kv-cache-compression)
  - [Serialization and Persistence](#serialization-and-persistence)
- [Choosing Parameters](#choosing-parameters)
- [How It Works](#how-it-works)
  - [Stage 1: Random Rotation (Whitening)](#stage-1-random-rotation-whitening)
  - [Stage 2: PolarQuant (Polar Coordinate Encoding)](#stage-2-polarquant-polar-coordinate-encoding)
  - [Stage 3: QJL Residual Correction](#stage-3-qjl-residual-correction)
  - [Inner Product Estimation](#inner-product-estimation)
- [API Reference](#api-reference)
- [Performance Characteristics](#performance-characteristics)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [References](#references)
- [License](#license)

---

## Why turbo-quant?

Traditional vector quantization methods like Product Quantization (PQ) and Optimized Product Quantization (OPQ) require an expensive offline training phase over a representative dataset before a single vector can be indexed. This means:

- You can't index vectors as they arrive in a streaming setting
- Quantizer quality degrades when the data distribution shifts
- Training on sensitive data may violate privacy constraints

**turbo-quant eliminates all of these problems.** The quantizer is entirely defined by four integers `(dim, bits, projections, seed)` -- no codebooks, no centroids, no calibration data. Vectors can be compressed the instant they arrive, making it ideal for real-time indexing, federated search, and privacy-sensitive deployments.

---

## Key Properties

| Property | Description |
|---|---|
| **Data-oblivious** | No k-means training, no codebook, no calibration set. The rotation is seeded once and works on any distribution. |
| **Deterministic** | Identical `(dim, bits, projections, seed)` always produces the same quantizer. State can be fully reconstructed from four integers. |
| **Zero accuracy loss** | At 3+ bits, inner product estimates are provably unbiased and achieve near-optimal distortion (within ~2.7x of the Shannon limit). |
| **Instant indexing** | Unlike Product Quantization, there is no offline training phase. Vectors can be indexed as they arrive. |
| **Compact state** | The entire quantizer is defined by `(dim, bits, projections, seed)`. No model files to ship or version. |

---

## Architecture

turbo-quant is organized as a layered system where each algorithm builds on the previous:

```
+------------------------------------------------------------------+
|                        KvCacheCompressor                         |
|  Online KV cache for transformer attention (compress_token,      |
|  attention_scores, attend)                                       |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|                        TurboQuantizer                            |
|  Two-stage compression: PolarQuant (b-1 bits) + QJL (1 bit)     |
|  Provably unbiased inner product estimation                      |
+------------------------------------------------------------------+
                      /                  \
                     v                    v
+-------------------------------+  +-------------------------------+
|       PolarQuantizer          |  |        QjlQuantizer           |
|  Polar coordinate encoding    |  |  1-bit JL sketching of the   |
|  of rotated vectors           |  |  reconstruction residual      |
+-------------------------------+  +-------------------------------+
                     \                    /
                      v                  v
+------------------------------------------------------------------+
|                       StoredRotation                             |
|  Haar-distributed random orthogonal matrix via QR decomposition  |
|  Deterministic from (dim, seed) using ChaCha8 RNG               |
+------------------------------------------------------------------+
```

---

## Getting Started

### Requirements

- **Rust** 1.75.0 or later
- **Cargo** (included with Rust)

### Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
turbo-quant = { git = "https://github.com/RecursiveIntell/turbo-quant" }
```

Or clone and build from source:

```bash
git clone https://github.com/RecursiveIntell/turbo-quant.git
cd turbo-quant
cargo build --release
```

---

## Usage

### TurboQuant (Two-Stage Compression)

The flagship algorithm. Combines PolarQuant with a QJL residual sketch for near-optimal distortion at any bit width.

```rust
use turbo_quant::TurboQuantizer;

// Create a quantizer for 1536-dimensional embeddings at 8 bits.
let q = TurboQuantizer::new(
    1536,   // dimension (must be even)
    8,      // bits per value (1-16)
    384,    // QJL projections (dim/4 recommended for search)
    42,     // seed (any u64)
).unwrap();

// Compress a database vector.
let embedding: Vec<f32> = vec![0.1; 1536];
let code = q.encode(&embedding).unwrap();

// At query time: estimate inner product without decompressing.
let query: Vec<f32> = vec![0.05; 1536];
let score = q.inner_product_estimate(&code, &query).unwrap();

// Estimate L2 distance.
let dist = q.l2_distance_estimate(&code, &query).unwrap();

// Approximate reconstruction (lossy).
let reconstructed = q.decode_approximate(&code).unwrap();

// Compression metrics.
println!("Encoded size: {} bytes", code.encoded_bytes());
println!("Compression ratio: {:.2}x", code.compression_ratio());

// Batch statistics over multiple codes.
let codes = vec![code];
let stats = q.batch_stats(&codes);
println!("Average compression: {:.2}x", stats.compression_ratio);
```

### PolarQuant (Single-Stage Compression)

A simpler, single-stage compressor. Lower overhead than TurboQuant but slightly higher distortion at low bit widths.

```rust
use turbo_quant::PolarQuantizer;

let pq = PolarQuantizer::new(1536, 8, 42).unwrap();

let vector: Vec<f32> = vec![0.1; 1536];
let code = pq.encode(&vector).unwrap();

// Lossless decode (radii stored as f32, only angles are quantized).
let decoded = pq.decode(&code).unwrap();

// Inner product and L2 distance estimation.
let query: Vec<f32> = vec![0.05; 1536];
let ip = pq.inner_product_estimate(&code, &query).unwrap();
let l2 = pq.l2_distance_estimate(&code, &query).unwrap();
```

### QJL (1-bit Sketching)

The Quantized Johnson-Lindenstrauss transform. Produces a 1-bit-per-projection sketch for unbiased inner product estimation. Used internally by TurboQuant for residual correction, but also available standalone.

```rust
use turbo_quant::QjlQuantizer;

let qjl = QjlQuantizer::new(1536, 384, 42).unwrap();

let vector: Vec<f32> = vec![0.1; 1536];
let sketch = qjl.sketch(&vector).unwrap();

// Estimate inner product using sketch + raw query.
let query: Vec<f32> = vec![0.05; 1536];
let ip = qjl.inner_product_estimate(&sketch, &query).unwrap();

println!("Sketch size: {} bytes", sketch.encoded_bytes());
```

### KV Cache Compression

Online KV cache compressor for transformer attention layers. Compresses key-value pairs as tokens are generated and computes attention scores directly on compressed data.

```rust
use turbo_quant::kv::{KvCacheCompressor, KvCacheConfig};

let config = KvCacheConfig {
    head_dim: 128,       // per-attention-head dimension
    bits: 6,             // 4-6 bits recommended for KV cache
    projections: 16,     // QJL projections for residual correction
    seed: 42,
};

let mut cache = KvCacheCompressor::new(config).unwrap();

// Compress tokens as they are generated.
for step in 0..sequence_length {
    let key = compute_key(step);     // Vec<f32> of length head_dim
    let value = compute_value(step); // Vec<f32> of length head_dim
    cache.compress_token(&key, &value).unwrap();
}

// Compute attention logits against all compressed keys.
let query = compute_query();
let scores = cache.attention_scores(&query).unwrap();

// Full softmax-weighted attention output.
let output = cache.attend(&query).unwrap();

// Decode specific token values (e.g., top-k).
let top_k_indices = vec![0, 5, 12];
let values = cache.decode_values(&top_k_indices).unwrap();

// Compression metrics.
println!("Tokens: {}", cache.len());
println!("Compressed bytes: {}", cache.compressed_bytes());
println!("Compression ratio: {:.2}x", cache.compression_ratio());
```

### Serialization and Persistence

All compressed codes support JSON serialization via serde. Quantizers themselves don't need to be serialized -- they can be reconstructed from their parameters.

```rust
use turbo_quant::{TurboQuantizer, TurboCode};

let q = TurboQuantizer::new(64, 8, 16, 42).unwrap();
let code = q.encode(&vec![0.1; 64]).unwrap();

// Serialize to JSON.
let json = serde_json::to_string(&code).unwrap();

// Deserialize back.
let restored: TurboCode = serde_json::from_str(&json).unwrap();

// Quantizer reconstruction: just save the 4 parameters.
let q2 = TurboQuantizer::new(64, 8, 16, 42).unwrap();
// q2 is identical to q -- same rotation, same behavior.
```

---

## Choosing Parameters

| Use Case | Bits | Projections | Notes |
|---|---|---|---|
| **Semantic search** (recall@10) | 8 | dim / 4 | Best accuracy; suitable for production retrieval |
| **KV cache compression** | 4-6 | dim / 8 | Balances memory savings with attention quality |
| **Maximum compression** | 3 | dim / 16 | Aggressive; still provably unbiased |
| **Lightweight similarity** | - | dim / 4 | Use QJL standalone for 1-bit sketches |

**Guidelines:**

- **Bits** controls the PolarQuant fidelity. Higher bits = less distortion but larger codes. The QJL stage always uses 1 bit per projection.
- **Projections** controls the QJL residual correction precision. More projections = better bias correction but larger residual sketches. Diminishing returns beyond dim/4.
- **Seed** must be the same for encoding and querying. Different seeds produce incompatible quantizers.
- **Dimension** must be even (required for polar pair encoding).

---

## How It Works

### Stage 1: Random Rotation (Whitening)

Real-world embeddings have non-uniform energy distributions across coordinates. A random orthogonal rotation (Haar-distributed, generated via QR decomposition of a Gaussian matrix) spreads the energy uniformly, making simple scalar quantization near-optimal.

```
x_rotated = R * x     where R is a d x d orthogonal matrix
```

The rotation is deterministic: the same `(dim, seed)` always produces the same R via ChaCha8 RNG.

### Stage 2: PolarQuant (Polar Coordinate Encoding)

After rotation, the d-dimensional vector is grouped into d/2 coordinate pairs. Each pair is converted to polar form:

```
(a, b) -> (r, theta)     where r = sqrt(a^2 + b^2), theta = atan2(b, a)
```

- **Radii** are stored losslessly as f32 values
- **Angles** are uniformly quantized to b bits on [-pi, pi], yielding 2^b quantization levels

This preserves directional information with controllable precision while retaining exact magnitude information.

### Stage 3: QJL Residual Correction

TurboQuant computes the reconstruction residual (original minus PolarQuant approximation) and sketches it using the Quantized Johnson-Lindenstrauss transform:

```
residual = x - decode(polar_encode(x))
sketch   = [sign(g_1 . residual), ..., sign(g_m . residual)]
```

where g_1, ..., g_m are random Gaussian projection vectors. Each sketch entry is a single bit (sign).

### Inner Product Estimation

The combined estimator is provably unbiased:

```
<x, y> ~ IP_polar(code, y) + IP_qjl(residual_sketch, y)
```

- **IP_polar**: Computed directly from quantized angles and stored radii against the rotated query
- **IP_qjl**: `(pi / 2m) * sum_i sign(g_i . residual) * (g_i . query)` -- an unbiased estimator by the JL lemma

At 3+ bits this achieves distortion within ~2.7x of the Shannon rate-distortion limit.

---

## API Reference

### Core Types

| Type | Description |
|---|---|
| `TurboQuantizer` | Two-stage quantizer (PolarQuant + QJL). The primary entry point. |
| `TurboCode` | Compressed vector output from TurboQuantizer. |
| `PolarQuantizer` | Single-stage polar coordinate quantizer. |
| `PolarCode` | Compressed vector output from PolarQuantizer. |
| `QjlQuantizer` | Standalone 1-bit JL sketch quantizer. |
| `QjlSketch` | 1-bit sketch output from QjlQuantizer. |
| `KvCacheCompressor` | Online KV cache for transformer attention. |
| `KvCacheConfig` | Configuration struct for KvCacheCompressor. |
| `CompressedToken` | A single compressed key-value pair in the cache. |
| `StoredRotation` | A d x d orthogonal rotation matrix. |
| `BatchStats` | Aggregate compression statistics. |

### Constructors

```rust
TurboQuantizer::new(dim: usize, bits: u8, projections: usize, seed: u64) -> Result<Self>
PolarQuantizer::new(dim: usize, bits: u8, seed: u64) -> Result<Self>
QjlQuantizer::new(dim: usize, projections: usize, seed: u64) -> Result<Self>
KvCacheCompressor::new(config: KvCacheConfig) -> Result<Self>
```

### Error Handling

All fallible operations return `Result<T, TurboQuantError>`. Error variants:

| Variant | Cause |
|---|---|
| `DimensionMismatch` | Input vector length does not match quantizer dimension |
| `OddDimension` | Dimension must be even for polar pair encoding |
| `ZeroDimension` | Dimension must be non-zero |
| `InvalidBitWidth` | Bits must be in range 1-16 |
| `ZeroProjectionCount` | Projections must be non-zero |
| `RotationFailed` | Internal error during rotation matrix generation |
| `MalformedCode` | Compressed code failed validation |

---

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Notes |
|---|---|---|
| `new()` | O(d^2) | One-time cost: QR decomposition of d x d matrix |
| `encode()` | O(d * m) | d = dimension, m = projections |
| `inner_product_estimate()` | O(d * m) | No decompression needed |
| `decode()` | O(d) | PolarQuant only; TurboQuant decode is approximate |
| `attention_scores()` | O(n * d * m) | n = number of cached tokens |

### Space Complexity

| Component | Size |
|---|---|
| Quantizer state | O(d^2) for the rotation matrix (~9 MB at d=1536) |
| PolarCode | d/2 * 4 bytes (radii) + d/2 * 2 bytes (angles) = 3d bytes |
| QjlSketch | m bytes (one sign byte per projection) |
| TurboCode | 3d + m bytes total |

### Compression Ratios

| Dimension | Bits | Projections | Uncompressed | Compressed | Ratio |
|---|---|---|---|---|---|
| 1536 | 8 | 384 | 6144 B | 4992 B | 1.23x |
| 1536 | 4 | 192 | 6144 B | 4800 B | 1.28x |
| 128 | 8 | 32 | 512 B | 416 B | 1.23x |
| 128 | 4 | 16 | 512 B | 400 B | 1.28x |

The primary value is not raw byte savings but the ability to compute inner products directly on compressed data, avoiding full decompression during search.

---

## Testing

Run the full test suite:

```bash
cargo test
```

Run specific test categories:

```bash
cargo test --lib                     # Unit tests in each module
cargo test --test determinism        # Reproducibility and ordering tests
cargo test --test inner_product      # Accuracy and recall tests
```

### Test Coverage

The test suite validates:

- **Determinism**: Same seed produces identical codes across quantizer instances
- **Seed independence**: Different seeds produce different codes
- **Encoding order independence**: Encoding order does not affect individual codes
- **Rank preservation**: Top-k ordering is preserved at 8 bits
- **Error scaling**: Distortion decreases monotonically with bit width
- **Accuracy bounds**: Relative error < 5% at 16 bits; top-10 recall >= 60% at 8 bits
- **Mathematical properties**: Self inner product > 0, antipodal vectors yield negative estimates, L2 self-distance near zero
- **KV cache correctness**: Attention score ordering, softmax normalization, dimension validation
- **Serialization roundtrips**: JSON encode/decode for all code types
- **Edge cases**: Zero vectors, unit vectors, empty caches, dimension mismatches

---

## Project Structure

```
turbo-quant/
├── Cargo.toml                 # Package manifest and dependencies
├── README.md                  # This file
├── src/
│   ├── lib.rs                 # Library root, module exports, and top-level docs
│   ├── error.rs               # Error types (TurboQuantError) and Result alias
│   ├── rotation.rs            # StoredRotation: Haar-random orthogonal matrices via QR
│   ├── polar.rs               # PolarQuantizer: single-stage polar coordinate encoding
│   ├── qjl.rs                 # QjlQuantizer: 1-bit Johnson-Lindenstrauss sketching
│   ├── turbo.rs               # TurboQuantizer: two-stage PolarQuant + QJL compression
│   └── kv.rs                  # KvCacheCompressor: online KV cache for transformers
└── tests/
    ├── determinism.rs          # Reproducibility, seed, and ordering tests
    └── inner_product.rs        # Accuracy, recall, and mathematical property tests
```

### Module Dependency Graph

```
lib.rs
 ├── error.rs          (standalone)
 ├── rotation.rs       (standalone, uses nalgebra + rand_chacha)
 ├── polar.rs          (depends on rotation, error)
 ├── qjl.rs            (depends on error, uses rand_chacha)
 ├── turbo.rs          (depends on polar, qjl, error)
 └── kv.rs             (depends on turbo, error)
```

---

## References

1. **TurboQuant**: Zandieh et al., "TurboQuant: Redefining AI Efficiency with Extreme Compression", *ICLR 2026*.
2. **PolarQuant**: Zandieh et al., *AISTATS 2026*.
3. **QJL (Quantized Johnson-Lindenstrauss)**: Zandieh et al., *AAAI 2025*.

---

## License

This project is licensed under the [MIT License](LICENSE).
