//! # turbo-quant
//!
//! Rust implementation of **TurboQuant**, **PolarQuant**, and **QJL** —
//! the vector quantization algorithms from Google Research (ICLR 2026).
//!
//! These algorithms compress high-dimensional vectors (embeddings, KV cache
//! entries) to 3–8 bits per value with **zero accuracy loss** and
//! **no dataset-specific calibration**.
//!
//! ## Key Properties
//!
//! - **Data-oblivious**: no k-means training, no codebook, no calibration set.
//!   The rotation is seeded once and works on any distribution.
//! - **Deterministic**: identical `(dim, bits, seed)` always produces the same
//!   quantizer. State can be fully reconstructed from four integers.
//! - **Zero accuracy loss**: at 3+ bits, inner product estimates are provably
//!   unbiased and achieve near-optimal distortion (within ~2.7× of Shannon limit).
//! - **Instant indexing**: unlike Product Quantization, there is no offline
//!   training phase. Vectors can be indexed as they arrive.
//!
//! ## Quick Start
//!
//! ```rust
//! use turbo_quant::{TurboQuantizer, PolarQuantizer};
//!
//! // Compress 1536-dimensional embeddings (OpenAI/sentence-transformer size).
//! let dim = 64; // use 1536 in production
//! let q = TurboQuantizer::new(dim, 8, 32, /* seed */ 42).unwrap();
//!
//! let database_vector: Vec<f32> = vec![0.1; dim]; // your embedding here
//! let query_vector: Vec<f32> = vec![0.1; dim];    // your query here
//!
//! // Compress the database vector (store this, not the raw f32 array).
//! let code = q.encode(&database_vector).unwrap();
//!
//! // At query time: estimate inner product without decompressing.
//! let score = q.inner_product_estimate(&code, &query_vector).unwrap();
//!
//! // Or just use PolarQuant for a simpler single-stage compressor.
//! let pq = PolarQuantizer::new(dim, 8, 42).unwrap();
//! let polar_code = pq.encode(&database_vector).unwrap();
//! let polar_score = pq.inner_product_estimate(&polar_code, &query_vector).unwrap();
//! ```
//!
//! ## Choosing Parameters
//!
//! | Use case | Recommended bits | Recommended projections |
//! |---|---|---|
//! | Semantic search (recall@10) | 8 | dim / 4 |
//! | KV cache compression | 4–6 | dim / 8 |
//! | Maximum compression | 3 | dim / 16 |
//!
//! ## References
//!
//! - TurboQuant: Zandieh et al., ICLR 2026
//! - PolarQuant: Zandieh et al., AISTATS 2026
//! - QJL: Zandieh et al., AAAI 2025

pub mod error;
pub mod kv;
pub mod polar;
pub mod qjl;
pub mod rotation;
pub mod turbo;

pub use error::{Result, TurboQuantError};
pub use kv::{CompressedToken, KvCacheCompressor, KvCacheConfig};
pub use polar::{PolarCode, PolarQuantizer};
pub use qjl::{QjlQuantizer, QjlSketch};
pub use rotation::{Rotation, StoredRotation};
pub use turbo::{BatchStats, TurboCode, TurboQuantizer};
