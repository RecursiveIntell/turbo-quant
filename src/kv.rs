//! KV cache compression for transformer attention.
//!
//! In a decoder-only transformer, the KV cache stores one key/value pair per
//! token per layer. With long context windows this dominates GPU memory.
//!
//! [`KvCacheCompressor`] provides an online interface: call [`compress_token`]
//! as each token is generated, then [`attention_scores`] to compute the full
//! attention logit vector against all stored compressed keys — without
//! decompressing them.
//!
//! # Memory Savings
//!
//! At 8 bits with d=128 (typical per-head dimension):
//! - Uncompressed key: 128 × 4 bytes = 512 bytes
//! - TurboCode key:   64 × 4 (radii) + 64 × 2 (angle indices) + 32 × 1 (QJL signs)
//!               =   256 + 128 + 32 = 416 bytes
//!
//! At 4 bits the savings are more dramatic. Values are compressed identically.
//!
//! # Usage
//!
//! ```rust
//! use turbo_quant::kv::{KvCacheCompressor, KvCacheConfig};
//!
//! let config = KvCacheConfig {
//!     head_dim: 64,        // per-attention-head dimension
//!     bits: 8,             // bit budget for keys and values
//!     projections: 16,     // QJL sketch projections
//!     seed: 42,
//! };
//!
//! let mut cache = KvCacheCompressor::new(config).unwrap();
//!
//! // At each generation step, compress the new key and value.
//! let key = vec![0.1f32; 64];
//! let value = vec![0.2f32; 64];
//! cache.compress_token(&key, &value).unwrap();
//!
//! // At attention time: compute logits against all compressed keys.
//! let query = vec![0.15f32; 64];
//! let scores = cache.attention_scores(&query).unwrap();
//! assert_eq!(scores.len(), 1); // one score per stored token
//!
//! // Retrieve approximate reconstructed values for the top-k tokens.
//! let values = cache.decode_values(&[0]).unwrap();
//! assert_eq!(values.len(), 1);
//! ```

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    error::{Result, TurboQuantError},
    turbo::{TurboCode, TurboQuantizer},
};

/// Configuration for a single attention head's KV cache compressor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct KvCacheConfig {
    /// Dimension of each attention head's key/value vectors.
    pub head_dim: usize,
    /// Bit budget per scalar value (2–16). 8 gives near-lossless recall.
    pub bits: u8,
    /// Number of QJL projections for the residual sketch.
    /// Rule of thumb: `head_dim / 4`.
    pub projections: usize,
    /// Deterministic seed for all random matrices.
    pub seed: u64,
}

/// A compressed KV cache entry for one token.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CompressedToken {
    pub compressed_key: TurboCode,
    pub compressed_value: TurboCode,
}

/// Online KV cache compressor for one attention head.
///
/// Tokens are appended in generation order. Attention scores are computed
/// across all stored compressed keys without decompression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvCacheCompressor {
    config: KvCacheConfig,
    key_quantizer: TurboQuantizer,
    value_quantizer: TurboQuantizer,
    tokens: Vec<CompressedToken>,
}

impl KvCacheCompressor {
    /// Create a new compressor for one attention head.
    pub fn new(config: KvCacheConfig) -> Result<Self> {
        let key_quantizer = TurboQuantizer::new(
            config.head_dim,
            config.bits,
            config.projections,
            config.seed,
        )?;
        // Value quantizer uses an independent seed offset so key and value
        // rotation matrices are uncorrelated.
        let value_quantizer = TurboQuantizer::new(
            config.head_dim,
            config.bits,
            config.projections,
            config.seed.wrapping_add(0x1234_5678_ABCD_EF00),
        )?;
        Ok(Self {
            config,
            key_quantizer,
            value_quantizer,
            tokens: Vec::new(),
        })
    }

    /// The number of tokens currently stored in this cache.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// True if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Compress and store one (key, value) token pair.
    ///
    /// Called once per generation step. O(d) time.
    pub fn compress_token(&mut self, key: &[f32], value: &[f32]) -> Result<()> {
        if key.len() != self.config.head_dim {
            return Err(TurboQuantError::DimensionMismatch {
                expected: self.config.head_dim,
                got: key.len(),
            });
        }
        if value.len() != self.config.head_dim {
            return Err(TurboQuantError::DimensionMismatch {
                expected: self.config.head_dim,
                got: value.len(),
            });
        }

        let compressed_key = self.key_quantizer.encode(key)?;
        let compressed_value = self.value_quantizer.encode(value)?;
        self.tokens.push(CompressedToken { compressed_key, compressed_value });
        Ok(())
    }

    /// Compute attention logits q·kᵢ for all stored tokens i.
    ///
    /// Returns a vector of length `self.len()`. Does not decompress keys.
    /// Suitable for softmax → attention weight computation.
    ///
    /// O(n·d) time where n is the number of stored tokens.
    pub fn attention_scores(&self, query: &[f32]) -> Result<Vec<f32>> {
        if query.len() != self.config.head_dim {
            return Err(TurboQuantError::DimensionMismatch {
                expected: self.config.head_dim,
                got: query.len(),
            });
        }

        self.tokens
            .iter()
            .map(|t| self.key_quantizer.inner_product_estimate(&t.compressed_key, query))
            .collect()
    }

    /// Softmax-weighted sum of approximate value reconstructions.
    ///
    /// This is the full compressed attention output for one head:
    /// `Σᵢ softmax(scores)ᵢ · decode(vᵢ)`
    ///
    /// Note: values are decoded (approximate reconstruction) rather than
    /// stored in compressed form, since weighted sums cannot be computed
    /// in compressed space. This is consistent with how KV cache compression
    /// is used in practice (compress keys for score computation; decode values
    /// for the weighted sum).
    pub fn attend(&self, query: &[f32]) -> Result<Vec<f32>> {
        if self.tokens.is_empty() {
            return Ok(vec![0.0f32; self.config.head_dim]);
        }

        let scores = self.attention_scores(query)?;

        // Numerically stable softmax.
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
        let sum_exp: f32 = exps.iter().sum();
        let weights: Vec<f32> = exps.iter().map(|e| e / sum_exp).collect();

        // Weighted sum of decoded values.
        let mut output = vec![0.0f32; self.config.head_dim];
        for (token, &weight) in self.tokens.iter().zip(weights.iter()) {
            let decoded = self.value_quantizer.decode_approximate(&token.compressed_value)?;
            for (out, val) in output.iter_mut().zip(decoded.iter()) {
                *out += weight * val;
            }
        }

        Ok(output)
    }

    /// Decode the value vector for specific token indices (for top-k attention).
    pub fn decode_values(&self, indices: &[usize]) -> Result<Vec<Vec<f32>>> {
        indices
            .iter()
            .map(|&i| {
                if i >= self.tokens.len() {
                    return Err(TurboQuantError::DimensionMismatch {
                        expected: self.tokens.len(),
                        got: i + 1,
                    });
                }
                self.value_quantizer
                    .decode_approximate(&self.tokens[i].compressed_value)
            })
            .collect()
    }

    /// Approximate total bytes used by all compressed tokens.
    pub fn compressed_bytes(&self) -> usize {
        self.tokens.iter().map(|t| {
            t.compressed_key.encoded_bytes() + t.compressed_value.encoded_bytes()
        }).sum()
    }

    /// Bytes that would be used if keys/values were stored uncompressed as f32.
    pub fn uncompressed_bytes(&self) -> usize {
        self.tokens.len() * 2 * self.config.head_dim * std::mem::size_of::<f32>()
    }

    /// Compression ratio: uncompressed / compressed.
    pub fn compression_ratio(&self) -> f32 {
        let compressed = self.compressed_bytes();
        if compressed == 0 {
            return 0.0;
        }
        self.uncompressed_bytes() as f32 / compressed as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_vec(dim: usize, seed: u64) -> Vec<f32> {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        use rand_distr::{Distribution, StandardNormal};
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        (0..dim).map(|_| StandardNormal.sample(&mut rng)).collect()
    }

    fn make_cache(dim: usize) -> KvCacheCompressor {
        KvCacheCompressor::new(KvCacheConfig {
            head_dim: dim,
            bits: 8,
            projections: dim / 4,
            seed: 42,
        }).unwrap()
    }

    #[test]
    fn empty_cache_returns_empty_scores() {
        let cache = make_cache(16);
        let scores = cache.attention_scores(&random_vec(16, 1)).unwrap();
        assert!(scores.is_empty());
    }

    #[test]
    fn token_count_increments_correctly() {
        let mut cache = make_cache(16);
        for i in 0..5 {
            cache.compress_token(&random_vec(16, i), &random_vec(16, i + 100)).unwrap();
        }
        assert_eq!(cache.len(), 5);
    }

    #[test]
    fn attention_scores_length_matches_token_count() {
        let mut cache = make_cache(16);
        for i in 0..10 {
            cache.compress_token(&random_vec(16, i), &random_vec(16, i + 100)).unwrap();
        }
        let scores = cache.attention_scores(&random_vec(16, 999)).unwrap();
        assert_eq!(scores.len(), 10);
    }

    #[test]
    fn highest_score_is_for_query_similar_key() {
        let dim = 16;
        let mut cache = make_cache(dim);

        // Add 9 random keys.
        for i in 0..9u64 {
            cache.compress_token(&random_vec(dim, i * 10), &random_vec(dim, i * 10 + 1)).unwrap();
        }

        // Add one key that is very close to the query.
        let query = random_vec(dim, 999);
        let similar_key: Vec<f32> = query.iter().map(|x| x + 0.001).collect();
        cache.compress_token(&similar_key, &random_vec(dim, 9000)).unwrap();

        let scores = cache.attention_scores(&query).unwrap();
        let best_idx = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        assert_eq!(best_idx, 9, "similar key should have highest attention score");
    }

    #[test]
    fn attend_output_has_correct_dimension() {
        let dim = 16;
        let mut cache = make_cache(dim);
        for i in 0..5u64 {
            cache.compress_token(&random_vec(dim, i), &random_vec(dim, i + 50)).unwrap();
        }
        let output = cache.attend(&random_vec(dim, 1)).unwrap();
        assert_eq!(output.len(), dim);
    }

    #[test]
    fn attend_weights_sum_to_one_implicitly() {
        // If all values are the same vector v, attend() should return ≈ v.
        let dim = 16;
        let mut cache = make_cache(dim);
        let v = random_vec(dim, 77);

        for i in 0..8u64 {
            cache.compress_token(&random_vec(dim, i), &v).unwrap();
        }

        let output = cache.attend(&random_vec(dim, 1)).unwrap();

        // Output should be close to v since all values are the same.
        let error: f32 = output.iter().zip(v.iter()).map(|(o, vi)| (o - vi).powi(2)).sum::<f32>().sqrt();
        let v_norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(error / v_norm < 0.15, "attend output too far from value: relative_error={:.4}", error / v_norm);
    }

    #[test]
    fn compression_ratio_is_above_one() {
        let mut cache = make_cache(64);
        for i in 0..20u64 {
            cache.compress_token(&random_vec(64, i), &random_vec(64, i + 1000)).unwrap();
        }
        let ratio = cache.compression_ratio();
        assert!(ratio > 1.0, "compression ratio should be > 1, got {ratio:.2}");
        println!("compression ratio at 8 bits, d=64: {ratio:.2}x");
    }

    #[test]
    fn wrong_key_dimension_is_rejected() {
        let mut cache = make_cache(16);
        let result = cache.compress_token(&random_vec(8, 0), &random_vec(16, 1));
        assert!(result.is_err());
    }

    #[test]
    fn wrong_query_dimension_is_rejected() {
        let mut cache = make_cache(16);
        cache.compress_token(&random_vec(16, 0), &random_vec(16, 1)).unwrap();
        let result = cache.attention_scores(&random_vec(8, 0));
        assert!(result.is_err());
    }

    #[test]
    fn key_and_value_quantizers_are_independent() {
        // Encode the same vector as both key and value.
        // The codes should differ because the quantizers use different seeds.
        let mut cache = make_cache(16);
        let v = random_vec(16, 42);
        cache.compress_token(&v, &v).unwrap();

        let token = &cache.tokens[0];
        // Different rotation seeds → different angle indices with overwhelming probability.
        assert_ne!(
            token.compressed_key.polar_code.angle_indices,
            token.compressed_value.polar_code.angle_indices,
            "key and value quantizers should use independent rotations"
        );
    }
}
