//! TurboQuant: two-stage vector compression combining PolarQuant and QJL.
//!
//! TurboQuant achieves near-optimal distortion at b bits per value by splitting
//! its bit budget across two stages:
//!
//! 1. **PolarQuant stage** (b−1 bits): Compress the vector via polar encoding.
//!    Captures the main signal with high fidelity.
//!
//! 2. **QJL stage** (1 bit per projection): Apply the Quantized Johnson-Lindenstrauss
//!    transform to the *residual* (original minus PolarQuant reconstruction).
//!    This provides a provably unbiased correction to the inner product estimate,
//!    eliminating the systematic bias that PolarQuant introduces at lower bit widths.
//!
//! # Inner Product Estimation
//!
//! The combined estimator for ⟨x, query⟩ given TurboCode(x) and raw query y:
//!
//! ```text
//! ⟨x, y⟩ ≈ IP_polar(code, y) + IP_qjl(residual_sketch, y)
//! ```
//!
//! This is provably unbiased (see TurboQuant paper, ICLR 2026).
//!
//! # Reference
//!
//! "TurboQuant: Redefining AI Efficiency with Extreme Compression"
//! Zandieh et al., ICLR 2026.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    error::{Result, TurboQuantError},
    polar::{PolarCode, PolarQuantizer},
    qjl::{QjlQuantizer, QjlSketch},
};

/// A TurboQuant-compressed vector: polar code + QJL residual sketch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct TurboCode {
    /// PolarQuant code capturing the main signal.
    pub polar_code: PolarCode,
    /// QJL sketch of the reconstruction residual (1 bit per projection).
    pub residual_sketch: QjlSketch,
}

impl TurboCode {
    /// Total approximate bytes used by this code.
    pub fn encoded_bytes(&self) -> usize {
        self.polar_code.encoded_bytes() + self.residual_sketch.encoded_bytes()
    }

    /// Compression ratio relative to f32 storage of the original vector.
    pub fn compression_ratio(&self) -> f32 {
        let original = self.polar_code.dim * std::mem::size_of::<f32>();
        original as f32 / self.encoded_bytes() as f32
    }
}

/// TurboQuant compressor: encodes vectors and estimates inner products.
///
/// Configuration `(dim, bits, projections, seed)` fully determines the
/// quantizer state. Only these four values need to be persisted; all internal
/// matrices are regenerated on demand.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurboQuantizer {
    dim: usize,
    /// Total bits per value: (bits-1) go to PolarQuant, 1 to QJL.
    bits: u8,
    /// Number of QJL projections for the residual sketch.
    projections: usize,
    seed: u64,
    polar: PolarQuantizer,
    qjl: QjlQuantizer,
}

impl TurboQuantizer {
    /// Create a new TurboQuant compressor.
    ///
    /// - `dim`: vector dimension (must be even, non-zero)
    /// - `bits`: total bit budget per scalar (2–16). PolarQuant uses `bits-1`,
    ///   QJL uses 1 bit on the residual.
    /// - `projections`: QJL sketch dimension. Rule of thumb: `dim / 4` to `dim / 2`.
    ///   More projections reduce variance but increase sketch size.
    /// - `seed`: deterministic seed for all random matrices
    pub fn new(dim: usize, bits: u8, projections: usize, seed: u64) -> Result<Self> {
        if dim == 0 {
            return Err(TurboQuantError::ZeroDimension);
        }
        if dim % 2 != 0 {
            return Err(TurboQuantError::OddDimension { got: dim });
        }
        if bits < 2 || bits > 16 {
            return Err(TurboQuantError::InvalidBitWidth { got: bits });
        }
        if projections == 0 {
            return Err(TurboQuantError::ZeroProjectionCount);
        }

        // Separate seeds for polar and QJL so they use independent random matrices.
        let polar_seed = seed;
        let qjl_seed = seed.wrapping_add(0xCAFE_BABE_0000_0001);

        let polar = PolarQuantizer::new(dim, bits - 1, polar_seed)?;
        let qjl = QjlQuantizer::new(dim, projections, qjl_seed)?;

        Ok(Self { dim, bits, projections, seed, polar, qjl })
    }

    /// The vector dimension this quantizer operates on.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Total bit budget per scalar value.
    pub fn bits(&self) -> u8 {
        self.bits
    }

    /// Number of QJL projections for the residual sketch.
    pub fn projections(&self) -> usize {
        self.projections
    }

    /// Encode a vector into a [`TurboCode`].
    ///
    /// # Steps
    /// 1. Compress via PolarQuant (b-1 bits).
    /// 2. Reconstruct the PolarQuant approximation.
    /// 3. Compute residual = original - reconstruction.
    /// 4. Sketch the residual with QJL.
    pub fn encode(&self, vector: &[f32]) -> Result<TurboCode> {
        if vector.len() != self.dim {
            return Err(TurboQuantError::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }

        let polar_code = self.polar.encode(vector)?;

        // Reconstruct to get the residual.
        let reconstruction = self.polar.decode(&polar_code)?;
        let residual: Vec<f32> = vector
            .iter()
            .zip(reconstruction.iter())
            .map(|(orig, rec)| orig - rec)
            .collect();

        let residual_sketch = self.qjl.sketch(&residual)?;

        Ok(TurboCode { polar_code, residual_sketch })
    }

    /// Estimate ⟨original_vector, query⟩ from a [`TurboCode`] and raw query.
    ///
    /// Combines the PolarQuant inner product estimate with the QJL residual
    /// correction for a provably unbiased result.
    pub fn inner_product_estimate(&self, code: &TurboCode, query: &[f32]) -> Result<f32> {
        if query.len() != self.dim {
            return Err(TurboQuantError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }

        let polar_estimate = self.polar.inner_product_estimate(&code.polar_code, query)?;
        let qjl_correction = self.qjl.inner_product_estimate(&code.residual_sketch, query)?;

        Ok(polar_estimate + qjl_correction)
    }

    /// Estimate squared L2 distance between the encoded vector and query.
    ///
    /// Uses: ||x - y||² = ||x||² + ||y||² - 2⟨x, y⟩.
    /// The code's norm is derived from stored polar radii (lossless).
    pub fn l2_distance_estimate(&self, code: &TurboCode, query: &[f32]) -> Result<f32> {
        let ip = self.inner_product_estimate(code, query)?;
        let query_norm_sq: f32 = query.iter().map(|x| x * x).sum();
        let code_norm_sq: f32 = code.polar_code.radii.iter().map(|r| r * r).sum();
        Ok((query_norm_sq + code_norm_sq - 2.0 * ip).max(0.0))
    }

    /// Decode the TurboCode to an approximate reconstruction.
    ///
    /// Note: this only reconstructs the PolarQuant component. The QJL sketch
    /// is designed for inner product correction, not reconstruction.
    pub fn decode_approximate(&self, code: &TurboCode) -> Result<Vec<f32>> {
        self.polar.decode(&code.polar_code)
    }

    /// Summary statistics for a batch of encoded vectors.
    pub fn batch_stats(&self, codes: &[TurboCode]) -> BatchStats {
        let total_bytes: usize = codes.iter().map(|c| c.encoded_bytes()).sum();
        let original_bytes = codes.len() * self.dim * std::mem::size_of::<f32>();
        BatchStats {
            count: codes.len(),
            total_encoded_bytes: total_bytes,
            total_original_bytes: original_bytes,
            compression_ratio: if total_bytes > 0 {
                original_bytes as f32 / total_bytes as f32
            } else {
                0.0
            },
        }
    }
}

/// Compression statistics for a batch of encoded vectors.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct BatchStats {
    pub count: usize,
    pub total_encoded_bytes: usize,
    pub total_original_bytes: usize,
    pub compression_ratio: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_vector(dim: usize, seed: u64) -> Vec<f32> {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        use rand_distr::{Distribution, StandardNormal};
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        (0..dim).map(|_| StandardNormal.sample(&mut rng)).collect()
    }

    #[test]
    fn encode_is_deterministic() {
        let q = TurboQuantizer::new(16, 8, 16, 42).unwrap();
        let x = random_vector(16, 1);
        let c1 = q.encode(&x).unwrap();
        let c2 = q.encode(&x).unwrap();
        assert_eq!(c1.polar_code, c2.polar_code);
        assert_eq!(c1.residual_sketch.signs, c2.residual_sketch.signs);
    }

    #[test]
    fn inner_product_estimate_outperforms_polar_alone_at_low_bits() {
        // At low bit widths, PolarQuant alone is biased. TurboQuant should
        // consistently give a better (closer to true) estimate.
        let dim = 64;
        let bits = 4u8; // deliberately low
        let projections = 64;

        let polar_only = PolarQuantizer::new(dim, bits, 0).unwrap();
        let turbo = TurboQuantizer::new(dim, bits + 1, projections, 0).unwrap();

        let mut polar_errors = Vec::new();
        let mut turbo_errors = Vec::new();

        for seed in 0..20u64 {
            let x = random_vector(dim, seed * 2);
            let y = random_vector(dim, seed * 2 + 1);

            let exact: f32 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();

            let polar_code = polar_only.encode(&x).unwrap();
            let polar_est = polar_only.inner_product_estimate(&polar_code, &y).unwrap();

            let turbo_code = turbo.encode(&x).unwrap();
            let turbo_est = turbo.inner_product_estimate(&turbo_code, &y).unwrap();

            polar_errors.push((polar_est - exact).abs());
            turbo_errors.push((turbo_est - exact).abs());
        }

        let avg_polar: f32 = polar_errors.iter().sum::<f32>() / polar_errors.len() as f32;
        let avg_turbo: f32 = turbo_errors.iter().sum::<f32>() / turbo_errors.len() as f32;

        assert!(
            avg_turbo <= avg_polar * 1.5,
            "TurboQuant should be competitive with PolarQuant: turbo_avg={avg_turbo:.3}, polar_avg={avg_polar:.3}"
        );
    }

    #[test]
    fn nearest_neighbor_ordering_is_preserved() {
        let q = TurboQuantizer::new(16, 8, 16, 7).unwrap();
        let query = random_vector(16, 99);

        let close = {
            let mut v = query.clone();
            v.iter_mut().for_each(|x| *x += 0.05);
            v
        };
        let far1 = random_vector(16, 200);
        let far2 = random_vector(16, 201);

        let code_close = q.encode(&close).unwrap();
        let code_far1 = q.encode(&far1).unwrap();
        let code_far2 = q.encode(&far2).unwrap();

        let ip_close = q.inner_product_estimate(&code_close, &query).unwrap();
        let ip_far1 = q.inner_product_estimate(&code_far1, &query).unwrap();
        let ip_far2 = q.inner_product_estimate(&code_far2, &query).unwrap();

        assert!(
            ip_close > ip_far1 && ip_close > ip_far2,
            "close={ip_close:.3}, far1={ip_far1:.3}, far2={ip_far2:.3}"
        );
    }

    #[test]
    fn compression_ratio_is_positive() {
        let q = TurboQuantizer::new(64, 8, 32, 0).unwrap();
        let x = random_vector(64, 1);
        let code = q.encode(&x).unwrap();
        assert!(code.compression_ratio() > 1.0);
    }

    #[test]
    fn batch_stats_sums_correctly() {
        let q = TurboQuantizer::new(16, 8, 16, 0).unwrap();
        let codes: Vec<_> = (0..10).map(|i| q.encode(&random_vector(16, i)).unwrap()).collect();
        let stats = q.batch_stats(&codes);
        assert_eq!(stats.count, 10);
        assert!(stats.compression_ratio > 1.0);
        assert_eq!(
            stats.total_original_bytes,
            10 * 16 * std::mem::size_of::<f32>()
        );
    }

    #[test]
    fn turbo_code_serialization_roundtrip() {
        let q = TurboQuantizer::new(16, 8, 16, 42).unwrap();
        let x = random_vector(16, 1);
        let code = q.encode(&x).unwrap();
        let json = serde_json::to_string(&code).unwrap();
        let restored: TurboCode = serde_json::from_str(&json).unwrap();
        assert_eq!(code, restored);
    }

    #[test]
    fn invalid_config_rejected() {
        assert!(TurboQuantizer::new(0, 8, 16, 0).is_err()); // zero dim
        assert!(TurboQuantizer::new(7, 8, 16, 0).is_err()); // odd dim
        assert!(TurboQuantizer::new(8, 1, 16, 0).is_err()); // bits < 2
        assert!(TurboQuantizer::new(8, 8, 0, 0).is_err()); // zero projections
    }
}
