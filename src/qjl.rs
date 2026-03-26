//! Quantized Johnson-Lindenstrauss (QJL) transform for unbiased inner product estimation.
//!
//! QJL projects a d-dimensional vector onto m random hyperplanes and records
//! only the sign of each projection. This costs exactly 1 bit per projection
//! dimension — zero memory overhead for quantization constants.
//!
//! # Mathematical Guarantee
//!
//! For random Gaussian projection vectors g₁, …, gₘ ~ N(0, I_d):
//!
//! ```text
//! E[sign(gᵢ · x) · (gᵢ · y)] = (2/π) · ⟨x, y⟩
//! ```
//!
//! This gives the unbiased estimator:
//!
//! ```text
//! ⟨x, y⟩ ≈ (π / 2m) · Σᵢ sign(gᵢ · x) · (gᵢ · y)
//! ```
//!
//! In TurboQuant, QJL is applied to the *residual* vector after PolarQuant
//! reconstruction, correcting bias introduced by angle quantization and pushing
//! the combined estimator toward the information-theoretic optimum.
//!
//! # Reference
//!
//! "Quantized Johnson-Lindenstrauss is Optimal for Inner Products"
//! Zandieh et al., AAAI 2025.

use std::f32::consts::PI;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::{Result, TurboQuantError};

/// A QJL sketch: the sign of each random projection, stored compactly as i8 (+1 or -1).
///
/// At query time the caller must have access to the same projection matrix
/// (regenerated from seed) to compute `g · query` for each row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct QjlSketch {
    /// The dimension of the original vector.
    pub dim: usize,
    /// Number of random projections (sketch dimension).
    pub projections: usize,
    /// Sign bits: each entry is +1 or -1.
    pub signs: Vec<i8>,
}

impl QjlSketch {
    /// Estimated bytes used by this sketch (just the sign bits).
    pub fn encoded_bytes(&self) -> usize {
        // Ideally packed as bitfields, but stored as i8 for simplicity.
        // A production implementation would use bitvec or u64 bitmasks.
        self.projections
    }
}

/// Projects vectors to QJL sketches and estimates inner products from sketches.
///
/// The projection matrix is never stored — it is regenerated from `seed` on demand,
/// keeping the quantizer footprint small regardless of d and m.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QjlQuantizer {
    dim: usize,
    projections: usize,
    seed: u64,
}

impl QjlQuantizer {
    /// Create a new QJL quantizer.
    ///
    /// - `dim`: input vector dimension
    /// - `projections` (m): number of random projections. Higher values reduce
    ///   estimation variance. Rule of thumb: m ≈ d / 4 for good accuracy.
    /// - `seed`: controls the random projection matrix deterministically
    pub fn new(dim: usize, projections: usize, seed: u64) -> Result<Self> {
        if dim == 0 {
            return Err(TurboQuantError::ZeroDimension);
        }
        if projections == 0 {
            return Err(TurboQuantError::ZeroProjectionCount);
        }
        Ok(Self { dim, projections, seed })
    }

    /// The input vector dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The number of random projections (sketch dimension).
    pub fn projections(&self) -> usize {
        self.projections
    }

    /// Project `vector` to a [`QjlSketch`].
    ///
    /// Computes sign(G · x) for random G ~ N(0, 1)^{m×d}, storing only the
    /// signs. The projection matrix is regenerated from seed.
    pub fn sketch(&self, vector: &[f32]) -> Result<QjlSketch> {
        if vector.len() != self.dim {
            return Err(TurboQuantError::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }

        let g = self.projection_matrix();
        let mut signs = Vec::with_capacity(self.projections);

        for row in g.iter() {
            let dot: f32 = row.iter().zip(vector.iter()).map(|(g, x)| g * x).sum();
            signs.push(if dot >= 0.0 { 1i8 } else { -1i8 });
        }

        Ok(QjlSketch {
            dim: self.dim,
            projections: self.projections,
            signs,
        })
    }

    /// Estimate ⟨x, query⟩ from a sketch of x and the raw query vector.
    ///
    /// Returns the unbiased QJL estimator:
    /// `(π / 2m) · Σᵢ signs[i] · (gᵢ · query)`
    pub fn inner_product_estimate(&self, sketch: &QjlSketch, query: &[f32]) -> Result<f32> {
        self.validate_sketch(sketch)?;
        if query.len() != self.dim {
            return Err(TurboQuantError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }

        let g = self.projection_matrix();
        let m = self.projections as f32;
        let scale = PI / (2.0 * m);

        let estimate: f32 = g
            .iter()
            .zip(sketch.signs.iter())
            .map(|(row, &sign)| {
                let g_dot_query: f32 = row.iter().zip(query.iter()).map(|(g, q)| g * q).sum();
                sign as f32 * g_dot_query
            })
            .sum();

        Ok(scale * estimate)
    }

    /// Generate the m×d projection matrix G from seed.
    ///
    /// Entries are i.i.d. N(0, 1). This is deterministic for a given
    /// `(dim, projections, seed)` triple.
    fn projection_matrix(&self) -> Vec<Vec<f32>> {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        use rand_distr::{Distribution, StandardNormal};

        let mut rng = ChaCha8Rng::seed_from_u64(self.seed.wrapping_add(0xDEAD_BEEF_1234_5678));
        let dist = StandardNormal;

        (0..self.projections)
            .map(|_| (0..self.dim).map(|_| dist.sample(&mut rng)).collect())
            .collect()
    }

    fn validate_sketch(&self, sketch: &QjlSketch) -> Result<()> {
        if sketch.dim != self.dim {
            return Err(TurboQuantError::DimensionMismatch {
                expected: self.dim,
                got: sketch.dim,
            });
        }
        if sketch.projections != self.projections {
            return Err(TurboQuantError::MalformedCode {
                reason: format!(
                    "sketch has {} projections, quantizer expects {}",
                    sketch.projections, self.projections
                ),
            });
        }
        Ok(())
    }
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
    fn sketch_is_deterministic() {
        let q = QjlQuantizer::new(16, 32, 42).unwrap();
        let x = random_vector(16, 1);
        let s1 = q.sketch(&x).unwrap();
        let s2 = q.sketch(&x).unwrap();
        assert_eq!(s1.signs, s2.signs);
    }

    #[test]
    fn signs_are_plus_or_minus_one() {
        let q = QjlQuantizer::new(8, 16, 7).unwrap();
        let x = random_vector(8, 2);
        let s = q.sketch(&x).unwrap();
        for sign in &s.signs {
            assert!(*sign == 1 || *sign == -1, "unexpected sign value: {sign}");
        }
    }

    #[test]
    fn inner_product_estimate_is_unbiased_over_many_samples() {
        // With many projections, the QJL estimator should converge to the true
        // inner product. We use large m to reduce variance for this test.
        let dim = 64;
        let m = 2048;
        let q = QjlQuantizer::new(dim, m, 0).unwrap();

        let x = random_vector(dim, 10);
        let y = random_vector(dim, 20);

        let exact: f32 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        let sketch = q.sketch(&x).unwrap();
        let estimated = q.inner_product_estimate(&sketch, &y).unwrap();

        let error = (estimated - exact).abs();
        // With m=2048, variance is very low. Allow 10% relative error.
        let tolerance = 0.10 * exact.abs() + 0.5;
        assert!(
            error < tolerance,
            "error={error:.3}, exact={exact:.3}, estimated={estimated:.3}"
        );
    }

    #[test]
    fn same_vector_gives_positive_self_similarity() {
        let q = QjlQuantizer::new(16, 256, 5).unwrap();
        let x = random_vector(16, 99);
        let sketch = q.sketch(&x).unwrap();
        let estimate = q.inner_product_estimate(&sketch, &x).unwrap();
        // ⟨x, x⟩ = ||x||² > 0
        assert!(estimate > 0.0, "self inner product should be positive, got {estimate}");
    }

    #[test]
    fn orthogonal_vectors_estimate_near_zero() {
        // x = [1, 0, 0, ...], y = [0, 1, 0, ...] are orthogonal, ⟨x,y⟩ = 0.
        let dim = 64;
        let q = QjlQuantizer::new(dim, 1024, 3).unwrap();

        let mut x = vec![0.0f32; dim];
        let mut y = vec![0.0f32; dim];
        x[0] = 1.0;
        y[1] = 1.0;

        let sketch = q.sketch(&x).unwrap();
        let estimate = q.inner_product_estimate(&sketch, &y).unwrap();

        assert!(
            estimate.abs() < 0.15,
            "orthogonal estimate should be near zero, got {estimate:.4}"
        );
    }

    #[test]
    fn zero_dim_rejected() {
        assert!(QjlQuantizer::new(0, 8, 0).is_err());
    }

    #[test]
    fn zero_projections_rejected() {
        assert!(QjlQuantizer::new(8, 0, 0).is_err());
    }

    #[test]
    fn sketch_serialization_roundtrip() {
        let q = QjlQuantizer::new(8, 16, 42).unwrap();
        let x = random_vector(8, 77);
        let sketch = q.sketch(&x).unwrap();
        let json = serde_json::to_string(&sketch).unwrap();
        let restored: QjlSketch = serde_json::from_str(&json).unwrap();
        assert_eq!(sketch, restored);
    }
}
