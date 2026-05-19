//! Quantized Johnson-Lindenstrauss (QJL) transform for approximate inner product estimation.
//!
//! QJL projects a d-dimensional vector onto m random hyperplanes and records
//! only the sign of each projection.
//!
//! # Mathematical Guarantee
//!
//! For random Gaussian projection vectors g₁, …, gₘ ~ N(0, I_d):
//!
//! ```text
//! E[sign(gᵢ · x) · (gᵢ · y)] = sqrt(2/π) · ⟨x, y⟩ / ||x||
//! ```
//!
//! This gives a sign-projection estimator:
//!
//! ```text
//! ⟨x, y⟩ ≈ (sqrt(π/2) · ||x|| / m) · Σᵢ sign(gᵢ · x) · (gᵢ · y)
//! ```
//!
//! In TurboQuant, QJL can be applied to the *residual* vector after PolarQuant
//! reconstruction. It is optional and should be benchmark-gated for each
//! workload.
//!
//! # Reference
//!
//! Quantized Johnson-Lindenstrauss sign-projection inner-product sketches.

use std::f32::consts::PI;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    bitpack,
    error::{Result, TurboQuantError},
};

/// A QJL sketch: the sign of each random projection.
///
/// At query time the caller must have access to the same projection matrix
/// (regenerated from seed) to compute `g · query` for each row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct QjlSketch {
    /// The dimension of the original vector.
    pub dim: usize,
    /// Number of random projections (sketch dimension).
    pub projections: usize,
    /// Projection signs. Each value must be -1 or 1.
    pub signs: Vec<i8>,
}

/// Explicit provenance needed for source-norm-scaled QJL estimates.
///
/// The legacy [`QjlSketch`] shape is preserved for `0.1.0` compatibility and
/// intentionally does not grow hidden fields. New storage or transport paths
/// that need the full sign-projection estimator should persist this sidecar
/// provenance with the sketch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct QjlSketchProvenanceV1 {
    /// Euclidean norm of the vector used to produce the sketch.
    pub source_norm: f32,
    /// Optional caller-owned digest tying this provenance to an external source.
    pub source_digest: Option<String>,
}

impl QjlSketchProvenanceV1 {
    /// Build explicit QJL provenance.
    pub fn new(source_norm: f32, source_digest: Option<String>) -> Result<Self> {
        validate_source_norm(source_norm)?;
        Ok(Self {
            source_norm,
            source_digest,
        })
    }
}

impl QjlSketch {
    /// Build a sketch from logical signs.
    pub fn from_parts(dim: usize, projections: usize, signs: &[i8]) -> Result<Self> {
        let sketch = Self {
            dim,
            projections,
            signs: signs.to_vec(),
        };
        sketch.validate_for(dim, projections)?;
        Ok(sketch)
    }

    /// Return the logical sign at `index`.
    pub fn sign(&self, index: usize) -> Result<i8> {
        if index >= self.projections {
            return Err(TurboQuantError::MalformedCode {
                reason: format!(
                    "sign index {index} is outside projection count {}",
                    self.projections
                ),
            });
        }
        Ok(self.signs[index])
    }

    /// Unpack all logical signs.
    pub fn signs(&self) -> Result<Vec<i8>> {
        self.validate_for(self.dim, self.projections)?;
        Ok(self.signs.clone())
    }

    /// Serialized payload bytes used by this sketch.
    pub fn encoded_bytes(&self) -> usize {
        bitpack::packed_len(self.signs.len(), 1).unwrap_or(usize::MAX)
    }

    /// Validate this sketch against an expected profile.
    pub fn validate_for(&self, dim: usize, projections: usize) -> Result<()> {
        if self.dim != dim {
            return Err(TurboQuantError::DimensionMismatch {
                expected: dim,
                got: self.dim,
            });
        }
        if self.projections != projections {
            return Err(TurboQuantError::MalformedCode {
                reason: format!(
                    "sketch has {} projections, expected {projections}",
                    self.projections
                ),
            });
        }
        if self.signs.len() != projections {
            return Err(TurboQuantError::MalformedCode {
                reason: format!(
                    "sketch has {} signs, expected {projections}",
                    self.signs.len()
                ),
            });
        }
        for (index, sign) in self.signs.iter().enumerate() {
            if *sign != -1 && *sign != 1 {
                return Err(TurboQuantError::MalformedCode {
                    reason: format!("sign {index} is {sign}, expected -1 or 1"),
                });
            }
        }
        Ok(())
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
    projection_matrix: Vec<f32>,
}

/// Query state prepared once for scoring multiple QJL sketches.
#[derive(Debug, Clone, PartialEq)]
pub struct QjlProjectedQuery {
    projected: Vec<f32>,
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
        let projection_matrix = generate_projection_matrix(dim, projections, seed);
        Ok(Self {
            dim,
            projections,
            seed,
            projection_matrix,
        })
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
        check_finite_vector(vector)?;

        let mut signs = Vec::with_capacity(self.projections);

        for row in self.projection_matrix.chunks_exact(self.dim) {
            let dot: f32 = row.iter().zip(vector.iter()).map(|(g, x)| g * x).sum();
            signs.push(if dot >= 0.0 { 1i8 } else { -1i8 });
        }

        QjlSketch::from_parts(self.dim, self.projections, &signs)
    }

    /// Project `vector` and return explicit source-norm provenance.
    pub fn sketch_with_provenance(
        &self,
        vector: &[f32],
        source_digest: Option<String>,
    ) -> Result<(QjlSketch, QjlSketchProvenanceV1)> {
        let sketch = self.sketch(vector)?;
        let source_norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
        let provenance = QjlSketchProvenanceV1::new(source_norm, source_digest)?;
        Ok((sketch, provenance))
    }

    /// Estimate ⟨x, query⟩ from a sketch of x and the raw query vector.
    ///
    /// This legacy method is transport-stable for the `0.1.0` [`QjlSketch`]
    /// shape, which does not store `||x||`. It uses a unit source norm as the
    /// compatibility scale rather than process-local hidden state. Use
    /// [`Self::inner_product_estimate_with_provenance`] when the source norm is
    /// available and should affect scoring.
    pub fn inner_product_estimate(&self, sketch: &QjlSketch, query: &[f32]) -> Result<f32> {
        let projected = self.project_query(query)?;
        self.inner_product_estimate_with_projected_query(sketch, &projected)
    }

    /// Estimate ⟨x, query⟩ using explicit source-norm provenance.
    ///
    /// Returns the QJL sign-projection estimator:
    /// `(sqrt(pi/2) * ||x|| / m) * sum_i signs[i] * (g_i . query)`.
    pub fn inner_product_estimate_with_provenance(
        &self,
        sketch: &QjlSketch,
        provenance: &QjlSketchProvenanceV1,
        query: &[f32],
    ) -> Result<f32> {
        validate_source_norm(provenance.source_norm)?;
        let projected = self.project_query(query)?;
        self.inner_product_estimate_with_projected_query_and_source_norm(
            sketch,
            provenance.source_norm,
            &projected,
        )
    }

    /// Project a query once so it can score multiple sketches without regenerating rows.
    pub fn project_query(&self, query: &[f32]) -> Result<QjlProjectedQuery> {
        if query.len() != self.dim {
            return Err(TurboQuantError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        check_finite_vector(query)?;

        let projected = self
            .projection_matrix
            .chunks_exact(self.dim)
            .map(|row| row.iter().zip(query.iter()).map(|(g, q)| g * q).sum())
            .collect::<Vec<f32>>();
        check_finite_vector(&projected)?;
        Ok(QjlProjectedQuery { projected })
    }

    /// Estimate inner product using a pre-projected query.
    pub fn inner_product_estimate_with_projected_query(
        &self,
        sketch: &QjlSketch,
        query: &QjlProjectedQuery,
    ) -> Result<f32> {
        self.inner_product_estimate_with_projected_query_and_source_norm(sketch, 1.0, query)
    }

    /// Estimate inner product using a pre-projected query and explicit source norm.
    pub fn inner_product_estimate_with_projected_query_and_source_norm(
        &self,
        sketch: &QjlSketch,
        source_norm: f32,
        query: &QjlProjectedQuery,
    ) -> Result<f32> {
        self.validate_sketch(sketch)?;
        validate_source_norm(source_norm)?;
        if query.projected.len() != self.projections {
            return Err(TurboQuantError::MalformedCode {
                reason: format!(
                    "projected query has {} values, expected {}",
                    query.projected.len(),
                    self.projections
                ),
            });
        }

        let m = self.projections as f32;
        let scale = (PI / 2.0).sqrt() * source_norm / m;

        let signs = sketch.signs()?;
        let estimate: f32 = query
            .projected
            .iter()
            .zip(signs.iter())
            .map(|(g_dot_query, &sign)| sign as f32 * g_dot_query)
            .sum();

        let score = scale * estimate;
        if !score.is_finite() {
            return Err(TurboQuantError::MalformedCode {
                reason: "qjl score is not finite".into(),
            });
        }
        Ok(score)
    }

    /// Generate the m×d projection matrix G from seed.
    ///
    /// Entries are i.i.d. N(0, 1). This is deterministic for a given
    /// `(dim, projections, seed)` triple.
    fn validate_sketch(&self, sketch: &QjlSketch) -> Result<()> {
        sketch.validate_for(self.dim, self.projections)
    }
}

fn generate_projection_matrix(dim: usize, projections: usize, seed: u64) -> Vec<f32> {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rand_distr::{Distribution, StandardNormal};

    let mut rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(0xDEAD_BEEF_1234_5678));
    let dist = StandardNormal;

    (0..projections * dim)
        .map(|_| dist.sample(&mut rng))
        .collect()
}

fn check_finite_vector(vector: &[f32]) -> Result<()> {
    if let Some((index, _)) = vector
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(TurboQuantError::NonFiniteInput { index });
    }
    Ok(())
}

fn validate_source_norm(source_norm: f32) -> Result<()> {
    if !source_norm.is_finite() || source_norm < 0.0 {
        return Err(TurboQuantError::MalformedCode {
            reason: format!("QJL source norm must be finite and non-negative, got {source_norm}"),
        });
    }
    Ok(())
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
        for sign in &s.signs().unwrap() {
            assert!(*sign == 1 || *sign == -1, "unexpected sign value: {sign}");
        }
    }

    #[test]
    fn inner_product_estimate_is_close_over_many_samples() {
        // With many projections, the QJL estimator should converge to the true
        // inner product. We use large m to reduce variance for this test.
        let dim = 64;
        let m = 2048;
        let q = QjlQuantizer::new(dim, m, 0).unwrap();

        let x = random_vector(dim, 10);
        let y = random_vector(dim, 20);

        let exact: f32 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        let sketch = q.sketch(&x).unwrap();
        let (_sketch, provenance) = q.sketch_with_provenance(&x, None).unwrap();
        let estimated = q
            .inner_product_estimate_with_provenance(&sketch, &provenance, &y)
            .unwrap();

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
        assert!(
            estimate > 0.0,
            "self inner product should be positive, got {estimate}"
        );
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

    #[test]
    fn legacy_score_is_stable_after_serde_roundtrip() {
        let q = QjlQuantizer::new(16, 64, 42).unwrap();
        let x = random_vector(16, 77);
        let y = random_vector(16, 88);
        let sketch = q.sketch(&x).unwrap();
        let before = q.inner_product_estimate(&sketch, &y).unwrap();
        let json = serde_json::to_string(&sketch).unwrap();
        let restored: QjlSketch = serde_json::from_str(&json).unwrap();
        let after = q.inner_product_estimate(&restored, &y).unwrap();
        assert!((before - after).abs() <= f32::EPSILON);
    }

    #[test]
    fn explicit_provenance_survives_serde_roundtrip() {
        let q = QjlQuantizer::new(16, 256, 42).unwrap();
        let x = random_vector(16, 77);
        let y = random_vector(16, 88);
        let (sketch, provenance) = q
            .sketch_with_provenance(&x, Some("source:test".into()))
            .unwrap();
        let before = q
            .inner_product_estimate_with_provenance(&sketch, &provenance, &y)
            .unwrap();
        let json = serde_json::to_string(&(sketch, provenance)).unwrap();
        let (restored_sketch, restored_provenance): (QjlSketch, QjlSketchProvenanceV1) =
            serde_json::from_str(&json).unwrap();
        let after = q
            .inner_product_estimate_with_provenance(&restored_sketch, &restored_provenance, &y)
            .unwrap();
        assert!((before - after).abs() <= f32::EPSILON);
    }
}
