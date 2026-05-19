//! Explicit packed sidecar payloads.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    bitpack,
    error::{Result, TurboQuantError},
    polar::PolarCode,
    qjl::QjlSketch,
    radius::{CompressedRadiiV1, RadiusCodecProfileV1},
    turbo::TurboCode,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct PackedPolarCode {
    pub dim: usize,
    pub bits: u8,
    pub radii: CompressedRadiiV1,
    pub packed_angle_indices: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct PackedQjlSketch {
    pub dim: usize,
    pub projections: usize,
    pub packed_signs: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct PackedTurboCode {
    pub polar_code: PackedPolarCode,
    pub residual_sketch: PackedQjlSketch,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct PackedCompressionStatsV1 {
    pub raw_fp32_bytes: usize,
    pub fp16_baseline_bytes: usize,
    pub legacy_logical_bytes: usize,
    pub packed_sidecar_bytes: usize,
}

impl PackedPolarCode {
    pub fn from_polar(code: &PolarCode, radius_profile: RadiusCodecProfileV1) -> Result<Self> {
        code.validate_for(code.dim, code.bits)?;
        Ok(Self {
            dim: code.dim,
            bits: code.bits,
            radii: CompressedRadiiV1::compress(&code.radii, radius_profile)?,
            packed_angle_indices: bitpack::pack_indices(&code.angle_indices, code.bits)?,
        })
    }

    pub fn unpack(&self) -> Result<PolarCode> {
        let pairs = checked_pairs(self.dim)?;
        let radii = self.radii.decompress()?;
        if radii.len() != pairs {
            return Err(TurboQuantError::MalformedCode {
                reason: format!("packed polar has {} radii, expected {pairs}", radii.len()),
            });
        }
        let angle_indices = bitpack::unpack_indices(&self.packed_angle_indices, pairs, self.bits)?;
        let code = PolarCode {
            dim: self.dim,
            bits: self.bits,
            radii,
            angle_indices,
        };
        code.validate_for(self.dim, self.bits)?;
        Ok(code)
    }

    pub fn encoded_bytes(&self) -> usize {
        self.radii.encoded_bytes() + self.packed_angle_indices.len()
    }
}

impl PackedQjlSketch {
    pub fn from_qjl(sketch: &QjlSketch) -> Result<Self> {
        sketch.validate_for(sketch.dim, sketch.projections)?;
        Ok(Self {
            dim: sketch.dim,
            projections: sketch.projections,
            packed_signs: bitpack::pack_signs(&sketch.signs)?,
        })
    }

    pub fn unpack(&self) -> Result<QjlSketch> {
        let signs = bitpack::unpack_signs(&self.packed_signs, self.projections)?;
        let sketch = QjlSketch {
            dim: self.dim,
            projections: self.projections,
            signs,
        };
        sketch.validate_for(self.dim, self.projections)?;
        Ok(sketch)
    }

    pub fn encoded_bytes(&self) -> usize {
        self.packed_signs.len()
    }
}

impl PackedTurboCode {
    pub fn from_turbo(code: &TurboCode, radius_profile: RadiusCodecProfileV1) -> Result<Self> {
        Ok(Self {
            polar_code: PackedPolarCode::from_polar(&code.polar_code, radius_profile)?,
            residual_sketch: PackedQjlSketch::from_qjl(&code.residual_sketch)?,
        })
    }

    pub fn unpack(&self) -> Result<TurboCode> {
        Ok(TurboCode {
            polar_code: self.polar_code.unpack()?,
            residual_sketch: self.residual_sketch.unpack()?,
        })
    }

    pub fn encoded_bytes(&self) -> usize {
        self.polar_code.encoded_bytes() + self.residual_sketch.encoded_bytes()
    }

    pub fn stats(&self) -> PackedCompressionStatsV1 {
        let dim = self.polar_code.dim;
        let packed_sidecar_bytes = self.encoded_bytes();
        PackedCompressionStatsV1 {
            raw_fp32_bytes: dim * 4,
            fp16_baseline_bytes: dim * 2,
            legacy_logical_bytes: self
                .unpack()
                .map(|code| {
                    code.polar_code.radii.len() * 4
                        + code.polar_code.angle_indices.len() * 2
                        + code.residual_sketch.signs.len()
                })
                .unwrap_or(usize::MAX),
            packed_sidecar_bytes,
        }
    }
}

fn checked_pairs(dim: usize) -> Result<usize> {
    if dim == 0 || dim % 2 != 0 {
        return Err(TurboQuantError::MalformedCode {
            reason: format!("packed polar dimension must be positive and even, got {dim}"),
        });
    }
    Ok(dim / 2)
}
