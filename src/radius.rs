//! Radius compression profiles for packed sidecar payloads.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::{Result, TurboQuantError};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub enum RadiusCodecProfileV1 {
    F32,
    BlockLinearU16,
    BlockLogU8,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct CompressedRadiiV1 {
    pub profile: RadiusCodecProfileV1,
    pub count: usize,
    pub min: f32,
    pub max: f32,
    pub payload: Vec<u8>,
}

impl CompressedRadiiV1 {
    pub fn compress(radii: &[f32], profile: RadiusCodecProfileV1) -> Result<Self> {
        validate_radii(radii)?;
        match profile {
            RadiusCodecProfileV1::F32 => {
                let mut payload = Vec::with_capacity(radii.len() * 4);
                for radius in radii {
                    payload.extend_from_slice(&radius.to_le_bytes());
                }
                Ok(Self {
                    profile,
                    count: radii.len(),
                    min: 0.0,
                    max: 0.0,
                    payload,
                })
            }
            RadiusCodecProfileV1::BlockLinearU16 => {
                let (min, max) = min_max(radii);
                let mut payload = Vec::with_capacity(radii.len() * 2);
                let span = (max - min).max(f32::EPSILON);
                for radius in radii {
                    let normalized = ((*radius - min) / span).clamp(0.0, 1.0);
                    let quantized = (normalized * u16::MAX as f32).round() as u16;
                    payload.extend_from_slice(&quantized.to_le_bytes());
                }
                Ok(Self {
                    profile,
                    count: radii.len(),
                    min,
                    max,
                    payload,
                })
            }
            RadiusCodecProfileV1::BlockLogU8 => {
                let logged = radii
                    .iter()
                    .map(|value| value.max(f32::MIN_POSITIVE).ln())
                    .collect::<Vec<f32>>();
                let (min, max) = min_max(&logged);
                let mut payload = Vec::with_capacity(radii.len());
                let span = (max - min).max(f32::EPSILON);
                for value in &logged {
                    let normalized = ((*value - min) / span).clamp(0.0, 1.0);
                    payload.push((normalized * u8::MAX as f32).round() as u8);
                }
                Ok(Self {
                    profile,
                    count: radii.len(),
                    min,
                    max,
                    payload,
                })
            }
        }
    }

    pub fn decompress(&self) -> Result<Vec<f32>> {
        match self.profile {
            RadiusCodecProfileV1::F32 => {
                if self.payload.len() != self.count * 4 {
                    return Err(TurboQuantError::MalformedCode {
                        reason: format!(
                            "f32 radius payload has {} bytes, expected {}",
                            self.payload.len(),
                            self.count * 4
                        ),
                    });
                }
                self.payload
                    .chunks_exact(4)
                    .enumerate()
                    .map(|(index, chunk)| {
                        let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                        if !value.is_finite() || value < 0.0 {
                            return Err(TurboQuantError::MalformedCode {
                                reason: format!("radius {index} is not finite and non-negative"),
                            });
                        }
                        Ok(value)
                    })
                    .collect()
            }
            RadiusCodecProfileV1::BlockLinearU16 => {
                if self.payload.len() != self.count * 2 {
                    return Err(TurboQuantError::MalformedCode {
                        reason: format!(
                            "linear u16 radius payload has {} bytes, expected {}",
                            self.payload.len(),
                            self.count * 2
                        ),
                    });
                }
                validate_range(self.min, self.max)?;
                let span = self.max - self.min;
                Ok(self
                    .payload
                    .chunks_exact(2)
                    .map(|chunk| {
                        let value = u16::from_le_bytes([chunk[0], chunk[1]]) as f32;
                        self.min + span * (value / u16::MAX as f32)
                    })
                    .collect())
            }
            RadiusCodecProfileV1::BlockLogU8 => {
                if self.payload.len() != self.count {
                    return Err(TurboQuantError::MalformedCode {
                        reason: format!(
                            "log u8 radius payload has {} bytes, expected {}",
                            self.payload.len(),
                            self.count
                        ),
                    });
                }
                validate_range(self.min, self.max)?;
                let span = self.max - self.min;
                Ok(self
                    .payload
                    .iter()
                    .map(|value| (self.min + span * (*value as f32 / u8::MAX as f32)).exp())
                    .collect())
            }
        }
    }

    pub fn encoded_bytes(&self) -> usize {
        match self.profile {
            RadiusCodecProfileV1::F32 => self.payload.len(),
            RadiusCodecProfileV1::BlockLinearU16 | RadiusCodecProfileV1::BlockLogU8 => {
                self.payload.len() + 8
            }
        }
    }
}

fn validate_radii(radii: &[f32]) -> Result<()> {
    for (index, radius) in radii.iter().enumerate() {
        if !radius.is_finite() || *radius < 0.0 {
            return Err(TurboQuantError::MalformedCode {
                reason: format!("radius {index} is not finite and non-negative"),
            });
        }
    }
    Ok(())
}

fn validate_range(min: f32, max: f32) -> Result<()> {
    if !min.is_finite() || !max.is_finite() || min > max {
        return Err(TurboQuantError::MalformedCode {
            reason: "radius codec range is malformed".into(),
        });
    }
    Ok(())
}

fn min_max(values: &[f32]) -> (f32, f32) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    values
        .iter()
        .copied()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), value| {
            (min.min(value), max.max(value))
        })
}
