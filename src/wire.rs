//! Deterministic compact wire encoding for [`TurboCode`].
//!
//! The wire bytes are a derived acceleration artifact. They are bound to the
//! quantizer profile on decode and are never authoritative over raw f32 vectors.

use crate::{
    bitpack,
    error::{Result, TurboQuantError},
    polar::PolarCode,
    qjl::QjlSketch,
    rotation::RotationKind,
    turbo::{TurboCode, TurboMode, TurboQuantizer},
};

/// Magic bytes for TurboCode wire format v1.
pub const TURBO_CODE_WIRE_MAGIC: &[u8; 4] = b"TQW1";

const VERSION: u16 = 1;
const VARIANT_TURBO_CODE: u8 = 1;

/// Encoder/decoder for TurboCode wire format v1.
pub struct TurboCodeWireV1;

impl TurboCodeWireV1 {
    /// Encode a validated TurboCode using the supplied quantizer profile.
    pub fn encode(code: &TurboCode, profile: &TurboQuantizer) -> Result<Vec<u8>> {
        code.validate_for(
            profile.dim(),
            profile.bits(),
            profile.projections(),
            profile.mode(),
        )?;

        let dim = checked_u32(profile.dim(), "dimension")?;
        let polar_bits = code.polar_code.bits;
        let qjl_projections = checked_u32(profile.projections(), "projection count")?;
        let polar_block_count = checked_u32(code.polar_code.radii.len(), "polar block count")?;
        let qjl_sign_count = checked_u32(
            match profile.mode() {
                TurboMode::PolarOnly => 0,
                TurboMode::PolarWithQjl => code.residual_sketch.projections,
            },
            "qjl sign count",
        )?;
        let packed_angle_indices =
            bitpack::pack_indices(&code.polar_code.angle_indices, polar_bits)?;
        let packed_signs = match profile.mode() {
            TurboMode::PolarOnly => Vec::new(),
            TurboMode::PolarWithQjl => bitpack::pack_signs(&code.residual_sketch.signs)?,
        };
        let payload_len = checked_u64(
            code.polar_code.radii.len() * 4 + packed_angle_indices.len() + packed_signs.len(),
            "payload length",
        )?;

        let mut bytes = Vec::with_capacity(42 + payload_len as usize);
        bytes.extend_from_slice(TURBO_CODE_WIRE_MAGIC);
        bytes.extend_from_slice(&VERSION.to_le_bytes());
        bytes.extend_from_slice(&rotation_flag(profile.rotation_kind()).to_le_bytes());
        bytes.push(VARIANT_TURBO_CODE);
        bytes.push(0);
        bytes.extend_from_slice(&dim.to_le_bytes());
        bytes.push(polar_bits);
        bytes.extend_from_slice(&[0, 0, 0]);
        bytes.extend_from_slice(&qjl_projections.to_le_bytes());
        bytes.extend_from_slice(&profile.seed().to_le_bytes());
        bytes.extend_from_slice(&polar_block_count.to_le_bytes());
        bytes.extend_from_slice(&qjl_sign_count.to_le_bytes());
        bytes.extend_from_slice(&payload_len.to_le_bytes());

        for radius in &code.polar_code.radii {
            bytes.extend_from_slice(&radius.to_le_bytes());
        }
        bytes.extend_from_slice(&packed_angle_indices);
        bytes.extend_from_slice(&packed_signs);
        Ok(bytes)
    }

    /// Decode and validate TurboCode wire bytes against the supplied profile.
    pub fn decode(bytes: &[u8], profile: &TurboQuantizer) -> Result<TurboCode> {
        let mut cursor = WireCursor::new(bytes);
        if cursor.read_exact(TURBO_CODE_WIRE_MAGIC.len())? != TURBO_CODE_WIRE_MAGIC {
            return Err(TurboQuantError::MalformedCode {
                reason: "wrong TurboQuant wire magic".into(),
            });
        }
        let version = cursor.read_u16()?;
        if version != VERSION {
            return Err(TurboQuantError::MalformedCode {
                reason: format!("unsupported TurboQuant wire version {version}"),
            });
        }
        let wire_rotation_flag = cursor.read_u16()?;
        let expected_rotation_flag = rotation_flag(profile.rotation_kind());
        if wire_rotation_flag != expected_rotation_flag {
            return Err(TurboQuantError::MalformedCode {
                reason: format!(
                    "wire rotation flag {wire_rotation_flag} does not match quantizer profile flag {expected_rotation_flag}"
                ),
            });
        }
        let variant = cursor.read_u8()?;
        if variant != VARIANT_TURBO_CODE {
            return Err(TurboQuantError::MalformedCode {
                reason: format!("unsupported TurboQuant wire variant {variant}"),
            });
        }
        let reserved = cursor.read_u8()?;
        if reserved != 0 {
            return Err(TurboQuantError::MalformedCode {
                reason: "nonzero TurboQuant wire reserved byte".into(),
            });
        }

        let dim = cursor.read_u32()? as usize;
        let polar_bits = cursor.read_u8()?;
        let reserved2 = cursor.read_exact(3)?;
        if reserved2 != [0, 0, 0] {
            return Err(TurboQuantError::MalformedCode {
                reason: "nonzero TurboQuant wire reserved bytes".into(),
            });
        }
        let qjl_projections = cursor.read_u32()? as usize;
        let seed = cursor.read_u64()?;
        let polar_block_count = cursor.read_u32()? as usize;
        let qjl_sign_count = cursor.read_u32()? as usize;
        let payload_len = cursor.read_u64()?;
        let payload_start = cursor.offset();

        let expected_polar_bits = match profile.mode() {
            TurboMode::PolarOnly => profile.bits(),
            TurboMode::PolarWithQjl => profile.bits() - 1,
        };
        if dim != profile.dim()
            || polar_bits != expected_polar_bits
            || qjl_projections != profile.projections()
        {
            return Err(TurboQuantError::MalformedCode {
                reason: "wire header does not match quantizer profile".into(),
            });
        }
        if seed != profile.seed() {
            return Err(TurboQuantError::MalformedCode {
                reason: format!(
                    "wire seed {seed} does not match quantizer profile seed {}",
                    profile.seed()
                ),
            });
        }
        if polar_block_count != profile.dim() / 2 {
            return Err(TurboQuantError::MalformedCode {
                reason: format!(
                    "wire polar block count {polar_block_count} does not match dimension {}",
                    profile.dim()
                ),
            });
        }
        let expected_qjl_sign_count = match profile.mode() {
            TurboMode::PolarOnly => 0,
            TurboMode::PolarWithQjl => profile.projections(),
        };
        if qjl_sign_count != expected_qjl_sign_count {
            return Err(TurboQuantError::MalformedCode {
                reason: format!(
                    "wire sign count {qjl_sign_count} does not match expected {expected_qjl_sign_count}"
                ),
            });
        }
        let angle_bytes = bitpack::packed_len(polar_block_count, polar_bits)?;
        let sign_bytes = match profile.mode() {
            TurboMode::PolarOnly => 0,
            TurboMode::PolarWithQjl => profile.projections().div_ceil(8),
        };
        let residual_bytes = sign_bytes;
        let expected_payload_len = checked_u64(
            polar_block_count * 4 + angle_bytes + residual_bytes,
            "expected payload length",
        )?;
        if payload_len != expected_payload_len {
            return Err(TurboQuantError::MalformedCode {
                reason: format!(
                    "TurboQuant wire payload length {payload_len} does not match expected {expected_payload_len}"
                ),
            });
        }
        if payload_len > cursor.remaining_len() as u64 {
            return Err(TurboQuantError::MalformedCode {
                reason: "TurboQuant wire payload length exceeds remaining bytes".into(),
            });
        }

        let mut radii = Vec::with_capacity(polar_block_count);
        for _ in 0..polar_block_count {
            radii.push(cursor.read_f32()?);
        }
        let packed_angle_indices = cursor.read_exact(angle_bytes)?.to_vec();
        let angle_indices =
            bitpack::unpack_indices(&packed_angle_indices, polar_block_count, polar_bits)?;
        let residual_sketch = match profile.mode() {
            TurboMode::PolarOnly => QjlSketch {
                dim: profile.dim(),
                projections: 0,
                signs: Vec::new(),
            },
            TurboMode::PolarWithQjl => {
                let packed_signs = cursor.read_exact(sign_bytes)?.to_vec();
                let signs = bitpack::unpack_signs(&packed_signs, profile.projections())?;
                QjlSketch {
                    dim: profile.dim(),
                    projections: profile.projections(),
                    signs,
                }
            }
        };
        if cursor.offset() - payload_start != payload_len as usize {
            return Err(TurboQuantError::MalformedCode {
                reason: "TurboQuant wire payload length mismatch".into(),
            });
        }
        cursor.finish()?;

        let code = TurboCode {
            polar_code: PolarCode {
                dim: profile.dim(),
                bits: polar_bits,
                radii,
                angle_indices,
            },
            residual_sketch,
        };
        code.validate_for(
            profile.dim(),
            profile.bits(),
            profile.projections(),
            profile.mode(),
        )?;
        Ok(code)
    }
}

fn checked_u32(value: usize, field: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| TurboQuantError::MalformedCode {
        reason: format!("{field} {value} does not fit u32 wire field"),
    })
}

fn checked_u64(value: usize, field: &str) -> Result<u64> {
    u64::try_from(value).map_err(|_| TurboQuantError::MalformedCode {
        reason: format!("{field} {value} does not fit u64 wire field"),
    })
}

fn rotation_flag(kind: RotationKind) -> u16 {
    match kind {
        RotationKind::Auto => 0,
        RotationKind::FastHadamard => 1,
        RotationKind::StoredQr => 2,
    }
}

struct WireCursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

/// Decoded TurboQuant wire header. The wire format carries the full
/// quantizer profile (dim, bits, projections, seed, mode, rotation kind)
/// in the first 44 bytes, so a `TurboCode` can be reconstructed from
/// the wire bytes alone — no external quantizer required.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TurboCodeWireHeader {
    /// Original vector dimension.
    pub dim: usize,
    /// Polar-code bits per angle (b in the paper; b-1 for PolarWithQjl mode).
    pub polar_bits: u8,
    /// QJL projection count for the residual sketch.
    pub qjl_projections: usize,
    /// Seed used to derive the projection state.
    pub seed: u64,
    /// Number of polar code blocks (≈ dim / 2).
    pub polar_block_count: usize,
    /// QJL sign count (0 for PolarOnly mode).
    pub qjl_sign_count: usize,
    /// Length of the payload section following the header.
    pub payload_len: u64,
    /// Rotation kind embedded in the wire.
    pub rotation_kind: RotationKind,
}

impl TurboCodeWireV1 {
    /// Parse just the 44-byte wire header. This is the public entry point
    /// for callers that want to extract the quantizer profile from the
    /// wire format without validating against a specific quantizer instance.
    pub fn parse_header(bytes: &[u8]) -> Result<TurboCodeWireHeader> {
        if bytes.len() < 44 {
            return Err(TurboQuantError::MalformedCode {
                reason: format!("TurboQuant wire header is {} bytes, need 44", bytes.len()),
            });
        }
        if &bytes[0..4] != TURBO_CODE_WIRE_MAGIC {
            return Err(TurboQuantError::MalformedCode {
                reason: "wrong TurboQuant wire magic".into(),
            });
        }
        let version = u16::from_le_bytes(bytes[4..6].try_into().unwrap());
        if version != VERSION {
            return Err(TurboQuantError::MalformedCode {
                reason: format!("unsupported TurboQuant wire version {version}"),
            });
        }
        let wire_rotation_flag = u16::from_le_bytes(bytes[6..8].try_into().unwrap());
        let rotation_kind = match wire_rotation_flag {
            0 => RotationKind::Auto,
            1 => RotationKind::FastHadamard,
            2 => RotationKind::StoredQr,
            _ => {
                return Err(TurboQuantError::MalformedCode {
                    reason: format!("unknown TurboQuant rotation flag {wire_rotation_flag}"),
                })
            }
        };
        let variant = bytes[8];
        if variant != VARIANT_TURBO_CODE {
            return Err(TurboQuantError::MalformedCode {
                reason: format!("unsupported TurboQuant wire variant {variant}"),
            });
        }
        let reserved = bytes[9];
        if reserved != 0 {
            return Err(TurboQuantError::MalformedCode {
                reason: "nonzero TurboQuant wire reserved byte".into(),
            });
        }
        let dim = u32::from_le_bytes(bytes[10..14].try_into().unwrap()) as usize;
        let polar_bits = bytes[14];
        let reserved2: [u8; 3] = bytes[15..18].try_into().unwrap();
        if reserved2 != [0, 0, 0] {
            return Err(TurboQuantError::MalformedCode {
                reason: "nonzero TurboQuant wire reserved bytes".into(),
            });
        }
        let qjl_projections = u32::from_le_bytes(bytes[18..22].try_into().unwrap()) as usize;
        let seed = u64::from_le_bytes(bytes[22..30].try_into().unwrap());
        let polar_block_count = u32::from_le_bytes(bytes[30..34].try_into().unwrap()) as usize;
        let qjl_sign_count = u32::from_le_bytes(bytes[34..38].try_into().unwrap()) as usize;
        let payload_len = u64::from_le_bytes(bytes[38..46].try_into().unwrap());
        Ok(TurboCodeWireHeader {
            dim,
            polar_bits,
            qjl_projections,
            seed,
            polar_block_count,
            qjl_sign_count,
            payload_len,
            rotation_kind,
        })
    }
}

impl<'a> WireCursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn offset(&self) -> usize {
        self.offset
    }

    fn remaining_len(&self) -> usize {
        self.bytes.len().saturating_sub(self.offset)
    }

    fn read_exact(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or_else(|| TurboQuantError::MalformedCode {
                reason: "wire offset overflow".into(),
            })?;
        if end > self.bytes.len() {
            return Err(TurboQuantError::MalformedCode {
                reason: "truncated TurboQuant wire artifact".into(),
            });
        }
        let out = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(out)
    }

    fn read_u8(&mut self) -> Result<u8> {
        Ok(self.read_exact(1)?[0])
    }

    fn read_u16(&mut self) -> Result<u16> {
        let bytes = self.read_exact(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn read_u32(&mut self) -> Result<u32> {
        let bytes = self.read_exact(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_u64(&mut self) -> Result<u64> {
        let bytes = self.read_exact(8)?;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn read_f32(&mut self) -> Result<f32> {
        let bytes = self.read_exact(4)?;
        Ok(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn finish(&self) -> Result<()> {
        if self.offset != self.bytes.len() {
            return Err(TurboQuantError::MalformedCode {
                reason: "trailing bytes in TurboQuant wire artifact".into(),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_quantizer(dim: usize, seed: u64) -> TurboQuantizer {
        // Use the simplest possible profile: PolarWithQjl, 8-bit, 32 projections.
        TurboQuantizer::new(dim, 8, 32, seed).expect("quantizer build")
    }

    #[test]
    fn parse_header_round_trips_encoded_code() {
        let q = make_quantizer(128, 42);
        let vector: Vec<f32> = (0..128).map(|i| (i as f32 / 128.0) - 0.5).collect();
        let code = q.encode(&vector).expect("encode");
        let wire = TurboCodeWireV1::encode(&code, &q).expect("wire encode");

        let header = TurboCodeWireV1::parse_header(&wire).expect("parse header");
        assert_eq!(header.dim, 128);
        assert_eq!(header.qjl_projections, 32);
        assert_eq!(header.seed, 42);
        assert!(header.polar_block_count > 0);
        assert!(header.payload_len > 0);
    }

    #[test]
    fn parse_header_rejects_short_buffer() {
        let bytes = vec![0u8; 10];
        let result = TurboCodeWireV1::parse_header(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn parse_header_rejects_bad_magic() {
        let mut bytes = vec![0u8; 44];
        bytes[0..4].copy_from_slice(b"XXXX");
        let result = TurboCodeWireV1::parse_header(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn parse_header_rejects_unsupported_version() {
        let mut bytes = vec![0u8; 44];
        bytes[0..4].copy_from_slice(TURBO_CODE_WIRE_MAGIC);
        // version = 99 (unsupported)
        bytes[4..6].copy_from_slice(&99u16.to_le_bytes());
        let result = TurboCodeWireV1::parse_header(&bytes);
        assert!(result.is_err());
    }
}
