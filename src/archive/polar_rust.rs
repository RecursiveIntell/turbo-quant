//! Archived Rust implementation of polar hot paths, replaced by C kernels (c-kernels/polar.c).
//! Kept for reference and verification. The C kernels produce identical output.
//!
//! Original location: src/polar.rs
//! Archived: 2026-07-12

#![allow(dead_code)]

use std::f32::consts::PI;

/// Archived: encode_pair — encode (a, b) into (radius, quantized angle index).
///
/// Matches the original Rust:
///   theta = b.atan2(a)                    ∈ [−π, π]
///   normalized = (theta + PI) / (2*PI)     → [0, 1)
///   idx = floor(normalized * levels) % levels
fn encode_pair_rust_archived(a: f32, b: f32, bits: u8) -> (f32, u16) {
    let r = (a * a + b * b).sqrt();
    let theta = b.atan2(a); // ∈ [−π, π]
    let levels = 1u32 << bits;
    // Map [−π, π] → [0, 1) → [0, levels)
    let normalized = (theta + PI) / (2.0 * PI);
    let idx = (normalized * levels as f32).floor() as u32 % levels;
    (r, idx as u16)
}

/// Archived: dequantize_angle — convert angle index back to angle value.
///
/// Matches the original Rust:
///   (idx / levels) * 2*PI - PI    ∈ [−π, π)
fn dequantize_angle_rust_archived(angle_index: u16, bits: u8) -> f32 {
    let levels = 1u32 << bits;
    let idx = angle_index as f32;
    (idx / levels as f32) * (2.0 * PI) - PI
}

/// Archived: encode hot path — rotate, then encode each pair.
fn polar_encode_rust_archived(
    rotated: &[f32],
    dim: usize,
    bits: u8,
) -> (Vec<f32>, Vec<u16>) {
    let pairs = dim / 2;
    let mut radii = Vec::with_capacity(pairs);
    let mut angle_indices = Vec::with_capacity(pairs);

    for i in 0..pairs {
        let a = rotated[2 * i];
        let b = rotated[2 * i + 1];
        let (r, idx) = encode_pair_rust_archived(a, b, bits);
        radii.push(r);
        angle_indices.push(idx);
    }
    (radii, angle_indices)
}

/// Archived: decode hot path — reconstruct rotated vector from radii + angle indices.
fn polar_decode_rust_archived(
    radii: &[f32],
    angle_indices: &[u16],
    dim: usize,
    bits: u8,
) -> Vec<f32> {
    let pairs = dim / 2;
    let mut rotated = vec![0.0f32; dim];

    for i in 0..pairs {
        let theta = dequantize_angle_rust_archived(angle_indices[i], bits);
        let r = radii[i];
        rotated[2 * i] = r * theta.cos();
        rotated[2 * i + 1] = r * theta.sin();
    }
    rotated
}