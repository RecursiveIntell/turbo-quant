//! Archived Rust implementation of QJL hot paths, replaced by C kernels (c-kernels/qjl.c).
//! Kept for reference and verification. The C kernels produce identical output.
//!
//! Original location: src/qjl.rs
//! Archived: 2026-07-12

#![allow(dead_code)]

use std::f32::consts::PI;

/// Archived: sketch hot path — project vector against matrix, take sign of each projection.
fn qjl_sketch_rust_archived(
    vector: &[f32],
    dim: usize,
    projection_matrix: &[f32],
    projections: usize,
) -> Vec<i8> {
    let mut signs = Vec::with_capacity(projections);
    for row in projection_matrix.chunks_exact(dim) {
        let dot: f32 = row.iter().zip(vector.iter()).map(|(g, x)| g * x).sum();
        signs.push(if dot >= 0.0 { 1i8 } else { -1i8 });
    }
    signs
}

/// Archived: project_query hot path — multiply query by projection matrix.
fn qjl_project_query_rust_archived(
    query: &[f32],
    dim: usize,
    projection_matrix: &[f32],
    projections: usize,
) -> Vec<f32> {
    projection_matrix
        .chunks_exact(dim)
        .map(|row| row.iter().zip(query.iter()).map(|(g, q)| g * q).sum())
        .collect::<Vec<f32>>()
}

/// Archived: inner_product_estimate hot path — dot product of signs with projected query.
fn qjl_ip_estimate_rust_archived(
    signs: &[i8],
    projected_query: &[f32],
    projections: usize,
    source_norm: f32,
) -> f32 {
    let m = projections as f32;
    let scale = (PI / 2.0).sqrt() * source_norm / m;
    let estimate: f32 = projected_query
        .iter()
        .zip(signs.iter())
        .map(|(g_dot_query, &sign)| sign as f32 * g_dot_query)
        .sum();
    scale * estimate
}