//! Archived Rust implementation of fwht_normalized, replaced by C kernel (c-kernels/fwht.c).
//! Kept for reference and verification. The C kernel produces identical output.
//!
//! Original location: src/rotation.rs
//! Archived: 2026-07-12

#[allow(dead_code)]
fn fwht_normalized_rust_archived(values: &mut [f32]) {
    let n = values.len();
    let mut step = 1;
    while step < n {
        let block = step * 2;
        for start in (0..n).step_by(block) {
            for offset in 0..step {
                let a = values[start + offset];
                let b = values[start + offset + step];
                values[start + offset] = a + b;
                values[start + offset + step] = a - b;
            }
        }
        step = block;
    }
    let scale = (n as f32).sqrt().recip();
    for value in values {
        *value *= scale;
    }
}