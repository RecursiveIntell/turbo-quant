//! Byte accounting helpers for sidecar and shadow-mode receipts.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ByteAccountingV1 {
    pub vector_count: usize,
    pub dim: usize,
    pub raw_fp32_bytes: usize,
    pub fp16_baseline_bytes: usize,
    pub sidecar_bytes: usize,
    pub exact_shadow_bytes: usize,
    pub resident_bytes: usize,
}

impl ByteAccountingV1 {
    pub fn new(
        vector_count: usize,
        dim: usize,
        sidecar_bytes: usize,
        keep_exact_shadow: bool,
    ) -> Self {
        let raw_fp32_bytes = vector_count * dim * 4;
        let fp16_baseline_bytes = vector_count * dim * 2;
        let exact_shadow_bytes = if keep_exact_shadow { raw_fp32_bytes } else { 0 };
        Self {
            vector_count,
            dim,
            raw_fp32_bytes,
            fp16_baseline_bytes,
            sidecar_bytes,
            exact_shadow_bytes,
            resident_bytes: sidecar_bytes + exact_shadow_bytes,
        }
    }
}
