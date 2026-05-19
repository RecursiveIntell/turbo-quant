use turbo_quant::{CompressedToken, KvCacheConfig, PolarCode, QjlSketch, TurboCode};

#[test]
fn legacy_0_1_public_struct_literals_compile() {
    let polar = PolarCode {
        dim: 4,
        bits: 8,
        radii: vec![1.0, 1.0],
        angle_indices: vec![0, 1],
    };
    let qjl = QjlSketch {
        dim: 4,
        projections: 2,
        signs: vec![1, -1],
    };
    let turbo = TurboCode {
        polar_code: polar,
        residual_sketch: qjl,
    };
    let _token = CompressedToken {
        compressed_key: turbo.clone(),
        compressed_value: turbo,
    };
    let _config = KvCacheConfig {
        head_dim: 4,
        bits: 8,
        projections: 2,
        seed: 42,
    };
}
