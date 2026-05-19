use turbo_quant::{
    CompressedToken, KvCacheCompressor, KvCacheConfig, PolarCode, PolarQuantizer, QjlQuantizer,
    QjlSketch, TurboCode, TurboQuantizer,
};

fn main() -> Result<(), turbo_quant::TurboQuantError> {
    let dim = 4;
    let vector = vec![0.1_f32, 0.2, 0.3, 0.4];
    let query = vec![0.2_f32, 0.1, 0.4, 0.3];

    let polar_quantizer = PolarQuantizer::new(dim, 8, 42)?;
    let polar_code = polar_quantizer.encode(&vector)?;
    let _polar_score = polar_quantizer.inner_product_estimate(&polar_code, &query)?;

    let qjl_quantizer = QjlQuantizer::new(dim, 2, 42)?;
    let qjl_sketch = qjl_quantizer.sketch(&vector)?;
    let _qjl_score = qjl_quantizer.inner_product_estimate(&qjl_sketch, &query)?;

    let manual_polar = PolarCode {
        dim,
        bits: 8,
        radii: vec![1.0, 1.0],
        angle_indices: vec![0, 1],
    };
    let manual_qjl = QjlSketch {
        dim,
        projections: 2,
        signs: vec![1, -1],
    };
    let manual_turbo = TurboCode {
        polar_code: manual_polar,
        residual_sketch: manual_qjl,
    };
    let _manual_token = CompressedToken {
        compressed_key: manual_turbo.clone(),
        compressed_value: manual_turbo,
    };

    let turbo_quantizer = TurboQuantizer::new(dim, 8, 2, 42)?;
    let turbo_code = turbo_quantizer.encode(&vector)?;
    let _turbo_score = turbo_quantizer.inner_product_estimate(&turbo_code, &query)?;

    let config = KvCacheConfig {
        head_dim: dim,
        bits: 8,
        projections: 2,
        seed: 42,
    };
    let mut cache = KvCacheCompressor::new(config)?;
    cache.compress_token(&vector, &query)?;
    let scores = cache.attention_scores(&query)?;
    assert_eq!(scores.len(), 1);
    let values = cache.decode_values(&[0])?;
    assert_eq!(values.len(), 1);

    Ok(())
}
