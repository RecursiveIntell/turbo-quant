use turbo_quant::{
    PackedTurboCode, RadiusCodecProfileV1, SearchOptions, TurboQuantizer, TurboSidecarIndex,
};

fn vector(dim: usize, seed: u64) -> Vec<f32> {
    (0..dim)
        .map(|index| ((index as u64 + 17 * seed) as f32 * 0.031).sin())
        .collect()
}

#[test]
fn packed_turbo_roundtrips_to_legacy_shape() {
    let q = TurboQuantizer::new(16, 8, 8, 42).unwrap();
    let code = q.encode(&vector(16, 1)).unwrap();
    let packed = PackedTurboCode::from_turbo(&code, RadiusCodecProfileV1::BlockLinearU16).unwrap();
    let restored = packed.unpack().unwrap();
    assert_eq!(restored.polar_code.dim, code.polar_code.dim);
    assert_eq!(restored.polar_code.bits, code.polar_code.bits);
    assert_eq!(
        restored.polar_code.angle_indices,
        code.polar_code.angle_indices
    );
    assert_eq!(restored.residual_sketch.signs, code.residual_sketch.signs);
    assert!(
        packed.encoded_bytes()
            < code.polar_code.radii.len() * 4
                + code.polar_code.angle_indices.len() * 2
                + code.residual_sketch.signs.len()
    );
}

#[test]
fn packed_malformed_payload_is_rejected() {
    let q = TurboQuantizer::new(16, 8, 8, 42).unwrap();
    let code = q.encode(&vector(16, 2)).unwrap();
    let mut packed = PackedTurboCode::from_turbo(&code, RadiusCodecProfileV1::F32).unwrap();
    packed.polar_code.packed_angle_indices.push(0);
    assert!(packed.unpack().is_err());
}

#[test]
fn sidecar_index_returns_deterministic_candidates_and_receipt() {
    let q = TurboQuantizer::new(16, 8, 8, 42).unwrap();
    let mut index = TurboSidecarIndex::new(q);
    index.add("b", &vector(16, 2), None).unwrap();
    index.add("a", &vector(16, 2), None).unwrap();

    let (candidates, receipt) = index
        .search(
            &vector(16, 2),
            SearchOptions {
                top_k: 2,
                oversample: 1,
            },
        )
        .unwrap();
    assert_eq!(candidates.len(), 2);
    assert_eq!(candidates[0].id, "a");
    assert!(receipt.exact_rerank_required);
    assert_eq!(receipt.byte_accounting.raw_fp32_bytes, 2 * 16 * 4);
}
