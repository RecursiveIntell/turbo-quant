use turbo_quant::{PolarCode, QjlSketch, TurboCode, TurboMode, TurboQuantError, TurboQuantizer};

fn vector(dim: usize) -> Vec<f32> {
    (0..dim).map(|idx| idx as f32 / dim as f32 - 0.5).collect()
}

fn assert_malformed(result: turbo_quant::Result<impl std::fmt::Debug>) {
    match result {
        Err(TurboQuantError::MalformedCode { .. }) => {}
        other => panic!("expected malformed-code error, got {other:?}"),
    }
}

#[test]
fn nan_inf_input_vectors_are_rejected() {
    let q = TurboQuantizer::new(8, 8, 8, 42).unwrap();
    let mut nan = vector(8);
    nan[3] = f32::NAN;
    assert!(matches!(
        q.encode(&nan),
        Err(TurboQuantError::NonFiniteInput { index: 3 })
    ));

    let mut inf = vector(8);
    inf[4] = f32::INFINITY;
    assert!(matches!(
        q.inner_product_estimate(&q.encode(&vector(8)).unwrap(), &inf),
        Err(TurboQuantError::NonFiniteInput { index: 4 })
    ));
}

#[test]
fn polar_code_rejects_nonfinite_and_negative_radius() {
    let mut code = TurboQuantizer::new(8, 8, 8, 42)
        .unwrap()
        .encode(&vector(8))
        .unwrap();

    code.polar_code.radii[0] = f32::NAN;
    assert_malformed(code.validate_for(8, 8, 8, TurboMode::PolarWithQjl));

    code.polar_code.radii[0] = -1.0;
    assert_malformed(code.validate_for(8, 8, 8, TurboMode::PolarWithQjl));
}

#[test]
fn polar_code_rejects_out_of_range_angle() {
    let q = TurboQuantizer::new(8, 4, 8, 42).unwrap();
    let mut code = q.encode(&vector(8)).unwrap();
    code.polar_code.angle_indices[0] = 1 << code.polar_code.bits;
    assert_malformed(q.inner_product_estimate(&code, &vector(8)));
}

#[test]
fn qjl_sketch_rejects_invalid_sign_shape_and_values() {
    let q = TurboQuantizer::new(8, 8, 8, 42).unwrap();
    let mut code = q.encode(&vector(8)).unwrap();

    code.residual_sketch.signs.clear();
    assert_malformed(q.inner_product_estimate(&code, &vector(8)));

    code.residual_sketch.signs = vec![1; 7];
    assert_malformed(q.l2_distance_estimate(&code, &vector(8)));

    code.residual_sketch.signs = vec![1; 8];
    code.residual_sketch.signs[0] = 0;
    assert_malformed(q.inner_product_estimate(&code, &vector(8)));
}

#[test]
fn malformed_turbo_code_rejected_by_all_score_and_decode_paths() {
    let q = TurboQuantizer::new(8, 8, 8, 42).unwrap();
    let query = vector(8);
    let bad = TurboCode {
        polar_code: PolarCode {
            dim: 8,
            bits: 7,
            radii: vec![1.0; 4],
            angle_indices: vec![0; 4],
        },
        residual_sketch: QjlSketch {
            dim: 8,
            projections: 8,
            signs: Vec::new(),
        },
    };

    assert_malformed(q.inner_product_estimate(&bad, &query));
    assert_malformed(q.l2_distance_estimate(&bad, &query));
    assert_malformed(q.decode_approximate(&bad));
}

#[test]
fn prepared_turbo_score_matches_unprepared_score() {
    let q = TurboQuantizer::new(16, 8, 16, 42).unwrap();
    let code = q.encode(&vector(16)).unwrap();
    let query = vector(16);
    let prepared = q.prepare_query(&query).unwrap();

    let unprepared = q.inner_product_estimate(&code, &query).unwrap();
    let prepared_score = q.inner_product_estimate_prepared(&code, &prepared).unwrap();
    assert!((unprepared - prepared_score).abs() <= 1e-5);
}

#[test]
fn turbo_quant_binary_wire_roundtrips_and_rejects_tampering() {
    let q = TurboQuantizer::new(16, 8, 16, 42).unwrap();
    let bytes = q.encode_to_bytes(&vector(16)).unwrap();
    let code = q.decode_code_from_bytes(&bytes).unwrap();
    code.validate_for(16, 8, 16, TurboMode::PolarWithQjl)
        .unwrap();

    let score = q
        .score_inner_product_from_bytes(&bytes, &vector(16))
        .unwrap();
    assert!(score.is_finite());

    let mut bad_magic = bytes.clone();
    bad_magic[0] = b'X';
    assert_malformed(q.decode_code_from_bytes(&bad_magic));

    let mut trailing = bytes.clone();
    trailing.push(0);
    assert_malformed(q.decode_code_from_bytes(&trailing));

    let mut bad_header = bytes;
    bad_header[13] = 7;
    assert_malformed(q.decode_code_from_bytes(&bad_header));
}
