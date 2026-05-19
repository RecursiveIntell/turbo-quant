use turbo_quant::{PolarCode, QjlSketch, TurboCode, TurboQuantizer};

fn valid_code() -> (TurboQuantizer, TurboCode) {
    let q = TurboQuantizer::new(8, 8, 8, 42).unwrap();
    let vector = (0..8).map(|i| i as f32 * 0.125 + 0.1).collect::<Vec<_>>();
    let code = q.encode(&vector).unwrap();
    (q, code)
}

#[test]
fn negative_polar_radius_rejected() {
    let (q, mut code) = valid_code();
    code.polar_code.radii[0] = -1.0;
    assert!(q.inner_product_estimate(&code, &[0.1; 8]).is_err());
}

#[test]
fn nan_polar_radius_rejected() {
    let (q, mut code) = valid_code();
    code.polar_code.radii[0] = f32::NAN;
    assert!(q.inner_product_estimate(&code, &[0.1; 8]).is_err());
}

#[test]
fn infinite_polar_radius_rejected() {
    let (q, mut code) = valid_code();
    code.polar_code.radii[0] = f32::INFINITY;
    assert!(q.inner_product_estimate(&code, &[0.1; 8]).is_err());
}

#[test]
fn out_of_range_polar_angle_rejected() {
    let (q, mut code) = valid_code();
    code.polar_code.angle_indices[0] = 1 << code.polar_code.bits;
    assert!(q.inner_product_estimate(&code, &[0.1; 8]).is_err());
}

#[test]
fn qjl_invalid_sign_rejected() {
    let q = TurboQuantizer::new(8, 8, 9, 42).unwrap();
    let mut code = q
        .encode(&(0..8).map(|i| i as f32 * 0.125 + 0.1).collect::<Vec<_>>())
        .unwrap();
    code.residual_sketch.signs[0] = 0;
    assert!(q.inner_product_estimate(&code, &[0.1; 8]).is_err());
}

#[test]
fn qjl_sign_length_rejected() {
    let (q, mut code) = valid_code();
    code.residual_sketch.signs.clear();
    assert!(q.inner_product_estimate(&code, &[0.1; 8]).is_err());
}

#[test]
fn query_nan_rejected() {
    let (q, code) = valid_code();
    let mut query = vec![0.1; 8];
    query[0] = f32::NAN;
    assert!(q.inner_product_estimate(&code, &query).is_err());
}

#[test]
fn query_infinity_rejected() {
    let (q, code) = valid_code();
    let mut query = vec![0.1; 8];
    query[0] = f32::INFINITY;
    assert!(q.inner_product_estimate(&code, &query).is_err());
}

#[test]
fn mismatched_polar_residual_dimensions_rejected() {
    let (q, mut code) = valid_code();
    code.residual_sketch.dim = 16;
    assert!(q.inner_product_estimate(&code, &[0.1; 8]).is_err());
}

#[test]
fn direct_malformed_shapes_rejected() {
    let q = TurboQuantizer::new(8, 8, 8, 42).unwrap();
    let code = TurboCode {
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
    assert!(q.inner_product_estimate(&code, &[0.1; 8]).is_err());
}
