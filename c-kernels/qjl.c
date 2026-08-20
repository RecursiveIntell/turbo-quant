#include "turbo_quant.h"

/* QJL sketch: project vector against random matrix, take sign of each projection.
 * QJL IP estimate: dot product of projected query with sketch signs.
 * QJL project_query: multiply query by projection matrix (returns floats).
 *
 * Rust original archived in src/archive/qjl_rust.rs
 */

void tq_qjl_sketch(const float *vector, size_t dim,
                   const float *proj_matrix, size_t projections,
                   int8_t *out_signs) {
    for (size_t p = 0; p < projections; p++) {
        const float *row = proj_matrix + p * dim;
        float dot = 0.0f;
        for (size_t d = 0; d < dim; d++) {
            dot += vector[d] * row[d];
        }
        out_signs[p] = (dot >= 0.0f) ? 1 : -1;
    }
}

float tq_qjl_ip_estimate(const int8_t *sketch_signs, size_t projections,
                          const float *projected_query) {
    float sum = 0.0f;
    for (size_t p = 0; p < projections; p++) {
        sum += (float)sketch_signs[p] * projected_query[p];
    }
    return sum;
}

void tq_qjl_project_query(const float *query, size_t dim,
                          const float *proj_matrix, size_t projections,
                          float *out_projected) {
    for (size_t p = 0; p < projections; p++) {
        const float *row = proj_matrix + p * dim;
        float dot = 0.0f;
        for (size_t d = 0; d < dim; d++) {
            dot += query[d] * row[d];
        }
        out_projected[p] = dot;
    }
}