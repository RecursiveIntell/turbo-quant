#include "turbo_quant.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Polar quantization: encode a pair (a,b) into radius and quantized angle index.
 *
 * Matches the Rust encode_pair exactly:
 *   theta = atan2(b, a)              ∈ [−π, π]
 *   normalized = (theta + PI) / (2*PI)   maps to [0, 1)
 *   idx = floor(normalized * levels) % levels
 *
 * Rust original archived in src/archive/polar_rust.rs
 */

void tq_polar_encode_pair(float a, float b, uint8_t bits,
                          float *out_radius, uint16_t *out_index) {
    float r = sqrtf(a * a + b * b);
    float theta = atan2f(b, a);            /* ∈ [−π, π] */
    uint32_t levels = (uint32_t)1 << bits;
    float normalized = (theta + (float)M_PI) / (2.0f * (float)M_PI);
    /* floorf matches Rust .floor(); fmodf matches Rust % (both operate on the
     * float domain — the % levels is a final safety clamp for the boundary
     * case where normalized is exactly 1.0 due to rounding). */
    uint32_t idx = (uint32_t)(floorf(normalized * (float)levels)) % levels;

    *out_radius = r;
    *out_index = (uint16_t)idx;
}

/* Dequantize angle index back to angle value.
 * Matches Rust dequantize_angle exactly:
 *   (idx / levels) * 2*PI - PI     ∈ [−π, π)
 */
float tq_polar_dequantize_angle(uint16_t angle_index, uint8_t bits) {
    uint32_t levels = (uint32_t)1 << bits;
    float idx = (float)angle_index;
    return (idx / (float)levels) * 2.0f * (float)M_PI - (float)M_PI;
}