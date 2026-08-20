#include "turbo_quant.h"
#include <math.h>

/* Fast Walsh-Hadamard Transform, in-place, normalized by 1/sqrt(n).
 *
 * Butterfly: for each step, pair up elements and compute (a+b, a-b).
 * This is the same algorithm as the Rust fwht_normalized, but with
 * explicit loop control for potential auto-vectorization.
 *
 * Rust original archived in src/archive/fwht_rust.rs
 */
void tq_fwht_normalized(float *values, size_t n) {
    size_t step = 1;
    while (step < n) {
        size_t block = step * 2;
        for (size_t start = 0; start < n; start += block) {
            for (size_t offset = 0; offset < step; offset++) {
                float a = values[start + offset];
                float b = values[start + offset + step];
                values[start + offset]       = a + b;
                values[start + offset + step] = a - b;
            }
        }
        step = block;
    }
    float scale = 1.0f / sqrtf((float)n);
    for (size_t i = 0; i < n; i++) {
        values[i] *= scale;
    }
}
