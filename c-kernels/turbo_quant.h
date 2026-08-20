#ifndef TURBO_QUANT_H
#define TURBO_QUANT_H

#include <stddef.h>
#include <stdint.h>

/* fwht.c — Fast Walsh-Hadamard Transform, normalized */
void tq_fwht_normalized(float *values, size_t n);

/* bitpack.c — bit-level packing/unpacking */
size_t tq_packed_len(size_t count, uint8_t bits);
int tq_pack_indices(const uint16_t *indices, size_t count, uint8_t bits, uint8_t *out);
int tq_unpack_indices(const uint8_t *packed, size_t count, uint8_t bits, uint16_t *out);
int tq_pack_signs(const int8_t *signs, size_t count, uint8_t *out);
int tq_unpack_signs(const uint8_t *packed, size_t count, int8_t *out);

/* qjl.c — Quantized Johnson-Lindenstrauss */
void tq_qjl_sketch(const float *vector, size_t dim,
                   const float *proj_matrix, size_t projections,
                   int8_t *out_signs);
float tq_qjl_ip_estimate(const int8_t *sketch_signs, size_t projections,
                          const float *projected_query);
void tq_qjl_project_query(const float *query, size_t dim,
                          const float *proj_matrix, size_t projections,
                          float *out_projected);

/* polar.c — Polar quantization */
void tq_polar_encode_pair(float a, float b, uint8_t bits,
                          float *out_radius, uint16_t *out_index);
float tq_polar_dequantize_angle(uint16_t angle_index, uint8_t bits);

#endif /* TURBO_QUANT_H */