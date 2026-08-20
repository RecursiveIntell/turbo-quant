#include "turbo_quant.h"
#include <string.h>

/* Bitpacking helpers — pack/unpack integer indices and signs into bytes.
 *
 * Packing order is little-endian within each byte: first logical value
 * starts at bit 0 of byte 0.
 *
 * Rust original archived in src/archive/bitpack_rust.rs
 */

static int validate_bits(uint8_t bits) {
    return bits > 0 && bits <= 16;
}

static uint32_t levels_for_bits(uint8_t bits) {
    return (uint32_t)1 << bits;
}

size_t tq_packed_len(size_t count, uint8_t bits) {
    if (!validate_bits(bits)) return 0;
    size_t total_bits = count * bits;
    return (total_bits + 7) / 8;
}

static void write_bits(uint8_t *bytes, size_t bit_offset, uint8_t bits, uint32_t value) {
    size_t byte_idx = bit_offset / 8;
    uint8_t bit_idx = (uint8_t)(bit_offset % 8);
    uint8_t remaining = bits;

    while (remaining > 0) {
        uint8_t avail = 8 - bit_idx;
        uint8_t to_write = remaining < avail ? remaining : avail;
        uint8_t mask = (uint8_t)((1 << to_write) - 1);
        uint8_t val_bits = (uint8_t)(value & mask);
        bytes[byte_idx] |= (val_bits << bit_idx);
        value >>= to_write;
        bit_idx += to_write;
        remaining -= to_write;
        if (bit_idx == 8) {
            bit_idx = 0;
            byte_idx++;
        }
    }
}

static uint32_t read_bits(const uint8_t *bytes, size_t bit_offset, uint8_t bits) {
    size_t byte_idx = bit_offset / 8;
    uint8_t bit_idx = (uint8_t)(bit_offset % 8);
    uint32_t result = 0;
    uint8_t shift = 0;
    uint8_t remaining = bits;

    while (remaining > 0) {
        uint8_t avail = 8 - bit_idx;
        uint8_t to_read = remaining < avail ? remaining : avail;
        uint8_t mask = (uint8_t)((1 << to_read) - 1);
        uint8_t val_bits = (uint8_t)((bytes[byte_idx] >> bit_idx) & mask);
        result |= (uint32_t)val_bits << shift;
        shift += to_read;
        bit_idx += to_read;
        remaining -= to_read;
        if (bit_idx == 8) {
            bit_idx = 0;
            byte_idx++;
        }
    }
    return result;
}

int tq_pack_indices(const uint16_t *indices, size_t count, uint8_t bits, uint8_t *out) {
    if (!validate_bits(bits)) return -1;
    uint32_t levels = levels_for_bits(bits);
    size_t packed_len = tq_packed_len(count, bits);
    if (packed_len == 0 && count > 0) return -1;
    memset(out, 0, packed_len);
    for (size_t i = 0; i < count; i++) {
        if ((uint32_t)indices[i] >= levels) return -1;
        write_bits(out, i * bits, bits, (uint32_t)indices[i]);
    }
    return 0;
}

int tq_unpack_indices(const uint8_t *packed, size_t count, uint8_t bits, uint16_t *out) {
    if (!validate_bits(bits)) return -1;
    for (size_t i = 0; i < count; i++) {
        out[i] = (uint16_t)read_bits(packed, i * bits, bits);
    }
    return 0;
}

int tq_pack_signs(const int8_t *signs, size_t count, uint8_t *out) {
    size_t packed_len = (count + 7) / 8;
    memset(out, 0, packed_len);
    for (size_t i = 0; i < count; i++) {
        if (signs[i] == 1) {
            out[i / 8] |= (uint8_t)(1 << (i % 8));
        } else if (signs[i] != -1) {
            return -1;
        }
    }
    return 0;
}

int tq_unpack_signs(const uint8_t *packed, size_t count, int8_t *out) {
    for (size_t i = 0; i < count; i++) {
        out[i] = (packed[i / 8] & (1 << (i % 8))) ? 1 : -1;
    }
    return 0;
}
