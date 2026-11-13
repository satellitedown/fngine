// Modified for fngine from llama.cpp/ggml:ggml/src/ggml-common.h, ggml/src/ggml-impl.h, ggml/src/ggml-quants.h @ 3cf03257f219afbe7334045ff7c6a06ac68c627d. See NOTICE.
// SPDX-License-Identifier: MIT (see LICENSE in this directory)
#pragma once
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>

#ifdef __cplusplus
#define IQ_RESTRICT __restrict__
#define IQ_STATIC_ASSERT(c, m) static_assert(c, m)
extern "C" {
#else
#define IQ_RESTRICT restrict
#define IQ_STATIC_ASSERT(c, m) _Static_assert(c, m)
#endif

typedef uint16_t ggml_half;
typedef uint32_t ggml_half2;
typedef uint16_t ggml_fp16_t;
typedef struct { uint16_t bits; } ggml_bf16_t;

#define QK_K 256

// Named struct tags. nvcc 13.0 aborts in lexical.c (find_allocated_name_reference)
// when a .cu includes these as anonymous typedef structs.
#define QK2_0 64
typedef struct block_q2_0 {
    ggml_half d;              // delta (scale)
    uint8_t qs[QK2_0 / 4];   // 2 bits per element
} block_q2_0;
IQ_STATIC_ASSERT(sizeof(block_q2_0) == sizeof(ggml_half) + QK2_0 / 4, "wrong q2_0 block size/padding");

#define QK8_0 32
typedef struct block_q8_0 {
    ggml_half d;       // delta
    int8_t  qs[QK8_0]; // quants
} block_q8_0;
IQ_STATIC_ASSERT(sizeof(block_q8_0) == sizeof(ggml_half) + QK8_0, "wrong q8_0 block size/padding");

#define QK8_1 32
typedef struct block_q8_1 {
    union {
        struct {
            ggml_half d; // delta
            ggml_half s; // d * sum(qs[i])
        };
        ggml_half2 ds;
    };
    int8_t qs[QK8_1]; // quants
} block_q8_1;
IQ_STATIC_ASSERT(sizeof(block_q8_1) == 2*sizeof(ggml_half) + QK8_1, "wrong q8_1 block size/padding");

typedef struct block_q8_K {
    float   d;              // delta
    int8_t  qs[QK_K];       // quants
    int16_t bsums[QK_K/16]; // sum of quants in groups of 16
} block_q8_K;
IQ_STATIC_ASSERT(sizeof(block_q8_K) == sizeof(float) + QK_K + QK_K/16*sizeof(int16_t), "wrong q8_K block size/padding");

typedef struct block_iq2_s {
    ggml_half d;
    uint8_t qs[QK_K/4];
    uint8_t qh[QK_K/32];
    uint8_t scales[QK_K/32];
} block_iq2_s;
IQ_STATIC_ASSERT(sizeof(block_iq2_s) == sizeof(ggml_half) + QK_K/4 + QK_K/16, "wrong iq2_s block size/padding");

typedef struct block_iq3_xxs {
    ggml_half d;
    uint8_t qs[3*QK_K/8];
} block_iq3_xxs;
IQ_STATIC_ASSERT(sizeof(block_iq3_xxs) == sizeof(ggml_half) + 3*(QK_K/8), "wrong iq3_xxs block size/padding");

typedef struct block_iq4_xs {
    ggml_half d;
    uint16_t scales_h;
    uint8_t  scales_l[QK_K/64];
    uint8_t  qs[QK_K/2];
} block_iq4_xs;
IQ_STATIC_ASSERT(sizeof(block_iq4_xs) == sizeof(ggml_half) + sizeof(uint16_t) + QK_K/64 + QK_K/2, "wrong iq4_xs block size/padding");

#define IQ3S_N_SCALE (QK_K/64)
typedef struct block_iq3_s {
    ggml_half d;
    uint8_t qs[QK_K/4];
    uint8_t qh[QK_K/32];
    uint8_t signs[QK_K/8];
    uint8_t scales[IQ3S_N_SCALE];
} block_iq3_s;
IQ_STATIC_ASSERT(sizeof(block_iq3_s) == sizeof(ggml_half) + 13*(QK_K/32) + IQ3S_N_SCALE, "wrong iq3_s block size/padding");

#define QK4_NL 32
typedef struct block_iq4_nl {
    ggml_half d;
    uint8_t qs[QK4_NL/2];
} block_iq4_nl;
IQ_STATIC_ASSERT(sizeof(block_iq4_nl) == sizeof(ggml_half) + QK4_NL/2, "wrong iq4_nl block size/padding");

// ---- FP16 / BF16 conversion (ggml-impl.h) ----
static inline float ggml_iq_fp32_from_bits(uint32_t w) {
    union {
        uint32_t as_bits;
        float as_value;
    } fp32;
    fp32.as_bits = w;
    return fp32.as_value;
}

static inline uint32_t ggml_iq_fp32_to_bits(float f) {
    union {
        float as_value;
        uint32_t as_bits;
    } fp32;
    fp32.as_value = f;
    return fp32.as_bits;
}

static inline float ggml_compute_fp16_to_fp32(ggml_fp16_t h) {
    const uint32_t w = (uint32_t) h << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;

    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
#if (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    const float exp_scale = 0x1.0p-112f;
#else
    const float exp_scale = ggml_iq_fp32_from_bits(UINT32_C(0x7800000));
#endif
    const float normalized_value = ggml_iq_fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = ggml_iq_fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign |
        (two_w < denormalized_cutoff ? ggml_iq_fp32_to_bits(denormalized_value) : ggml_iq_fp32_to_bits(normalized_value));
    return ggml_iq_fp32_from_bits(result);
}

static inline ggml_fp16_t ggml_compute_fp32_to_fp16(float f) {
#if (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    const float scale_to_inf = 0x1.0p+112f;
    const float scale_to_zero = 0x1.0p-110f;
#else
    const float scale_to_inf = ggml_iq_fp32_from_bits(UINT32_C(0x77800000));
    const float scale_to_zero = ggml_iq_fp32_from_bits(UINT32_C(0x08800000));
#endif
    float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    const uint32_t w = ggml_iq_fp32_to_bits(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    if (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }

    base = ggml_iq_fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    const uint32_t bits = ggml_iq_fp32_to_bits(base);
    const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
    const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}

static inline float ggml_compute_bf16_to_fp32(ggml_bf16_t h) {
    union {
        float f;
        uint32_t i;
    } u;
    u.i = (uint32_t)h.bits << 16;
    return u.f;
}

/**
 * Converts float32 to brain16.
 *
 * This is binary identical with Google Brain float conversion.
 * Floats shall round to nearest even, and NANs shall be quiet.
 * Subnormals aren't flushed to zero, except perhaps when used.
 * This code should vectorize nicely if using modern compilers.
 */
static inline ggml_bf16_t ggml_compute_fp32_to_bf16(float s) {
    ggml_bf16_t h;
    union {
        float f;
        uint32_t i;
    } u;
    u.f = s;
    if ((u.i & 0x7fffffff) > 0x7f800000) { /* nan */
        h.bits = (u.i >> 16) | 64; /* force to quiet */
        return h;
    }
    h.bits = (u.i + (0x7fff + ((u.i >> 16) & 1))) >> 16;
    return h;
}

#define GGML_FP16_TO_FP32(x) ggml_compute_fp16_to_fp32(x)
#define GGML_FP32_TO_FP16(x) ggml_compute_fp32_to_fp16(x)
#define GGML_BF16_TO_FP32(x) ggml_compute_bf16_to_fp32(x)
#define GGML_FP32_TO_BF16(x) ggml_compute_fp32_to_bf16(x)

// ---- reference (de)quantizers (ggml-quants.c) ----
void dequantize_row_iq2_s  (const block_iq2_s   * IQ_RESTRICT x, float * IQ_RESTRICT y, int64_t k);
void dequantize_row_iq3_xxs(const block_iq3_xxs * IQ_RESTRICT x, float * IQ_RESTRICT y, int64_t k);
void dequantize_row_iq3_s  (const block_iq3_s   * IQ_RESTRICT x, float * IQ_RESTRICT y, int64_t k);
void dequantize_row_iq4_xs (const block_iq4_xs  * IQ_RESTRICT x, float * IQ_RESTRICT y, int64_t k);
void dequantize_row_iq4_nl (const block_iq4_nl  * IQ_RESTRICT x, float * IQ_RESTRICT y, int64_t k);
void dequantize_row_q2_0   (const block_q2_0    * IQ_RESTRICT x, float * IQ_RESTRICT y, int64_t k);
void dequantize_row_q8_0   (const block_q8_0    * IQ_RESTRICT x, float * IQ_RESTRICT y, int64_t k);
void dequantize_row_q8_K   (const block_q8_K    * IQ_RESTRICT x, float * IQ_RESTRICT y, int64_t k);

void quantize_row_q8_0_ref(const float * IQ_RESTRICT x, block_q8_0 * IQ_RESTRICT y, int64_t k);
void quantize_row_q8_1_ref(const float * IQ_RESTRICT x, block_q8_1 * IQ_RESTRICT y, int64_t k);
void quantize_row_q8_K_ref(const float * IQ_RESTRICT x, block_q8_K * IQ_RESTRICT y, int64_t k);
void quantize_row_q2_0_ref(const float * IQ_RESTRICT x, block_q2_0 * IQ_RESTRICT y, int64_t k);
// quant_weights may be NULL. Returns bytes written.
size_t quantize_iq4_nl(const float * IQ_RESTRICT src, void * IQ_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights);

// ---- generic dot products (ggml-cpu/quants.c); nrc must be 1 ----
void ggml_vec_dot_iq2_s_q8_K_generic(int n, float * IQ_RESTRICT s, size_t bs, const void * IQ_RESTRICT vx, size_t bx, const void * IQ_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq3_xxs_q8_K_generic(int n, float * IQ_RESTRICT s, size_t bs, const void * IQ_RESTRICT vx, size_t bx, const void * IQ_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq3_s_q8_K_generic(int n, float * IQ_RESTRICT s, size_t bs, const void * IQ_RESTRICT vx, size_t bx, const void * IQ_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq4_xs_q8_K_generic(int n, float * IQ_RESTRICT s, size_t bs, const void * IQ_RESTRICT vx, size_t bx, const void * IQ_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq4_nl_q8_0_generic(int n, float * IQ_RESTRICT s, size_t bs, const void * IQ_RESTRICT vx, size_t bx, const void * IQ_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q2_0_q8_0_generic(int n, float * IQ_RESTRICT s, size_t bs, const void * IQ_RESTRICT vx, size_t bx, const void * IQ_RESTRICT vy, size_t by, int nrc);

#ifdef __cplusplus
}
#endif
