// AVX2 + AVX-VNNI + AVX-NE-CONVERT CPU expert kernels.
// Same layouts and public API as iq_cpu.cpp (the AVX-512 path). Exactly one of the two is compiled.
#include "cpu/cpu_isa.h"
#if FNGINE_CPU_BACKEND_AVX2
// Modified for fngine from llama.cpp/ggml:ggml/src/ggml-cpu/arch/x86/quants.c @ 3cf03257f219afbe7334045ff7c6a06ac68c627d. See NOTICE.
// SPDX-License-Identifier: MIT
//
// Block semantics follow ggml's ggml_vec_dot_{iq2_s,iq3_xxs,iq3_s,iq4_xs}_q8_K and
// ggml_vec_dot_iq4_nl_q8_0 (grid tables, sign bits, sub-block scales, IQ4 nibble table, final
// 1/8 and 1/4 factors). fngine restructures them into a decode-once GEMM (see iq_cpu.h):
//  * each GGUF block is expanded once into biased uint8 weights in a lane-permuted order that
//    makes every int32 lane belong to one sub-block, so the sub-block scale is a per-lane float;
//  * a register-blocked vpdpbusd micro-kernel runs over (rows x tokens) tiles with exact int32
//    sums per block and one FP32 scale step per block.
// Q2_0 (value (q - 1) * d) is new: it uses the same biased form, w + 128 = q + 127.
#include "cpu/iq_cpu.h"

#include "cpu/q8_cpu.h"
#include "ggml-iq/ggml-iq-tables.h"

#include <immintrin.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fngine::cpu {
namespace {

inline float fp16(uint16_t h) { return _cvtsh_ss(h); }

inline uint64_t load_u64(const void* p) {
  uint64_t v;
  std::memcpy(&v, p, 8);
  return v;
}
inline uint32_t load_u32(const void* p) {
  uint32_t v;
  std::memcpy(&v, p, 4);
  return v;
}
inline uint16_t load_u16(const void* p) {
  uint16_t v;
  std::memcpy(&v, p, 2);
  return v;
}

// ------------------------------------------------------------------ geometry
constexpr int kChunk = 64;

constexpr int chunks_of(ActFormat a) { return a == ActFormat::Q8_K ? 4 : 1; }
constexpr int group_elems_of(ActFormat a) { return a == ActFormat::Q8_K ? QK_K : 64; }
constexpr int wgroup_bytes(int C) { return C * kChunk + 64; }
constexpr int agroup_bytes(int C) { return C * kChunk + 128; }

int groups_of(ActFormat a, int32_t k) {
  const int ge = group_elems_of(a);
  return (k + ge - 1) / ge;
}

void check_k(WeightFormat fmt, int32_t k) {
  const FormatInfo fi = format_info(fmt);
  if (k <= 0 || k % fi.block_elems != 0)
    throw std::invalid_argument(std::string("row length not a multiple of the ") + format_name(fmt) +
                                " block size");
}

// Weight rows stream from DRAM (cold experts); rows are decoded in address order, so a fixed
// prefetch distance also covers the start of the next row.
constexpr int kPrefetchBytes = 1024;
inline void prefetch_ahead(const void* p) {
  const char* c = static_cast<const char*>(p) + kPrefetchBytes;
  _mm_prefetch(c, _MM_HINT_T0);
  _mm_prefetch(c + 64, _MM_HINT_T0);
}

inline __m256i cvt2ps_to_pbh(__m256 a_hi, __m256 b_lo) {
  __m128i lo = (__m128i)_mm256_cvtneps_avx_pbh(b_lo);
  __m128i hi = (__m128i)_mm256_cvtneps_avx_pbh(a_hi);
  return _mm256_set_m128i(hi, lo);
}

inline float hsum256_ps(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  __m128 s = _mm_add_ps(lo, hi);
  s = _mm_add_ps(s, _mm_movehl_ps(s, s));
  s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
  return _mm_cvtss_f32(s);
}

// ------------------------------------------------------------------ weight decoders
// Helper: apply sign bits to 8 dwords (32 bytes) of grid magnitudes `g` (< 128) and add 128.
// `s32` holds in each dword `j` (0..7) the 16 sign bits for lane `j`:
//   byte 0 bits 0..3 -> c=0, byte 0 bits 4..7 -> c=1,
//   byte 1 bits 0..3 -> c=2, byte 1 bits 4..7 -> c=3.
template <int C>
inline __m256i bias_sign_256(__m256i g, __m256i s32) {
  static_assert(C >= 0 && C < 4);
  const __m256i shuf = (C < 2)
      ? _mm256_setr_epi8(0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12,
                         0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12)
      : _mm256_setr_epi8(1, 1, 1, 1, 5, 5, 5, 5, 9, 9, 9, 9, 13, 13, 13, 13,
                         1, 1, 1, 1, 5, 5, 5, 5, 9, 9, 9, 9, 13, 13, 13, 13);
  const __m256i bit_mask = _mm256_set1_epi32((C & 1) ? 0x80402010 : 0x08040201);
  const __m256i rep = _mm256_shuffle_epi8(s32, shuf);
  const __m256i is_neg = _mm256_cmpeq_epi8(_mm256_and_si256(rep, bit_mask), bit_mask);
  const __m256i sign_mul = _mm256_or_si256(is_neg, _mm256_set1_epi8(1));
  return _mm256_add_epi8(_mm256_sign_epi8(g, sign_mul), _mm256_set1_epi8(char(0x80)));
}

// Q8_K formats: one 256-element block -> [4 x 64 B w+128 (lane-permuted)][64 B F].
void dec_iq2_s(const block_iq2_s& x, uint8_t* dst) {
  const float d = fp16(x.d) * 0.125f;
  const auto* grid32 = reinterpret_cast<const int*>(iq2s_grid);
  const __m256i qh_shift_p0 = _mm256_setr_epi32(0, 4, 0, 4, 0, 4, 0, 4);
  const __m256i qh_shift_p1 = _mm256_setr_epi32(2, 6, 2, 6, 2, 6, 2, 6);
  const __m256i mask3 = _mm256_set1_epi32(3);
  const __m256i maskff = _mm256_set1_epi32(0xFF);
  const __m256i dup_pair = _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3);

  for (int half = 0; half < 2; ++half) {
    // 8 uint16_t of qs -> low byte is p=0 (c=0,1), high byte is p=1 (c=2,3)
    const __m256i qs16 = _mm256_cvtepu16_epi32(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(x.qs + 16 * half)));
    // 4 bytes of qh -> duplicate each byte to a pair of dwords (even/odd j_lane)
    const __m128i qh4 = _mm_cvtsi32_si128(static_cast<int>(load_u32(x.qh + 4 * half)));
    const __m256i qh8 = _mm256_permutevar8x32_epi32(_mm256_cvtepu8_epi32(qh4), dup_pair);
    const __m256i s32 = _mm256_cvtepu16_epi32(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(x.qs + QK_K / 8 + 16 * half)));

    // p = 0 (c = 0, 1)
    {
      const __m256i lo = _mm256_and_si256(qs16, maskff);
      const __m256i hi = _mm256_and_si256(_mm256_srlv_epi32(qh8, qh_shift_p0), mask3);
      const __m256i gi2 = _mm256_slli_epi32(_mm256_or_si256(lo, _mm256_slli_epi32(hi, 8)), 1);
      const __m256i g0 = _mm256_i32gather_epi32(grid32 + 0, gi2, 4);
      const __m256i g1 = _mm256_i32gather_epi32(grid32 + 1, gi2, 4);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + 0 * kChunk + 32 * half),
                          bias_sign_256<0>(g0, s32));
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + 1 * kChunk + 32 * half),
                          bias_sign_256<1>(g1, s32));
    }
    // p = 1 (c = 2, 3)
    {
      const __m256i lo = _mm256_srli_epi32(qs16, 8);
      const __m256i hi = _mm256_and_si256(_mm256_srlv_epi32(qh8, qh_shift_p1), mask3);
      const __m256i gi2 = _mm256_slli_epi32(_mm256_or_si256(lo, _mm256_slli_epi32(hi, 8)), 1);
      const __m256i g2 = _mm256_i32gather_epi32(grid32 + 0, gi2, 4);
      const __m256i g3 = _mm256_i32gather_epi32(grid32 + 1, gi2, 4);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + 2 * kChunk + 32 * half),
                          bias_sign_256<2>(g2, s32));
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + 3 * kChunk + 32 * half),
                          bias_sign_256<3>(g3, s32));
    }
    // Scales: 4 bytes -> 8 nibbles
    const __m128i sc4 = _mm_cvtsi32_si128(static_cast<int>(load_u32(x.scales + 4 * half)));
    const __m256i sc8 = _mm256_permutevar8x32_epi32(_mm256_cvtepu8_epi32(sc4), dup_pair);
    const __m256i nib = _mm256_and_si256(_mm256_srlv_epi32(sc8, qh_shift_p0), _mm256_set1_epi32(15));
    const __m256i odd = _mm256_or_si256(_mm256_slli_epi32(nib, 1), _mm256_set1_epi32(1));
    _mm256_storeu_ps(reinterpret_cast<float*>(dst + 4 * kChunk + 32 * half),
                     _mm256_mul_ps(_mm256_cvtepi32_ps(odd), _mm256_set1_ps(d)));
  }
}

void dec_iq3_xxs(const block_iq3_xxs& x, uint8_t* dst) {
  const float d = fp16(x.d) * 0.25f;
  const auto* grid32 = reinterpret_cast<const int*>(iq3xxs_grid);
  const uint8_t* scales_and_signs = x.qs + QK_K / 4;
  const __m256i dup_pair = _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3);
  const __m256i shift0_14 = _mm256_setr_epi32(0, 14, 0, 14, 0, 14, 0, 14);
  const __m256i maskff = _mm256_set1_epi32(0xFF);
  const __m256i pop4 = _mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
                                        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);

  for (int half = 0; half < 2; ++half) {
    const __m256i q = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x.qs + 32 * half));
    const __m128i aux4 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(scales_and_signs + 16 * half));
    const __m256i aux8 = _mm256_permutevar8x32_epi32(_mm256_castsi128_si256(aux4), dup_pair);

    // Extract the two 7-bit sign indices per lane into byte 0 and byte 1 of each dword:
    const __m256i aux_sh = _mm256_srlv_epi32(aux8, shift0_14);
    const __m256i s_lo = _mm256_and_si256(aux_sh, _mm256_set1_epi32(127));
    const __m256i s_hi = _mm256_and_si256(_mm256_srli_epi32(aux_sh, 7), _mm256_set1_epi32(127));
    const __m256i s7 = _mm256_or_si256(s_lo, _mm256_slli_epi32(s_hi, 8));
    // Add odd-parity bit 7 to match ksigns_iq2xs:
    const __m256i pcnt = _mm256_and_si256(
        _mm256_add_epi8(
            _mm256_shuffle_epi8(pop4, _mm256_and_si256(s7, _mm256_set1_epi8(0x0f))),
            _mm256_shuffle_epi8(pop4, _mm256_and_si256(_mm256_srli_epi16(s7, 4), _mm256_set1_epi8(0x0f)))),
        _mm256_set1_epi8(1));
    const __m256i s32 = _mm256_or_si256(s7, _mm256_slli_epi16(pcnt, 7));

    const __m256i g0 = _mm256_i32gather_epi32(grid32, _mm256_and_si256(q, maskff), 4);
    const __m256i g1 = _mm256_i32gather_epi32(grid32, _mm256_and_si256(_mm256_srli_epi32(q, 8), maskff), 4);
    const __m256i g2 = _mm256_i32gather_epi32(grid32, _mm256_and_si256(_mm256_srli_epi32(q, 16), maskff), 4);
    const __m256i g3 = _mm256_i32gather_epi32(grid32, _mm256_srli_epi32(q, 24), 4);

    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + 0 * kChunk + 32 * half), bias_sign_256<0>(g0, s32));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + 1 * kChunk + 32 * half), bias_sign_256<1>(g1, s32));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + 2 * kChunk + 32 * half), bias_sign_256<2>(g2, s32));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + 3 * kChunk + 32 * half), bias_sign_256<3>(g3, s32));

    const __m256i ls = _mm256_srli_epi32(aux8, 28);
    const __m256i odd = _mm256_or_si256(_mm256_slli_epi32(ls, 1), _mm256_set1_epi32(1));
    _mm256_storeu_ps(reinterpret_cast<float*>(dst + 4 * kChunk + 32 * half),
                     _mm256_mul_ps(_mm256_cvtepi32_ps(odd), _mm256_set1_ps(d)));
  }
}

void dec_iq3_s(const block_iq3_s& x, uint8_t* dst) {
  const float d = fp16(x.d);
  const auto* grid32 = reinterpret_cast<const int*>(iq3s_grid);
  const __m256i dup_pair = _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3);
  const __m256i qh_sh0 = _mm256_setr_epi32(0, 4, 0, 4, 0, 4, 0, 4);
  const __m256i qh_sh1 = _mm256_setr_epi32(1, 5, 1, 5, 1, 5, 1, 5);
  const __m256i qh_sh2 = _mm256_setr_epi32(2, 6, 2, 6, 2, 6, 2, 6);
  const __m256i qh_sh3 = _mm256_setr_epi32(3, 7, 3, 7, 3, 7, 3, 7);
  const __m256i maskff = _mm256_set1_epi32(0xFF);
  const __m256i mask1 = _mm256_set1_epi32(1);

  for (int half = 0; half < 2; ++half) {
    const __m256i q = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x.qs + 32 * half));
    const __m128i qh4 = _mm_cvtsi32_si128(static_cast<int>(load_u32(x.qh + 4 * half)));
    const __m256i qh8 = _mm256_permutevar8x32_epi32(_mm256_cvtepu8_epi32(qh4), dup_pair);
    const __m256i s32 = _mm256_cvtepu16_epi32(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(x.signs + 16 * half)));

    const __m256i idx0 = _mm256_or_si256(
        _mm256_and_si256(q, maskff),
        _mm256_slli_epi32(_mm256_and_si256(_mm256_srlv_epi32(qh8, qh_sh0), mask1), 8));
    const __m256i idx1 = _mm256_or_si256(
        _mm256_and_si256(_mm256_srli_epi32(q, 8), maskff),
        _mm256_slli_epi32(_mm256_and_si256(_mm256_srlv_epi32(qh8, qh_sh1), mask1), 8));
    const __m256i idx2 = _mm256_or_si256(
        _mm256_and_si256(_mm256_srli_epi32(q, 16), maskff),
        _mm256_slli_epi32(_mm256_and_si256(_mm256_srlv_epi32(qh8, qh_sh2), mask1), 8));
    const __m256i idx3 = _mm256_or_si256(
        _mm256_srli_epi32(q, 24),
        _mm256_slli_epi32(_mm256_and_si256(_mm256_srlv_epi32(qh8, qh_sh3), mask1), 8));

    const __m256i g0 = _mm256_i32gather_epi32(grid32, idx0, 4);
    const __m256i g1 = _mm256_i32gather_epi32(grid32, idx1, 4);
    const __m256i g2 = _mm256_i32gather_epi32(grid32, idx2, 4);
    const __m256i g3 = _mm256_i32gather_epi32(grid32, idx3, 4);

    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + 0 * kChunk + 32 * half), bias_sign_256<0>(g0, s32));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + 1 * kChunk + 32 * half), bias_sign_256<1>(g1, s32));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + 2 * kChunk + 32 * half), bias_sign_256<2>(g2, s32));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + 3 * kChunk + 32 * half), bias_sign_256<3>(g3, s32));

    // 2 bytes of x.scales per half -> 4 nibbles -> 8 lanes (each nibble shared by a pair of j_lanes)
    const uint16_t sc2 = load_u16(x.scales + 2 * half);
    const __m256i sc_vec = _mm256_set1_epi32(sc2);
    const __m256i pair_shift = _mm256_setr_epi32(0, 0, 4, 4, 8, 8, 12, 12);
    const __m256i nib = _mm256_and_si256(_mm256_srlv_epi32(sc_vec, pair_shift), _mm256_set1_epi32(15));
    const __m256i odd = _mm256_or_si256(_mm256_slli_epi32(nib, 1), mask1);
    _mm256_storeu_ps(reinterpret_cast<float*>(dst + 4 * kChunk + 32 * half),
                     _mm256_mul_ps(_mm256_cvtepi32_ps(odd), _mm256_set1_ps(d)));
  }
}

void dec_iq4_xs(const block_iq4_xs& x, uint8_t* dst) {
  const float d = fp16(x.d);
  alignas(16) static constexpr uint8_t kIq4Biased[16] = {
      uint8_t(-127 + 128), uint8_t(-104 + 128), uint8_t(-83 + 128), uint8_t(-65 + 128),
      uint8_t(-49 + 128),  uint8_t(-35 + 128),  uint8_t(-22 + 128), uint8_t(-10 + 128),
      uint8_t(1 + 128),    uint8_t(13 + 128),   uint8_t(25 + 128),  uint8_t(38 + 128),
      uint8_t(53 + 128),   uint8_t(69 + 128),   uint8_t(89 + 128),  uint8_t(113 + 128)};
  const __m256i tbl = _mm256_broadcastsi128_si256(_mm_load_si128(reinterpret_cast<const __m128i*>(kIq4Biased)));
  const __m256i mask0f = _mm256_set1_epi8(0x0f);
  float* F = reinterpret_cast<float*>(dst + 4 * kChunk);

  for (int half = 0; half < 2; ++half) {
    // Two 32-byte loads cover 4 sub-blocks (ib = 4*half .. 4*half + 3 -> 8 j_lanes)
    const __m256i q01 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x.qs + 64 * half + 0));
    const __m256i q23 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x.qs + 64 * half + 32));

    // In q01: low 128b is ib=0 (lo nib = j0, hi nib = j1), high 128b is ib=1 (lo nib = j2, hi nib = j3)
    const __m256i w_even_02 = _mm256_shuffle_epi8(tbl, _mm256_and_si256(q01, mask0f));
    const __m256i w_odd_13 = _mm256_shuffle_epi8(tbl, _mm256_and_si256(_mm256_srli_epi16(q01, 4), mask0f));
    const __m256i w_even_46 = _mm256_shuffle_epi8(tbl, _mm256_and_si256(q23, mask0f));
    const __m256i w_odd_57 = _mm256_shuffle_epi8(tbl, _mm256_and_si256(_mm256_srli_epi16(q23, 4), mask0f));

    // Interleave even and odd j_lanes:
    // unpacklo_epi32 gives [c0_j0, c0_j1, c1_j0, c1_j1 | c0_j2, c0_j3, c1_j2, c1_j3]
    // unpackhi_epi32 gives [c2_j0, c2_j1, c3_j0, c3_j1 | c2_j2, c2_j3, c3_j2, c3_j3]
    const __m256i c01_03 = _mm256_unpacklo_epi32(w_even_02, w_odd_13);
    const __m256i c23_03 = _mm256_unpackhi_epi32(w_even_02, w_odd_13);
    const __m256i c01_47 = _mm256_unpacklo_epi32(w_even_46, w_odd_57);
    const __m256i c23_47 = _mm256_unpackhi_epi32(w_even_46, w_odd_57);

    // Within each 128b lane, dwords 0,1 are c0 (or c2) and dwords 2,3 are c1 (or c3):
    const __m256i c0_packed = _mm256_unpacklo_epi64(c01_03, c01_47);
    const __m256i c1_packed = _mm256_unpackhi_epi64(c01_03, c01_47);
    const __m256i c2_packed = _mm256_unpacklo_epi64(c23_03, c23_47);
    const __m256i c3_packed = _mm256_unpackhi_epi64(c23_03, c23_47);

    // c0_packed has [j0, j1, j4, j5 | j2, j3, j6, j7] -> permute 64-bit lanes [0, 2, 1, 3] (0xD8)
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + 0 * kChunk + 32 * half),
                        _mm256_permute4x64_epi64(c0_packed, 0xD8));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + 1 * kChunk + 32 * half),
                        _mm256_permute4x64_epi64(c1_packed, 0xD8));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + 2 * kChunk + 32 * half),
                        _mm256_permute4x64_epi64(c2_packed, 0xD8));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + 3 * kChunk + 32 * half),
                        _mm256_permute4x64_epi64(c3_packed, 0xD8));

    for (int ib = 4 * half; ib < 4 * half + 4; ++ib) {
      const int ls = ((x.scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) |
                     (((x.scales_h >> (2 * ib)) & 3) << 4);
      const float dl = d * (ls - 32);
      F[2 * ib + 0] = dl;
      F[2 * ib + 1] = dl;
    }
  }
}

// Q8_0 formats: one 64-element group -> [64 B w+128 (natural order)][64 B F].
void dec_iq4_nl(const block_iq4_nl* xa, const block_iq4_nl* xb, uint8_t* dst) {
  alignas(16) static constexpr uint8_t kIq4Biased[16] = {
      uint8_t(-127 + 128), uint8_t(-104 + 128), uint8_t(-83 + 128), uint8_t(-65 + 128),
      uint8_t(-49 + 128),  uint8_t(-35 + 128),  uint8_t(-22 + 128), uint8_t(-10 + 128),
      uint8_t(1 + 128),    uint8_t(13 + 128),   uint8_t(25 + 128),  uint8_t(38 + 128),
      uint8_t(53 + 128),   uint8_t(69 + 128),   uint8_t(89 + 128),  uint8_t(113 + 128)};
  const __m256i tbl = _mm256_broadcastsi128_si256(_mm_load_si128(reinterpret_cast<const __m128i*>(kIq4Biased)));
  const __m128i mask0f = _mm_set1_epi8(0x0f);
  const __m128i qa = _mm_loadu_si128(reinterpret_cast<const __m128i*>(xa->qs));
  const __m256i nib_a = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(qa, 4), mask0f),
                                         _mm_and_si128(qa, mask0f));
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), _mm256_shuffle_epi8(tbl, nib_a));
  if (xb) {
    const __m128i qb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(xb->qs));
    const __m256i nib_b = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(qb, 4), mask0f),
                                           _mm_and_si128(qb, mask0f));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + 32), _mm256_shuffle_epi8(tbl, nib_b));
  } else {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + 32), _mm256_set1_epi8(char(0x80)));
  }
  _mm256_storeu_ps(reinterpret_cast<float*>(dst + kChunk), _mm256_set1_ps(fp16(xa->d)));
  _mm256_storeu_ps(reinterpret_cast<float*>(dst + kChunk + 32),
                   _mm256_set1_ps(xb ? fp16(xb->d) : 0.0f));
}

void dec_q2_0(const block_q2_0& x, uint8_t* dst) {
  const __m128i qb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x.qs));
  const __m256i d0 = _mm256_cvtepu8_epi32(qb);
  const __m256i d1 = _mm256_cvtepu8_epi32(_mm_srli_si128(qb, 8));
  const __m256i m3 = _mm256_set1_epi8(3);
  const __m256i c127 = _mm256_set1_epi8(127);

  auto expand_2b = [&](__m256i d) {
    const __m256i lo = _mm256_or_si256(d, _mm256_slli_epi32(d, 6));
    const __m256i hi = _mm256_or_si256(_mm256_slli_epi32(d, 12), _mm256_slli_epi32(d, 18));
    const __m256i combined = _mm256_blend_epi16(lo, hi, 0xAA);
    return _mm256_add_epi8(_mm256_and_si256(combined, m3), c127);
  };
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), expand_2b(d0));
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + 32), expand_2b(d1));
  const __m256 vd = _mm256_set1_ps(fp16(x.d));
  _mm256_storeu_ps(reinterpret_cast<float*>(dst + kChunk), vd);
  _mm256_storeu_ps(reinterpret_cast<float*>(dst + kChunk + 32), vd);
}

template <class Block, void (*Dec)(const Block&, uint8_t*)>
void decode_rows_q8k(const void* rows, std::size_t row_bytes, int32_t nrows, int nblk, uint8_t* dst) {
  constexpr int WG = wgroup_bytes(4);
  for (int r = 0; r < nrows; ++r) {
    const auto* x = reinterpret_cast<const Block*>(static_cast<const uint8_t*>(rows) + std::size_t(r) * row_bytes);
    for (int b = 0; b < nblk; ++b, dst += WG) {
      prefetch_ahead(x + b);
      Dec(x[b], dst);
    }
  }
}

// ------------------------------------------------------------------ activation packing
void pack_group_q8K(const block_q8_K& x, uint8_t* dst) {
  const int32_t* qs32 = reinterpret_cast<const int32_t*>(x.qs);
  __m256i sum_lo = _mm256_setzero_si256();
  __m256i sum_hi = _mm256_setzero_si256();
  const __m256i k128 = _mm256_set1_epi8(char(0x80));
  for (int c = 0; c < 4; ++c) {
    int32_t* d32 = reinterpret_cast<int32_t*>(dst + kChunk * c);
    for (int j = 0; j < 16; ++j) {
      d32[j] = qs32[4 * j + c];
    }
    const __m256i vlo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(d32));
    const __m256i vhi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(d32 + 8));
    sum_lo = _mm256_add_epi32(sum_lo, _mm256_dpbusd_epi32(_mm256_setzero_si256(), k128, vlo));
    sum_hi = _mm256_add_epi32(sum_hi, _mm256_dpbusd_epi32(_mm256_setzero_si256(), k128, vhi));
  }
  _mm256_storeu_ps(reinterpret_cast<float*>(dst + 4 * kChunk), _mm256_set1_ps(x.d));
  _mm256_storeu_ps(reinterpret_cast<float*>(dst + 4 * kChunk + 32), _mm256_set1_ps(x.d));
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + 4 * kChunk + 64),
                      _mm256_sub_epi32(_mm256_setzero_si256(), sum_lo));
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + 4 * kChunk + 96),
                      _mm256_sub_epi32(_mm256_setzero_si256(), sum_hi));
}

void pack_group_q8_0(const block_q8_0* a, const block_q8_0* b, uint8_t* dst) {
  const __m256i lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a->qs));
  const __m256i hi = b ? _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b->qs)) : _mm256_setzero_si256();
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), lo);
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + 32), hi);
  _mm256_storeu_ps(reinterpret_cast<float*>(dst + kChunk), _mm256_set1_ps(fp16(a->d)));
  _mm256_storeu_ps(reinterpret_cast<float*>(dst + kChunk + 32), _mm256_set1_ps(b ? fp16(b->d) : 0.0f));

  const __m256i k128 = _mm256_set1_epi8(char(0x80));
  const __m256i acorr_lo = _mm256_sub_epi32(
      _mm256_setzero_si256(), _mm256_dpbusd_epi32(_mm256_setzero_si256(), k128, lo));
  const __m256i acorr_hi = _mm256_sub_epi32(
      _mm256_setzero_si256(), _mm256_dpbusd_epi32(_mm256_setzero_si256(), k128, hi));
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + kChunk + 64), acorr_lo);
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + kChunk + 96), acorr_hi);
}

// ------------------------------------------------------------------ micro-kernel
// R decoded rows (stride ws) x N packed activation rows over G groups of C chunks.
template <int C, int R, int N>
void micro(const uint8_t* w, std::size_t ws, const uint8_t* const* a, int G, float* out,
           std::size_t ldo) {
  constexpr int WG = wgroup_bytes(C), AG = agroup_bytes(C);
  __m256 acc_lo[R][N], acc_hi[R][N];
#pragma GCC unroll 16
  for (int r = 0; r < R; ++r)
#pragma GCC unroll 16
    for (int t = 0; t < N; ++t) {
      acc_lo[r][t] = _mm256_setzero_ps();
      acc_hi[r][t] = _mm256_setzero_ps();
    }
  const uint8_t* ap[N];
#pragma GCC unroll 16
  for (int t = 0; t < N; ++t) ap[t] = a[t];
  for (int g = 0; g < G; ++g) {
    const std::size_t wo = std::size_t(g) * WG, ao = std::size_t(g) * AG;
    __m256i is_lo[R][N], is_hi[R][N];
#pragma GCC unroll 16
    for (int t = 0; t < N; ++t) {
      const __m256i corr_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ap[t] + ao + C * kChunk + 64));
      const __m256i corr_hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ap[t] + ao + C * kChunk + 96));
#pragma GCC unroll 16
      for (int r = 0; r < R; ++r) {
        is_lo[r][t] = corr_lo;
        is_hi[r][t] = corr_hi;
      }
    }
#pragma GCC unroll 4
    for (int c = 0; c < C; ++c) {
      __m256i av_lo[N], av_hi[N];
#pragma GCC unroll 16
      for (int t = 0; t < N; ++t) {
        av_lo[t] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ap[t] + ao + c * kChunk));
        av_hi[t] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ap[t] + ao + c * kChunk + 32));
      }
#pragma GCC unroll 16
      for (int r = 0; r < R; ++r) {
        const __m256i wv_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + r * ws + wo + c * kChunk));
        const __m256i wv_hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + r * ws + wo + c * kChunk + 32));
#pragma GCC unroll 16
        for (int t = 0; t < N; ++t) {
          is_lo[r][t] = _mm256_dpbusd_epi32(is_lo[r][t], wv_lo, av_lo[t]);
          is_hi[r][t] = _mm256_dpbusd_epi32(is_hi[r][t], wv_hi, av_hi[t]);
        }
      }
    }
#pragma GCC unroll 16
    for (int r = 0; r < R; ++r) {
      const __m256 flo = _mm256_loadu_ps(reinterpret_cast<const float*>(w + r * ws + wo + C * kChunk));
      const __m256 fhi = _mm256_loadu_ps(reinterpret_cast<const float*>(w + r * ws + wo + C * kChunk + 32));
#pragma GCC unroll 16
      for (int t = 0; t < N; ++t) {
        const __m256 dlo = _mm256_loadu_ps(reinterpret_cast<const float*>(ap[t] + ao + C * kChunk));
        const __m256 dhi = _mm256_loadu_ps(reinterpret_cast<const float*>(ap[t] + ao + C * kChunk + 32));
        acc_lo[r][t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(is_lo[r][t]), _mm256_mul_ps(flo, dlo), acc_lo[r][t]);
        acc_hi[r][t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(is_hi[r][t]), _mm256_mul_ps(fhi, dhi), acc_hi[r][t]);
      }
    }
  }
#pragma GCC unroll 16
  for (int r = 0; r < R; ++r)
#pragma GCC unroll 16
    for (int t = 0; t < N; ++t) {
      const __m256 sum8 = _mm256_add_ps(acc_lo[r][t], acc_hi[r][t]);
      out[r * ldo + t] = hsum256_ps(sum8);
    }
}

using MicroFn = void (*)(const uint8_t*, std::size_t, const uint8_t* const*, int, float*, std::size_t);

constexpr int kRows = 4;
template <int C>
constexpr int kTokens = C == 4 ? 3 : 4;

template <int C, int R, int... Ns>
constexpr std::array<MicroFn, sizeof...(Ns)> micro_row(std::integer_sequence<int, Ns...>) {
  return {&micro<C, R, Ns + 1>...};
}
template <int C>
struct MicroTable {
  static constexpr auto seq = std::make_integer_sequence<int, kTokens<C>>{};
  static constexpr std::array<std::array<MicroFn, kTokens<C>>, kRows> fns = {
      micro_row<C, 1>(seq), micro_row<C, 2>(seq), micro_row<C, 3>(seq), micro_row<C, 4>(seq)};
};

template <int C>
void gemm_impl(const uint8_t* w, std::size_t ws, int nrows, const uint8_t* const* a, int T, int G,
               float* out, std::size_t ldo) {
  constexpr int N = kTokens<C>;
  for (int t0 = 0; t0 < T; t0 += N) {
    const int nt = std::min(N, T - t0);
    for (int r0 = 0; r0 < nrows; r0 += kRows) {
      const int nr = std::min(kRows, nrows - r0);
      MicroTable<C>::fns[nr - 1][nt - 1](w + std::size_t(r0) * ws, ws, a + t0, G,
                                         out + std::size_t(r0) * ldo + t0, ldo);
    }
  }
}

// ------------------------------------------------------------------ silu
inline __m256 exp256(__m256 x) {
  // Cephes expf: exp(x) = 2^n * p(r), n = round(x / ln2), r = x - n ln2.
  x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-103.0f)), _mm256_set1_ps(88.5f));
  const __m256 n = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f)),
                                    _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m256 r = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.693359375f), x);
  r = _mm256_fnmadd_ps(n, _mm256_set1_ps(-2.12194440e-4f), r);
  __m256 y = _mm256_set1_ps(1.9875691500e-4f);
  y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(1.3981999507e-3f));
  y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(8.3334519073e-3f));
  y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(4.1665795894e-2f));
  y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(1.6666665459e-1f));
  y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(5.0000001201e-1f));
  y = _mm256_fmadd_ps(y, _mm256_mul_ps(r, r), _mm256_add_ps(r, _mm256_set1_ps(1.0f)));

  // 2^n scaling via two exponent splits
  const __m256i ni = _mm256_cvtps_epi32(n);
  const __m256i n1 = _mm256_srai_epi32(ni, 1);
  const __m256i n2 = _mm256_sub_epi32(ni, n1);
  const __m256 p1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(n1, _mm256_set1_epi32(127)), 23));
  const __m256 p2 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(n2, _mm256_set1_epi32(127)), 23));
  return _mm256_mul_ps(_mm256_mul_ps(y, p1), p2);
}

// ------------------------------------------------------------------ scratch for cpu_dot
struct Scratch {
  uint8_t* p = nullptr;
  std::size_t cap = 0;
  ~Scratch() { std::free(p); }
  uint8_t* get(std::size_t bytes) {
    if (bytes > cap) {
      std::free(p);
      cap = (bytes + 4095) & ~std::size_t{4095};
      p = static_cast<uint8_t*>(std::aligned_alloc(64, cap));
      if (!p) throw std::bad_alloc();
    }
    return p;
  }
};

void dot_impl(WeightFormat fmt, const void* row, int nblk, int T,
              const uint8_t* (*act_at)(const void*, std::size_t, int), const void* acts,
              std::size_t stride, float* out) {
  const FormatInfo fi = format_info(fmt);
  const int32_t k = nblk * fi.block_elems;
  thread_local Scratch wbuf, abuf;
  thread_local std::vector<const void*> ptrs;
  const std::size_t pb = packed_act_bytes(fi.act, k);
  uint8_t* ap = abuf.get(pb * std::size_t(std::max(T, 1)));
  ptrs.resize(std::size_t(std::max(T, 1)));
  for (int t = 0; t < T; ++t) {
    const uint8_t* src = act_at(acts, stride, t);
    if (fi.act == ActFormat::Q8_K) pack_act_q8_K(reinterpret_cast<const block_q8_K*>(src), k, ap + pb * t);
    else pack_act_q8_0(reinterpret_cast<const block_q8_0*>(src), k, ap + pb * t);
    ptrs[std::size_t(t)] = ap + pb * t;
  }
  uint8_t* wp = wbuf.get(decoded_row_bytes(fmt, k));
  decode_rows(fmt, row, weight_row_bytes(fmt, k), 1, k, wp);
  gemm_decoded(fmt, wp, 1, k, ptrs.data(), T, out, std::size_t(T));
}

}  // namespace

// ------------------------------------------------------------------ public API
FormatInfo format_info(WeightFormat fmt) {
  switch (fmt) {
    case WeightFormat::IQ2_S: return {QK_K, int32_t(sizeof(block_iq2_s)), ActFormat::Q8_K};
    case WeightFormat::IQ3_XXS: return {QK_K, int32_t(sizeof(block_iq3_xxs)), ActFormat::Q8_K};
    case WeightFormat::IQ3_S: return {QK_K, int32_t(sizeof(block_iq3_s)), ActFormat::Q8_K};
    case WeightFormat::IQ4_XS: return {QK_K, int32_t(sizeof(block_iq4_xs)), ActFormat::Q8_K};
    case WeightFormat::IQ4_NL: return {QK4_NL, int32_t(sizeof(block_iq4_nl)), ActFormat::Q8_0};
    case WeightFormat::Q2_0: return {QK2_0, int32_t(sizeof(block_q2_0)), ActFormat::Q8_0};
  }
  throw std::invalid_argument("unknown weight format " + std::to_string(int32_t(fmt)));
}

const char* format_name(WeightFormat fmt) {
  switch (fmt) {
    case WeightFormat::IQ2_S: return "IQ2_S";
    case WeightFormat::IQ3_XXS: return "IQ3_XXS";
    case WeightFormat::IQ3_S: return "IQ3_S";
    case WeightFormat::IQ4_XS: return "IQ4_XS";
    case WeightFormat::IQ4_NL: return "IQ4_NL";
    case WeightFormat::Q2_0: return "Q2_0";
  }
  return "?";
}

bool parse_weight_format(std::string_view name, WeightFormat* out) {
  static constexpr WeightFormat kAll[] = {WeightFormat::IQ2_S, WeightFormat::IQ3_XXS,
                                          WeightFormat::IQ3_S, WeightFormat::IQ4_XS,
                                          WeightFormat::IQ4_NL, WeightFormat::Q2_0};
  for (WeightFormat f : kAll) {
    if (name == format_name(f)) {
      *out = f;
      return true;
    }
  }
  return false;
}

bool is_weight_format(int32_t id) {
  switch (WeightFormat(id)) {
    case WeightFormat::IQ2_S:
    case WeightFormat::IQ3_XXS:
    case WeightFormat::IQ3_S:
    case WeightFormat::IQ4_XS:
    case WeightFormat::IQ4_NL:
    case WeightFormat::Q2_0: return true;
  }
  return false;
}

std::size_t weight_row_bytes(WeightFormat fmt, int32_t k) {
  check_k(fmt, k);
  const FormatInfo fi = format_info(fmt);
  return std::size_t(k / fi.block_elems) * std::size_t(fi.block_bytes);
}

std::size_t act_row_bytes(ActFormat act, int32_t k) {
  if (act == ActFormat::Q8_K) {
    if (k % QK_K != 0) throw std::invalid_argument("Q8_K row length not a multiple of 256");
    return std::size_t(k / QK_K) * sizeof(block_q8_K);
  }
  if (k % QK8_0 != 0) throw std::invalid_argument("Q8_0 row length not a multiple of 32");
  return std::size_t(k / QK8_0) * sizeof(block_q8_0);
}

std::size_t decoded_row_bytes(WeightFormat fmt, int32_t k) {
  check_k(fmt, k);
  const ActFormat a = format_info(fmt).act;
  return std::size_t(groups_of(a, k)) * wgroup_bytes(chunks_of(a));
}

std::size_t packed_act_bytes(ActFormat act, int32_t k) {
  act_row_bytes(act, k);  // validates k
  return std::size_t(groups_of(act, k)) * agroup_bytes(chunks_of(act));
}

void decode_rows(WeightFormat fmt, const void* rows, std::size_t row_bytes, int32_t nrows,
                 int32_t k, void* dst_v) {
  check_k(fmt, k);
  auto* dst = static_cast<uint8_t*>(dst_v);
  const int nblk = k / format_info(fmt).block_elems;
  switch (fmt) {
    case WeightFormat::IQ2_S:
      decode_rows_q8k<block_iq2_s, dec_iq2_s>(rows, row_bytes, nrows, nblk, dst);
      return;
    case WeightFormat::IQ3_XXS:
      decode_rows_q8k<block_iq3_xxs, dec_iq3_xxs>(rows, row_bytes, nrows, nblk, dst);
      return;
    case WeightFormat::IQ3_S:
      decode_rows_q8k<block_iq3_s, dec_iq3_s>(rows, row_bytes, nrows, nblk, dst);
      return;
    case WeightFormat::IQ4_XS:
      decode_rows_q8k<block_iq4_xs, dec_iq4_xs>(rows, row_bytes, nrows, nblk, dst);
      return;
    case WeightFormat::IQ4_NL: {
      constexpr int WG = wgroup_bytes(1);
      for (int r = 0; r < nrows; ++r) {
        const auto* x = reinterpret_cast<const block_iq4_nl*>(static_cast<const uint8_t*>(rows) +
                                                              std::size_t(r) * row_bytes);
        for (int b = 0; b < nblk; b += 2, dst += WG) {
          prefetch_ahead(x + b);
          dec_iq4_nl(x + b, b + 1 < nblk ? x + b + 1 : nullptr, dst);
        }
      }
      return;
    }
    case WeightFormat::Q2_0: {
      constexpr int WG = wgroup_bytes(1);
      for (int r = 0; r < nrows; ++r) {
        const auto* x = reinterpret_cast<const block_q2_0*>(static_cast<const uint8_t*>(rows) +
                                                            std::size_t(r) * row_bytes);
        for (int b = 0; b < nblk; ++b, dst += WG) {
          prefetch_ahead(x + b);
          dec_q2_0(x[b], dst);
        }
      }
      return;
    }
  }
  throw std::invalid_argument("unknown weight format");
}

void pack_act_q8_K(const block_q8_K* x, int32_t k, void* dst) {
  act_row_bytes(ActFormat::Q8_K, k);
  auto* d = static_cast<uint8_t*>(dst);
  for (int b = 0; b < k / QK_K; ++b) pack_group_q8K(x[b], d + std::size_t(b) * agroup_bytes(4));
}

void pack_act_q8_0(const block_q8_0* x, int32_t k, void* dst) {
  act_row_bytes(ActFormat::Q8_0, k);
  auto* d = static_cast<uint8_t*>(dst);
  const int nb = k / QK8_0;
  for (int b = 0, g = 0; b < nb; b += 2, ++g)
    pack_group_q8_0(x + b, b + 1 < nb ? x + b + 1 : nullptr, d + std::size_t(g) * agroup_bytes(1));
}

void quantize_pack_bf16(ActFormat act, const uint16_t* x, int32_t k, void* dst) {
  act_row_bytes(act, k);
  auto* d = static_cast<uint8_t*>(dst);
  if (act == ActFormat::Q8_K) {
    block_q8_K q;
    for (int b = 0; b < k / QK_K; ++b) {
      quantize_row_q8_K_bf16(x + std::size_t(b) * QK_K, &q, QK_K);
      pack_group_q8K(q, d + std::size_t(b) * agroup_bytes(4));
    }
  } else {
    block_q8_0 q[2];
    const int nb = k / QK8_0;
    for (int b = 0, g = 0; b < nb; b += 2, ++g) {
      const int n = std::min(2, nb - b);
      quantize_row_q8_0_bf16(x + std::size_t(b) * QK8_0, q, n * QK8_0);
      pack_group_q8_0(&q[0], n == 2 ? &q[1] : nullptr, d + std::size_t(g) * agroup_bytes(1));
    }
  }
}

void quantize_pack_f32(ActFormat act, const float* x, int32_t k, void* dst) {
  act_row_bytes(act, k);
  auto* d = static_cast<uint8_t*>(dst);
  if (act == ActFormat::Q8_K) {
    block_q8_K q;
    for (int b = 0; b < k / QK_K; ++b) {
      quantize_row_q8_K_f32(x + std::size_t(b) * QK_K, &q, QK_K);
      pack_group_q8K(q, d + std::size_t(b) * agroup_bytes(4));
    }
  } else {
    block_q8_0 q[2];
    const int nb = k / QK8_0;
    for (int b = 0, g = 0; b < nb; b += 2, ++g) {
      const int n = std::min(2, nb - b);
      quantize_row_q8_0_f32(x + std::size_t(b) * QK8_0, q, n * QK8_0);
      pack_group_q8_0(&q[0], n == 2 ? &q[1] : nullptr, d + std::size_t(g) * agroup_bytes(1));
    }
  }
}

void quantize_pack_q8_0_half(const float* x32, int32_t group, int32_t half, void* dst) {
  block_q8_0 q;
  quantize_row_q8_0_f32(x32, &q, QK8_0);
  auto* d = static_cast<uint8_t*>(dst) + std::size_t(group) * agroup_bytes(1);
  const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q.qs));
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(d + 32 * half), v);
  _mm256_storeu_ps(reinterpret_cast<float*>(d + kChunk + 32 * half), _mm256_set1_ps(fp16(q.d)));

  const __m256i k128 = _mm256_set1_epi8(char(0x80));
  const __m256i acorr = _mm256_sub_epi32(
      _mm256_setzero_si256(), _mm256_dpbusd_epi32(_mm256_setzero_si256(), k128, v));
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(d + kChunk + 64 + 32 * half), acorr);
}

void gemm_decoded(WeightFormat fmt, const void* wdec, int32_t nrows, int32_t k,
                  const void* const* acts, int32_t T, float* out, std::size_t ldo) {
  if (nrows <= 0 || T <= 0) return;
  const ActFormat a = format_info(fmt).act;
  const int G = groups_of(a, k);
  const std::size_t ws = decoded_row_bytes(fmt, k);
  const auto* w = static_cast<const uint8_t*>(wdec);
  const auto* ap = reinterpret_cast<const uint8_t* const*>(acts);
  if (a == ActFormat::Q8_K) gemm_impl<4>(w, ws, nrows, ap, T, G, out, ldo);
  else gemm_impl<1>(w, ws, nrows, ap, T, G, out, ldo);
}

namespace {

inline void store_bf16_quarter(const uint8_t* biased_16, float s, uint16_t* dst) {
  const __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(biased_16));
  const __m128i q = _mm_xor_si128(b, _mm_set1_epi8(char(0x80)));
  const __m256 f_lo = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q)), _mm256_set1_ps(s));
  const __m256 f_hi = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q, 8))), _mm256_set1_ps(s));
  const __m256i bf = cvt2ps_to_pbh(f_hi, f_lo);
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), bf);
}

}  // namespace

// BF16 = RNE(float(w) * F) of the decoded layout, in natural element order. Each row segment is
// expanded by one decode_rows call into an L1-resident scratch, then converted in registers
// (Q8_K groups are transposed back from the VNNI lane order with shuffles).
void decode_rows_bf16(WeightFormat fmt, const void* rows, std::size_t row_bytes, int32_t nrows,
                      int32_t k, uint16_t* dst) {
  check_k(fmt, k);
  const FormatInfo fi = format_info(fmt);
  const int C = chunks_of(fi.act);
  const int elems = group_elems_of(fi.act);
  const int WG = wgroup_bytes(C);
  const int G = groups_of(fi.act, k);
  constexpr int kScratchBytes = 16384;
  const int seg_groups = kScratchBytes / WG;  // Q8_K: 51 groups, Q8_0: 128 groups
  // A Q8_0 segment boundary must fall on a whole GGUF block (32 elements, always true at 64).
  alignas(64) uint8_t scratch[kScratchBytes];
  for (int r = 0; r < nrows; ++r) {
    const auto* row = static_cast<const uint8_t*>(rows) + std::size_t(r) * row_bytes;
    uint16_t* out = dst + std::size_t(r) * k;
    for (int g0 = 0; g0 < G; g0 += seg_groups) {
      const int g1 = std::min(G, g0 + seg_groups);
      const int base = g0 * elems;
      const int seg_k = std::min(g1 * elems, k) - base;
      decode_rows(fmt, row + std::size_t(base / fi.block_elems) * fi.block_bytes, row_bytes, 1, seg_k,
                  scratch);
      for (int g = g0; g < g1; ++g) {
        const uint8_t* grp = scratch + std::size_t(g - g0) * WG;
        const auto* F = reinterpret_cast<const float*>(grp + C * kChunk);
        uint16_t* o = out + std::size_t(g) * elems;
        if (C == 4) {
          for (int j = 0; j < 16; ++j) {
            alignas(16) uint32_t sub_elem[4] = {
              reinterpret_cast<const uint32_t*>(grp)[j],
              reinterpret_cast<const uint32_t*>(grp + 64)[j],
              reinterpret_cast<const uint32_t*>(grp + 128)[j],
              reinterpret_cast<const uint32_t*>(grp + 192)[j]
            };
            store_bf16_quarter(reinterpret_cast<const uint8_t*>(sub_elem), F[j], o + 16 * j);
          }
        } else {
          const int live = std::min(elems, k - g * elems);
          store_bf16_quarter(grp + 0, F[0], o + 0);
          store_bf16_quarter(grp + 16, F[0], o + 16);
          if (live == 64) {
            store_bf16_quarter(grp + 32, F[8], o + 32);
            store_bf16_quarter(grp + 48, F[8], o + 48);
          }
        }
      }
    }
  }
}

namespace {
template<int R, int T>
void bf16_micro(const uint16_t* weights, int k, const float* x_eo,
                float* out, std::size_t ldo) {
  __m256 acc_lo[R][T];
  __m256 acc_hi[R][T];
#pragma GCC unroll 8
  for (int r = 0; r < R; ++r) {
#pragma GCC unroll 8
    for (int t = 0; t < T; ++t) {
      acc_lo[r][t] = _mm256_setzero_ps();
      acc_hi[r][t] = _mm256_setzero_ps();
    }
  }
  const int stride_t = k;  // k floats per token (for each of even_lo, odd_lo, even_hi, odd_hi -> 32 floats per i-step)
  for (int i = 0; i < k; i += 32) {
    __m256 x_even_lo[T], x_odd_lo[T], x_even_hi[T], x_odd_hi[T];
#pragma GCC unroll 8
    for (int t = 0; t < T; ++t) {
      const float* a = x_eo + std::size_t(t) * stride_t + i;
      x_even_lo[t] = _mm256_loadu_ps(a + 0);
      x_odd_lo[t]  = _mm256_loadu_ps(a + 8);
      x_even_hi[t] = _mm256_loadu_ps(a + 16);
      x_odd_hi[t]  = _mm256_loadu_ps(a + 24);
    }
#pragma GCC unroll 8
    for (int r = 0; r < R; ++r) {
      const uint16_t* w = weights + std::size_t(r) * k + i;
      const __m256 w_even_lo = _mm256_cvtneebf16_ps(reinterpret_cast<const __m256bh*>(w));
      const __m256 w_odd_lo  = _mm256_cvtneobf16_ps(reinterpret_cast<const __m256bh*>(w));
      const __m256 w_even_hi = _mm256_cvtneebf16_ps(reinterpret_cast<const __m256bh*>(w + 16));
      const __m256 w_odd_hi  = _mm256_cvtneobf16_ps(reinterpret_cast<const __m256bh*>(w + 16));
#pragma GCC unroll 8
      for (int t = 0; t < T; ++t) {
        acc_lo[r][t] = _mm256_fmadd_ps(w_even_lo, x_even_lo[t], acc_lo[r][t]);
        acc_lo[r][t] = _mm256_fmadd_ps(w_odd_lo, x_odd_lo[t], acc_lo[r][t]);
        acc_hi[r][t] = _mm256_fmadd_ps(w_even_hi, x_even_hi[t], acc_hi[r][t]);
        acc_hi[r][t] = _mm256_fmadd_ps(w_odd_hi, x_odd_hi[t], acc_hi[r][t]);
      }
    }
  }
#pragma GCC unroll 8
  for (int r = 0; r < R; ++r) {
#pragma GCC unroll 8
    for (int t = 0; t < T; ++t) {
      const __m256 sum8 = _mm256_add_ps(acc_lo[r][t], acc_hi[r][t]);
      out[std::size_t(r) * ldo + t] = hsum256_ps(sum8);
    }
  }
}

template<int T>
void bf16_col_strip(const uint16_t* weights, int nrows, int k, const float* x_eo,
                    float* out, std::size_t ldo) {
  int r = 0;
  for (; r + 2 <= nrows; r += 2)
    bf16_micro<2, T>(weights + std::size_t(r) * k, k, x_eo, out + std::size_t(r) * ldo, ldo);
  for (; r < nrows; ++r)
    bf16_micro<1, T>(weights + std::size_t(r) * k, k, x_eo, out + std::size_t(r) * ldo, ldo);
}
}  // namespace

void gemm_bf16(const uint16_t* weights, int32_t nrows, int32_t k,
               const void* const* acts, int32_t T, float* out, std::size_t ldo) {
  if (nrows <= 0 || T <= 0) return;
  thread_local Scratch xbuf;
  float* x_eo = reinterpret_cast<float*>(xbuf.get(std::size_t(T) * k * sizeof(float)));
  for (int t = 0; t < T; ++t) {
    const uint16_t* a = static_cast<const uint16_t*>(acts[t]);
    float* dst = x_eo + std::size_t(t) * k;
    for (int i = 0; i < k; i += 32) {
      _mm256_storeu_ps(dst + i + 0,  _mm256_cvtneebf16_ps(reinterpret_cast<const __m256bh*>(a + i)));
      _mm256_storeu_ps(dst + i + 8,  _mm256_cvtneobf16_ps(reinterpret_cast<const __m256bh*>(a + i)));
      _mm256_storeu_ps(dst + i + 16, _mm256_cvtneebf16_ps(reinterpret_cast<const __m256bh*>(a + i + 16)));
      _mm256_storeu_ps(dst + i + 24, _mm256_cvtneobf16_ps(reinterpret_cast<const __m256bh*>(a + i + 16)));
    }
  }
  int t = 0;
  for (; t + 3 <= T; t += 3)
    bf16_col_strip<3>(weights, nrows, k, x_eo + std::size_t(t) * k, out + t, ldo);
  if (t + 2 <= T) {
    bf16_col_strip<2>(weights, nrows, k, x_eo + std::size_t(t) * k, out + t, ldo);
    t += 2;
  }
  if (t < T)
    bf16_col_strip<1>(weights, nrows, k, x_eo + std::size_t(t) * k, out + t, ldo);
}

void silu_mul(const float* g, const float* u, float* h, int32_t n) {
  for (int32_t i = 0; i < n; i += 8) {
    const int rem = n - i;
    if (rem >= 8) {
      const __m256 gv = _mm256_loadu_ps(g + i);
      const __m256 uv = _mm256_loadu_ps(u + i);
      const __m256 e = exp256(_mm256_sub_ps(_mm256_setzero_ps(), gv));
      const __m256 s = _mm256_div_ps(gv, _mm256_add_ps(_mm256_set1_ps(1.0f), e));
      _mm256_storeu_ps(h + i, _mm256_mul_ps(s, uv));
    } else {
      alignas(32) float gb[8] = {0}, ub[8] = {0}, hb[8];
      std::memcpy(gb, g + i, rem * sizeof(float));
      std::memcpy(ub, u + i, rem * sizeof(float));
      const __m256 gv = _mm256_load_ps(gb);
      const __m256 uv = _mm256_load_ps(ub);
      const __m256 e = exp256(_mm256_sub_ps(_mm256_setzero_ps(), gv));
      const __m256 s = _mm256_div_ps(gv, _mm256_add_ps(_mm256_set1_ps(1.0f), e));
      _mm256_store_ps(hb, _mm256_mul_ps(s, uv));
      std::memcpy(h + i, hb, rem * sizeof(float));
    }
  }
}

void cpu_dot(WeightFormat fmt, const void* row, int nblk, const void* acts, int T,
             std::size_t act_stride_bytes, float* out) {
  dot_impl(
      fmt, row, nblk, T,
      [](const void* a, std::size_t s, int t) { return static_cast<const uint8_t*>(a) + std::size_t(t) * s; },
      acts, act_stride_bytes, out);
}

void cpu_dot_ptr(WeightFormat fmt, const void* row, int nblk, const void* const* acts, int T,
                 float* out) {
  dot_impl(
      fmt, row, nblk, T,
      [](const void* a, std::size_t, int t) {
        return static_cast<const uint8_t*>(static_cast<const void* const*>(a)[t]);
      },
      acts, 0, out);
}

}  // namespace fngine::cpu
#endif
