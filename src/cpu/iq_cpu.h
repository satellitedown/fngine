// Modified for fngine from llama.cpp/ggml:ggml/src/ggml-cpu/arch/x86/quants.c @ 3cf03257f219afbe7334045ff7c6a06ac68c627d. See NOTICE.
// SPDX-License-Identifier: MIT
#pragma once

// CPU expert matmul for the GGUF expert weight formats.
// AVX-512 VNNI/VBMI/BITALG/BF16 (iq_cpu.cpp: Zen 4/5, Sapphire Rapids, Granite Rapids)
// or AVX2 + FMA + AVX-VNNI + AVX-NE-CONVERT (iq_cpu_avx2.cpp: Arrow Lake / Core Ultra 200S).
// cpu_isa.h compiles exactly one. Both use the layouts below.
//
// Two kernel-native layouts, both built from 64-element "chunks" (one zmm, or two ymm, of int8):
//
//  * Decoded weights: each GGUF block is expanded ONCE (grid lookups, signs applied) into
//    biased uint8 values w + 128, plus one float vector of per-lane scales:
//        group = [C x 64 B  w + 128][64 B float F]
//  * Packed activations (Q8_K / Q8_0 quantization, ggml reference semantics):
//        group = [C x 64 B  int8 q][64 B float DA][64 B int32 ACORR = -128 * sum of the lane]
//
//  Q8_K weight formats (IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS): group = one 256-element block, C = 4.
//    Elements are permuted so that int32 lane j always holds elements 16j..16j+15 of the block
//    (chunk c, lane j, byte b <- element 16j + 4c + b); every lane then lies in a single
//    sub-block and the sub-block scale becomes a per-lane factor F[j] that is the same for all
//    four chunks. DA = the Q8_K block scale.
//  Q8_0 weight formats (IQ4_NL, Q2_0): group = 64 elements in natural order, C = 1; lanes 0-7
//    and 8-15 are the two 32-element halves. DA lanes = the two Q8_0 block scales.
//
// Dot product of weight row r and activation row t, per group (identical for every tile shape,
// so results do not depend on T or on a token's position in the batch):
//    isum = ACORR_t + sum_c vpdpbusd(w_r[c], q_t[c])          exact int32 per lane
//    acc  = fma(float(isum), F_r * DA_t, acc)                     float(isum) is exact (< 2^24)
// and finally out = horizontal sum of acc's 16 lanes.
#include "ggml-iq/ggml-iq.h"

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace fngine::cpu {

// Values are the GGML type ids (Q2_0 = 42 in the vendored ggml).
enum class WeightFormat : int32_t {
  IQ3_XXS = 18,
  IQ4_NL = 20,
  IQ3_S = 21,
  IQ2_S = 22,
  IQ4_XS = 23,
  Q2_0 = 42,
};

enum class ActFormat : int32_t { Q8_K, Q8_0 };

struct FormatInfo {
  int32_t block_elems;  // weight elements per GGUF block
  int32_t block_bytes;  // bytes per GGUF block
  ActFormat act;        // activation quantization the format pairs with
};

// Throws std::invalid_argument for unknown values.
FormatInfo format_info(WeightFormat fmt);
const char* format_name(WeightFormat fmt);
// Accepts the manifest names ("IQ2_S", "IQ3_XXS", "IQ3_S", "IQ4_XS", "IQ4_NL", "Q2_0").
bool parse_weight_format(std::string_view name, WeightFormat* out);
bool is_weight_format(int32_t ggml_type_id);

// Bytes of one GGUF weight row of k elements (k % block_elems == 0).
std::size_t weight_row_bytes(WeightFormat fmt, int32_t k);
// Bytes of one ggml activation row (block_q8_K / block_q8_0) of k elements.
std::size_t act_row_bytes(ActFormat act, int32_t k);

// ------------------------------------------------------------------ kernel-native layouts
// k: Q8_K formats need k % 256 == 0; Q8_0 formats k % 32 == 0 (Q2_0: k % 64 == 0); a trailing
// half group is zero-padded. All buffers must be 64-byte aligned.
std::size_t decoded_row_bytes(WeightFormat fmt, int32_t k);
std::size_t packed_act_bytes(ActFormat act, int32_t k);

// Expands nrows GGUF rows (row i at rows + i * row_bytes) into dst (row stride
// decoded_row_bytes). Prefetches ahead within each row.
void decode_rows(WeightFormat fmt, const void* rows, std::size_t row_bytes, int32_t nrows,
                 int32_t k, void* dst);

// ggml activation blocks -> packed row.
void pack_act_q8_K(const block_q8_K* x, int32_t k, void* dst);
void pack_act_q8_0(const block_q8_0* x, int32_t k, void* dst);
// Quantize (bit-identical to quantize_row_q8_{K,0}_ref) and pack one row.
void quantize_pack_bf16(ActFormat act, const uint16_t* x, int32_t k, void* dst);
void quantize_pack_f32(ActFormat act, const float* x, int32_t k, void* dst);
// Quantizes 32 floats as one Q8_0 block and stores it as half `half` (0/1) of packed Q8_0
// group `group` of dst (the other half is left untouched).
void quantize_pack_q8_0_half(const float* x32, int32_t group, int32_t half, void* dst);

// out[r * ldo + t] = dot(decoded row r, packed activation row acts[t]) for r < nrows, t < T.
// wdec rows are contiguous (stride decoded_row_bytes(fmt, k)). Keep nrows small (<= 8) so the
// decoded rows stay in L1 while the token tiles stream through.
void gemm_decoded(WeightFormat fmt, const void* wdec, int32_t nrows, int32_t k,
                  const void* const* acts, int32_t T, float* out, std::size_t ldo);

// Prefill uses the same BF16-dequantized weights and BF16 activations as GPU grouped GEMM.
// Decode keeps the Q8/VNNI path above. Caller supplies contiguous uint16 BF16 weight rows;
// activation rows may be noncontiguous. All accumulation and outputs are FP32.
void decode_rows_bf16(WeightFormat fmt, const void* rows, std::size_t row_bytes, int32_t nrows,
                      int32_t k, uint16_t* dst);
void gemm_bf16(const uint16_t* weights, int32_t nrows, int32_t k,
               const void* const* acts, int32_t T, float* out, std::size_t ldo);

// h[i] = silu(g[i]) * u[i] (vectorized exp, max rel. error ~2 ulp; deterministic).
void silu_mul(const float* g, const float* u, float* h, int32_t n);

// ------------------------------------------------------------------ convenience
// One GGUF row against T ggml-format activation rows (row t at acts + t * act_stride_bytes, or
// acts[t]): out[t] = dot. `nblk` counts GGUF weight blocks. Uses thread-local scratch; meant for
// tests and cost probes, the expert pool uses the layouts above directly.
void cpu_dot(WeightFormat fmt, const void* row, int nblk, const void* acts, int T,
             std::size_t act_stride_bytes, float* out);
void cpu_dot_ptr(WeightFormat fmt, const void* row, int nblk, const void* const* acts, int T,
                 float* out);

}  // namespace fngine::cpu
