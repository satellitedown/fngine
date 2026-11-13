// AVX2 + AVX-VNNI + AVX-NE-CONVERT Q8_K / Q8_0 quantizers.
// Bit-exact contract matches q8_cpu.cpp (the AVX-512 path). Exactly one of the two is compiled.
#include "cpu/cpu_isa.h"
#if FNGINE_CPU_BACKEND_AVX2
#include "cpu/q8_cpu.h"

#include <immintrin.h>

#include <cstring>

// Bit-exactness notes (this file is compiled with -ffp-contract=off, see CMakeLists.txt):
//  * Q8_K: the reference rounds with nearest_int() (add 1.5*2^23, round-to-nearest-even), which
//    equals vcvtps2dq under the default MXCSR rounding mode; the signed max is the FIRST element
//    of largest magnitude.
//  * Q8_0: the reference uses id = 1/(amax/127) and roundf() (half away from zero).
namespace fngine::cpu {
namespace {

inline __m256 load8(const float* p) { return _mm256_loadu_ps(p); }
inline __m256 load8(const uint16_t* p) {
  const __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
  return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16));
}
inline float scalar_at(const float* p, int i) { return p[i]; }
inline float scalar_at(const uint16_t* p, int i) {
  const uint32_t w = uint32_t(p[i]) << 16;
  float f;
  std::memcpy(&f, &w, 4);
  return f;
}

inline __m256 abs8(__m256 v) {
  return _mm256_and_ps(v, _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff)));
}

inline float hmax256(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  __m128 m = _mm_max_ps(lo, hi);
  m = _mm_max_ps(m, _mm_movehl_ps(m, m));
  m = _mm_max_ss(m, _mm_shuffle_ps(m, m, 1));
  return _mm_cvtss_f32(m);
}

inline int32_t hsum_epi32(__m256i v) {
  __m128i lo = _mm256_castsi256_si128(v);
  __m128i hi = _mm256_extracti128_si256(v, 1);
  __m128i sum = _mm_add_epi32(lo, hi);
  sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(1, 0, 3, 2)));
  sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(2, 3, 0, 1)));
  return _mm_cvtsi128_si32(sum);
}

inline __m128i pack16_epi32_to_epi8(__m256i lo, __m256i hi) {
  __m128i a0 = _mm256_castsi256_si128(lo);
  __m128i a1 = _mm256_extracti128_si256(lo, 1);
  __m128i b0 = _mm256_castsi256_si128(hi);
  __m128i b1 = _mm256_extracti128_si256(hi, 1);
  __m128i p0 = _mm_packs_epi32(a0, a1);
  __m128i p1 = _mm_packs_epi32(b0, b1);
  return _mm_packs_epi16(p0, p1);
}

template <class T>
void q8K_block(const T* x, block_q8_K* y) {
  __m256 v[32];
  __m256 amaxv = _mm256_setzero_ps();
  for (int i = 0; i < 32; ++i) {
    v[i] = load8(x + 8 * i);
    amaxv = _mm256_max_ps(amaxv, abs8(v[i]));
  }
  const float amax = hmax256(amaxv);
  if (!amax) {
    y->d = 0;
    std::memset(y->qs, 0, sizeof(y->qs));
    std::memset(y->bsums, 0, sizeof(y->bsums));
    return;
  }
  const __m256 amax8 = _mm256_set1_ps(amax);
  int idx = 0;
  for (int i = 0; i < 32; ++i) {
    const __m256 cmp = _mm256_cmp_ps(abs8(v[i]), amax8, _CMP_EQ_OQ);
    const int m = _mm256_movemask_ps(cmp);
    if (m) {
      idx = 8 * i + __builtin_ctz(m);
      break;
    }
  }
  const float max = scalar_at(x, idx);
  const float iscale = -127.f / max;
  const __m256 is = _mm256_set1_ps(iscale);
  const __m256i lim = _mm256_set1_epi32(127);
  for (int i = 0; i < 16; ++i) {
    __m256i q0 = _mm256_cvtps_epi32(_mm256_mul_ps(v[2 * i], is));
    __m256i q1 = _mm256_cvtps_epi32(_mm256_mul_ps(v[2 * i + 1], is));
    q0 = _mm256_min_epi32(q0, lim);
    q1 = _mm256_min_epi32(q1, lim);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(y->qs + 16 * i), pack16_epi32_to_epi8(q0, q1));
    const __m256i sum = _mm256_add_epi32(q0, q1);
    y->bsums[i] = static_cast<int16_t>(hsum_epi32(sum));
  }
  y->d = 1.0f / iscale;
}

inline __m256i roundf8(__m256 x) {
  __m256 t = _mm256_round_ps(x, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
  const __m256 diff = _mm256_sub_ps(x, t);
  const __m256 up = _mm256_cmp_ps(abs8(diff), _mm256_set1_ps(0.5f), _CMP_GE_OQ);
  const __m256 one = _mm256_or_ps(
      _mm256_set1_ps(1.0f),
      _mm256_and_ps(x, _mm256_castsi256_ps(_mm256_set1_epi32(int(0x80000000u)))));
  t = _mm256_add_ps(t, _mm256_and_ps(up, one));
  return _mm256_cvttps_epi32(t);
}

template <class T>
void q8_0_block(const T* x, block_q8_0* y) {
  const __m256 v0 = load8(x);
  const __m256 v1 = load8(x + 8);
  const __m256 v2 = load8(x + 16);
  const __m256 v3 = load8(x + 24);
  const __m256 m01 = _mm256_max_ps(abs8(v0), abs8(v1));
  const __m256 m23 = _mm256_max_ps(abs8(v2), abs8(v3));
  const float amax = hmax256(_mm256_max_ps(m01, m23));
  const float d = amax / ((1 << 7) - 1);
  const float id = d ? 1.0f / d : 0.0f;
  y->d = GGML_FP32_TO_FP16(d);
  const __m256 m = _mm256_set1_ps(id);
  const __m256i q0 = roundf8(_mm256_mul_ps(v0, m));
  const __m256i q1 = roundf8(_mm256_mul_ps(v1, m));
  const __m256i q2 = roundf8(_mm256_mul_ps(v2, m));
  const __m256i q3 = roundf8(_mm256_mul_ps(v3, m));
  _mm_storeu_si128(reinterpret_cast<__m128i*>(y->qs), pack16_epi32_to_epi8(q0, q1));
  _mm_storeu_si128(reinterpret_cast<__m128i*>(y->qs + 16), pack16_epi32_to_epi8(q2, q3));
}

}  // namespace

void quantize_row_q8_K_bf16(const uint16_t* x, block_q8_K* y, int64_t k) {
  for (int64_t i = 0; i < k / QK_K; ++i) q8K_block(x + i * QK_K, y + i);
}
void quantize_row_q8_K_f32(const float* x, block_q8_K* y, int64_t k) {
  for (int64_t i = 0; i < k / QK_K; ++i) q8K_block(x + i * QK_K, y + i);
}
void quantize_row_q8_0_bf16(const uint16_t* x, block_q8_0* y, int64_t k) {
  for (int64_t i = 0; i < k / QK8_0; ++i) q8_0_block(x + i * QK8_0, y + i);
}
void quantize_row_q8_0_f32(const float* x, block_q8_0* y, int64_t k) {
  for (int64_t i = 0; i < k / QK8_0; ++i) q8_0_block(x + i * QK8_0, y + i);
}

}  // namespace fngine::cpu
#endif
