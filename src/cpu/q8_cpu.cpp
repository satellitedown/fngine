#include "cpu/cpu_isa.h"
#if FNGINE_CPU_BACKEND_AVX512
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

inline __m512 load16(const float* p) { return _mm512_loadu_ps(p); }
inline __m512 load16(const uint16_t* p) {
  const __m256i h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
  return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(h), 16));
}
inline float scalar_at(const float* p, int i) { return p[i]; }
inline float scalar_at(const uint16_t* p, int i) {
  const uint32_t w = uint32_t(p[i]) << 16;
  float f;
  std::memcpy(&f, &w, 4);
  return f;
}

inline __m512 abs16(__m512 v) {
  return _mm512_castsi512_ps(
      _mm512_and_si512(_mm512_castps_si512(v), _mm512_set1_epi32(0x7fffffff)));
}

template <class T>
void q8K_block(const T* x, block_q8_K* y) {
  __m512 v[16];
  __m512 amaxv = _mm512_setzero_ps();
  for (int i = 0; i < 16; ++i) {
    v[i] = load16(x + 16 * i);
    amaxv = _mm512_max_ps(amaxv, abs16(v[i]));
  }
  const float amax = _mm512_reduce_max_ps(amaxv);
  if (!amax) {
    y->d = 0;
    std::memset(y->qs, 0, sizeof(y->qs));
    std::memset(y->bsums, 0, sizeof(y->bsums));
    return;
  }
  const __m512 amax16 = _mm512_set1_ps(amax);
  int idx = 0;
  for (int i = 0; i < 16; ++i) {
    const __mmask16 m = _mm512_cmp_ps_mask(abs16(v[i]), amax16, _CMP_EQ_OQ);
    if (m) {
      idx = 16 * i + __builtin_ctz(m);
      break;
    }
  }
  const float max = scalar_at(x, idx);
  const float iscale = -127.f / max;
  const __m512 is = _mm512_set1_ps(iscale);
  const __m512i lim = _mm512_set1_epi32(127);
  for (int i = 0; i < 16; ++i) {
    __m512i q = _mm512_cvtps_epi32(_mm512_mul_ps(v[i], is));
    q = _mm512_min_epi32(q, lim);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(y->qs + 16 * i), _mm512_cvtepi32_epi8(q));
    y->bsums[i] = static_cast<int16_t>(_mm512_reduce_add_epi32(q));
  }
  y->d = 1 / iscale;
}

inline __m512i roundf16(__m512 x) {
  __m512 t = _mm512_roundscale_ps(x, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
  const __m512 diff = _mm512_sub_ps(x, t);  // exact
  const __mmask16 up = _mm512_cmp_ps_mask(abs16(diff), _mm512_set1_ps(0.5f), _CMP_GE_OQ);
  const __m512 one = _mm512_castsi512_ps(_mm512_or_si512(
      _mm512_castps_si512(_mm512_set1_ps(1.0f)),
      _mm512_and_si512(_mm512_castps_si512(x), _mm512_set1_epi32(int(0x80000000u)))));
  t = _mm512_mask_add_ps(t, up, t, one);
  return _mm512_cvttps_epi32(t);
}

template <class T>
void q8_0_block(const T* x, block_q8_0* y) {
  const __m512 v0 = load16(x), v1 = load16(x + 16);
  const float amax = _mm512_reduce_max_ps(_mm512_max_ps(abs16(v0), abs16(v1)));
  const float d = amax / ((1 << 7) - 1);
  const float id = d ? 1.0f / d : 0.0f;
  y->d = GGML_FP32_TO_FP16(d);
  const __m512 m = _mm512_set1_ps(id);
  const __m512i q0 = roundf16(_mm512_mul_ps(v0, m));
  const __m512i q1 = roundf16(_mm512_mul_ps(v1, m));
  _mm_storeu_si128(reinterpret_cast<__m128i*>(y->qs), _mm512_cvtepi32_epi8(q0));
  _mm_storeu_si128(reinterpret_cast<__m128i*>(y->qs + 16), _mm512_cvtepi32_epi8(q1));
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
