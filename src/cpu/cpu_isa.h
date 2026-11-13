#pragma once

// Compile-time CPU expert backend. FNGINE_CPU_ARCH (-march, default native) decides:
//   AVX-512 F/BW/DQ/VL + VNNI + BF16 + VBMI + BITALG + VPOPCNTDQ
//     AMD Zen 4/5, Intel Sapphire Rapids, Granite Rapids and later
//   AVX2 + FMA + AVX-VNNI + AVX-NE-CONVERT + F16C
//     Intel Arrow Lake / Core Ultra 200S (no AVX-512)
// AVX-512 is selected when the target has both.
//
// CMake defines FNGINE_CPU_BACKEND_AVX512 and FNGINE_CPU_BACKEND_AVX2 (0/1) on every
// target, including translation units that are not themselves compiled with -march.
// A manual object compile can omit those defines; this header then reads the -m macros.

#if !defined(FNGINE_CPU_BACKEND_AVX512) && !defined(FNGINE_CPU_BACKEND_AVX2)
#  if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__) && defined(__AVX512DQ__) && \
      defined(__AVX512VNNI__) && defined(__AVX512BF16__) && defined(__AVX512VBMI__) &&                    \
      defined(__AVX512BITALG__) && defined(__AVX512VPOPCNTDQ__)
#    define FNGINE_CPU_BACKEND_AVX512 1
#    define FNGINE_CPU_BACKEND_AVX2 0
#  elif defined(__AVX2__) && defined(__FMA__) && defined(__AVXVNNI__) && defined(__AVXNECONVERT__) && defined(__F16C__)
#    define FNGINE_CPU_BACKEND_AVX512 0
#    define FNGINE_CPU_BACKEND_AVX2 1
#  else
#    error "fngine CPU expert path needs AVX-512 (F/BW/DQ/VL/VNNI/BF16/VBMI/BITALG/VPOPCNTDQ) or AVX2+FMA+AVX-VNNI+AVX-NE-CONVERT+F16C. Set FNGINE_CPU_ARCH to a -march that provides one (default: native)."
#  endif
#endif

#if !defined(FNGINE_CPU_BACKEND_AVX512) || !defined(FNGINE_CPU_BACKEND_AVX2)
#  error "set both FNGINE_CPU_BACKEND_AVX512 and FNGINE_CPU_BACKEND_AVX2 to 0 or 1"
#endif
#if (FNGINE_CPU_BACKEND_AVX512 + FNGINE_CPU_BACKEND_AVX2) != 1
#  error "select exactly one fngine CPU expert backend"
#endif

namespace fngine::cpu {

inline const char* cpu_backend_name() noexcept {
#if FNGINE_CPU_BACKEND_AVX512
  return "AVX-512";
#else
  return "AVX2+AVX-VNNI+AVX-NE-CONVERT";
#endif
}

// True when the CPU running this process has the ISA this binary was compiled for.
inline bool cpu_backend_usable() noexcept {
  __builtin_cpu_init();
#if FNGINE_CPU_BACKEND_AVX512
  return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
         __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512dq") &&
         __builtin_cpu_supports("avx512vnni") && __builtin_cpu_supports("avx512bf16") &&
         __builtin_cpu_supports("avx512vbmi") && __builtin_cpu_supports("avx512bitalg") &&
         __builtin_cpu_supports("avx512vpopcntdq") && __builtin_cpu_supports("f16c");
#else
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") &&
         __builtin_cpu_supports("avxvnni") && __builtin_cpu_supports("avxneconvert") &&
         __builtin_cpu_supports("f16c");
#endif
}

}  // namespace fngine::cpu
