# Modified for fngine from cinference:cmake/Dependencies.cmake @ 383e5db. See NOTICE.
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBURING REQUIRED IMPORTED_TARGET liburing)

# Repository-pinned header dependencies. No configure-time downloads.
add_library(fngine::json INTERFACE IMPORTED GLOBAL)
target_include_directories(fngine::json INTERFACE ${PROJECT_SOURCE_DIR}/third_party)

add_library(fngine::httplib INTERFACE IMPORTED GLOBAL)
target_include_directories(fngine::httplib INTERFACE
  ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)

add_subdirectory(third_party/llama-jinja EXCLUDE_FROM_ALL)
add_subdirectory(third_party/spdlog EXCLUDE_FROM_ALL)

# Internal headers: public API in include/, implementation in src/.
function(fngine_internal_includes target)
  target_include_directories(${target} PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/third_party)
endfunction()

function(fngine_cuda_target target)
  set_target_properties(${target} PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --expt-relaxed-constexpr>)
endfunction()

# Host code on the CPU expert path uses one of two backends, chosen from FNGINE_CPU_ARCH
# (-march, default native). AVX-512 wins when the target has both.
#   * AVX-512 F/BW/DQ/VL with VNNI, BF16, VBMI, BITALG and VPOPCNTDQ
#     (Zen 4/5, Intel Sapphire Rapids, Granite Rapids and later)
#   * AVX2 + FMA + AVX-VNNI + AVX-NE-CONVERT + F16C
#     (Intel Arrow Lake / Core Ultra 200S, which have no AVX-512)
# GCC 13 has no -march=arrowlake; -march=native selects that backend there.
set(FNGINE_CPU_ARCH "native" CACHE STRING "-march target for the CPU expert path")
set(FNGINE_CPU_FLAGS -O3 -march=${FNGINE_CPU_ARCH})

include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_FLAGS "-march=${FNGINE_CPU_ARCH}")
if(NOT "${FNGINE_CPU_ARCH}" STREQUAL "${FNGINE_CPU_ARCH_CHECKED}")
  unset(FNGINE_MARCH_HAS_AVX512 CACHE)
  unset(FNGINE_MARCH_HAS_AVX2_VNNI CACHE)
  set(FNGINE_CPU_ARCH_CHECKED "${FNGINE_CPU_ARCH}" CACHE INTERNAL
      "FNGINE_CPU_ARCH last used to detect the CPU expert backend")
endif()
check_cxx_source_compiles("
#if !defined(__AVX512F__) || !defined(__AVX512BW__) || !defined(__AVX512VL__) || !defined(__AVX512DQ__) || !defined(__AVX512VNNI__) || !defined(__AVX512BF16__) || !defined(__AVX512VBMI__) || !defined(__AVX512BITALG__) || !defined(__AVX512VPOPCNTDQ__)
#error no avx512 expert backend
#endif
int main() { return 0; }
" FNGINE_MARCH_HAS_AVX512)
check_cxx_source_compiles("
#if !defined(__AVX2__) || !defined(__FMA__) || !defined(__AVXVNNI__) || !defined(__AVXNECONVERT__) || !defined(__F16C__)
#error no avx2 expert backend
#endif
int main() { return 0; }
" FNGINE_MARCH_HAS_AVX2_VNNI)
unset(CMAKE_REQUIRED_FLAGS)
if(FNGINE_MARCH_HAS_AVX512)
  set(FNGINE_CPU_BACKEND_AVX512 1)
  set(FNGINE_CPU_BACKEND_AVX2 0)
  set(FNGINE_CPU_BACKEND_NAME "AVX-512")
elseif(FNGINE_MARCH_HAS_AVX2_VNNI)
  set(FNGINE_CPU_BACKEND_AVX512 0)
  set(FNGINE_CPU_BACKEND_AVX2 1)
  set(FNGINE_CPU_BACKEND_NAME "AVX2+AVX-VNNI+AVX-NE-CONVERT")
else()
  message(FATAL_ERROR
    "FNGINE_CPU_ARCH='${FNGINE_CPU_ARCH}' enables neither the AVX-512 expert backend "
    "(F/BW/DQ/VL/VNNI/BF16/VBMI/BITALG/VPOPCNTDQ) nor AVX2+FMA+AVX-VNNI+AVX-NE-CONVERT+F16C")
endif()
add_compile_definitions(
  FNGINE_CPU_BACKEND_AVX512=${FNGINE_CPU_BACKEND_AVX512}
  FNGINE_CPU_BACKEND_AVX2=${FNGINE_CPU_BACKEND_AVX2})
message(STATUS "CPU expert backend (${FNGINE_CPU_ARCH}): ${FNGINE_CPU_BACKEND_NAME}")

# Test oracles: no reassociation or contraction.
set(FNGINE_ORACLE_FLAGS -fno-fast-math -ffp-contract=off)
