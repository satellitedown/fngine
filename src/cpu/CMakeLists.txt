add_library(fngine_cpu STATIC
  expert_pool.cpp
  iq_cpu.cpp
  iq_cpu_avx2.cpp
  q8_cpu.cpp
  q8_cpu_avx2.cpp
)
fngine_internal_includes(fngine_cpu)
target_compile_options(fngine_cpu PRIVATE ${FNGINE_CPU_FLAGS})
# The quantizers must be bit-identical to ggml's reference: no FMA contraction.
# Both ISA files are compiled; cpu_isa.h keeps exactly one of them.
set_source_files_properties(q8_cpu.cpp q8_cpu_avx2.cpp PROPERTIES COMPILE_OPTIONS -ffp-contract=off)
target_link_libraries(fngine_cpu PUBLIC fngine_core Threads::Threads)
