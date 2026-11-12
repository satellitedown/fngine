// Modified for fngine from cinference:src/ops/kernel/argmax.cuh @ 383e5db. See NOTICE.
// (Also merges cinference:src/ops/launcher/argmax.cu.)
//
// FP32 logits [T][248320]. One CTA reduces an 8-items-per-thread tile and an atomic CAS winner
// selects the exact value/lower-id maximum per column (the output is zeroed first).

#include "core/device.h"
#include "core/model_dims.h"
#include "kernels/common/math.h"
#include "kernels/head/head.h"

#include <climits>
#include <cstdint>
#include <math_constants.h>
#include <stdexcept>

namespace fngine::kernels {
namespace {

inline constexpr int kArgmaxBlock          = 512;
inline constexpr int kArgmaxItemsPerThread = 8;
// Many columns: a smaller block gives the 2-D grid enough resident CTAs.
inline constexpr int kArgmaxWideBlock      = 128;
inline constexpr int kArgmaxWideColumns    = 8;

__device__ __forceinline__ bool argmax_better(float value, std::int32_t index, float best_value,
                                              std::int32_t best_index) {
    return value > best_value || (value == best_value && index < best_index);
}

__device__ __forceinline__ void argmax_warp_reduce(float& value, std::int32_t& index) {
    constexpr unsigned int kMask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other_value        = __shfl_down_sync(kMask, value, offset);
        const std::int32_t other_index = __shfl_down_sync(kMask, index, offset);
        if (argmax_better(other_value, other_index, value, index)) {
            value = other_value;
            index = other_index;
        }
    }
}

__device__ __forceinline__ void argmax_block_reduce(float& value, std::int32_t& index) {
    __shared__ float warp_values[16];
    __shared__ std::int32_t warp_indices[16];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    argmax_warp_reduce(value, index);
    if (lane == 0) {
        warp_values[warp]  = value;
        warp_indices[warp] = index;
    }
    __syncthreads();

    value = (lane < (blockDim.x >> 5)) ? warp_values[lane] : -CUDART_INF_F;
    index = (lane < (blockDim.x >> 5)) ? warp_indices[lane] : INT32_MAX;
    if (warp == 0) { argmax_warp_reduce(value, index); }
}

__launch_bounds__(kArgmaxBlock) __global__
    void argmax_tiled_atomic_kernel(const float* logits, std::int32_t* out,
                                    std::int32_t valid_rows, std::int32_t physical_rows) {
    const std::int32_t t    = static_cast<std::int32_t>(blockIdx.y);
    const std::int64_t base = static_cast<std::int64_t>(t) * physical_rows;
    const std::int32_t tile_start =
        static_cast<std::int32_t>(blockIdx.x) * blockDim.x * kArgmaxItemsPerThread;

    float best_value        = -CUDART_INF_F;
    std::int32_t best_index = INT32_MAX;
#pragma unroll
    for (int item = 0; item < kArgmaxItemsPerThread; ++item) {
        const std::int32_t v = tile_start + threadIdx.x + item * blockDim.x;
        if (v < valid_rows) {
            const float value = logits[base + v];
            if (argmax_better(value, v, best_value, best_index)) {
                best_value = value;
                best_index = v;
            }
        }
    }

    argmax_block_reduce(best_value, best_index);
    if (threadIdx.x != 0 || best_index == INT32_MAX) { return; }

    int current = out[t];
    while (true) {
        const float current_value = logits[base + current];
        if (!argmax_better(best_value, best_index, current_value, current)) { break; }
        const int observed = atomicCAS(reinterpret_cast<int*>(out + t), current, best_index);
        if (observed == current) { break; }
        current = observed;
    }
}

} // namespace

void argmax(const float* logits, std::int32_t T, std::int32_t token_domain, std::int32_t* out,
            cudaStream_t stream) {
    if (T == 0) { return; }
    if (token_domain <= 0 || token_domain > dims::kVocab) {
        throw std::invalid_argument("argmax: token_domain must be in [1, 248320]");
    }
    const int block        = T <= kArgmaxWideColumns ? kArgmaxBlock : kArgmaxWideBlock;
    const int tiled_blocks = div_up(token_domain, block * kArgmaxItemsPerThread);
    CUDA_CHECK(cudaMemsetAsync(out, 0, static_cast<std::size_t>(T) * sizeof(std::int32_t), stream));
    const dim3 grid(static_cast<unsigned int>(tiled_blocks), static_cast<unsigned int>(T));
    argmax_tiled_atomic_kernel<<<grid, block, 0, stream>>>(logits, out, token_domain, dims::kVocab);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace fngine::kernels
