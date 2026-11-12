// Modified for fngine from cinference:src/ops/kernel/sampling_device.cuh @ 383e5db. See NOTICE.
#pragma once

// Shared implementation primitives for sampling and speculative acceptance over FP32 logits.
// The ordering key is exact for finite logits (including numeric-zero ties); candidate storage is
// bounded by the semantic top-20 cap and all global staging is supplied by the caller.

#include "kernels/common/score_id_order.cuh"
#include "kernels/head/head.h"
#include "kernels/head/sampling_workspace.h"

#include <cub/block/block_merge_sort.cuh>
#include <cub/warp/warp_merge_sort.cuh>

#include <climits>
#include <cstdint>
#include <math_constants.h>

namespace fngine::kernels {

using SamplingPartialSort =
    cub::BlockMergeSort<unsigned long long, kSamplerBlock, kSamplerItemsPerThread>;
using SamplingGroupSort =
    cub::BlockMergeSort<unsigned long long, kSamplerGroupBlock, kSamplerGroupItemsPerThread>;
inline constexpr int kSamplingTileWarps = kSamplerBlock / 32;

using SamplingTileWarpSort = cub::WarpMergeSort<unsigned long long, kSamplerItemsPerThread>;

struct SamplingTileTopKStorage {
    typename SamplingTileWarpSort::TempStorage warp[kSamplingTileWarps];
    unsigned long long candidates[kSamplingTileWarps * kSamplerCandidateCap];
};

struct SamplingKeyGreater {
    __device__ __forceinline__ bool operator()(unsigned long long a, unsigned long long b) const {
        return a > b;
    }
};

__device__ inline unsigned long long sampling_block_max_key(unsigned long long key,
                                                            unsigned long long* warp_keys) {
    constexpr unsigned int kMask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        const unsigned long long other = __shfl_down_sync(kMask, key, offset);
        if (other > key) { key = other; }
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) { warp_keys[warp] = key; }
    __syncthreads();
    key = (warp == 0 && lane < (blockDim.x >> 5)) ? warp_keys[lane] : 0ull;
    if (warp == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            const unsigned long long other = __shfl_down_sync(kMask, key, offset);
            if (other > key) { key = other; }
        }
    }
    return key;
}

__host__ __device__ __forceinline__ unsigned long long sampling_splitmix64(unsigned long long x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// Uniform float in [0,1) from (seed, position, purpose, sub). Pure function of its inputs so it
// is safe under CUDA-graph replay (no mutable RNG state).
__host__ __device__ __forceinline__ float sampling_uniform(unsigned long long seed, int position,
                                                           int purpose, unsigned int sub) {
    unsigned long long key = seed;
    key                    = sampling_splitmix64(
        key ^ (static_cast<unsigned long long>(static_cast<unsigned int>(position)) *
               0xD1B54A32D192ED03ull));
    key = sampling_splitmix64(
        key ^ (static_cast<unsigned long long>(static_cast<unsigned int>(purpose)) << 21) ^
        (static_cast<unsigned long long>(sub) * 0x2545F4914F6CDD1Dull));
    const unsigned int bits = static_cast<unsigned int>(key >> 40); // 24 bits
    return static_cast<float>(bits) * (1.0f / 16777216.0f);
}

__device__ __forceinline__ unsigned long long sampling_sort_key(float v, int idx) {
    return score_id_order_key(v, idx);
}

__device__ __forceinline__ float sampling_key_float(unsigned long long key) {
    return score_from_order_key(key);
}

__device__ __forceinline__ int sampling_key_index(unsigned long long key) {
    return id_from_order_key(key);
}

__device__ __forceinline__ int sampling_candidate_cap(const SamplingConfig& cfg,
                                                      std::int32_t vocab) {
    // top_k is clamped to the pipeline cap: top_k <= 0 (no explicit limit) or a top_k larger
    // than the cap both select the full kSamplerCandidateCap set.
    int cap = kSamplerCandidateCap;
    if (cfg.top_k > 0 && cfg.top_k < cap) { cap = cfg.top_k; }
    if (vocab < cap) { cap = vocab; }
    return cap;
}

__device__ __forceinline__ int sampling_partial_offset(const SamplingWorkspace& workspace, int col,
                                                       int partial, int j) {
    return ((col * workspace.partial_stride + partial) * kSamplerCandidateCap) + j;
}

// Sorts each warp's keys, then merges the eight warp-sorted lists into the tile's top `cap`
// keys at partial slot `partial` of column `col`.
__device__ inline void sampling_store_tile_topk(unsigned long long (&keys)[kSamplerItemsPerThread],
                                                int cap, SamplingWorkspace workspace, int col,
                                                int partial, SamplingTileTopKStorage& storage) {
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    SamplingTileWarpSort(storage.warp[warp]).Sort(keys, SamplingKeyGreater{});
#pragma unroll
    for (int item = 0; item < kSamplerItemsPerThread; ++item) {
        const int rank = lane * kSamplerItemsPerThread + item;
        if (rank < kSamplerCandidateCap) {
            storage.candidates[warp * kSamplerCandidateCap + rank] = keys[item];
        }
    }
    __syncthreads();

    if (warp != 0) { return; }
    if (lane < kSamplingTileWarps) {
        constexpr unsigned int kMergeMask = (1u << kSamplingTileWarps) - 1u;
        int position                      = 0;
        for (int rank = 0; rank < cap; ++rank) {
            const unsigned long long key =
                storage.candidates[lane * kSamplerCandidateCap + position];
            const unsigned int high     = static_cast<unsigned int>(key >> 32);
            const unsigned int max_high = __reduce_max_sync(kMergeMask, high);
            const unsigned int low      = high == max_high ? static_cast<unsigned int>(key) : 0u;
            const unsigned int max_low  = __reduce_max_sync(kMergeMask, low);
            const unsigned int winners =
                __ballot_sync(kMergeMask, high == max_high && low == max_low);
            const int source = __ffs(static_cast<int>(winners)) - 1;
            if (lane == 0) {
                const int off = sampling_partial_offset(workspace, col, partial, rank);
                workspace.partial_keys[off] =
                    (static_cast<unsigned long long>(max_high) << 32) | max_low;
            }
            if (lane == source) { ++position; }
        }
    }
}

__device__ __forceinline__ int sampling_dist_offset(int col, int j) {
    return col * kSamplerCandidateCap + j;
}

// Applies presence/frequency penalties to a raw logit. `overlay`/`overlay_len` carry a
// round-local count overlay: tokens already committed earlier in the current speculative round
// but not yet flushed to the global `token_counts`. For speculative verify column `col` the
// overlay is exactly drafts[0..col-1] (statically known, since column `col` is only consumed when
// every earlier draft was accepted), so the penalty at each column sees the same prefix a
// per-token sampler would. Non-speculative callers pass no overlay.
__device__ __forceinline__ float sampling_adjusted_logit(float raw, int v, const SamplingConfig& c,
                                                         const std::int32_t* overlay = nullptr,
                                                         int overlay_len             = 0) {
    float x = raw;
    if (c.presence_penalty == 0.0f && c.frequency_penalty == 0.0f) { return x; }
    int cnt = c.token_counts != nullptr ? c.token_counts[v] : 0;
    for (int j = 0; j < overlay_len; ++j) {
        if (overlay[j] == v) { ++cnt; }
    }
    if (cnt > 0) { x -= c.presence_penalty; }
    if (c.frequency_penalty != 0.0f) { x -= c.frequency_penalty * static_cast<float>(cnt); }
    return x;
}

// Turns the sorted candidates cand_val[0..n) into the normalized truncated support prob[0..*n).
__device__ inline void sampling_normalize_support(const SamplingConfig& cfg, float* cand_val,
                                                  int* cand_idx, float* prob, int* n_support,
                                                  int n) {
    const int tid = threadIdx.x;
    if (tid == 0) {
        const float inv_temp = 1.0f / cfg.temperature;
        const float m        = cand_val[0] * inv_temp;
        float sum            = 0.0f;
        for (int j = 0; j < n; ++j) {
            const float e = __expf(cand_val[j] * inv_temp - m);
            prob[j]       = e;
            sum += e;
        }
        const float e0           = prob[0];
        const float min_p_thresh = (cfg.min_p > 0.0f) ? cfg.min_p * e0 : -1.0f;
        const bool top_p_active  = (cfg.top_p < 1.0f);
        const float top_p_target = cfg.top_p * sum;
        float cum                = 0.0f;
        int support              = 0;
        for (int j = 0; j < n; ++j) {
            if (min_p_thresh >= 0.0f && prob[j] < min_p_thresh) { break; }
            cum += prob[j];
            support = j + 1;
            if (top_p_active && cum >= top_p_target) { break; }
        }
        if (support < 1) { support = 1; }
        float ssum = 0.0f;
        for (int j = 0; j < support; ++j) { ssum += prob[j]; }
        const float inv = 1.0f / ssum;
        for (int j = 0; j < support; ++j) { prob[j] *= inv; }
        *n_support = support;
    }
    __syncthreads();
}

// thread-0 helper: inverse-CDF pick over a normalized `prob[0..n-1]` support, optionally
// excluding `exclude` (a rejected draft token) and renormalizing over the remainder. `u` is a
// uniform in [0,1). Returns the chosen vocab id.
__device__ __forceinline__ int sampling_pick_from_support(const int* cand_idx, const float* prob,
                                                          int n, int exclude, float u) {
    float mass = 0.0f;
    for (int j = 0; j < n; ++j) {
        if (cand_idx[j] == exclude) { continue; }
        mass += prob[j];
    }
    if (mass <= 0.0f) {
        // Degenerate: support is only the excluded token. Return it (caller only reaches this
        // when accept probability was ~1, i.e. it will not happen).
        return cand_idx[0];
    }
    const float goal = u * mass;
    float acc        = 0.0f;
    int picked       = -1;
    for (int j = 0; j < n; ++j) {
        if (cand_idx[j] == exclude) { continue; }
        acc += prob[j];
        picked = cand_idx[j];
        if (goal < acc) { return picked; }
    }
    return picked;
}

// thread-0 helper: inverse-CDF pick from the residual max(0, p - q) of speculative sampling,
// renormalized. p is `prob[0..n)` over cand_idx, q is `q_prob[0..q_n)` over q_idx (ids absent
// from q have q = 0). `u` is a uniform in [0,1). Returns the chosen vocab id.
__device__ __forceinline__ int sampling_pick_residual(const int* cand_idx, const float* prob, int n,
                                                      const std::int32_t* q_idx, const float* q_prob,
                                                      int q_n, float u) {
    float residual[kSamplerCandidateCap];
    float mass = 0.0f;
    for (int j = 0; j < n; ++j) {
        float q = 0.0f;
        for (int t = 0; t < q_n; ++t) {
            if (q_idx[t] == cand_idx[j]) {
                q = q_prob[t];
                break;
            }
        }
        residual[j] = prob[j] > q ? prob[j] - q : 0.0f;
        mass += residual[j];
    }
    if (mass <= 0.0f) {
        // p <= q everywhere, so p == q and a rejection has probability zero up to rounding.
        return cand_idx[0];
    }
    const float goal = u * mass;
    float acc        = 0.0f;
    int picked       = cand_idx[0];
    for (int j = 0; j < n; ++j) {
        if (!(residual[j] > 0.0f)) { continue; }
        acc += residual[j];
        picked = cand_idx[j];
        if (goal < acc) { return picked; }
    }
    return picked;
}

} // namespace fngine::kernels
