// Modified for fngine from cinference:src/ops/kernel/sampling.cuh @ 383e5db. See NOTICE.
// (Also merges cinference:src/ops/launcher/sampling.cu.)
//
// Two-launch route: 256-thread/2-item partial tiles produce each tile's top candidates, then
// group blocks merge 25 tiles each and the last group to finish per column draws the token.
// fngine logits are FP32 [T][248320]; all columns share one SamplingConfig.

#include "core/device.h"
#include "core/model_dims.h"
#include "kernels/head/head.h"
#include "kernels/head/sampling_device.cuh"

#include <stdexcept>
#include <string>

namespace fngine::kernels {
namespace {

// What sample_drafts() adds to sample(): a row -> token map, a penalty overlay, a position
// offset and a draw counter for the RNG key, the distribution output, and no token_counts
// increment.
struct DraftArgs {
    const std::int32_t* id_map   = nullptr;
    const std::int32_t* overlay  = nullptr;
    std::int32_t overlay_len     = 0;
    std::int32_t position_offset = 0;
    const std::uint32_t* draw    = nullptr;
    DraftDistribution* dist      = nullptr;
    std::int32_t count_tokens    = 1;
};

__launch_bounds__(kSamplerBlock) __global__
    void sampling_partial_topk_kernel(const float* logits, const SamplingConfig* cfg_ptr,
                                      std::int32_t token_domain, std::int32_t physical_rows,
                                      SamplingWorkspace workspace, DraftArgs draft) {
    const int col            = static_cast<int>(blockIdx.y);
    const int partial        = static_cast<int>(blockIdx.x);
    const SamplingConfig cfg = *cfg_ptr;
    if (partial == 0 && threadIdx.x == 0) { workspace.group_done[col] = 0; }

    __shared__ typename SamplingPartialSort::TempStorage sort_storage;
    __shared__ unsigned long long greedy_warp_keys[kSamplerBlock / 32];
    unsigned long long keys[kSamplerItemsPerThread];

    const bool greedy       = !(cfg.temperature > 0.0f);
    const bool penalties    = cfg.presence_penalty != 0.0f || cfg.frequency_penalty != 0.0f;
    const int cap           = greedy ? 1 : sampling_candidate_cap(cfg, token_domain);
    const std::int64_t base = static_cast<std::int64_t>(col) * physical_rows;
    const int tile_start    = partial * kSamplerPartialTileItems;
#pragma unroll
    for (int item = 0; item < kSamplerItemsPerThread; ++item) {
        const int v = tile_start + item * blockDim.x + threadIdx.x;
        if (v < token_domain) {
            const float raw = logits[base + v];
            const float x   = penalties ? sampling_adjusted_logit(raw, draft.id_map ? draft.id_map[v] : v,
                                                                  cfg, draft.overlay, draft.overlay_len)
                                        : raw;
            keys[item]      = sampling_sort_key(x, v);
        } else {
            keys[item] = 0ull;
        }
    }
    if (greedy) {
        unsigned long long best = keys[0];
#pragma unroll
        for (int item = 1; item < kSamplerItemsPerThread; ++item) {
            if (keys[item] > best) { best = keys[item]; }
        }
        best = sampling_block_max_key(best, greedy_warp_keys);
        if (threadIdx.x == 0) {
            const int off               = sampling_partial_offset(workspace, col, partial, 0);
            workspace.partial_keys[off] = best;
        }
        return;
    }
    SamplingPartialSort(sort_storage).Sort(keys, SamplingKeyGreater{});

#pragma unroll
    for (int item = 0; item < kSamplerItemsPerThread; ++item) {
        const int rank = threadIdx.x * kSamplerItemsPerThread + item;
        if (rank < cap) {
            const int off               = sampling_partial_offset(workspace, col, partial, rank);
            workspace.partial_keys[off] = keys[item];
        }
    }
}

__launch_bounds__(kSamplerGroupBlock) __global__ void sampling_group_finalize_sample_kernel(
    std::int32_t* out, const SamplingConfig* cfg_ptr, const std::int32_t* logical_positions,
    std::int32_t purpose, std::int32_t token_domain, std::int32_t partial_blocks,
    std::int32_t group_count, SamplingWorkspace workspace, DraftArgs draft) {
    const int group          = static_cast<int>(blockIdx.x);
    const int col            = static_cast<int>(blockIdx.y);
    const int tid            = threadIdx.x;
    const SamplingConfig cfg = *cfg_ptr;
    __shared__ typename SamplingGroupSort::TempStorage sort_storage;
    __shared__ float cand_val[kSamplerCandidateCap];
    __shared__ int cand_idx[kSamplerCandidateCap];
    __shared__ float prob[kSamplerCandidateCap];
    __shared__ int n_support;
    __shared__ int is_last;
    __shared__ unsigned long long greedy_warp_keys[kSamplerGroupBlock / 32];
    unsigned long long keys[kSamplerGroupItemsPerThread];

    const bool greedy = !(cfg.temperature > 0.0f);
    const int cap     = greedy ? 1 : sampling_candidate_cap(cfg, token_domain);
    // The preceding partial launch initializes group_done[col], so caller-owned workspace does
    // not rely on prior contents or a separate memset launch.

    const int group_begin = group * kSamplerPartialsPerGroup;
    int group_partials    = partial_blocks - group_begin;
    if (group_partials < 0) { group_partials = 0; }
    if (group_partials > kSamplerPartialsPerGroup) { group_partials = kSamplerPartialsPerGroup; }

    if (greedy) {
        unsigned long long best = 0ull;
        for (int p = tid; p < group_partials; p += blockDim.x) {
            const int off = sampling_partial_offset(workspace, col, group_begin + p, 0);
            if (workspace.partial_keys[off] > best) { best = workspace.partial_keys[off]; }
        }
        best = sampling_block_max_key(best, greedy_warp_keys);
        if (tid == 0) {
            const int out_off = sampling_partial_offset(workspace, col, partial_blocks + group, 0);
            workspace.partial_keys[out_off] = best;
            __threadfence();
            const int done = atomicAdd(&workspace.group_done[col], 1) + 1;
            is_last        = (done == group_count) ? 1 : 0;
        }
        __syncthreads();
        if (!is_last) { return; }
        __threadfence();

        best = 0ull;
        for (int p = tid; p < group_count; p += blockDim.x) {
            const int off = sampling_partial_offset(workspace, col, partial_blocks + p, 0);
            if (workspace.partial_keys[off] > best) { best = workspace.partial_keys[off]; }
        }
        __syncthreads();
        best = sampling_block_max_key(best, greedy_warp_keys);
        if (tid == 0) {
            const int row    = sampling_key_index(best);
            const int picked = draft.id_map ? draft.id_map[row] : row;
            out[col]         = picked;
            if (draft.count_tokens && cfg.token_counts != nullptr) {
                atomicAdd(&cfg.token_counts[picked], 1);
            }
            if (draft.dist != nullptr) { draft.dist[col].n = 0; }
            workspace.group_done[col] = 0;
        }
        return;
    }

    const int group_n = group_partials * cap;
#pragma unroll
    for (int item = 0; item < kSamplerGroupItemsPerThread; ++item) {
        const int p = item * blockDim.x + tid;
        if (p < group_n) {
            const int partial = group_begin + p / cap;
            const int j       = p - (p / cap) * cap;
            const int off     = sampling_partial_offset(workspace, col, partial, j);
            keys[item]        = workspace.partial_keys[off];
        } else {
            keys[item] = 0ull;
        }
    }
    SamplingGroupSort(sort_storage).Sort(keys, SamplingKeyGreater{});

#pragma unroll
    for (int item = 0; item < kSamplerGroupItemsPerThread; ++item) {
        const int rank = tid * kSamplerGroupItemsPerThread + item;
        if (rank < cap) {
            const int out_off =
                sampling_partial_offset(workspace, col, partial_blocks + group, rank);
            workspace.partial_keys[out_off] = keys[item];
        }
    }
    __syncthreads();

    if (tid == 0) {
        __threadfence();
        const int done = atomicAdd(&workspace.group_done[col], 1) + 1;
        is_last        = (done == group_count) ? 1 : 0;
    }
    __syncthreads();
    if (!is_last) { return; }
    __threadfence();

    const int final_n = group_count * cap;
#pragma unroll
    for (int item = 0; item < kSamplerGroupItemsPerThread; ++item) {
        const int p = item * blockDim.x + tid;
        if (p < final_n) {
            const int partial = partial_blocks + p / cap;
            const int j       = p - (p / cap) * cap;
            const int off     = sampling_partial_offset(workspace, col, partial, j);
            keys[item]        = workspace.partial_keys[off];
        } else {
            keys[item] = 0ull;
        }
    }
    __syncthreads();
    SamplingGroupSort(sort_storage).Sort(keys, SamplingKeyGreater{});

#pragma unroll
    for (int item = 0; item < kSamplerGroupItemsPerThread; ++item) {
        const int rank = tid * kSamplerGroupItemsPerThread + item;
        if (rank < cap) {
            cand_val[rank] = sampling_key_float(keys[item]);
            cand_idx[rank] = sampling_key_index(keys[item]);
        }
    }
    __syncthreads();

    sampling_normalize_support(cfg, cand_val, cand_idx, prob, &n_support, cap);
    if (tid == 0) {
        const int support = n_support;
        const float u     = sampling_uniform(cfg.seed, logical_positions[col] + draft.position_offset,
                                             purpose, draft.draw != nullptr ? *draft.draw : 0u);
        float acc         = 0.0f;
        int row           = cand_idx[support - 1];
        for (int j = 0; j < support; ++j) {
            acc += prob[j];
            if (u < acc) {
                row = cand_idx[j];
                break;
            }
        }
        const int picked = draft.id_map ? draft.id_map[row] : row;
        out[col]         = picked;
        if (draft.count_tokens && cfg.token_counts != nullptr) {
            atomicAdd(&cfg.token_counts[picked], 1);
        }
        if (draft.dist != nullptr) {
            DraftDistribution& q = draft.dist[col];
            q.n                  = support;
            for (int j = 0; j < support; ++j) {
                q.ids[j]   = draft.id_map ? draft.id_map[cand_idx[j]] : cand_idx[j];
                q.probs[j] = prob[j];
            }
        }
        workspace.group_done[col] = 0;
    }
}

__global__ void increment_token_counts_kernel(const std::int32_t* token_ids, std::int32_t count,
                                              std::int32_t* token_counts) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) { atomicAdd(&token_counts[token_ids[index]], 1); }
}

void check_route(std::int32_t token_domain, std::int32_t columns, const char* what) {
    if (token_domain > dims::kVocab || !sampler_route_ok(token_domain, columns)) {
        throw std::invalid_argument(std::string(what) +
                                    ": token_domain must be in (256, 248320] and columns in "
                                    "[1, 16]");
    }
}

} // namespace

std::size_t sampling_workspace_bytes(std::int32_t token_domain, std::int32_t max_columns) {
    check_route(token_domain, max_columns, "sampling_workspace_bytes");
    return make_sampling_workspace_layout(token_domain, max_columns).bytes;
}

namespace {

void launch_sample(const float* logits, std::int32_t T, std::int32_t token_domain,
                   std::int32_t row_stride, const SamplingConfig* config,
                   const std::int32_t* positions, std::int32_t purpose, std::int32_t* out,
                   const DraftArgs& draft, void* workspace, std::size_t workspace_bytes,
                   cudaStream_t stream, const char* what) {
    check_route(token_domain, T, what);
    const SamplingWorkspaceLayout layout = make_sampling_workspace_layout(token_domain, T);
    if (workspace_bytes < layout.bytes) {
        throw std::invalid_argument(std::string(what) + ": workspace too small");
    }
    const std::int32_t partial_blocks = sampler_partial_blocks(token_domain);
    const std::int32_t groups         = sampler_group_count(partial_blocks);
    const SamplingWorkspace scratch   = layout.bind(workspace);
    const dim3 partial_grid(static_cast<unsigned int>(partial_blocks),
                            static_cast<unsigned int>(T));
    sampling_partial_topk_kernel<<<partial_grid, kSamplerBlock, 0, stream>>>(
        logits, config, token_domain, row_stride, scratch, draft);
    CUDA_CHECK(cudaGetLastError());
    const dim3 group_grid(static_cast<unsigned int>(groups), static_cast<unsigned int>(T));
    sampling_group_finalize_sample_kernel<<<group_grid, kSamplerGroupBlock, 0, stream>>>(
        out, config, positions, purpose, token_domain, partial_blocks, groups, scratch, draft);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void sample(const float* logits, std::int32_t T, std::int32_t token_domain,
            const SamplingConfig* config, const std::int32_t* positions, std::int32_t purpose,
            std::int32_t* out, void* workspace, std::size_t workspace_bytes,
            cudaStream_t stream) {
    if (T == 0) { return; }
    launch_sample(logits, T, token_domain, dims::kVocab, config, positions, purpose, out, DraftArgs{},
                  workspace, workspace_bytes, stream, "sample");
}

void sample_drafts(const float* logits, std::int32_t rows, std::int32_t token_domain,
                   std::int32_t row_stride, const std::int32_t* id_map, const SamplingConfig* config,
                   const std::int32_t* positions, std::int32_t position_offset,
                   const std::uint32_t* draw, const std::int32_t* overlay,
                   std::int32_t overlay_len, std::int32_t* out, DraftDistribution* dist,
                   void* workspace, std::size_t workspace_bytes, cudaStream_t stream) {
    if (rows == 0) { return; }
    if (row_stride < token_domain || overlay_len < 0 || (overlay_len > 0 && overlay == nullptr)) {
        throw std::invalid_argument("sample_drafts: bad row stride or overlay");
    }
    DraftArgs draft;
    draft.id_map          = id_map;
    draft.overlay         = overlay;
    draft.overlay_len     = overlay_len;
    draft.position_offset = position_offset;
    draft.draw            = draw;
    draft.dist            = dist;
    draft.count_tokens    = 0;
    launch_sample(logits, rows, token_domain, row_stride, config, positions, kSamplePurposeDraft, out,
                  draft, workspace, workspace_bytes, stream, "sample_drafts");
}

void increment_token_counts(const std::int32_t* token_ids, std::int32_t count,
                            std::int32_t* token_counts, cudaStream_t stream) {
    if (count <= 0) { return; }
    constexpr int kBlock = 256;
    increment_token_counts_kernel<<<div_up(count, kBlock), kBlock, 0, stream>>>(
        token_ids, count, token_counts);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace fngine::kernels
