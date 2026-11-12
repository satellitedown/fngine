// Modified for fngine from cinference:src/ops/kernel/speculative_round.cuh @ 383e5db. See NOTICE.
// (Also merges cinference:src/ops/launcher/speculative_round.cu.)
//
// Single-request chain verification over FP32 logits [k+1][248320]. Trees, sparse proposals,
// KV compaction and batch > 1 are not part of fngine.

#include "core/device.h"
#include "core/model_dims.h"
#include "kernels/head/head.h"
#include "kernels/head/sampling_device.cuh"

#include <stdexcept>
#include <string>

namespace fngine::kernels {
namespace {

__global__ void speculative_prepare_verify_inputs_kernel(const std::int32_t* anchor,
                                                         const std::int32_t* drafts,
                                                         const std::int32_t* base_position,
                                                         const std::int32_t* current_extent,
                                                         std::int32_t* verify_ids,
                                                         std::int32_t* positions, std::int32_t k) {
    const int T = k + 1;
    int extent  = *current_extent;
    extent      = extent < 0 ? 0 : (extent > k ? k : extent);
    for (int j = static_cast<int>(threadIdx.x); j < T; j += blockDim.x) {
        verify_ids[j] = j == 0 ? *anchor : (j <= extent ? drafts[j - 1] : *anchor);
        if (positions != nullptr) { positions[j] = *base_position + (j <= extent ? j : extent); }
    }
}

__device__ __forceinline__ void
speculative_store_accept_result(const std::int32_t* drafts, std::int32_t k,
                                std::int32_t accepted_count, std::int32_t terminal_token,
                                std::int32_t* length, std::int32_t* anchor,
                                std::int32_t* licensed_tokens, std::int32_t* licensed_count,
                                std::int32_t* accepted, const SamplingConfig& config) {
    for (int i = 0; i <= k; ++i) { licensed_tokens[i] = 0; }
    for (int i = 0; i < accepted_count; ++i) { licensed_tokens[i] = drafts[i]; }
    licensed_tokens[accepted_count] = terminal_token;

    const int produced = accepted_count + 1;
    *licensed_count    = produced;
    *accepted          = accepted_count;
    *anchor            = terminal_token;
    *length += produced;
    if (config.token_counts != nullptr) {
        for (int i = 0; i < produced; ++i) { atomicAdd(&config.token_counts[licensed_tokens[i]], 1); }
    }
}

// Per (partial tile, column): the tile's top candidates of the penalty-adjusted column logits.
// Column col's penalty overlay is drafts[0..col) (the column is consumed only when that prefix
// was accepted); applying it before top-k lets it change the candidate set. Raw-greedy rows
// (temperature <= 0, no penalties) use target_tokens and skip this launch's work.
__launch_bounds__(kSamplerBlock) __global__ void speculative_sampling_partial_topk_kernel(
    const float* logits, const std::int32_t* drafts, const std::int32_t* current_extent,
    const SamplingConfig* config, std::int32_t token_domain, std::int32_t physical_rows,
    std::int32_t k, SamplingWorkspace workspace) {
    const int col     = static_cast<int>(blockIdx.y);
    const int partial = static_cast<int>(blockIdx.x);
    int extent        = *current_extent;
    extent            = extent < 0 ? 0 : (extent > k ? k : extent);
    if (col > extent) { return; }
    const SamplingConfig cfg = *config;
    const bool greedy        = !(cfg.temperature > 0.0f);
    const bool penalties     = cfg.presence_penalty != 0.0f || cfg.frequency_penalty != 0.0f;
    if (greedy && !penalties) { return; }
    if (partial == 0 && threadIdx.x == 0) {
        workspace.group_done[col] = 0;
        if (col == 0) { *workspace.speculative_finalize_count = 0; }
    }

    __shared__ SamplingTileTopKStorage topk_storage;
    __shared__ unsigned long long greedy_warp_keys[kSamplerBlock / 32];

    const int cap           = greedy ? 1 : sampling_candidate_cap(cfg, token_domain);
    const std::int64_t base = static_cast<std::int64_t>(col) * physical_rows;
    const int tile_start    = partial * kSamplerPartialTileItems;

    unsigned long long keys[kSamplerItemsPerThread];
#pragma unroll
    for (int item = 0; item < kSamplerItemsPerThread; ++item) {
        const int v = tile_start + item * blockDim.x + threadIdx.x;
        if (v < token_domain) {
            const float raw      = logits[base + v];
            const float adjusted = penalties ? sampling_adjusted_logit(raw, v, cfg, drafts, col)
                                             : raw;
            keys[item]           = sampling_sort_key(adjusted, v);
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
    sampling_store_tile_topk(keys, cap, workspace, col, partial, topk_storage);
}

// Commits the round's accepted tokens plus one correction/bonus token, then advances the
// length. Greedy (temperature <= 0) keeps the longest draft prefix whose target argmax matches,
// then takes the target argmax at the divergence column. Sampling (temperature > 0) runs
// distribution-correct speculative sampling: drafts[i], drawn from q_i, is accepted with
// probability min(1, p_i(drafts[i]) / q_i(drafts[i])) under the truncated target distribution
// p_i; the first rejection resamples from max(0, p_i - q_i); a bonus is drawn from the last
// column when every draft accepts. q_i is draft_dist[i] for drafts drawn by sample_drafts()
// and one-hot otherwise. Each (group, column) block merges its tiles; the last group of a
// column publishes the column distribution, and the last column to finish performs the
// sequential accept/commit.
__launch_bounds__(kSamplerGroupBlock) __global__ void speculative_sampling_group_finalize_kernel(
    const std::int32_t* target_tokens, const std::int32_t* drafts,
    const std::int32_t* current_extent, std::int32_t* length, std::int32_t* anchor,
    std::int32_t* licensed_tokens, std::int32_t* licensed_count, std::int32_t* accepted,
    const SamplingConfig* config, std::int32_t token_domain, std::int32_t k,
    std::int32_t partial_blocks, std::int32_t group_count, SamplingWorkspace workspace,
    const DraftDistribution* draft_dist, const std::int32_t* draft_kind) {
    const int group = static_cast<int>(blockIdx.x);
    const int col   = static_cast<int>(blockIdx.y);
    const int tid   = threadIdx.x;
    int extent      = *current_extent;
    extent          = extent < 0 ? 0 : (extent > k ? k : extent);
    if (col > extent) { return; }
    const SamplingConfig cfg = *config;
    const bool greedy        = !(cfg.temperature > 0.0f);
    const bool penalties     = cfg.presence_penalty != 0.0f || cfg.frequency_penalty != 0.0f;

    if (greedy && !penalties) {
        if (tid == 0 && col == 0 && group == 0) {
            int a = 0;
            while (a < extent && target_tokens[a] == drafts[a]) { ++a; }
            speculative_store_accept_result(drafts, k, a, target_tokens[a], length, anchor,
                                            licensed_tokens, licensed_count, accepted, cfg);
        }
        return;
    }

    __shared__ SamplingTileTopKStorage topk_storage;
    __shared__ float cand_val[kSamplerCandidateCap];
    __shared__ int cand_idx[kSamplerCandidateCap];
    __shared__ float prob[kSamplerCandidateCap];
    __shared__ int n_support;
    __shared__ int is_last_group;
    unsigned long long keys[kSamplerGroupItemsPerThread];

    const int cap = greedy ? 1 : sampling_candidate_cap(cfg, token_domain);
    // The preceding partial launch initializes all caller-owned counters. CUDA stream ordering
    // makes those writes visible before this launch begins.

    const int group_begin = group * kSamplerPartialsPerGroup;
    int group_partials    = partial_blocks - group_begin;
    if (group_partials < 0) { group_partials = 0; }
    if (group_partials > kSamplerPartialsPerGroup) { group_partials = kSamplerPartialsPerGroup; }
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
    sampling_store_tile_topk(keys, cap, workspace, col, partial_blocks + group, topk_storage);
    __syncthreads();

    if (tid == 0) {
        __threadfence();
        const int done = atomicAdd(&workspace.group_done[col], 1) + 1;
        is_last_group  = (done == group_count) ? 1 : 0;
    }
    __syncthreads();
    if (!is_last_group) { return; }
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
    sampling_store_tile_topk(keys, cap, workspace, col, partial_blocks + group, topk_storage);
    __syncthreads();

    if (tid < cap) {
        const int off = sampling_partial_offset(workspace, col, partial_blocks + group, tid);
        const unsigned long long key = workspace.partial_keys[off];
        cand_val[tid]                = sampling_key_float(key);
        cand_idx[tid]                = sampling_key_index(key);
    }
    __syncthreads();

    if (greedy) {
        if (tid == 0) {
            workspace.dist_idx[sampling_dist_offset(col, 0)] = cand_idx[0];
            workspace.group_done[col]                        = 0;
            __threadfence();
            const int done_cols = atomicAdd(workspace.speculative_finalize_count, 1) + 1;
            if (done_cols == extent + 1) {
                __threadfence();
                int a     = 0;
                int tstar = 0;
                for (int i = 0; i <= extent; ++i) {
                    const int selected = workspace.dist_idx[sampling_dist_offset(i, 0)];
                    if (i < extent && selected == drafts[i]) {
                        a = i + 1;
                        continue;
                    }
                    tstar = selected;
                    break;
                }
                speculative_store_accept_result(drafts, k, a, tstar, length, anchor,
                                                licensed_tokens, licensed_count, accepted, cfg);
                *workspace.speculative_finalize_count = 0;
            }
        }
        return;
    }

    sampling_normalize_support(cfg, cand_val, cand_idx, prob, &n_support, cap);

    if (tid == 0) {
        workspace.dist_support[col] = n_support;
        for (int j = 0; j < n_support; ++j) {
            const int off            = sampling_dist_offset(col, j);
            workspace.dist_idx[off]  = cand_idx[j];
            workspace.dist_prob[off] = prob[j];
        }
        workspace.group_done[col] = 0;
        __threadfence();
        const int done_cols = atomicAdd(workspace.speculative_finalize_count, 1) + 1;
        if (done_cols == extent + 1) {
            __threadfence();
            const int L        = *length;
            const bool coupled = draft_dist != nullptr && draft_kind != nullptr && *draft_kind != 0;
            int a              = 0;
            int tstar          = 0;
            for (int i = 0; i <= extent; ++i) {
                const int n            = workspace.dist_support[i];
                const int* dist_idx    = workspace.dist_idx + sampling_dist_offset(i, 0);
                const float* dist_prob = workspace.dist_prob + sampling_dist_offset(i, 0);
                if (i < extent) {
                    const int d = drafts[i];
                    float pd    = 0.0f;
                    for (int j = 0; j < n; ++j) {
                        if (dist_idx[j] == d) {
                            pd = dist_prob[j];
                            break;
                        }
                    }
                    // q(d) of the distribution the draft was drawn from; one-hot (q(d) = 1) for a
                    // fixed draft, or when the stored distribution does not hold the draft.
                    const DraftDistribution* q = nullptr;
                    float qd                   = 1.0f;
                    if (coupled) {
                        const DraftDistribution& dd = draft_dist[i];
                        const int q_n = dd.n < kDraftSupport ? dd.n : kDraftSupport;
                        for (int j = 0; j < q_n; ++j) {
                            if (dd.ids[j] == d && dd.probs[j] > 0.0f) {
                                q  = &dd;
                                qd = dd.probs[j];
                                break;
                            }
                        }
                    }
                    const float u =
                        sampling_uniform(cfg.seed, L + i + 1, kSamplePurposeSpeculativeAccept, 0u);
                    if (u * qd < pd) {
                        a = i + 1;
                        continue;
                    }
                    const float ur = sampling_uniform(cfg.seed, L + i + 1,
                                                      kSamplePurposeSpeculativeCorrection, 0u);
                    tstar = q != nullptr
                                ? sampling_pick_residual(dist_idx, dist_prob, n, q->ids, q->probs,
                                                         q->n < kDraftSupport ? q->n : kDraftSupport,
                                                         ur)
                                : sampling_pick_from_support(dist_idx, dist_prob, n, d, ur);
                    break;
                }
                const float u =
                    sampling_uniform(cfg.seed, L + extent + 1, kSamplePurposeSpeculativeBonus, 0u);
                tstar = sampling_pick_from_support(dist_idx, dist_prob, n, -1, u);
            }
            speculative_store_accept_result(drafts, k, a, tstar, length, anchor, licensed_tokens,
                                            licensed_count, accepted, cfg);
            *workspace.speculative_finalize_count = 0;
        }
    }
}

void check_drafts(std::int32_t token_domain, std::int32_t k, const char* what) {
    if (k < 1 || token_domain > dims::kVocab || !sampler_route_ok(token_domain, k + 1)) {
        throw std::invalid_argument(std::string(what) +
                                    ": token_domain must be in (256, 248320] and k in [1, 15]");
    }
}

} // namespace

void speculative_prepare_verify_inputs(const std::int32_t* anchor, const std::int32_t* drafts,
                                       const std::int32_t* base_position,
                                       const std::int32_t* current_extent, std::int32_t k,
                                       std::int32_t* verify_ids, std::int32_t* positions,
                                       cudaStream_t stream) {
    if (k < 1 || k + 1 > 32) {
        throw std::invalid_argument("speculative_prepare_verify_inputs: k must be in [1, 31]");
    }
    speculative_prepare_verify_inputs_kernel<<<1, 32, 0, stream>>>(
        anchor, drafts, base_position, current_extent, verify_ids, positions, k);
    CUDA_CHECK(cudaGetLastError());
}

std::size_t speculative_accept_workspace_bytes(std::int32_t token_domain,
                                               std::int32_t max_drafts) {
    check_drafts(token_domain, max_drafts, "speculative_accept_workspace_bytes");
    return make_sampling_workspace_layout(token_domain, max_drafts + 1).bytes;
}

void speculative_accept(const std::int32_t* target_tokens, const float* logits,
                        const std::int32_t* drafts, const std::int32_t* current_extent,
                        std::int32_t k, std::int32_t token_domain, const SamplingConfig* config,
                        std::int32_t* length, std::int32_t* anchor,
                        std::int32_t* licensed_tokens, std::int32_t* licensed_count,
                        std::int32_t* accepted, void* workspace, std::size_t workspace_bytes,
                        cudaStream_t stream, const DraftDistribution* draft_dist,
                        const std::int32_t* draft_kind) {
    check_drafts(token_domain, k, "speculative_accept");
    const std::int32_t cols              = k + 1;
    const SamplingWorkspaceLayout layout = make_sampling_workspace_layout(token_domain, cols);
    if (workspace_bytes < layout.bytes) {
        throw std::invalid_argument("speculative_accept: workspace too small");
    }
    const std::int32_t partial_blocks = sampler_partial_blocks(token_domain);
    const std::int32_t groups         = sampler_group_count(partial_blocks);
    const SamplingWorkspace scratch   = layout.bind(workspace);
    const dim3 partial_grid(static_cast<unsigned int>(partial_blocks),
                            static_cast<unsigned int>(cols));
    speculative_sampling_partial_topk_kernel<<<partial_grid, kSamplerBlock, 0, stream>>>(
        logits, drafts, current_extent, config, token_domain, dims::kVocab, k, scratch);
    CUDA_CHECK(cudaGetLastError());
    const dim3 group_grid(static_cast<unsigned int>(groups), static_cast<unsigned int>(cols));
    speculative_sampling_group_finalize_kernel<<<group_grid, kSamplerGroupBlock, 0, stream>>>(
        target_tokens, drafts, current_extent, length, anchor, licensed_tokens, licensed_count,
        accepted, config, token_domain, k, partial_blocks, groups, scratch, draft_dist,
        draft_kind);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace fngine::kernels
