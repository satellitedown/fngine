#pragma once

// Model head and token-selection kernels: embedding gather, FP32-logit argmax, sampling,
// speculative verification/acceptance, and the MTP round transition.
//
// Logits are FP32 [T][dims::kVocab] (row stride 248320). Only vocabulary ids in
// [0, token_domain) participate; token_domain must lie in (256, 248320]. T (the number of logit
// columns) is at most 16. All launchers are CUDA-graph capturable: dynamic per-step values
// (positions, lengths, extents, counts) are read from device memory.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "core/model_dims.h"

#include <cstddef>
#include <cstdint>

namespace fngine::kernels {

// ---------------------------------------------------------------- embedding

// X[t][s][d] = float(table[tokens[t]][d]) for all 4 streams s; emb[t][d] = table[tokens[t]][d]
// when emb is non-null. `table` is BF16 [248320][2560] and may be a host-mapped pointer
// (zero-copy gather); it must be 16-byte aligned. tokens is device I32 [T] with ids in
// [0, 248320). X is FP32 [T][4][2560].
void embed_gather(const __nv_bfloat16* table, const std::int32_t* tokens, std::int32_t T,
                  float* X, __nv_bfloat16* emb, cudaStream_t stream);

// ---------------------------------------------------------------- argmax

// out[t] = min argmax_{v < token_domain} logits[t][v] over FP32 [T][248320] logits (ties go to
// the lower id).
void argmax(const float* logits, std::int32_t T, std::int32_t token_domain, std::int32_t* out,
            cudaStream_t stream);

// ---------------------------------------------------------------- sampling

// Counter-based RNG subkey. Distinct purposes keep draws at the same logical position separate.
enum SamplePurpose : std::int32_t {
    kSamplePurposePrefill               = 0,
    kSamplePurposeDecode                = 1,
    kSamplePurposeSpeculativeAccept     = 2,
    kSamplePurposeSpeculativeCorrection = 3,
    kSamplePurposeSpeculativeBonus      = 4,
    kSamplePurposeDraft                 = 5,
};

// Device-resident sampling parameters. token_counts is an optional device I32 [vocab]
// occurrence-count array of the committed generated tokens, used by both penalties; the
// sampling and acceptance kernels increment it for every token they emit.
struct SamplingConfig {
    float temperature          = 0.0f; // <= 0 => greedy argmax
    std::int32_t top_k         = 20;   // contract [1,20]; <=0 or >=20 keeps 20 candidates
    float top_p                = 1.0f; // >= 1 => disabled
    float min_p                = 0.0f; // <= 0 => disabled
    float presence_penalty     = 0.0f;
    float frequency_penalty    = 0.0f;
    unsigned long long seed    = 0;
    std::int32_t* token_counts = nullptr;
};

// Workspace for sample() over at most `max_columns` (<= 16) columns.
[[nodiscard]] std::size_t sampling_workspace_bytes(std::int32_t token_domain,
                                                   std::int32_t max_columns);

/**
 * One token per logit column t < T, all columns sharing the device-resident `config`.
 * adjusted_v = logits[t][v] - presence_penalty*(c_v>0) - frequency_penalty*c_v with
 * c_v = token_counts[v] (0 when null). temperature <= 0: out[t] = min argmax adjusted_v.
 * Otherwise candidates are sorted by adjusted_v descending (lower id breaks ties); top_k keeps
 * min(top_k,20) of them; weights exp((adjusted_v - max)/temperature); min_p drops the suffix
 * below min_p*max_weight; top_p keeps the shortest prefix whose cumulative weight reaches
 * top_p times the total candidate weight; at least one candidate remains; the support is
 * renormalized and one id is drawn with the counter RNG keyed by
 * (seed, positions[t], purpose). Every selected token increments token_counts when non-null.
 * positions is device I32 [T]; out is device I32 [T].
 */
void sample(const float* logits, std::int32_t T, std::int32_t token_domain,
            const SamplingConfig* config, const std::int32_t* positions, std::int32_t purpose,
            std::int32_t* out, void* workspace, std::size_t workspace_bytes, cudaStream_t stream);

// The truncated distribution a draft token was drawn from (token ids, descending probability).
// n == 0: the draft is a fixed choice (an argmax or a looked-up token), a one-hot proposal.
inline constexpr std::int32_t kDraftSupport = 20;
struct alignas(16) DraftDistribution {
    std::int32_t n = 0;
    std::int32_t ids[kDraftSupport] = {};
    float probs[kDraftSupport]      = {};
};

/**
 * Draft tokens from draft-head logits FP32 [rows][row_stride] over `token_domain` draft rows;
 * `id_map` (device I32 [token_domain], or null for the identity) gives the token of each row,
 * which is what token_counts and the penalty overlay refer to. Row r's overlay is
 * overlay[0..overlay_len): tokens that precede the draft and are not yet in token_counts.
 *   temperature <= 0: out[r] = token of the penalty-adjusted row argmax; dist[r].n = 0.
 *   Otherwise: row r's truncated distribution q_r (sample() rules) is stored in dist[r] and
 *     out[r] is drawn from it with the counter RNG keyed by
 *     (seed, positions[r] + position_offset, kSamplePurposeDraft, *draw). `draw` (device U32,
 *     or null for 0) separates repeated draws for one position: a draft redrawn after a
 *     rejection is then independent of the discarded one.
 * token_counts are read, never incremented. positions is device I32 [rows]; out is device I32
 * [rows]; dist is device [rows]. The workspace is sample()'s.
 */
void sample_drafts(const float* logits, std::int32_t rows, std::int32_t token_domain,
                   std::int32_t row_stride, const std::int32_t* id_map, const SamplingConfig* config,
                   const std::int32_t* positions, std::int32_t position_offset,
                   const std::uint32_t* draw, const std::int32_t* overlay,
                   std::int32_t overlay_len, std::int32_t* out, DraftDistribution* dist,
                   void* workspace, std::size_t workspace_bytes, cudaStream_t stream);

// token_counts[token_ids[i]] += 1 for i < count (device arrays).
void increment_token_counts(const std::int32_t* token_ids, std::int32_t count,
                            std::int32_t* token_counts, cudaStream_t stream);

// ---------------------------------------------------------------- speculative round

/**
 * Verification inputs for k drafts (current extent P = clamp(*current_extent, 0, k)):
 *   verify_ids[j] = *anchor for j == 0, drafts[j-1] for 0 < j <= P, *anchor otherwise;
 *   positions[j]  = *base_position + min(j, P)   (skipped when positions is null).
 * verify_ids/positions are device I32 [k+1].
 */
void speculative_prepare_verify_inputs(const std::int32_t* anchor, const std::int32_t* drafts,
                                       const std::int32_t* base_position,
                                       const std::int32_t* current_extent, std::int32_t k,
                                       std::int32_t* verify_ids, std::int32_t* positions,
                                       cudaStream_t stream);

// Workspace for speculative_accept() with up to `max_drafts` (<= 15) drafts.
[[nodiscard]] std::size_t speculative_accept_workspace_bytes(std::int32_t token_domain,
                                                             std::int32_t max_drafts);

/**
 * Accepts a prefix of the k drafts against the verification logits FP32 [k+1][248320] and
 * emits one correction/bonus token. P = clamp(*current_extent, 0, k) drafts are live.
 *   Greedy (temperature <= 0): A = longest prefix with drafts[i] == argmax of column i
 *     (penalty-adjusted; without penalties target_tokens[i], the raw argmax, is used), terminal
 *     = argmax of column A.
 *   Sampling: column i's truncated distribution p_i (sample() rules, penalty overlay =
 *     drafts[0..i)) accepts drafts[i] when u q_i(drafts[i]) < p_i(drafts[i]) with
 *     u = rng(seed, L+i+1, SpeculativeAccept), where q_i is the distribution the draft was drawn
 *     from: draft_dist[i] when `draft_dist` is set, *draft_kind != 0 and its n > 0 and it
 *     holds the draft; otherwise one-hot (q_i(drafts[i]) = 1). On the first rejection the
 *     terminal is drawn from max(0, p_i - q_i) normalized (one-hot: p_i with drafts[i] removed;
 *     rng purpose SpeculativeCorrection); when all P drafts are accepted the terminal is drawn
 *     from p_P (purpose SpeculativeBonus, position L+P+1). Every emitted token is then
 *     distributed as its p_i: speculative sampling (Leviathan et al.), with acceptance
 *     probability sum_x min(p_i(x), q_i(x)). The one-hot rule is valid for any draft chosen
 *     independently of this call's random numbers.
 * L = *length before the call. Effects: licensed_tokens[0..A) = accepted drafts,
 * licensed_tokens[A] = terminal, rest 0 (device I32 [k+1]); *licensed_count = A+1;
 * *accepted = A; *anchor = terminal; *length += A+1; token_counts (if any) += emitted tokens.
 * target_tokens is device I32 [k+1] (argmax() of the logits; read only on the raw-greedy path).
 * draft_dist (device [k]) and draft_kind (device I32) may both be null.
 */
void speculative_accept(const std::int32_t* target_tokens, const float* logits,
                        const std::int32_t* drafts, const std::int32_t* current_extent,
                        std::int32_t k, std::int32_t token_domain, const SamplingConfig* config,
                        std::int32_t* length, std::int32_t* anchor,
                        std::int32_t* licensed_tokens, std::int32_t* licensed_count,
                        std::int32_t* accepted, void* workspace, std::size_t workspace_bytes,
                        cudaStream_t stream, const DraftDistribution* draft_dist = nullptr,
                        const std::int32_t* draft_kind = nullptr);

/**
 * MTP round transition after acceptance (k = verify width - 1, A = clamp(*accepted, 0, k)):
 *   alignment_ids[j] = verify_ids[j+1] for j < A, else *next_anchor   (device I32 [k+1]; the
 *     MTP bridge input tokens);
 *   *next_extent = clamp(min(*remaining_budget - *licensed_count - 1,
 *                            max_context - *updated_frontier - 1), 0, k);
 *   for s < max(k-1, 1): ar_positions[s] = *updated_frontier + s,
 *     ar_valid[s] = (s + 1 < *next_extent)   (the extra autoregressive draft steps).
 */
void mtp_prepare_next_round(const std::int32_t* verify_ids, std::int32_t k,
                            const std::int32_t* next_anchor, const std::int32_t* accepted,
                            const std::int32_t* updated_frontier,
                            const std::int32_t* remaining_budget,
                            const std::int32_t* licensed_count, std::int32_t max_context,
                            std::int32_t* alignment_ids, std::int32_t* next_extent,
                            std::int32_t* ar_positions, std::int32_t* ar_valid,
                            cudaStream_t stream);

} // namespace fngine::kernels
