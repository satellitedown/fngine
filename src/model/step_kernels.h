#pragma once

// Small device-side glue for decode/verify steps: round transition after acceptance, selection of
// the accepted column's MTP state, and single-token step bookkeeping.
#include "kernels/gdn/gdn.h"
#include "kernels/head/head.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace fngine::model {

inline constexpr int32_t kStepMaxTokens = 16;
inline constexpr int32_t kMtpDrafts = 3;

// Device-resident step inputs, refreshed from device-mapped host memory at the start of every graph.
struct alignas(16) StepInDev {
  int32_t tokens[kStepMaxTokens];     // [anchor, d1..dk]
  int32_t positions[kStepMaxTokens];  // base + j
  int32_t drafts[kStepMaxTokens];     // d1..dk (tokens[1..k])
  int32_t k;                          // drafts in this step (T = k + 1)
  int32_t length;                     // committed length (RNG position base)
  uint32_t seq;                       // exchange sequence number
  int32_t draft_kind;                 // 1: drafts are the previous step's MTP drafts d1..dk
};

// Device-resident step outputs, stored to device-mapped host memory at the end of every graph.
struct alignas(16) StepOutDev {
  int32_t licensed[kStepMaxTokens];  // accepted drafts then the correction/bonus token
  int32_t licensed_count;            // accepted + 1
  int32_t accepted;
  int32_t anchor;                    // last licensed token (next step's anchor)
  int32_t length;
  int32_t mtp_drafts[4];             // next MTP drafts d1..d3
  int32_t target[kStepMaxTokens];    // per-column argmax of the verify logits (diagnostics)
  float target_gap[kStepMaxTokens];  // optional diagnostics: top-1 minus top-2 before MTP
};

// Device scratch of the step (MTP bridge/AR bookkeeping).
struct alignas(16) StepScratch {
  int32_t alignment_ids[kStepMaxTokens];  // MTP bridge input tokens
  int32_t mtp_token[kStepMaxTokens];      // per-column MTP draft
  int32_t ar_positions[4];
  int32_t ar_token[4];
};

// Device-resident distributions the MTP drafts were drawn from. next[i] belongs to
// out.mtp_drafts[i] and is read by the following step's acceptance; bridge[c] is the bridge
// column c's (the column of the accepted count becomes next[0]).
struct alignas(16) StepDraftDists {
  kernels::DraftDistribution bridge[kStepMaxTokens];
  kernels::DraftDistribution next[4];
};

// dst <- src, `bytes` a multiple of 16: step inputs/outputs between device memory and
// device-mapped host memory. A kernel moves them, not a memcpy: copy engines serve the expert-swap
// transfers, behind which a small upload waits 30-50 µs.
void step_copy(void* dst, const void* src, std::size_t bytes, cudaStream_t s);

// T = 1 bookkeeping after sample(): licensed = {sampled}, count 1, accepted 0, anchor = sampled,
// length += 1.
void step_finish_single(const StepInDev* in, StepOutDev* out, cudaStream_t s);

// Optional verification diagnostics. No launch on the production path.
void step_logit_gaps(const float* logits, int32_t rows, StepOutDev* out, cudaStream_t s);

// After speculative_accept (verify, T = k + 1): GDN fold commit count = accepted + 1.
void step_set_gdn_commit(const StepOutDev* out, kernels::GdnStepControl* control, cudaStream_t s);

// MTP round transition: alignment_ids[j] = j < A ? in.tokens[j+1] : out.anchor (j < T);
// ar_positions[i] = in.positions[0] + A + 1 + i.
void step_mtp_transition(const StepInDev* in, const StepOutDev* out, int32_t T,
                         StepScratch* scratch, cudaStream_t s);

// dst[0] = src[A] for an FP32 [T][10240] stream tensor (A = out->accepted),
// out->mtp_drafts[0] = scratch->ar_token[0] = scratch->mtp_token[A] and
// dists->next[0] = dists->bridge[A].
void step_select_accepted(const float* src, StepOutDev* out, float* dst, StepScratch* scratch,
                          StepDraftDists* dists, cudaStream_t s);

// After MTP AR step i: out->mtp_drafts[i + 1] = scratch->ar_token[i + 1] = scratch->mtp_token[0]
// and dst <- src row 0 (the next AR step's hidden).
void step_ar_advance(const float* src, float* dst, StepOutDev* out, StepScratch* scratch, int32_t i,
                     cudaStream_t s);

// Draft head construction: dst row i = src row vocab[i] (FP8 codes of k bytes + BF16 scale).
void gather_fp8_rows(const uint8_t* src_codes, const __nv_bfloat16* src_scales, const int32_t* vocab,
                     int32_t rows, int32_t k, uint8_t* dst_codes, __nv_bfloat16* dst_scales,
                     cudaStream_t s);

}  // namespace fngine::model
