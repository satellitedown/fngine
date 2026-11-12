#include "model/step_kernels.h"

#include "core/device.h"
#include "core/model_dims.h"

#include <stdexcept>

namespace fngine::model {
namespace {

__global__ void copy16_kernel(int4* dst, const int4* src, int n) {
  for (int i = int(threadIdx.x); i < n; i += int(blockDim.x)) dst[i] = src[i];
}

__global__ void finish_single_kernel(const StepInDev* in, StepOutDev* out) {
  if (threadIdx.x != 0) return;
  out->licensed_count = 1;
  out->accepted = 0;
  out->anchor = out->licensed[0];
  out->length = in->length + 1;
}

__global__ void set_commit_kernel(const StepOutDev* out, kernels::GdnStepControl* control) {
  if (threadIdx.x == 0) control->commit = out->accepted + 1;
}

__global__ void logit_gaps_kernel(const float* logits, StepOutDev* out) {
  __shared__ float first[256], second[256];
  const int lane = threadIdx.x;
  const float* row = logits + std::size_t(blockIdx.x) * dims::kVocab;
  float a = -INFINITY, b = -INFINITY;
  for (int i = lane; i < dims::kVocab; i += blockDim.x) {
    const float v = row[i];
    if (v >= a) { b = a; a = v; }
    else b = fmaxf(b, v);
  }
  first[lane] = a;
  second[lane] = b;
  __syncthreads();
  for (int stride = 128; stride; stride >>= 1) {
    if (lane < stride) {
      second[lane] = fmaxf(fminf(first[lane], first[lane + stride]),
                           fmaxf(second[lane], second[lane + stride]));
      first[lane] = fmaxf(first[lane], first[lane + stride]);
    }
    __syncthreads();
  }
  if (lane == 0) out->target_gap[blockIdx.x] = first[0] - second[0];
}

__global__ void mtp_transition_kernel(const StepInDev* in, const StepOutDev* out, int32_t T,
                                      StepScratch* scratch) {
  const int32_t A = out->accepted;
  const int j = threadIdx.x;
  if (j < T) scratch->alignment_ids[j] = j < A ? in->tokens[j + 1] : out->anchor;
  if (j < 4) scratch->ar_positions[j] = in->positions[0] + A + 1 + j;
}

__global__ void select_accepted_kernel(const float* src, StepOutDev* out, float* dst,
                                       StepScratch* scratch, StepDraftDists* dists) {
  const int32_t A = out->accepted;
  const float4* s = reinterpret_cast<const float4*>(src + std::size_t(A) * dims::kStreamHidden);
  float4* d = reinterpret_cast<float4*>(dst);
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < dims::kStreamHidden / 4;
       i += blockDim.x * gridDim.x)
    d[i] = s[i];
  if (blockIdx.x != 0) return;
  constexpr int kWords = int(sizeof(kernels::DraftDistribution) / sizeof(int32_t));
  if (int(threadIdx.x) < kWords)
    reinterpret_cast<int32_t*>(&dists->next[0])[threadIdx.x] =
        reinterpret_cast<const int32_t*>(&dists->bridge[A])[threadIdx.x];
  if (threadIdx.x == 0) {
    const int32_t tok = scratch->mtp_token[A];
    out->mtp_drafts[0] = tok;
    scratch->ar_token[0] = tok;
  }
}

__global__ void ar_advance_kernel(const float* src, float* dst, StepOutDev* out,
                                  StepScratch* scratch, int32_t i) {
  const float4* s = reinterpret_cast<const float4*>(src);
  float4* d = reinterpret_cast<float4*>(dst);
  for (int t = threadIdx.x + blockIdx.x * blockDim.x; t < dims::kStreamHidden / 4;
       t += blockDim.x * gridDim.x)
    d[t] = s[t];
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const int32_t tok = scratch->mtp_token[0];
    out->mtp_drafts[i + 1] = tok;
    scratch->ar_token[i + 1] = tok;
  }
}

__global__ void gather_rows_kernel(const uint8_t* src_codes, const __nv_bfloat16* src_scales,
                                   const int32_t* vocab, int32_t k, uint8_t* dst_codes,
                                   __nv_bfloat16* dst_scales) {
  const int32_t row = blockIdx.x;
  const int32_t src = vocab[row];
  const auto* s = reinterpret_cast<const uint4*>(src_codes + std::size_t(src) * k);
  auto* d = reinterpret_cast<uint4*>(dst_codes + std::size_t(row) * k);
  for (int i = threadIdx.x; i < k / 16; i += blockDim.x) d[i] = s[i];
  if (threadIdx.x == 0) dst_scales[row] = src_scales[src];
}

}  // namespace

void step_copy(void* dst, const void* src, std::size_t bytes, cudaStream_t s) {
  if (bytes % 16 != 0) throw std::invalid_argument("step_copy: size must be a multiple of 16");
  copy16_kernel<<<1, 32, 0, s>>>(static_cast<int4*>(dst), static_cast<const int4*>(src), int(bytes / 16));
  CUDA_CHECK(cudaGetLastError());
}

void step_finish_single(const StepInDev* in, StepOutDev* out, cudaStream_t s) {
  finish_single_kernel<<<1, 32, 0, s>>>(in, out);
  CUDA_CHECK(cudaGetLastError());
}

void step_logit_gaps(const float* logits, int32_t rows, StepOutDev* out, cudaStream_t s) {
  logit_gaps_kernel<<<rows, 256, 0, s>>>(logits, out);
  CUDA_CHECK(cudaGetLastError());
}

void step_set_gdn_commit(const StepOutDev* out, kernels::GdnStepControl* control, cudaStream_t s) {
  set_commit_kernel<<<1, 32, 0, s>>>(out, control);
  CUDA_CHECK(cudaGetLastError());
}

void step_mtp_transition(const StepInDev* in, const StepOutDev* out, int32_t T,
                         StepScratch* scratch, cudaStream_t s) {
  mtp_transition_kernel<<<1, 32, 0, s>>>(in, out, T, scratch);
  CUDA_CHECK(cudaGetLastError());
}

void step_select_accepted(const float* src, StepOutDev* out, float* dst, StepScratch* scratch,
                          StepDraftDists* dists, cudaStream_t s) {
  select_accepted_kernel<<<10, 256, 0, s>>>(src, out, dst, scratch, dists);
  CUDA_CHECK(cudaGetLastError());
}

void step_ar_advance(const float* src, float* dst, StepOutDev* out, StepScratch* scratch,
                     int32_t i, cudaStream_t s) {
  ar_advance_kernel<<<10, 256, 0, s>>>(src, dst, out, scratch, i);
  CUDA_CHECK(cudaGetLastError());
}

void gather_fp8_rows(const uint8_t* src_codes, const __nv_bfloat16* src_scales, const int32_t* vocab,
                     int32_t rows, int32_t k, uint8_t* dst_codes, __nv_bfloat16* dst_scales,
                     cudaStream_t s) {
  if (k % 16 != 0) throw std::invalid_argument("gather_fp8_rows: k must be a multiple of 16");
  gather_rows_kernel<<<rows, 128, 0, s>>>(src_codes, src_scales, vocab, k, dst_codes, dst_scales);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace fngine::model
