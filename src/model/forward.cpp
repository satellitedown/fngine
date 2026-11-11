#include "model/forward.h"

#include "core/device.h"
#include "core/model_dims.h"
#include "kernels/gdn/gdn.h"
#include "kernels/gr/gr.h"
#include "kernels/head/head.h"
#include "kernels/mtp/mtp.h"
#include "kernels/ple/ple.h"
#include "kernels/qsa/qsa.h"

#include <stdexcept>

namespace fngine::model {
namespace k = fngine::kernels;

namespace {

void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess)
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

k::Fp8RowWeight fp8(const Fp8Weight& w) { return {w.codes, w.scales, w.n, w.k}; }
k::Bf16Weight bf16(const Bf16Weight& w) { return {w.data, w.n, w.k}; }

}  // namespace

Forward::Forward(const ForwardContext& ctx) : ctx_(ctx) {
  if (!ctx_.weights || !ctx_.state || !ctx_.ws || !ctx_.cublas)
    throw std::invalid_argument("Forward: weights, state, workspace and cublas are required");
  if (ctx_.dense_bf16 && !ctx_.weights->has_bf16())
    throw std::invalid_argument("Forward: dense_bf16 requires BF16 weight copies");
  if (ctx_.placement) {
    for (int l = 0; l < dims::kLayers; ++l) {
      layouts_[std::size_t(l)] = k::make_moe_expert_layout(
          ctx_.placement->slot_base(l), int64_t(ctx_.placement->blob_stride(l)),
          ctx_.gate_up_format[std::size_t(l)], ctx_.down_format[std::size_t(l)]);
    }
  }
  const ExpertBlobs& mtp = ctx_.weights->mtp().experts;
  if (mtp.base)
    mtp_layout_ = k::make_moe_expert_layout(mtp.base, int64_t(mtp.blob_stride), k::MoeFormat::IQ4_NL,
                                            k::MoeFormat::IQ4_NL);
  // The lists name the FP8 dense weights; BF16 validation mode reads other copies.
  if (ctx_.prefetch_stream && ctx_.prefetch_budget > 0 && !ctx_.dense_bf16) {
    check(cudaEventCreateWithFlags(&fork_, cudaEventDisableTiming), "prefetch fork event");
    check(cudaEventCreateWithFlags(&join_, cudaEventDisableTiming), "prefetch join event");
    build_prefetch_lists();
  }
  if (ctx_.aux_stream && ctx_.fp8_path != k::Fp8LinearPath::A8) {
    check(cudaEventCreateWithFlags(&aux_fork_, cudaEventDisableTiming), "aux fork event");
    check(cudaEventCreateWithFlags(&aux_join_, cudaEventDisableTiming), "aux join event");
  }
}

Forward::~Forward() {
  if (fork_) cudaEventDestroy(fork_);
  if (join_) cudaEventDestroy(join_);
  if (aux_fork_) cudaEventDestroy(aux_fork_);
  if (aux_join_) cudaEventDestroy(aux_join_);
}

void Forward::build_prefetch_lists() {
  const DenseWeights& W = *ctx_.weights;
  std::array<L2PrefetchList, dims::kLayers> lists{};
  for (int32_t l = 0; l < dims::kLayers; ++l) {
    L2PrefetchList& list = lists[std::size_t(l)];
    auto add = [&](const void* p, uint64_t bytes) {
      if (!p || bytes < 4096) return;  // tiny tensors are not worth a request
      if (list.count >= kMaxL2Ranges) throw std::logic_error("L2 prefetch list overflow");
      list.ranges[list.count++] = L2Range{p, bytes};
    };
    auto fp8w = [&](const Fp8Weight& w) {
      add(w.codes, uint64_t(w.n) * uint64_t(w.k));
      add(w.scales, uint64_t(w.n) * sizeof(__nv_bfloat16));
    };
    auto bf16w = [&](const Bf16Weight& w) { add(w.data, uint64_t(w.n) * uint64_t(w.k) * sizeof(__nv_bfloat16)); };
    auto f32w = [&](const F32Weight& w) { add(w.data, uint64_t(w.count) * sizeof(float)); };
    auto gr = [&](const GrWeights& w) {
      f32w(w.hc_norm);
      bf16w(w.inject);
      fp8w(w.down);
      fp8w(w.up);
    };
    const int32_t next = l + 1;
    if (next == dims::kLayers) {
      // Last layer: the final mixer, then the head of lm_head (rows are read in order).
      const MixerWeights& m = W.mixer();
      f32w(m.hc_norm);
      fp8w(m.down);
      fp8w(m.up);
      add(W.lm_head().scales, uint64_t(W.lm_head().n) * sizeof(__nv_bfloat16));
      add(W.lm_head().codes, uint64_t(W.lm_head().n) * uint64_t(W.lm_head().k));
      continue;
    }
    const LayerWeights& L = W.layer(next);
    if (next == dims::kPleLayer) {
      const PleWeights& p = W.ple();
      fp8w(p.key);
      fp8w(p.value);
      f32w(p.norm_key);
      f32w(p.norm_query);
      f32w(p.norm_conv);
      f32w(p.conv);
    }
    gr(L.attn_gr);
    if (L.qsa) {
      fp8w(L.attn.qkv);
      bf16w(L.attn.idx_qk);
      fp8w(L.attn.o);
    } else {
      fp8w(L.gdn.qkvz);
      bf16w(L.gdn.ab);
      f32w(L.gdn.conv);
      fp8w(L.gdn.out);
    }
    gr(L.mlp_gr);
    bf16w(L.moe.router);
    fp8w(L.moe.shared_gate_up);
    fp8w(L.moe.shared_down);
  }
  prefetch_lists_ = DeviceBuffer(sizeof(lists));
  prefetch_lists_.copy_from_host(lists.data(), sizeof(lists));
}

void Forward::prefetch_next(int32_t l, cudaStream_t s) {
  if (!fork_ || l < 0 || l >= dims::kLayers) return;
  const auto* lists = static_cast<const L2PrefetchList*>(prefetch_lists_.p);
  check(cudaEventRecord(fork_, s), "prefetch fork");
  check(cudaStreamWaitEvent(ctx_.prefetch_stream, fork_, 0), "prefetch fork wait");
  // Gate: only layers with CPU-tier work leave DRAM idle long enough to pay off.
  check(l2_prefetch(lists + l, ctx_.prefetch_budget, &ctx_.ws->moe_decode.work->n_cold,
                    ctx_.prefetch_stream),
        "l2_prefetch");
  pending_join_ = true;
}

void Forward::join_prefetch(cudaStream_t s) {
  if (!pending_join_) return;
  check(cudaEventRecord(join_, ctx_.prefetch_stream), "prefetch join");
  check(cudaStreamWaitEvent(s, join_, 0), "prefetch join wait");
  pending_join_ = false;
}

cudaStream_t Forward::fork_background(cudaStream_t s) {
  if (!fork_) return nullptr;
  check(cudaEventRecord(fork_, s), "background fork");
  check(cudaStreamWaitEvent(ctx_.prefetch_stream, fork_, 0), "background fork wait");
  pending_join_ = true;
  return ctx_.prefetch_stream;
}

cudaStream_t Forward::fork_aux(const StepIo& io, cudaStream_t s) {
  if (!aux_fork_ || io.pass == Pass::Prefill) return nullptr;
  check(cudaEventRecord(aux_fork_, s), "aux fork");
  check(cudaStreamWaitEvent(ctx_.aux_stream, aux_fork_, 0), "aux fork wait");
  return ctx_.aux_stream;
}

void Forward::join_aux(cudaStream_t s) {
  check(cudaEventRecord(aux_join_, ctx_.aux_stream), "aux join");
  check(cudaStreamWaitEvent(s, aux_join_, 0), "aux join wait");
}

void Forward::dense(const __nv_bfloat16* x, int32_t T, const Fp8Weight& w,
                    const k::LinearEpilogueParams& ep, cudaStream_t s) {
  Workspace& ws = *ctx_.ws;
  if (ctx_.dense_bf16) {
    k::linear_bf16(x, T, {w.bf16, w.n, w.k}, ep, ctx_.cublas, ws.linear_ws, ws.linear_ws_bytes, s);
  } else {
    // lm_head has no A8 path; it always runs A16.
    const k::Fp8LinearPath path = w.n == dims::kVocab ? k::Fp8LinearPath::Auto : ctx_.fp8_path;
    k::linear_fp8(x, T, fp8(w), ep, ws.linear_ws, ws.linear_ws_bytes, s, path);
  }
}

void Forward::dense(const __nv_bfloat16* x, int32_t T, const Bf16Weight& w,
                    const k::LinearEpilogueParams& ep, cudaStream_t s) {
  Workspace& ws = *ctx_.ws;
  k::linear_bf16(x, T, bf16(w), ep, ctx_.cublas, ws.linear_ws, ws.linear_ws_bytes, s);
}

void Forward::embed(const StepIo& io, cudaStream_t s) {
  Workspace& ws = *ctx_.ws;
  k::embed_gather(ctx_.weights->embedding().device, io.tokens, io.T, ws.X, ws.emb, s);
}

void Forward::gr_front(const float* X, int32_t T, const float* hc_norm,
                       const __nv_bfloat16* inject, const Fp8Weight& down, cudaStream_t s) {
  Workspace& ws = *ctx_.ws;
  if (!ctx_.dense_bf16 && ctx_.fp8_path != k::Fp8LinearPath::A8 && T <= k::kGrNormDownMaxTokens) {
    k::gr_norm_down(X, T, hc_norm, inject, down.codes, down.scales, ws.Xn, inject ? ws.inj : nullptr,
                    ws.r, s);
    return;
  }
  k::gr_norm(X, T, hc_norm, inject, ws.Xn, inject ? ws.inj : nullptr, s);
  dense(ws.Xn, T, down, k::linear_silu_quarter(ws.r), s);
}

void Forward::gr_read(const GrWeights& w, const float* X, int32_t T, cudaStream_t s) {
  Workspace& ws = *ctx_.ws;
  gr_front(X, T, w.hc_norm.data, w.inject.data, w.down, s);
  dense(ws.r, T, w.up, k::linear_gr_mix(ws.mix, ws.Xn), s);
}

void Forward::final_mixer(const MixerWeights& w, const float* X, int32_t rows,
                          __nv_bfloat16* mix_out, cudaStream_t s) {
  Workspace& ws = *ctx_.ws;
  gr_front(X, rows, w.hc_norm.data, nullptr, w.down, s);
  dense(ws.r, rows, w.up, k::linear_gr_mix(mix_out, ws.Xn), s);
}

void Forward::lm_head(const __nv_bfloat16* mix, int32_t rows, float* logits, cudaStream_t s) {
  if (rows > ctx_.ws->options.logit_rows) throw std::invalid_argument("lm_head: too many rows");
  dense(mix, rows, ctx_.weights->lm_head(), k::linear_store_f32(logits), s);
}

void Forward::draft_head(const __nv_bfloat16* mix, int32_t rows, float* logits, cudaStream_t s) {
  if (rows > ctx_.ws->options.logit_rows) throw std::invalid_argument("draft_head: too many rows");
  if (!ctx_.weights->draft_head().codes) throw std::logic_error("draft_head: no draft vocabulary");
  dense(mix, rows, ctx_.weights->draft_head(), k::linear_store_f32(logits), s);
}

void Forward::ple(const StepIo& io, cudaStream_t s) {
  Workspace& ws = *ctx_.ws;
  const PleWeights& w = ctx_.weights->ple();
  k::PleRowsWait wait{};
  if (io.pass != Pass::Prefill && ctx_.exchange) {
    ExchangeControl* control = &ctx_.exchange->control;
    wait.ready = &control->ple_ready.value;
    wait.expected = io.seq;
    wait.error = &control->error.value;
  }
  k::ple_dequant(ctx_.ple_rows, io.T, ws.E, wait, s);
  dense(ws.E, io.T, w.key, k::linear_store_bf16(ws.K), s);
  dense(ws.E, io.T, w.value, k::linear_store_bf16(ws.V), s);
  const k::PleMixWeights mw{w.norm_key.data, w.norm_query.data, w.norm_conv.data, w.conv.data};
  k::ple_mix(ws.X, ws.K, ws.V, mw, ctx_.state->ple_conv_state(), ws.c_record, io.T,
             /*commit_state=*/io.pass != Pass::Verify, ws.ple_ws, ws.ple_ws_bytes, s);
}

void Forward::gdn(int32_t l, const StepIo& io, cudaStream_t s) {
  Workspace& ws = *ctx_.ws;
  ModelState& st = *ctx_.state;
  const GdnWeights& w = ctx_.weights->layer(l).gdn;
  const int32_t gi = dims::gdn_index(l);
  const int32_t T = io.T;
  // Decode/verify: ab + gating only feed the recurrence, so they run beside qkvz + conv.
  const cudaStream_t aux = fork_aux(io, s);
  if (aux) {
    dense(ws.mix, T, w.ab, k::linear_store_f32(ws.ab), aux);
    k::gdn_gating(ws.ab, w.A_log.data, w.dt_bias.data, ws.g, ws.beta, T, aux);
  }
  dense(ws.mix, T, w.qkvz, k::linear_store_bf16(ws.qkvz), s);
  if (!aux) dense(ws.mix, T, w.ab, k::linear_store_f32(ws.ab), s);
  switch (io.pass) {
    case Pass::Prefill:
      k::gdn_conv_prefill(ws.qkvz, w.conv.data, st.gdn_conv(gi), st.gdn_conv(gi), ws.conv_out, T, s);
      k::gdn_gating(ws.ab, w.A_log.data, w.dt_bias.data, ws.g, ws.beta, T, s);
      k::gdn_prefill(ws.conv_out, ws.g, ws.beta, st.gdn_recurrent(gi), st.gdn_recurrent(gi), ws.core,
                     T, ws.gdn_ws, ws.gdn_ws_bytes, s);
      break;
    case Pass::Decode:
      k::gdn_conv_decode(ws.qkvz, w.conv.data, st.gdn_pool(), gi, st.gdn_control(), ws.conv_out, s);
      if (aux) join_aux(s);
      else k::gdn_gating(ws.ab, w.A_log.data, w.dt_bias.data, ws.g, ws.beta, T, s);
      k::gdn_recurrent_decode(ws.conv_out, ws.g, ws.beta, st.gdn_pool(), gi, st.gdn_control(),
                              ws.core, s);
      break;
    case Pass::Verify:
      k::gdn_conv_record(ws.qkvz, w.conv.data, st.gdn_pool(), gi, st.gdn_control(),
                         st.gdn_records(), T, ws.conv_out, s);
      if (aux) join_aux(s);
      else k::gdn_gating(ws.ab, w.A_log.data, w.dt_bias.data, ws.g, ws.beta, T, s);
      k::gdn_recurrent_record(ws.conv_out, ws.g, ws.beta, st.gdn_pool(), gi, st.gdn_control(),
                              st.gdn_records(), T, ws.core, s);
      break;
  }
  k::gdn_out_norm(ws.core, ws.qkvz + dims::kGdnConvChannels, dims::kGdnQkvzRows, w.norm.data,
                  ws.normed, T, s);
  dense(ws.normed, T, w.out, k::linear_gr_write(ws.X, ws.inj), s);
}

void Forward::qsa(const QsaWeights& w, int32_t cache, float* X, const StepIo& io, cudaStream_t s) {
  Workspace& ws = *ctx_.ws;
  const QsaLayerState& st = ctx_.state->qsa(cache);
  const int32_t T = io.T;
  // Decode/verify: the indexer projection and ring/key update run beside qkv + norm/rope + append.
  const cudaStream_t aux = fork_aux(io, s);
  if (aux) {
    dense(ws.mix, T, w.idx_qk, k::linear_store_bf16(ws.idx), aux);
    k::indexer_prepare(ws.idx, io.positions, T, w.idx_q_norm.data, w.idx_k_norm.data, st.ring,
                       st.block_keys, ws.q_idx, aux);
  }
  dense(ws.mix, T, w.qkv, k::linear_store_bf16(ws.qkv), s);
  if (!aux) dense(ws.mix, T, w.idx_qk, k::linear_store_bf16(ws.idx), s);
  k::qsa_qk_norm_rope(ws.qkv, w.q_norm.data, w.k_norm.data, io.positions, T, ws.q, ws.k, s);
  k::qsa_kv_append(st.kv, ws.k, ws.qkv + dims::kQsaQProjRows + dims::kQsaKvHeads * dims::kQsaHeadDim,
                   dims::kQsaQkvRows, io.positions, T, s);
  if (aux) {
    join_aux(s);
  } else {
    k::indexer_prepare(ws.idx, io.positions, T, w.idx_q_norm.data, w.idx_k_norm.data, st.ring,
                       st.block_keys, ws.q_idx, s);
  }
  const int32_t max_blocks =
      io.pass == Pass::Prefill ? std::max<int32_t>(1, (io.last_position + 1) / 4) : ctx_.state->max_blocks();
  k::indexer_score_topk(ws.q_idx, st.block_keys, io.positions, T, max_blocks, ws.score_ws,
                        ws.score_ws_bytes, ws.selected, ws.counts, s);
  k::qsa_attention(st.kv, ws.q, ws.qkv, io.positions, ws.selected, ws.counts, T, ws.attn_ws,
                   ws.attn_ws_bytes, ws.attn, s);
  dense(ws.attn, T, w.o, k::linear_gr_write(X, ws.inj), s);
}

void Forward::moe(const MoeWeights& w, int32_t l, float* X, const StepIo& io, cudaStream_t s) {
  Workspace& ws = *ctx_.ws;
  const int32_t T = io.T;
  const bool mtp = l < 0;
  // Abliterated packs (docs/engineering-notes.md, "Abliterated packs") edit the routed-expert sum
  // along the refusal direction in the combine kernels. The MTP layer is left alone: the
  // transferred abliteration does not touch the MTP head.
  const float* const ablate =
      (!mtp && ctx_.ablation_directions != nullptr)
          ? ctx_.ablation_directions + std::size_t(l) * std::size_t(dims::kHidden)
          : nullptr;
  const float ablate_scale = ablate != nullptr ? ctx_.ablation_scales[std::size_t(l)] : 1.0f;
  const k::MoeExpertLayout& layout = mtp ? mtp_layout_ : layouts_[std::size_t(l)];
  const int32_t* slot_of = (mtp || !ctx_.placement)
                               ? nullptr
                               : ctx_.placement->slot_of_device() + std::size_t(l) * dims::kExperts;
  DecodeExchange* exchange = mtp ? nullptr : ctx_.exchange;
  dense(ws.mix, T, w.router, k::linear_store_f32(ws.router), s);
  if (io.pass == Pass::Prefill) {
    PrefillStream* ps = ctx_.prefill_stream;
    if (!ps) throw std::logic_error("Forward: prefill requires the prefill expert tables");
    const uint8_t* const* blobs = mtp ? ps->mtp_table() : ps->table(l);
    ExchangeControl* control = exchange ? &exchange->control : nullptr;
    const PrefillExchangeView view = exchange ? ctx_.prefill : PrefillExchangeView{};
    check(k::moe_route(ws.router, ws.mix, blobs, mtp ? 0 : l, io.seq, control, view, T,
                       ws.moe_prefill, s), "moe_route");
    dense(ws.mix, T, w.shared_gate_up, k::linear_swiglu(ws.shared_h), s);
    dense(ws.shared_h, T, w.shared_down,
          k::linear_shared_scale(ws.y_shared, ws.router + dims::kExperts, dims::kRouterRows), s);
    // Streamed experts: their ring buffer must be filled before the first blob read.
    if (!mtp) ps->wait_ready(l, s);
    check(k::moe_grouped_gate_up(layout, blobs, ws.mix, T, ws.moe_prefill, s), "moe_grouped_gate_up");
    check(k::moe_grouped_down(layout, blobs, T, ws.moe_prefill, s), "moe_grouped_down");
    check(k::moe_prefill_combine(ws.moe_prefill, ws.y_shared, control, view, mtp ? 0 : l, io.seq,
                                 ws.inj, X, nullptr, T, ctx_.moe_timeout_cycles, ablate,
                                 ablate_scale, s),
          "moe_prefill_combine");
    if (!mtp) ps->release(l, s);
  } else {
    uint32_t* counts = (ctx_.routing_counts && !mtp)
                           ? ctx_.routing_counts + std::size_t(l) * dims::kExperts
                           : nullptr;
    check(k::moe_select_publish(ws.router, ws.mix, slot_of, mtp ? 0 : l, io.seq, exchange, T,
                                ws.moe_decode, s, counts), "moe_select_publish");
    // The shared expert only meets the routed experts in the combine, so it runs beside the hot
    // experts (its down epilogue reads the shared-gate logit of the router above). The fork comes
    // after routing: sharing SMs with the single-CTA router selection would delay both the CPU
    // tier and the hot experts.
    const cudaStream_t aux = fork_aux(io, s);
    const cudaStream_t shared = aux ? aux : s;
    dense(ws.mix, T, w.shared_gate_up, k::linear_swiglu(ws.shared_h), shared);
    dense(ws.shared_h, T, w.shared_down,
          k::linear_shared_scale(ws.y_shared, ws.router + dims::kExperts, dims::kRouterRows), shared);
    check(k::moe_hot_gate_up(layout, T, ws.moe_decode, s), "moe_hot_gate_up");
    check(k::moe_hot_down(layout, T, ws.moe_decode, s), "moe_hot_down");
    if (aux) join_aux(s);
    // The prefetch competes for DRAM with the expert work above (hot experts and, on the aux
    // stream, the shared expert), so it starts when the layer only waits for the CPU tier.
    if (!mtp) prefetch_next(l, s);
    check(k::moe_combine(ws.moe_decode, ws.y_shared, exchange, mtp ? 0 : l, io.seq, ws.inj, X,
                         nullptr, T, ctx_.moe_timeout_cycles, ablate, ablate_scale, s),
          "moe_combine");
  }
}

void Forward::layer(int32_t l, const StepIo& io, cudaStream_t s) {
  Workspace& ws = *ctx_.ws;
  const LayerWeights& L = ctx_.weights->layer(l);
  if (l == dims::kPleLayer) ple(io, s);
  gr_read(L.attn_gr, ws.X, io.T, s);
  if (L.qsa) {
    qsa(L.attn, dims::qsa_index(l), ws.X, io, s);
  } else {
    gdn(l, io, s);
  }
  gr_read(L.mlp_gr, ws.X, io.T, s);
  moe(L.moe, l, ws.X, io, s);
}

void Forward::mtp_layer(const float* x_src, const StepIo& io, cudaStream_t s) {
  Workspace& ws = *ctx_.ws;
  const MtpWeights& m = ctx_.weights->mtp();
  if (!m.experts.base) throw std::logic_error("MTP experts are not loaded");
  if (x_src == ws.X_mtp) throw std::invalid_argument("mtp_layer: x_src may not alias ws.X_mtp");
  const int32_t T = io.T;
  // Hidden branch first: x_src may live in a buffer the embedding gather below reuses.
  k::rmsnorm_bf16(x_src, T, dims::kStreamHidden, m.pre_fc_norm_hidden.data, ws.hid_n, s);
  dense(ws.hid_n, T * dims::kStreams, m.fc_hidden, k::linear_store_bf16(ws.h), s);
  // Embedding branch (embed_gather also writes 4 streams; X_mtp is overwritten by the merge).
  k::embed_gather(ctx_.weights->embedding().device, io.tokens, T, ws.X_mtp, ws.emb, s);
  k::rmsnorm_bf16(ws.emb, T, dims::kHidden, m.pre_fc_norm_embedding.data, ws.emb_n, s);
  dense(ws.emb_n, T, m.fc_embedding, k::linear_store_bf16(ws.e), s);
  k::mtp_merge_streams(ws.h, ws.e, T, ws.X_mtp, s);
  // One QSA decoder layer without PLE; all 512 experts on the GPU.
  gr_read(m.attn_gr, ws.X_mtp, T, s);
  qsa(m.attn, kMtpCache, ws.X_mtp, io, s);
  gr_read(m.mlp_gr, ws.X_mtp, T, s);
  moe(m.moe, -1, ws.X_mtp, io, s);
}

}  // namespace fngine::model
