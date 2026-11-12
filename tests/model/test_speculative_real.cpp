// Speculative decoding checks on the real model (plan M9):
//   1. greedy with automatic speculation produces the same tokens as plain greedy decode; a
//      divergence is allowed only where both runs' top-2 logit gaps are < 0.05;
//   2. MTP first-draft acceptance on a greedy chat prompt >= 0.55;
//   3. rollback: after 50 verify rounds with mixed accepted/rejected drafts of width 2..9, the GDN
//      state, PLE conv state, indexer rings, the KV of committed positions and the indexer keys of
//      committed blocks equal a plain T=1 decode of the same committed tokens within 1e-3 relative;
//   4. session restore: a snapshot taken mid-prompt and restored after 64 decoded tokens gives
//      byte-identical state and the identical greedy continuation (the engine-level reuse test
//      cannot compare outputs: decode-built and prefill-built states legitimately differ);
//   5. sampling: MTP drafts verified against the distributions they were drawn from are accepted
//      more often than MTP drafts verified as fixed tokens.
// Needs FNGINE_TEST_PACK and ~25 GB VRAM / ~30 GB RAM; returns 77 otherwise.
#include "common/test_util.h"
#include "core/device.h"
#include "core/model_dims.h"
#include "engine/speculation.h"
#include "model/runtime.h"
#include "model/step_graph.h"
#include "text/frontend.h"
#include "text/prompt_lookup.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <span>
#include <vector>

using namespace fngine;

namespace {

constexpr int kTokens = 256;

std::vector<int32_t> chat_prompt(const text::Frontend& fe, const std::string& user, bool thinking = false) {
  PromptInput in;
  in.messages.push_back({ChatRole::User, user, {}, {}, {}});
  in.options.enable_thinking = thinking;
  return fe.prepare(in).token_ids;
}


struct PlainRun {
  std::vector<int32_t> tokens;  // generated tokens
  std::vector<float> gaps;      // top-2 gap of the logits that produced tokens[i]
};

PlainRun plain_greedy(model::ModelRuntime& rt, const std::vector<int32_t>& prompt) {
  model::StepRunner runner(rt, true, false, true);
  runner.set_sampling(kernels::SamplingConfig{});
  rt.reset_sequence();
  rt.prefill(prompt);
  PlainRun r;
  model::StepResult s = runner.first_token(int32_t(prompt.size()));
  r.tokens.push_back(s.licensed[0]);
  r.gaps.push_back(s.target_gap[0]);
  while (int(r.tokens.size()) < kTokens) {
    s = runner.step(r.tokens.back(), {});
    r.tokens.push_back(s.licensed[0]);
    r.gaps.push_back(s.target_gap[0]);
  }
  return r;
}

struct SpecRun {
  std::vector<int32_t> tokens;
  std::vector<float> gaps;
  int rounds = 0, first_offered = 0, first_accepted = 0, offered = 0, accepted = 0;
};

SpecRun spec_greedy(model::ModelRuntime& rt, const std::vector<int32_t>& prompt, bool automatic) {
  model::StepRunner runner(rt, true, true, true);
  runner.set_sampling(kernels::SamplingConfig{});
  rt.reset_sequence();
  rt.prefill(prompt);
  SpecRun r;
  model::StepResult s = runner.first_token(int32_t(prompt.size()));
  r.tokens.push_back(s.licensed[0]);
  r.gaps.push_back(s.target_gap[0]);
  engine::SpeculationController controller(engine::SpeculationMode::Auto);
  text::PromptLookup lookup;
  std::vector<int32_t> context = prompt;
  context.push_back(s.licensed[0]);
  while (int(r.tokens.size()) < kTokens) {
    int32_t look[8];
    const int nlook = int(lookup.propose(context, look));
    const auto choice = automatic
        ? controller.decide(s.mtp_count, nlook, std::exp(lookup.log_probability()))
        : engine::SpecDecision{engine::SpecSource::Mtp, s.mtp_count, 0};
    const int k = std::min(choice.k, kTokens - int(r.tokens.size()) - 1);
    const int32_t* proposal = choice.source == engine::SpecSource::Lookup ? look : s.mtp_drafts;
    s = runner.step(r.tokens.back(), std::span<const int32_t>(proposal, std::size_t(k)));
    controller.record_step(k + 1, s.seconds, 0.0);
    if (nlook) lookup.observe(s.tokens());
    ++r.rounds;
    if (k > 0 && choice.source == engine::SpecSource::Mtp) {
      controller.record_mtp_outcome(k, s.accepted);
      r.offered += k;
      r.accepted += s.accepted;
      ++r.first_offered;
      if (s.accepted >= 1) ++r.first_accepted;
    } else if (k > 0 && choice.source == engine::SpecSource::Lookup) {
      controller.record_lookup_outcome(0, k, s.accepted);
    } else {
      controller.record_plain_round();
    }
    for (int i = 0; i < s.count; ++i) {
      r.tokens.push_back(s.licensed[i]);
      r.gaps.push_back(s.target_gap[i]);
      context.push_back(s.licensed[i]);
    }
  }
  r.tokens.resize(kTokens);
  return r;
}

// Sampling with all MTP drafts every round. `coupled`: the drafts are declared as the previous
// step's MTP drafts, so the acceptance uses the distributions they were drawn from. The flat
// sampling distribution (temperature 1.5 over the top 20) separates the two acceptance rules:
// a drawn draft d survives with probability p(d) as a fixed token and min(1, p(d)/q(d)) coupled.
SpecRun spec_sampled(model::ModelRuntime& rt, const std::vector<int32_t>& prompt, bool coupled) {
  model::StepRunner runner(rt, true, true);
  kernels::SamplingConfig config;
  config.temperature = 1.5f;
  config.top_k = 20;
  config.seed = 0x5eed;
  runner.set_sampling(config);
  rt.reset_sequence();
  rt.prefill(prompt);
  SpecRun r;
  model::StepResult s = runner.first_token(int32_t(prompt.size()));
  r.tokens.push_back(s.licensed[0]);
  while (int(r.tokens.size()) < kTokens) {
    const int k = std::min<int>(s.mtp_count, kTokens - int(r.tokens.size()) - 1);
    int32_t drafts[model::kMtpDrafts];
    std::copy(std::begin(s.mtp_drafts), std::end(s.mtp_drafts), drafts);
    s = runner.step(r.tokens.back(), std::span<const int32_t>(drafts, std::size_t(k)), coupled);
    ++r.rounds;
    r.offered += k;
    r.accepted += s.accepted;
    if (k > 0) {
      ++r.first_offered;
      if (s.accepted >= 1) ++r.first_accepted;
    }
    for (int i = 0; i < s.count; ++i) r.tokens.push_back(s.licensed[i]);
  }
  return r;
}

// ---------------------------------------------------------------- rollback comparison
struct StateImage {
  std::vector<unsigned char> snapshot;               // GDN + PLE + rings
  std::vector<std::vector<__nv_bfloat16>> k, v;      // BF16 KV of committed positions per cache
  std::vector<std::vector<__nv_bfloat16>> blocks;    // committed block keys per cache
};

StateImage capture(model::ModelRuntime& rt, int32_t positions) {
  StateImage im;
  im.snapshot.resize(rt.snapshot_bytes());
  rt.snapshot_to(im.snapshot.data());
  const int32_t pages = (positions + 63) / 64;
  const int32_t blocks = positions / 4;
  for (int c = 0; c < model::kQsaCaches - 1; ++c) {  // main layers (MTP not maintained in T=1 run)
    const auto& q = rt.state().qsa(c);
    const size_t elems = size_t(positions) * 2 * 256;
    im.k.emplace_back(elems);
    im.v.emplace_back(elems);
    // Include the committed part of the tail page; uncommitted slots are scratch.
    for (int page = 0; page < pages; ++page) {
      const int nt = std::min(64, positions - page * 64);
      for (int h = 0; h < 2; ++h) {
        const size_t src = size_t(page * 2 + h) * 64 * 256;
        const size_t dst = size_t(page * 64) * 512 + h * 256;
        CUDA_CHECK(cudaMemcpy2D(im.k.back().data() + dst, 512 * 2,
            static_cast<const __nv_bfloat16*>(q.kv.k) + src, 256 * 2,
            256 * 2, nt, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy2D(im.v.back().data() + dst, 512 * 2,
            static_cast<const __nv_bfloat16*>(q.kv.v) + src, 256 * 2,
            256 * 2, nt, cudaMemcpyDeviceToHost));
      }
    }
    im.blocks.emplace_back(size_t(blocks) * 128);
    CUDA_CHECK(cudaMemcpy(im.blocks.back().data(), q.block_keys, size_t(blocks) * 128 * 2,
                          cudaMemcpyDeviceToHost));
  }
  return im;
}

// rms(a - b) / rms(a): the plan's "within 1e-3 relative" bound.
double rel_diff_bf16(const std::vector<__nv_bfloat16>& a, const std::vector<__nv_bfloat16>& b) {
  double dd = 0, sq = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double x = test::bf16_to_float(a[i]), y = test::bf16_to_float(b[i]);
    dd += (x - y) * (x - y);
    sq += x * x;
  }
  return sq > 0 ? std::sqrt(dd / sq) : std::sqrt(dd);
}

}  // namespace

int main() {
  const std::string pack = test::env_or_skip("FNGINE_TEST_PACK");
  if (pack.empty() || test::skip_without_gpu()) return test::kSkip;
  const text::Frontend fe = text::Frontend::load(std::filesystem::path(pack) / "tokenizer");
  bool ok = true;

  {
    model::RuntimeOptions ro;
    ro.pack = pack;
    ro.max_context = 4096;
    ro.prefill_chunk = 1024;
    ro.adaptive_experts = false;  // hold placement fixed for the numerical comparison
    model::ModelRuntime rt(ro);
    const std::vector<std::string> prompts = {
        "Write a C function that reverses a singly linked list, then explain it in two sentences.",
        "List the first twenty prime numbers and explain how you would test whether 9973 is prime.",
        "Summarize the causes of the French Revolution in five bullet points.",
    };
    int total_first_offered = 0, total_first_accepted = 0;
    for (std::size_t p = 0; p < prompts.size(); ++p) {
      const std::vector<int32_t> ids = chat_prompt(fe, prompts[p]);
      const PlainRun plain = plain_greedy(rt, ids);
      const SpecRun spec = spec_greedy(rt, ids, true);
      const SpecRun mtp = spec_greedy(rt, ids, false);
      total_first_offered += mtp.first_offered;
      total_first_accepted += mtp.first_accepted;
      int diverge = -1;
      for (int i = 0; i < kTokens; ++i)
        if (plain.tokens[size_t(i)] != spec.tokens[size_t(i)]) { diverge = i; break; }
      const bool allowed = diverge < 0 ||
          (plain.gaps[size_t(diverge)] < 0.05f && spec.gaps[size_t(diverge)] < 0.05f);
      std::printf("[check1] prompt %zu: %s (first divergence %d, top-2 gaps plain %.4f / auto %.4f); "
                  "auto rounds %d, MTP accepted %d/%d drafts\n",
                  p, allowed ? "PASS" : "FAIL", diverge,
                  diverge < 0 ? 0.0f : plain.gaps[size_t(diverge)],
                  diverge < 0 ? 0.0f : spec.gaps[size_t(diverge)],
                  spec.rounds, mtp.accepted, mtp.offered);
      ok = ok && allowed;
    }
    const double first_rate = double(total_first_accepted) / std::max(1, total_first_offered);
    std::printf("[check2] MTP first-draft acceptance %.3f (%s >= 0.55)\n", first_rate,
                first_rate >= 0.55 ? "PASS" : "FAIL");
    ok = ok && first_rate >= 0.55;

    {
      int offered[2] = {}, accepted[2] = {};
      for (const std::string& prompt : prompts) {
        const std::vector<int32_t> ids = chat_prompt(fe, prompt, true);
        for (int coupled = 0; coupled < 2; ++coupled) {
          const SpecRun run = spec_sampled(rt, ids, coupled != 0);
          offered[coupled] += run.first_offered;
          accepted[coupled] += run.first_accepted;
        }
      }
      const double fixed = double(accepted[0]) / std::max(1, offered[0]);
      const double coupled = double(accepted[1]) / std::max(1, offered[1]);
      const bool pass5 = coupled > fixed + 0.1;
      std::printf("[check5] sampled first MTP draft accepted: %.3f against its distribution (%d of %d), "
                  "%.3f as a fixed token (%d of %d) -> %s\n",
                  coupled, accepted[1], offered[1], fixed, accepted[0], offered[0], pass5 ? "PASS" : "FAIL");
      ok = ok && pass5;
    }

    // A previously captured MTP graph must not write drafts beyond the final cache page.
    // The target still processes every remaining position; only unusable proposals stop.
    model::StepRunner tail_runner(rt, true, true);
    tail_runner.set_sampling(kernels::SamplingConfig{});
    const auto tail_ids = chat_prompt(fe, prompts.front());
    std::vector<int32_t> tail_prompt(std::size_t(ro.max_context - 3), tail_ids.front());
    rt.reset_sequence();
    rt.prefill(tail_prompt);
    auto tail = tail_runner.first_token(int32_t(tail_prompt.size()));
    FNGINE_CHECK(tail.mtp_count == model::kMtpDrafts);
    tail = tail_runner.step(tail.licensed[0], {});
    FNGINE_CHECK(tail.mtp_count == model::kMtpDrafts);
    const int32_t tail_anchor = tail.licensed[0];
    const auto tail_history = rt.ngram_history();
    std::vector<unsigned char> before_tail(rt.snapshot_bytes());
    rt.snapshot_to(before_tail.data());
    tail = tail_runner.step(tail.licensed[0], {});
    FNGINE_CHECK(tail.mtp_count == 0);
    const int32_t penultimate = tail.licensed[0];
    tail = tail_runner.step(tail.licensed[0], {});
    FNGINE_CHECK(tail.mtp_count == 0 && rt.position() == ro.max_context);
    const int32_t final = tail.licensed[0];
    const auto capture_tail = [&] {
      std::vector<unsigned char> bytes(rt.snapshot_bytes());
      rt.snapshot_to(bytes.data());
      for (int c = 0; c < model::kQsaCaches; ++c) {
        const auto& q = rt.state().qsa(c);
        const auto kv_bytes = kernels::qsa_kv_layer_bytes(rt.state().kv_format(), rt.state().capacity());
        const auto key_bytes = std::size_t(rt.state().max_blocks()) * 128 * sizeof(__nv_bfloat16);
        const auto at = bytes.size();
        bytes.resize(at + kv_bytes + key_bytes);
        CUDA_CHECK(cudaMemcpy(bytes.data() + at, q.kv.k, kv_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(bytes.data() + at + kv_bytes, q.block_keys, key_bytes, cudaMemcpyDeviceToHost));
      }
      return bytes;
    };
    const auto mtp_tail_state = capture_tail();
    bool context_rejected = false;
    try {
      (void)tail_runner.step(tail.licensed[0], {});
    } catch (const std::length_error&) {
      context_rejected = true;
    }
    FNGINE_CHECK(context_rejected);
    rt.restore_from(before_tail.data(), ro.max_context - 2, tail_history);
    model::StepRunner plain_tail_runner(rt, true, false);
    plain_tail_runner.set_sampling(kernels::SamplingConfig{});
    auto plain_tail = plain_tail_runner.step(tail_anchor, {});
    FNGINE_CHECK(plain_tail.count == 1 && plain_tail.licensed[0] == penultimate);
    plain_tail = plain_tail_runner.step(plain_tail.licensed[0], {});
    FNGINE_CHECK(plain_tail.count == 1 && plain_tail.licensed[0] == final);
    FNGINE_CHECK(capture_tail() == mtp_tail_state);

    // first_token has its own MTP launch path: at the final prompt position it must leave
    // persistent state unchanged and sample exactly the same target token as a plain runner.
    rt.restore_from(before_tail.data(), ro.max_context - 2, tail_history);
    rt.prefill(std::span<const int32_t>(&tail_anchor, 1));
    const auto before_first = capture_tail();
    const auto mtp_first = tail_runner.first_token(rt.position());
    const auto plain_first = plain_tail_runner.first_token(rt.position());
    FNGINE_CHECK(mtp_first.mtp_count == 0 && mtp_first.count == 1 &&
                 mtp_first.licensed[0] == plain_first.licensed[0]);
    FNGINE_CHECK(capture_tail() == before_first);
  }

  {
    // Rollback: BF16 KV so cache contents compare as floats.
    model::RuntimeOptions ro;
    ro.pack = pack;
    ro.max_context = 4096;
    ro.prefill_chunk = 1024;
    ro.kv = kernels::QsaKvFormat::BF16;
    ro.mtp = false;
    ro.adaptive_experts = false;
    model::ModelRuntime rt(ro);
    const std::vector<int32_t> ids = chat_prompt(fe, "Explain how a hash table handles collisions, with an example in Python.");
    const PlainRun ref = plain_greedy(rt, ids);  // greedy continuation used as the draft oracle

    model::StepRunner runner(rt, true, false);
    runner.set_sampling(kernels::SamplingConfig{});
    rt.reset_sequence();
    rt.prefill(ids);
    std::vector<int32_t> processed;  // tokens processed by verify steps, in order
    model::StepResult s = runner.first_token(int32_t(ids.size()));
    int32_t anchor = s.licensed[0];
    std::mt19937 rng(1234);
    int rounds_with_reject = 0, rounds_full_accept = 0;
    for (int round = 0; round < 50; ++round) {
      const int k = 1 + int(rng() % 8);
      // Oracle continuation after `anchor`: the plain run's tokens at the matching index when the
      // sequences agree so far; corrupt each draft with probability 0.3.
      const std::size_t gen = processed.size() + 1;  // index of the token after anchor in ref
      std::vector<int32_t> drafts;
      for (int j = 0; j < k; ++j) {
        int32_t d = gen + size_t(j) < ref.tokens.size() ? ref.tokens[gen + size_t(j)] : 1000 + j;
        if (rng() % 10 < 3) d = int32_t(rng() % 200000);
        drafts.push_back(d);
      }
      processed.push_back(anchor);
      s = runner.step(anchor, drafts);
      for (int j = 0; j < s.accepted; ++j) processed.push_back(s.licensed[j]);
      anchor = s.licensed[s.count - 1];
      if (s.accepted < k) ++rounds_with_reject; else ++rounds_full_accept;
    }
    const int32_t P = rt.position();
    const StateImage spec_state = capture(rt, P);

    rt.reset_sequence();
    rt.prefill(ids);
    for (const int32_t t : processed) runner.step(t, {});
    if (rt.position() != P) {
      std::printf("[check3] FAIL position mismatch %d vs %d\n", rt.position(), P);
      return 1;
    }
    const StateImage plain_state = capture(rt, P);

    // Snapshot regions: GDN recurrent (FP32) and conv/PLE/rings (BF16).
    const auto regions = rt.state().snapshot_regions();
    double worst = 0;
    size_t off = 0;
    for (std::size_t r = 0; r < regions.size(); ++r) {
      const size_t bytes = regions[r].bytes;
      double d;
      if (r == 0) {
        const float* a = reinterpret_cast<const float*>(spec_state.snapshot.data() + off);
        const float* b = reinterpret_cast<const float*>(plain_state.snapshot.data() + off);
        double dd = 0, sq = 0;
        for (size_t i = 0; i < bytes / 4; ++i) {
          dd += double(a[i] - b[i]) * double(a[i] - b[i]);
          sq += double(a[i]) * a[i];
        }
        d = sq > 0 ? std::sqrt(dd / sq) : std::sqrt(dd);
      } else {
        // Only raw keys in the committed partial block are live. Other ring entries may
        // legitimately contain rejected drafts, which the next verify overwrites.
        const size_t start = r >= 3 ? size_t((P / 4 * 4) % 16) * 128 * 2 : 0;
        const size_t live = r >= 3 ? size_t(P % 4) * 128 * 2 : bytes;
        std::vector<__nv_bfloat16> a(live / 2), b(live / 2);
        std::memcpy(a.data(), spec_state.snapshot.data() + off + start, live);
        std::memcpy(b.data(), plain_state.snapshot.data() + off + start, live);
        d = rel_diff_bf16(a, b);
      }
      std::printf("[check3] snapshot region %zu (%s): rms(diff)/rms %.3e\n", r,
                  r == 0 ? "GDN recurrent" : r == 1 ? "GDN conv" : r == 2 ? "PLE conv" : "indexer ring", d);
      worst = std::max(worst, d);
      off += (bytes + 255) / 256 * 256;
    }
    for (std::size_t c = 0; c < spec_state.k.size(); ++c) {
      const double dk = rel_diff_bf16(spec_state.k[c], plain_state.k[c]);
      const double dv = rel_diff_bf16(spec_state.v[c], plain_state.v[c]);
      const double db = rel_diff_bf16(spec_state.blocks[c], plain_state.blocks[c]);
      worst = std::max({worst, dk, dv, db});
      if (c == 0 || c + 1 == spec_state.k.size())
        std::printf("[check3] QSA cache %zu: K %.3e V %.3e block keys %.3e\n", c, dk, dv, db);
    }
    std::printf("[check3] %d rounds with a rejection, %d fully accepted; committed %zu tokens; worst rms(diff)/rms %.3e\n",
                rounds_with_reject, rounds_full_accept, processed.size(), worst);
    const bool pass3 = worst <= 1e-3 && rounds_with_reject > 0 && rounds_full_accept > 0;
    std::printf("[check3] %s\n", pass3 ? "PASS" : "FAIL");
    ok = ok && pass3;

    // Session restore: a snapshot restored after the sequence moved on reproduces the original
    // state bytes and continuation exactly (same process, fixed placement, same chunk widths).
    const std::size_t m = ids.size() / 2;
    const std::span<const int32_t> head(ids.data(), m), tail(ids.data() + m, ids.size() - m);
    auto continuation = [&](int n) {
      std::vector<int32_t> out;
      model::StepResult r = runner.first_token(int32_t(ids.size()));
      out.push_back(r.licensed[0]);
      while (int(out.size()) < n) {
        r = runner.step(out.back(), {});
        out.push_back(r.licensed[0]);
      }
      return out;
    };
    rt.reset_sequence();
    rt.prefill(head);
    std::vector<unsigned char> saved(rt.snapshot_bytes()), restored(rt.snapshot_bytes());
    rt.snapshot_to(saved.data());
    const io::NgramHistory history = rt.ngram_history();
    rt.prefill(tail);
    const std::vector<int32_t> first = continuation(64);  // also moves KV/state/history past m
    rt.restore_from(saved.data(), int32_t(m), history);
    rt.snapshot_to(restored.data());
    rt.prefill(tail);
    const std::vector<int32_t> second = continuation(64);
    const bool pass4 = saved == restored && first == second;
    std::printf("[check4] session snapshot at %zu of %zu prompt tokens: state %s, continuation %s -> %s\n",
                m, ids.size(), saved == restored ? "identical" : "DIFFERS",
                first == second ? "identical" : "DIFFERS", pass4 ? "PASS" : "FAIL");
    ok = ok && pass4;
  }
  std::printf("%s\n", ok ? "PASS speculative checks" : "FAIL speculative checks");
  return ok ? 0 : 1;
}
