#include "engine/speculation.h"

#include <algorithm>
#include <cmath>

#include "text/prompt_lookup.h"

namespace fngine::engine {

static_assert(SpeculationController::kLookupWindows[0] == int(text::PromptLookup::kWindows[0]) &&
                  SpeculationController::kLookupWindows[1] == int(text::PromptLookup::kWindows[1]) &&
                  SpeculationController::kLookupWindows[2] == int(text::PromptLookup::kWindows[2]),
              "lookup windows must match PromptLookup");
static_assert(1 + SpeculationController::kMaxLookupDrafts <= SpeculationController::kMaxWidth);

std::optional<SpeculationMode> parse_speculation_mode(std::string_view s) {
  if (s == "auto") return SpeculationMode::Auto;
  if (s == "mtp") return SpeculationMode::Mtp;
  if (s == "lookup") return SpeculationMode::Lookup;
  if (s == "none") return SpeculationMode::None;
  return std::nullopt;
}

const char* to_string(SpeculationMode m) {
  switch (m) {
    case SpeculationMode::Auto: return "auto";
    case SpeculationMode::Mtp: return "mtp";
    case SpeculationMode::Lookup: return "lookup";
    case SpeculationMode::None: return "none";
  }
  return "?";
}

const char* to_string(SpecSource s) {
  switch (s) {
    case SpecSource::None: return "none";
    case SpecSource::Mtp: return "mtp";
    case SpecSource::Lookup: return "lookup";
  }
  return "?";
}

SpeculationController::SpeculationController(SpeculationMode mode, int max_mtp, int max_lookup)
    : mode_(mode),
      max_mtp_(std::clamp(max_mtp, 0, kMaxMtpDrafts)),
      max_lookup_(std::clamp(max_lookup, 0, kMaxLookupDrafts)) {}

int SpeculationController::window_index(int window) {
  for (size_t i = 0; i < kLookupWindows.size(); ++i)
    if (kLookupWindows[i] == window) return int(i);
  return -1;
}

double SpeculationController::lookup_acceptance(int window) const {
  const int i = window_index(window);
  return i >= 0 ? lookup_accept_[size_t(i)] : *std::min_element(lookup_accept_.begin(), lookup_accept_.end());
}

double SpeculationController::lookup_prob(float token_prob, int window) const {
  if (token_prob > 0.0f && token_prob <= 1.0f) return double(token_prob);  // false for NaN
  return lookup_acceptance(window);
}

double SpeculationController::step_latency(int T) const {
  const double t = double(std::clamp(T, 1, kMaxWidth));
  return base_ * (1.0 + token_share_ * t) + cpu_delay_ * cpu_token_ * t;
}

double SpeculationController::expected_tokens(SpecSource source, int k, double p) const {
  double tokens = 1.0, run = 1.0;
  if (source == SpecSource::Mtp) {
    for (int i = 0; i < std::clamp(k, 0, kMaxMtpDrafts); ++i) tokens += (run *= mtp_accept_[size_t(i)]);
  } else if (source == SpecSource::Lookup) {
    for (int i = 0; i < std::clamp(k, 0, kMaxLookupDrafts); ++i) tokens += (run *= p);
  }
  return tokens;
}

double SpeculationController::expected_value(SpecSource source, int k, double lookup_token_prob) const {
  if (source == SpecSource::Mtp) k = std::clamp(k, 0, kMaxMtpDrafts);
  else if (source == SpecSource::Lookup) k = std::clamp(k, 0, kMaxLookupDrafts);
  else k = 0;
  return expected_tokens(source, k, lookup_prob(float(lookup_token_prob), 0)) / step_latency(1 + k);
}

SpecDecision SpeculationController::decide(int mtp_drafts_available, int lookup_drafts_available,
                                           float lookup_token_prob, int lookup_window) {
  const bool use_mtp = mode_ == SpeculationMode::Auto || mode_ == SpeculationMode::Mtp;
  const bool use_lookup = mode_ == SpeculationMode::Auto || mode_ == SpeculationMode::Lookup;
  const int kmtp = use_mtp ? std::clamp(mtp_drafts_available, 0, max_mtp_) : 0;
  const int klook = use_lookup ? std::clamp(lookup_drafts_available, 0, max_lookup_) : 0;
  const double p = lookup_prob(lookup_token_prob, lookup_window);
  const auto value = [&](SpecSource source, int k) {
    return expected_tokens(source, k, p) / step_latency(1 + k);
  };

  SpecDecision best{SpecSource::None, 0, 1.0 / step_latency(1)};
  // Walk widths in increasing order; a candidate must beat the best by a relative margin, so
  // ties keep the smaller T (and MTP over lookup at equal T).
  for (int k = 1; k <= std::max(kmtp, klook); ++k) {
    if (k <= kmtp) {
      const double v = value(SpecSource::Mtp, k);
      if (v > best.value * (1.0 + 1e-9)) best = {SpecSource::Mtp, k, v};
    }
    if (k <= klook) {
      const double v = value(SpecSource::Lookup, k);
      if (v > best.value * (1.0 + 1e-9)) best = {SpecSource::Lookup, k, v};
    }
  }
  // Probe: one draft more or fewer than the best choice, from its source (from plain decode: the
  // first draft of a source that has one), so that γ keeps being observed. The rounds between
  // probes grow with the throughput the model expects a probe to lose, which keeps that loss
  // within kProbeBudget.
  if ((kmtp > 0 || klook > 0) && ++since_probe_ >= kProbeInterval) {
    const SpecSource source = best.source != SpecSource::None
                                  ? best.source
                                  : (kmtp > 0 ? SpecSource::Mtp : SpecSource::Lookup);
    const int available = source == SpecSource::Mtp ? kmtp : klook;
    const int wider = best.k + 1 <= available ? best.k + 1 : 0;
    const int narrower = best.k - 1 >= 1 ? best.k - 1 : 0;
    const int k = probe_wider_ ? (wider ? wider : narrower) : (narrower ? narrower : wider);
    if (k > 0) {
      const double probe_value = value(source, k);
      // Tokens the best choice commits during the probe's step, less the probe's own, in rounds
      // of the best choice.
      const double lost_rounds =
          std::max(0.0, best.value - probe_value) * step_latency(1 + k) /
          (best.value * step_latency(1 + best.k));
      const double interval = std::min(double(kMaxProbeInterval), lost_rounds / kProbeBudget);
      if (double(since_probe_) >= interval) {
        probe_wider_ = k < best.k;
        best = {source, k, probe_value, true};
        since_probe_ = 0;
        ++probes_;
      }
    }
  }
  ++stats_[size_t(best.source)].decisions;
  return best;
}

void SpeculationController::record_step(int T, double seconds, double cpu_seconds) {
  if (T < 1 || T > kMaxWidth || !(seconds > 0.0) || !std::isfinite(seconds)) return;
  if (!(cpu_seconds >= 0.0) || !std::isfinite(cpu_seconds)) cpu_seconds = 0.0;
  ++measured_[size_t(T - 1)];
  const double t = double(T);
  if (!measured_any_) {
    // The first step sets the level and the CPU work per token.
    base_ = std::max(0.1 * seconds, seconds - cpu_delay_ * cpu_seconds) / (1.0 + token_share_ * t);
    cpu_token_ = cpu_seconds / t;
    recent_width_ = t;
    recent_cpu_ = cpu_seconds;
    measured_any_ = true;
    return;
  }
  const double error = seconds - (base_ * (1.0 + token_share_ * t) + cpu_delay_ * cpu_seconds);
  // Normalized LMS: c from this step's CPU work against the recent steps', γ from its width
  // against theirs (the regularizers keep steps near the recent values from moving them).
  const double dc = cpu_seconds - recent_cpu_;
  const double dt = t - recent_width_;
  const double c_floor = 0.05 * base_;
  cpu_delay_ = std::clamp(cpu_delay_ + kCpuDelayRate * error * dc / (dc * dc + c_floor * c_floor),
                          0.0, kMaxCpuDelay);
  if (std::fabs(dt) >= 0.5)
    token_share_ = std::clamp(token_share_ + kTokenShareRate * error * dt / (base_ * (dt * dt + 0.25)),
                              kMinTokenShare, kMaxTokenShare);
  base_ = std::max(1e-6, base_ + kBaseAlpha * error / (1.0 + token_share_ * t));
  cpu_token_ += kCpuTokenAlpha * (cpu_seconds / t - cpu_token_);
  recent_width_ += kBaseAlpha * (t - recent_width_);
  recent_cpu_ += kBaseAlpha * (cpu_seconds - recent_cpu_);
}

void SpeculationController::age_mtp(int offered) {
  for (int i = std::max(offered, 0); i < kMaxMtpDrafts; ++i)
    mtp_weight_[size_t(i)] = std::max(kMinWeight, mtp_weight_[size_t(i)] * kWeightDecay);
}

// A round that offered `offered` drafts and accepted the first `accepted` observes position i
// (1-based) only if 1..i-1 were accepted: positions 1..min(offered, accepted + 1), 1 if accepted
// and 0 for the first rejection.
void SpeculationController::record_mtp_outcome(int offered, int accepted) {
  if (offered <= 0) return;
  accepted = std::clamp(accepted, 0, offered);
  const int observed = std::min({offered, accepted + 1, kMaxMtpDrafts});
  for (int i = 0; i < observed; ++i) {
    double& w = mtp_weight_[size_t(i)];
    w = std::min(kMaxWeight, w + 1.0);
    mtp_accept_[size_t(i)] += ((i < accepted ? 1.0 : 0.0) - mtp_accept_[size_t(i)]) / w;
  }
  age_mtp(offered);
  auto& s = stats_[size_t(SpecSource::Mtp)];
  ++s.rounds;
  s.drafts_offered += uint64_t(offered);
  s.drafts_accepted += uint64_t(accepted);
}

void SpeculationController::record_lookup_outcome(int window, int offered, int accepted) {
  if (offered <= 0) return;
  accepted = std::clamp(accepted, 0, offered);
  if (const int w = window_index(window); w >= 0) {
    // One per-token estimate: every observed token of the round updates it.
    double& a = lookup_accept_[size_t(w)];
    for (int i = 1; i <= std::min(offered, accepted + 1); ++i) a += kLookupAlpha * ((i <= accepted ? 1.0 : 0.0) - a);
  }
  auto& s = stats_[size_t(SpecSource::Lookup)];
  ++s.rounds;
  s.drafts_offered += uint64_t(offered);
  s.drafts_accepted += uint64_t(accepted);
}

void SpeculationController::record_plain_round() {
  age_mtp(0);
  ++stats_[size_t(SpecSource::None)].rounds;
}

double SpeculationController::tokens_per_round() const {
  uint64_t rounds = 0, tokens = 0;
  for (const auto& s : stats_) {
    rounds += s.rounds;
    tokens += s.rounds + s.drafts_accepted;
  }
  return rounds ? double(tokens) / double(rounds) : 0.0;
}

}  // namespace fngine::engine
