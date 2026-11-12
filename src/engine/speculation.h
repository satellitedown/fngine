#pragma once

// Cost-model speculation controller (pure host logic).
//
// Each decode round verifies T = 1 + k tokens in one step: the next token plus k drafts from one
// source — MTP (up to 3 drafts produced by the previous step) or prompt lookup (up to 8 drafts).
// The controller picks the (source, k) maximizing expected committed tokens per second:
//
//   value(source, k) = (1 + Σ_{i=1..k} Π_{j=1..i} a_j) / L(1 + k)
//
// where a_j is the probability that draft j is accepted given drafts 1..j-1 were, and L(T) is the
// expected step latency at width T. k = 0 is plain decode (source None, T = 1). Exact ties
// (relative 1e-9) go to the smaller T; between sources at equal T, MTP wins.
//
// Latency model. A step of width T whose CPU expert tier worked C seconds takes
//
//   seconds = D (1 + γ T) + c C,        L(T) = D (1 + γ T) + c κ T
//
//  - D (seconds): the work every step does once (dense weights, head, drafting). It follows the
//    measured steps quickly (EMA α = 0.2 of the model's error): step time changes with the
//    engine's state and with other GPU work, and such changes hit every width alike. A per-width
//    estimate would charge them to the width in use, and a width abandoned for that reason would
//    keep its bad estimate for good.
//  - γ: GPU cost of one more token relative to D (its hot experts). Starts at 0.08 and is learned
//    from steps whose width differs from the recent ones.
//  - c: how much of the CPU tier's work delays the step (the GPU's own expert work hides the
//    rest). Starts at 0.7 and is learned from the step-to-step variation of C.
//  - κ (seconds per token): CPU-tier work per verified token, EMA (α = 0.05) of C / T. This is
//    what moves most between workloads (it follows the share of routed experts that are not on
//    the GPU), and every step measures it.
//  γ and c move by normalized LMS steps (rates 0.08 and 0.03) on the model's error, within
//  [0.01, 0.5] and [0, 1.5].
//  - Probes supply the widths γ is learned from, and observations of the acceptance positions the
//    best choice leaves out: a round verifies one draft fewer or one more than the best choice
//    (alternating, as available; never plain decode). Probes come at least 32 decisions apart,
//    and further apart (up to 128) the more throughput the model expects one to lose, so that
//    they cost at most 0.4% of it.
// Acceptance:
//  - MTP: per-position conditional acceptance averages a_1..a_3 starting at 0.7/0.65/0.65. A
//    round that offered n drafts and accepted m observes position i for i ≤ min(n, m + 1) only
//    (position i is observed only when 1..i-1 were accepted): 1 for i ≤ m, 0 for i = m + 1.
//    An observation moves a_i by 1/w_i of the difference, where the weight w_i counts the
//    position's observations (it starts at 10 and stops at 50) and loses 1% (down to 5) in
//    every round that does not offer the position. A long-observed estimate therefore moves
//    slowly (2% per observation), while one the controller stopped offering, which only probes
//    still observe, follows them quickly instead of keeping its last verdict.
//  - Prompt lookup: one per-token conditional acceptance EMA (α = 0.05) per window (8, 4, 2 — the
//    PromptLookup windows), starting at 0.8. A round's per-token probability is PromptLookup's
//    own estimate (exp(log_probability())) when the caller passes one in (0, 1]; otherwise the
//    EMA of the proposal's window, or the lowest window EMA when the window is unknown.

#include <array>
#include <cstdint>
#include <optional>
#include <string_view>

namespace fngine::engine {

enum class SpeculationMode { Auto, Mtp, Lookup, None };
enum class SpecSource { None, Mtp, Lookup };

// Parses "auto" | "mtp" | "lookup" | "none".
std::optional<SpeculationMode> parse_speculation_mode(std::string_view s);
const char* to_string(SpeculationMode m);
const char* to_string(SpecSource s);

struct SpecDecision {
  SpecSource source = SpecSource::None;
  int k = 0;              // drafts to verify; step width T = 1 + k
  double value = 0.0;     // expected committed tokens per second of the choice
  bool probe = false;     // a probe round: one draft off the best choice
  int width() const { return 1 + k; }
};

struct SpecSourceStats {
  uint64_t decisions = 0;        // rounds decide() chose this source (None: plain decode)
  uint64_t rounds = 0;           // outcomes recorded
  uint64_t drafts_offered = 0;
  uint64_t drafts_accepted = 0;
  // Committed tokens per recorded round (accepted drafts + the verified token).
  double tokens_per_round() const {
    return rounds ? double(drafts_accepted + rounds) / double(rounds) : 0.0;
  }
  double acceptance() const {
    return drafts_offered ? double(drafts_accepted) / double(drafts_offered) : 0.0;
  }
};

class SpeculationController {
 public:
  static constexpr int kMaxWidth = 9;
  static constexpr int kMaxMtpDrafts = 3;
  static constexpr int kMaxLookupDrafts = 8;
  static constexpr std::array<int, 3> kLookupWindows{8, 4, 2};
  static constexpr std::array<double, kMaxMtpDrafts> kMtpAcceptance{0.7, 0.65, 0.65};
  static constexpr double kStartWeight = 10.0;   // observations a starting value counts as
  static constexpr double kMaxWeight = 50.0, kMinWeight = 5.0;
  static constexpr double kWeightDecay = 0.99;   // per round that does not offer the position
  static constexpr double kLookupAlpha = 0.05;   // lookup window EMAs
  // Latency model before any measurement (this engine on an RTX 5090: about 6.5 ms at T = 1).
  static constexpr double kBaseSeconds = 0.0055;       // D
  static constexpr double kTokenShare = 0.08;          // γ
  static constexpr double kCpuDelay = 0.7;             // c
  static constexpr double kCpuTokenSeconds = 0.0008;   // κ
  static constexpr double kBaseAlpha = 0.2;      // D, and the recent width and CPU work
  static constexpr double kCpuTokenAlpha = 0.05; // κ
  static constexpr double kTokenShareRate = 0.08;
  static constexpr double kCpuDelayRate = 0.03;
  static constexpr double kMinTokenShare = 0.01, kMaxTokenShare = 0.5;
  static constexpr double kMaxCpuDelay = 1.5;
  static constexpr int kProbeInterval = 32;      // least decisions between probes
  static constexpr int kMaxProbeInterval = 128;
  static constexpr double kProbeBudget = 0.004;  // share of throughput probes may cost

  explicit SpeculationController(SpeculationMode mode, int max_mtp = kMaxMtpDrafts,
                                 int max_lookup = kMaxLookupDrafts);

  // Chooses the source and draft count for the next round. `*_available` are the drafts the
  // sources can supply now; `lookup_token_prob` is PromptLookup's per-token probability
  // (exp(log_probability())), or ≤ 0 / NaN to use the window EMA of `lookup_window` (8, 4, 2;
  // 0 = unknown). Pinned modes only consider their own source (None → always plain decode).
  SpecDecision decide(int mtp_drafts_available, int lookup_drafts_available, float lookup_token_prob,
                      int lookup_window = 0);

  // Measured wall time of one verify/decode step of width T (1..9) and the time its CPU expert
  // tier worked (summed over the layers).
  void record_step(int T, double seconds, double cpu_seconds);
  // Outcome of a round that verified `offered` drafts of which the first `accepted` matched.
  void record_mtp_outcome(int offered, int accepted);
  void record_lookup_outcome(int window, int offered, int accepted);
  // A plain decode round (k = 0) committed one token.
  void record_plain_round();

  // Model state.
  double step_latency(int T) const;  // L(T) as used by decide()
  double base_seconds() const { return base_; }             // D
  double token_share() const { return token_share_; }       // γ
  double cpu_delay() const { return cpu_delay_; }           // c
  double cpu_token_seconds() const { return cpu_token_; }   // κ
  uint64_t steps_measured(int T) const { return measured_[size_t(T - 1)]; }
  double mtp_acceptance(int position) const { return mtp_accept_[size_t(position - 1)]; }  // 1..3
  double lookup_acceptance(int window) const;                                             // 8/4/2
  double expected_value(SpecSource source, int k, double lookup_token_prob = 0.0) const;

  SpeculationMode mode() const { return mode_; }
  const SpecSourceStats& stats(SpecSource s) const { return stats_[size_t(s)]; }
  uint64_t probes() const { return probes_; }
  // Committed tokens per round over all recorded rounds.
  double tokens_per_round() const;

 private:
  static int window_index(int window);
  double lookup_prob(float token_prob, int window) const;
  // Expected committed tokens of `k` drafts from `source` (lookup at per-token probability p).
  double expected_tokens(SpecSource source, int k, double p) const;
  // MTP positions past `offered` were not offered this round: their weights decay.
  void age_mtp(int offered);

  SpeculationMode mode_;
  int max_mtp_;
  int max_lookup_;
  double base_ = kBaseSeconds;
  double token_share_ = kTokenShare;
  double cpu_delay_ = kCpuDelay;
  double cpu_token_ = kCpuTokenSeconds;
  double recent_width_ = 1.0;  // EMAs of the measured widths and CPU-tier seconds
  double recent_cpu_ = 0.0;
  bool measured_any_ = false;
  std::array<uint64_t, kMaxWidth> measured_{};
  std::array<double, kMaxMtpDrafts> mtp_accept_ = kMtpAcceptance;
  std::array<double, kMaxMtpDrafts> mtp_weight_{kStartWeight, kStartWeight, kStartWeight};
  std::array<double, kLookupWindows.size()> lookup_accept_{0.8, 0.8, 0.8};
  std::array<SpecSourceStats, 3> stats_{};
  int since_probe_ = 0;
  bool probe_wider_ = false;  // the direction the next probe tries first
  uint64_t probes_ = 0;
};

}  // namespace fngine::engine
