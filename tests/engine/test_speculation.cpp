// Speculation controller: decision boundaries against hand-computed values, the latency model
// (level, CPU-tier work, learned shares), probes, acceptance estimates, pinned modes and ties.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>

#include "engine/speculation.h"

using fngine::engine::SpecDecision;
using fngine::engine::SpecSource;
using fngine::engine::SpeculationController;
using fngine::engine::SpeculationMode;

namespace {

int g_failures = 0;

void check(bool ok, const char* what, int line) {
  if (!ok) {
    std::printf("FAIL line %d: %s\n", line, what);
    ++g_failures;
  }
}
#define CHECK(cond) check((cond), #cond, __LINE__)

bool near(double a, double b, double tol = 1e-12) { return std::fabs(a - b) <= tol * std::max(1.0, std::fabs(b)); }

bool is(const SpecDecision& d, SpecSource s, int k) { return d.source == s && d.k == k; }

constexpr int kProbe = SpeculationController::kProbeInterval;
constexpr int kMaxProbe = SpeculationController::kMaxProbeInterval;

// Deterministic uniform numbers in [0, 1).
struct Lcg {
  uint64_t state = 0x243f6a8885a308d3ull;
  double next() {
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    return double(state >> 11) / double(1ull << 53);
  }
};

// A step of width T takes base (1 + share T) + delay C, where its CPU tier works
// C = cpu_token T (1 ± cpu_noise) seconds; the whole is measured within ± noise.
struct Law {
  double base = 0.0055, share = 0.08, delay = 0.7, cpu_token = 0.0008;
  double cpu_noise = 0.5, noise = 0.05;
  double accept = 0.62;
  double mean(int T) const { return base * (1.0 + share * T) + delay * cpu_token * T; }
};

// Runs `rounds` MTP rounds under `law`. Returns the committed tokens per second of the last
// `tail` rounds (probes included) relative to the best fixed draft count.
double simulate(SpeculationController& c, const Law& law, int rounds, int tail, Lcg& rng) {
  double tokens = 0.0, seconds = 0.0;
  for (int round = 0; round < rounds; ++round) {
    const SpecDecision d = c.decide(3, 0, 0.0f);
    const int T = 1 + d.k;
    const double cpu = law.cpu_token * T * (1.0 + law.cpu_noise * (2.0 * rng.next() - 1.0));
    const double step = (law.base * (1.0 + law.share * T) + law.delay * cpu) *
                        (1.0 + law.noise * (2.0 * rng.next() - 1.0));
    c.record_step(T, step, cpu);
    int accepted = 0;
    while (accepted < d.k && rng.next() < law.accept) ++accepted;
    if (d.k > 0) c.record_mtp_outcome(d.k, accepted);
    else c.record_plain_round();
    if (round >= rounds - tail) {
      tokens += 1.0 + accepted;
      seconds += law.mean(T);
    }
  }
  double best = 0.0, expected = 1.0, run = 1.0;
  for (int k = 0; k <= 3; ++k) {
    if (k > 0) expected += (run *= law.accept);
    best = std::max(best, expected / law.mean(1 + k));
  }
  return tokens / seconds / best;
}

void test_priors() {
  // No measurements: L(T) = 5.5 ms (1 + .08 T) + .7 * .8 ms T = 5.5 + T ms.
  SpeculationController c(SpeculationMode::Auto);
  CHECK(near(c.step_latency(1), 0.0065));
  CHECK(near(c.step_latency(4), 0.0095));
  // MTP a = .7/.65/.65: k0 1/6.5 = 153.8, k1 1.7/7.5 = 226.7, k2 2.155/8.5 = 253.5,
  // k3 2.45075/9.5 = 258.0 -> k = 3.
  CHECK(near(c.expected_value(SpecSource::Mtp, 1), 1.7 / 0.0075));
  CHECK(near(c.expected_value(SpecSource::Mtp, 3), 2.45075 / 0.0095));
  SpeculationController m(SpeculationMode::Mtp);
  const auto dm = m.decide(3, 8, 0.0f);
  CHECK(is(dm, SpecSource::Mtp, 3));
  CHECK(near(dm.value, 2.45075 / 0.0095));
  CHECK(!dm.probe);
  CHECK(is(m.decide(2, 8, 0.0f), SpecSource::Mtp, 2));
  // Lookup p = .8: k4 3.3616/10.5 = 320.2 < k5 3.68928/11.5 = 320.8 > k6 3.951424/12.5 = 316.1,
  // above every MTP width -> auto picks lookup k = 5.
  const auto da = c.decide(3, 8, 0.0f);
  CHECK(is(da, SpecSource::Lookup, 5));
  CHECK(near(da.value, 3.68928 / 0.0115));
  // No drafts available -> plain decode.
  CHECK(is(c.decide(0, 0, 0.0f), SpecSource::None, 0));
}

void test_level() {
  // The first measurement sets the level for every width.
  SpeculationController c(SpeculationMode::Mtp);
  c.record_step(4, 0.019, 0.0032);
  CHECK(near(c.step_latency(4), 0.019));
  CHECK(c.step_latency(1) < c.step_latency(2) && c.step_latency(2) < c.step_latency(4));
  CHECK(near(c.cpu_token_seconds(), 0.0008));
  CHECK(c.steps_measured(4) == 1 && c.steps_measured(1) == 0);
  // Invalid samples are ignored.
  c.record_step(4, -1.0, 0.0);
  c.record_step(0, 0.010, 0.0);
  c.record_step(4, std::nan(""), 0.0);
  CHECK(near(c.step_latency(4), 0.019));
  CHECK(c.steps_measured(4) == 1);

  // A slowdown seen at one width moves every width, so it cannot turn the controller against
  // the width in use (with per-width estimates it did, and the abandoned width kept its bad
  // estimate for good); once it passes, the level and the choice are back.
  SpeculationController s(SpeculationMode::Mtp);
  for (int i = 0; i < 400; ++i) s.record_mtp_outcome(3, 2);  // a3 -> 0: two drafts are best
  for (int i = 0; i < 60; ++i) s.record_step(3, 0.0085, 0.0024);
  CHECK(is(s.decide(3, 0, 0.0f), SpecSource::Mtp, 2));
  for (int i = 0; i < 12; ++i) s.record_step(3, 0.0300, 0.0024);
  CHECK(s.step_latency(3) > 0.025);
  CHECK(s.step_latency(2) > 0.020 && s.step_latency(4) > s.step_latency(3));
  CHECK(s.decide(3, 0, 0.0f).k >= 2);
  for (int i = 0; i < 60; ++i) s.record_step(3, 0.0085, 0.0024);
  CHECK(near(s.step_latency(3), 0.0085, 1e-2));
  CHECK(is(s.decide(3, 0, 0.0f), SpecSource::Mtp, 2));
}

void test_cpu_work() {
  // The CPU tier's work per token is measured by every step, so the width follows it within a few
  // dozen steps. With a = 1/.65/.65 (after the outcomes below): the prior model verifies three
  // drafts (3.07/9.5); at 12 ms of CPU work per token, L(T) = 5.5 + 8.84 T ms, one draft is
  // best (2/23.2 > 2.65/32.0 > 3.07/40.9); without CPU work, L(T) = 5.5 + .44 T ms, three again.
  SpeculationController c(SpeculationMode::Mtp);
  for (int i = 0; i < 400; ++i) c.record_mtp_outcome(1, 1);
  CHECK(is(c.decide(3, 0, 0.0f), SpecSource::Mtp, 3));
  Lcg rng;
  for (int i = 0; i < 80; ++i) {
    const double cpu = 0.012 * 3 * (0.6 + 0.8 * rng.next());
    c.record_step(3, 0.0055 * 1.24 + 0.7 * cpu, cpu);
  }
  CHECK(std::fabs(c.cpu_token_seconds() - 0.012) < 0.0024);
  CHECK(is(c.decide(3, 0, 0.0f), SpecSource::Mtp, 1));
  for (int i = 0; i < 160; ++i) c.record_step(2, 0.0055 * 1.16, 0.0);
  CHECK(c.cpu_token_seconds() < 0.0001);
  CHECK(is(c.decide(3, 0, 0.0f), SpecSource::Mtp, 3));
}

void test_learning() {
  Lcg rng;
  {  // The prior model, noisy: near the best fixed width from the start.
    SpeculationController c(SpeculationMode::Mtp);
    CHECK(simulate(c, Law{}, 2000, 2000, rng) > 0.97);
  }
  {  // Much CPU work per token (L(T) = 5.5 + 2.12 T ms): two drafts beat three by 5%. Every step
     // measures it, so the controller is there within a few hundred rounds.
    SpeculationController c(SpeculationMode::Mtp);
    Law law;
    law.cpu_token = 0.0024;
    CHECK(simulate(c, law, 1000, 700, rng) > 0.97);
  }
  {  // The GPU's own expert work is the expensive part (share .25, L(T) = 5.5 + 1.94 T ms): two
     // drafts beat three by 5%; the prior verifies three. Learned from probes.
    SpeculationController c(SpeculationMode::Mtp);
    Law law;
    law.share = 0.25;
    CHECK(simulate(c, law, 12000, 4000, rng) > 0.97);
    CHECK(std::fabs(c.token_share() - 0.25) < 0.08);
  }
  {  // CPU work that hardly delays the step (delay .15 of 2.4 ms per token, L(T) = 5.5 + .8 T
     // ms): three drafts stay best; with the prior delay (.7) the model would drop one.
    SpeculationController c(SpeculationMode::Mtp);
    Law law;
    law.cpu_token = 0.0024;
    law.delay = 0.15;
    CHECK(simulate(c, law, 3000, 2000, rng) > 0.97);
    CHECK(std::fabs(c.cpu_delay() - 0.15) < 0.15);
  }
  {  // Speculation that never pays (share 1: L(2) = 1.5 L(1); acceptance .3): plain decode but
     // for the probes, which stay within their budget.
    SpeculationController c(SpeculationMode::Mtp);
    Law law;
    law.share = 1.0;
    law.cpu_token = 0.0;
    law.accept = 0.3;
    CHECK(simulate(c, law, 12000, 4000, rng) > 0.97);
    CHECK(c.probes() <= uint64_t(12000 / kProbe));
  }
  {  // The learned shares stay inside their bounds whatever the measurements claim.
    SpeculationController c(SpeculationMode::Mtp);
    for (int i = 0; i < 2000; ++i) c.record_step(i % 2 ? 4 : 2, i % 2 ? 0.001 : 0.050, i % 2 ? 0.004 : 0.0);
    CHECK(c.token_share() >= SpeculationController::kMinTokenShare);
    CHECK(c.cpu_delay() >= 0.0);
    SpeculationController d(SpeculationMode::Mtp);
    for (int i = 0; i < 2000; ++i) d.record_step(i % 2 ? 4 : 2, i % 2 ? 0.500 : 0.001, i % 2 ? 0.004 : 0.0);
    CHECK(d.token_share() <= SpeculationController::kMaxTokenShare);
    CHECK(d.cpu_delay() <= SpeculationController::kMaxCpuDelay);
    CHECK(d.step_latency(1) > 0.0 && std::isfinite(d.step_latency(9)));
  }
}

void test_probes() {
  {  // Best MTP k = 2 of 3 (a3 = 0): probes of k = 1 and k = 3 alternate, at least kProbe
     // decisions apart.
    SpeculationController c(SpeculationMode::Mtp);
    for (int i = 0; i < 400; ++i) c.record_mtp_outcome(3, 2);
    int probes = 0, last = 0, since = 0;
    for (int i = 1; i <= 16 * kProbe; ++i) {
      const SpecDecision d = c.decide(3, 0, 0.0f);
      ++since;
      if (!d.probe) {
        CHECK(is(d, SpecSource::Mtp, 2));
        continue;
      }
      ++probes;
      CHECK(since >= kProbe && since <= kMaxProbe);
      CHECK(d.source == SpecSource::Mtp && (d.k == 1 || d.k == 3));
      CHECK(d.k != last);
      CHECK(near(d.value, c.expected_value(SpecSource::Mtp, d.k)));
      last = d.k;
      since = 0;
    }
    CHECK(probes >= 8);
    CHECK(c.probes() == uint64_t(probes));
  }
  {  // Best k = 3 of 3: only the narrower width can be probed.
    SpeculationController c(SpeculationMode::Mtp);
    for (int i = 0; i < 400; ++i) c.record_mtp_outcome(3, 3);  // a -> 1
    for (int i = 1; i <= 2 * kMaxProbe; ++i) {
      const SpecDecision d = c.decide(3, 0, 0.0f);
      CHECK(is(d, SpecSource::Mtp, d.probe ? 2 : 3));
    }
    CHECK(c.probes() >= 2);
  }
  {  // Best k = 1: the probe never falls back to plain decode.
    SpeculationController c(SpeculationMode::Mtp);
    for (int i = 0; i < 400; ++i) c.record_mtp_outcome(3, 1);  // a1 -> 1, a2 -> 0
    for (int i = 1; i <= 2 * kMaxProbe; ++i) {
      const SpecDecision d = c.decide(3, 0, 0.0f);
      CHECK(is(d, SpecSource::Mtp, d.probe ? 2 : 1));
    }
    CHECK(c.probes() >= 2);
  }
  {  // Plain decode is best: the probe takes one draft, and only in a round that has one.
    SpeculationController c(SpeculationMode::Auto);
    for (int i = 0; i < 400; ++i) c.record_mtp_outcome(3, 0);  // a1 -> 0
    for (int i = 0; i < 4 * kMaxProbe; ++i) CHECK(is(c.decide(0, 0, 0.0f), SpecSource::None, 0));
    CHECK(c.probes() == 0);
    int first = 0;
    for (int i = 1; i <= kMaxProbe && !first; ++i) {
      const SpecDecision d = c.decide(3, 0, 0.0f);
      if (d.probe) {
        CHECK(is(d, SpecSource::Mtp, 1));
        first = i;
      } else {
        CHECK(is(d, SpecSource::None, 0));
      }
    }
    CHECK(first >= kProbe);
  }
  {  // The more throughput a probe is expected to lose, the rarer it is: dropping a third draft
     // that barely pays (priors) against one draft that always fails at 5 ms of CPU work per
     // token (L(1) = 9.4 ms, L(2) = 13.4 ms).
    SpeculationController cheap(SpeculationMode::Mtp), costly(SpeculationMode::Mtp);
    for (int i = 0; i < 400; ++i) costly.record_mtp_outcome(3, 0);
    costly.record_step(1, 0.00944, 0.005);
    for (int i = 0; i < 8 * kMaxProbe; ++i) {
      cheap.decide(3, 0, 0.0f);
      costly.decide(3, 0, 0.0f);
    }
    CHECK(cheap.probes() > 2 * costly.probes());
    CHECK(costly.probes() >= 8);
  }
  {  // Lookup rounds probe lookup widths.
    SpeculationController c(SpeculationMode::Lookup);
    for (int i = 1; i < kProbe; ++i) CHECK(is(c.decide(0, 8, 0.8f), SpecSource::Lookup, 5));
    const SpecDecision d = c.decide(0, 8, 0.8f);
    CHECK(d.probe && d.source == SpecSource::Lookup && (d.k == 4 || d.k == 6));
  }
  {  // No speculation, no probes.
    SpeculationController c(SpeculationMode::None);
    for (int i = 0; i < 4 * kProbe; ++i) CHECK(is(c.decide(3, 8, 0.99f), SpecSource::None, 0));
    CHECK(c.probes() == 0);
  }
}

void test_latency_shapes() {
  {  // Nearly free extra tokens (no CPU work, the token share at its floor): more drafts always
     // help -> max k per source.
    SpeculationController c(SpeculationMode::Auto);
    for (int i = 0; i < 2000; ++i) c.record_step(i % 2 ? 4 : 2, 0.010, 0.0);
    CHECK(near(c.token_share(), SpeculationController::kMinTokenShare));
    CHECK(is(c.decide(3, 0, 0.0f), SpecSource::Mtp, 3));
    // Lookup p = .8 over 8 drafts beats MTP's 2.45 tokens.
    CHECK(is(c.decide(3, 8, 0.0f), SpecSource::Lookup, 8));
    // Fewer lookup drafts available than the maximum: capped.
    CHECK(is(c.decide(3, 5, 0.0f), SpecSource::Lookup, 5));
  }
  {  // Boundary at the prior model (L(2) / L(1) = 7.5 / 6.5): one draft pays above an
     // acceptance of 2 / 13.
    SpeculationController c(SpeculationMode::Lookup, 3, 1);
    CHECK(is(c.decide(0, 1, 0.153f), SpecSource::None, 0));
    CHECK(is(c.decide(0, 1, 0.155f), SpecSource::Lookup, 1));
  }
}

void test_ties() {
  {  // Same width, different sources: the higher per-token acceptance wins (MTP a1 = .7).
    SpeculationController c(SpeculationMode::Auto);
    CHECK(is(c.decide(1, 1, 0.75f), SpecSource::Lookup, 1));
    CHECK(is(c.decide(1, 1, 0.65f), SpecSource::Mtp, 1));
    CHECK(is(c.decide(1, 1, 0.70f), SpecSource::Mtp, 1));  // equal: MTP
  }
  {  // All drafts rejected for long: gains of ~1e-9 tokens are ties -> T = 1.
    SpeculationController c(SpeculationMode::Auto);
    for (int i = 0; i < 2000; ++i) {
      c.record_mtp_outcome(3, 0);
      c.record_lookup_outcome(2, 8, 0);
    }
    CHECK(c.mtp_acceptance(1) < 1e-15);
    CHECK(is(c.decide(3, 8, 0.0f, 2), SpecSource::None, 0));
  }
}

void test_pinned() {
  SpeculationController none(SpeculationMode::None);
  CHECK(is(none.decide(3, 8, 0.99f), SpecSource::None, 0));

  SpeculationController mtp(SpeculationMode::Mtp);
  for (int i = 0; i < 40; ++i) mtp.record_mtp_outcome(3, 3);
  CHECK(is(mtp.decide(3, 8, 0.99f), SpecSource::Mtp, 3));  // lookup would win in auto
  CHECK(is(mtp.decide(2, 8, 0.99f), SpecSource::Mtp, 2));
  CHECK(is(mtp.decide(0, 8, 0.99f), SpecSource::None, 0));

  SpeculationController look(SpeculationMode::Lookup);
  CHECK(is(look.decide(3, 0, 0.0f), SpecSource::None, 0));
  CHECK(is(look.decide(3, 4, 0.99f), SpecSource::Lookup, 4));

  SpeculationController capped(SpeculationMode::Auto, 2, 3);  // max_mtp = 2, max_lookup = 3
  for (int i = 0; i < 40; ++i) capped.record_mtp_outcome(2, 2);
  CHECK(is(capped.decide(3, 0, 0.0f), SpecSource::Mtp, 2));
  CHECK(is(capped.decide(0, 8, 0.99f), SpecSource::Lookup, 3));

  CHECK(fngine::engine::parse_speculation_mode("auto") == SpeculationMode::Auto);
  CHECK(fngine::engine::parse_speculation_mode("lookup") == SpeculationMode::Lookup);
  CHECK(!fngine::engine::parse_speculation_mode("tree").has_value());
}

void test_acceptance() {
  const double first = 1.0 / (SpeculationController::kStartWeight + 1.0);  // first observation
  SpeculationController c(SpeculationMode::Auto);
  // Offered 3, accepted 1: a1 rises, a2 (the rejection) falls, a3 was not observed.
  c.record_mtp_outcome(3, 1);
  CHECK(near(c.mtp_acceptance(1), 0.7 + first * 0.3));
  CHECK(near(c.mtp_acceptance(2), 0.65 - first * 0.65));
  CHECK(near(c.mtp_acceptance(3), 0.65));
  // First draft rejected -> only a1 updates.
  const double a2 = c.mtp_acceptance(2);
  c.record_mtp_outcome(3, 0);
  CHECK(c.mtp_acceptance(1) < 0.7 + first * 0.3);
  CHECK(near(c.mtp_acceptance(2), a2));
  c.record_mtp_outcome(3, 3);
  CHECK(near(c.mtp_acceptance(3), 0.65 + first * 0.35));

  // The estimates follow the conditional frequencies: d1 accepted 3 of 4 times, d2 a third of
  // those, d3 never.
  SpeculationController f(SpeculationMode::Auto);
  for (int i = 0; i < 4000; ++i) f.record_mtp_outcome(3, i % 4 == 0 ? 0 : (i % 2 ? 1 : 2));
  CHECK(std::fabs(f.mtp_acceptance(1) - 0.75) < 0.1);
  CHECK(std::fabs(f.mtp_acceptance(2) - 1.0 / 3.0) < 0.1);
  CHECK(f.mtp_acceptance(3) < 0.05);

  // An estimate that turned the controller away from a position is observed by probes only.
  // Once the position went unoffered for long, a few observations move it; the same
  // observations barely move an estimate that is offered every round.
  SpeculationController stale(SpeculationMode::Mtp), fresh(SpeculationMode::Mtp);
  for (int i = 0; i < 400; ++i) {
    stale.record_mtp_outcome(3, 1);  // a1 -> 1, a2 -> 0
    fresh.record_mtp_outcome(3, 1);
  }
  CHECK(stale.mtp_acceptance(2) < 0.01);
  CHECK(is(stale.decide(3, 0, 0.0f), SpecSource::Mtp, 1));
  for (int i = 0; i < 300; ++i) stale.record_mtp_outcome(1, 1);  // the second draft is not offered
  CHECK(stale.mtp_acceptance(2) < 0.01);
  for (int i = 0; i < 4; ++i) {
    stale.record_mtp_outcome(2, 2);
    fresh.record_mtp_outcome(2, 2);
  }
  CHECK(stale.mtp_acceptance(2) > 0.4);
  CHECK(stale.decide(3, 0, 0.0f).k >= 2);
  CHECK(fresh.mtp_acceptance(2) < 0.1);
  CHECK(is(fresh.decide(3, 0, 0.0f), SpecSource::Mtp, 1));
  // The same for the first draft after plain decode rounds.
  SpeculationController plain(SpeculationMode::Mtp);
  for (int i = 0; i < 400; ++i) plain.record_mtp_outcome(3, 0);  // a1 -> 0
  CHECK(is(plain.decide(3, 0, 0.0f), SpecSource::None, 0));
  for (int i = 0; i < 300; ++i) plain.record_plain_round();
  plain.record_mtp_outcome(1, 1);
  plain.record_mtp_outcome(1, 1);
  CHECK(plain.decide(3, 0, 0.0f).k >= 1);

  const double alpha = SpeculationController::kLookupAlpha;

  // Lookup window 4: offered 5, accepted 2 -> two rises and one fall; other windows stay .8.
  c.record_lookup_outcome(4, 5, 2);
  const double w4 = ((0.8 + alpha * 0.2) + alpha * (1.0 - (0.8 + alpha * 0.2))) * (1.0 - alpha);
  CHECK(near(c.lookup_acceptance(4), w4));
  CHECK(near(c.lookup_acceptance(8), 0.8));
  CHECK(near(c.lookup_acceptance(2), 0.8));
  CHECK(near(c.lookup_acceptance(0), w4));  // unknown window -> lowest EMA

  // The window EMA drives decide() when no PromptLookup probability is given; a given
  // probability overrides it. Prior model: one draft pays above p = 2 / 13.
  SpeculationController d(SpeculationMode::Lookup);
  for (int i = 0; i < 200; ++i) d.record_lookup_outcome(2, 1, 0);  // window 2 -> ~0
  CHECK(is(d.decide(0, 8, 0.0f, 2), SpecSource::None, 0));
  CHECK(is(d.decide(0, 8, 0.0f, 8), SpecSource::Lookup, 5));       // window 8 still .8
  CHECK(is(d.decide(0, 8, 0.1f, 8), SpecSource::None, 0));
  CHECK(is(d.decide(0, 8, std::nanf(""), 8), SpecSource::Lookup, 5));

  // Stats.
  SpeculationController s(SpeculationMode::Auto);
  s.decide(0, 0, 0.0f);
  s.record_plain_round();
  s.record_mtp_outcome(3, 2);
  s.record_mtp_outcome(3, 3);
  s.record_lookup_outcome(8, 8, 8);
  CHECK(s.stats(SpecSource::None).decisions == 1 && s.stats(SpecSource::None).rounds == 1);
  CHECK(s.stats(SpecSource::Mtp).rounds == 2 && s.stats(SpecSource::Mtp).drafts_accepted == 5);
  CHECK(near(s.stats(SpecSource::Mtp).tokens_per_round(), 3.5));
  CHECK(near(s.stats(SpecSource::Mtp).acceptance(), 5.0 / 6.0));
  CHECK(near(s.stats(SpecSource::Lookup).tokens_per_round(), 9.0));
  CHECK(near(s.tokens_per_round(), (1.0 + 3.0 + 4.0 + 9.0) / 4.0));
}

}  // namespace

int main() {
  test_priors();
  test_level();
  test_cpu_work();
  test_learning();
  test_probes();
  test_latency_shapes();
  test_ties();
  test_pinned();
  test_acceptance();
  if (g_failures) {
    std::printf("%d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("PASS speculation\n");
  return 0;
}
