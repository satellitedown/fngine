#include "cpu/expert_pool.h"

#include "core/cpu_topology.h"
#include "cpu/cpu_isa.h"

#include <immintrin.h>
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <stdexcept>

namespace fngine::cpu {
namespace {

constexpr int kHidden = dims::kHidden;          // 2560
constexpr int kInter = dims::kExpertInter;      // 640
constexpr int kExperts = dims::kExperts;        // 512
constexpr int kTopK = dims::kTopK;              // 10
constexpr int kLayers = kExchangeLayers;        // 48
// Phase A task: (expert, 32 intermediate rows = one Q8_0 block of h, <= 128 tokens).
constexpr int kChunkA = 32;
constexpr int kChunksA = kInter / kChunkA;      // 20
constexpr int kTileA = 4;                       // gate rows (+ the same up rows) per decode
constexpr int kTilesA = kChunkA / kTileA;       // 8
constexpr int kSplitA = 128;                    // max tokens per Phase A task
// Phase B task: 32 output rows over every cold expert of the layer.
constexpr int kChunkB = 32;
constexpr int kChunksB = kHidden / kChunkB;     // 80
constexpr int kTileB = 8;                       // down rows per decode
static_assert(kInter % kChunkA == 0 && kChunkA == QK8_0 && kChunkA % kTileA == 0);
static_assert(kHidden % kChunkB == 0 && kChunkB % kTileB == 0);

uint64_t now_ns() {
  return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                      std::chrono::steady_clock::now().time_since_epoch())
                      .count());
}

void futex_wait(std::atomic<uint32_t>* addr, uint32_t expected) {
  syscall(SYS_futex, reinterpret_cast<uint32_t*>(addr), FUTEX_WAIT_PRIVATE, expected, nullptr,
          nullptr, 0);
}
void futex_wake_all(std::atomic<uint32_t>* addr) {
  syscall(SYS_futex, reinterpret_cast<uint32_t*>(addr), FUTEX_WAKE_PRIVATE, INT_MAX, nullptr,
          nullptr, 0);
}
void short_sleep() {
  timespec ts{0, 20'000};
  nanosleep(&ts, nullptr);
}

struct AlignedBuf {
  uint8_t* p = nullptr;
  explicit AlignedBuf(std::size_t bytes) {
    const std::size_t n = std::max<std::size_t>(64, (bytes + 63) & ~std::size_t{63});
    p = static_cast<uint8_t*>(std::aligned_alloc(64, n));
    if (!p) throw std::bad_alloc();
    std::memset(p, 0, n);
  }
  ~AlignedBuf() { std::free(p); }
  AlignedBuf(const AlignedBuf&) = delete;
  AlignedBuf& operator=(const AlignedBuf&) = delete;
  float* f() const { return reinterpret_cast<float*>(p); }
};

struct Group {
  int32_t expert;
  int32_t begin, end;  // pair range
  const uint8_t* blob;
};

struct TaskA {
  int32_t group, chunk, p0, p1;  // p0/p1: pair offsets inside the group
};

struct LayerInfo {
  WeightFormat gu_format = WeightFormat::IQ2_S, down_format = WeightFormat::IQ4_NL;
  ActFormat gu_act = ActFormat::Q8_K;
  std::size_t gu_row_bytes = 0, down_row_bytes = 0;  // GGUF rows
  std::size_t gu_dec_bytes = 0;                      // decoded gate/up row
  std::size_t act_bytes = 0;                         // packed x row
  std::size_t gate_offset = 0, up_offset = 0, down_offset = 0;
};

// Published input of one layer, independent of the mode.
struct LayerView {
  int rows = 0;
  int n_cold = 0;
  const int32_t* count = nullptr;   // [rows]
  const int32_t* expert = nullptr;  // [rows][kTopK]
  const float* weight = nullptr;    // [rows][kTopK]
  const uint16_t* x = nullptr;      // [rows][kHidden] bf16 bits
  float* y = nullptr;               // [rows][kHidden]
};

struct TimingCell {
  std::atomic<uint64_t> steps{0}, rows{0}, pairs{0}, experts{0}, wait_ns{0}, compute_ns{0};
};

inline void add_relaxed(std::atomic<uint64_t>& a, uint64_t v) {
  a.fetch_add(v, std::memory_order_relaxed);
}

}  // namespace

struct ExpertPool::Impl {
  struct Worker {
    int idx = 0;
    int cpu = -1;
    int sense = 0;
    std::unique_ptr<AlignedBuf> dec;   // decoded weight rows of the current tile
    std::unique_ptr<AlignedBuf> guf;   // [kTilesA][2 * kTileA][kSplitA] gate/up outputs
    std::unique_ptr<AlignedBuf> outb;  // [kTileB][max_rows] down outputs
    std::unique_ptr<AlignedBuf> yb;    // [max_rows][kChunkB] Phase B accumulator
    std::thread th;
  };

  PoolConfig cfg;
  int n = 0;
  int max_rows = 0;
  int max_pairs = 0;
  std::size_t h_row_bytes = 0;  // packed Q8_0 row of h (640)
  std::size_t step_h_stride = 0;  // Q8 decode or natural BF16 prefill
  std::array<LayerInfo, kLayers> info{};
  std::vector<std::unique_ptr<Worker>> workers;

  // ---- shared state of the layer being processed (one at a time, see `exits`)
  alignas(64) std::atomic<int> q_next{0};
  alignas(64) std::atomic<int> a_next{0};
  alignas(64) std::atomic<int> b_next{0};
  alignas(64) std::atomic<int> finish{0};
  // Workers that left a processed layer (summed over the step). The leader rewrites the shared
  // lists of processed layer k only once exits == n * k, i.e. nobody still reads layer k - 1.
  alignas(64) std::atomic<uint64_t> exits{0};
  alignas(64) int n_rows = 0;
  int n_pairs = 0;
  int n_groups = 0;
  int n_tasks = 0;
  uint64_t t_ready = 0;
  std::vector<Group> groups;       // ascending expert id (Phase B accumulation order)
  std::vector<TaskA> tasks;        // Phase A, largest experts first
  std::vector<int32_t> order;      // group indices by descending token count
  std::vector<int32_t> pair_row;
  std::vector<float> pair_w;
  std::vector<const void*> pair_act;  // token activation row, BF16 prefill or packed Q8 decode
  std::vector<const void*> pair_h;    // intermediate row, format follows the step mode
  std::unique_ptr<AlignedBuf> act;    // decode-only quantized inputs; prefill reads published BF16
  std::unique_ptr<AlignedBuf> hbuf;   // max(decode Q8 intermediates, prefill BF16 intermediates)
  std::array<int32_t, kExperts> counts{};
  std::array<int32_t, kExperts> cursor{};

  alignas(64) std::atomic<uint32_t> gen{0};
  alignas(64) std::atomic<int> active{0};
  alignas(64) std::atomic<int> parked{0};
  alignas(64) std::atomic<int> bar_count{0};
  alignas(64) std::atomic<int> bar_sense{0};
  alignas(64) std::atomic<bool> stop{false};
  uint32_t step_seq = 0;
  PoolMode step_mode = PoolMode::Decode;
  uint32_t last_seq = 0;

  std::array<TimingCell, kLayers> timing;
  std::atomic<bool> failed{false};
  mutable std::mutex fail_mu;
  std::string fail_msg;

  uint64_t park_ns() const { return uint64_t(cfg.park_after_ms) * 1'000'000ull; }

  void fail(const std::string& msg) {
    std::lock_guard<std::mutex> lk(fail_mu);
    if (!failed.load(std::memory_order_relaxed)) fail_msg = msg;
    failed.store(true, std::memory_order_release);
  }

  bool aborted() const {
    return stop.load(std::memory_order_relaxed) ||
           host_load_acquire(cfg.decode->control.abort.value) != 0;
  }

  void reset_counters() {
    q_next.store(0, std::memory_order_relaxed);
    a_next.store(0, std::memory_order_relaxed);
    b_next.store(0, std::memory_order_relaxed);
    finish.store(0, std::memory_order_relaxed);
  }

  // 1: ready, 0: stale (the GPU already moved past this step), -1: abort.
  int wait_ready(int l, uint32_t seq) const {
    const uint32_t& cell = cfg.decode->control.ready[l].value;
    uint64_t start = 0;
    unsigned spins = 0;
    for (;;) {
      const uint32_t v = host_load_acquire(cell);
      if (v == seq) return 1;
      if (int32_t(v - seq) > 0) return 0;
      _mm_pause();
      if ((++spins & 1023u) == 0) {
        if (aborted()) return -1;
        const uint64_t t = now_ns();
        if (start == 0) start = t;
        else if (t - start > park_ns()) short_sleep();  // long GPU phase: poll gently
      }
    }
  }

  bool barrier(Worker& w) {
    const int s = (w.sense ^= 1);
    if (bar_count.fetch_add(1, std::memory_order_acq_rel) == n - 1) {
      bar_count.store(0, std::memory_order_relaxed);
      bar_sense.store(s, std::memory_order_release);
      return true;
    }
    unsigned spins = 0;
    while (bar_sense.load(std::memory_order_acquire) != s) {
      _mm_pause();
      if ((++spins & 1023u) == 0 && aborted()) return false;
    }
    return true;
  }

  bool wait_exits(uint64_t target) const {
    unsigned spins = 0;
    while (exits.load(std::memory_order_acquire) < target) {
      _mm_pause();
      if ((++spins & 1023u) == 0 && aborted()) return false;
    }
    return true;
  }

  LayerView view(int l, PoolMode mode) const {
    LayerView v;
    if (mode == PoolMode::Decode) {
      const MoeDecodePublish& p = cfg.decode->publish[l];
      v.rows = p.tokens;
      v.n_cold = p.n_cold;
      v.count = p.cold_count;
      v.expert = &p.cold_expert[0][0];
      v.weight = &p.cold_weight[0][0];
      v.x = reinterpret_cast<const uint16_t*>(&p.x[0][0]);
      v.y = &cfg.decode->result[l].y[0][0];
    } else {
      const PrefillExchangeView& p = cfg.prefill;
      v.rows = p.header[1];
      v.n_cold = p.header[2];
      v.count = p.cold_count;
      v.expert = p.cold_expert;
      v.weight = p.cold_weight;
      v.x = reinterpret_cast<const uint16_t*>(p.x);
      v.y = p.y;
    }
    return v;
  }

  int clamp_rows(int rows) const { return std::clamp(rows, 0, max_rows); }

  // Leader only: group cold pairs by expert (ascending id; rows ascending inside an expert) and
  // build the Phase A task list (largest experts first, split into <= kSplitA token ranges).
  void group(int l, const LayerView& v) {
    const int rows = clamp_rows(v.rows);
    if (rows != v.rows)
      fail("layer " + std::to_string(l) + ": " + std::to_string(v.rows) +
           " published rows exceed the exchange capacity " + std::to_string(max_rows));
    n_rows = rows;
    counts.fill(0);
    const void* const* slots = cfg.host_slot_of + std::size_t(l) * kExperts;
    for (int r = 0; r < rows; ++r) {
      const int c = std::clamp(v.count[r], 0, kTopK);
      for (int j = 0; j < c; ++j) {
        const int e = v.expert[r * kTopK + j];
        if (e < 0 || e >= kExperts) {
          fail("layer " + std::to_string(l) + ": cold expert id " + std::to_string(e) + " out of range");
          continue;
        }
        ++counts[e];
      }
    }
    int off = 0, ng = 0;
    for (int e = 0; e < kExperts; ++e) {
      if (counts[e] == 0) continue;
      const auto* blob = static_cast<const uint8_t*>(slots[e]);
      if (!blob) {
        fail("layer " + std::to_string(l) + ": cold expert " + std::to_string(e) + " has no host blob");
        counts[e] = 0;
        continue;
      }
      groups[ng++] = Group{e, off, off + counts[e], blob};
      cursor[e] = off;
      off += counts[e];
    }
    const LayerInfo& li = info[l];
    for (int r = 0; r < rows; ++r) {
      const int c = std::clamp(v.count[r], 0, kTopK);
      for (int j = 0; j < c; ++j) {
        const int e = v.expert[r * kTopK + j];
        if (e < 0 || e >= kExperts || counts[e] == 0) continue;
        const int p = cursor[e]++;
        pair_row[p] = r;
        pair_w[p] = v.weight[r * kTopK + j];
        pair_act[p] = step_mode == PoolMode::Prefill
                          ? static_cast<const void*>(v.x + std::size_t(r) * kHidden)
                          : act->p + std::size_t(r) * li.act_bytes;
        pair_h[p] = hbuf->p + std::size_t(p) * step_h_stride;
      }
    }
    n_pairs = off;
    n_groups = ng;
    order.resize(std::size_t(ng));
    for (int i = 0; i < ng; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
      const int ta = groups[a].end - groups[a].begin, tb = groups[b].end - groups[b].begin;
      return ta != tb ? ta > tb : a < b;
    });
    int nt = 0;
    for (const int gi : order) {
      const int te = groups[gi].end - groups[gi].begin;
      for (int p0 = 0; p0 < te; p0 += kSplitA)
        for (int c = 0; c < kChunksA; ++c) tasks[nt++] = TaskA{gi, c, p0, std::min(te, p0 + kSplitA)};
    }
    n_tasks = nt;
  }

  // Gate/up -> SwiGLU: BF16 h for prefill, packed Q8_0 half group for decode.
  void phase_a(Worker& w, const LayerInfo& li, const TaskA& tk) const {
    const Group& g = groups[tk.group];
    const int nt = tk.p1 - tk.p0;
    const void* const* acts = pair_act.data() + g.begin + tk.p0;
    const uint8_t* gate = g.blob + li.gate_offset;
    const uint8_t* up = g.blob + li.up_offset;
    uint8_t* dec = w.dec->p;
    float* guf = w.guf->f();
    for (int tile = 0; tile < kTilesA; ++tile) {
      const std::size_t row = std::size_t(tk.chunk * kChunkA + tile * kTileA);
      float* output = guf + std::size_t(tile) * 2 * kTileA * kSplitA;
      if (step_mode == PoolMode::Prefill) {
        auto* bf16 = reinterpret_cast<uint16_t*>(dec);
        decode_rows_bf16(li.gu_format, gate + row * li.gu_row_bytes, li.gu_row_bytes,
                         kTileA, kHidden, bf16);
        decode_rows_bf16(li.gu_format, up + row * li.gu_row_bytes, li.gu_row_bytes,
                         kTileA, kHidden, bf16 + kTileA * kHidden);
        gemm_bf16(bf16, 2 * kTileA, kHidden, acts, nt, output, kSplitA);
      } else {
        decode_rows(li.gu_format, gate + row * li.gu_row_bytes, li.gu_row_bytes, kTileA, kHidden, dec);
        decode_rows(li.gu_format, up + row * li.gu_row_bytes, li.gu_row_bytes, kTileA, kHidden,
                    dec + kTileA * li.gu_dec_bytes);
        gemm_decoded(li.gu_format, dec, 2 * kTileA, kHidden, acts, nt, output, kSplitA);
      }
    }
    alignas(64) float gv[kChunkA], uv[kChunkA], hv[kChunkA];
    for (int t = 0; t < nt; ++t) {
      for (int r = 0; r < kChunkA; ++r) {
        const float* tilep = guf + std::size_t(r / kTileA) * 2 * kTileA * kSplitA;
        gv[r] = tilep[std::size_t(r % kTileA) * kSplitA + t];
        uv[r] = tilep[std::size_t(kTileA + r % kTileA) * kSplitA + t];
      }
      silu_mul(gv, uv, hv, kChunkA);
      auto* h = hbuf->p + std::size_t(g.begin + tk.p0 + t) * step_h_stride;
      if (step_mode == PoolMode::Prefill) {
#if FNGINE_CPU_BACKEND_AVX512
        _mm512_storeu_si512(reinterpret_cast<uint16_t*>(h) + tk.chunk * kChunkA,
                            (__m512i)_mm512_cvtne2ps_pbh(_mm512_load_ps(hv + 16),
                                                       _mm512_load_ps(hv)));
#else
        // AVX-NE-CONVERT has a 256-bit convert. Two stores cover the 32 BF16 lanes.
        auto cvt2ps_to_pbh = [](__m256 a_hi, __m256 b_lo) -> __m256i {
          __m128i lo = (__m128i)_mm256_cvtneps_avx_pbh(b_lo);
          __m128i hi = (__m128i)_mm256_cvtneps_avx_pbh(a_hi);
          return _mm256_set_m128i(hi, lo);
        };
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(reinterpret_cast<uint16_t*>(h) + tk.chunk * kChunkA),
                            cvt2ps_to_pbh(_mm256_load_ps(hv + 8), _mm256_load_ps(hv)));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(reinterpret_cast<uint16_t*>(h) + tk.chunk * kChunkA + 16),
                            cvt2ps_to_pbh(_mm256_load_ps(hv + 24), _mm256_load_ps(hv + 16)));
#endif
      } else {
        quantize_pack_q8_0_half(hv, tk.chunk / 2, tk.chunk % 2, h);
      }
    }
  }

  // 32 output rows over every cold expert (ascending id): y[row][n] = sum w * down_n . h.
  void phase_b(Worker& w, const LayerInfo& li, const LayerView& v, int task) const {
    const int r0 = task * kChunkB;
    const int rows = n_rows;
    float* yb = w.yb->f();
    float* outb = w.outb->f();
    std::fill(yb, yb + std::size_t(rows) * kChunkB, 0.0f);
    for (int gi = 0; gi < n_groups; ++gi) {
      const Group& g = groups[gi];
      const int te = g.end - g.begin;
      const void* const* hs = pair_h.data() + g.begin;
      const int32_t* prow = pair_row.data() + g.begin;
      const float* pw = pair_w.data() + g.begin;
      for (int tile = 0; tile < kChunkB / kTileB; ++tile) {
        const std::size_t row = std::size_t(r0 + tile * kTileB);
        const auto* down = g.blob + li.down_offset + row * li.down_row_bytes;
        if (step_mode == PoolMode::Prefill) {
          auto* bf16 = reinterpret_cast<uint16_t*>(w.dec->p);
          decode_rows_bf16(li.down_format, down, li.down_row_bytes, kTileB, kInter, bf16);
          gemm_bf16(bf16, kTileB, kInter, hs, te, outb, std::size_t(te));
        } else {
          decode_rows(li.down_format, down, li.down_row_bytes, kTileB, kInter, w.dec->p);
          gemm_decoded(li.down_format, w.dec->p, kTileB, kInter, hs, te, outb, std::size_t(te));
        }
        for (int i = 0; i < kTileB; ++i) {
          const float* o = outb + std::size_t(i) * te;
          const int col = tile * kTileB + i;
          for (int p = 0; p < te; ++p) yb[std::size_t(prow[p]) * kChunkB + col] += pw[p] * o[p];
        }
      }
    }
    for (int row = 0; row < rows; ++row)
      std::memcpy(v.y + std::size_t(row) * kHidden + r0, yb + std::size_t(row) * kChunkB,
                  kChunkB * sizeof(float));
  }

  bool process_layer(Worker& w, int l, uint32_t seq, const LayerView& v, int processed) {
    const LayerInfo& li = info[l];
    if (w.idx == 0) {
      if (!wait_exits(uint64_t(n) * uint64_t(processed))) return false;
      t_ready = now_ns();
      group(l, v);
    }
    const int rows = clamp_rows(v.rows);
    if (step_mode == PoolMode::Decode) {
      for (int r; (r = q_next.fetch_add(1, std::memory_order_relaxed)) < rows;)
        quantize_pack_bf16(li.gu_act, v.x + std::size_t(r) * kHidden, kHidden,
                           act->p + std::size_t(r) * li.act_bytes);
    }
    if (!barrier(w)) return false;
    for (int t; (t = a_next.fetch_add(1, std::memory_order_relaxed)) < n_tasks;)
      phase_a(w, li, tasks[std::size_t(t)]);
    if (!barrier(w)) return false;
    for (int t; (t = b_next.fetch_add(1, std::memory_order_relaxed)) < kChunksB;)
      phase_b(w, li, v, t);
    finish_layer(l, seq, true);
    return true;
  }

  // Last of the n workers publishes done[l]; every worker then counts itself out.
  void finish_layer(int l, uint32_t seq, bool computed) {
    if (finish.fetch_add(1, std::memory_order_acq_rel) == n - 1) {
      if (computed) {
        TimingCell& tc = timing[l];
        add_relaxed(tc.steps, 1);
        add_relaxed(tc.rows, uint64_t(n_rows));
        add_relaxed(tc.pairs, uint64_t(n_pairs));
        add_relaxed(tc.experts, uint64_t(n_groups));
      }
      const uint64_t t0 = t_ready;
      reset_counters();
      host_store_release(cfg.decode->control.done[l].value, seq);
      if (computed) add_relaxed(timing[l].compute_ns, now_ns() - t0);
    }
    exits.fetch_add(1, std::memory_order_acq_rel);
  }

  void run_step(Worker& w, uint32_t seq, PoolMode mode) {
    w.sense = 0;
    int processed = 0;
    for (int l = 0; l < kLayers; ++l) {
      const uint64_t t0 = w.idx == 0 ? now_ns() : 0;
      const int r = wait_ready(l, seq);
      if (r < 0) return;
      if (w.idx == 0) add_relaxed(timing[l].wait_ns, now_ns() - t0);
      if (r == 0) continue;
      const LayerView v = view(l, mode);
      if (v.n_cold <= 0) {
        // Prefill: the region is shared by all layers, so the GPU waits on done[l] for every
        // layer; acknowledge once every worker has read this layer's header.
        if (mode == PoolMode::Prefill) {
          finish_layer(l, seq, false);
          ++processed;
        }
        continue;
      }
      if (!process_layer(w, l, seq, v, processed)) return;
      ++processed;
    }
  }

  void worker_main(Worker& w) {
    if (w.cpu >= 0) pin_current_thread(w.cpu);
    uint32_t seen = 0;
    for (;;) {
      uint32_t g = gen.load(std::memory_order_acquire);
      if (g == seen) {
        uint64_t start = now_ns();
        unsigned spins = 0;
        while ((g = gen.load(std::memory_order_acquire)) == seen) {
          if (stop.load(std::memory_order_relaxed)) return;
          _mm_pause();
          if ((++spins & 1023u) == 0 && now_ns() - start > park_ns()) {
            parked.fetch_add(1, std::memory_order_seq_cst);
            if (gen.load(std::memory_order_seq_cst) == seen && !stop.load()) futex_wait(&gen, seen);
            parked.fetch_sub(1, std::memory_order_seq_cst);
            start = now_ns();
          }
        }
      }
      if (stop.load(std::memory_order_acquire)) return;
      seen = g;
      run_step(w, step_seq, step_mode);
      active.fetch_sub(1, std::memory_order_acq_rel);
    }
  }
};

ExpertPool::ExpertPool(const PoolConfig& cfg) : impl_(std::make_unique<Impl>()) {
  Impl& I = *impl_;
  if (cfg.worker_cpus.empty()) throw std::invalid_argument("ExpertPool: no worker CPUs");
  if (!cfg.decode) throw std::invalid_argument("ExpertPool: decode exchange is required");
  if (!cfg.host_slot_of) throw std::invalid_argument("ExpertPool: host_slot_of is required");
  if (cfg.park_after_ms < 0) throw std::invalid_argument("ExpertPool: negative park_after_ms");
  const PrefillExchangeView& pv = cfg.prefill;
  if (pv.capacity < 0) throw std::invalid_argument("ExpertPool: negative prefill capacity");
  if (pv.capacity > 0 && (!pv.header || !pv.row_token || !pv.cold_count || !pv.cold_expert ||
                          !pv.cold_weight || !pv.x || !pv.y))
    throw std::invalid_argument("ExpertPool: incomplete prefill exchange view");
  I.cfg = cfg;
  I.n = int(cfg.worker_cpus.size());
  I.max_rows = std::max<int>(kDecodeExchangeTokens, pv.capacity);
  I.max_pairs = I.max_rows * kTopK;
  I.h_row_bytes = packed_act_bytes(ActFormat::Q8_0, kInter);
  std::size_t max_act = 0, max_dec = 0;
  for (int l = 0; l < kLayers; ++l) {
    const ExpertLayerDesc& d = cfg.layers[l];
    const FormatInfo gu = format_info(d.gu_format);
    const FormatInfo dn = format_info(d.down_format);
    if (kHidden % gu.block_elems != 0)
      throw std::invalid_argument("ExpertPool: gate/up format incompatible with K=2560");
    if (dn.act != ActFormat::Q8_0 || kInter % dn.block_elems != 0)
      throw std::invalid_argument("ExpertPool: down format must consume Q8_0 activations");
    LayerInfo& li = I.info[l];
    li.gu_format = d.gu_format;
    li.down_format = d.down_format;
    li.gu_act = gu.act;
    li.gu_row_bytes = weight_row_bytes(d.gu_format, kHidden);
    li.down_row_bytes = weight_row_bytes(d.down_format, kInter);
    li.gu_dec_bytes = decoded_row_bytes(d.gu_format, kHidden);
    li.act_bytes = packed_act_bytes(gu.act, kHidden);
    li.gate_offset = d.gate_offset;
    li.up_offset = d.up_offset;
    li.down_offset = d.down_offset;
    max_act = std::max(max_act, li.act_bytes);
    max_dec = std::max({max_dec, 2 * kTileA * li.gu_dec_bytes,
                        kTileB * decoded_row_bytes(d.down_format, kInter)});
  }
  I.groups.resize(kExperts);
  I.tasks.resize(std::size_t(kChunksA) * std::size_t(kExperts + I.max_pairs / kSplitA + 1));
  I.pair_row.resize(std::size_t(I.max_pairs));
  I.pair_w.resize(std::size_t(I.max_pairs));
  I.pair_act.resize(std::size_t(I.max_pairs));
  I.pair_h.resize(std::size_t(I.max_pairs));
  I.act = std::make_unique<AlignedBuf>(std::size_t(kDecodeExchangeTokens) * max_act);
  I.hbuf = std::make_unique<AlignedBuf>(std::max(
      std::size_t(kDecodeExchangeTokens) * kTopK * I.h_row_bytes,
      std::size_t(pv.capacity) * kTopK * kInter * sizeof(uint16_t)));
  n_workers_ = I.n;
  for (int i = 0; i < I.n; ++i) {
    auto w = std::make_unique<Impl::Worker>();
    w->idx = i;
    w->cpu = cfg.worker_cpus[i];
    w->dec = std::make_unique<AlignedBuf>(
        std::max(max_dec, std::size_t(2 * kTileA) * kHidden * sizeof(uint16_t)));
    w->guf = std::make_unique<AlignedBuf>(std::size_t(kTilesA) * 2 * kTileA * kSplitA * sizeof(float));
    w->outb = std::make_unique<AlignedBuf>(std::size_t(kTileB) * I.max_rows * sizeof(float));
    w->yb = std::make_unique<AlignedBuf>(std::size_t(I.max_rows) * kChunkB * sizeof(float));
    I.workers.push_back(std::move(w));
  }
  for (auto& w : I.workers) {
    Impl::Worker* wp = w.get();
    wp->th = std::thread([&I, wp] { I.worker_main(*wp); });
  }
}

ExpertPool::~ExpertPool() {
  Impl& I = *impl_;
  I.stop.store(true, std::memory_order_seq_cst);
  I.gen.fetch_add(1, std::memory_order_seq_cst);
  futex_wake_all(&I.gen);
  for (auto& w : I.workers)
    if (w->th.joinable()) w->th.join();
}

void ExpertPool::begin(uint32_t seq, PoolMode mode) {
  Impl& I = *impl_;
  if (seq == 0) throw std::invalid_argument("ExpertPool::begin: seq must be nonzero");
  if (seq == I.last_seq) throw std::invalid_argument("ExpertPool::begin: seq reused");
  if (mode == PoolMode::Prefill && I.cfg.prefill.capacity == 0)
    throw std::invalid_argument("ExpertPool::begin: no prefill exchange configured");
  wait_idle();
  I.reset_counters();
  I.exits.store(0, std::memory_order_relaxed);
  I.bar_count.store(0, std::memory_order_relaxed);
  I.bar_sense.store(0, std::memory_order_relaxed);
  I.step_seq = seq;
  I.step_mode = mode;
  I.step_h_stride = mode == PoolMode::Prefill ? kInter * sizeof(uint16_t) : I.h_row_bytes;
  I.last_seq = seq;
  I.active.store(I.n, std::memory_order_relaxed);
  I.gen.fetch_add(1, std::memory_order_seq_cst);
  if (I.parked.load(std::memory_order_seq_cst) > 0) futex_wake_all(&I.gen);
}

void ExpertPool::wait_idle() {
  Impl& I = *impl_;
  unsigned spins = 0;
  while (I.active.load(std::memory_order_acquire) != 0) {
    _mm_pause();
    if ((++spins & 4095u) == 0) std::this_thread::yield();
  }
}

bool ExpertPool::failed() const { return impl_->failed.load(std::memory_order_acquire); }

std::string ExpertPool::failure() const {
  std::lock_guard<std::mutex> lk(impl_->fail_mu);
  return impl_->fail_msg;
}

std::array<LayerTiming, kExchangeLayers> ExpertPool::timings() const {
  std::array<LayerTiming, kExchangeLayers> out{};
  for (int l = 0; l < kLayers; ++l) {
    const TimingCell& c = impl_->timing[l];
    out[l].steps = c.steps.load(std::memory_order_relaxed);
    out[l].rows = c.rows.load(std::memory_order_relaxed);
    out[l].pairs = c.pairs.load(std::memory_order_relaxed);
    out[l].experts = c.experts.load(std::memory_order_relaxed);
    out[l].wait_ns = c.wait_ns.load(std::memory_order_relaxed);
    out[l].compute_ns = c.compute_ns.load(std::memory_order_relaxed);
  }
  return out;
}

void ExpertPool::reset_timings() {
  for (TimingCell& c : impl_->timing) {
    c.steps.store(0, std::memory_order_relaxed);
    c.rows.store(0, std::memory_order_relaxed);
    c.pairs.store(0, std::memory_order_relaxed);
    c.experts.store(0, std::memory_order_relaxed);
    c.wait_ns.store(0, std::memory_order_relaxed);
    c.compute_ns.store(0, std::memory_order_relaxed);
  }
}

}  // namespace fngine::cpu
