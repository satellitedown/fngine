// fngine::Engine: public API over ModelRuntime + StepRunner + text::Frontend.
//
// Threading: one engine thread (pinned to CpuTopology::engine_cpu) constructs and owns the
// runtime and performs every runtime call. submit() enqueues into a FIFO (max_pending_requests
// waiting, one active); GenerationHandle::wait() pumps the request's events (start, progress,
// timing, deltas) to the caller's OutputSink on the caller thread. Cancellation (handle
// destruction, CancellationView, sink failure) is honoured between prefill chunks and decode
// steps. Prompt rendering/tokenization runs on the caller thread (the frontend is immutable).
//
// Thinking-budget control: when the output session requests the forced close suffix, the
// pending anchor and all control tokens but the last are processed by plain 1-token steps whose
// sampled outputs are discarded; the last control token becomes the next anchor. Discarded
// samples still count in the penalty token counts and leave an MTP KV entry built with the
// discarded token at those positions (draft quality only; verification stays exact); MTP drafts
// are not offered for the round after the control suffix.
#include "fngine/engine.h"

#include "core/cpu_topology.h"
#include "core/exchange.h"
#include "core/model_dims.h"
#include "engine/session.h"
#include "engine/speculation.h"
#include "model/runtime.h"
#include "model/step_graph.h"
#include "text/frontend.h"
#include "text/output_session.h"
#include "text/prompt_lookup.h"

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <string_view>
#include <deque>
#include <exception>
#include <future>
#include <limits>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>
#include <utility>

namespace fngine {
namespace {

using Clock = std::chrono::steady_clock;

double seconds_between(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

std::uint64_t ns_between(Clock::time_point a, Clock::time_point b) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count());
}

constexpr int kLayers           = 48;
constexpr int kRoutedPerToken   = 10;
constexpr int kMaxLookupDrafts  = 8;
constexpr int kMaxPositionStats = 8;

struct Event {
    enum class Kind : std::uint8_t { Start, Progress, Timing, Delta } kind = Kind::Delta;
    GenerationStart start;
    PromptProgress progress;
    GenerationTimingObservation timing;
    OutputDelta delta;
};

// Thrown from the prefill hook to abort a cancelled prompt at a chunk boundary.
struct PrefillCancelled {
    std::int32_t position = 0;
};

engine::SpeculationMode to_mode(SpeculativeBackend backend) {
    switch (backend) {
    case SpeculativeBackend::None:
        return engine::SpeculationMode::None;
    case SpeculativeBackend::Mtp:
        return engine::SpeculationMode::Mtp;
    case SpeculativeBackend::Lookup:
        return engine::SpeculationMode::Lookup;
    case SpeculativeBackend::Auto:
        return engine::SpeculationMode::Auto;
    }
    return engine::SpeculationMode::Auto;
}

} // namespace

// ---------------------------------------------------------------------------------------------
// Request state shared by the engine thread and the handle.

struct RequestState {
    text::PreparedText prompt;
    PromptSummary summary;
    PromptPreparationStats preparation;
    RequestOptions options;
    ResolvedSamplingParameters sampling;
    OutputConsumerMode mode = OutputConsumerMode::Aggregate;
    GenerationObservationOptions observation;
    Clock::time_point submitted;
    Clock::time_point deadline = Clock::time_point::max();
    std::atomic<bool> cancelled{false};

    std::mutex mu;
    std::condition_variable cv;
    std::deque<Event> events;
    bool done = false;
    GenerationResult result;
    std::exception_ptr error;

    [[nodiscard]] bool streaming() const noexcept { return mode == OutputConsumerMode::Streaming; }

    void push(Event event) {
        {
            std::lock_guard lock(mu);
            events.push_back(std::move(event));
        }
        cv.notify_all();
    }

    void finish(GenerationResult value) {
        {
            std::lock_guard lock(mu);
            result = std::move(value);
            done   = true;
        }
        cv.notify_all();
    }

    void fail(std::exception_ptr exception) {
        {
            std::lock_guard lock(mu);
            error = std::move(exception);
            done  = true;
        }
        cv.notify_all();
    }
};

// ---------------------------------------------------------------------------------------------
// PreparedPrompt

class PreparedPrompt::Impl {
public:
    text::PreparedText text;
    PromptSummary summary;
};

PreparedPrompt::PreparedPrompt() noexcept = default;
PreparedPrompt::~PreparedPrompt()         = default;
PreparedPrompt::PreparedPrompt(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
PreparedPrompt::PreparedPrompt(PreparedPrompt&&) noexcept            = default;
PreparedPrompt& PreparedPrompt::operator=(PreparedPrompt&&) noexcept = default;

const PromptSummary& PreparedPrompt::summary() const noexcept {
    static const PromptSummary empty{};
    return impl_ ? impl_->summary : empty;
}

const PromptPreparationStats& PreparedPrompt::preparation_stats() const noexcept {
    static const PromptPreparationStats empty{};
    return impl_ ? impl_->text.stats : empty;
}

PreparedPrompt::operator bool() const noexcept { return impl_ != nullptr; }

// ---------------------------------------------------------------------------------------------
// Engine::Impl

class Engine::Impl {
public:
    explicit Impl(EngineOptions engine_options);
    ~Impl();

    Impl(const Impl&)            = delete;
    Impl& operator=(const Impl&) = delete;

    void enqueue(const std::shared_ptr<RequestState>& request);

    EngineOptions options;
    text::Frontend frontend;

    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::shared_ptr<RequestState>> queue;
    bool stopping  = false;
    bool active    = false;
    bool available = false;
    std::shared_ptr<RequestState> current; // the active request (cancelled on shutdown)
    RuntimeStats stats;
    LoadSummary load;
    MemorySummary memory;
    std::mt19937_64 seed_rng{std::random_device{}()};

private:
    void run(std::promise<void>* ready);
    void start_runtime();
    void process(RequestState& request);
    void publish_stats();
    // Decode accounting of one decode/verify step. Returns the CPU expert tier's compute time
    // (seconds, summed over the layers).
    double account_decode_step(std::int32_t width, std::int32_t licensed);
    // Re-reads the cumulative pool counters so that decode accounting excludes prefill work.
    void sync_pool_counters();

    std::thread worker_;

    // Engine-thread state.
    std::unique_ptr<model::ModelRuntime> rt_;
    std::unique_ptr<model::StepRunner> runner_;
    std::unique_ptr<engine::SpeculationController> controller_;
    std::unique_ptr<engine::SessionStore> sessions_;
    text::PromptLookup lookup_;
    std::vector<std::int32_t> seq_; // tokens of the runtime state, incl. the pending anchor
    RuntimeStats local_;            // engine-thread copy, published under mu
    std::array<std::uint64_t, kLayers> pool_pairs_{};   // cumulative pool counters at last read
    std::array<std::uint64_t, kLayers> pool_compute_{};
    std::array<double, kLayers> decode_wait_ns_{};       // decode-step CPU expert time per layer
    std::uint64_t decode_routed_      = 0;
    std::uint64_t decode_cold_        = 0;
    double decode_mass_               = 0.0; // routing mass of decode tokens (48 per token)
    double decode_cold_mass_          = 0.0; // of which CPU-computed
};

Engine::Impl::Impl(EngineOptions engine_options)
    : options(std::move(engine_options)),
      frontend([&] {
          if (options.pack_path.empty()) throw std::invalid_argument("EngineOptions.pack_path is empty");
          if (options.max_context == 0) throw std::invalid_argument("max_context must be positive");
          if (options.max_context >
              std::uint32_t(std::numeric_limits<std::int32_t>::max() -
                            (kernels::kQsaKvPageTokens - 1)))
              throw std::invalid_argument("max_context exceeds the page-aligned position range");
          try {
              return text::Frontend::load(options.pack_path / "tokenizer",
                                          text::FrontendOptions{options.chat_template_path,
                                                                options.max_context});
          } catch (const std::exception& error) {
              throw StartupError(std::string("cannot load the pack tokenizer: ") + error.what());
          }
      }()) {
    std::promise<void> ready;
    auto started = ready.get_future();
    worker_      = std::thread([this, &ready] { run(&ready); });
    try {
        started.get();
    } catch (...) {
        worker_.join();
        throw;
    }
}

Engine::Impl::~Impl() {
    {
        std::lock_guard lock(mu);
        stopping = true;
        if (current) current->cancelled.store(true);
    }
    cv.notify_all();
    if (worker_.joinable()) worker_.join();
}

void Engine::Impl::enqueue(const std::shared_ptr<RequestState>& request) {
    {
        std::lock_guard lock(mu);
        if (!available || stopping) {
            throw RequestError(RequestErrorKind::Unavailable, "engine is not available");
        }
        if (queue.size() >= options.max_pending_requests) {
            throw RequestError(RequestErrorKind::Overloaded, "inference request queue is full");
        }
        queue.push_back(request);
    }
    cv.notify_all();
}

void Engine::Impl::start_runtime() {
    const auto t0 = Clock::now();
    if (cudaSetDevice(options.device) != cudaSuccess) {
        throw StartupError("cannot select CUDA device " + std::to_string(options.device));
    }
    std::size_t free_before = 0, total = 0;
    cudaMemGetInfo(&free_before, &total);

    const bool mtp = options.speculative.backend == SpeculativeBackend::Auto ||
                     options.speculative.backend == SpeculativeBackend::Mtp;
    model::RuntimeOptions ro;
    ro.pack             = options.pack_path;
    ro.max_context      = static_cast<std::int32_t>(options.max_context);
    ro.prefill_chunk    = static_cast<std::int32_t>(options.prefill_chunk);
    switch (options.kv_cache) {
    case KvCacheStorage::K8V8:
        ro.kv = kernels::QsaKvFormat::K8V8;
        break;
    case KvCacheStorage::K8V4:
        ro.kv = kernels::QsaKvFormat::K8V4;
        break;
    case KvCacheStorage::BF16:
        ro.kv = kernels::QsaKvFormat::BF16;
        break;
    }
    ro.dense_bf16       = options.dense_bf16;
    ro.cpu_workers      = static_cast<int>(options.cpu_workers);
    ro.vram_reserve_mib = options.vram_reserve_mib;
    ro.ngram_cache_mib  = options.ngram_cache_mib;
    ro.mtp              = mtp;
    ro.adaptive_experts = options.adaptive_experts;
    ro.prefill_stream   = options.prefill_stream;
    ro.prefill_borrow   = options.prefill_borrow;
    ro.ablation         = options.ablation;
    // Routing telemetry persists to <pack>/routing-stats.bin; FNGINE_NO_ROUTING_STATS=1 turns the
    // file off (A/B measurements that must not change the next start's placement).
    {
        const char* off           = std::getenv("FNGINE_NO_ROUTING_STATS");
        ro.persist_routing_stats  = !(off && *off && std::string_view(off) != "0");
    }
    rt_                 = std::make_unique<model::ModelRuntime>(ro);
    runner_             = std::make_unique<model::StepRunner>(*rt_, options.use_cuda_graph, mtp);
    const int max_mtp   = options.speculative.max_mtp_drafts == 0
                              ? engine::SpeculationController::kMaxMtpDrafts
                              : static_cast<int>(std::min<std::uint32_t>(
                                    options.speculative.max_mtp_drafts,
                                    engine::SpeculationController::kMaxMtpDrafts));
    const int max_look  = options.speculative.max_lookup_drafts == 0
                              ? engine::SpeculationController::kMaxLookupDrafts
                              : static_cast<int>(std::min<std::uint32_t>(
                                    options.speculative.max_lookup_drafts,
                                    engine::SpeculationController::kMaxLookupDrafts));
    controller_ = std::make_unique<engine::SpeculationController>(
        to_mode(options.speculative.backend), max_mtp, max_look);
    sessions_ = std::make_unique<engine::SessionStore>(options.snapshots);

    const model::PlacementPlan& plan = rt_->placement().plan();
    sync_pool_counters();
    std::lock_guard lock(mu);
    load.architecture         = "qwen4_exp";
    load.model_name           = "Qwen3.8-Flash-Next";
    load.pack_path            = options.pack_path;
    load.load_seconds         = seconds_between(t0, Clock::now());
    load.host_to_device_bytes = rt_->weights().vram_bytes() + plan.vram_bytes;
    memory.device             = options.device;
    memory.max_context        = options.max_context;
    memory.kv_cache           = options.kv_cache;
    memory.device_total_bytes = total;
    memory.device_free_at_start = free_before;
    memory.dense_weight_bytes   = rt_->weights().vram_bytes();
    memory.state_bytes          = rt_->fixed_vram_bytes();
    memory.expert_pool_bytes    = plan.vram_bytes;
    memory.gpu_experts          = static_cast<std::uint32_t>(plan.gpu_experts + std::int64_t(plan.elastic.size()));
    memory.total_experts        = static_cast<std::uint32_t>(plan.gpu_experts + plan.cpu_experts);
    memory.predicted_gpu_hit_rate = plan.predicted_hit_rate;
    memory.host_expert_bytes      = plan.host_bytes;
    memory.host_pinned_bytes      = plan.host_bytes;
    memory.ngram_cache_bytes      = options.ngram_cache_mib << 20;
    memory.snapshot_slots         = options.snapshots;
    available                     = true;
}

void Engine::Impl::run(std::promise<void>* ready) {
    try {
        const CpuTopology topology = CpuTopology::detect();
        if (!pin_current_thread(topology.engine_cpu)) {
            spdlog::warn("cannot pin the engine thread to CPU {}", topology.engine_cpu);
        }
    } catch (const std::exception& error) {
        spdlog::warn("CPU topology detection failed: {}", error.what());
    }
    try {
        start_runtime();
    } catch (...) {
        runner_.reset();
        rt_.reset();
        ready->set_exception(std::current_exception());
        return;
    }
    ready->set_value();

    for (;;) {
        std::shared_ptr<RequestState> request;
        {
            std::unique_lock lock(mu);
            if (!cv.wait_for(lock, std::chrono::minutes(10),
                             [this] { return stopping || !queue.empty(); })) {
                lock.unlock();
                try {
                    rt_->collect_routing_stats();
                } catch (const std::exception& error) {
                    spdlog::warn("routing stats collection failed: {}", error.what());
                }
                continue;
            }
            if (stopping) break;
            request = std::move(queue.front());
            queue.pop_front();
            active  = true;
            current = request;
        }
        try {
            process(*request);
        } catch (...) {
            request->fail(std::current_exception());
        }
        try {
            rt_->collect_routing_stats();
        } catch (const std::exception& error) {
            spdlog::warn("routing stats collection failed: {}", error.what());
        }
        publish_stats();
        std::lock_guard lock(mu);
        active = false;
        current.reset();
    }

    std::deque<std::shared_ptr<RequestState>> abandoned;
    {
        std::lock_guard lock(mu);
        available = false;
        abandoned.swap(queue);
    }
    for (auto& request : abandoned) {
        request->fail(std::make_exception_ptr(
            RequestError(RequestErrorKind::Unavailable, "engine is shutting down")));
    }
    runner_.reset();
    rt_.reset();
}

void Engine::Impl::publish_stats() {
    const io::NgramStats ngram = rt_->ngram().stats();
    local_.ngram_lookups       = ngram.lookups;
    local_.ngram_cache_hits    = ngram.hits;
    local_.ngram_cache_hit_rate =
        ngram.lookups ? static_cast<double>(ngram.hits) / static_cast<double>(ngram.lookups) : 0.0;
    local_.routed_pairs     = decode_routed_;
    local_.gpu_routed_pairs = decode_routed_ - std::min(decode_cold_, decode_routed_);
    local_.gpu_hit_rate_by_routing_mass =
        decode_mass_ > 0.0 ? 1.0 - std::min(decode_cold_mass_, decode_mass_) / decode_mass_ : 0.0;
    local_.decode_routing_mass     = decode_mass_;
    local_.decode_gpu_routing_mass = decode_mass_ - std::min(decode_cold_mass_, decode_mass_);
    local_.expert_swaps            = rt_->expert_swaps();
    local_.cpu_wait_us_by_layer.assign(kLayers, 0.0);
    double total_ns = 0.0;
    for (int l = 0; l < kLayers; ++l) {
        total_ns += decode_wait_ns_[std::size_t(l)];
        if (local_.decode_steps != 0) {
            local_.cpu_wait_us_by_layer[std::size_t(l)] =
                decode_wait_ns_[std::size_t(l)] / 1000.0 / static_cast<double>(local_.decode_steps);
        }
    }
    local_.cpu_wait_us_per_layer =
        local_.decode_steps ? total_ns / 1000.0 / static_cast<double>(local_.decode_steps * kLayers)
                            : 0.0;
    local_.accepted_tokens_per_round_mtp =
        local_.mtp_rounds ? static_cast<double>(local_.mtp_accepted_tokens) /
                                static_cast<double>(local_.mtp_rounds)
                          : 0.0;
    local_.accepted_tokens_per_round_lookup =
        local_.lookup_rounds ? static_cast<double>(local_.lookup_accepted_tokens) /
                                   static_cast<double>(local_.lookup_rounds)
                             : 0.0;
    std::lock_guard lock(mu);
    const std::uint32_t waiting = stats.waiting_requests;
    const bool is_active        = stats.active_request;
    stats                       = local_;
    stats.waiting_requests      = waiting;
    stats.active_request        = is_active;
    memory.snapshot_occupied    = sessions_->occupied();
    memory.host_pinned_bytes    = rt_->placement().plan().host_bytes + sessions_->pinned_bytes();
}

void Engine::Impl::sync_pool_counters() {
    const auto layer_timings = rt_->pool().timings();
    for (int l = 0; l < kLayers; ++l) {
        pool_pairs_[std::size_t(l)]   = layer_timings[std::size_t(l)].pairs;
        pool_compute_[std::size_t(l)] = layer_timings[std::size_t(l)].compute_ns;
    }
}

double Engine::Impl::account_decode_step(std::int32_t width, std::int32_t licensed) {
    const auto layer_timings = rt_->pool().timings();
    std::uint64_t cold       = 0;
    double cpu_ns            = 0.0;
    for (int l = 0; l < kLayers; ++l) {
        const auto& t   = layer_timings[std::size_t(l)];
        cold           += t.pairs - pool_pairs_[std::size_t(l)];
        const auto layer_ns = static_cast<double>(t.compute_ns - pool_compute_[std::size_t(l)]);
        cpu_ns += layer_ns;
        decode_wait_ns_[std::size_t(l)] += layer_ns;
        pool_pairs_[std::size_t(l)]   = t.pairs;
        pool_compute_[std::size_t(l)] = t.compute_ns;
    }
    decode_cold_ += cold;
    decode_routed_ += static_cast<std::uint64_t>(width) * kRoutedPerToken * kLayers;
    // Routing mass: each token's renormalized top-10 weights sum to 1 per layer; the cold part
    // is read back from the host-mapped exchange the router published for this step.
    const DecodeExchange* exchange = rt_->exchange_host();
    double cold_mass               = 0.0;
    for (int l = 0; l < kLayers; ++l) {
        const MoeDecodePublish& pub = exchange->publish[l];
        const int tokens            = std::clamp(pub.tokens, 0, kDecodeExchangeTokens);
        for (int t = 0; t < tokens; ++t) {
            const int count = std::clamp(pub.cold_count[t], 0, int(dims::kTopK));
            for (int j = 0; j < count; ++j) cold_mass += double(pub.cold_weight[t][j]);
        }
    }
    decode_cold_mass_ += cold_mass;
    decode_mass_ += static_cast<double>(width) * kLayers;
    ++local_.decode_steps;
    local_.committed_decode_tokens += static_cast<std::uint64_t>(licensed);
    return cpu_ns * 1e-9;
}

void Engine::Impl::process(RequestState& rq) {
    const auto t_begin = Clock::now();
    GenerationResult result;
    result.prompt                     = rq.summary;
    result.thinking.configured_budget = rq.options.execution.thinking.budget;
    result.speculative.backend        = options.speculative.backend;
    result.timings.prepare_seconds    = rq.preparation.seconds;

    if (rq.cancelled.load()) {
        result.finish_reason = FinishReason::Cancelled;
        rq.finish(std::move(result));
        return;
    }
    if (t_begin > rq.deadline) {
        throw RequestError(RequestErrorKind::QueueTimeout,
                           "inference request expired while waiting in the queue");
    }

    const std::vector<std::int32_t>& prompt = rq.prompt.token_ids;
    const auto n                            = static_cast<std::int32_t>(prompt.size());
    const std::int32_t capacity             = rt_->options().max_context;
    if (n == 0) throw RequestError(RequestErrorKind::InvalidRequest, "prompt is empty");
    if (n >= capacity) {
        throw RequestError(RequestErrorKind::ContextLengthExceeded,
                           "prompt of " + std::to_string(n) + " tokens leaves no room in context " +
                               std::to_string(capacity));
    }
    const std::uint32_t requested  = rq.options.execution.requested_output_tokens;
    const auto room                = static_cast<std::uint32_t>(capacity - n);
    const std::uint32_t effective  = std::min(requested, room);
    const FinishReason limit_reason =
        requested <= room ? FinishReason::OutputLimit : FinishReason::ContextCapacity;

    text::OutputSession session = frontend.make_output_session(
        rq.prompt, rq.options.stop, rq.options.output, rq.options.execution.thinking);
    try {
        session.validate_generation_capacity(effective);
    } catch (const std::invalid_argument& error) {
        throw RequestError(RequestErrorKind::ThinkingBudgetCapacityInsufficient, error.what());
    }

    kernels::SamplingConfig sampling;
    sampling.temperature       = rq.sampling.temperature;
    sampling.top_k             = rq.sampling.top_k;
    sampling.top_p             = rq.sampling.top_p;
    sampling.min_p             = rq.sampling.min_p;
    sampling.presence_penalty  = rq.sampling.presence_penalty;
    sampling.frequency_penalty = rq.sampling.frequency_penalty;
    sampling.seed              = rq.sampling.seed;
    runner_->set_sampling(sampling);

    // Session reuse.
    std::int32_t reused = 0;
    try {
        if (rq.prompt.allow_prefix_reuse && rq.options.execution.allow_prefix_reuse) {
            const std::size_t lcp = engine::common_prefix(seq_, prompt);
            reused                = sessions_->restore_best(*rt_, lcp, prompt.size());
        }
        if (reused == 0) {
            rt_->reset_sequence();
            sessions_->clear();
        } else {
            sessions_->drop_above(reused);
            spdlog::info("reused {} tokens", reused);
        }
    } catch (...) {
        seq_.clear();
        sessions_->clear();
        throw;
    }
    seq_.assign(prompt.begin(), prompt.end());
    result.reused_prompt_tokens = static_cast<std::uint32_t>(reused);
    ++local_.requests;
    local_.reused_prompt_tokens += static_cast<std::uint64_t>(reused);

    if (rq.streaming()) {
        Event start;
        start.kind  = Event::Kind::Start;
        start.start = GenerationStart{rq.summary, static_cast<std::uint32_t>(reused)};
        rq.push(std::move(start));
    }
    const auto push_progress = [&](std::int32_t position) {
        if (!rq.streaming() || !rq.observation.prompt_progress) return;
        Event e;
        e.kind     = Event::Kind::Progress;
        e.progress = PromptProgress{static_cast<std::uint32_t>(n),
                                    static_cast<std::uint32_t>(reused),
                                    static_cast<std::uint32_t>(position),
                                    ns_between(t_begin, Clock::now())};
        rq.push(std::move(e));
    };
    push_progress(reused);

    std::vector<TokenId> generated;
    const auto publish = [&](text::PublishedOutput output) {
        for (OutputDelta& delta : output) {
            if (delta.text.empty()) continue;
            (delta.channel == OutputChannel::Reasoning ? result.reasoning : result.content) +=
                delta.text;
            if (rq.streaming()) {
                Event e;
                e.kind  = Event::Kind::Delta;
                e.delta = std::move(delta);
                rq.push(std::move(e));
            }
        }
    };
    Clock::time_point t_first{};
    const auto push_timing = [&] {
        if (!rq.streaming() || !rq.observation.live_timings) return;
        Event e;
        e.kind   = Event::Kind::Timing;
        e.timing = GenerationTimingObservation{static_cast<std::uint32_t>(generated.size()),
                                               ns_between(t_begin, t_first),
                                               ns_between(t_first, Clock::now())};
        rq.push(std::move(e));
    };
    const auto feed = [&](std::span<const TokenId> tokens) {
        const auto remaining = static_cast<std::uint32_t>(effective - generated.size());
        text::OutputDecision decision = session.preview_model(tokens, remaining, limit_reason);
        publish(session.commit_preview());
        generated.insert(generated.end(), tokens.begin(),
                         tokens.begin() + static_cast<std::ptrdiff_t>(decision.accepted_tokens));
        push_timing();
        return decision;
    };

    text::OutputDecision decision;
    try {
        // Prefill.
        const auto t_prefill = Clock::now();
        model::PrefillHooks hooks;
        hooks.after_chunk = [&](std::int32_t position) {
            if (position < n && position % engine::kSnapshotInterval == 0) {
                sessions_->take(*rt_);
            }
            push_progress(position);
            if (rq.cancelled.load() && position < n) throw PrefillCancelled{position};
        };
        try {
            rt_->prefill(std::span<const std::int32_t>(prompt).subspan(std::size_t(reused)),
                         &hooks);
        } catch (const PrefillCancelled& cancelled) {
            // The state is consistent at the chunk boundary; the MTP KV of position - 1 was built
            // with prompt[position].
            seq_.resize(std::size_t(cancelled.position) + 1);
            sessions_->drop_above(cancelled.position);
            local_.computed_prefill_tokens += std::uint64_t(cancelled.position - reused);
            result.finish_reason = FinishReason::Cancelled;
            result.timings.prefill_seconds = seconds_between(t_prefill, Clock::now());
            result.timings.total_seconds =
                rq.preparation.seconds + seconds_between(rq.submitted, Clock::now());
            rq.finish(std::move(result));
            return;
        }
        local_.computed_prefill_tokens += std::uint64_t(n - reused);
        sessions_->take(*rt_); // prompt end

        model::StepResult step = runner_->first_token(n);
        sync_pool_counters();
        t_first                = Clock::now();
        result.timings.prefill_seconds = seconds_between(t_prefill, t_first);
        result.timings.prompt_wall_seconds = seconds_between(t_begin, t_first);
        result.timings.first_token_seconds =
            rq.preparation.seconds + seconds_between(rq.submitted, t_first);
        local_.prefill_seconds += result.timings.prefill_seconds;

        TokenId anchor = step.licensed[0];
        seq_.push_back(anchor);
        std::array<std::int32_t, model::kMtpDrafts> mtp_drafts{};
        std::copy(std::begin(step.mtp_drafts), std::end(step.mtp_drafts), mtp_drafts.begin());
        int mtp_available = step.mtp_count;
        decision          = feed(std::span<const TokenId>(&anchor, 1));

        const bool lookup_enabled = options.speculative.backend == SpeculativeBackend::Auto ||
                                    options.speculative.backend == SpeculativeBackend::Lookup;
        result.speculative.accepted_per_position.assign(kMaxPositionStats, 0);
        const auto t_decode = Clock::now();
        // A step is timed with the drawing of the next MTP drafts inside. Its latency reaches
        // the controller only after the decision about those drafts, so that whether sampled
        // drafts are offered cannot depend on what was drawn (model::StepRunner::step).
        struct {
            int width      = 0;
            double seconds = 0.0, cpu_seconds = 0.0;
        } timed;
        const auto record_timed = [&] {
            if (timed.width > 0) {
                controller_->record_step(timed.width, timed.seconds, timed.cpu_seconds);
            }
            timed.width = 0;
        };
        while (!decision.finished()) {
            if (decision.continuation == text::ContinuationAction::ApplyTargetControl) {
                record_timed();
                const auto control_span = session.pending_control_tokens();
                const std::vector<TokenId> control(control_span.begin(), control_span.end());
                std::vector<TokenId> forced{anchor};
                forced.insert(forced.end(), control.begin(), control.end() - 1);
                for (const TokenId token : forced) {
                    if (rt_->position() + 1 > capacity) break;
                    const model::StepResult r = runner_->step(token, {});
                    controller_->record_step(1, r.seconds, account_decode_step(1, 0));
                }
                seq_.insert(seq_.end(), control.begin(), control.end());
                const auto remaining = static_cast<std::uint32_t>(effective - generated.size());
                decision             = session.preview_control(control, remaining);
                publish(session.commit_preview());
                generated.insert(generated.end(), control.begin(), control.end());
                push_timing();
                anchor        = control.back();
                mtp_available = 0;
                continue;
            }
            if (rq.cancelled.load()) {
                decision = session.preview_terminal(FinishReason::Cancelled);
                publish(session.commit_preview());
                break;
            }
            const std::int32_t pos = rt_->position();
            const auto remaining   = static_cast<std::uint32_t>(effective - generated.size());
            const std::uint32_t model_remaining = session.model_token_budget_remaining(remaining);
            if (model_remaining == 0 || pos + 1 > capacity) {
                decision = session.preview_terminal(FinishReason::ContextCapacity);
                publish(session.commit_preview());
                break;
            }
            const int kmax = std::max(
                0, std::min({kMaxLookupDrafts, static_cast<int>(model_remaining) - 1,
                             capacity - pos - 1}));

            std::array<TokenId, kMaxLookupDrafts> look{};
            int look_n    = 0;
            float look_p  = 0.0F;
            if (lookup_enabled && kmax > 0) {
                look_n = static_cast<int>(lookup_.propose(
                    seq_, std::span<TokenId>(look.data(), std::size_t(kmax))));
                if (look_n > 0) look_p = std::exp(lookup_.log_probability());
            }
            const engine::SpecDecision choice =
                controller_->decide(std::min(mtp_available, kmax), look_n, look_p);
            record_timed();
            std::span<const std::int32_t> drafts;
            if (choice.source == engine::SpecSource::Mtp) {
                drafts = std::span<const std::int32_t>(mtp_drafts.data(), std::size_t(choice.k));
            } else if (choice.source == engine::SpecSource::Lookup) {
                drafts = std::span<const std::int32_t>(look.data(), std::size_t(choice.k));
            }

            const model::StepResult r =
                runner_->step(anchor, drafts, choice.source == engine::SpecSource::Mtp);
            const int width           = 1 + choice.k;
            timed                     = {width, r.seconds, account_decode_step(width, r.count)};
            if (look_n > 0) lookup_.observe(r.tokens());
            ++result.speculative.rounds;
            switch (choice.source) {
            case engine::SpecSource::Mtp:
                controller_->record_mtp_outcome(choice.k, r.accepted);
                ++result.speculative.mtp_rounds;
                result.speculative.mtp_drafted += std::uint64_t(choice.k);
                result.speculative.mtp_accepted += std::uint64_t(r.accepted);
                ++local_.mtp_rounds;
                local_.mtp_drafted_tokens += std::uint64_t(choice.k);
                local_.mtp_accepted_tokens += std::uint64_t(r.accepted);
                break;
            case engine::SpecSource::Lookup:
                controller_->record_lookup_outcome(0, choice.k, r.accepted);
                ++result.speculative.lookup_rounds;
                result.speculative.lookup_drafted += std::uint64_t(choice.k);
                result.speculative.lookup_accepted += std::uint64_t(r.accepted);
                ++local_.lookup_rounds;
                local_.lookup_drafted_tokens += std::uint64_t(choice.k);
                local_.lookup_accepted_tokens += std::uint64_t(r.accepted);
                break;
            case engine::SpecSource::None:
                controller_->record_plain_round();
                ++result.speculative.plain_steps;
                break;
            }
            result.speculative.drafted_tokens += std::uint64_t(choice.k);
            result.speculative.accepted_tokens += std::uint64_t(r.accepted);
            for (int i = 0; i < std::min(r.accepted, kMaxPositionStats); ++i) {
                ++result.speculative.accepted_per_position[std::size_t(i)];
            }

            const auto licensed = r.tokens();
            seq_.insert(seq_.end(), licensed.begin(), licensed.end());
            anchor = licensed.back();
            std::copy(std::begin(r.mtp_drafts), std::end(r.mtp_drafts), mtp_drafts.begin());
            mtp_available = r.mtp_count;
            decision      = feed(licensed);
            if (local_.decode_steps % 32 == 0) publish_stats();
        }
        record_timed();
        const auto t_end = Clock::now();
        result.timings.decode_seconds          = seconds_between(t_decode, t_end);
        result.timings.generation_wall_seconds = seconds_between(t_first, t_end);
        local_.decode_seconds += result.timings.decode_seconds;
        sessions_->take(*rt_); // generation end
    } catch (...) {
        // The runtime state is unknown: the next request starts a fresh sequence.
        seq_.clear();
        sessions_->clear();
        throw;
    }

    result.generated_token_ids = std::move(generated);
    result.finish_reason       = decision.finish_reason;
    result.matched_stop_string = session.matched_stop_string();
    result.tool_calls          = session.take_tool_calls();
    result.tool_call_parse     = session.tool_call_parse_diagnostics();
    result.reasoning_tokens    = session.reasoning_tokens();
    result.thinking            = session.thinking_stats();
    result.timings.total_seconds =
        rq.preparation.seconds + seconds_between(rq.submitted, Clock::now());
    // Publish request counters before waking waiters (bench snapshots them immediately).
    publish_stats();
    rq.finish(std::move(result));
}

// ---------------------------------------------------------------------------------------------
// GenerationHandle

class GenerationHandle::Impl {
public:
    // `engine` keeps the Engine::Impl (and its worker) alive while the handle exists.
    Impl(std::shared_ptr<const void> engine_impl, std::shared_ptr<RequestState> request_state)
        : engine(std::move(engine_impl)), state(std::move(request_state)) {}

    ~Impl() {
        if (!consumed) state->cancelled.store(true);
    }

    Impl(const Impl&)            = delete;
    Impl& operator=(const Impl&) = delete;

    std::shared_ptr<const void> engine;
    std::shared_ptr<RequestState> state;
    bool consumed = false;
};

GenerationHandle::GenerationHandle() noexcept = default;
GenerationHandle::~GenerationHandle()         = default;
GenerationHandle::GenerationHandle(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
GenerationHandle::GenerationHandle(GenerationHandle&&) noexcept            = default;
GenerationHandle& GenerationHandle::operator=(GenerationHandle&&) noexcept = default;

GenerationHandle::operator bool() const noexcept { return impl_ != nullptr && !impl_->consumed; }

const ResolvedSamplingParameters& GenerationHandle::resolved_sampling() const noexcept {
    static const ResolvedSamplingParameters empty{};
    return impl_ ? impl_->state->sampling : empty;
}

GenerationResult GenerationHandle::wait(OutputSink* sink, const CancellationView& cancellation) {
    if (!impl_ || impl_->consumed) throw std::logic_error("generation handle is empty or consumed");
    RequestState& st = *impl_->state;
    if (st.streaming() != (sink != nullptr)) {
        throw std::invalid_argument("GenerationHandle wait sink does not match its consumer mode");
    }
    impl_->consumed = true;
    try {
        std::unique_lock lock(st.mu);
        for (;;) {
            while (!st.events.empty()) {
                Event event = std::move(st.events.front());
                st.events.pop_front();
                lock.unlock();
                switch (event.kind) {
                case Event::Kind::Start:
                    sink->start(event.start);
                    break;
                case Event::Kind::Progress:
                    sink->progress(event.progress);
                    break;
                case Event::Kind::Timing:
                    sink->timing(event.timing);
                    break;
                case Event::Kind::Delta:
                    sink->publish(std::move(event.delta));
                    break;
                }
                lock.lock();
            }
            if (st.done) break;
            st.cv.wait_for(lock, std::chrono::milliseconds(20));
            if (!st.done && st.events.empty()) {
                lock.unlock();
                if (cancellation.requested()) st.cancelled.store(true);
                lock.lock();
            }
        }
        if (st.error) std::rethrow_exception(st.error);
        return std::move(st.result);
    } catch (...) {
        st.cancelled.store(true);
        throw;
    }
}

// ---------------------------------------------------------------------------------------------
// Engine

Engine::Engine(EngineOptions options) : impl_(std::make_shared<Impl>(std::move(options))) {}
Engine::~Engine()                            = default;
Engine::Engine(Engine&&) noexcept            = default;
Engine& Engine::operator=(Engine&&) noexcept = default;

PreparedPrompt Engine::prepare(PromptInput input, const PreparationControl& control) const {
    auto impl     = std::make_unique<PreparedPrompt::Impl>();
    impl->text    = impl_->frontend.prepare(std::move(input), control);
    impl->summary = PromptSummary{impl->text.starts_in_reasoning,
                                  static_cast<std::uint32_t>(impl->text.token_ids.size())};
    return PreparedPrompt(std::move(impl));
}

PreparedPrompt Engine::prepare_tokens(std::vector<TokenId> token_ids,
                                      bool allow_prefix_reuse) const {
    auto impl     = std::make_unique<PreparedPrompt::Impl>();
    impl->text    = impl_->frontend.prepare_tokens(std::move(token_ids), allow_prefix_reuse);
    impl->summary = PromptSummary{false, static_cast<std::uint32_t>(impl->text.token_ids.size())};
    return PreparedPrompt(std::move(impl));
}

std::vector<TokenId> Engine::tokenize_text(std::string_view text) const {
    return impl_->frontend.tokenize_text(text);
}

std::uint32_t Engine::count_tokens(PromptInput input, const PreparationControl& control) const {
    return impl_->frontend.count_tokens(std::move(input), control);
}

ModelSamplingDefaults Engine::sampling_defaults() const {
    return impl_->frontend.sampling_defaults();
}

GenerationHandle Engine::submit(PreparedPrompt prompt, RequestOptions options,
                                OutputConsumerMode consumer_mode,
                                GenerationObservationOptions observation,
                                std::chrono::steady_clock::time_point pending_deadline) {
    if (!prompt) throw std::invalid_argument("prepared prompt is empty");
    auto state         = std::make_shared<RequestState>();
    state->submitted   = Clock::now();
    state->prompt      = std::move(prompt.impl_->text);
    state->summary     = prompt.impl_->summary;
    state->preparation = state->prompt.stats;
    state->mode        = consumer_mode;
    state->observation = observation;
    state->deadline =
        pending_deadline != std::chrono::steady_clock::time_point{}
            ? pending_deadline
            : state->submitted + std::chrono::milliseconds(impl_->options.pending_timeout_ms);
    std::uint64_t seed = 0;
    if (impl_->options.seed) {
        seed = *impl_->options.seed;
    } else {
        std::lock_guard lock(impl_->mu);
        seed = impl_->seed_rng();
    }
    state->sampling = impl_->frontend.resolve_sampling(options.execution.sampling,
                                                       state->summary.starts_in_reasoning, seed);
    state->options  = std::move(options);
    prompt.impl_.reset();

    if (state->options.execution.requested_output_tokens == 0) {
        GenerationResult result;
        result.prompt                     = state->summary;
        result.finish_reason              = FinishReason::OutputLimit;
        result.thinking.configured_budget = state->options.execution.thinking.budget;
        result.timings.prepare_seconds    = state->preparation.seconds;
        result.timings.total_seconds      = state->preparation.seconds;
        if (state->streaming()) {
            Event start;
            start.kind  = Event::Kind::Start;
            start.start = GenerationStart{state->summary, 0};
            state->push(std::move(start));
        }
        state->finish(std::move(result));
    } else {
        impl_->enqueue(state);
    }
    return GenerationHandle(std::make_unique<GenerationHandle::Impl>(impl_, std::move(state)));
}

GenerationResult Engine::generate(PreparedPrompt prompt, RequestOptions options, OutputSink* sink,
                                  const CancellationView& cancellation) {
    GenerationHandle handle =
        submit(std::move(prompt), std::move(options),
               sink != nullptr ? OutputConsumerMode::Streaming : OutputConsumerMode::Aggregate);
    return handle.wait(sink, cancellation);
}

const EngineOptions& Engine::options() const { return impl_->options; }

LoadSummary Engine::load_summary() const {
    std::lock_guard lock(impl_->mu);
    return impl_->load;
}

MemorySummary Engine::memory_summary() const {
    std::lock_guard lock(impl_->mu);
    return impl_->memory;
}

RuntimeStats Engine::runtime_stats() const {
    std::lock_guard lock(impl_->mu);
    RuntimeStats stats      = impl_->stats;
    stats.waiting_requests  = static_cast<std::uint32_t>(impl_->queue.size());
    stats.active_request    = impl_->active;
    return stats;
}

bool Engine::is_available() const {
    std::lock_guard lock(impl_->mu);
    return impl_->available;
}

// Peaks are not tracked separately: memory_summary() reports the planned allocations, which are
// fixed for the engine's lifetime.
void Engine::reset_memory_peaks() noexcept {}

} // namespace fngine
