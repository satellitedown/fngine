// Host-only support for fngine-bench: options, test matrix, corpus, statistics and reports.
// Structure adapted from cinference:bench/inference/ninfer_bench_support.h @ 383e5db
// (Apache-2.0, see NOTICE).
#pragma once

#include "fngine/types.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace fngine::bench {

inline constexpr int kSchemaVersion                  = 1;
inline constexpr std::string_view kReportType        = "fngine_bench_report";
inline constexpr int kDefaultNGen                    = 256;
inline constexpr int kDefaultRepetitions             = 3;
inline constexpr int kDefaultWarmup                  = 1;
inline constexpr std::string_view kDefaultCorpusRoot = "/home/obelisk/Projects/cinference";

// One test = a fresh (non-reused) prompt of n_prompt corpus tokens followed by n_gen decoded
// tokens. Prefill tok/s = n_prompt / prefill_seconds; decode tok/s = (generated-1) /
// decode_seconds (the first token belongs to the prefill round).
struct BenchTest {
    int n_prompt = 0;
    int n_gen    = 0;
    std::string label;

    [[nodiscard]] std::uint32_t requested_output_tokens() const;
    [[nodiscard]] std::uint32_t required_context() const;
};

enum class OutputFormat { Table, Json };

struct BenchOptions {
    EngineOptions engine;
    std::vector<int> n_prompt;
    int n_gen       = kDefaultNGen;
    int repetitions = kDefaultRepetitions;
    int warmup      = kDefaultWarmup;
    // Whitespace-separated token-id file; empty = built from corpus_root text at startup.
    std::filesystem::path corpus_path;
    std::filesystem::path corpus_root{std::string(kDefaultCorpusRoot)};
    bool ctx_explicit   = false;
    bool coding_chat    = false;
    // Decode with sampling (the model's default preset unless overridden) instead of greedy;
    // every measured request gets its own fixed seed.
    bool sample = false;
    std::optional<float> temperature;
    std::optional<float> top_p;
    OutputFormat output = OutputFormat::Table;
    std::filesystem::path output_file;
    bool help_requested = false;
};

// Engine counters attributable to one repetition (difference of two RuntimeStats snapshots,
// with the rates recomputed over the interval).
struct StatsDelta {
    std::uint64_t routed_pairs          = 0;
    std::uint64_t gpu_routed_pairs      = 0;
    double gpu_hit_rate_by_routing_mass = 0.0;
    double cpu_wait_us_per_layer        = 0.0;
    std::uint64_t decode_steps          = 0;
    std::uint64_t mtp_rounds            = 0;
    std::uint64_t mtp_accepted_tokens   = 0;
    std::uint64_t lookup_rounds         = 0;
    std::uint64_t lookup_accepted_tokens = 0;
    std::uint64_t ngram_lookups         = 0;
    std::uint64_t ngram_cache_hits      = 0;
    std::uint64_t reused_prompt_tokens  = 0;
    // Decode routing mass in the interval (all, GPU-computed) and completed expert swaps.
    double routing_mass                 = 0.0;
    double gpu_routing_mass             = 0.0;
    std::uint64_t expert_swaps          = 0;
};

StatsDelta stats_delta(const RuntimeStats& before, const RuntimeStats& after);

struct RepTiming {
    GenerationTimings timings;
    SpeculativeStats speculative;
    std::uint32_t prompt_tokens           = 0;
    std::uint32_t reused_prompt_tokens    = 0;
    std::uint32_t generated_output_tokens = 0;
    StatsDelta engine;
};

struct TestResult {
    BenchTest test;
    std::vector<RepTiming> reps;
};

struct Stats {
    double mean   = 0.0;
    double stddev = 0.0;
    int count     = 0;
};

struct BenchEnvironment {
    std::string gpu_name;
    std::string pack_path;
    LoadSummary load;
    MemorySummary memory;
    EngineOptions engine;
    int repetitions = 0;
    int warmup      = 0;
    std::string corpus_source;
    std::size_t corpus_tokens = 0;
};

BenchOptions parse_args(int argc, char** argv);
std::string usage_text(std::string_view program);

std::vector<BenchTest> expand_tests(const BenchOptions& options);
// Context to request: the largest test requirement (unless --ctx was given, which must cover it).
std::uint32_t resolve_max_context(const std::vector<BenchTest>& tests,
                                  std::optional<std::uint32_t> explicit_context);

std::vector<TokenId> load_corpus_ids(const std::filesystem::path& path);
// Deterministic text corpus: <root>/README.md then every *.md under <root>/docs in sorted path
// order, concatenated with blank lines, tokenized, and repeated until at least min_tokens.
std::vector<TokenId>
build_text_corpus(const std::filesystem::path& root,
                  const std::function<std::vector<TokenId>(std::string_view)>& tokenize,
                  std::size_t min_tokens);
std::vector<TokenId> prompt_slice(const std::vector<TokenId>& corpus, int n_prompt);

Stats compute_stats(const std::vector<double>& values);
std::vector<double> prefill_tok_s_series(const TestResult& result);
std::vector<double> decode_tok_s_series(const TestResult& result);

std::string format_table(const BenchEnvironment& env, const std::vector<TestResult>& results);
std::string format_json(const BenchEnvironment& env, const std::string& command,
                        const std::vector<TestResult>& results);

std::string json_escape(std::string_view value);

} // namespace fngine::bench
