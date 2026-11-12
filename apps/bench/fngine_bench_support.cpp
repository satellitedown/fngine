// Structure adapted from cinference:bench/inference/ninfer_bench_support.cpp @ 383e5db
// (Apache-2.0, see NOTICE).
#include "fngine_bench_support.h"

#include "serve/cli_common.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace fngine::bench {
namespace {

int parse_positive(std::string_view text, const char* label) {
    int value         = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (text.empty() || result.ec != std::errc{} || result.ptr != text.data() + text.size() ||
        value <= 0) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + std::string(text));
    }
    return value;
}

int parse_nonnegative(std::string_view text, const char* label) {
    int value         = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (text.empty() || result.ec != std::errc{} || result.ptr != text.data() + text.size() ||
        value < 0) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + std::string(text));
    }
    return value;
}

std::vector<int> parse_int_list(std::string_view value, const char* label) {
    std::vector<int> out;
    std::size_t start = 0;
    while (true) {
        const std::size_t comma      = value.find(',', start);
        const std::string_view piece = value.substr(
            start, comma == std::string_view::npos ? std::string_view::npos : comma - start);
        out.push_back(parse_positive(piece, label));
        if (comma == std::string_view::npos) { break; }
        start = comma + 1;
    }
    return out;
}

std::string number(double value) {
    if (!std::isfinite(value)) return "null";
    std::ostringstream out;
    out << std::setprecision(10) << value;
    return out.str();
}

std::string rate_cell(const std::vector<double>& values) {
    if (values.empty()) { return "-"; }
    const Stats stats = compute_stats(values);
    std::ostringstream out;
    out << std::fixed << std::setprecision(2) << stats.mean;
    if (stats.count > 1) { out << " \u00b1 " << std::fixed << std::setprecision(2) << stats.stddev; }
    return out.str();
}

double ratio(std::uint64_t numerator, std::uint64_t denominator) {
    return denominator == 0 ? 0.0
                            : static_cast<double>(numerator) / static_cast<double>(denominator);
}

// Aggregated engine counters over all measured repetitions of one test.
StatsDelta total_engine(const TestResult& result) {
    StatsDelta out;
    for (const RepTiming& rep : result.reps) {
        out.routed_pairs += rep.engine.routed_pairs;
        out.gpu_routed_pairs += rep.engine.gpu_routed_pairs;
        out.mtp_rounds += rep.engine.mtp_rounds;
        out.mtp_accepted_tokens += rep.engine.mtp_accepted_tokens;
        out.lookup_rounds += rep.engine.lookup_rounds;
        out.lookup_accepted_tokens += rep.engine.lookup_accepted_tokens;
        out.ngram_lookups += rep.engine.ngram_lookups;
        out.ngram_cache_hits += rep.engine.ngram_cache_hits;
        out.reused_prompt_tokens += rep.engine.reused_prompt_tokens;
        out.routing_mass += rep.engine.routing_mass;
        out.gpu_routing_mass += rep.engine.gpu_routing_mass;
        out.expert_swaps += rep.engine.expert_swaps;
        out.decode_steps += rep.engine.decode_steps;
        out.gpu_hit_rate_by_routing_mass =
            out.routing_mass > 0.0 ? out.gpu_routing_mass / out.routing_mass : 0.0;
        out.cpu_wait_us_per_layer += rep.engine.cpu_wait_us_per_layer *
                                     static_cast<double>(rep.engine.decode_steps);
    }
    if (out.decode_steps) out.cpu_wait_us_per_layer /= static_cast<double>(out.decode_steps);
    return out;
}

std::string percent(double fraction) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(1) << 100.0 * fraction << '%';
    return out.str();
}

} // namespace

std::uint32_t BenchTest::requested_output_tokens() const {
    return static_cast<std::uint32_t>(n_gen) + 1U;
}

std::uint32_t BenchTest::required_context() const {
    // Prompt + output + speculative verify slack (up to 8 drafts beyond the last token).
    return static_cast<std::uint32_t>(n_prompt) + requested_output_tokens() + 16U;
}

StatsDelta stats_delta(const RuntimeStats& before, const RuntimeStats& after) {
    StatsDelta d;
    d.routed_pairs                 = after.routed_pairs - before.routed_pairs;
    d.gpu_routed_pairs             = after.gpu_routed_pairs - before.gpu_routed_pairs;
    const auto steps = after.decode_steps - before.decode_steps;
    d.decode_steps = steps;
    d.cpu_wait_us_per_layer = steps ?
        (after.cpu_wait_us_per_layer * static_cast<double>(after.decode_steps) -
         before.cpu_wait_us_per_layer * static_cast<double>(before.decode_steps)) /
            static_cast<double>(steps) : 0.0;
    d.mtp_rounds                   = after.mtp_rounds - before.mtp_rounds;
    d.mtp_accepted_tokens          = after.mtp_accepted_tokens - before.mtp_accepted_tokens;
    d.lookup_rounds                = after.lookup_rounds - before.lookup_rounds;
    d.lookup_accepted_tokens       = after.lookup_accepted_tokens - before.lookup_accepted_tokens;
    d.ngram_lookups                = after.ngram_lookups - before.ngram_lookups;
    d.ngram_cache_hits             = after.ngram_cache_hits - before.ngram_cache_hits;
    d.routing_mass                 = after.decode_routing_mass - before.decode_routing_mass;
    d.gpu_routing_mass             = after.decode_gpu_routing_mass - before.decode_gpu_routing_mass;
    d.gpu_hit_rate_by_routing_mass =
        d.routing_mass > 0.0 ? d.gpu_routing_mass / d.routing_mass : 0.0;
    d.expert_swaps                 = after.expert_swaps - before.expert_swaps;
    d.reused_prompt_tokens         = after.reused_prompt_tokens - before.reused_prompt_tokens;
    return d;
}

std::string usage_text(std::string_view program) {
    if (program.empty()) { program = "fngine-bench"; }
    std::ostringstream out;
    out << "Usage: " << program << " --pack PATH [options]\n\n"
        << "Throughput benchmark over fngine::Engine. Each test prefills a fresh P-token prompt\n"
        << "(no snapshot reuse) and then decodes N tokens.\n\n"
        << "Options:\n"
        << "  -p, --n-prompt <list>       prompt lengths, e.g. 1024,4096,32768 (default 1024)\n"
        << "  -n, --n-gen <n>             decoded tokens per test (default " << kDefaultNGen
        << "; 0 = prefill only)\n"
        << "  -r, --repetitions <n>       measured repetitions (default " << kDefaultRepetitions
        << ")\n"
        << "  --warmup <n>                discarded repetitions (default " << kDefaultWarmup
        << ")\n"
        << "  --corpus <file>             whitespace-separated token ids\n"
        << "  --corpus-root <dir>         text corpus root when --corpus is absent (default "
        << kDefaultCorpusRoot << ")\n"
        << "  --coding-chat              two linked-list coding turns instead of the corpus\n"
        << "  --sample                   sample (model default preset, one fixed seed per request)\n"
        << "                             instead of greedy decoding\n"
        << "  --temperature <f>          with --sample: temperature override\n"
        << "  --top-p <f>                with --sample: top-p override\n"
        << "  -o, --output <table|json>   output format (default table)\n"
        << "  -f, --output-file <path>    also write the report to a file\n"
        << "  -h, --help                  show this help\n\nEngine:\n"
        << serve::engine_flags_usage()
        << "  (--ctx defaults to the largest test requirement)\n";
    return out.str();
}

BenchOptions parse_args(int argc, char** argv) {
    BenchOptions options;
    for (int i = 1; i < argc; ++i) {
        std::string_view arg(argv[i]);
        const char* attached = nullptr;
        if (arg.size() > 2 && arg[0] == '-' &&
            (arg[1] == 'p' || arg[1] == 'n' || arg[1] == 'r' || arg[1] == 'o')) {
            attached = argv[i] + 2;
            arg = arg.substr(0, 2);
        }
        const serve::RequireValue value = [&](const char* flag) -> const char* {
            if (attached) return std::exchange(attached, nullptr);
            if (++i >= argc) { throw std::invalid_argument(std::string(flag) + " needs a value"); }
            return argv[i];
        };
        if (arg == "-h" || arg == "--help") {
            options.help_requested = true;
            return options;
        }
        if (arg == "--ctx") options.ctx_explicit = true;
        if (serve::parse_engine_flag(arg, value, options.engine)) continue;
        if (arg == "-p" || arg == "--n-prompt") {
            auto parsed = parse_int_list(value("--n-prompt"), "n-prompt");
            options.n_prompt.insert(options.n_prompt.end(), parsed.begin(), parsed.end());
        } else if (arg == "-n" || arg == "--n-gen") {
            options.n_gen = parse_nonnegative(value("--n-gen"), "n-gen");
        } else if (arg == "-r" || arg == "--repetitions") {
            options.repetitions = parse_positive(value("--repetitions"), "repetitions");
        } else if (arg == "--warmup") {
            options.warmup = parse_nonnegative(value("--warmup"), "warmup");
        } else if (arg == "--coding-chat") {
            options.coding_chat = true;
        } else if (arg == "--sample") {
            options.sample = true;
        } else if (arg == "--temperature") {
            options.temperature = std::stof(value("--temperature"));
        } else if (arg == "--top-p") {
            options.top_p = std::stof(value("--top-p"));
        } else if (arg == "--corpus") {
            options.corpus_path = value("--corpus");
        } else if (arg == "--corpus-root") {
            options.corpus_root = value("--corpus-root");
        } else if (arg == "-o" || arg == "--output") {
            const std::string_view selected = value("--output");
            if (selected == "table") {
                options.output = OutputFormat::Table;
            } else if (selected == "json") {
                options.output = OutputFormat::Json;
            } else {
                throw std::invalid_argument("--output must be table or json");
            }
        } else if (arg == "-f" || arg == "--output-file") {
            options.output_file = value("--output-file");
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
    }
    if (!options.sample && (options.temperature || options.top_p))
        throw std::invalid_argument("--temperature and --top-p need --sample");
    if (options.coding_chat) {
        if (!options.n_prompt.empty() || !options.corpus_path.empty() ||
            options.corpus_root != kDefaultCorpusRoot) {
            throw std::invalid_argument("--coding-chat cannot be combined with prompt lengths or a corpus");
        }
        if (!options.ctx_explicit) {
            const std::uint64_t context = std::max<std::uint64_t>(
                8192, 2ULL * static_cast<unsigned>(options.n_gen) + 1024);
            if (context > std::numeric_limits<std::uint32_t>::max())
                throw std::invalid_argument("coding chat requires too much context");
            options.engine.max_context = static_cast<std::uint32_t>(context);
        }
        serve::validate_engine_options(options.engine);
        return options;
    }
    if (options.n_prompt.empty()) options.n_prompt.push_back(1024);
    options.engine.max_context = resolve_max_context(
        expand_tests(options),
        options.ctx_explicit ? std::optional<std::uint32_t>(options.engine.max_context)
                             : std::nullopt);
    serve::validate_engine_options(options.engine);
    return options;
}

std::vector<BenchTest> expand_tests(const BenchOptions& options) {
    std::vector<BenchTest> tests;
    if (options.coding_chat) return tests;
    for (const int p : options.n_prompt) {
        std::string label = "pp" + std::to_string(p);
        if (options.n_gen > 0) label += "+tg" + std::to_string(options.n_gen);
        tests.push_back(BenchTest{p, options.n_gen, std::move(label)});
    }
    return tests;
}

std::uint32_t resolve_max_context(const std::vector<BenchTest>& tests,
                                  std::optional<std::uint32_t> explicit_context) {
    std::uint32_t required = 0;
    for (const BenchTest& test : tests) required = std::max(required, test.required_context());
    if (explicit_context) {
        if (*explicit_context < required) {
            throw std::invalid_argument("--ctx " + std::to_string(*explicit_context) +
                                        " is below the largest test requirement " +
                                        std::to_string(required));
        }
        return *explicit_context;
    }
    return required;
}

std::vector<TokenId> load_corpus_ids(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file) throw std::invalid_argument("cannot open corpus: " + path.string());
    std::vector<TokenId> ids;
    long long value = 0;
    while (file >> value) {
        if (value < 0 || value > std::numeric_limits<TokenId>::max()) {
            throw std::invalid_argument("corpus token id out of range: " + std::to_string(value));
        }
        ids.push_back(static_cast<TokenId>(value));
    }
    if (!file.eof()) throw std::invalid_argument("corpus contains a non-integer token");
    if (ids.empty()) throw std::invalid_argument("corpus is empty: " + path.string());
    return ids;
}

std::vector<TokenId>
build_text_corpus(const std::filesystem::path& root,
                  const std::function<std::vector<TokenId>(std::string_view)>& tokenize,
                  std::size_t min_tokens) {
    std::vector<std::filesystem::path> files;
    if (std::filesystem::exists(root / "README.md")) files.push_back(root / "README.md");
    std::vector<std::filesystem::path> docs;
    if (std::filesystem::is_directory(root / "docs")) {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(root / "docs")) {
            if (entry.is_regular_file() && entry.path().extension() == ".md") {
                docs.push_back(entry.path());
            }
        }
    }
    std::sort(docs.begin(), docs.end());
    files.insert(files.end(), docs.begin(), docs.end());
    if (files.empty()) {
        throw std::invalid_argument("no README.md or docs/*.md under corpus root " +
                                    root.string());
    }
    std::string text;
    for (const auto& path : files) {
        std::ifstream file(path, std::ios::binary);
        std::ostringstream bytes;
        bytes << file.rdbuf();
        if (!text.empty()) text += "\n\n";
        text += bytes.str();
    }
    const std::vector<TokenId> once = tokenize(text);
    if (once.empty()) throw std::invalid_argument("corpus text tokenized to nothing");
    std::vector<TokenId> corpus;
    corpus.reserve(std::max(min_tokens, once.size()));
    while (corpus.size() < min_tokens || corpus.empty()) {
        corpus.insert(corpus.end(), once.begin(), once.end());
    }
    return corpus;
}

std::vector<TokenId> prompt_slice(const std::vector<TokenId>& corpus, int n_prompt) {
    if (n_prompt <= 0) throw std::invalid_argument("prompt length must be positive");
    if (static_cast<std::size_t>(n_prompt) > corpus.size()) {
        throw std::invalid_argument("corpus has " + std::to_string(corpus.size()) +
                                    " tokens, fewer than the requested prompt " +
                                    std::to_string(n_prompt));
    }
    return std::vector<TokenId>(corpus.begin(), corpus.begin() + n_prompt);
}

Stats compute_stats(const std::vector<double>& values) {
    Stats stats;
    stats.count = static_cast<int>(values.size());
    if (values.empty()) return stats;
    double sum = 0.0;
    for (const double v : values) sum += v;
    stats.mean = sum / static_cast<double>(values.size());
    if (values.size() > 1) {
        double sq = 0.0;
        for (const double v : values) sq += (v - stats.mean) * (v - stats.mean);
        stats.stddev = std::sqrt(sq / static_cast<double>(values.size() - 1));
    }
    return stats;
}

std::vector<double> prefill_tok_s_series(const TestResult& result) {
    std::vector<double> out;
    for (const RepTiming& rep : result.reps) {
        const double tokens = static_cast<double>(rep.prompt_tokens - rep.reused_prompt_tokens);
        if (rep.timings.prefill_seconds > 0.0 && tokens > 0.0) {
            out.push_back(tokens / rep.timings.prefill_seconds);
        }
    }
    return out;
}

std::vector<double> decode_tok_s_series(const TestResult& result) {
    std::vector<double> out;
    if (result.test.n_gen == 0) return out;
    for (const RepTiming& rep : result.reps) {
        if (rep.generated_output_tokens > 1 && rep.timings.decode_seconds > 0.0) {
            out.push_back(static_cast<double>(rep.generated_output_tokens - 1) /
                          rep.timings.decode_seconds);
        }
    }
    return out;
}

std::string format_table(const BenchEnvironment& env, const std::vector<TestResult>& results) {
    std::ostringstream out;
    out << "fngine-bench  gpu: " << env.gpu_name << "  pack: " << env.pack_path << '\n'
        << "spec: " << speculative_backend_name(env.engine.speculative.backend)
        << "  kv: " << kv_cache_storage_name(env.engine.kv_cache)
        << "  dense: " << (env.engine.dense_bf16 ? "bf16" : "fp8")
        << "  ctx: " << env.engine.max_context << "  prefill-chunk: " << env.engine.prefill_chunk
        << "  cpu-workers: " << env.engine.cpu_workers << "  gpu experts: "
        << env.memory.gpu_experts << '/' << env.memory.total_experts
        << "  reps: " << env.repetitions << " (+" << env.warmup << " warmup)\n\n";
    out << std::left << std::setw(18) << "test" << std::right << std::setw(20) << "prefill tok/s"
        << std::setw(20) << "decode tok/s" << std::setw(10) << "gpu-hit" << std::setw(8) << "swaps"
        << std::setw(14) << "cpu-wait/lyr" << std::setw(10) << "mtp a/r" << std::setw(10) << "lkp a/r"
        << std::setw(10) << "ngram-hit" << '\n';
    for (const TestResult& result : results) {
        const StatsDelta engine = total_engine(result);
        std::ostringstream wait, mtp, lookup;
        wait << std::fixed << std::setprecision(1) << engine.cpu_wait_us_per_layer << " us";
        mtp << std::fixed << std::setprecision(2)
            << ratio(engine.mtp_accepted_tokens, engine.mtp_rounds);
        lookup << std::fixed << std::setprecision(2)
               << ratio(engine.lookup_accepted_tokens, engine.lookup_rounds);
        out << std::left << std::setw(18) << result.test.label << std::right << std::setw(20)
            << rate_cell(prefill_tok_s_series(result)) << std::setw(20)
            << rate_cell(decode_tok_s_series(result)) << std::setw(10)
            << percent(engine.routing_mass > 0.0 ? engine.gpu_routing_mass / engine.routing_mass
                                                 : engine.gpu_hit_rate_by_routing_mass)
            << std::setw(8) << engine.expert_swaps << std::setw(14) << wait.str()
            << std::setw(10) << mtp.str() << std::setw(10) << lookup.str() << std::setw(10)
            << percent(ratio(engine.ngram_cache_hits, engine.ngram_lookups)) << '\n';
    }
    return out.str();
}

std::string json_escape(std::string_view value) {
    std::ostringstream out;
    for (const unsigned char c : value) {
        switch (c) {
        case '"':
            out << "\\\"";
            break;
        case '\\':
            out << "\\\\";
            break;
        case '\n':
            out << "\\n";
            break;
        case '\r':
            out << "\\r";
            break;
        case '\t':
            out << "\\t";
            break;
        default:
            if (c < 0x20) {
                out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c)
                    << std::dec << std::setfill(' ');
            } else {
                out << c;
            }
        }
    }
    return out.str();
}

std::string format_json(const BenchEnvironment& env, const std::string& command,
                        const std::vector<TestResult>& results) {
    std::ostringstream out;
    const auto stat = [&](std::string_view name, const std::vector<double>& values) {
        const Stats s = compute_stats(values);
        out << "      \"" << name << "_mean\": " << (values.empty() ? "null" : number(s.mean))
            << ",\n      \"" << name << "_stddev\": " << (values.empty() ? "null" : number(s.stddev))
            << ",\n";
    };
    out << "{\n  \"type\": \"" << kReportType << "\",\n  \"schema_version\": " << kSchemaVersion
        << ",\n  \"command\": \"" << json_escape(command) << "\",\n  \"environment\": {\n"
        << "    \"gpu\": \"" << json_escape(env.gpu_name) << "\",\n"
        << "    \"pack\": \"" << json_escape(env.pack_path) << "\",\n"
        << "    \"model\": \"" << json_escape(env.load.model_name) << "\",\n"
        << "    \"spec\": \"" << speculative_backend_name(env.engine.speculative.backend) << "\",\n"
        << "    \"kv\": \"" << kv_cache_storage_name(env.engine.kv_cache)
        << "\",\n    \"dense\": \"" << (env.engine.dense_bf16 ? "bf16" : "fp8") << "\",\n"
        << "    \"max_context\": " << env.engine.max_context << ",\n"
        << "    \"prefill_chunk\": " << env.engine.prefill_chunk << ",\n"
        << "    \"cpu_workers\": " << env.engine.cpu_workers << ",\n"
        << "    \"use_cuda_graph\": " << (env.engine.use_cuda_graph ? "true" : "false") << ",\n"
        << "    \"adaptive_experts\": " << (env.engine.adaptive_experts ? "true" : "false") << ",\n"
        << "    \"prefill_stream\": " << (env.engine.prefill_stream ? "true" : "false") << ",\n"
        << "    \"prefill_borrow\": " << (env.engine.prefill_borrow ? "true" : "false") << ",\n"
        << "    \"gpu_experts\": " << env.memory.gpu_experts << ",\n"
        << "    \"total_experts\": " << env.memory.total_experts << ",\n"
        << "    \"predicted_gpu_hit_rate\": " << number(env.memory.predicted_gpu_hit_rate) << ",\n"
        << "    \"repetitions\": " << env.repetitions << ",\n"
        << "    \"warmup\": " << env.warmup << ",\n"
        << "    \"corpus\": \"" << json_escape(env.corpus_source) << "\",\n"
        << "    \"corpus_tokens\": " << env.corpus_tokens << "\n  },\n  \"tests\": [\n";
    for (std::size_t t = 0; t < results.size(); ++t) {
        const TestResult& result = results[t];
        const StatsDelta engine  = total_engine(result);
        out << "    {\n      \"label\": \"" << json_escape(result.test.label) << "\",\n"
            << "      \"n_prompt\": " << result.test.n_prompt << ",\n"
            << "      \"n_gen\": " << result.test.n_gen << ",\n";
        stat("prefill_tok_s", prefill_tok_s_series(result));
        stat("decode_tok_s", decode_tok_s_series(result));
        out << "      \"gpu_hit_rate_by_routing_mass\": "
            << number(engine.gpu_hit_rate_by_routing_mass) << ",\n"
            << "      \"interval_gpu_hit_rate_by_routing_mass\": "
            << number(engine.routing_mass > 0.0 ? engine.gpu_routing_mass / engine.routing_mass : 0.0)
            << ",\n      \"expert_swaps\": " << engine.expert_swaps << ",\n"
            << "      \"gpu_routed_pairs\": " << engine.gpu_routed_pairs << ",\n"
            << "      \"routed_pairs\": " << engine.routed_pairs << ",\n"
            << "      \"cpu_wait_us_per_layer\": " << number(engine.cpu_wait_us_per_layer)
            << ",\n      \"accepted_tokens_per_round_mtp\": "
            << number(ratio(engine.mtp_accepted_tokens, engine.mtp_rounds)) << ",\n"
            << "      \"mtp_rounds\": " << engine.mtp_rounds << ",\n"
            << "      \"accepted_tokens_per_round_lookup\": "
            << number(ratio(engine.lookup_accepted_tokens, engine.lookup_rounds)) << ",\n"
            << "      \"lookup_rounds\": " << engine.lookup_rounds << ",\n"
            << "      \"ngram_cache_hit_rate\": "
            << number(ratio(engine.ngram_cache_hits, engine.ngram_lookups)) << ",\n"
            << "      \"reused_prompt_tokens\": " << engine.reused_prompt_tokens << ",\n"
            << "      \"reps\": [";
        for (std::size_t r = 0; r < result.reps.size(); ++r) {
            const RepTiming& rep = result.reps[r];
            out << (r == 0 ? "" : ", ") << "{\"prefill_seconds\": "
                << number(rep.timings.prefill_seconds)
                << ", \"decode_seconds\": " << number(rep.timings.decode_seconds)
                << ", \"total_seconds\": " << number(rep.timings.total_seconds)
                << ", \"prompt_tokens\": " << rep.prompt_tokens
                << ", \"reused_prompt_tokens\": " << rep.reused_prompt_tokens
                << ", \"generated_tokens\": " << rep.generated_output_tokens
                << ", \"spec_rounds\": " << rep.speculative.rounds
                << ", \"spec_accepted\": " << rep.speculative.accepted_tokens
                << ", \"gpu_hit_rate_by_routing_mass\": "
                << number(rep.engine.routing_mass > 0.0 ? rep.engine.gpu_routing_mass / rep.engine.routing_mass : 0.0)
                << ", \"expert_swaps\": " << rep.engine.expert_swaps << '}';
        }
        out << "]\n    }" << (t + 1 == results.size() ? "\n" : ",\n");
    }
    out << "  ]\n}\n";
    return out.str();
}

} // namespace fngine::bench
