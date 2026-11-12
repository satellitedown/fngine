// Modified for fngine from cinference:bench/inference/ninfer_bench.cpp @ 383e5db. See NOTICE.
// fngine changes: pp+tg tests over fresh prompts, engine RuntimeStats deltas per repetition,
// text corpus generated with the pack tokenizer when no id corpus is given.
#include "fngine_bench_support.h"

#include "fngine/engine.h"

#include <cuda_runtime.h>

#include <exception>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string command_line(int argc, char** argv) {
    std::ostringstream out;
    for (int i = 0; i < argc; ++i) out << (i == 0 ? "" : " ") << argv[i];
    return out.str();
}

std::string gpu_name(int device) {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return "unknown";
    return prop.name;
}

// `request_index` numbers the benchmark's requests; with --sample it is the request's seed.
fngine::RequestOptions benchmark_request(const fngine::bench::BenchTest& test,
                                         const fngine::bench::BenchOptions& options,
                                         std::uint64_t request_index) {
    fngine::RequestOptions request;
    request.execution.requested_output_tokens = test.requested_output_tokens();
    request.execution.allow_prefix_reuse      = false; // measure full prefill every repetition
    if (options.sample) {
        request.execution.sampling.temperature = options.temperature;
        request.execution.sampling.top_p       = options.top_p;
        request.execution.sampling.seed        = request_index;
    } else {
        request.execution.sampling.temperature = 0.0F; // deterministic, comparable acceptance
    }
    request.stop.include_model_defaults       = false; // decode exactly n_gen tokens
    request.output.raw                        = true;
    return request;
}

fngine::bench::RepTiming measured_result(const fngine::GenerationResult& result,
                                       const fngine::RuntimeStats& before,
                                       const fngine::RuntimeStats& after) {
    fngine::bench::RepTiming rep;
    rep.timings                 = result.timings;
    rep.speculative             = result.speculative;
    rep.prompt_tokens           = result.prompt.prompt_tokens;
    rep.reused_prompt_tokens    = result.reused_prompt_tokens;
    rep.generated_output_tokens = static_cast<std::uint32_t>(result.generated_token_ids.size());
    rep.engine                  = fngine::bench::stats_delta(before, after);
    return rep;
}

fngine::bench::RepTiming run_repetition(fngine::Engine& engine,
                                        const fngine::bench::BenchTest& test,
                                        const fngine::bench::BenchOptions& options,
                                        std::uint64_t request_index,
                                        const std::vector<fngine::TokenId>& corpus) {
    const fngine::RuntimeStats before = engine.runtime_stats();
    fngine::PreparedPrompt prompt =
        engine.prepare_tokens(fngine::bench::prompt_slice(corpus, test.n_prompt), false);
    const fngine::GenerationResult result =
        engine.generate(std::move(prompt), benchmark_request(test, options, request_index));
    return measured_result(result, before, engine.runtime_stats());
}

std::vector<fngine::bench::TestResult>
run_coding_chat(fngine::Engine& engine, const fngine::bench::BenchOptions& options) {
    std::vector<fngine::bench::TestResult> results{
        {{0, options.n_gen, "coding-turn1"}, {}}, {{0, options.n_gen, "coding-turn2"}, {}}};
    std::uint64_t request_index = 0;
    for (int repetition = -options.warmup; repetition < options.repetitions; ++repetition) {
        fngine::PromptInput input;
        input.messages.reserve(3);
        input.messages.push_back({.role = fngine::ChatRole::User,
            .content = "Write a C function that reverses a singly linked list. Include an explanation and tests."});
        for (std::size_t turn = 0; turn < results.size(); ++turn) {
            std::cerr << "[fngine-bench] coding chat repetition " << repetition + 1
                      << ", turn " << turn + 1 << '\n';
            const auto before = engine.runtime_stats();
            auto request = benchmark_request(results.front().test, options, request_index++);
            request.output.raw = false;
            request.execution.sampling.presence_penalty = 0.0F;
            request.execution.sampling.frequency_penalty = 0.0F;
            auto generated = engine.generate(engine.prepare(input), request);
            if (repetition >= 0)
                results[turn].reps.push_back(measured_result(generated, before, engine.runtime_stats()));
            if (turn == 0) {
                input.messages.push_back({.role = fngine::ChatRole::Assistant,
                    .content = std::move(generated.content),
                    .reasoning_content = std::move(generated.reasoning)});
                input.messages.push_back({.role = fngine::ChatRole::User,
                    .content = "Now adapt it for a doubly linked list and explain the pointer updates."});
            }
        }
    }
    return results;
}

} // namespace

int main(int argc, char** argv) {
    fngine::bench::BenchOptions options;
    try {
        options = fngine::bench::parse_args(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "fngine-bench: " << error.what() << '\n'
                  << fngine::bench::usage_text(argc > 0 ? argv[0] : "fngine-bench");
        return 1;
    }
    if (options.help_requested) {
        std::cout << fngine::bench::usage_text(argc > 0 ? argv[0] : "fngine-bench");
        return 0;
    }
    try {
        const std::vector<fngine::bench::BenchTest> tests = fngine::bench::expand_tests(options);
        std::cerr << "[fngine-bench] loading " << options.engine.pack_path.string() << '\n';
        fngine::Engine engine(options.engine);

        fngine::bench::BenchEnvironment env;
        env.gpu_name    = gpu_name(options.engine.device);
        env.pack_path   = options.engine.pack_path.string();
        env.load        = engine.load_summary();
        env.memory      = engine.memory_summary();
        env.engine      = engine.options();
        env.repetitions = options.repetitions;
        env.warmup      = options.warmup;

        std::size_t longest = 0;
        for (const auto& test : tests) longest = std::max<std::size_t>(longest, test.n_prompt);
        std::vector<fngine::TokenId> corpus;
        if (options.coding_chat) {
            env.corpus_source = "built-in two-turn linked-list coding chat";
        } else if (!options.corpus_path.empty()) {
            corpus            = fngine::bench::load_corpus_ids(options.corpus_path);
            env.corpus_source = options.corpus_path.string();
        } else {
            corpus = fngine::bench::build_text_corpus(
                options.corpus_root,
                [&](std::string_view text) { return engine.tokenize_text(text); }, longest);
            env.corpus_source = options.corpus_root.string() + " (README.md + docs/**/*.md)";
        }
        env.corpus_tokens = corpus.size();

        std::vector<fngine::bench::TestResult> results =
            options.coding_chat ? run_coding_chat(engine, options)
                                : std::vector<fngine::bench::TestResult>{};
        for (std::size_t i = 0; i < tests.size(); ++i) {
            const auto& test = tests[i];
            std::cerr << "[fngine-bench] test " << (i + 1) << '/' << tests.size() << ' '
                      << test.label << '\n';
            std::uint64_t request_index = 0;
            for (int w = 0; w < options.warmup; ++w)
                (void)run_repetition(engine, test, options, request_index++, corpus);
            fngine::bench::TestResult result{test, {}};
            for (int r = 0; r < options.repetitions; ++r) {
                result.reps.push_back(run_repetition(engine, test, options, request_index++, corpus));
            }
            results.push_back(std::move(result));
        }

        const std::string text = options.output == fngine::bench::OutputFormat::Json
                                     ? fngine::bench::format_json(env, command_line(argc, argv),
                                                                  results)
                                     : fngine::bench::format_table(env, results);
        std::cout << text;
        if (!options.output_file.empty()) {
            std::ofstream file(options.output_file);
            if (!file || !(file << text)) {
                throw std::runtime_error("cannot write " + options.output_file.string());
            }
        }
        return 0;
    } catch (const fngine::StartupError& error) {
        std::cerr << "fngine-bench: startup failed: " << error.what() << '\n';
        return error.exit_code();
    } catch (const std::exception& error) {
        std::cerr << "fngine-bench: " << error.what() << '\n';
        return 1;
    }
}
