# fngine

Text-only C++20/CUDA engine for Qwen3.8-Flash-Next (`qwen4_exp`), targeting an RTX 5090 (32 GiB) and approximately 60 GiB host RAM. Routed experts retain their native GSQ-RCO GGUF IQ formats; dense weights are BF16-derived FP8-row, the n-gram table is page-packed IQ4_NL on NVMe, and MTP experts are resident IQ4_NL. See `NOTICE` and `LICENSE` for upstream licensing.

**To run the published decensored (heretic) model, use the installer:** [flash-next-abliterated-fngine](https://github.com/satellitedown/flash-next-abliterated-fngine) builds this engine at a pinned commit with an isolated CUDA toolchain, downloads and verifies the pack, starts the server and updates both. One command (Linux, RTX 5090, 64 GB RAM); choose 1 to install, then 3 to start:

```sh
curl -fsSL https://raw.githubusercontent.com/satellitedown/flash-next-abliterated-fngine/main/get.sh | bash
```

The rest of this README covers building and developing the engine directly.

## Status and verification

The integrated build has recorded speculative/state equivalence, actual HTTP/tool/reasoning checks, near-256K-context retrieval, executable coding output, and throughput measurements under `results/`. **The original HF-reference numerical thresholds are not all met.** The validator still returns failure for those gates; practical output smoke checks do not replace broad model-quality evaluation.

Latest regression (2026-10-15, run on the heretic pack): **46/46 C++/CUDA tests passed, with no skips**, including the routed-expert ablation kernels, native-position indexer/RoPE boundaries, speculative context-tail state equivalence, real-model generation, streaming and session restore. Earlier evidence: `results/context-extension.json`. The performance-work verification also passed **23/23 Python tooling tests** (`results/verification.json`); those Python tests were not rerun for the C++ changes.

The target workstation's PCIe link was observed at **x4**, not x16, with approximately 11.7–14 GB/s host-to-device bandwidth measured across runs. Check slot/BIOS lane settings or reseat the card if appropriate; the prefill streaming cost model adapts to measured bandwidth. Desktop VRAM use and other GPU applications reduce expert residency. Stop competing GPU workloads before measuring; record desktop load and GPU power cap.

## Build

Requires Linux x86_64, CUDA 13.3 (the installer pins 13.3.73 with cuBLAS 13.6.0.2), a compatible host C++20 compiler, CMake ≥ 3.28, Ninja and liburing. The CUDA architecture is `120a` (RTX 5090). The CPU expert path needs AVX-512 F/BW/DQ/VL with VNNI, BF16, VBMI, BITALG and VPOPCNTDQ (AMD Zen 4/5, Intel Sapphire Rapids and later); it compiles for the building CPU by default, and `-DFNGINE_CPU_ARCH=<march>` targets another.

```sh
cmake --preset dev
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Use `cmake --preset release` for a release configuration. Tests requiring real data use:

```sh
export FNGINE_TEST_PACK=/mnt/data/fngine/pack
export FNGINE_TEST_REF=/mnt/data/fngine/ref
ctest --test-dir build --output-on-failure
```

## Fetch, convert and reference

The GGUF input is the two-shard ISTA-DASLab GSQ-RCO IQ3_S build. Set `GGUF1` and `GGUF2` to those existing shard paths. Conversion copies routed-expert bytes without requantization and fetches official dense BF16 weights separately. Allow sufficient NVMe capacity for inputs, pack and references.

```sh
uv venv --python 3.12 tools/.venv
uv pip install --python tools/.venv/bin/python -r tools/requirements.txt
tools/.venv/bin/python tools/fetch_hf.py --repo Qwen/Qwen3.8-Flash-Next --out /mnt/data/fngine/hf
tools/.venv/bin/python tools/convert.py --gguf "$GGUF1" --ngram-gguf "$GGUF2" --hf /mnt/data/fngine/hf --out /mnt/data/fngine/pack
tools/.venv/bin/python tools/verify_pack.py --pack /mnt/data/fngine/pack --gguf "$GGUF1" --ngram-gguf "$GGUF2" --hf /mnt/data/fngine/hf
tools/.venv/bin/python tools/reference/run.py golden --out /mnt/data/fngine/ref
tools/.venv/bin/python tools/reference/run.py calibrate --out /mnt/data/fngine/ref
tools/.venv/bin/python tools/reference/make_fixtures.py --hf /mnt/data/fngine/hf --out tests/fixtures
tools/.venv/bin/pytest tools/tests
```

Golden/reference work streams model layers and is expensive. Supply the generated `routing-counts.npy` to conversion with `--profile /mnt/data/fngine/ref/routing-counts.npy`; it becomes `profile.bin`, the prompt (prefill) routing profile.

Two optional pack files tune decode. `decode-profile.bin` holds decode routing counts (u64 [48][512]) collected by generating 768 greedy tokens for each of 52 chat prompts (English coding and tool use, markdown documents and tables, Chinese, mathematics); expert placement ranks by it. `draft-vocab.bin` lists the 32,768 most frequent token ids of a local code and English corpus (all special and single-byte tokens included); the MTP drafter then scores only those `lm_head` rows. Drafts are verified exactly, so the vocabulary affects acceptance, never output. Corpus roots default to this workstation's source trees; pass `--root DIR` (repeatable) elsewhere.

```sh
tools/.venv/bin/python tools/draft_vocab.py --pack /mnt/data/fngine/pack
tools/.venv/bin/python tools/decode_prompts.py --out /mnt/data/fngine/prompts
FNGINE_NO_ROUTING_STATS=1 build/apps/fngine-validate --pack /mnt/data/fngine/pack --ref /mnt/data/fngine/prompts \
  --golden $(seq -s, -f 'p%02g' 1 52) --no-adaptive --generate 768 --decode-width 1 \
  --dump-decode-counts /mnt/data/fngine/pack/decode-profile.bin
```

### Abliterated packs

A pack may carry an optional `ablation` manifest section that names `ablation.bin`: FP32 unit refusal directions `[48][2560]` at `directions_offset` (4096-aligned) and, optionally, FP32 per-layer scales `[48]` at `scales_offset` (otherwise the section's `scale`, default 1). The engine then edits each main layer's **routed-expert** output along that layer's direction, `y_routed −= scale·r·(r·y_routed)`, inside the decode/verify and prefill MoE combine kernels, for GPU-resident and CPU-tier experts alike. The routed experts' packed bytes stay as they are: because the routed output is a weighted sum of linear down projections, this is exactly equivalent to orthogonalizing every expert's down projection (`W −= scale·r·rᵀW`). The shared expert is added afterwards unedited, and the MTP layer is never edited. The other residual writers (attention/GDN output projections, shared-expert down projections) carry the edit in `dense.bin` itself.

With ablation present, one CTA per token performs the combine so the dot product is a block reduction; packs without the section run the original kernels unchanged. `--no-ablation` (engine CLIs, `fngine-validate`) ignores the section, which is the control for A/B checks. `tools/reference/run.py golden --abliteration WORK` generates matching reference goldens: the edited dense tensors override the official ones, and the reference's GGUF expert sum receives the same edit.

**`/mnt/data/fngine/pack` is now the heretic pack** (2026-10-15): the decensoring of [trohrbaugh/Qwen3.8-Flash-Next-heretic-2](https://huggingface.co/trohrbaugh/Qwen3.8-Flash-Next-heretic-2) (Heretic, one direction per layer) transferred onto the fngine pack. Its 21 edited dense writers (attention/GDN output projections of layers 32–47, shared-expert down projections of layers 31–35) are in `dense.bin`; its routed-expert edit (layers 31–35, per-layer scales 0.45–1.71, 0 elsewhere) is in `ablation.bin`. It is published as [satellitedown/Qwen3.8-Flash-Next-heretic-fngine](https://huggingface.co/satellitedown/Qwen3.8-Flash-Next-heretic-fngine), whose `heretic/` folder records how it was built and verified; the installer serves it with this engine. Against the original model's goldens (fixed placement, no prefill streaming) it measured G2 top-1 0.8993 / KLD 0.0354, against 0.9106 / 0.0307 for the un-abliterated pack, and it refuses 1 of 32 mildly sensitive prompts (the un-abliterated pack: 8), with the same capability and speed. It replaced the huihui-ai abliteration (scale 1.955 in all 48 layers), which measured G2 0.7867 / 0.2040 against the original model and, by the two HF references, predicts ordinary text with 8–10% higher perplexity.

## CLI

```sh
build/apps/fngine --pack /mnt/data/fngine/pack --prompt 'Write a haiku about GPUs' --greedy --max-new 128 --stats
build/apps/fngine --pack /mnt/data/fngine/pack --prompt 'List three primes' --no-thinking --print-token-ids
```

Shared options include `--ctx 131072`, `--prefill-chunk 4096`, `--cpu-workers 0` (one expert worker per physical core; the engine and n-gram I/O threads run on the SMT siblings of the first core of each CCD), `--ngram-cache-mib 1024`, `--snapshots 6`, `--spec auto|mtp|lookup|none`, `--dense fp8|bf16`, and `--kv k8v8|k8v4|bf16`. **K8V8 is the current default**, superseding the original K8V4 plan. `--no-cuda-graph` selects eager decode. `--no-prefill-stream` disables PCIe expert streaming; `--no-borrow` reserves prompt workspace rather than borrowing resident expert slots. `--no-adaptive-experts` disables in-session expert swaps and the elastic expert slots in the unused context. `--no-ablation` ignores an abliterated pack's ablation section (see Abliterated packs). `FNGINE_NO_ROUTING_STATS=1` disables persistent placement telemetry for controlled comparisons.

Prefill keeps activations, dequantized expert weights and SwiGLU intermediates in BF16 on both CPU and GPU, with FP32 accumulation. CPU decode retains its Q8/VNNI kernels. The streaming cost model measures the active CPU prefill path and PCIe transfer path at startup. Per chunk and layer it streams the CPU-resident experts most frequent in prompt routing (the pack's prefill profile) while their PCIe time stays below the CPU time they save plus 2.5 µs per chunk token, the allowance for the layer's GPU attention and dense work that the copies also overlap. The ring has two buffers of 192 expert slots; a layer's copies run while the previous layer computes.

Borrowing uses a contiguous group of resident expert slots (filled at startup with the lowest-ranked residents) for the prompt workspace and the streaming ring: the experts in those slots keep RAM backups and are CPU-computed during prefill. The workspace view is sized for the prompt's largest chunk and final absolute KV frontier, including any reused prefix, rather than the configured context capacity. Only the experts whose slots prefill actually wrote (that view plus the ring slots used) are re-uploaded: in the engine test, short prompts re-uploaded 43–168 of 1,393 loan experts. The permanent decode workspace and captured graph pointers stay fixed; expert bytes and maps are restored before decode. If RAM registration fails, streaming, restoration and swaps use two pinned 64-MiB staging buffers instead of disabling the cold-tier transfers.

Decode and verify steps count routing per (layer, expert); prompt routing is not counted, because prompt text routes differently from the model's own output. The counts persist in `<pack>/routing-stats.bin` on shutdown and every ten minutes, with integer half-decay per flush. Startup ranks experts for GPU residency by decode routing frequency (`decode-profile.bin` plus `routing-stats.bin`, weight 0.8) and the prefill profile (weight 0.2, which covers routing the generation prompts miss, such as continuing raw documents).

Adaptive replacement runs every decode/verify step. A kernel queued behind the step folds the routing counters into a per-step EMA (decay 0.999; startup residents begin at 1) and reports each layer's hottest CPU expert and coldest GPU expert; at the step boundary, up to four layers whose pair clears a 2× EMA ratio start a swap, and the copies (low-priority stream) overlap the next step. Experts in backed slots (the loan group and the elastic slots below) keep their RAM blobs, so evicting one needs no download: it falls back to its RAM copy and the incoming expert is uploaded into its slot. Other victims go through one spare slot per layer: the incoming expert is uploaded into the spare, and the victim is copied back to RAM before its slot becomes the spare.

With adaptive replacement, the device state shares the expert pool's allocation, and the next-ranked experts after the regular residents occupy the unwritten tails of the KV planes and indexer keys of all 13 attention caches (elastic slots; 869 experts at `--ctx 131072`). A sequence writes these planes only up to its length; before a step or prefill chunk reaches an elastic slot's bytes, its expert returns to its RAM copy and the slot retires. A new or truncated sequence frees retired slots, and adaptive replacement refills them with the hottest CPU experts.

Decode and verify steps prefetch the next layer's dense weights into L2 (80 MiB by default, `FNGINE_L2_PREFETCH_MIB` overrides, 0 disables) on a parallel graph branch while the GPU waits for the CPU expert tier. The speculation controller measures step latency per verify width, excluding CUDA-graph capture, and extrapolates unmeasured widths from the measured slope (prior 0.25 of a decode step per extra token).

The single-sequence session cache uses pinned host snapshots at every 4096-token prefill boundary, prompt end and generation end. Prefix reuse is logged as `reused N tokens`. A reusable snapshot must leave at least one prompt token to prefill. With MTP the saved boundary's next token must also match: the MTP KV at position `p-1` depends on token `p`, so the safe current rule is `p <= min(common_prefix-1, prompt_length-1)`, not the plan's looser boundary rule. Position-addressed KV remains resident; recurrent states and n-gram history are restored from snapshots.

### Native 256K capacity

`--ctx 262144` enables the pack's native **262,144-token total context**, including generated output. The library/CLI default remains 131,072. No RoPE extrapolation or precision reduction is used; FP8 dense weights, the existing mixed expert pack and K8V8 KV remain unchanged. Values beyond the pack's native positional limit are rejected. The logical requested limit, not page padding, bounds generation; MTP proposals are omitted only at the final positions where their extra writes would exceed that limit.

At 256K, K8V8 position state reserves an additional 1.739 GiB and decode score scratch another 2 MiB. Existing elastic expert slots reclaim most unused context storage: the measured short-context resident count changed from 11,783 to 11,774 experts. As a real prompt grows, those slots return to KV storage; longer prompts still require more prefill work and may reduce expert residency.

Same-binary 128K/256K/256K/128K runs, one warmup and three measured repetitions per process, with the desktop running and routing-stat persistence disabled:

| Workload | 128K decode tok/s, mean ± sample SD | 256K decode tok/s, mean ± sample SD |
|---|---:|---:|
| Coding chat, turn 1 | 178.3 ± 6.7 | 191.2 ± 14.7 |
| Coding chat, turn 2 | 192.5 ± 10.1 | 197.3 ± 15.2 |
| Raw corpus, 64-token prompt | 158.7 ± 26.6 | 150.9 ± 14.8 |
| Raw corpus, 512-token prompt | 144.6 ± 41.5 | 145.5 ± 27.7 |
| Raw corpus, 4,096-token prompt | 152.8 ± 13.8 | 148.5 ± 19.2 |

No chat-generation slowdown was observed in this comparison. Short coding-chat prefill increased by 15–19 ms; total request time decreased in both chat cases. Raw-corpus results were mixed and variable: this is not a claim of universally faster or cost-free capacity. Full samples and commands are in `results/context-extension.json`.

Fresh SSE requests, with zero reused prompt tokens, recovered the planted code `COBALT-68157` at all three lengths:

| Request prompt tokens | Code's raw-document token offset | Prefill seconds |
|---:|---:|---:|
| 180,032 | 155,000 | 108.101 |
| 200,032 | 175,000 | 124.437 |
| 261,832 | 241,000 | 166.902 |

Each stream delivered usage and `[DONE]`, with keep-alive comments during prefill. A subsequent short arithmetic request returned `391`; an over-capacity prompt returned HTTP 400 with `context_length_exceeded`. These establish working long-context retrieval and boundary handling, not broad long-context quality or steady-state long-context decode speed.

## OpenAI-compatible server

```sh
build/apps/fngine-serve --pack /mnt/data/fngine/pack --ctx 262144 --host 127.0.0.1 --port 8002 --model-id qwen3.8-flash-next
curl http://127.0.0.1:8002/health
curl http://127.0.0.1:8002/v1/models
curl -N http://127.0.0.1:8002/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"qwen3.8-flash-next","stream":true,"messages":[{"role":"user","content":"List three prime numbers."}]}'
```

Supports non-stream and SSE Chat Completions, `reasoning_content`, function tools, developer messages, usage, and `enable_thinking:false`. Text-only: media inputs and unsupported multi-completion/concurrency controls are rejected. One request executes at a time, with a FIFO queue of eight pending requests by default. This is not a multi-user KV scheduler. Use loopback by default; do not expose an unauthenticated endpoint to an untrusted network.

OMP provider entry (merge into your own configuration; no OMP configuration is modified by fngine):

```json
{
  "api": "openai-completions",
  "baseUrl": "http://127.0.0.1:8002/v1"
}
```

For that 256K server, set the matching model entry in `~/.omp/agent/models.yml` to `contextWindow: 262144`; `maxTokens: 32768` can remain unchanged. The context limit includes the output allowance.

For a provider named `fngine`, merge this into OMP's **`~/.omp/agent/config.yml`**:

```yaml
providers:
  maxInFlightRequests:
    fngine: 1
```

This admission limit is shared across local OMP processes using the same config root. It keeps main turns, subagents and background requests from flooding a single inference worker. The server's default pending deadline is 30 seconds; without client admission control, a healthy long generation can cause queued requests to fail with `inference request expired while waiting in the queue` (503), or a full queue to return 429. Increasing the stream idle timeout does not fix this queue timeout. Serializing this provider's LLM requests preserves its existing one-at-a-time GPU execution; other providers and tool execution remain independent.

With that limit in a temporary OMP configuration and retries disabled, a real 200,217-token OMP request completed successfully. A second OMP process waited 128.1 seconds, then returned `OMP_FNGINE_OK`; the server admitted it only after the first request finished. Normal stream watchdog defaults were retained. This checks cross-process admission beyond the server's 30-second deadline without increasing that deadline; evidence is in `results/context-extension.json`.

Actual HTTP/model checks passed for nonthinking and developer messages, two-turn prefix reuse (42 cached tokens), SSE reasoning/content/usage, and a structured `get_weather` tool call. A 100,045-token request recovered the planted middle-of-document code `INDIGO-73921`; prefill took 111.985 s (893.4 tok/s). Evidence: `results/server-smoke.json`. These are smoke checks, not a broad quality benchmark.

`results/quality-smoke.json` records the real model's linked-list reversal function: it compiled with warnings-as-errors, AddressSanitizer and UndefinedBehaviorSanitizer, then passed pointer/value preservation and double-reversal checks for lengths 0 through 1,024. The arithmetic, prime-number and haiku samples were also coherent/correct for their prompts.

## Correctness contracts

```sh
build/apps/fngine-validate --pack /mnt/data/fngine/pack --ref /mnt/data/fngine/ref
build/apps/fngine-validate --pack /mnt/data/fngine/pack --ref /mnt/data/fngine/ref --dense bf16 --kv bf16 --dump-layers
```

`--dense bf16` reads `dense_bf16.bin`, the BF16 copies of the FP8 dense tensors. Nothing else uses that file, so serving installs may omit it even though the manifest lists it; the pack then loads without BF16 copies.

Original numerical targets on G1/G2: default top-1 agreement at least 96% and mean KLD at most 0.03; dense-BF16/KV-BF16 at least 98.5% and KLD at most 0.01. G0 BF16 per-layer stream cosine target is 0.999. Streaming prefill versus CPU-only prefill targets KLD at most 1e-3. These targets are not all met: BF16 mode measured G1 top-1 91.70% / KLD 0.04596 and G2 95.20% / 0.01006, with minimum G0 layer cosine 0.992795; split-path KLD measured 0.009466. The implementation retains the selected weight formats and reports these fidelity limits rather than weakening the validator or claiming a numerical pass.

Default FP8/K8V8 measured G1 top-1 82.81% / KLD 0.09811 and G2 90.95% / 0.03200 against HF BF16 in the latest run (`results/validation-default.json`; the previous build's run recorded 84.67% / 0.07333 and 90.78% / 0.03038). Prefill expert placement between CPU and GPU now follows the measured cost model, so separate runs round differently: four runs of the same build gave G2 90.86–91.20% / 0.0300–0.0320 and G1 79.3–85.0% / 0.093–0.133 (see Performance work below). `--dense bf16 --kv bf16` gives closer reference agreement in these measurements (`results/validation-bf16.json`) and consumes more VRAM, but also does not meet all original gates.

Speculative greedy output must match plain decoding over 256 tokens on three prompts, except positions where both top-two gaps are below 0.05; first MTP draft acceptance target is 0.55. After 50 mixed accept/reject rounds, committed recurrent/conv/KV/indexer state must agree within 1e-3 relative.

Validation defaults to the HF BF16 goldens specified by the build plan. `--ref-dtype fp32` selects separately generated precision-diagnostic goldens; those results do not replace the BF16-reference acceptance check. `--compare-logits FILE` additionally enforces the split-path KLD limit and records it in JSON. Timing and repeatability diagnostics do not report an accuracy pass. Forced-swap validation requires `--no-borrow --prefill-chunk 16 --swap-every-chunk`, because live prompt scratch cannot be used as swap storage.

Prefill chunks accept positive multiples of 16; the former multiple-of-64 restriction prevented the planned forced-swap check despite the workspace supporting 16 rows. The corrected path completed **21,248 expert swaps** across G0/G1/G2 with 16-token chunks. G2 measured top-1 90.92% / KLD 0.03039 (`results/validation-forced-swaps.json`): close to the default run, but still an explicit numerical validation failure.

Measured speculative verification is recorded in `results/speculation.json`: all three 256-token greedy comparisons were exact, first-draft MTP acceptance was 84.2%, and every compared committed state region had zero relative error after 50 mixed rounds. The small-width FP8 kernels use a consistent reduction order across decode and verify; kernel oracle coverage passed 265/265 cases. With the 32,768-row draft vocabulary (October 2026) the same check measures 74.1% first-draft acceptance (82.9% with the full `lm_head`): the code-and-documentation vocabulary misses names and rare words of the prose prompt (132/367 drafts accepted versus 161/282), while the drafter reads 80 MiB instead of 606 MiB per draft step; coding-chat decode is 4% (turn 1) and 20% (turn 2) faster with it. Greedy outputs stay exact either way, and a fourth check restores a mid-prompt session snapshot after 64 decoded tokens and requires byte-identical state and an identical continuation.

## Benchmark baseline

```sh
build/apps/fngine-bench --pack /mnt/data/fngine/pack -p1024,4096,32768 -n256 -r3 --spec auto -ojson -f results/baseline.json
build/apps/fngine-bench --pack /mnt/data/fngine/pack -p32768 -n256 -r3 --spec none --no-prefill-stream -o table
```

Reports prefill/decode tokens per second, decode routing-mass GPU hit rate, CPU wait microseconds per layer, accepted tokens per round separately for MTP and lookup, expert swaps, and n-gram cache hit rate. Repetitions disable prefix reuse. Decode throughput excludes the first generated token (charged to prefill); interval counters exclude warmup and preceding requests.

| Engine / evidence | Prompt tokens | Prefill tok/s | Decode tok/s |
|---|---:|---:|---:|
| Strata historical IQ3_S log, MTP + int8 KV | 44 | — | 104.5 |
| Strata historical IQ3_S log, MTP + int8 KV | 38,861 | 2,507.9 | 108.0 |
| fngine, FP8 + K8V8, auto speculation | 1,024 | 870.4 | 100.0 |
| fngine, FP8 + K8V8, auto speculation | 4,096 | 1,549.1 | 102.4 |
| fngine, FP8 + K8V8, auto speculation | 32,768 | 1,732.7 | 92.9 |

fngine figures are means of three repetitions, with one warmup, from `results/baseline.json`. The corpus is cinference's README and documentation. GPU routing-mass hit rates were 83.8%, 83.8% and 81.9%; CPU wait was 147, 152 and 165 microseconds per layer respectively. `results/baseline-environment.json` records the 575-W power cap, x4 PCIe link and 1,286 MiB pre-run GPU usage. Persistence was disabled during measurement; in-session adaptive swaps remained enabled.

Strata numbers are historical on this workstation, **not apples-to-apples measurements**: prompt/corpus, cache residency, KV representation, GPU load and sampling can differ. Record a new paired run before claiming a speedup.

With speculation and adaptive swaps disabled, three paired 32,768-token runs measured **1,866.1 tok/s split prefill versus 1,505.5 tok/s CPU-only cold-expert prefill** (about 24% faster), using the same corpus, chunk size and warmup policy. Evidence: `results/prefill-split.json` and `results/prefill-cpu.json`. Resident experts still run on the GPU in both modes. Desktop VRAM variation changed residency by 27 experts; decoded continuations also differed, so their decode rates are not a matched-workload comparison.

For the two-turn coding workload, run:

```sh
FNGINE_NO_ROUTING_STATS=1 build/apps/fngine-bench --pack /mnt/data/fngine/pack --coding-chat -n512 -r1 --warmup 0 --spec none -ojson -f results/adaptive-on.json
FNGINE_NO_ROUTING_STATS=1 build/apps/fngine-bench --pack /mnt/data/fngine/pack --coding-chat -n512 -r1 --warmup 0 --spec none --no-adaptive-experts -ojson -f results/adaptive-off.json
```

`--coding-chat` asks for singly linked-list code, then adapts the actual first answer to a doubly linked list. Each turn uses the normal benchmark token limit and reports separate counters; `n_prompt: 0` denotes a variable chat prompt, with actual lengths in each repetition. Prefix reuse is disabled. Repetitions retain in-session cache adaptation; use zero warmups and one repetition to measure the first two turns from startup.

The recorded short-chat run **did not show an adaptive-cache speedup**:

| Cache policy | Turn 1 decode tok/s | Turn 1 GPU routing mass | Turn 2 decode tok/s | Turn 2 GPU routing mass | Swaps |
|---|---:|---:|---:|---:|---:|
| Static | 108.0 | 86.70% | 103.1 | 84.95% | 0 |
| Adaptive | 102.0 | 85.60% | 99.7 | 84.92% | 96 |

These are single-run observations, not statistically established differences. Adaptive mode began with 65 fewer resident experts because of spare slots and desktop VRAM variation, and generated continuations can differ across CPU/GPU placements. The per-step adaptive placement of the fourth pass (see Performance work) decodes this workload 9–16% faster than static placement at width 1 over three interleaved rounds.

## Performance work (October 2026)

The tables above describe the engine before this work. Paired runs of the previous build (source snapshot `baseline-v1`) and this tree used the same pack, machine state and commands: `FNGINE_NO_ROUTING_STATS=1`, no `routing-stats.bin`, three repetitions after one warmup, 575-W cap, PCIe x4, desktop compositor running on the same GPU (see the remaining bottlenecks). Evidence: `results/perf-2026-10/` (the first, second and third passes' pairs are in `round1/`, `round2/` and `round3/`).

```sh
export FNGINE_NO_ROUTING_STATS=1
build/apps/fngine-bench --pack /mnt/data/fngine/pack -p1024,4096,32768 -n256 -r3 --spec auto -ojson -f current-corpus.json
build/apps/fngine-bench --pack /mnt/data/fngine/pack --coding-chat -n512 -r3 --spec auto -ojson -f current-coding.json
build/apps/fngine-bench --pack /mnt/data/fngine/pack --coding-chat -n512 -r3 --spec auto --ctx 131072 -ojson -f current-ctx131072.json
build/apps/fngine-bench --pack /mnt/data/fngine/pack -p64,512 -n64 -r3 --spec auto -ojson -f current-short.json
build/apps/fngine-bench --pack /mnt/data/fngine/pack -p1024 -n256 -r3 --spec none -ojson -f current-nospec.json
```

| Workload (auto speculation unless noted) | Prefill tok/s, before → after | Decode tok/s, before → after | Decode GPU routing mass |
|---|---:|---:|---:|
| Coding chat, turn 1 (512 tokens) | 110.0 → 195.5 | 116.8 → **181.2** | 89.9% → 97.5% |
| Coding chat, turn 2 | 511.5 → 714.3 | 104.9 → **196.4** | 87.9% → 97.3% |
| Coding chat at `--ctx 131072`, turn 1 | 99.6 → 194.3 | 99.7 → **187.5** | 87.5% → 97.7% |
| Coding chat at `--ctx 131072`, turn 2 | 467.6 → 715.1 | 103.3 → **212.9** | 86.8% → 96.9% |
| Corpus, 64 + 64 tokens | 116.4 → 192.0 | 107.5 → 158.0 | 87.2% → 90.0% |
| Corpus, 512 + 64 | 558.9 → 574.1 | 89.3 → 124.0 | 89.3% → 94.3% |
| Corpus, 1,024 + 256 | 876.1 → 1,107.9 | 108.5 → 146.8 | 90.0% → 91.6% |
| Corpus, 4,096 + 256 | 1,560.5 → **2,935.3** | 92.2 → **172.2** | 84.6% → 92.4% |
| Corpus, 32,768 + 256 | 1,714.5 → **3,287.0** | 94.4 → **155.5** | 84.8% → 88.2% |
| Corpus, 1,024 + 256, no speculation | 925.9 → 1,099.0 | 83.9 → 120.1 | 87.6% → 95.3% |

Chat decode, the only decode workload of the server API, is 55–87% faster in this pair at the bench's automatic context and 88–106% faster at the server's default `--ctx 131072`, where the previous build loses experts to the reserved context (the third pass's pair measured 68–69%, 175.0 and 168.8 tok/s; the second pass's 59–74%, 180.5 and 179.5 tok/s; the first pass's 155.0 and 172.4). The corpus rows decode a raw continuation of cinference's documentation (the 4,096-token slice ends inside a table of numbers). Adaptive placement now follows that routing as well (92.4% routing mass at 4,096 + 256, against 71.5% in the third pass), and these rows decode 35–87% faster. Prefill is 3–92% faster. The 64-token prefill depends on how many experts the startup cost measurement streams, so it varies between processes of one build. Rows move by up to 15% between sessions for both builds; compare within a pair.

Changes, each measured on its own:

- **CPU prompt-expert dequantization.** `decode_rows_bf16` expands each row with one `decode_rows` call into an L1 scratch and undoes the VNNI lane order with register shuffles, instead of a call and gathers per 256-element group. The BF16 values are bit-identical (exact-dequantization test). One core went from 4.1 to 10.5 G weights/s for IQ3_XXS and from 6.1 to 22.4 for IQ4_NL; the pool's 32-token prefill went from 193 to 136 µs per expert (layer 2); the measured B_cpu rose from 13 to 22 GB/s and 32K prefill from 1,463 to 1,775 tok/s.
- **Stream planning by prompt routing.** Streamed experts are ranked by the prefill profile instead of the decode-oriented placement score: 32K prefill 1,775 → 2,101 tok/s.
- **Ring geometry and copy overlap.** Two buffers of 192 slots replace four of 64, and copies may exceed the CPU time they save by 2.5 µs per chunk token, because they also overlap the layer's GPU attention and dense work. With 192-slot buffers, 32K prefill measured 2,509 tok/s with no allowance, 2,548–2,669 with 1–3 µs per token and 2,506 with 4 µs.
- **CPU cost model.** F_cpu is the slope between 1 and 128 tokens. At 32 tokens it measured 3.6–10.6 TFLOP/s across starts, above the roughly 7 TFLOP/s BF16 peak of the 10 worker cores; at 128 tokens it measures 4.5–4.9.
- **Loan restore of written ranges.** The borrowed workspace view is sized per prompt and only experts under written bytes are re-uploaded: 64-token prompts restore in 28 ms instead of 251 ms (prefill 118 → 200 tok/s); 2,048 tokens 150 ms; 8,192 tokens the full 240 ms.
- **L2 prefetch.** Width-1 decode on G1 with fixed placement: 9.47 → 8.92 ms per step; prefetching more than 80 MiB per layer stopped helping.
- **Decode-driven placement.** Held-out chat prompts p17–p20 (768 tokens each, width 1, no speculation, fixed placement) decoded at 70.2 tok/s on average under the previous placement and 90.3 tok/s under a profile of 36 other prompts. The shipped 0.8/0.2 weighting with the 52-prompt profile covers, on held-out routing counts, 85.8% of chat-decode routing (−0.5 points versus 0.95/0.05) and 81.6% of document-continuation routing (+1.8 points).
- **MTP draft head.** The drafter scores 32,768 draft-vocabulary rows (80 MiB) instead of `lm_head` (606 MiB) per draft step; see the acceptance trade-off under correctness contracts.
- **Speculation controller.** Graph capture is no longer timed as a step, so a newly captured width is not mistaken for a slow one, and unmeasured widths extrapolate from the measured slope (prior 0.25 instead of 0.6 of a decode step per extra token).

Decode kernels and scheduling (second pass). Teacher-forced timings use fixed placement (`--expert-vram-mib 22000 --no-adaptive`), where routing and therefore the CPU tier's work are identical between the compared builds:

- **Fused GR read.** `gr_norm_down` computes the four-stream RMS norm and the 10,240 → 320 down projection in one kernel (clusters of four CTAs, one per stream; partials reduced in rank 0 through distributed shared memory): 8.6 → 5.7 µs at one token, 10.4 → 7.9 µs at four.
- **Programmatic dependent launch.** The decode-width linear kernels (GEMV, SIMT, K-split MMA), the fused GR read and the hot-expert down projection launch as programmatic dependents (`kernels/common/pdl.cuh`): each requests its first weight codes, then waits for its predecessor. Reads of a predecessor's output go through `pdl_coherent`, because ptxas schedules `ld.global.nc` loads (what a `const __restrict__` pointer compiles to) above `griddepcontrol.wait`; without it the GR read raced at width 4. Width 4: 22.79 → 22.26 ms per step over three paired rounds; width 1: 1–2%.
- **Auxiliary stream.** GDN `ab` + gating, the QSA indexer projection and preparation, and the MoE shared expert (forked after routing) run on a second stream one priority level below the step stream.
- **Routing selection.** `select_publish` also counts routing (replacing a separate kernel), only threads that wrote host memory fence before the ready flag, and the top-10 selection no longer spills to local memory: 8.3 → about 5 µs per layer.
- **Hot experts.** The gate/up kernel stages its codebook in shared memory, and the down projection streams its weights while gate/up finishes (PDL). Gate/up + down for ten experts at one token: IQ3_XXS 21.6 → 19.7 µs, IQ2_S 21.6 → 20.5, IQ3_S 23.6 → 22.6, IQ4_XS 25.0 → 23.6.
- **CPU placement.** One expert worker per physical core (12 instead of 10); the engine and n-gram I/O threads run on SMT siblings. Coding-chat decode, same build otherwise: 171.7 / 159.5 → 179.8 / 173.7 tok/s (turn 1 / turn 2). Using the remaining SMT siblings as ten more workers made width-1 steps slower (11.2–11.9 → 14.4–14.9 ms).
- **Routing statistics** are collected once per adaptive planning period rather than every step.

Measured without gain and not kept: warming the next layer's cold experts into CPU caches (even with perfect knowledge of them), an LRU VRAM cache of cold experts (about 56 MB of promotions per step), L2 prefetch of the down matrix from the gate/up CTAs, and moving the n-gram gather off the step thread (at most about 30 µs per step).

Fidelity is unchanged within run-to-run variation (`results/perf-2026-10/fidelity.json`). G2 (8,192 positions, FP8/K8V8, teacher-forced against HF BF16): previous build 91.02–91.21% top-1 and 0.0300–0.0321 mean KLD over four expert-placement caps; this build 90.86–91.20% and 0.0300–0.0320 over four runs. G1 swings more within one build than between builds (previous build 76.3–89.6% / 0.062–0.179 across caps; this build 79.3–85.0% / 0.093–0.133 across identical runs; two runs of the same configuration differed by 0.013 mean KLD). The decode path with fixed placement measured G2 91.03% / 0.0302 before and 91.06% / 0.0309 after, and widths 1 and 4 give identical logits. Streaming no longer stops at 64 experts per layer, so the CPU/GPU split follows the startup cost measurement and separate processes round prefill differently; within a process plans are cached per chunk width and repeat exactly.

After the second pass: G0 99.08% / 0.0082, G1 82.03% / 0.0970, G2 90.93% / 0.0303 (default), and decode-path G2 91.05% / 0.0323 at widths 1 and 4 with identical logits (top-1 agreement 1.0, mean KLD 4e-13). The fused GR read accounts for the decode-path change from 91.06% / 0.0309: it moves logits by a mean KLD of 1.1e-2, the same as reordering the unfused kernel's warp reduction. Programmatic dependent launch leaves the logits bit-identical. Speculative greedy output stays exact on all three prompts, with first-draft acceptance 75.6%.

Decode scheduling (third pass), same fixed-placement method:

- **Programmatic dependent launch on the remaining decode kernels:** the GDN convolution, staged recurrence, output norm and gating, the BF16 router GEMV, routing selection, the hot-expert gate/up projection and the combine, and on the QSA side q/k norm + RoPE, KV append, the indexer (prepare, ring update, score, select) and attention with its split merge. Before `pdl_wait()` a kernel now reads only immutable data (weights, codebooks, scales): a back-to-back launch of the same kernel writes the recurrent state and the KV cache, and reading the GDN state early failed the multi-step recurrence test until the wait moved ahead of it. Width 1 on G1 in interleaved rounds: 12.75–12.84 → 12.56–12.63 ms per step; width 4 was not resolvable under the time-slice noise described below. Logits stay bit-identical.
- **Two-stage K-split for 8 tokens at K = 2,560** (`k2560_store.cuh`: `gdn.qkvz`, `qsa.qkv`, `ple.key`, `ple.value`, the MTP fc projections), same reduction order: 16,384 × 2,560 29.8 → 28.7 µs, 10,240 × 2,560 19.5 → 18.6 µs, 2,560 × 2,560 6.9 → 6.4 µs; 13,312 × 2,560 unchanged.
- **One host poller per CPU-tier wait.** GPU reads of one host cache line serialize: a flag round trip takes 1.5 µs with one polling CTA, 3.9 µs with five and 16.8 µs with twenty (27 µs while the CPU streams DRAM). Every combine CTA used to poll `done[l]` (3 per token in decode, up to 48 at width 16, and 3 per chunk token in prefill); now CTA (0, 0) polls and relays the outcome through device memory (`exchange_wait_sys_relayed`). Inside a full step the gain is below the time-slice noise.

Measured without gain and not kept: CCD-affine CPU task queues, CPU prefetch distances of 1–8 KiB, a three-stage K-split for `out_proj` (2,560 × 6,144), 256-thread hot gate/up CTAs (12.6 → 13.7 µs), and a non-temporal or flushed `done` store or slower polling.

After the third pass: decode-path G2 logits are bit-identical to the second pass at width 1 (width 4: top-1 agreement 1.0, mean KLD 4e-13). Two processes of the default path measured G0 98.15% / 0.0083, G1 83.30% / 0.107 and 81.69% / 0.134, G2 90.77% / 0.0320 and 90.64% / 0.0301; within one process, repeated prefills are bit-identical at every layer. Speculative greedy output stays exact on all three prompts, with first-draft acceptance 74.0–75.0%.

Expert residency (fourth pass), coding chat with automatic speculation unless noted; measurements in `results/perf-2026-10/round4.json`:

- **Per-step adaptive swaps, planned on the GPU.** The previous policy collected the routing counters on the host every 256 decoded tokens and planned up to 32 swaps per period, and loan slots never moved. A recorded routing trace of it measured 96.2 cold experts and 196 MB of CPU-tier weight reads per step (6.8% of routed pairs). Replayed against the engine's swap timing, per-step planning predicted 56–59 cold experts per step. Now a kernel queued behind every decode/verify step folds the counters into a per-step EMA (decay 0.999; startup residents start at 1) and writes each layer's hottest CPU expert and coldest GPU expert to mapped host memory. The step boundary starts up to four swaps, at most one per layer, whose EMA ratio is at least 2, and their copies start after the next step's launch, so they overlap it. With a budget of eight the engine measured 50.7 cold experts per step (103 MB, 3.9% of routed pairs); with the shipped four, width-4 steps averaged 1.56 instead of 2.49 cold experts per layer. At width 1 (no speculation), adaptive placement decodes 9–16% faster than static placement (130.0 / 131.2 against 119.2 / 113.1 tok/s, turn 1 / turn 2, three rounds each). Budgets of 4, 8 and 16 were within run-to-run variation; 4 keeps PCIe bursts smallest. A first version copied the counters to the host each step and waited 486 µs per step, because the copy queued behind 2 MB swap downloads on the copy engine; the step boundary now takes 6–16 µs.
- **Backed slots.** Experts in the loan group and in elastic slots keep their RAM copies, so evicting them needs no download: the victim falls back to its RAM copy at the boundary that starts the swap, and the incoming expert is uploaded into its slot. Only regular victims use the layer's spare.
- **Elastic context.** `--ctx 131072` (the server default) reserves 1.6 GiB for the KV planes and indexer keys of the 13 attention caches. Without elastic slots the GPU held 861 fewer experts than at the bench's automatic 8,192, and coding chat decoded 176.7 / 183.3 instead of 197.5 / 210.0 tok/s. With 869 elastic experts in the unwritten tails it decodes 197.5 / 198.4 against 195.6 / 206.2 at 8,192 (two interleaved rounds each; turn-2 routing mass 97.3%, previously 95.9%).
- **Step inputs and outputs** move between device memory and mapped host memory by a small kernel instead of memcpy nodes: a 1 KB copy queued behind 2.2 MB swap uploads on the copy engine took 33–50 µs instead of 2–3 µs.
- **L2 prefetch after the shared expert.** The prefetch branch now forks after the shared expert (on the auxiliary stream) has joined, so the two no longer compete for DRAM: teacher-forced G1 at width 3, 12.48 → 11.95 ms per step; width 1 unchanged (8.78 against 8.86).
- **Kernel preloading.** The swap kernels are loaded at startup. Their first launch is the first kernel issued while a step graph runs, and CUDA's lazy module loading waits for the device there, which deadlocked against the step's spinning exchange wait until the n-gram row timeout.

Evaluated and not changed: limiting MTP to two drafts (coding chat 195.4 / 199.2 → 178.7 / 180.1 tok/s, so the controller's width-4 choice holds); skipping the MTP drafter in prompt-lookup rounds (the drafter costs 0.74–0.78 ms per step and coding chat chose lookup in 1–5% of rounds); verify trees (an estimate with the controller's value model: measured first-draft acceptance 0.73, the controller's priors 0.6 / 0.5 for later drafts, an assumed 0.85 top-2 hit rate for the first draft, and an extra verify token costing about 0.19 of a width-1 step as measured on G1, so a top-2 first level would add about 10% tokens per round for about 36% more step time); and removing the CPU tier's barrier between its gate/up and down phases (a CPU-active layer with 1.52 cold experts takes 72 µs against a 63 µs DRAM floor, which bounds everything outside the transfer at 2–3% of a step). Through the server with thinking, sampling (temperature 0.6, top-p 0.95, top-k 20) decoded as fast as greedy, 167–188 tok/s, with 51–66% of drafts accepted against 55–72%.

After the fourth pass: decode-path G2 logits are bit-identical at width 1 (width 4: top-1 agreement 1.0, mean KLD 4e-13). One process of the default path measured G0 98.77% / 0.0089, G1 85.21% / 0.0826, G2 91.10% / 0.0295. Speculative greedy output stays exact on all three prompts, with first-draft acceptance 72.9%.

Aggressive kernel and speculation experiments (fifth pass), with raw measurements in `results/perf-2026-10/round5.json`:

- **Exact-width GR norm/down specialization.** The existing one-/two-token kernels now expose their exact token count to the compiler and allocate only that many columns of shared scratch. The launch geometry, weight reuse, arithmetic order and wider-token path stay unchanged. Across 64 rotating weight sets, one token measured 3.206 → 2.884 µs and two tokens 4.510 → 3.934 µs; four tokens remained effectively unchanged at 5.528 → 5.502 µs. Complete outputs at widths 1, 2, 3, 4, 8 and 16 are byte-identical to the fourth-pass kernel.
- **Rejected despite faster microbenchmarks:** splitting GR tokens across more clusters and skipping empty QSA split-workspace traffic. The combined version regressed small-context speculative chat; GR token tiling alone also regressed corpus runs. Wider GR batches therefore retain the original register-held weight reuse rather than duplicating that work across token tiles.
- **Other measured rejections:** exact integer-tensor-core MoE down projections, separately quantizing the MoE intermediate once, alternate IQ sign decoding, a fourth MTP draft, a 65,536-entry draft vocabulary, the full draft vocabulary, and empirical conditional-acceptance priors. None justified replacing the existing path. The original model pack, three-draft limit, 32,768-entry draft vocabulary, precision and numerical thresholds are unchanged.
- **Verification fixes:** the elastic-expert test now reads its asynchronous slot-map updates on the producer stream before asserting; no inference synchronization was added. QSA tests poison scratch and cover the first token and a 32-key chunk boundary.

Final paired decode rates (tok/s, mean ± sample SD; six measured repetitions per build and case, baseline/current then current/baseline, one discarded warmup per process):

| Workload | Fourth pass | Fifth pass |
| --- | ---: | ---: |
| Coding turn 1, 8K capacity | 184.0 ± 12.9 | 204.9 ± 12.7 |
| Coding turn 2, 8K capacity | 181.3 ± 5.4 | 204.0 ± 22.1 |
| Coding turn 1, 128K capacity | 187.9 ± 13.2 | 185.4 ± 12.1 |
| Coding turn 2, 128K capacity | 195.1 ± 13.3 | 185.8 ± 9.4 |
| Corpus, 1K prompt | 151.7 ± 29.1 | 138.2 ± 23.7 |
| Corpus, 4K prompt | 142.1 ± 18.0 | 145.9 ± 10.5 |
| Corpus, 32K prompt | 95.8 ± 12.6 | 115.8 ± 12.2 |
| No speculation, 1K prompt | 117.3 ± 13.4 | 116.8 ± 17.5 |

The mixed signs and process variance do **not** establish an across-the-board whole-model speedup. The retained claim is the exact-width kernel improvement, not a peak token-rate result or a language-rewrite gain. Corpus cases share one process configured for the largest prompt; coding cases use 512 generated tokens, corpus/plain cases 256. The desktop remained running.

Final verification: 44/44 C++/CUDA tests and 23/23 Python tests passed; CUDA memcheck reported zero errors for GR and QSA attention. Against the frozen fourth-pass build, G2 at widths 1, 2 and 4 has top-1 agreement 1.0 and mean/max KLD 0 over 127 saved full-logit rows per width. Three 256-token speculative continuations match plain greedy output exactly; mixed acceptance/rejection state checks and session restoration pass. The pre-existing HF-reference numerical gates still fail, with thresholds unchanged.

Hot-expert streaming and step-tail overlap (sixth pass), against the frozen fifth-pass build; numbers and raw outputs in `results/perf-2026-10/round6.json` and `round6/`:

- **Hot experts.** Each fifth-pass hot-expert CTA staged one whole tile and then decoded it: a width-4 layer launched 3,200 gate/up and 1,600 down CTAs. Alone at width 4, gate/up streamed 1.01–1.29 TB/s and down 0.88 TB/s; a gate/up variant that only staged its weights (no decode) reached 1.45–1.6 TB/s against 1.0–1.16 TB/s for the full kernel in the same measurement, so DRAM sat idle while the CTAs decoded and between waves.
  - **Persistent hot_down.** As many 128-thread CTAs as fit on the SMs take contiguous ranges of (expert, 32-row) tiles and keep two staged (cp.async double buffering), so the next tile streams in while the current one is decoded. A CTA quantizes an expert's h rows once for all its tiles of that expert, instead of once per 64-row tile, and triggers its dependent first thing. Alone: 29.3–29.4 → 19.7–20.1 µs at width 4, 11.8–12.1 → 7.5–7.8 µs at width 1.
  - **Streaming hot_gate_up.** Its decode needs several CTAs per SM (a double-buffered persistent version slowed width 4 from 30.2 to 37.8 µs), so each persistent CTA keeps one tile buffer and streams it in five K slices: once slice i of the current tile is decoded, slice i − 1 of the next tile is requested into the space of the current tile's slice i − 1. The one-slice gap keeps the refill clear of the slices still in flight for any 16-byte row alignment (`warp_stage_slice`). Alone at width 4: 30.2 / 29.0 / 34.1 / 37.9 → 28.4 / 27.5 / 31.9 / 35.5 µs (L0 / L2 / L17 / L47); width 1 is 0.1–0.3 µs slower.
  - **Combine beside hot_down.** The CPU-tier wait and the PCIe read of the CPU experts' result run before `pdl_wait()`, so the combine waits for the CPU tier while hot_down runs. A capture of the same launch sequence (shared expert on the side stream, joined before the combine) keeps both edges into the combine programmatic.

  Decode MoE pipeline per step (`kernels_moe_experts --bench-only`: select, gate/up, down and combine; 16 steps per graph so the weights stream from DRAM; minimum of 300 replays; the fifth-pass kernels in an earlier build of the same bench):

  | Layer formats | Width 1, 10 experts | Width 4, 28 experts / 40 pairs |
  | --- | ---: | ---: |
  | L0 IQ3_XXS/IQ4_NL | 26.9 → 25.2 µs | 61.5 → 52.1 µs |
  | L2 IQ2_S/IQ4_NL | 26.9 → 24.1 µs | 60.2 → 50.7 µs |
  | L17 IQ3_S/IQ4_NL | 28.7 → 26.0 µs | 65.3 → 55.5 µs |
  | L47 IQ4_XS/IQ4_NL | 30.1 → 26.8 µs | 69.1 → 60.0 µs |

- **GR norm/down preloads.** The fused kernel read the inject rows (cluster 0) and the down row scales (rank 0) after `pdl_wait()`, that is, after the previous kernel finished. Both now load with the weights. GR read pair (norm/down + up, fifth-pass and current `gr.cu` in one harness, two interleaved runs): width 4 9.70 → 9.07–9.08 µs with L2-resident weights and 10.80–10.81 → 9.25–9.27 µs across 16 rotating weight sets; width 1 unchanged with L2-resident weights (5.35–5.36 → 5.35–5.38 µs), 6.12–6.13 → 5.99–6.01 µs across 16 sets.
- **Verify folds beside the MTP drafts.** The GDN fold reads and writes back 113 MB of fp32 recurrent state (36 layers × 48 heads × 128 × 128), a median 110 µs per width-4 verify step. It and the PLE fold only feed the next step, and the MTP drafts (one QSA layer without PLE) read neither state, so both now run on the prefetch stream beside the drafts, the GDN fold on 64 CTAs that sweep its 13,824 tiles. nsys medians (coding chat, 720–738 width-4 verify steps per build): from the end of `lm_head` to the end of the step 820 µs in the fifth pass and 756 µs now; the draft chain took 679 and 735 µs, as it now shares DRAM with the fold. With 170 CTAs the fold took 145 instead of 461 µs, but the drafts 810 µs and the step tail 832 µs.

Paired runs, `--ctx 262144`, automatic speculation, a fifth-pass then a sixth-pass process per round, one warm-up and two measured repetitions per process (decode tok/s, mean ± SD; time per speculative round):

| Workload | Fifth pass | Sixth pass | Time per round |
| --- | ---: | ---: | ---: |
| Coding turn 1 (n = 12) | 176.5 ± 15.3 | 182.6 ± 12.9 | 14.74 → 13.70 ms |
| Coding turn 2 (n = 12) | 191.7 ± 10.8 | 209.5 ± 18.9 | 14.59 → 13.38 ms |
| Corpus, 1K prompt (n = 6) | 143.6 ± 21.6 | 147.3 ± 7.0 | 17.40 → 17.00 ms |
| Corpus, 4K prompt (n = 6) | 138.8 ± 15.1 | 135.6 ± 9.7 | 20.80 → 21.11 ms |
| Corpus, 32K prompt (n = 6) | 114.0 ± 19.1 | 124.4 ± 10.4 | 18.75 → 15.55 ms |

The speculation controller sizes each round from measured step latencies, and default-path numerics vary between processes of one build (the golden runs below), so generated text and round counts vary (coding turn 1: 172–262 rounds). The 1K and 4K corpus differences are within that variance, and prefill moved within noise. A scratch build that added 10 µs at the end of every hot_down call decoded 9.3% and 6.9% fewer coding tokens per second with the default adaptive placement (n = 6), but +1.8% and +2.1% with fixed placement (`--no-adaptive-experts`, predicted GPU hit rate 79% instead of 85%; n = 4). That is consistent with hot-expert time mattering where a layer does not wait for the CPU tier anyway (`round6/hot-down-delay.json`).

Measured and not kept: uniform 1-, 2- or 3-slice refill windows for the streaming gate/up (slower than the eager refill), forcing 3–6 gate/up CTAs per SM, three or four hot_down stages, hot_down reading a q8 copy of h from global memory, copying the x rows to the host early in select (select with a host exchange 4.7 → 4.8 µs), and the 170-CTA fold.

Sixth-pass verification: 46/46 tests passed (`kernels_gdn_fold` now also folds with a 37-CTA swept grid, bit-exact). In `test_speculative_real`, greedy decoding with automatic speculation produced the same 256 tokens as plain greedy decoding for all three prompts. After 50 verify rounds (37 with a rejected draft), the GDN recurrent, GDN conv and PLE conv states, the indexer rings and the KV equal a plain decode exactly. Against the frozen fifth-pass build, decode-path logits with fixed placement are identical (top-1 agreement 1.0, mean and max KLD 0) for G1 at widths 1 and 4 and for G2 at width 4. Default path (adaptive placement): fifth pass G1 87.6% / 0.063 and 89.4% / 0.061, G2 91.56% / 0.0266 and 91.58% / 0.0265; sixth pass G1 87.0% / 0.078, 89.5% / 0.055 and 90.1% / 0.061, G2 91.25% / 0.0263, 91.30% / 0.0266 and 91.48% / 0.0267. The HF-reference gates still fail, with thresholds unchanged.

Remaining bottlenecks on this machine:

- **GPU time slicing with the desktop compositor (largest).** The 5090 also drives the 3,840 × 2,160 desktop. Hyprland submits GPU work about 300 times per second, also with the monitor off (DPMS), and each submission takes the GPU from the CUDA context for one ~1.2 ms time slice: a spinning one-thread kernel is descheduled 19–27% of the time, and a watchdog kernel inside fngine's context measured 20–29% during decode. Pausing Hyprland (`SIGSTOP`) removes every gap; pausing Xwayland, nautilus, quickshell or cinmux changes nothing. With Hyprland paused only while the timed steps ran, G1 decode with fixed placement took 8.48 instead of 11.70–12.09 ms per width-1 step and 17.38 instead of 24.17–24.42 ms per width-4 step (28–30% less); the CPU tier also ran 13–17% faster (NVIDIA's interrupts land on CPU 10, a worker's core, and on CPU 12, the SMT sibling of worker 0). Driving the monitor from the Ryzen's integrated Radeon (a motherboard video output) leaves the 5090 to CUDA; otherwise, find what makes the compositor submit continuously. Every throughput number in this README was measured with the compositor running. Evidence: `results/perf-2026-10/compositor.json`.
- **Host DRAM speed.** The installed kit (Corsair CMH64GX5M2D6000Z40) carries DDR5-6000 CL40 XMP/EXPO profiles in its SPD, yet reads measure 49–57 GB/s, which suggests it runs at JEDEC speed (check with `sudo dmidecode -t memory`). The decode CPU tier is bound by this bandwidth.
- **GPU memory clock:** under CUDA load the card runs in P1 with memory at 13,801 of 14,001 MHz (−1.4%).
- **PCIe x4** at ~11.8 GB/s (DMA 11.3 GB/s host-to-device, 13.1 device-to-host; zero-copy reads 9–12 GB/s): long-prompt prefill streams ~12.5 GiB per 4,096-token chunk at roughly 70% link utilization. On this board (MSI PRO X870E-P WIFI) `PCI_E1` is CPU PCIe 5.0 x16 without lane sharing and the root port advertises x16, yet the card trains at x4: check seating, the slot and any riser. An x16 link mainly speeds prefill streaming, swaps and loading; decode moves only a few tens of KB per layer across the link.
- **CPU prompt experts:** BF16 GEMM at ~4.5 TFLOP/s and dequantization at ~22 GB/s, measured on 10 Zen 5 cores; the cost model predicts about 1 s of CPU expert time per 4,096-token chunk.
- **Decode CPU tier:** bound by host DRAM, which reads at 49–55 GB/s here (12–24 threads), at 35–65 µs per cold expert. Adaptive placement now keeps the GPU's wait for it at 43–47 µs per layer in coding chat with automatic speculation (the previous build: 150–175 µs) and 13 µs at width 1, about a sixth of a step: 39–50% of the layers of a width-3 or width-4 step have no cold expert, and the others average 2.0–2.6 (the previous policy: 25–32% and 2.6–3.3). With fixed placement on G1 at width 1, a CPU-active layer has 3.04 cold experts and takes 118 µs with the compositor paused (57 GB/s), 128–143 µs with it running.
- **Decode step:** 8.2–8.9 ms per width-1 step on held-out chat prompts (p17–p20, default placement without swaps) after the second pass, CPU-tier waits included; single repetitions after the third pass measured 7.5–10.1 ms under the compositor's time slicing. Kernels inside a step run back to back (idle gaps of about 1 µs or less, about 0.2 ms per step in total); the host gap between steps is about 42 µs.
- **Prompt-loan restore** after a full 4,096-token chunk: ~0.24 s.

