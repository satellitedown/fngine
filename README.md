# fngine

A C++20/CUDA inference engine that runs **Qwen3.8-Flash-Next** on one RTX 5090 (32 GB) with 64 GB of system RAM, with up to 256K context.

The model doesn't fit in VRAM, so fngine splits it:

- **Experts:** about half of the 24,576 routed experts sit on the GPU. The rest stay in RAM and run on the CPU (AVX-512), or stream over PCIe for long prompts. Placement follows what you actually use and adapts every decode step.
- **Everything else:** FP8 dense weights, an 8-bit KV cache (K8V8), and the model's n-gram table read from NVMe on demand.
- **Speculative decoding** with the model's MTP head and prompt lookup. Drafts are checked exactly, so output matches plain decoding.

## Speed

RTX 5090 on a PCIe x4 link, desktop running on the same GPU:

| Workload | Speed |
|---|---:|
| Coding chat, decode | 180–210 tok/s |
| Document continuation, decode | 125–160 tok/s |
| 32K-token prompt, prefill | ~3,300 tok/s |
| 262K-token prompt, prefill | 167 s (~1,570 tok/s) |

## Install

The easiest way to run it is the installer for the published decensored (heretic) model, [flash-next-abliterated-fngine](https://github.com/satellitedown/flash-next-abliterated-fngine). It builds fngine with its own CUDA toolchain, downloads and verifies the model, and starts the server (Linux, RTX 5090, 64 GB RAM):

```sh
curl -fsSL https://raw.githubusercontent.com/satellitedown/flash-next-abliterated-fngine/main/get.sh | bash
```

Choose 1 to install, then 3 to start.

## Build from source

Needs Linux x86_64, CUDA 13.3, a C++20 compiler, CMake 3.28+, Ninja and liburing. The CPU expert path needs AVX-512 with VNNI, BF16 and VBMI (AMD Zen 4/5, Intel Sapphire Rapids or newer).

```sh
cmake --preset dev        # --preset release skips the tests
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Make a model pack

fngine loads a pack converted from the two-shard ISTA-DASLab GSQ-RCO IQ3_S GGUF (`GGUF1`, `GGUF2`) and Qwen's official BF16 dense weights. Expert bytes are copied as they are, not requantized.

```sh
uv venv --python 3.12 tools/.venv
uv pip install --python tools/.venv/bin/python -r tools/requirements.txt
tools/.venv/bin/python tools/fetch_hf.py --repo Qwen/Qwen3.8-Flash-Next --out /mnt/data/fngine/hf
tools/.venv/bin/python tools/convert.py --gguf "$GGUF1" --ngram-gguf "$GGUF2" --hf /mnt/data/fngine/hf --out /mnt/data/fngine/pack
tools/.venv/bin/python tools/verify_pack.py --pack /mnt/data/fngine/pack --gguf "$GGUF1" --ngram-gguf "$GGUF2" --hf /mnt/data/fngine/hf
```

For full decode speed, also add the optional draft vocabulary and decode profile to the pack ([how](docs/engineering-notes.md#fetch-convert-and-reference)).

## Run

```sh
# One prompt
build/apps/fngine --pack /mnt/data/fngine/pack --prompt 'Write a haiku about GPUs' --max-new 128 --stats

# OpenAI-compatible server
build/apps/fngine-serve --pack /mnt/data/fngine/pack --ctx 262144 --host 127.0.0.1 --port 8002 --model-id qwen3.8-flash-next
```

The server handles Chat Completions with streaming, reasoning, tool calls and `enable_thinking: false`. It is text-only, runs one request at a time with up to eight queued, and has no authentication, so keep it on localhost. Close other GPU apps first: any VRAM they hold means fewer experts on the GPU.

Common options:

- `--ctx N`: context size, 131072 by default, up to 262144
- `--spec auto|mtp|lookup|none`: speculative decoding mode
- `--kv k8v8|k8v4|bf16`: KV cache format

### Use with OMP

Add a provider with `"api": "openai-completions"` and `"baseUrl": "http://127.0.0.1:8002/v1"`, and set its model's `contextWindow` to 262144. Then allow one request at a time in `~/.omp/agent/config.yml` and restart running OMP sessions:

```yaml
providers:
  maxInFlightRequests:
    fngine: 1
```

Without the limit, subagents queue behind long generations and can fail with 503 or 429 errors.

## Status

- All 46 C++/CUDA tests pass (2026-10-15), including speculative-decoding equivalence, long-context state and real-model generation.
- Long prompts work: planted codes were found in 180K-, 200K- and 262K-token requests.
- Not all of the original accuracy targets against the BF16 reference are met; the validator reports this rather than hiding it. Smoke checks pass (working code, tool calls, arithmetic), but this is not a broad quality evaluation.

Every option, the validation details and the full performance record: [docs/engineering-notes.md](docs/engineering-notes.md).

## License

Apache 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for upstream licensing.
