<img width="1774" height="887" alt="image" src="https://github.com/user-attachments/assets/dc91bc31-0cd4-42e6-bc7a-67ffb277efe0" />


<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/slogan-dark.svg">
  <img alt="NVMAI is the fastest SSD/NVMe streamer for LLMs on Mac - M1 through M6."
       src="assets/slogan-light.svg" width="660">
</picture>

# NVMAI

### Core Benefits

- NVMAI streams LLM's faster than any other similar project.
- Run large MOE LLM's on low RAM Apple Silicon Macs by keeping the AI model on SSD/NVMe. 
- A 125B model on 8 GB of RAM. NVMAI streams experts straight from SSD, so model size is bounded by your disk space, not your memory.
- You set the RAM budget. NVMAI stays inside it. Give it 4 GB or 16 GB — it holds the line, so your Mac stays responsive while the model runs.
- Apple Neural Engine acceleration for prompt processing, 2.3× faster than the GPU cores.
- Our own Metal kernels, our own engine. Purpose-built for Apple silicon and engineered against the hardware's real bandwidth limit, not a portability layer's.
- No MLX. No GGUF. NVMAI ships its own high-speed model format and a converter that builds it straight from the original weights.
  

### Supported LLMs

- **Qwen3.8-Flash-Next 125B-A6B**
- **Qwen-AgentWorld 35B-A3B**
- **Ornith 1.5 35B-A3B**
- **Qwen 3.6 35B-A3B**


Every model id ends in the routed-expert width, read from the manifest rather
than parsed from the name, so two quantizations of the same weights stay
distinguishable. Ask `/v1/models` rather than assuming an id.

### Special Features

- **Bounded expert RAM:** The resident expert cache is sized per family from
  the model's own expert stride and clamped to half of physical memory, so a
  smaller Mac is not handed a budget tuned on a larger one. `--ram-budget`
  overrides it with any size. Model state, KV cache, and runtime scratch use
  additional memory.
- **Long context:** Native RoPE supports up to 262K tokens, while optional YaRN
  extends the context to 512K or 1M tokens.
- **Compressed KV cache:** Live attention state can use 16-bit, 8-bit, or 4-bit
  storage independently of the installed model quantization, with 8-bit as the
  default.
- **Thinking mode:** Ornith and Qwen support truthful Off/On reasoning control;
  their chat templates do not define Low, Medium, or High effort levels.
- **MTP off by default:** Native speculative decoding remains experimental and
  disabled because measured Ornith runs showed no speed benefit and it
  currently requires greedy decoding, native RoPE, and prompt-cache reuse off.

### Performance Improvements

- **Tiled Top-K sampling:** Production sampling (Top-K 1–64) runs a
  three-stage tiled GPU reduction, cutting per-token sampling cost from
  15.5 ms to 1.4 ms with a token-for-token identical stream — the main
  source of the v4.6 decode gain.
- **ANE prefill (experimental, opt-in):** `NVMAI_PREFILL_ANE=on` runs
  full-attention prefill blocks on the Neural Engine from a one-time
  exported Core ML sidecar, roughly halving long-prompt time to first
  token; short prompts and decode are untouched.
- **Follow-up cache:** Exact live and multi-prefix prompt-state reuse avoids
  repeating compatible prefill work across conversation turns.
- **Concise mode:** An optional terse system prompt reduces generated text for
  workloads that benefit from it; standard responses are the default because
  they generalized more reliably in the coding/tooling qualification.
- **Fast alias:** The chat-only `-fast` model alias strips coding-agent
  boilerplate before prefill for quicker direct answers, while the base alias
  preserves tools and agent loops.

### Usage

- **OpenAI-compatible server:** A loopback Chat Completions and Responses API
  includes launch scripts for starting NVMAI and connecting supported coding
  clients.
- **Tested coding CLIs:** The launch workflow supports Codex, Qwen Code, and
  OpenCode against the local server.
- **Mac app and tools:** NVMAI also provides a native Mac app, direct CLI
  generation, streaming responses, and client-authorized function-tool calls.

## Benchmarks

Peak decode on a base 8-core M3 MacBook Pro with 24 GB, generating 512 tokens
of continuous English prose. Each row is the median of fresh-process runs after
one discarded warmup, on an idle machine, at the shipped defaults for that
model -- nothing is pinned for the benchmark that a user would not get.


| Model | Quantization | Peak decode |
| --- | --- | ---: |
| Ornith 1.5 35B-A3B | 4-bit | **22.65 tok/s** |
| Qwen 3.6 35B-A3B | 4-bit | **19.21 tok/s** |
| Qwen-AgentWorld 35B-A3B | 4-bit | **18.57 tok/s** |
| Qwen3.8-Flash-Next 125B-A6B | 4-bit | **5.25 tok/s** |
| Ornith 1.5 35B-A3B | 8-bit | **11.89 tok/s** |
| Qwen 3.6 35B-A3B | 8-bit | **9.73 tok/s** |
| Qwen-AgentWorld 35B-A3B | 8-bit | **9.46 tok/s** |
| Qwen3.8-Flash-Next 125B-A6B | 8-bit | **2.03 tok/s** |


Settings: temperature `0.6`, Top-P `0.95`, Top-K `20`, presence penalty `0.0`,
native 262K context, prompt cache on, 8-bit KV, and MTP off. The Qwen 3.6 and
Qwen-AgentWorld rows were measured 2026-09-04 on the story-generation prompt,
medians of three (4-bit) and two (8-bit) runs, on the builds the installer
makes today. The Qwen3.8-Flash-Next rows are from the same day and prompt at
that family's shipped defaults (temperature `1.0`, Top-P `0.95`), the median
of three runs at 4-bit and a single run at 8-bit. The Ornith rows are from
2026-08-30, measured as the best of four prompts on the mlx-community build
the installer used then; they will be re-measured when Ornith is rebuilt from
its bf16 release.

Every install is built by `tools/install_models.sh` from the model's own
bf16 release, one shard in flight at a time, quantized here to group-64
affine with the router, the shared-expert gate, the DeltaNet gating
projections and every norm kept at bf16 in both widths; no third-party
quantization is used. Qwen-AgentWorld is Qwen's agentic fine-tune of the
35B-A3B geometry. Qwen3.8-Flash-Next is a 125B model with 6B active
parameters, against 35B/3B for the others. Decode is bound by streaming routed experts from SSD, so the
4-bit rows run roughly twice the 8-bit ones, and the 125B model is slower again
because 512 experts at top-10 spread across a far wider working set.

Against the publication before 4.6's, Ornith 8-bit gained **+16.9%** (10.17)
and Qwen 3.6 8-bit **+14.3%** (11.12), from the per-family expert-cache sizing
and speculative prefetch. The Qwen3.8-Flash-Next 4-bit row is not comparable
with the 6.82 published on 2026-08-30: that figure was the best of four prompts,
where this one is the prose prompt alone, and it was taken with the routing
top-k on a single GPU thread (since replaced, +5%). The two 4-bit 35B
configurations did not change and reproduce their published numbers to within
0.5%, which is what makes the other three readable as gains rather than drift.

These are larger than the same changes measured in an interleaved A/B (+21.3%,
+5.9%, +5.5%). Interleaving keeps every configuration page-cache-warm, which
flatters the baseline far more than the tuned configuration, because the tuned
one barely reaches the disk. Measured cold, the way a server actually starts,
the gains are two to three times what the interleaved test credited.

ANE prefill is on by default for Ornith and Qwen 3.6. On a 9,316-token prompt
it measured **3.14x** faster end to end at 4-bit (313.6 s -> 99.9 s) and
**1.91x** at 8-bit; `NVMAI_PREFILL_ANE=off` opts out.

[Full benchmark results](https://github.com/Pummelchen/NVMAI/wiki/Benchmarks)

## Core Links

- [Getting started](https://github.com/Pummelchen/NVMAI/wiki/Getting-Started)
- [Features](https://github.com/Pummelchen/NVMAI/wiki/Features)
- [Local server and launchers](https://github.com/Pummelchen/NVMAI/wiki/OpenAI-Compatible-Server)
- [Runtime controls](https://github.com/Pummelchen/NVMAI/wiki/Runtime-Controls)
- [Benchmarks](https://github.com/Pummelchen/NVMAI/wiki/Benchmarks)
- [Changelog](https://github.com/Pummelchen/NVMAI/wiki/Changelog)

## Credits

NVMAI is a focused fork of
[drumih/turbo-fieldfare](https://github.com/drumih/turbo-fieldfare), which
provides the bounded-memory runtime, installer, CLI, Mac app, and local server.
The Qwen 3.6 integration was created by
[NeelM0906](https://github.com/NeelM0906) in
[upstream PR #29](https://github.com/drumih/turbo-fieldfare/pull/29). Concise
mode is derived from the
[Nail-Qwen3.6-35B-A3B](https://huggingface.co/peculiar-ragdoll/Nail-Qwen3.6-35B-A3B-MLX)
chat template by [peculiar-ragdoll](https://huggingface.co/peculiar-ragdoll).
