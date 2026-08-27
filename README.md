<img width="1774" height="887" alt="image" src="https://github.com/user-attachments/assets/dc91bc31-0cd4-42e6-bc7a-67ffb277efe0" />

> [!NOTE]
> **Coming soon: Qwen3.8-Flash-Next (125B-A6B).** NVMAI will support Qwen's
> new flagship sparse-MoE model — 125B parameters with only 6B active per
> token, a natural fit for SSD-streamed inference on 16–32 GB Macs. The port
> is in development on the
> [`qwen38-flash-next` branch](https://github.com/Pummelchen/NVMAI/tree/qwen38-flash-next)
> ([verified design record](https://github.com/Pummelchen/NVMAI/blob/qwen38-flash-next/docs/qwen38-flash-next-port.md));
> installation will be enabled once an mlx-community quantized release is
> available. Progress is tracked in
> [issue #2](https://github.com/Pummelchen/NVMAI/issues/2).

# NVMAI

## Core Benefits

### Project Purpose

- **SSD-streamed inference:** NVMAI runs large mixture-of-experts (MOE) models on
  low-RAM Apple Silicon Macs by keeping routed experts on SSD/NVMe storage and
  loading only the experts selected for each token.

### Supported LLMs

- **Verified models:** NVMAI supports text-only Ornith 1.5 35B-A3B and Qwen
  3.6 35B-A3B in 4-bit and 8-bit quantization.
- **Adaptable runtime:** The shared Qwen3.5-MoE parser, repacker, and inference
  path make other tensor-compatible Qwen-based MoE models straightforward to
  add after validation.

### Special Features

- **Bounded expert RAM:** The server accepts common 1, 2, 4, 8, or 16 GiB RAM
  budgets to limit the resident expert cache, while model state, KV cache, and
  runtime scratch use additional memory.
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

## Ornith 1.5 35B A3B

> [!IMPORTANT]
> NVMAI now supports text-only **Ornith-1.5-35B-A3B** installation and inference
> in 4-bit and 8-bit. It reuses the verified Qwen3.5-MoE runtime and keeps routed
> experts SSD-streamed with the same bounded-memory design. Ornith's native MTP
> draft is available as an optional experimental sidecar; current M3 benchmarks
> do not show a speed benefit. Vision is not included, and Qwen 3.6 remains
> supported. **Ornith 1.5 8-bit is the default installer, app, launcher,
> benchmark, and real-inference test baseline; Concise and Thinking default
> to off.**

[![Published Ornith 1.5 benchmark overview](assets/stats.png)](https://ornith.ai/ornith_1_5.html)

Ornith is a 35B mixture-of-experts model with approximately 3B active
parameters per token. The chart above is publisher-supplied, not an NVMAI
measurement. See the [Ornith 1.5 announcement](https://ornith.ai/ornith_1_5.html)
and [model card](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B).

## Benchmarks

NVMAI v4.6 median results on a base 8-core M3 MacBook Pro with 24 GB. Ornith
generated 512 tokens of continuous plain English about an ordinary day in a
small town; each row used one discarded warmup and three fresh-process runs.

| Quantization | Median decode | Median wall time | Change from v4.1 |
| --- | ---: | ---: | ---: |
| 4-bit | **22.53 tok/s** | 25.59 s | **+37.0%** |
| 8-bit | **9.40 tok/s** | 59.48 s | **+7.4%** |

Settings: temperature `0.6`, Top-P `0.95`, Top-K `20`, presence penalty `0.0`,
native 262K context, prompt cache on, 8-bit KV, and MTP off. The gain over
v4.1 is the tiled Top-K sampler; the sampled token stream at a fixed seed is
unchanged. With the experimental opt-in ANE prefill enabled, a 6,103-token
prompt additionally measured **2.31x** faster prefill (132.9 s → 57.5 s) with
decode speed unchanged.

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
