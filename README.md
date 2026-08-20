<img width="2456" height="930" alt="image" src="https://github.com/user-attachments/assets/737b0ff8-1f55-4456-bc32-89a532cbd716" />


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

- **Follow-up cache:** Exact live and multi-prefix prompt-state reuse avoids
  repeating compatible prefill work across conversation turns.
- **Concise mode:** The launch scripts enable a terse system prompt by default
  to reduce unnecessary generated tokens and response time without changing
  model weights; standard mode remains selectable.
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
> supported. **Ornith 1.5 4-bit is now the default installer, app, launcher,
> benchmark, and real-inference test baseline.**

[![Published Ornith 1.5 benchmark overview](assets/stats.png)](https://ornith.ai/ornith_1_5.html)

Ornith is a 35B mixture-of-experts model with approximately 3B active
parameters per token. The chart above is publisher-supplied, not an NVMAI
measurement. See the [Ornith 1.5 announcement](https://ornith.ai/ornith_1_5.html)
and [model card](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B).

## Benchmarks

Fastest single-run decode results from the verified Ornith 1.5 benchmark set on
a base M3 MacBook Pro with 24 GB:

| Quantization | Best decode | Prompt |
| --- | ---: | --- |
| 4-bit | **7.34 tok/s** | Short |
| 8-bit | **4.38 tok/s** | Medium |

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
