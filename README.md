<img width="1774" height="887" alt="image" src="https://github.com/user-attachments/assets/dc91bc31-0cd4-42e6-bc7a-67ffb277efe0" />

> [!NOTE]
> **Coming soon: Qwen3.8-Flash-Next (125B-A6B).** NVMAI will support Qwen's
> new flagship sparse-MoE model — 125B parameters with only 6B active per
> token, a natural fit for SSD-streamed inference on 16–32 GB Macs. The port
> is in development on `main`
> ([verified design record](https://github.com/Pummelchen/NVMAI/blob/main/docs/qwen38-flash-next-port.md));
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

- **Qwen3.8-Flash-Next 125B-A6B** — 4-bit. 48 layers of gated DeltaNet with
  sparse-indexed attention, 512 experts at top-10, hyper-connection residual
  streams and hashed n-gram embeddings. Text-only; the checkpoint's vision
  tower is not repacked. Served as `qwen3.8-flash-next_4-Bit`.
- **Ornith 1.5 35B-A3B** — 4-bit and 8-bit. Served as
  `ornith-1.5-35b-a3b_4-Bit` / `_8-Bit`, each with a `-fast` chat-only alias.
- **Qwen 3.6 35B-A3B** — 4-bit and 8-bit. Served as `qwen3.6-35b-a3b_4-Bit` /
  `_8-Bit`, each with a `-fast` chat-only alias.
- **Adaptable runtime:** The shared Qwen3.5-MoE parser, repacker, and inference
  path make other tensor-compatible Qwen-based MoE models straightforward to
  add after validation.

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

Peak decode on a base 8-core M3 MacBook Pro with 24 GB, generating 512 tokens
of continuous English prose. Each row is the median of three fresh-process
runs after one discarded warmup, on an idle machine.

**Qwen3.8-Flash-Next 125B-A6B**

| Quantization | Peak decode |
| --- | ---: |
| 4-bit | **5.03 tok/s** |
| 8-bit | not released |

**Ornith 1.5 35B-A3B**

| Quantization | Peak decode |
| --- | ---: |
| 4-bit | **22.53 tok/s** |
| 8-bit | **10.17 tok/s** |

**Qwen 3.6 35B-A3B**

| Quantization | Peak decode |
| --- | ---: |
| 4-bit | **23.23 tok/s** |
| 8-bit | **11.12 tok/s** |

Settings: temperature `0.6`, Top-P `0.95`, Top-K `20`, presence penalty `0.0`,
native 262K context, prompt cache on, 8-bit KV, and MTP off.

Qwen3.8-Flash-Next is a 125B model with 6B active parameters, against 35B/3B
for the other two, and its runtime has had no throughput work yet -- it is
here as a reference point, not a tuned result. Decode is bound by streaming
routed experts from SSD, so the 4-bit rows run roughly twice the 8-bit ones.

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
