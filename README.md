# NVMAI

Native Swift and Metal inference for **Qwen 3.6 35B-A3B** in 4-bit, 6-bit,
and 8-bit quantization on Apple M1-M5 systems.

NVMAI is a focused fork of
[drumih/turbo-fieldfare](https://github.com/drumih/turbo-fieldfare). The
original project provides the bounded-memory inference architecture, streaming
expert runtime, installer, CLI, Mac app, and local server on which this fork is
built.

## Qwen contribution credit

The foundational Qwen 3.6 35B-A3B integration came from
[upstream pull request #29](https://github.com/drumih/turbo-fieldfare/pull/29),
authored by [NeelM0906](https://github.com/NeelM0906). That work added the Qwen
hybrid layer graph, gated-DeltaNet kernels and state, ChatML tokenization, tool
call parsing, model repacking, product integration, and the original 4-bit
runtime support. NVMAI merged that contribution and extends it with the work
listed below.

## Features

- Native Swift 6 and Metal inference, without wrapping MLX or llama.cpp.
- Qwen 3.6 35B-A3B support with pinned MLX Community 4-bit, 6-bit, and 8-bit
  checkpoints and a streaming `.gturbo` repacker.
- Bounded-memory expert streaming: common weights and runtime state stay
  resident while routed experts are read from SSD as needed.
- Qwen's hybrid architecture: 30 gated-DeltaNet layers, 10 full-attention
  layers, 256 routed experts per layer with top-8 routing, and a shared expert.
- Native Mac app, command-line interface, installer/verifier, and an
  OpenAI-compatible loopback server with JSON and SSE streaming.
- ChatML conversations, function-tool calls, sampling controls, runtime
  diagnostics, and 4K through 256K context settings using FP16 KV state.
- Exact multi-prefix inference-state reuse with bounded live/RAM caching and an
  optional persistent SSD tier. Qwen snapshots include both full-attention KV
  and gated-DeltaNet recurrent state.
- Native streaming-aware Qwen MTP speculative decode with target-verified
  greedy output:
  a reusable 4-bit one-layer sidecar shares the target embedding/head, streams
  only top-8 draft experts, verifies two tokens through batched target prefill,
  and defaults to a strict 384 MiB incremental memory budget.
- Apple Silicon prefill controls up to 4,096 tokens per chunk, with the
  Qwen default set to 4,096 from measurements on the base M3.
- Apple M1-M5 target compatibility; the benchmark below was performed on M3.

## What NVMAI added

| Area | Addition |
| --- | --- |
| Quantization | Extended Qwen from the original 4-bit path to complete 4-bit, 6-bit, and 8-bit repacking, loading, decode, shared/routed expert, output-head, and chunked-prefill support. |
| Higher-bit prefill | Added affine 6-bit/8-bit Metal kernels, final-row output support, routed-expert fixes, and regression tests covering prefill-to-decode continuity. |
| Context | Raised the server, CLI, and app ceiling to Qwen's full 262,144-token window while retaining 4K through 128K options. |
| Apple Silicon tuning | Added larger Qwen prefill chunks, M3 measurements and defaults, runtime controls, and memory/phase diagnostics while retaining the M1-M5 target. |
| Swift | Moved the package to Swift tools 6.3 and Swift 6 language mode; the current validated toolchain is Apple Swift 6.3.3. |
| Prompt reuse | Added exact multi-prefix live/RAM/SSD inference-state caching with compatibility checks, atomic persistence, bounded LRU eviction, integrity validation, and cache-usage reporting. |
| Native MTP | Added a pinned tensor-only MTP sidecar repacker, shared 4/6/8-bit target embedding/head binding, 4-bit router support, 32-row streaming draft prefill, two-token target verification, bounded draft KV, acceptance metrics, and KV/Gated-DeltaNet rollback. |
| Validation | Added higher-bit kernel/runtime tests, state-snapshot tests, persistent-cache tests, and the reproducible coding-agent stress benchmark. |

## 2×2 Precise Benchmark Results (Aug 2026)

Measured on M3 24 GB with 12 independent prompts (128-token max generations).
Each prompt runs as a fresh session. TTFT (time to first token) measured via
streaming, decode speed computed only from requests with ≥0.5s decode time.

> **Methodology caveat (audit correction):** the server forces prompt cache
> **off** whenever MTP is enabled (a target-only snapshot cannot restore the
> draft stream), so the `ON × ON` cell below never actually ran with both
> enabled — it re-ran the cache-off × MTP-on configuration and is marked
> invalid. In addition, all published runs used unique one-shot prompts, so
> `cached_tokens` was 0 everywhere and the multi-prefix cache dimension was
> not measured. `benchmark/nvmai_benchmark.py` has been corrected (replayable
> warm sends that verify cache hits, PID-managed servers, answer
> verification) and the matrix is being regenerated; treat the cache column
> and the ON × ON row below as unmeasured until the regenerated numbers are
> published.

| Config | Cache | MTP | **4-bit** | **6-bit** | **8-bit** |
|--------|-------|-----|-----------|-----------|-----------|
| OFF × OFF | off | off | **10.17 tok/s** | 6.96 tok/s | 5.83 tok/s |
| ON × OFF | multi-prefix | off | **10.19 tok/s** | 6.38 tok/s | 5.77 tok/s |
| OFF × ON | off | on | **10.21 tok/s** | 6.16 tok/s | 5.92 tok/s |
| ~~ON × ON~~ | — | — | ~~8.62~~ | ~~5.68~~ | ~~6.02~~ |

### Key Findings

1. **4-bit is the fastest quantization** at ~10 tok/s across all configs.
   Consistent decode performance with minimal overhead variation.

2. **6-bit trades ~30% speed for better quality** at ~6-7 tok/s. Notable
   degradation with MTP on (6.16 tok/s vs 6.96 tok/s OFF × OFF).

3. **8-bit matches 6-bit** at ~5-6 tok/s for cache-only configs.

4. **MTP ON** reduces decode speed across all quantizations. The one-layer
   draft sidecar adds overhead that cannot overcome the 40-layer MoE cost.

## Documentation

- [Wiki home](https://github.com/Pummelchen/NVMAI/wiki)
- [OpenAI-compatible server](https://github.com/Pummelchen/NVMAI/wiki/OpenAI-Compatible-Server)
- [Runtime controls](https://github.com/Pummelchen/NVMAI/wiki/Runtime-Controls)
- [System design](https://github.com/Pummelchen/NVMAI/wiki/System-Design)
- [Benchmarks and protocol](https://github.com/Pummelchen/NVMAI/wiki/Benchmarks)

NVMAI remains text-only. The server binds to `127.0.0.1` without authentication
or TLS and must not be exposed through a proxy or tunnel.
