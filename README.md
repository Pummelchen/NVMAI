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
  diagnostics, and 4K through 128K context settings using FP16 KV state.
- Explicit OpenCode `coding-lean` and `prompt-only` request profiles. Filtering
  is opt-in through headers and never inferred from `User-Agent`.
- Exact multi-prefix inference-state reuse with bounded live/RAM caching and an
  optional persistent SSD tier. Qwen snapshots include both full-attention KV
  and gated-DeltaNet recurrent state.
- Apple Silicon prefill controls up to 4,096 tokens per chunk, with a 1,024-token
  Qwen default selected from measurements on the base M3.
- Apple M1-M5 target compatibility; the benchmark below was performed on M3.

## What NVMAI added

| Area | Addition |
| --- | --- |
| Quantization | Extended Qwen from the original 4-bit path to complete 4-bit, 6-bit, and 8-bit repacking, loading, decode, shared/routed expert, output-head, and chunked-prefill support. |
| Higher-bit prefill | Added affine 6-bit/8-bit Metal kernels, final-row output support, routed-expert fixes, and regression tests covering prefill-to-decode continuity. |
| Context | Raised the server and app choices to 128K while retaining 4K, 8K, 16K, 32K, and 64K options. |
| Apple Silicon tuning | Added larger Qwen prefill chunks, M3 measurements and defaults, runtime controls, and memory/phase diagnostics while retaining the M1-M5 target. |
| Swift | Moved the package to Swift tools 6.3 and Swift 6 language mode; the current validated toolchain is Apple Swift 6.3.3. |
| OpenCode | Added explicit `coding-lean` and `prompt-only` profiles to reduce unnecessary client guidance and tool-schema overhead without unsafe heuristic detection. |
| Prompt reuse | Added exact multi-prefix live/RAM/SSD inference-state caching with compatibility checks, atomic persistence, bounded LRU eviction, integrity validation, and cache-usage reporting. |
| Validation | Added higher-bit kernel/runtime tests, OpenCode boundary tests, state-snapshot tests, persistent-cache tests, and the reproducible coding-agent stress benchmark. |

## M3 24 GB test results

These measurements were recorded at commit `2ddf68e` on a MacBook Pro
`Mac15,3` with a base 8-core Apple M3 (4 performance + 4 efficiency cores),
24 GB unified memory, macOS 26.6, and Apple Swift 6.3.3.

The workload used 10 coding conversations with an initial request and two
follow-ups: 30 requests per configuration and 120 successful requests in
total. It used the OpenCode `coding-lean` profile, 4,096-token context,
temperature `0.2`, Top-K `64`, Top-P `0.95`, and at most 128 generated tokens.
Cache-on used 64 entries, 512 MiB RAM, and a 4 GiB SSD tier.

| Quant | Cache | Requests | Prompt tokens | Cached tokens | Generated tokens | Decode tok/s | End-to-end output tok/s | Mean TTFT | Total wall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4-bit | Not measured | — | — | — | — | — | — | — | — |
| 6-bit | Off | 30 | 5,882 | 0 | 2,198 | 6.99 | 4.11 | 7.35 s | 535.3 s |
| 6-bit | On | 30 | 5,948 | 4,396 | 2,217 | 5.61 | 4.09 | 4.69 s | 541.8 s |
| 8-bit | Off | 30 | 6,217 | 0 | 2,474 | 5.31 | 3.35 | 9.02 s | 738.0 s |
| 8-bit | On | 30 | 6,321 | 4,769 | 2,491 | 5.16 | 3.80 | 5.56 s | 655.0 s |

The local 4-bit checkpoint had been removed before this stress test, so no
4-bit result is inferred. On the 20 follow-up requests per quantization, cache
reuse reduced computed prefill by 87.0% for 6-bit and 87.8% for 8-bit. It cut
mean time to first token by 52.8% and 55.7%, respectively. Follow-up wall time
improved by 5.6% for 6-bit and 19.7% for 8-bit; caching reduces repeated prefill
but does not increase steady-state decode speed.

`Decode tok/s` is measured between the first and last visible streamed content
token. `End-to-end output tok/s` includes prefill, generation, and cache
publication. These are workload measurements, not performance ceilings. See
the [complete report and raw evidence](benchmark-results/cache-stress-20260803T110702Z/RESULTS.md)
for the protocol, follow-up-only tables, validation, and caveats.

## Documentation

- [OpenAI-compatible server and OpenCode setup](docs/OPENAI_SERVER.md)
- [Runtime controls](docs/RUNTIME_CONTROLS.md)
- [System design](docs/SYSTEM_DESIGN.md)
- [Benchmark protocol](docs/COMMUNITY_BENCHMARKS.md)
- [Qwen performance notes](docs/QWEN36_PERFORMANCE.md)

NVMAI remains text-only. The server binds to `127.0.0.1` without authentication
or TLS and must not be exposed through a proxy or tunnel.
