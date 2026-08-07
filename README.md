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
- Parallel expert pread fills across the idle CPU cores: the decode's expert
  fetch is single-flight, so the cache-miss fills run concurrently on a
  mostly-idle machine — measured +28% decode on the 4-bit M3 (9.98 -> 12.80
  tok/s on 512-token generations). Disable with `NVMAI_PARALLEL_IO=0`.
- Expert-cache default 64 slots/layer (~4 GB) with the resident weights
  pinned via mlock at load: measured +10% decode over the 32-slot default
  on the 4-bit M3 (interleaved A/B); up to 128 slots/layer configurable
  for larger hosts. The pin is required — a bigger cache alone evicts the
  file-backed attention weights. Opt out of the pin with `NVMAI_NO_PIN=1`.
- Per-kernel GPU diagnostics: `NVMAI_KERNEL_STATS=1` logs each decode
  command buffer's GPU span by role (attention block vs router vs head vs
  MoE) in the server footer, and `NVMAI_RUNNER_STATS=1` adds the per-token
  stage split (cb1/IO/cb2/head/rdadvise).
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

## Benchmark — NVMAI 2.0

Measured on M3 24 GB, 4-bit checkpoint, current 2.0 build (64 expert-cache
slots + resident pin + MoE phase-1 rewrite). Full measurement history:
[Optimization-Journey](https://github.com/Pummelchen/NVMAI/wiki/Optimization-Journey).

| Stat | Value |
| --- | --- |
| Decode, 256-token greedy essay (interleaved A/B) | **11.91 / 11.45 tok/s** (64 slots + pin) |
| — reference: 32 slots + pin | 10.76 / 10.19 tok/s (+11.5%) |
| Decode, 512-token greedy essay | 14.37 tok/s |
| Routed MoE phase-1 GPU | 10.60 ms/token |
| Long-gen 512-token (code-gen) | 4-bit 7.32 / 6-bit 4.54 / 8-bit 3.66 tok/s |
| Throughput envelope (digit / count / essay / coding), 4-bit | 14.48 / 11.34 / 9.33 / 8.05 tok/s |
| Throughput envelope (digit / count / essay / coding), 6-bit | 6.63 / 6.21 / 6.00 / 5.04 tok/s |
| Throughput envelope (digit / count / essay / coding), 8-bit | 5.56 / 5.33 / 4.90 / 3.90 tok/s |
| 2×2 cache × MTP matrix (off×off / on×off / off×on), 4-bit | 9.58 / 11.17 / 6.77 tok/s |
| 2×2 cache × MTP matrix (off×off / on×off / off×on), 6-bit | 4.44 / 4.34 / 4.12 tok/s |
| 2×2 cache × MTP matrix (off×off / on×off / off×on), 8-bit | 3.19 / 3.55 / 3.15 tok/s |

## Documentation

- [Wiki home](https://github.com/Pummelchen/NVMAI/wiki)
- [OpenAI-compatible server](https://github.com/Pummelchen/NVMAI/wiki/OpenAI-Compatible-Server)
- [Runtime controls](https://github.com/Pummelchen/NVMAI/wiki/Runtime-Controls)
- [System design](https://github.com/Pummelchen/NVMAI/wiki/System-Design)
- [Benchmarks and protocol](https://github.com/Pummelchen/NVMAI/wiki/Benchmarks)

NVMAI remains text-only. The server binds to `127.0.0.1` without authentication
or TLS and must not be exposed through a proxy or tunnel.
