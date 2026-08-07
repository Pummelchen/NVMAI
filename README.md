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

## Performance — long generations (code-generation style)

Measured on M3 24 GB, current build (parallel pread fills + SSE fix). A
512-token greedy completion from a coding prompt (Levenshtein + closest-match
functions), server-footer decode rate, 3 runs per quant after a discarded
warmup:

| Quantization | Decode |
| --- | --- |
| **4-bit** | **9.9 tok/s** (9.76 / 10.16 / 9.88) |
| 6-bit | 4.9 tok/s (4.90 / 4.91 / 4.90) |
| 8-bit | 4.1 tok/s (4.10 / 4.13 / 4.11) |

Decode rate is workload-dependent: the same 4-bit build does ~12.8 tok/s on
repetitive prose (high expert-cache locality) and ~9.9 on this coding prompt,
which routes through a more diverse expert set. Reproduce with
`benchmark/nvmai_longgen.py`.

### Throughput envelope (512-token greedy, by workload locality)

The decode rate is a function of expert-cache locality: the more
repetitively the routing reuses the same experts, the fewer expert preads
run and the faster decode goes. The four workloads below rank from
maximally repetitive (digit cycles — the measured peak) to diverse
(coding). At higher quantizations the weight-read cost dominates, so the
envelope flattens. Reproduce with `benchmark/nvmai_maxthroughput.py`.

#### 4-bit

| Workload | Decode | Locality |
| --- | --- | --- |
| **Digit cycles (peak)** | **15.9 tok/s** | maximal |
| Counting 1-1000 | 13.4 tok/s | high |
| Essay prose | 11.2 tok/s | medium |
| Coding (diverse routing) | 8.0 tok/s | low |

#### 6-bit

| Workload | Decode | Locality |
| --- | --- | --- |
| **Digit cycles** | **6.0 tok/s** | maximal |
| Essay prose | 5.9 tok/s | medium |
| Counting 1-1000 | 5.8 tok/s | high |
| Coding | 4.7 tok/s | low |

#### 8-bit

| Workload | Decode | Locality |
| --- | --- | --- |
| **Counting 1-1000** | **5.4 tok/s** | high |
| Digit cycles | 5.2 tok/s | maximal |
| Essay prose | 4.4 tok/s | medium |
| Coding | 3.8 tok/s | low |

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
> not measured. The corrected harness (`benchmark/nvmai_benchmark.py`) has
> since regenerated the matrix across all three quantizations with
> footer-derived decode rates, per-cell temperature, answer verification,
> and a multi-turn continuation probe — the table below is the current
> result (4-bit: 7.69/7.39/5.95, 6-bit: 4.33/4.37/2.72, 8-bit: 3.81/3.78/3.09
> tok/s; MTP confirmed slower at every quant; the cache probe hits on
> continuation in every cell). The current 4-bit decode on long generations
> is **~12.8 tok/s** with the parallel pread fills (see the wiki
> [Optimization-Journey](https://github.com/Pummelchen/NVMAI/wiki/Optimization-Journey)
> for the full measurement history).
>
> **Why these numbers look lower than older README stats:** the earlier
> client-measured rates (~10 tok/s) were inflated by a server bug that
> buffered the whole SSE response (client decode_time collapsed to ~0; the
> old result JSONs contain rates like 51,150 tok/s). The rates below are the
> server's own footer (`decode_tok_s`) on 12 short prompts (2-53 tokens
> each), where per-request overhead is large; the same build decodes long
> generations at ~12.8 tok/s. Same-methodology before/after the parallel
> pread fills: 9.98 -> 12.80 tok/s (+28%).

| Config | Cache | MTP | **4-bit** | **6-bit** | **8-bit** |
|--------|-------|-----|-----------|-----------|-----------|
| OFF × OFF | off | off | **7.69 tok/s** | 4.33 tok/s | 3.81 tok/s |
| ON × OFF | multi-prefix | off | **7.39 tok/s** | 4.37 tok/s | 3.78 tok/s |
| OFF × ON | off | on | **5.95 tok/s** | 2.72 tok/s | 3.09 tok/s |
| ~~ON × ON~~ | — | — | — | — | — |

### Key Findings

1. **4-bit is the fastest quantization** at ~7.4-7.7 tok/s across configs
   (footer-derived decode rates on the 12-prompt matrix). Decode on longer
   generations is ~12.8 tok/s with the parallel pread fills.

2. **6-bit trades ~45% speed for better quality** at ~4.3-4.4 tok/s. The
   MTP penalty is largest here (2.72 tok/s, -37%): the 4-bit draft sidecar
   is least aligned with a 6-bit target.

3. **8-bit is slower still** at ~3.8 tok/s for cache configs; the MTP
   penalty is the smallest of the three (-19%).

4. **MTP ON reduces decode speed at every quantization** (7.69 -> 5.95,
   4.33 -> 2.72, 3.81 -> 3.09 tok/s at 75-81% acceptance). The one-layer
   draft sidecar adds overhead that cannot overcome the 40-layer MoE cost.

5. **The multi-prefix cache hits on multi-turn continuation** (38/47/47 KV
   tokens restored in the probe across quants) but identical-prompt replay
   misses by design — the KV prefix runs through the generated assistant
   turn. Cache saves prefill/TTFT, not per-token decode.

## Documentation

- [Wiki home](https://github.com/Pummelchen/NVMAI/wiki)
- [OpenAI-compatible server](https://github.com/Pummelchen/NVMAI/wiki/OpenAI-Compatible-Server)
- [Runtime controls](https://github.com/Pummelchen/NVMAI/wiki/Runtime-Controls)
- [System design](https://github.com/Pummelchen/NVMAI/wiki/System-Design)
- [Benchmarks and protocol](https://github.com/Pummelchen/NVMAI/wiki/Benchmarks)

NVMAI remains text-only. The server binds to `127.0.0.1` without authentication
or TLS and must not be exposed through a proxy or tunnel.
