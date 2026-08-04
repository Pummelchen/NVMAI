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
- Explicit OpenCode `coding-lean` and `prompt-only` request profiles. Filtering
  is opt-in through headers and never inferred from `User-Agent`.
- Ready-to-use project `opencode.jsonc` with the `NVMAI` provider, the existing
  `qwen3.6-35b-a3b` model ID, and the full 262,144-token context limit.
- Exact multi-prefix inference-state reuse with bounded live/RAM caching and an
  optional persistent SSD tier. Qwen snapshots include both full-attention KV
  and gated-DeltaNet recurrent state.
- Native streaming-aware Qwen MTP speculative decode with target-verified
  greedy output:
  a reusable 4-bit one-layer sidecar shares the target embedding/head, streams
  only top-8 draft experts, verifies two tokens through batched target prefill,
  and defaults to a strict 384 MiB incremental memory budget.
- Apple Silicon prefill controls up to 4,096 tokens per chunk, with a 1,024-token
  Qwen default selected from measurements on the base M3.
- Apple M1-M5 target compatibility; the benchmark below was performed on M3.

## What NVMAI added

| Area | Addition |
| --- | --- |
| Quantization | Extended Qwen from the original 4-bit path to complete 4-bit, 6-bit, and 8-bit repacking, loading, decode, shared/routed expert, output-head, and chunked-prefill support. |
| Higher-bit prefill | Added affine 6-bit/8-bit Metal kernels, final-row output support, routed-expert fixes, and regression tests covering prefill-to-decode continuity. |
| Context | Raised the server, CLI, and app ceiling to Qwen's full 262,144-token window while retaining 4K through 128K options. |
| Apple Silicon tuning | Added larger Qwen prefill chunks, M3 measurements and defaults, runtime controls, and memory/phase diagnostics while retaining the M1-M5 target. |
| Swift | Moved the package to Swift tools 6.3 and Swift 6 language mode; the current validated toolchain is Apple Swift 6.3.3. |
| OpenCode | Added explicit `coding-lean` and `prompt-only` profiles to reduce unnecessary client guidance and tool-schema overhead without unsafe heuristic detection. |
| Prompt reuse | Added exact multi-prefix live/RAM/SSD inference-state caching with compatibility checks, atomic persistence, bounded LRU eviction, integrity validation, and cache-usage reporting. |
| Native MTP | Added a pinned tensor-only MTP sidecar repacker, shared 4/6/8-bit target embedding/head binding, 4-bit router support, 32-row streaming draft prefill, two-token target verification, bounded draft KV, acceptance metrics, and KV/Gated-DeltaNet rollback. |
| Validation | Added higher-bit kernel/runtime tests, OpenCode boundary tests, state-snapshot tests, persistent-cache tests, and the reproducible coding-agent stress benchmark. |

## MTP Benchmark Results (Aug 2026)

MTP uses a 1-layer 4-bit draft sidecar that shares the target embedding and
head, predicts one token ahead, and is verified by a single 40-layer target pass.
The design caps MTP KV at 65,536 tokens (128 MiB) and streams only top-8 draft
experts. It runs only for pure-greedy requests (`temperature: 0`).

Benchmarks were run on M3 24 GB with `max_completion_tokens: 128`, temperature
`0.2` for baseline and `0.0` for MTP, 30 requests across 10 multi-turn
conversations. The MTP sidecar was the pinned 4-bit version from MLX Community
(`qwen3.6-35b-a3b-mtp-4bit`).

### 4-bit MTP matrix

| Condition | Cache | MTP | Decode tok/s | E2E tok/s | Total wall (30 req) |
| --- | --- | --- | ---: | ---: | ---: |
| Baseline | Off | Off | **8.03** | **8.03** | 302 s |
| Cache only | On | Off | 7.04 | 7.13 | 341 s |
| MTP only | Off | On | 5.38 | 5.45 | 431 s |
| Cache + MTP | On | On | 5.23 | 5.30 | 443 s |

### 8-bit MTP comparison

| Condition | Cache | MTP | Decode tok/s | E2E tok/s | Total wall |
| --- | --- | --- | ---: | ---: | ---: |
| Baseline | Off | Off | 4.42 | 4.43 | 347 s |
| Cache only | On | Off | 4.42 | 4.42 | 347 s |
| MTP only | Off | On | 4.44 | 4.45 | 346 s |
| Cache + MTP | On | On | 4.39 | 4.39 | 350 s |

### Takeaways

1. **Cache does not accelerate short generations.** With 128-token completions
   and prompt contexts under 100 tokens, there is no measurable prefix reuse
   benefit from multi-prefix caching. The cache shines at longer prompts where
   repeated structures appear across many turns.

2. **MTP is architecturally correct but negative-speedup at this scale.**
   The server logs confirmed acceptance rates of 40–87% and 1.4–1.87 tokens
   emitted per target pass. However, the draft sidecar's forward pass (embedding
   lookup, 2D projection, full 1-layer attention + MoE + head) adds ~30–45%
   overhead per step. The 1.55× token benefit cannot overcome that GPU cost on
   Apple Silicon where the target pass itself is SSD-bandwidth bound.

3. **MTP may work better on larger targets.** For a 100+ layer model where the
   target pass dominates total compute, the 1-layer draft becomes cheaper
   relative to verification, and even a 55–60% acceptance rate could approach
   the 1.25–1.6× goal. On a 40-layer MoE, the draft overhead is a larger
   fraction of the total.

4. **8-bit MTP is statistically neutral.** The 1–2% variation between MTP-on
   and MTP-off at 8-bit sits within measurement noise (~±5%). The draft sidecar
   cost roughly cancels any acceptance benefit at this quantization.

The 1.25–1.6× target remains a published benchmark goal, not a current result.
Clean paired measurements with longer generations and larger contexts are needed
before MTP can be enabled by default.

## M3 24 GB test results

The table retains the faster complete 30-request run for each configuration
from two benchmark passes at source-equivalent commits `2ddf68e`, `cae9375`,
and `c74f11f`. Selection uses whole-run end-to-end output tok/s; metrics are
never mixed between runs. The rerun won for 4-bit cache-on and 6-bit cache-on,
while the prior run remained faster for the other four rows. All runs used a
MacBook Pro `Mac15,3` with a base 8-core Apple M3 (4 performance + 4 efficiency
cores), 24 GB unified memory, macOS 26.6, and Apple Swift 6.3.3.

The workload used 10 coding conversations with an initial request and two
follow-ups: 30 requests per configuration per pass and 360 successful requests
across both passes. It used the OpenCode `coding-lean` profile, 4,096-token
context, temperature `0.2`, Top-K `64`, Top-P `0.95`, and at most 128 generated
tokens.
Cache-on used 64 entries, 512 MiB RAM, and a 4 GiB SSD tier.

| Quant | Cache | Requests | Prompt tokens | Cached tokens | Generated tokens | Decode tok/s | End-to-end output tok/s | Mean TTFT | Total wall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4-bit | Off | 30 | 6,337 | 0 | 2,669 | 12.41 | 7.57 | 4.59 s | 352.4 s |
| 4-bit | On | 30 | 6,382 | 4,827 | 2,521 | 9.89 | 7.44 | 2.71 s | 339.1 s |
| 6-bit | Off | 30 | 5,882 | 0 | 2,198 | 6.99 | 4.11 | 7.35 s | 535.3 s |
| 6-bit | On | 30 | 5,948 | 4,396 | 2,217 | 5.66 | 4.15 | 4.60 s | 534.0 s |
| 8-bit | Off | 30 | 6,217 | 0 | 2,474 | 5.31 | 3.35 | 9.02 s | 738.0 s |
| 8-bit | On | 30 | 6,321 | 4,769 | 2,491 | 5.16 | 3.80 | 5.56 s | 655.0 s |

On the 20 follow-up requests per quantization, cache reuse reduced computed
prefill by 88.0% for 4-bit, 87.0% for 6-bit, and 87.8% for 8-bit. It cut mean
time to first token by 57.9%, 55.3%, and 55.7%, respectively. Follow-up wall
time improved by 12.8%, 9.4%, and 19.7%; caching reduces repeated prefill but
does not increase steady-state decode speed.

`Decode tok/s` is measured between the first and last visible streamed content
token. `End-to-end output tok/s` includes prefill, generation, and cache
publication. These are workload measurements, not performance ceilings. See
the [rerun and best-of-two report](benchmark-results/cache-stress-all6-20260803T150721Z/RESULTS.md)
and the prior raw evidence for
[4-bit](benchmark-results/cache-stress-4bit-20260803T125234Z/RESULTS.md) and
[6-bit/8-bit](benchmark-results/cache-stress-20260803T110702Z/RESULTS.md) for
the protocol, validation, and caveats.

## Documentation

- [Wiki home](https://github.com/Pummelchen/NVMAI/wiki)
- [Getting started](https://github.com/Pummelchen/NVMAI/wiki/Getting-Started)
- [OpenAI-compatible server and OpenCode setup](https://github.com/Pummelchen/NVMAI/wiki/OpenAI-Compatible-Server)
- [Runtime controls](https://github.com/Pummelchen/NVMAI/wiki/Runtime-Controls)
- [System design](https://github.com/Pummelchen/NVMAI/wiki/System-Design)
- [Benchmarks and protocol](https://github.com/Pummelchen/NVMAI/wiki/Benchmarks)

NVMAI remains text-only. The server binds to `127.0.0.1` without authentication
or TLS and must not be exposed through a proxy or tunnel.
