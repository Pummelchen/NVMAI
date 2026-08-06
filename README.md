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
| Prompt reuse | Added exact multi-prefix live/RAM/SSD inference-state caching with compatibility checks, atomic persistence, bounded LRU eviction, integrity validation, and cache-usage reporting. |
| Native MTP | Added a pinned tensor-only MTP sidecar repacker, shared 4/6/8-bit target embedding/head binding, 4-bit router support, 32-row streaming draft prefill, two-token target verification, bounded draft KV, acceptance metrics, and KV/Gated-DeltaNet rollback. |
| Validation | Added higher-bit kernel/runtime tests, state-snapshot tests, persistent-cache tests, and the reproducible coding-agent stress benchmark. |

## 2×2 Precise Benchmark Results (Aug 2026)

Measured on M3 24 GB with 12 independent prompts (128-token max generations).
Each prompt runs as a fresh session. TTFT (time to first token) measured via
streaming, decode speed computed only from requests with ≥0.5s decode time.

| Config | Cache | MTP | Warm Decode | Cold TTFT | Overhead/req | Last 6 Avg |
|--------|-------|-----|-------------|-----------|--------------|------------|
| **1. OFF × OFF** | off | off | **5.51 tok/s** | 4.84 s | 6.36 s | 5.88 |
| **2. ON × OFF** | on | off | **5.55 tok/s** | 4.02 s | 6.19 s | 6.00 |
| **3. OFF × ON** | off | on | **5.13 tok/s** | 4.48 s | 6.01 s | 5.28 |
| **4. ON × ON** | on | on | **4.55 tok/s** | 4.01 s | 6.29 s | 4.68 |

### Key Findings

1. **Real decode speed: ~5.5 tok/s**. The previous "1.5 tok/s" figure included
   ~6.3s fixed overhead per request (HTTP dispatch, model access, response
   serialization). For short responses (<20 tokens), overhead dominates total
   wall time.

2. **Cache ON (multi-prefix)** improves cold TTFT by ~25% (4.0s vs 4.8s) with
   identical warm decode speed. Prefill optimization works as expected.

3. **MTP ON** reduces decode speed by ~0.4–1.0 tok/s. The 1-layer draft sidecar
   overhead (embedding + attention + MoE + head) adds ~30–45% per-step cost
   that cannot be overcome on a 40-layer MoE.

4. **Best config: Cache ON + MTP OFF** at 5.55 tok/s warm decode, 4.0s cold
   TTFT, 6.2s fixed overhead per request.

See [`benchmarks/bench-2x2-precise.py`](benchmarks/bench-2x2-precise.py) for
the benchmark script. Results in
[`benchmark-results/bench-2x2-precise-*`](benchmark-results/).

## Benchmark Test Prompts

The following 12-prompt test suite exercises a broad capability spectrum. Each
prompt runs as an independent session (no conversation history). Results are
logged per prompt and averaged across the suite for each 2×2 config
(cache ON/OFF × MTP ON/OFF).

| # | Capability | Prompt | Expected Answer |
|---|-----------|--------|-----------------|
| 1 | Basic fact | What is the capital of France? Answer with only the city. | Paris |
| 2 | Arithmetic | Calculate 17 × 24. Answer with only the number. | 408 |
| 3 | Instruction following | Return exactly the word BLUE in uppercase. | BLUE |
| 4 | Classification | Classify as positive, negative, or neutral: "The product works as expected." | neutral |
| 5 | Extraction | Extract the email address: Contact Ana at ana@example.com tomorrow. | ana@example.com |
| 6 | Structured output | Return JSON with keys "name" and "age" for: Maya is 31. | Valid JSON; exact values |
| 7 | Logic | All bloops are razzies. No razzies are lazzies. Can a bloop be a lazzy? Answer yes or no. | no |
| 8 | Context use | Context: The access code is 7391. What is the access code? Answer only with the code. | 7391 |
| 9 | Refusal/honesty | What is the access code? Do not guess if none was provided. | States it was not provided |
| 10 | Summarization | Summarize in five words: The server failed because its disk was full. | Semantic scoring |
| 11 | Transformation | Convert to lowercase: Hello WORLD 123! | hello world 123! |
| 12 | Simple coding | Write a Python expression that returns the largest value in nums. | e.g. max(nums) |

## Documentation

- [Wiki home](https://github.com/Pummelchen/NVMAI/wiki)
- [Getting started](https://github.com/Pummelchen/NVMAI/wiki/Getting-Started)
- [OpenAI-compatible server](https://github.com/Pummelchen/NVMAI/wiki/OpenAI-Compatible-Server)
- [Runtime controls](https://github.com/Pummelchen/NVMAI/wiki/Runtime-Controls)
- [System design](https://github.com/Pummelchen/NVMAI/wiki/System-Design)
- [Benchmarks and protocol](https://github.com/Pummelchen/NVMAI/wiki/Benchmarks)

NVMAI remains text-only. The server binds to `127.0.0.1` without authentication
or TLS and must not be exposed through a proxy or tunnel.
