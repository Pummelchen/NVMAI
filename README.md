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
runtime support. NVMAI merges that contribution and extends it further.

## Benchmark — NVMAI 2.0

Measured on M3 24 GB, current 2.0 build (64 expert-cache slots + resident
pin + MoE phase-1 rewrite). Full measurement history:
[Optimization-Journey](https://github.com/Pummelchen/NVMAI/wiki/Optimization-Journey).

#### 4-bit

| Stat | Value |
| --- | --- |
| Decode, 256-token greedy essay (interleaved A/B) | **11.91 / 11.45 tok/s** (64 slots + pin) |
| — reference: 32 slots + pin | 10.76 / 10.19 tok/s (+11.5%) |
| Decode, 512-token greedy essay | 14.37 tok/s |
| Routed MoE phase-1 GPU | 10.60 ms/token |
| Long-gen 512-token (code-gen) | 7.32 tok/s |
| Throughput envelope (digit / count / essay / coding) | 14.48 / 11.34 / 9.33 / 8.05 tok/s |
| 2×2 cache × MTP matrix (off×off / on×off / off×on) | 9.58 / 11.17 / 6.77 tok/s |

#### 6-bit

| Stat | Value |
| --- | --- |
| Long-gen 512-token (code-gen) | 4.54 tok/s |
| Throughput envelope (digit / count / essay / coding) | 6.63 / 6.21 / 6.00 / 5.04 tok/s |
| 2×2 cache × MTP matrix (off×off / on×off / off×on) | 4.44 / 4.34 / 4.12 tok/s |

#### 8-bit

| Stat | Value |
| --- | --- |
| Long-gen 512-token (code-gen) | 3.66 tok/s |
| Throughput envelope (digit / count / essay / coding) | 5.56 / 5.33 / 4.90 / 3.90 tok/s |
| 2×2 cache × MTP matrix (off×off / on×off / off×on) | 3.19 / 3.55 / 3.15 tok/s |

## Documentation

- [Wiki](https://github.com/Pummelchen/NVMAI/wiki)

NVMAI remains text-only. The server binds to `127.0.0.1` without authentication
or TLS and must not be exposed through a proxy or tunnel.
