# NVMAI

Native Swift and Metal inference for **Qwen 3.6 35B-A3B** in 4-bit, 6-bit,
and 8-bit quantization on Apple M1-M5 systems. Optional concise mode injects a
per-quantization terse system prompt for direct, lean answers.

## References and credits

NVMAI is a focused fork of
[drumih/turbo-fieldfare](https://github.com/drumih/turbo-fieldfare). The
original project provides the bounded-memory inference architecture, streaming
expert runtime, installer, CLI, Mac app, and local server on which this fork is
built.

The foundational Qwen 3.6 35B-A3B integration came from
[upstream pull request #29](https://github.com/drumih/turbo-fieldfare/pull/29),
authored by [NeelM0906](https://github.com/NeelM0906). That work added the Qwen
hybrid layer graph, gated-DeltaNet kernels and state, ChatML tokenization, tool
call parsing, model repacking, product integration, and the original 4-bit
runtime support. NVMAI merges that contribution and extends it further.

NVMAI 3.2's concise mode is derived from the
[Nail-Qwen3.6-35B-A3B](https://huggingface.co/peculiar-ragdoll/Nail-Qwen3.6-35B-A3B-MLX)
chat template by [peculiar-ragdoll](https://huggingface.co/peculiar-ragdoll),
which force-appends a terseness prompt to every request. Measured on NVMAI's
own 4/6/8-bit builds, the built-in prompts cut answer tokens by 55-61% with no
measurable correctness loss on the sampled questions.

## Changelog

[Changelog](https://github.com/Pummelchen/NVMAI/wiki/Changelog)

## FAQ

[FAQ](https://github.com/Pummelchen/NVMAI/wiki/FAQ)

## Benchmark — NVMAI 3.2

Measured on M3 24 GB, current 3.2 build (64 expert-cache slots + resident
pin + MoE phase-1 rewrite). Concise-mode answer tokens: temp 0, deterministic,
8 chat questions, 512-token cap (`+` = baseline hit the cap, so its true total
is higher). Full measurement history:
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
| Concise mode — answer tokens (8-question set) | 3,788+ → **1,480** (−61%) |
| — concise prompt | standard (4-bit) |

#### 6-bit

| Stat | Value |
| --- | --- |
| Decode, 256-token greedy essay (interleaved A/B) | **7.55 / 7.34 tok/s** (64 slots + pin) |
| — reference: 32 slots + pin | 6.20 / 6.68 tok/s (+15.5%) |
| Decode, 512-token greedy essay | 7.26 tok/s |
| Routed MoE phase-1 GPU | 20.34 ms/token |
| Long-gen 512-token (code-gen) | 4.54 tok/s |
| Throughput envelope (digit / count / essay / coding) | 6.63 / 6.21 / 6.00 / 5.04 tok/s |
| 2×2 cache × MTP matrix (off×off / on×off / off×on) | 4.44 / 4.34 / 4.12 tok/s |
| Concise mode — answer tokens (8-question set) | 3,714+ → **1,680** (−55%) |
| — concise prompt | standard (6-bit) |

#### 8-bit

| Stat | Value |
| --- | --- |
| Decode, 256-token greedy essay (interleaved A/B) | **6.66 / 6.44 tok/s** (64 slots + pin) |
| — reference: 32 slots + pin | 5.46 / 5.74 tok/s (+17%) |
| Decode, 512-token greedy essay | 5.84 tok/s |
| Routed MoE phase-1 GPU | 16.81 ms/token |
| Long-gen 512-token (code-gen) | 3.66 tok/s |
| Throughput envelope (digit / count / essay / coding) | 5.56 / 5.33 / 4.90 / 3.90 tok/s |
| 2×2 cache × MTP matrix (off×off / on×off / off×on) | 3.19 / 3.55 / 3.15 tok/s |
| Concise mode — answer tokens (8-question set) | 3,635+ → **1,570** (−57%) |
| — concise prompt | standard (8-bit) |

## Launch scripts

`tools/` ships one launch script per quantization, in two flavors: the
plain scripts start the server with the production defaults (prompt cache ON,
MTP OFF); the `_concise` scripts add `NVMAI_CONCISE_MODE=1`. Build once
(`swift build -c release`), then run a script — it checks the binary and
model, stops any stale NVMAIServer on the port, and starts the server in the
foreground (Ctrl-C to stop). The server binds to `127.0.0.1`.

| Script | Model | Port | Mode |
| --- | --- | --- | --- |
| `launch_4bit.sh` / `launch_4bit_concise.sh` | 4-bit | 8081 | default / concise |
| `launch_6bit.sh` / `launch_6bit_concise.sh` | 6-bit | 8082 | default / concise |
| `launch_8bit.sh` / `launch_8bit_concise.sh` | 8-bit | 8083 | default / concise |

**What concise ON changes.** Concise mode injects a per-quantization terse
system prompt (the same standard prompt for 4/6/8-bit) into every
request. Answers lead with the answer and skip preamble, restated questions,
filler, and closing codas such as "let me know if you have questions". A
system prompt you send yourself is kept, with the concise prompt appended
after it.

**What concise ON does NOT change.** Decode rate is unchanged — tok/s is bound
by memory bandwidth, not by the prompt. Concise saves answer *length*, so the
time-to-answer shrinks with it: measured on M3 24 GB (temp 0, 8 chat
questions) answer tokens drop 55-61% per quant, roughly doubling the
wall-clock work per answer.

**When to leave it off.** Terseness can clip nuance on complex answers (the
sampled 4-bit prompt occasionally dropped actionable detail). If an answer
seems too clipped, use the plain script instead. The CLI offers the same
choice per run (`--concise`); the Mac app exposes a concise-mode setting.

The server also implements the OpenAI Responses API (`POST /v1/responses`),
so current Codex CLI versions connect directly (`base_url` +
`wire_api = "responses"`) with no proxy — see the wiki server guide.

## Documentation

- [Wiki](https://github.com/Pummelchen/NVMAI/wiki)

NVMAI remains text-only. The server binds to `127.0.0.1` without authentication
or TLS and must not be exposed through a proxy or tunnel.
