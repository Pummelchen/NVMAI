<img width="1122" height="1402" alt="image" src="https://github.com/user-attachments/assets/b807e4e2-f26f-4cae-9998-3a7fdbe03290" />

# NVMAI

**Answers in seconds, not minutes.** The `<model>-fast` model alias delivers a
**10×–65× reduction in response time** for coding CLIs — Codex, Qwen Code, and
OpenCode. Same model, same weights, same answers: the CLIs' agent boilerplate
is stripped before prefill, so a question that took minutes now answers in
seconds. (Introduced in v3.3; see the callout below for what is new in 3.5.)

**The Paris test** — "What is the capital of France?" (4-bit, wall clock):

| CLI | Base model | `-fast` alias | Speed-up |
| --- | ---: | ---: | --- |
| Codex `exec` | ~250 s | 24.6 s | **~10×** |
| OpenCode `run` | ~218 s | 5.9 s | **~37×** |
| Qwen Code `-p` | >300 s (stalled) | 4.6 s | **>65×** |

**New in 3.5:** production-readiness hardening — full-codebase audit fixes
across the OpenAI-compatible server (multi-tool-call streaming, request
validation, corrected temperature/top_k defaults), the Mac app's
decode-service IPC, the runtime, installer, and Metal kernels — plus a fully
green test suite (658 tests / 120 suites) and a `LIMIT=N` fastest-first
shortcut for the coding-CLI benchmark harness.

**Unreleased (since 3.5)**

- **Model residency.** `--lazy-load` binds the port immediately and defers the
  load to the first inference request; `--idle-unload-seconds <n>` releases the
  weights after `n` idle seconds and reloads transparently on the next request;
  `POST /v1/models/unload` releases them on demand, draining in-flight requests
  first. Measured on M3 24 GB, 4-bit: **2.9 MB idle → 5.7 GB loaded → 245 MB
  after unload**. Both flags default to off, so existing invocations are
  unchanged. See the [server guide](https://github.com/Pummelchen/NVMAI/wiki/OpenAI-Compatible-Server#model-residency).
- **Correctness.** Fixed a data race in the softmax kernel that could NaN an
  entire probability row and make the sampler emit an out-of-range token id.
  The prompt cache now keys on the post-strip view of a request, so a `-fast`
  conversation keeps its strip on cached follow-up turns instead of replaying
  the CLI's `<system-reminder>` scaffolding.
- **Operability.** A stale install receipt (from moving or renaming a model
  directory) now names the one-line `--verify-install` recovery instead of
  failing opaquely.
- **Production gates.** `tools/lint.sh` (force-cast and function-length
  ratchet) and a ThreadSanitizer CI job, both run by CI; the warning gate now
  matches compiler diagnostics only. The superseded `tools/responses_bridge.py`
  proxy is gone — the server has spoken the Responses API natively since 3.4.

680 tests / 121 suites.

Native Swift and Metal inference for **Qwen 3.6 35B-A3B** in 4-bit, 6-bit,
and 8-bit quantization on Apple M1-M5 systems. Optional concise mode injects a
per-quantization terse system prompt for direct, lean answers. The fast alias
is available on every quantization — how it works, the trade-off, and the full
4/6/8-bit matrix are on the
[wiki Fast alias page](https://github.com/Pummelchen/NVMAI/wiki/OpenAI-Compatible-Server#fast-alias).

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

## Core Links

- [Wiki](https://github.com/Pummelchen/NVMAI/wiki)
- [Changelog](https://github.com/Pummelchen/NVMAI/wiki/Changelog)
- [FAQ](https://github.com/Pummelchen/NVMAI/wiki/FAQ)

## Benchmarks

**Decode-rate reference (NVMAI 3.2 build).** Measured on M3 24 GB, 3.2 build
(64 expert-cache slots + resident pin + MoE phase-1 rewrite). Concise-mode
answer tokens: temp 0, deterministic, 8 chat questions, 512-token cap (`+` =
baseline hit the cap, so its true total is higher). Full measurement history:
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

#### Coding-CLI 72-combo benchmark (in progress)

The 72-combo harness (`benchmark/combos.sh`) runs 3 coding CLIs (Codex, Qwen
Code, OpenCode) × 2 models (`full` / `-fast`) × 3 quantizations × 2 modes
(default/concise) × 2 reasoning (off/on), ordered fastest-first. Prompt:
`difference of swift and c++ in detail` (4-bit M3 24 GB). Decode = completion
tokens ÷ wall time, so `full` rows include the large agent-prompt prefill
while `fast` rows are near-pure decode. **Partial — 23 of 72 combos** (the run
was stopped and is resumable with `bash benchmark/combos.sh`). The table lists
the combos at ≥ 5.0 tok/s. Answer Quality is the rubric score (correctness 40
/ coverage 30 / structure 20 / examples 10), judging only the final answer
text:

| tok/s | Config | CLI | Model | Wall (s) | Comp. tok | Answer Quality |
| ---: | --- | --- | --- | ---: | ---: | ---: |
| 11.9 | 4-bit default, no thinking | Qwen Code | fast | 377.0 | 4,478 | 90 |
| 11.4 | 4-bit concise, no thinking | Qwen Code | fast | 254.4 | 2,888 | 87 |
| 11.0 | 4-bit default, no thinking | OpenCode | fast | 410.8 | 4,521 | 91 |
| 11.0 | 4-bit default, no thinking | Codex | fast | 360.9 | 3,982 | 88 |
| 10.9 | 4-bit concise, no thinking | OpenCode | fast | 227.8 | 2,494 | 83 |
| 10.6 | 4-bit concise, no thinking | Codex | fast | 257.3 | 2,724 | 78 |
| 6.0 | 6-bit concise, no thinking | Qwen Code | fast | 365.9 | 2,193 | 82 |
| 5.8 | 6-bit concise, no thinking | OpenCode | fast | 464.4 | 2,695 | 82 |
| 5.8 | 6-bit concise, thinking | Qwen Code | fast | 461.9 | 2,684 | 82 |
| 5.7 | 6-bit concise, thinking | OpenCode | fast | 487.8 | 2,757 | 81 |
| 5.7 | 6-bit concise, no thinking | Codex | fast | 390.4 | 2,215 | 80 |
| 5.5 | 6-bit concise, thinking | Codex | fast | 457.7 | 2,500 | 81 |

## Launchers

`tools/` ships two launchers that ask the same five questions (CLI, model,
quantization, mode, reasoning):

- **`tools/server_launcher.sh`** starts the NVMAIServer only, in the
  foreground (Ctrl-C to stop). The quantization/mode/reasoning answers drive
  the server; the CLI/model answers print the matching `tools/cli_launcher.sh`
  command to connect a coding CLI afterwards. Once the server is up it
  prints the OpenAI API setup (base URL, API key, the chosen model ID) for
  pointing any OpenAI-compatible client at it. The server binds to
  `127.0.0.1`: 4-bit → 8081, 6-bit → 8082, 8-bit → 8083.
- **`tools/cli_launcher.sh`** starts any of the three coding CLIs (Codex,
  Qwen Code, OpenCode): it stops any stale `NVMAIServer`, starts a fresh one
  via `server_launcher.sh` for the chosen quantization/mode/reasoning, wires
  the CLI's provider config to the chosen model, and execs into the CLI's
  TUI. The server keeps running after the CLI exits.

Both accept the same positional arguments:

```bash
tools/server_launcher.sh [codex|qwen|opencode] [fast|full] [4|6|8] [default|concise] [nothink|think]
tools/cli_launcher.sh    [codex|qwen|opencode] [fast|full] [4|6|8] [default|concise] [nothink|think]
```

Run either with no arguments and it asks the same five questions in order —
CLI, model, quantization, mode, reasoning. Every choice has a default
(**codex / full / 4-bit / default / thinking on**), so pressing Enter through
all prompts launches that configuration. Or pass them positionally, e.g.:

```bash
tools/cli_launcher.sh codex fast 4 concise nothink   # Codex, fast alias, 4-bit, terse, no reasoning
tools/cli_launcher.sh qwen full 6 default think      # Qwen Code, full agent, 6-bit, full answers, reasoning
tools/server_launcher.sh qwen fast 4 concise think   # server only, foreground (Ctrl-C to stop)
```

Each `cli_launcher.sh` run stops any running `NVMAIServer`, starts a fresh
one on the chosen quantization, points the selected CLI at the chosen model
(`-fast` alias or base model), and execs into the CLI's TUI.

Both launchers default to the full model (agentic tool loop) with reasoning
on; choose `fast` for seconds-per-answer chat and `nothink` for direct
answers without the reasoning pass.

The server also implements the OpenAI Responses API (`POST /v1/responses`),
so current Codex CLI versions connect directly (`base_url` +
`wire_api = "responses"`) with no proxy — see the wiki server guide.
