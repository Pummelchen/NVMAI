<img width="1122" height="1402" alt="image" src="https://github.com/user-attachments/assets/b807e4e2-f26f-4cae-9998-3a7fdbe03290" />

# NVMAI

**Answers in seconds, not minutes.** NVMAI v3.3 introduces a **10×–65×
reduction in response time** for coding CLIs — Codex, Qwen Code, and
OpenCode — with the new `<model>-fast` model alias. Same model, same weights,
same answers: the CLIs' agent boilerplate is stripped before prefill, so a
question that took minutes now answers in seconds.

**The Paris test** — "What is the capital of France?" (4-bit, wall clock):

| CLI | Base model | `-fast` alias | Speed-up |
| --- | ---: | ---: | --- |
| Codex `exec` | ~250 s | 24.6 s | **~10×** |
| OpenCode `run` | ~218 s | 5.9 s | **~37×** |
| Qwen Code `-p` | >300 s (stalled) | 4.6 s | **>65×** |

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

## Coding CLI launcher

`tools/nvmai-cli.sh` is the fastest way to get a coding CLI talking to NVMAI:
one command builds nothing, starts a fresh server, wires the CLI's provider
config, and hands the terminal over.

```bash
tools/nvmai-cli.sh [codex|qwen|opencode] [fast|full] [4|6|8] [default|concise] [nothink|think]
```

Run it with no arguments and it asks five questions in order — CLI, model,
quantization, mode, reasoning. Every choice has a default (**codex / full /
4-bit / default / thinking on**), so pressing Enter through all prompts
launches that configuration. Or pass them positionally, e.g.:

```bash
tools/nvmai-cli.sh codex fast 4 concise nothink   # Codex, fast alias, 4-bit, terse, no reasoning
tools/nvmai-cli.sh qwen full 6 default think      # Qwen Code, full agent, 6-bit, full answers, reasoning
```

Each run stops any running `NVMAIServer`, starts a fresh one on the chosen
quantization, points the selected CLI at the chosen model (`-fast` alias or
base model), and execs into the CLI's TUI. Ctrl-C on the server ends the
session.

### Parameters

| Argument | Choices (default) | What it does | Pro | Con |
| --- | --- | --- | --- | --- |
| CLI | `codex` (default), `qwen`, `opencode` | Which coding CLI to launch | Codex: OpenAI agent, tool loop; Qwen Code: full agent config; OpenCode: lightweight, easy model picker | Each CLI brings its own agent prompt; Qwen Code's auto-memory is disabled for speed |
| Model | `full` (default), `fast` | `fast` = the `<model>-fast` alias (CLI boilerplate stripped, seconds-per-answer chat); `full` = the base model (keeps the CLI's agentic tool loop) | **fast:** 10×-65× wall-time reduction, ~90-98% less prefill; **full:** agent can read/edit files and run tools | **fast:** no tool calls (chat-only); **full:** multi-thousand-token prefill takes minutes |
| Quantization | `4` (default), `6`, `8` | Which quantized model the server loads (4-bit → port 8081, 6-bit → 8082, 8-bit → 8083) | **4-bit:** fastest decode, smallest memory; **6-bit:** balance; **8-bit:** best fidelity | **4-bit:** most quantization error; **8-bit:** slowest and heaviest |
| Mode | `default` (default), `concise` | Whether the server injects a terse-answer system prompt | **default:** full answers; **concise:** ~55-61% fewer answer tokens, near-identical correctness on sampled questions | **concise:** can clip nuance on complex answers; switch back to default if replies feel too short |
| Reasoning | `think` (default), `nothink` | Whether the model reasons before answering | **think:** shows its work on hard questions | **think:** adds wall time and tokens; choose `nothink` for fast answers |

### Env overrides

| Variable | Default | Purpose |
| --- | --- | --- |
| `NVMAI_PORT` | `8081` | Port for the server (and the CLI configs it writes) |
| `CODEX_HOME_NVMAI` | `~/.codex-nvmai` | Where the Codex config is written (dedicated, so your real `~/.codex` is untouched) |
| `QWEN_HOME_NVMAI` | `~/.qwen-nvmai` | Where the Qwen Code settings are written (dedicated home, real `~/.qwen` untouched) |
| `CODEX`, `QWEN`, `OPENCODE` | default install paths | Override the CLI binary to launch |
| `NVMAI_STRIP_TAGS` | `system-reminder` | Comma-separated in-message scaffolding tags the fast alias strips |

The launcher defaults to the full model (agentic tool loop) with reasoning
on; choose `fast` for seconds-per-answer chat and `nothink` for direct
answers without the reasoning pass.

## Core Links

- [Wiki](https://github.com/Pummelchen/NVMAI/wiki)
- [Changelog](https://github.com/Pummelchen/NVMAI/wiki/Changelog)
- [FAQ](https://github.com/Pummelchen/NVMAI/wiki/FAQ)

NVMAI remains text-only. The server binds to `127.0.0.1` without authentication
or TLS and must not be exposed through a proxy or tunnel.
