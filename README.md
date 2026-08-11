# NVMAI

Native Swift and Metal inference for **Qwen 3.6 35B-A3B** in 4-bit, 6-bit,
and 8-bit quantization on Apple M1-M5 systems. Optional concise mode injects a
per-quantization terse system prompt for direct, lean answers.

Every model also serves a `<model>-fast` alias (e.g.
`qwen3.6-35b-a3b-fast`): the same weights, but coding-CLI boilerplate — agent
system prompts, tool definitions, and in-message scaffolding — is stripped
before prefill, turning multi-thousand-token requests into a few hundred
tokens. Coding CLIs answer simple questions in seconds instead of minutes.
See [Fast alias](#fast-alias) below.

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

## Benchmark — NVMAI 3.3

**Coding-CLI latency, fast alias.** See
[Fast alias](#fast-alias) for the 3-CLI × 3-quant wall-clock matrix
("What is the capital of France?", `qwen3.6-35b-a3b-fast`): Codex 24.6-32.8 s,
Qwen Code 4.6-12.8 s, OpenCode 5.9-10.2 s across 4/6/8-bit. On 4-bit that is
**~10× (Codex), ~37× (OpenCode), and >65× (Qwen Code) faster than the base
model** — a 90-98% reduction in wall time for the same question.

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

## Fast alias

Coding CLIs (Codex, Qwen Code, OpenCode) send several thousand tokens of
boilerplate with every request: their agent system prompt, tool definitions,
and in-message scaffolding such as Qwen Code's `<system-reminder>` blocks.
On 4-bit that prefill alone can take 4+ minutes for a one-line question.

The `<model>-fast` alias (e.g. `qwen3.6-35b-a3b-fast`) serves the same
weights but applies a **CLI-strip heuristic** before encoding:

- system/developer guidance is dropped;
- tool definitions and tool-call history are dropped;
- `<system-reminder>` blocks (or any tag in `NVMAI_STRIP_TAGS`) inside user
  messages are dropped.

What remains is the real user/assistant conversation — a few hundred tokens.
Measured on the "What is the capital of France?" question (4-bit), the fast
alias answers in seconds where the base model takes minutes:

| CLI | 4-bit base | 4-bit fast | Saving |
| --- | --- | --- | --- |
| Codex | ~250 s | 24.6 s | **~10× faster (≈90%)** |
| OpenCode | ~218 s | 5.9 s | **~37× faster (≈97%)** |
| Qwen Code | >300 s (stalled) | 4.6 s | **>65× faster (>98%)** |

**Full matrix (fast alias, "What is the capital of France?", wall clock).**
Measured on M3 24 GB, macOS 26.6, Swift 6.3.3, commit 14b3f35 + fast-alias
change, one quant per server process, prompt-cache ON / MTP OFF / concise ON,
each CLI pointed at `qwen3.6-35b-a3b-fast`:

| CLI | 4-bit | 6-bit | 8-bit |
| --- | ---: | ---: | ---: |
| Codex `exec` | 24.6 s | 24.7 s | 32.8 s |
| Qwen Code `-p` | 4.6 s | 6.9 s | 12.8 s |
| OpenCode `run` | 5.9 s | 8.2 s | 10.2 s |

All nine cells answered correctly. Codex's wall time includes its client-side
agent overhead; server-side generation for the same request was 16.3 s
(4-bit), 28.3 s (6-bit), 28.3 s (8-bit) at prompt=1,210 tokens. Qwen Code and
OpenCode sent a 19-token prompt after the strip and generated in 2.9-11.8 s.
The Qwen Code cells also disable its post-turn auto-memory subagent
(`memory.enableManagedAutoMemory=false`, as `nvmai-cli.sh` configures).

**Trade-off.** The strip disables the CLI's agentic tool loop for that
request: no tool calls, no file edits. Use the base model when the agent's
tools are needed, the fast alias for direct chat-style questions.

**How it works.** The alias is per-request: request `model` is
`qwen3.6-35b-a3b-fast` (advertised by `/v1/models`), and the strip runs only
for requests that name it. The server environment variable
`NVMAI_STRIP_CLI_PROMPT=1` enables the same heuristic for *all* requests as a
fallback; `NVMAI_STRIP_TAGS` sets the in-message tag list (default
`system-reminder`). Each stripped request logs a strip report, e.g.:

```
strip v2 system=1 tools=59 reminders=5604chars messageFallback=0 requestFallback=false prompt=229
```

If a CLI rewrites its bloat template, the `reminders=`/`prompt=` counters
change visibly in the server log — the early-warning signal that the heuristic
needs a tag-list update or a rebuild.

`tools/nvmai-cli.sh` launches Codex, Qwen Code, and OpenCode against the fast
alias by default; set `NVMAI_STRIP_CLI_PROMPT=0` (or point the CLI config at
the base model) to keep the full agent prompt and tool loop.

## Documentation

- [Wiki](https://github.com/Pummelchen/NVMAI/wiki)

NVMAI remains text-only. The server binds to `127.0.0.1` without authentication
or TLS and must not be exposed through a proxy or tunnel.
