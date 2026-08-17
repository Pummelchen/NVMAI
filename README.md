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

**New in 3.8 — performance investigation release**

A measured investigation into decode throughput. One real fix, new instrumentation
that made it findable, and a documented ceiling so the next attempt starts from
evidence rather than assumption.

- **Decode is ~7% faster.** The shared dense MLP was encoded *after* the router
  readback, so the GPU sat idle through the whole round trip before that work was
  submitted — it depends only on the post-attention norm and could always have
  been queued earlier. GPU idle fell from 19.9 to 16.7 ms/token and occupancy rose
  from 57.9% to 61.1%, verified byte-identical against golden reference output.
- **Instrumentation you can act on.** `NVMAI_KERNEL_STATS` now reports true GPU
  occupancy by merging overlapping command-buffer intervals — the previous metric
  summed them and could exceed 100% — plus per-transition gap attribution showing
  *where* the GPU idles. `TURBO_FIELDFARE_PHASES` reports active experts per
  layer, and `NVMAI_ROUTE_TRACE` dumps real per-layer routing.
- **A C99/NEON expert kernel**, 3.4× faster than its Swift equivalent, with a new
  `NVMAIKernelsC` target for hot loops where Swift's vector types do not lower
  well. Validated against an independent reference. Not yet on the decode path —
  see the ceiling below for why.
- **The throughput ceiling is now known.** Batch-1 decode moves ~1.8 GB per token
  and this hardware sustains ~64 GB/s, so ~36 tok/s is the maximum for 4-bit on an
  M3. Measured, not estimated: eliminating GPU idle, running entirely on the ANE,
  and running entirely on the CPU all converge on the same number, because none of
  them is the bottleneck. Adding compute units *costs* throughput — CPU load
  raises GPU-busy 45%, ANE load 89%, since every unit draws on one memory
  controller.
- **Expert-cache defaults are now derived per quantization** from a ~1 GB budget
  and the model's own expert stride: 16 slots at 4-bit, 8 at 8-bit. The previous
  fixed default of 64 slots was both slower and four times larger. An earlier
  version of this note reported 6-bit and 8-bit at 6.7 and 1.6 tok/s; those figures
  were taken at 64 slots and no longer apply.

Full method, including nine approaches that measured out negative and why:
[performance investigation](https://github.com/Pummelchen/NVMAI/blob/main/docs/cpu-coexecution-plan.md).

689 tests / 124 suites.

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

Measured on M3 MacBook Pro, 24 GB, at the shipped defaults: `--max-context
262144`, prompt cache off, and the expert-cache budget the engine derives for each
quantization (16 slots at 4-bit, 8 at 8-bit — about 1 GB either way). Greedy,
temperature 0. Prompt sizes are 25 / 452 / 3532 tokens.

| | 4-bit | | 8-bit | |
| --- | ---: | ---: | ---: | ---: |
| prompt | prefill | decode | prefill | decode |
| short (25 tok) | 2.50 s | **13.61 tok/s** | 4.34 s | **5.64 tok/s** |
| medium (452 tok) | 6.04 s | 12.46 tok/s | 11.19 s | 4.94 tok/s |
| long (3532 tok) | 47.15 s | 7.54 tok/s | 61.84 s | 3.83 tok/s |
| expert-cache RAM | | 1.05 GB | | 1.00 GB |

Both quantizations stream routed experts from SSD inside that ~1 GB budget, which
is the point of the project: a 35B MoE that leaves the machine usable. Prefill is
GPU-bound (97.4% occupancy at maximum clocks), so it is insensitive to the cache
budget; decode is bound by memory bandwidth.

Larger expert caches are **slower**, not faster, at the shipped 262144 context —
4-bit measures 13.61 tok/s at 1 GB against 9.85 at 4 GB — because the KV
reservation is already large enough that extra slot memory pushes the machine into
pressure. Full matrix (2 quantizations × 5 budgets × 2 cache policies × 3 prompt
sizes) and the reasoning behind every default:
[docs/v4-core-design.md](docs/v4-core-design.md).


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
