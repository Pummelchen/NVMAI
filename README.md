<img width="1122" height="1402" alt="image" src="https://github.com/user-attachments/assets/b807e4e2-f26f-4cae-9998-3a7fdbe03290" />
## Announcement — Ornith-1.5-35B-A3B will be NVMAI's new default model

NVMAI will work on support for **Ornith-1.5-35B-A3B** as its new default AI model,
replacing Qwen 3.6 35B-A3B. It is a 35B mixture-of-experts model that activates
only ~3B parameters per token, so it fits the same bounded-memory architecture and
Apple M1–M5 hardware NVMAI already targets — while outperforming Qwen 3.6 35B-A3B
on the coding and agentic benchmarks that drive this project.

![LLM Performance Evaluation](assets/stats.png)

Source: [Ornith-1.5 announcement](https://ornith.ai/ornith_1_5.html) and
[ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B).


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

**New in 3.9 — streaming and memory release**

Four changes to how weights and context reach the GPU, all verified byte-identical
against golden reference output on 4-bit and 8-bit.

- **The default context no longer costs throughput.** KV storage was sized for
  `--max-context`, so the shipped 262144 reserved 512 MiB per layer and 20.0 GiB
  across 40 — lazily touched, but the mappings alone cost real time on a 24 GB
  machine. It now starts at 8192 tokens and doubles on demand. The gap between
  running at 8192 and at the 262144 default went from **1.63× to 1.02× — gone.**
- **Expert reads bypass the page cache by default.** The slot budget is now the
  machine's actual footprint rather than a figure the OS quietly supplements.
  `NVMAI_BOUNDED_IO=0` restores the old behaviour.
- **`--ram-budget` is the knob.** Give it a size (`8G`, `2G`, `512M`) and the slot
  count is derived from it and the model's own expert stride. Defaults are derived
  per quantization: 128 slots at 4-bit, 64 at 8-bit. `--expert-cache-slots` still
  overrides.
- **6-bit is withdrawn.** Its non-power-of-two packing measured 46.8 GB/s against 60
  for both 4-bit and 8-bit, and a 26 GB model does not fit 24 GB. A 6-bit model now
  refuses to load with an explanation rather than a generic error.

Also: a C99/NEON target for hot loops, a parallel expert reader, and instrumentation
for GPU occupancy, per-transition idle attribution and real routing traces.

Method, every measurement, and the approaches that were tried and abandoned:
[docs/v4-core-design.md](docs/v4-core-design.md).

711 tests / 128 suites.

Native Swift and Metal inference for **Qwen 3.6 35B-A3B** in 4-bit and
8-bit quantization on Apple M1-M5 systems. Optional concise mode injects a
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

Measured on M3 MacBook Pro, 24 GB, `--max-context 262144`, prompt cache off,
greedy, temperature 0, prompt sizes 25 / 452 / 3532 tokens.

**These figures are from 3.8**, taken with the page-cache read policy and a 1 GB
expert budget. 3.9 changes both defaults, and the machine used for this table is not
currently quiet enough to re-measure honestly — absolute throughput here swings
about 2× with thermal and memory state, which is itself documented in
[docs/v4-core-design.md](docs/v4-core-design.md). The relative result quoted in the
3.9 notes above (the default-context penalty going from 1.63× to 1.02×) is a
same-conditions comparison and does not depend on machine state. Treat the table as
a floor for 3.9 rather than a description of it, and re-measure on a quiet machine
before quoting it.

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
  `127.0.0.1`: 4-bit → 8081, 8-bit → 8083.
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
tools/cli_launcher.sh qwen full 8 default think      # Qwen Code, full agent, 8-bit, full answers, reasoning
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
