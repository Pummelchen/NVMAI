<img width="1122" height="1402" alt="image" src="https://github.com/user-attachments/assets/b807e4e2-f26f-4cae-9998-3a7fdbe03290" />
# NVMAI

> [!IMPORTANT]
> **Roadmap update — Ornith 1.5**
>
> NVMAI plans to adopt **Ornith-1.5-35B-A3B** as its next default model.
> Qwen 3.6 35B-A3B remains the current supported model while the new integration
> is implemented, validated, and benchmarked. Ornith uses a 35B
> mixture-of-experts architecture with approximately 3B active parameters per
> token; its published coding and agentic results make it a strong candidate for
> NVMAI's bounded-memory Apple Silicon runtime.

[![Published Ornith 1.5 benchmark overview](assets/stats.png)](https://ornith.ai/ornith_1_5.html)

*Publisher-supplied benchmark overview; this is not an NVMAI performance
measurement. See the [Ornith 1.5 announcement](https://ornith.ai/ornith_1_5.html)
and [model card](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B).*

**Answers in seconds, not minutes.** The `<model>-fast` model alias delivers a
**10×–65× reduction in response time** for coding CLIs — Codex, Qwen Code, and
OpenCode. Same model, same weights, same answers: the CLIs' agent boilerplate
is stripped before prefill, so a question that took minutes now answers in
seconds. (Introduced in v3.3; see the current release summary below.)

**The Paris test** — "What is the capital of France?" (4-bit, wall clock):

| CLI | Base model | `-fast` alias | Speed-up |
| --- | ---: | ---: | --- |
| Codex `exec` | ~250 s | 24.6 s | **~10×** |
| OpenCode `run` | ~218 s | 5.9 s | **~37×** |
| Qwen Code `-p` | >300 s (stalled) | 4.6 s | **>65×** |

## Current release — 3.9

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
[docs/v4-core-design.md](docs/v4-core-design.md). See the
[full release history](https://github.com/Pummelchen/NVMAI/wiki/Changelog).

711 tests / 128 suites.

Native Swift and Metal inference for **Qwen 3.6 35B-A3B** in 4-bit and
8-bit quantization on Apple M1-M5 systems. Optional concise mode injects a
per-quantization terse system prompt for direct, lean answers. The fast alias
is available on both supported quantizations — how it works, the trade-off, and
the historical 4/6/8-bit matrix are on the
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
- [Launcher guide](https://github.com/Pummelchen/NVMAI/wiki/OpenAI-Compatible-Server#launchers)
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

Under the 3.8 policy, both quantizations streamed routed experts from SSD inside
that ~1 GB budget. Prefill was GPU-bound (97.4% occupancy at maximum clocks),
while decode was bound by memory bandwidth.

With the 3.8 page-cache policy and a fixed 262144 context, larger expert caches
were slower: 4-bit measured 13.61 tok/s at 1 GB against 9.85 at 4 GB. These are
historical results, not the 3.9 bounded-I/O default. The full matrix
(2 quantizations × 5 budgets × 2 cache policies × 3 prompt sizes) and the
reasoning behind the current defaults are in
[docs/v4-core-design.md](docs/v4-core-design.md).
