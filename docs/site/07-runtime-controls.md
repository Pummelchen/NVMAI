> **Category:** Guides
> **Status:** draft v1 — to be reviewed before posting
> **Wiki source:** Runtime-Controls (the control tables, paraphrased; all defaults and ranges verified)

# Runtime controls

The rules of the road: **start with the defaults. Change one setting at a time. Reload the model when you change a load-time control.** The defaults are tuned per family and are the configuration the benchmarks and the launchers run. Treat the controls below as the lever you reach for *only* when the default doesn't fit your workload.

## Generation controls

| Control | Flag | Default | What it does |
| --- | --- | --- | --- |
| Response limit | `--max-new` | 1,024 | Max generated tokens. |
| Context scaling | `--rope-scaling none\|yarn` | `none` | Native RoPE or YaRN. |
| Context | `--max-context` | Native: CLI/app 4K, server 262K; YaRN: 1M | Prompt + response capacity. |
| KV cache | `--kv-bits 16\|8\|4` | 8 | Live attention-state storage precision. |
| Temperature | `--temperature` | 0.6 | `0` is greedy; positive samples. |
| Top-K | `--top-k` | 20 | Keep at most K candidates; `0` disables. |
| Top-P | `--top-p` | 0.95 | Nucleus truncation. |
| Presence penalty | API `presence_penalty` | 0.0 | Only `0.0` is currently implemented. |
| Repetition penalty | `--repetition-penalty` | 1.0 | Penalize repeated tokens. |
| Seed | `--seed` | Off | Reproducible sampling. |
| Stop text | `--stop` | None | Repeat the flag for multiple. |
| Concise answers | `--concise` | Off | Adds a shorter-answer system prompt. |
| Thinking | `--thinking off\|on` | Off | The model's binary reasoning branch. |

The shared runtime applies **temperature 0.6, Top-P 0.95, Top-K 20, presence penalty 0.0** to Qwen and Ornith at every supported weight precision. Native context supports up to **262,144** prompt-plus-response tokens; YaRN extends to **524,288** or **1,048,576**.

## Context and KV — the two things that eat memory

KV state uses selectable 16/8/4-bit storage (8-bit default) and **grows from 8,192 tokens** on demand rather than reserving max context at load. For the production Qwen/Ornith topology, full-length attention KV uses roughly:

| Context | 16-bit KV | 8-bit KV | 4-bit KV |
| --- | --- | --- | --- |
| 512K | 10.0 GiB | 5.31 GiB | 2.81 GiB |
| 1M | 20.0 GiB | 10.63 GiB | 5.63 GiB |

**Use 4-bit KV when 1M must fit on a 24 GB Mac.** The model weights keep their installed 4/8-bit format — `--kv-bits` changes only the live attention cache. YaRN changes RoPE extrapolation and *can affect quality beyond the native window*; it is not currently compatible with MTP.

```bash
# Native context, default 8-bit weights and KV
.build/release/NVMAICLI --model models/ornith-1.5_35B_A3B_8Bit \
  --prompt "Summarize this code" --max-context 262144

# 1M YaRN with 4-bit KV (1M is the YaRN default; --max-context omitted)
.build/release/NVMAICLI --model models/ornith-1.5_35B_A3B_8Bit \
  --prompt "Summarize this code" --rope-scaling yarn --kv-bits 4

# Explicit 512K YaRN
.build/release/NVMAICLI --model models/ornith-1.5_35B_A3B_8Bit \
  --prompt "Summarize this code" --rope-scaling yarn --max-context 524288 --kv-bits 8
```

## Model loading and I/O

These are the levers behind [SSD expert streaming](#03) and [the RAM budget](#04). Most people never touch them.

| Control | Default | Notes |
| --- | --- | --- |
| `--ram-budget 8G` | Per family | The byte budget for the expert cache; overrides the derived default. |
| `--expert-cache-slots` | Derived (app: 64) | Explicit slot count; overrides the derived value. |
| `--prefill-chunk` | 4096 (35B) | Larger = fewer repeated expert sweeps, more temp memory. |
| `--rdadvise` | `default` | `off` / `default` / `bounded` read advice. |
| `NVMAI_BOUNDED_IO` | On | Bypass the OS page cache so the budget is the *real* working set. |
| `NVMAI_PARALLEL_IO` | On | Fill cache misses concurrently. |
| `NVMAI_DECODE_EXPERT_EXECUTION` | `hit-fixup` | `hit-fixup` runs hits now, fixups misses; `barrier` waits. |
| `NVMAI_EXPERT_IO_BACKEND` | `pread` | `pread` / `metal`. |
| `NVMAI_SAMPLER_PATH` | `tiled` | The three-stage GPU Top-K reduction. |
| `NVMAI_PREFETCH_TOP_M` / `NVMAI_PREDICTIVE_PREFETCH` | Per family | Speculative read of the next layer's predicted experts. |
| `NVMAI_PREFILL_ANE` | On | Full-attention prefill on the Neural Engine (see [ANE prefill](#09)). |

The per-family expert-RAM and prefetch defaults differ because they're set by **how much of a token is expert I/O** — nearly 3× across the shipped families. The 4-bit 35B row is the cautionary tale: expert I/O is only ~7 ms of a 44 ms token there, so prefetch *regresses* (−4%) and is off; at 8-bit the same families stream twice the bytes and it pays (+5%). See [Benchmarking](#11) before flipping any of these.

## Server controls

| Flag | Default | Purpose |
| --- | --- | --- |
| `--rope-scaling` | `none` | Native RoPE or YaRN. |
| `--kv-bits` | 8 | 16/8/4-bit KV storage. |
| `--thinking` | `off` | Binary reasoning mode. |
| `--mtp-model` | Off | Native MTP sidecar; greedy + native RoPE only. |
| `--mtp-memory-mib` | 384 | Strict incremental MTP budget, 256–512. |
| `--queue-limit` | 4 | Max queued requests while one generation runs. |
| `--prompt-cache-mode` | `multi-prefix` | `off` / `single-prefix` / `multi-prefix`. |
| `--prompt-cache-entries` | 4 | Retained prefix count. |
| `--prompt-cache-memory-mib` | 256 | RAM snapshot budget. |
| `--prompt-cache-disk` | Off | Private persistent cache directory. |
| `--lazy-load` | Off | Defer model load to the first inference request. |
| `--idle-unload-seconds` | 0 | Unload after idle; implies lazy load. |

`--help` on `NVMAIServer` has the complete accepted ranges.

## Answer modes: Concise and Thinking

**Concise mode** shortens answers by adding a terse system prompt. It changes **answer length, not decode tokens-per-second** — if it's dropping nuance you need, turn it off. Surfaces: Mac app toggle, CLI `--concise`, server `NVMAI_CONCISE_MODE=1`, launcher 4th arg `concise`.

**Thinking mode** opens the model's reasoning branch. Ornith 1.5 and the compatible Qwen template expose **only `off` and `on`** — there are no Low/Medium/High effort levels and no thinking-token budget, and NVMAI won't dress prompt tricks up as if there were. Thinking can add substantial output and wall time. Every surface defaults to **off**.

## Practical presets (the settings that are known to work)

- **General launcher use:** Ornith 8-bit, `full`, concise off, native 262K, multi-prefix cache, 8-bit KV, MTP off.
- **Direct coding-CLI questions:** 8-bit, `fast` (the alias), optionally concise, thinking off.
- **Agent tool loops:** the **base** model (`full`), not the fast alias.
- **Shared Mac:** server with `--lazy-load --idle-unload-seconds N` plus a private disk prompt cache.
- **Benchmark:** fixed prompt, seed, context, controls — see [Benchmarking](#11).

## Where to go next

- What the RAM budget actually budgets: [The RAM budget](#04)
- Why the first token is slow (prefill) and how to extend context: [Long context and KV cache](#08)
- The ANE prefill flag in depth: [ANE prefill and the Metal engine](#09)

*Version at time of writing: NVMAI 5.1.*
