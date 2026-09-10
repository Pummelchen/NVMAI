> **Category:** Guides
> **Status:** draft v1 — to be reviewed before posting
> **Wiki source:** Runtime-Controls (model loading and I/O), System-Design (bounded expert streaming)

# The RAM budget and the bounded expert cache

NVMAI's promise is "you set the RAM budget, and it holds the line." This article is what that means mechanically and how to set it correctly.

## What the budget actually budgets

A model's RAM use has several parts, and only **one** of them is the knob you control:

| Part | Controlled by | Notes |
| --- | --- | --- |
| Routed-expert cache | **`--ram-budget`** (or derived slots) | *This* is the bounded, tunable part. |
| Shared / always-resident weights | The model | Attention, norms, embeddings, router. Fixed. |
| KV cache (live attention state) | `--kv-bits` + context | Grows with context, not load. |
| Runtime scratch (prefill, MTP, decode) | Various | Temporary. |

`--ram-budget` sets a **byte budget** for the routed-expert cache. The runtime derives the **slot count** from that budget and the model's own expert stride (how many bytes one expert occupies), so the byte figure is the real knob and the slot count is a derived display. Give it `--ram-budget 8G` and it will not exceed 8 GB for experts, whatever else the model wants.

## Why it's clamped to half of physical memory

The per-family tuned default is **clamped to half of physical RAM.** That's deliberate: a budget tuned on a 24 GB machine is wrong on a 16 GB one. So a 16 GB Mac loading Qwen3.8-Flash-Next gets 64 slots, not the 96 that measured fastest on 24 GB. A smaller Mac is not handed a budget that was sized for a bigger one. `--ram-budget` overrides the clamp with any size you like.

## Per-family defaults (and why they differ)

The default is not one number. It's governed by **how much of each token is spent on expert I/O**, which varies nearly 3× across the shipped families:

| Family | Expert RAM | Prefetch | Measured |
| --- | --- | --- | --- |
| Qwen3.8-Flash-Next 4-bit | 12 GiB (96 slots) | depth 1 | 5.74 → 6.96 tok/s (+21.3%) |
| Qwen 3.6 / Ornith 8-bit | 8 GiB (64 slots) | depth 1 | +5.5% / +5.9% |
| Qwen 3.6 / Ornith 4-bit | 8 GiB (128 slots) | off | prefetch measured **−4.2% / −3.9%** |

The 4-bit 35B row is the lesson in a single line: expert I/O there is only ~7 ms of a 44 ms token, so a speculative prefetch has almost nothing to recover and still steals SSD service from a demand read — it measures a *regression*, so it's off. At 8-bit the same families stream twice the bytes and the prefetch pays. **Larger budget is not automatically faster** — a bigger cache means more experts resident (good) but the *optimal* size depends on the drive and the model, so measure on your Mac.

## Smaller vs. larger budgets

- **Smaller budget** → less RAM, but more routed experts must come from the SSD each token → slower decode. Use it to leave room for the OS and other apps.
- **Larger budget** → fewer SSD reads, but only up to the point where your active expert set actually fits, and it costs RAM. Past that it buys nothing.
- **The sweet spot is a measurement, not a guess.** That's the whole philosophy of this project (see [Benchmarking](#11)).

## The other expert-I/O knobs

These are for when you've decided the default budget isn't right and want to dial the mechanism:

| Control | Default | What it does |
| --- | --- | --- |
| `--expert-cache-slots` | Derived (app: 64) | Explicit slot count; overrides the derived value. |
| `NVMAI_BOUNDED_IO` | On | Bypass the OS page cache so the budget is the *real* working set. | Off means the kernel keeps evicted experts around and your number is fiction. |
| `NVMAI_PARALLEL_IO` | On | Fill cache misses concurrently. |
| `NVMAI_DECODE_EXPERT_EXECUTION` | `hit-fixup` | `hit-fixup` runs hits immediately and fixups the misses; `barrier` waits. |
| `NVMAI_PREDICTIVE_PREFETCH` / `NVMAI_PREFETCH_TOP_M` | Per family | Speculatively read the next layer's predicted experts. |
| `--rdadvise` | `default` | `off` / `default` / `bounded` read advice for the SSD. |
| `--prefill-chunk` | 4096 (35B) | Larger chunks reduce repeated expert sweeps, use more temp memory. |

## Two things that are easy to get wrong

- **Hit/fixup changes scheduling, not the budget.** In-flight cache slots are leased until their Metal command completes; they can't be evicted or reused early. So `hit-fixup` vs `barrier` is a throughput question, not a memory one.
- **MTP has its own strict budget** (`--mtp-memory-mib`, 256–512, default 384), on top of the model's budget. It's incremental and enforced, not "a bit more."

## A practical rule

Start with the per-family default (don't pass anything). If your Mac is under pressure, lower `--ram-budget` and re-measure. If you have headroom and decode is SSD-bound, raise it — but verify it actually helped on *your* drive with *this* model. The number you want is a measurement, and [Benchmarking](#11) is how you get it.

## Where to go next

- The mechanism the budget bounds: [SSD expert streaming](#03)
- Every flag in one place: [Runtime controls](#07)
- Measuring the trade-off: [Benchmarking](#11)

*Version at time of writing: NVMAI 5.1.*
