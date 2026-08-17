# Using the idle CPU cores and the spare memory bus

NVMAI decodes with one CPU thread orchestrating and seven idle, and it uses
roughly half the memory bus. This is the plan for spending both.

Every number here was measured on the development machine (M3, 24 GiB, ~100
GB/s unified bus, 150 GB/s internal GPU bandwidth) with Qwen3.6-35B-A3B 4-bit,
192-256 token generations, prompt cache off, 128 expert-cache slots.

## What is actually idle

Decode wall is ~68 ms/token in the best state observed (14.7 tok/s). The GPU is
busy 32-33 ms of that. Occupancy measured across six runs: 39.5, 41.5, 42.9,
45.7, 48.1, 48.5 percent -- **the GPU is idle over half of every token**, and
that ratio is stable even though absolute milliseconds are not (see Method).

Where the 35 ms of idle goes:

| component | ms/token | share |
| --- | ---: | ---: |
| inter-command-buffer gaps | 18.9 | 53% |
| routed-expert pread | 7.8 | 22% |
| outside body (sample/detok) | 4.0 | 11% |
| command-buffer encoding | 1.9 | 5% |
| readback + plan | 1.8 | 5% |
| rdadvise | 1.0 | 3% |

## The two resources are real, and they are close to free

**Spare bandwidth.** A 6-thread NEON streaming read sustains 78 GB/s with the
GPU idle, and 45-60 GB/s *while the GPU is running inference*. It costs the GPU
nothing measurable:

| | tok/s | GPU busy/token | occupancy |
| --- | ---: | ---: | ---: |
| GPU alone | 13.222 | 33.15 ms | 43.8% |
| GPU + 6 CPU threads @ ~52 GB/s | 15.646 | 32.22 ms | 50.4% |

The GPU did not slow down. (It reads faster, but that is warm-up, not
causation -- the claim here is only that concurrent CPU traffic is not harmful.)

**Spare cores.** An int4 dequant + GEMV in NEON, matching the `.gturbo` layout
(4-bit weights, per-64 bf16 scale/bias, fp32 accumulate):

| threads | 13.5 MiB | throughput |
| ---: | ---: | ---: |
| 1 | 3.903 ms | 3.4 GiB/s |
| 4 | 1.093 ms | 12.1 GiB/s |
| 7 | **0.773 ms** | **17.1 GiB/s** |
| 8 | 0.817 ms | 16.1 GiB/s |

13.5 MiB is not an arbitrary size: it is exactly one layer's 8 active experts
(1.688 MiB each, from 16.875 GiB of packed experts over 40 layers x 256). The
per-layer wall budget is 68.1/40 = 1.7 ms.

**So the CPU can compute an entire layer's routed MoE in 0.773 ms, inside a
1.7 ms budget, during time the GPU is idle anyway.**

Note it saturates at 7 threads and consumes 17.1 GiB/s against ~50 GB/s
available. The CPU path is compute-bound on nibble unpacking, not bandwidth-
bound. That is the headroom a re-shard buys (Phase 2).

## Plan

### Phase 0 -- Consolidate command buffers: TRIED, DOES NOT WORK

The original reasoning: 182 command buffers per token, each apparently
carrying ~104 us of gap, and `attn_norm_qkv` -> `attn_tail_router` have no CPU
work between them, so merging them should reclaim ~40 buffers per token.

Implemented and measured. It does not help. Interleaved A/B, 6 samples per arm,
diagnostics on in both arms so `busy_per_token` could confirm the arms were
thermally comparable (27.83 vs 27.64 ms, +0.7%):

| arm | tok/s median | range |
| --- | ---: | ---: |
| split (current) | 21.02 | 18.32-22.27 |
| merged | 20.06 | 19.61-20.79 |

Merged is 4.6% slower by median and 2.4% by mean, which given split's sd of 1.4
is not a significant difference -- but it is emphatically not the predicted
+12-15%, so the premise is falsified either way.

The reason: committing `attnCB` early lets the GPU begin attention while the
CPU is still encoding the tail. The split was buying CPU/GPU pipelining within
the layer, and merging serialises encode-then-commit. The ~104 us figure came
from dividing measured idle by buffer count, which silently assumed the idle
*was* per-buffer overhead; most of it is dependency stall, and that does not
shrink by issuing fewer, larger buffers.

Do not retry this. If per-layer idle is attacked again, the target is the
dependency chain (attention -> router readback -> expert fetch -> MoE), not the
buffer count. Reverted in full; the reverted change is in the history if the
measurement needs repeating.

Phase 1 no longer depends on this. The worry was that adding a rendezvous to a
sync-saturated path would be expensive; since the path turns out not to be
sync-saturated -- fewer buffers did not help -- Phase 1 can go first.

### Phase 1 -- CPU computes a share of the routed experts

Per layer, split the 8 active experts between GPU and CPU, with one rendezvous
per layer to sum partial outputs. Coarse-grained on purpose: 40 rendezvous per
token, matching the sync count the layer loop already pays, rather than one per
matmul.

The CPU reads expert weights straight from the mmap'd file, so its share also
skips the pread into a GPU slot -- removing part of the 7.8 ms I/O and the
1.0 ms rdadvise along with the compute.

**Corrected estimate: ~6%, not the +25-40% first written here.** The original
figure came from checking that the CPU could do a layer inside the per-layer
budget. It can. What it never checked was whether the CPU is *faster than the
GPU already is* at the same work. It is not:

| | per expert |
| --- | ---: |
| GPU (`moe_phase1_2_routed`, 9.764 ms/token over 320 evals) | 0.031 ms |
| CPU, all 8 cores (0.516 ms/layer over 8) | 0.065 ms |

The GPU is 2.1x faster than all eight cores combined, measured with the GPU
otherwise idle -- under contention the CPU is worse still. Moving work from the
faster unit to the slower one only pays at the balance point: the expert phase
drops from 0.205 ms/layer to 0.139 ms, which is 2.6 ms/token, about 6%.

Worth doing as a cheap win once the plumbing exists, but it is not the main
event, and nothing here scales to 2x.

### Phase 2 -- Re-shard for a CPU-friendly layout

The CPU is compute-bound on 4-bit unpacking, not on bandwidth. Since the model
can be repacked, store the CPU-designated expert share in a layout that skips
that cost -- int8, or 4-bit pre-swizzled so a NEON lane loads without shift and
mask. Roughly 2x on the CPU path is plausible, which either raises the CPU's
share or frees cores.

This is a `NVMAIRepack` change plus a second expert section in the `.gturbo`
container. It does not need a re-download; it is a repack of installed weights.

Estimated +10-15% beyond Phase 1, and it is the phase that makes the split
tunable rather than fixed by dequant cost.

## The ceiling, and what a 2x would actually require

Adding parallel compute units cannot double throughput here, and the reason is
arithmetic rather than engineering. At 21 tok/s a token is 47.6 ms: 27.6 ms of
GPU-busy and 20 ms of idle. Doubling means 23.8 ms/token, which is less than
the GPU's own work. Even with the idle driven to zero -- infinite CPU, a
perfect ANE, no stalls -- the result is 36 tok/s, 1.7x. That bounds every
"use more units" idea in this document.

Decode at batch 1 is a serial dependency chain: layer L+1 needs layer L's
output, and within a layer the experts need the attention result. There is no
independent work to run alongside. Only the expert set can be split, and that
is the 6% above.

The lever that does reach 2x is moving fewer bytes. Per token the model reads
roughly 1.8 GB -- attention ~900 MB, experts 540 MB, lm_head ~240 MB, shared
~120 MB. At ~100 GB/s that is an 18 ms floor, so 4-bit caps out near 55 tok/s
and we are at 21, using about 39% of the bus.

| change | GPU busy | plausible total | tok/s |
| --- | ---: | ---: | ---: |
| now (4-bit) | 27.6 ms | 47.6 | 21 |
| + CPU expert split | 25.0 | 45.0 | 22 |
| 3-bit | 20.7 | ~33 | ~30 |
| 3-bit + idle halved | 20.7 | ~26 | ~38 |
| 2-bit experts / 3-bit attention + idle halved | ~15 | ~22 | ~45 |

Two things follow. Attention is the largest consumer, not the experts, so
byte-cutting should start there. And the table assumes GPU time scales with
bytes; that is consistent with the measured per-role rates (~55-65 GB/s across
attention and MoE alike) but it is an assumption, and the cheap way to test it
is to requantise to 3-bit and check whether `busy_per_token` falls
proportionally. If it does not, the GPU is not purely bandwidth-bound and every
row below the first is optimistic.

## What this does not do

Speculative decoding is not on this list and should not be revived by it. MTP
loses for a structural reason measured separately: verify cost tracks the
*union* of the rows' routed experts (1.585x at width 2) which cancels the
tokens emitted per pass (1.574), and wider blocks diverge because benefit is
capped at 1/(1-p) while the union grows unbounded. Adding CPU capacity does not
change that ratio.

## Risks

**Numerics.** CPU fp32 accumulation will not match the GPU bit-for-bit, so the
golden baseline changes. This needs an explicit decision before Phase 1 lands:
either re-baseline and accept that CPU/GPU split ratios are part of the
reproducibility contract, or constrain the CPU path to an accumulation order
that matches. The second is much harder and probably not worth it.

**Rendezvous cost.** Phase 1 adds a CPU/GPU sync per layer. If it lands before
Phase 0, it competes with the very overhead that dominates the idle budget.

**Thermal.** Loading seven cores raises package power and can downclock the
GPU. Absolute GPU time already varies ~1.7x with thermal state on this machine
(`attn_norm_qkv` measured 13.9 and 22.7 ms/token in different sessions, and it
touches no expert weights). Phase 1 must be validated with `busy_per_token`
reported alongside tok/s.

## Method

Absolute milliseconds here are not portable across sessions; ratios are. Any
A/B on this work must:

- interleave the arms (A/B/B/A), never sweep them in sequence -- a sequential
  sweep warms the page cache as it goes and manufactures a monotone "win"
- discard the first request after each server start
- take 4+ samples per arm and report the spread
- report `busy_per_token_ms` next to tok/s; if it moves between arms, the arms
  ran in different thermal states and the comparison is void

Two false positives came from skipping this: 32 MTP sidecar slots "beating" 8
by 11%, and 128 expert-cache slots "beating" 64 by 15%. Both vanished under an
interleaved retest.

`NVMAI_KERNEL_STATS=1` reports merged GPU occupancy; `TURBO_FIELDFARE_PHASES=1`
reports per-chunk route/tile/tail split and active experts per layer.
