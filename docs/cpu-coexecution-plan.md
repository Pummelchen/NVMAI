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

### Phase 0 -- Consolidate command buffers

182 command buffers per token (40 layers x ~4.5, plus head and embed), each
carrying ~104 us of gap. Not every boundary has a CPU dependency:
`attn_norm_qkv` -> `attn_softmax` -> `attn_tail_router` have none between them
and can merge, as can `shared_expert` with `moe_phase1_2_routed`. Target ~100
CBs/token.

Estimated +12-15%. Do this first: it is the largest single item, and every
co-execution design below adds rendezvous, which is cheaper on a path that is
not already sync-saturated.

### Phase 1 -- CPU computes a share of the routed experts

Per layer, split the 8 active experts between GPU and CPU, with one rendezvous
per layer to sum partial outputs. Coarse-grained on purpose: 40 rendezvous per
token, matching the sync count the layer loop already pays, rather than one per
matmul.

The CPU reads expert weights straight from the mmap'd file, so its share also
skips the pread into a GPU slot -- removing part of the 7.8 ms I/O and the
1.0 ms rdadvise along with the compute.

Balance point: CPU at 17.1 GiB/s does all 8 experts of a layer in 0.773 ms
(30.9 ms/token for all 40 layers), while the GPU's whole routed-MoE role is
9.8 ms/token. Giving the CPU everything would make it the critical path, so the
split should be tuned, starting near 3-4 experts of 8 and measured.

Estimated +25-40% on top of Phase 0. This is the main event.

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
