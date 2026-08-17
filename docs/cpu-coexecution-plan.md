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

## Killing the per-layer round trip: what it would take

After the shared-MLP fix the idle is ~16.7 ms/token, and the bulk of it is one
transition, `shared_expert -> moe_phase1_2_routed`, at 10.6 ms over ~35 layers.
Roughly 5.6 ms of that is pread; the rest is a CPU round trip per layer of
which only ~0.04 ms is actual CPU work.

Two cheap attacks on it have now failed. Merging command buffers costs the
CPU/GPU pipelining that committing `attnCB` early buys. Spinning instead of
blocking costs GPU clocks. Neither touches the cause.

The cause is that the CPU must see the routing before the experts can be
dispatched -- not to encode the dispatch, which could be done with GPU-side
indices, but to know which experts are *missing* and must be `pread` into
slots. Streaming is what forces the readback. Indirect dispatch alone does not
remove it: a kernel reading expert ids from a GPU buffer still cannot compute an
expert whose weights are not resident.

So the redesign is to stop streaming, and the enabler is already in the tree.
`ResidentBuffer` mmaps its range `PROT_READ, MAP_PRIVATE` and wraps it with
`makeBuffer(bytesNoCopy:)`, so the GPU already reads non-expert weights straight
out of mmap'd file pages. Applying the same to `packed_experts/layer_NN.bin`
would give the GPU the whole expert set addressably, with the OS page cache
deciding residency:

  - no pread, so the 5.6 ms goes
  - no slot bookkeeping and no per-layer argument buffer rebuilt from routing
  - no readback needed for correctness, so the round trip can go too, with the
    MoE kernel indexing experts from the router's own GPU output

Note what it also removes: at 128 slots the streamer allocates 128 x 1.688 MiB
x 40 layers, about 8.4 GB of anonymous memory. `ResidentBuffer` already carries
a comment recording that this hurts even when pinned. File-backed pages are
evictable; anonymous ones are not.

What it costs is the bounded-memory guarantee, which is a real property and not
one to trade away casually -- it is why the streamer exists. On this machine an
18 GB model against 24 GB of RAM mostly fits, and the 6/8-bit measurements
above show exactly what happens when it does not. Any move here should keep the
streaming path selectable rather than delete it.

This is a substantial piece of work, not a patch: it changes how weights reach
the GPU, and it needs the golden baseline plus a memory-pressure story before
it could be trusted.

### First prototype: the mechanism works, the memory story is unresolved

A standalone Metal program reading the packed expert files three ways, same
bytes each time:

| source | throughput |
| --- | ---: |
| one layer (432 MiB), pread into anonymous memory | 282.4 GB/s |
| one layer, zero-copy mmap | **404.3 GB/s** |
| all 40 layers mapped, full sweep (16.9 GiB) | 0.2 GB/s |

The first two settle the mechanism: the GPU reads mmap'd file pages 43% faster
than the pread'd anonymous slots used today, so there is no penalty for
dropping the copy. (Both are far above bus rate because a 432 MiB layer stays
resident and the kernel is a trivial streaming read; the comparison between
them is what matters, not the absolute.)

The third is the open question, and it is not as damning as it looks. Touching
all 16.9 GiB per pass thrashes a 24 GiB machine -- 83 s, and a second pass at
89 s with no caching benefit. But decode never does that. It reads 8 of 256
experts per layer, and the locality is high enough that 128 slots already hit
~92%. So this measures a worst case that the access pattern does not produce.

What it does establish is the failure mode: when the working set exceeds RAM,
mmap degrades to disk speed rather than degrading gracefully, and there is no
knob to bound it. The slot cache bounds it by construction. That is the
property being traded, and it is why the streaming path should stay selectable.

### Second prototype: the real access pattern, measured

`NVMAI_ROUTE_TRACE=<path>` dumps `position layer e0..e7` for every decode layer,
so the question can be answered from what decode actually does rather than from
a synthetic sweep. A 383-token generation at 4-bit, 128 slots:

| | |
| --- | --- |
| lifetime distinct experts/layer | min 139, median 189, max 255 of 256 |
| lifetime working set | **12.87 GiB of 16.88** |
| experts shared with the previous token | 3.01 of 8 (**38% reuse**) |
| 8-token window | 33.9 experts/layer -> 2.23 GiB |
| 32-token window | 76.1 -> 5.01 GiB |
| 128-token window | 131.4 (peak 232) -> 8.66 GiB |

This settles the question the full sweep could not. A normal-length generation
touches 76% of every expert weight in the model, and routing turns over fast
enough that only 38% of a layer's experts survive to the next token. The
working set is not a small hot subset that would sit comfortably in page cache;
it grows steadily with generation length toward the whole 16.9 GiB.

So on a 24 GiB machine an unbounded mmap would need ~13 GiB of expert pages
resident alongside everything else, and more for longer outputs. That is inside
RAM but not comfortably, and the failure mode past it is the 0.2 GB/s measured
above rather than graceful degradation. The slot cache bounds the same working
set to 8.4 GiB and reaches ~92% hits precisely because 128 slots covers the
mean of a 128-token window (131 experts/layer) even though not its peak (232).

The honest conclusion is that mmap is not the clear win the per-byte numbers
suggested. It reads 43% faster and avoids duplicating file pages into anonymous
slots, but it gives up the bound on a working set that this trace shows runs to
most of the model. Worth doing as a *selectable* mode for machines with headroom
over the model size; not worth doing as a replacement.

A replay harness driving both paths from the trace was attempted and is not
finished -- it holds the 8.4 GiB of slots while also mapping 16.9 GiB and never
frees between modes, so it thrashes the host rather than measuring it. The
trace and the analysis above stand on their own; a corrected harness would
refine the comparison rather than change the conclusion.

### The fused greedy head: TRIED, IT IS SLOWER

The server passes `forceLogitsHead: true` unconditionally, so the fused greedy
head never runs there, and it looked like free throughput for temperature-0
requests: the argmax happens on the GPU instead of moving a 151936-entry logit
vector to the CPU, which should save part of `head_ms` (3.7 ms/token) and most
of the `head_logits -> embed` gap (1.2 ms/token).

The CLI already selects it (`forceLogitsHead: !config.isPureGreedy`), so the
comparison needs no code change -- t=0 takes the fused path, t=0.7 the logits
path, same binary. Interleaved:

| head | tok/s | mean |
| --- | --- | ---: |
| logits (t=0.7) | 12.892, 12.446 | 12.67 |
| fused greedy (t=0) | 12.377, 12.206 | 12.29 |

The fused head is ~3% *slower*, and that is with the sampling work included on
the logits side. Avoiding the transfer does not pay for whatever the fused
kernel gives up -- most likely a full-vocabulary argmax inside one dispatch
against a plain GEMV the CPU then reduces.

So there is nothing to move to the server, which is the useful outcome: making
the head path per-request would have meant changing the sampling contract in
`RawCompletion`, shared by the CLI, the server and the decode service, and the
golden baseline is greedy-only so it would not have covered the sampled path
that change puts at risk.

## An idle core is not a free resource

Spinning on `MTLCommandBuffer.status` before parking looked like an obvious win:
`waitUntilCompleted` needs an OS wakeup after the GPU signals, gap attribution
put ~0.25 ms per layer in exactly that window, and seven cores sit idle while
the orchestrator waits. Interleaved A/B, 400 us budget:

| spin | tok/s | busy/token | occupancy |
| ---: | ---: | ---: | ---: |
| 0 (blocking) | 21.43, 21.27 | 27.26, 27.09 | 58.4%, 57.6% |
| 400 us | 17.77, 18.78 | 29.99, 30.03 | 53.3%, 56.4% |

15-17% slower, and `busy_per_token` rose 10%. That second number is the point:
the GPU's own compute slowed down. CPU and GPU share a package power budget on
this SoC, so occupying a core makes the GPU downclock.

This qualifies the bandwidth measurement above. A 6-thread *streaming read*
sustained 45-60 GB/s during inference without slowing the GPU; a tight polling
loop on one core cost 10% of GPU throughput. The difference is power draw, not
bandwidth. Any plan that spends CPU cycles has to be measured with
`busy_per_token` reported, because the cost lands somewhere the tok/s number
alone will not explain.

It also puts a caveat on Phase 1 that was not there when it was written. The
CPU expert kernel is a dense NEON loop across several cores -- far closer to
the spin than to the streaming read. Its ~6% estimate assumes the GPU keeps its
clocks, and this says that assumption needs testing before the work is trusted.

## Reshaping the weights: TRIED, NOTHING TO RECOVER

Reducing precision is not an option -- NVMAI offers 4/6/8-bit as a user-facing
quality choice -- so the lossless version of "move fewer bytes" is reshaping:
same values, better arrangement. Two measurements say there is nothing there.

Inlining each row's scales and biases next to its weights, so a row is one
contiguous read instead of three streams from distant regions, measured -1 to
-2%. The hardware prefetchers already handle three sequential streams.

More decisively, the GPU is close to saturated. Measured on 4-bit, 128 slots,
warm: 28.86 ms of GPU-busy per token against 1885 MiB of active weights, which
is **68.5 GB/s**. The best pure streaming read this machine produces -- no
compute, 6 threads, just summing bytes -- is 78 GB/s. So the GPU is at roughly
88% of what the memory system actually delivers, not 68% of a nominal 100.

A reshape pays when the layout causes wasted or inefficient reads. At 88% of
achievable bandwidth there is no such waste left to reclaim.

## Quant variants do not fit this machine

Measured on the 24 GiB development machine, same prompt, warm:

| quant | on disk | tok/s |
| --- | ---: | ---: |
| 4-bit | 18 GB | 18.83 |
| 6-bit | 26 GB | 6.68 |
| 8-bit | 34 GB | 1.64 |

6-bit and 8-bit exceed RAM and page continuously; 8-bit occupancy fell to 16.9%,
which is disk thrash rather than anything about the GPU. Treat any 6/8-bit
throughput number from this machine as invalid for kernel work.

This is also a product finding: the quality/speed choice is a cliff rather than
a gradient on this hardware class, and selecting 8-bit costs 11x throughput.
Worth a warning at model-selection time keyed to installed RAM.

## What this does not do

Speculative decoding is not on this list and should not be revived by it. MTP
loses for a structural reason measured separately: verify cost tracks the
*union* of the rows' routed experts (1.585x at width 2) which cancels the
tokens emitted per pass (1.574), and wider blocks diverge because benefit is
capped at 1/(1-p) while the union grows unbounded. Adding CPU capacity does not
change that ratio.

## mlx-dspark: what transfers and what does not

`https://github.com/ARahim3/mlx-dspark` reports roughly 1.3x on this same model
with EAGLE-family drafters (DSpark, semi-autoregressive) and a block-diffusion
variant (DFlash), plus a small-M quantised matmul and hardware-aware cap
calibration. Only one part of that is worth taking here.

**The drafter transfers.** NVMAI's MTP problem is acceptance, not
implementation: break-even needs p > 0.585 and the current drafter reaches
0.574, missing by 1.1 points. A trained EAGLE-family drafter attacks exactly
that. Running their acceptance range through the union costs measured here:

| acceptance | B(2) | vs C(2)=1.585 | net |
| ---: | ---: | ---: | ---: |
| 0.574 (today) | 1.574 | below | 0.99x, 0.85x with overheads |
| 0.75 | 1.75 | above | ~1.10x |
| 0.80 | 1.80 | above | ~1.14x |

Widening still does not pay even at p=0.80 -- width 3 gives ~1.16x, width 4
~1.15x -- because C(t) keeps climbing while B(t) saturates at 1/(1-p). So the
ceiling is ~1.15-1.3x, which is where their published figure lands. Two
independent derivations agreeing is the best evidence available that the model
above is sound.

**The small-M matmul does not transfer.** This was the first hypothesis in this
investigation and measurement killed it. The 2-row verify already amortises
through `useTwoRowProjection` -- one weight read feeding both rows -- and the
expert side is bounded by how many *distinct* experts must be read, not by
matmul shape. A better tile does not reduce the union.

**Block diffusion is a worse fit here than in a dense model.** DFlash-style
block drafting pays the union tax at every additional width, and the measured
curve is brutal for it: 5.18x at width 13, 11.25x at width 42, against benefit
capped at 2.35. Techniques that amortise well on dense models fight this
architecture.

**Priority.** The drafter is a real 1.15-1.3x but it is an ML project -- port
or train an EAGLE-family drafter for Qwen3.6, with model-quality risk and a new
artifact to validate. Against that, ~16.7 ms of GPU idle remains in a ~47 ms
token; closing it is worth up to 1.78x, is pure engineering, and part is
already banked. The two are independent and multiply, so finish the idle work
first and stack the drafter afterwards if it is still wanted.

One caveat on provenance: the description of their approach comes from reading
the repository earlier in this work, not from a fresh review. The numbers drawn
from NVMAI's own measurements stand on their own; their internals should be
re-verified before any port is committed to.

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

## Lossless compression: TRIED, THERE IS NOTHING TO COMPRESS

Question asked: can lossless compression cut the bytes moved from SSD, and
across the bus between RAM, CPU and GPU? Measured across the whole model. The
answer is no, and the first version of this section got it badly wrong by
sampling one layer.

Full scan, all 40 layers x 256 experts, weight regions only (gate, up, down):

| | zero 32-byte groups |
| --- | ---: |
| model-wide | **0.75%** |
| layer 0 | 27.60% |
| every other layer | 0.00-1.10%, median 0.01% |

Expert weights go 16.88 GiB -> 16.75 GiB. There is no sparsity to exploit.

Compressibility on representative layers, versus the layer-0 outlier:

| layer | nibble entropy | zlib | GPU-decodable per-group width |
| ---: | ---: | ---: | ---: |
| 0 | 3.155 / 4 bits | 57.5% | 71% |
| 20 | 3.712 | 93.7% | 100% |
| 39 | 3.705 | 93.5% | 100% |

Representative layers carry 3.71 of 4 bits, 93% of the maximum. That is what a
well-matched affine quantiser is supposed to produce -- the per-group min/max
spreads the codes close to uniform, and uniform codes are incompressible by
definition. zlib recovers 6.5%, and only by CPU decompression that cannot run
inline in a GPU kernel. The scheme a GPU could decode -- per-group bit width --
recovers exactly nothing.

Non-expert weights were already measured incompressible: 100% of groups need all
4 bits, and they are the larger consumer at ~900 MiB/token.

So the whole line of attack is closed. No SSD saving worth the decompression, no
bus saving available at all, and no repack that changes what fits in RAM.

### How the first version went wrong

An earlier revision of this section claimed 54.7% of expert weights were exactly
zero, projected ~+10% on 4-bit, and projected that a repack would shrink 6-bit
to ~13 GiB and 8-bit to ~17 GiB -- making both usable on a 24 GiB machine. All
of that was wrong. It came from sampling `layer_00`, which is a genuine outlier
at 27.6% zero groups against a 0.01% median.

The caveat was even written down at the time ("54.7% is one expert's figure ...
a full scan should precede any repack") and the case was built anyway. The scan
took four minutes. Sample breadth before building an economic case on a number,
and weight the theory: a 4-bit affine quantiser producing near-uniform codes was
the expected result, and it is what the model does everywhere except layer 0.

### Compress-in-transit as an architecture

The idea considered separately from storage format: a reader decides what is
needed, a feeder gathers it, compresses losslessly, ships the compressed package,
and the destination decompresses. Sound engineering -- it is what nvCOMP does
over PCIe -- and it fails here for three independent reasons, any one sufficient.

**1. The payload does not compress.** 92% on a representative layer (above). The
4-bit affine quantiser already *is* the compression, and at 3.71 of 4 bits of
entropy it is close to its own limit. A second lossless layer on top of a
near-entropy-limit encoding has nothing left to take.

**2. The decompressor is far too slow.** Measured on a representative layer,
64 MiB sample:

| codec | ratio | decompress |
| --- | ---: | ---: |
| zlib-1 | 92.3% | 0.51 GB/s |
| zlib-6 | 91.9% | 0.53 GB/s |
| lzma | 92.5% | 0.04 GB/s |

Expert bytes alone are 540 MiB/token, which at 21 tok/s is 11.9 GB/s sustained,
and the GPU reads at ~68 GB/s effective. Eight cores of zlib is ~4 GB/s -- three
times too slow to feed even the current rate, before counting that occupying all
eight cores downclocks the GPU (measured above at -15%). A much faster codec
(LZ4 class, ~4 GB/s/core) would reach ~32 GB/s across eight cores, still under
the GPU's read rate, to save 8% of bytes. The arithmetic never closes.

**3. On unified memory there is no CPU-to-GPU transfer to compress.** This is the
structural point. The architecture assumes a discrete GPU where a copy crosses a
link that compression can shrink. Apple Silicon has no such copy: the GPU cores
read the same physical RAM through the same memory controller the CPU uses. There
is no interposable step between "RAM" and "GPU" to put a codec in. Reducing
GPU-to-RAM traffic requires the data to sit compressed *in RAM* and be decoded
*inside the shader*, which is the per-group scheme that measured 0% available.

The SSD-to-RAM leg is the only one where the architecture applies at all, and
there it reduces to compressed-at-rest, since an SSD can only return stored
bytes. That buys 8% fewer disk bytes for a decompressor 20x slower than the read
path it replaces.

### nvCOMP specifically

Cannot be tried: nvCOMP is CUDA-only. Verified on this machine -- Apple M3, no
CUDA toolchain, and no `nvidia-nvcomp` distribution exists for any non-CUDA
platform. The equivalent here would be writing an entropy-decode kernel in
Metal, which is what nvCOMP's ANS codec does on NVIDIA hardware.

The entropy measurement already bounds what that could win. At 3.712 of 4 bits
per code, the floor for *any* lossless codec is 92.8% of current size. zlib
measured 92.0%, i.e. already at that floor -- which also means there is no
inter-symbol structure left for a cleverer codec to exploit. So a perfect Metal
ANS decoder saves ~7% of bytes: GPU-busy 27.6 -> 25.6 ms, token 47.3 -> 45.3,
about +4% at the absolute ceiling and before subtracting the shader ALU the
decode itself costs on a kernel that is already bandwidth-bound at 88% of
achievable.

Not worth building. The 7% ceiling is set by the quantiser having already done
the compression well.

## The decisive measurement: CPU work makes the GPU 45% slower

Everything above assumed the idle cores were close to free, on the strength of a
streaming-read test that pulled 45-60 GB/s during inference without hurting the
GPU. That test was wrong about the workload. Running the *actual* CPU expert
kernel (8 threads, int4 dequant + GEMV) as a load generator during decode:

| CPU load | tok/s | GPU busy/token |
| --- | ---: | ---: |
| off | 19.886 | 27.335 ms |
| 8 threads of real dequant work | 15.387 | **39.608 ms** |

GPU-busy rose **44.9%** and throughput fell 22.6%. A sequential streaming read is
prefetch-friendly and low-power; a dequant kernel is neither, and it competes
with the GPU for both the memory controller and the package power budget.

This kills CPU co-execution outright, and the arithmetic shows why it cannot be
recovered by tuning the split. At best the CPU absorbs ~32% of the work, but the
GPU then does the remaining 68% at 0.69x speed -- slower overall than the GPU
doing all of it alone. There is no split ratio that wins.

Phase 1 and Phase 2 above are therefore dead, not deferred. The CPUExpertFFN
kernel and NVMAIKernelsC target remain in the tree because they are correct,
tested, and cheap to keep, but nothing should call them from the decode path.

### What this leaves for a 2x

Three resources were nominated. Measured:

- **CPU cores**: net negative (above). Not idle capacity -- competing capacity.
- **ANE**: unreachable. Core ML only, no forced placement, fp16/int8 only, and a
  statically compiled graph cannot express per-token expert routing.
- **Memory bus**: only 38% utilised *averaged over the token*, but that average
  is dragged down by the idle. During the 27.6 ms the GPU is actually working it
  pulls ~65 GB/s against ~78 GB/s achievable on this machine -- 83%. The bus is
  not the slack it appears to be.

So the only real slack is the idle itself, 16.7 ms of a 47.3 ms token. Driving it
to zero yields 27.6 ms/token, 36 tok/s, **1.71x** -- and that is the credible
ceiling.

2x needs 23.65 ms/token. Reaching it requires the idle *and* raising GPU
bandwidth efficiency from ~65 to ~78 GB/s during compute, roughly +20% on the
kernels themselves. Whether that headroom exists is unmeasured; the GPU is
already at 83% of what a pure CPU streaming read achieves, so it is a narrow
target, not an obvious one.

## ANE: the blocker is the dependency chain, not the API

Corrections to the earlier dismissal, which was right in conclusion but thin on
detail. `MLModelConfiguration.computeUnits = .cpuAndNeuralEngine` does let you
*exclude* the GPU, even though no `.neuralEngineOnly` exists, so placement is
more controllable than "you cannot request ANE" suggested. `MLComputePlan`
exposes per-operation supported and preferred devices programmatically, and the
Core ML Instrument verifies which block actually executed -- both better than
inferring from timings. And current MPSGraph documentation does claim CPU, GPU
and Neural Engine, though `MPSGraphDeviceType` publicly exposes only `.metal`,
so ANE use there is framework-managed rather than directed.

None of that changes the outcome for this model, because the obstacle is not
reaching the ANE. It is that only half the network can go there.

Attention is the ANE-shaped part: dense, static shapes, ~11.7 ms/token of GPU
work, and non-expert weights are only 1.31 GiB at 4-bit -- about 2.6 GiB at int8,
which fits. Stateful Core ML models can even hold the KV cache now.

The routed experts cannot follow it. At int8 they are ~34 GB against 24 GB of
RAM, and expert selection changes every token while a Core ML graph is compiled
statically -- expressing top-8-of-256 would mean materialising all 256.

So attention on ANE and MoE on GPU forces the two to alternate, 40 times per
token, 80 handoffs. Measured per-layer GPU round trips already cost 0.2-0.3 ms,
and a Core ML invocation is not cheaper. That is ~12 ms/token of handoff against
11.7 ms of work moved -- negative before anything else is counted.

And there is now a further reason for doubt. CPU load was measured to raise
GPU-busy by 44.9% through shared package power; the ANE sits on the same die and
the same budget. It is more efficient per operation for the ops it supports, so
the penalty may be smaller, but "the ANE is idle therefore free" is the same
assumption that proved false for the cores.

The honest summary: for a *dense* model the ANE would be a real third resource.
For a sparse MoE with per-token routing it cannot hold the half that dominates,
and the alternation that follows costs more than it saves. Not an API limit, and
not something a v4.0 rewrite changes.

Untested, and testable if wanted: build one attention block as an fp16 Core ML
ML Program, confirm with the Core ML Instrument that it lands on ANE, then run it
in a loop during decode and watch `busy_per_token` -- the same harness that
settled the CPU question. That needs `coremltools`, which is not installed here.

## Measured: ANE load costs twice what CPU load costs

The ANE question was previously closed by argument. Now measured. Built an fp16
Core ML ML Program (24 stacked 2048x2048 linears, batch 64) with coremltools,
confirmed it reaches the ANE -- `CPU_AND_NE` runs it in 3.31 ms against 6.24 ms
for `CPU_ONLY`, 1.88x, at 3.9 TFLOP/s -- then ran it in a loop during decode:

| load | tok/s | GPU busy/token |
| --- | ---: | ---: |
| none | 20.43 | 27.26 ms |
| ANE at 3.9 TFLOP/s | 11.15 | **51.62 ms** |

GPU-busy **+89.3%**, throughput **-45.4%**. Two samples per arm, tight
(27.34/27.19 idle, 49.69/53.56 loaded). Twice the damage the CPU load did.

The mechanism is visible in the load generator itself: 192 MiB of fp16 weights
per call at 3.31 ms is ~58 GB/s of memory traffic. The ANE was not merely drawing
power, it was consuming most of the bus.

Caveat: driving it through Core ML from Python means some CPU marshalling is
included, so this is ANE-plus-some-CPU rather than ANE alone. The effect is
nearly double the pure-CPU case, so the ANE contribution dominates.

## Why every one of these attempts failed the same way

The three resources were nominated as idle: cores, ANE, bus. The unifying result
is that **only one of them was ever the constraint, and it is the bus.**

NVMAI moves ~1.8 GB per token. This machine delivers ~78 GB/s in practice. That
sets an 23 ms/token floor -- and during the 27.6 ms the GPU actually works, it
already pulls ~65 GB/s, 83% of that. Compute units are idle; *bandwidth* is not.

Every failed idea in this document is the same mistake in different clothing:

- CPU expert co-execution: adds a second bus consumer. -22.6%.
- ANE offload: adds a faster bus consumer. -45.4%.
- Compression: would reduce bus traffic, but the payload is at 93% of its entropy
  limit, so there is nothing to remove.
- Reshaping: would improve bus efficiency, but it is already at 88% of achievable.
- Spinning: trades bus-idle CPU cycles for GPU clocks. -15%.

Adding compute to a bandwidth-bound pipeline on a shared memory system cannot
help, and adding *fast* compute hurts more than slow compute. That is why the
two-GPU analogy does not carry: two discrete cards bring their own memory and
their own bandwidth. Everything on an M3 shares one controller.

### What this means for a v4.0

The ceiling is set by bytes / bandwidth, and both terms are fixed -- bytes by the
user's quantisation choice, bandwidth by the silicon. 1.8 GB at 78 GB/s is 23 ms,
so ~43 tok/s is the hard maximum for 4-bit on this machine, against 21 measured.

Closing that gap is entirely a matter of keeping the GPU fed: 16.7 ms of the
47.3 ms token is idle. Eliminate all of it and the result is 27.6 ms, 36 tok/s,
1.71x. The remaining step to 43 tok/s requires the GPU's own kernels to move from
65 to 78 GB/s, +20% on kernel bandwidth efficiency, which is unmeasured.

So a v4.0 should be a single-minded attack on GPU idle and kernel bandwidth
efficiency, with no CPU or ANE participation at all. 1.71x is the credible target;
2x needs the kernel efficiency to be there as well.

## 100% ANE, no GPU: a different route to the same ceiling

Worth asking, because the objection to ANE was contention, and running *only* on
ANE has no contention. Measured with a 64-layer 2048x2048 fp16 stack (512 MiB of
weights, far past overhead-bound):

| batch | ANE | CPU |
| ---: | ---: | ---: |
| 1 | **64.3 GB/s** | **63.8 GB/s** |
| 8 | 64.4 | 48.7 |
| 64 | 62.7 (4010 GFLOP/s) | 30.1 |

At batch 1 the ANE and the CPU are indistinguishable, 64.3 against 63.8 GB/s.
That is the whole story: batch-1 decode reads every weight exactly once and reuses
nothing, so it is purely bandwidth-bound and the compute unit does not matter. The
ANE's 4 TFLOP/s only shows up at batch 64, where reuse exists. It never exceeds
~64 GB/s of weight streaming at any batch size.

NVMAI's GPU achieves ~65 GB/s during its busy window. **The ANE and the GPU hit
the same wall, because it is the same wall.**

So an all-ANE engine would land at 1.8 GB / 64 GB/s = 28 ms/token, about 36 tok/s.
That is a legitimate architecture for reaching it -- Core ML executes the whole
graph without the per-layer CPU round trips that produce NVMAI's 16.7 ms of idle,
so the idle would be absent by construction rather than engineered away. But it is
the *same* 36 tok/s that eliminating GPU idle reaches, from the opposite direction.

It would also cost: no weight streaming, so the whole model resident (viable only
if Core ML's 4-bit palettisation holds ~18 GB, and worse for the 6/8-bit
variants); the MoE expressed as a gather over all 256 stacked experts, whose ANE
efficiency is unknown; and NVMAI's engine, prompt cache and quantisation choice
replaced wholesale.

### The corrected ceiling

An earlier estimate here used 78 GB/s -- the pure CPU streaming-read peak -- to
put the ceiling at 43 tok/s. That was wrong: no path that also *computes* reaches
78. Every compute path measured lands at 62-65 GB/s, whether GPU, ANE or CPU.

So the real ceiling for 4-bit on this M3 is **1.8 GB / ~64 GB/s = 28 ms/token,
about 36 tok/s**, against 21 measured. Maximum available gain is **1.71x**, and it
is entirely the GPU idle. Three independent routes -- eliminate the idle, run
all-ANE, or run all-CPU -- converge on the same 36 tok/s, which is what a
bandwidth ceiling looks like from the inside.

**2x is not achievable on this hardware with this model at 4-bit.** Not by
redesign, not by using every unit, not by compression. The remaining 1.71x is
real and worth building; the last 0.3x does not exist.

## ANE at 4/6/8-bit: hardware decompression, same ceiling, and a 6-bit anomaly

Same 64-layer 2048x2048 stack, batch 1, weights palettised with Core ML:

| format | ms/call | stored weights | GB/s of stored bytes |
| --- | ---: | ---: | ---: |
| fp16 | 8.47 | 512 MiB | 63.4 |
| 8-bit | 4.43 | 256 MiB | 60.6 |
| 6-bit | 4.31 | 192 MiB | **46.8** |
| 4-bit | 2.22 | 128 MiB | 60.4 |

**The ANE decompresses low-bit weights in hardware.** Time tracks *stored* bytes,
not fp16-equivalent size: fp16 to 4-bit is 4x fewer bytes and 3.8x faster. That is
textbook bandwidth-bound scaling and it independently confirms the central thesis
of this document -- bytes are the constraint, and nothing else is.

**And the ceiling is the same.** 60-63 GB/s of stored bytes at every bit width,
against the GPU's ~65 GB/s. So an all-ANE engine at 4-bit lands at 1.8 GB /
60 GB/s = 30 ms/token, ~33 tok/s -- marginally *worse* than the 36 tok/s the GPU
route reaches. The ANE is not a faster path, it is an equivalent one with a small
penalty.

**6-bit is the outlier and it is actionable.** 46.8 GB/s against 60 for both 4-bit
and 8-bit, and only 0.12 ms faster than 8-bit despite storing 25% fewer bytes.
That is the signature of a non-power-of-two packing being padded or unpacked
inefficiently. Worth checking whether NVMAI's own 6-bit GPU path has the same
shape: the earlier 6-bit measurement (6.68 tok/s) was dominated by the model not
fitting RAM, so a packing inefficiency would have been invisible underneath it. If
present, 6-bit users are paying twice -- once for not fitting, once for the
packing.

## GPU vs ANE, measured head to head

Identical workload -- 64 chained [B x 2048] x [2048 x 2048] fp16 matmuls, 512 MiB
of weights -- through each unit's optimised framework path: MPSMatrixMultiplication
for the GPU, Core ML for the ANE.

| batch | GPU | ANE |
| ---: | --- | --- |
| 1 | 8.99 ms, 59.7 GB/s, 60 GFLOP/s | 8.34 ms, 64.3 GB/s, 64 GFLOP/s |
| 8 | 8.45 ms, 63.5 GB/s, 508 GFLOP/s | 8.34 ms, 64.4 GB/s, 515 GFLOP/s |
| 64 | 16.96 ms, 31.7 GB/s, 2026 GFLOP/s | 8.57 ms, 62.7 GB/s, 4010 GFLOP/s |

The answer depends entirely on arithmetic intensity, and the crossover is sharp.

**At batch 1 they are the same**, within 7%. Both sit at ~60-64 GB/s of weight
reads and neither exceeds 64 GFLOP/s, because a single-row matmul reuses no weight
and the memory system decides the outcome. This is the regime NVMAI decode lives
in, which is why swapping units changes nothing.

**At batch 64 the ANE is 2x faster** -- 4010 against 2026 GFLOP/s -- and notably
it holds 62.7 GB/s while the GPU's effective weight bandwidth collapses to
31.7 GB/s, meaning the GPU has become compute-bound where the ANE has not.

Caveat on the GPU column: this is MPSMatrixMultiplication with one encode per
layer, not a hand-tuned kernel, and MPS is known to be weak on small-M shapes. The
batch-64 GPU figure is therefore a floor, not the GPU's ceiling. The batch-1 and
batch-8 rows are safe, since both units are bandwidth-bound there and the result is
a tie either way.

Practical reading: the ANE is the better dense-matmul engine once there is reuse to
exploit, and it also decompresses 4-bit weights in hardware. The GPU is the only
one that can run arbitrary kernels, any dtype, and dynamic control flow -- which is
what a per-token top-8-of-256 MoE router requires. For NVMAI's decode path neither
is faster, because neither is the bottleneck.

# v4.0 research: GPU vs ANE per workload

Benchmarked rather than assumed. ANE measured through Core ML at NVMAI's exact
shapes with 4-bit palettised weights, so bytes-moved matches what the GPU reads.
Per-op cost is the *marginal* cost from a slope (N and 2N repetitions of the op in
one graph), which removes Core ML's ~1 ms fixed invocation overhead. GPU figures
are NVMAI's own `NVMAI_KERNEL_STATS` roles, divided by 40 layers.

## ANE efficiency collapses with tensor size

| op | ANE ms/op | weight (4-bit) | GB/s |
| --- | ---: | ---: | ---: |
| qkv_proj 2048->9216 | 0.1414 | 9.0 MiB | 66.7 |
| o_proj 4096->2048 | 0.0682 | 4.0 MiB | 61.5 |
| router 2048->256 | 0.0053 | 0.2 MiB | 49.3 |
| expert_gate 2048->512 | 0.0204 | 0.5 MiB | 25.7 |
| expert_down 512->2048 | 0.0453 | 0.5 MiB | **11.6** |
| lm_head 2048->248320 | 3.9677 | 242.5 MiB | 64.1 |

This is the single most useful result for v4.0. The ANE reaches 62-67 GB/s on
tensors of 4 MiB and up, and falls to 12-26 GB/s below 1 MiB. `expert_down` is the
worst at 11.6 GB/s -- a 5.7x efficiency gap against `qkv_proj` for the same class
of operation, and note it is worse than `expert_gate` despite identical byte count,
so the narrow 512-wide input hurts as much as the small size.

**NVMAI's MoE is built entirely from 0.5 MiB tensors.** That is precisely the
regime where the ANE is weakest.

## Where each unit wins

| workload | GPU (measured) | ANE (measured) | winner |
| --- | ---: | ---: | --- |
| routed MoE, 8 experts/layer | 0.244 ms | 0.689 ms | **GPU 2.8x** |
| shared expert FFN | 0.054 ms | 0.086 ms | **GPU 1.6x** |
| o_proj + router | 0.063 ms | 0.074 ms | GPU 1.2x |
| lm_head (once per token) | 4.386 ms | 3.968 ms | ANE 1.1x |

The GPU wins decisively on everything the MoE is made of, which is 44% of its busy
time. The ANE is marginally better on `lm_head`, the one genuinely large tensor in
the model.

**Caveat that has to be stated:** the GPU's `attn_norm_qkv` role (0.348 ms/layer)
is a composite -- RMSNorm, QKV projection, RoPE, the SDPA itself and the output
gate -- while the ANE figure above is the projection alone. They are not
comparable, so the QKV comparison is deliberately absent from the table. Settling
it needs either a decomposed GPU role or a Core ML graph covering the whole
attention block. Until then, no claim either way.

## What this means for v4.0

A hybrid split is what the per-op numbers suggest: ANE for `lm_head`, GPU for
everything else. Best case that is 0.42 ms/token of the 47.3 ms budget, under 1%.

Any larger split runs into two measured walls. Alternating ANE and GPU per layer
costs 40 handoffs per token at 0.2-0.3 ms each -- 8-12 ms, larger than anything it
could save. And concurrent execution is worse than serial: ANE load raises GPU-busy
89%, because both draw on one memory controller.

So v4.0 stays GPU-only. The ANE research resolves to a single 1% opportunity that
is not worth the integration risk, and its real value is negative knowledge --
nobody needs to revisit this.

## 6-bit is dropped in v4.0

Decided, and the measurements support it. Palettised at 6 bits the ANE reaches
46.8 GB/s against 60 for both 4-bit and 8-bit, and is only 0.12 ms faster than
8-bit despite storing 25% fewer bytes -- the signature of a non-power-of-two
packing being padded. On a 24 GiB machine 6-bit also does not fit, measuring
6.7 tok/s against 4-bit's 18.8. It was costing users twice.

## The width axis changes the answer: prefill is compute-bound

Everything above concerns decode. Prefill is a different machine.

NVMAI prefill throughput against chunk width, 3752-token prompt, 4-bit, 128 slots:

| chunk | prefill | tok/s | ms/token |
| ---: | ---: | ---: | ---: |
| 128 | 82.76 s | 45.3 | 22.06 |
| 512 | 64.88 s | 57.8 | 17.29 |
| 1024 | 57.24 s | 65.5 | 15.26 |
| 2048 | 53.75 s | 69.8 | 14.33 |
| 4096 (default) | 53.43 s | **70.2** | 14.24 |

Monotonic and saturating -- the shipped default of 4096 is already optimal, and
larger chunks win because the expert union grows sublinearly, so reads amortise
over more tokens.

The important number is the byte budget:

| | expert bytes/token | ms/token |
| --- | ---: | ---: |
| decode (width 1) | 540 MiB | 47.3 |
| prefill (width 4096) | **4.22 MiB** | 14.24 |

128x fewer bytes for only 3.3x less time. **Prefill is not bandwidth-bound; it is
compute-bound.** At ~2.75B active parameters that is ~5.5 GFLOP/token, so 14.24 ms
is roughly **400-500 GFLOP/s -- about 13% of the M3 GPU's peak.**

### ANE at prefill widths

Marginal per-op cost, 4-bit palettised, NVMAI's real shapes, with a nonlinearity
between repetitions so `sum(W_i x)` cannot be folded to `(sum W_i) x`:

| shape | width 1 | width 256 | width 1024 |
| --- | ---: | ---: | ---: |
| qkv 2048->9216 | 247 GFLOP/s | 14250 | **13129** |
| expert_gate 2048->512 | 78 | 15065 | **16640** |
| expert_down 512->2048 | 34 | 7845 | **8223** |

The ANE goes from useless at width 1 (34-247 GFLOP/s) to **8-17 TFLOP/s** at
prefill widths -- a 100x swing driven entirely by weight reuse. Against NVMAI's
~400-500 GFLOP/s GPU prefill, the dense-matmul portion is 20-30x faster on ANE.

Treat the absolute figures with some caution: 13-16 TFLOP/s brushes the M3 ANE's
rated ~18 TOPS, which is only consistent if 4-bit palettised weights take an
int8 datapath. The *ratio* is robust regardless -- width 1 to width 1024 is a 100x
change measured on the same harness.

## Switching costs, measured

This is what decides whether any hybrid is viable:

| transition | cost | per token at decode | per token at prefill |
| --- | ---: | ---: | ---: |
| GPU per-layer round trip | 0.2-0.3 ms | 8-12 ms (40x) | negligible |
| Core ML invocation | ~0.16 ms | 6.4 ms (40x) | 0.00004 ms (1 per 4096) |

Same overhead, opposite verdict. A per-layer handoff during decode costs more than
the work it moves; one handoff per prefill chunk is amortised over 4096 tokens and
is free. **That, not the compute, is why decode must stay GPU-only and prefill is
worth revisiting.**

## Three-way summary

| workload | width | bound by | best unit |
| --- | ---: | --- | --- |
| decode, routed MoE | 1 | bandwidth | GPU (2.8x over ANE, 2.1x over CPU) |
| decode, everything else | 1 | bandwidth | GPU; all units within ~7% |
| lm_head | 1 | bandwidth | tie (ANE 1.1x) |
| **prefill** | **4096** | **compute** | **ANE, 20-30x on dense matmul** |

CPU is excluded throughout: it never wins a workload, and loading it costs the GPU
45%.

## The recommendation, and the check that must come first

Prefill is the one real opportunity, and for a long prompt it is large -- 53 s for
3752 tokens today.

But **do not start with Core ML.** The GPU is running prefill at 13% of its own
peak, and the first question is why. If NVMAI's prefill kernels are simply
inefficient at width 4096, fixing them is a contained kernel change that keeps the
engine, the quantisation choice and the KV cache intact. Adopting Core ML means a
second model artifact, all 256 experts resident, and handing the ANE-produced KV
cache to the GPU decode path across two frameworks -- by far the harder route, and
pointless if the GPU has 3-5x sitting in its own kernels.

So: profile prefill's GPU roles at width 4096 first. Only if the GPU is genuinely
near its ceiling does the Core ML prefill path become the right answer.
