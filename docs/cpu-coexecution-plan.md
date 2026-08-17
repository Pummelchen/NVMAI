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
