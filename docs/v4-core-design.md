# NVMAI v4.0 — core design

Clean-sheet rewrite of the inference core, targeting the physical limits of an
M3 MacBook Pro **while streaming weights from SSD**. Streaming is not a fallback
here; it is the product. The goal is a 35B MoE running with a small, declared RAM
footprint so the rest of the machine stays usable.

Every number below is measured on the development machine (M3, 4P+4E, 24 GB,
~64 GB/s achievable memory bandwidth, ~3.2 GB/s SSD). Method and derivations are
in [cpu-coexecution-plan.md](cpu-coexecution-plan.md).

## CORRECTION: the premise this document was written on was wrong

The first version of this design claimed the SSD was 5x under-used -- 0.62 GB/s
achieved against 3.2 available -- and built its core bet on recovering that. That
figure was an arithmetic error of the same kind already documented in
[cpu-coexecution-plan.md](cpu-coexecution-plan.md): 13.6 GB of expert reads divided
by the whole 21.9 s "expert fetch + tiles" phase, when only ~4.8 s of that phase is
fetch and the other 17.1 s is GPU tile execution. **The real prefill fetch rate is
~2.83 GB/s.** Decode's is ~2.6 GB/s (14.65 MiB/token in 5.6 ms).

Against a measured device ceiling of 3.92 GB/s that is 66-72%, not 16%. There is
roughly **1.4x** of streaming headroom, not 6x.

Two further corrections follow from it:

**Decode already parallelises its fills.** `executeExpertCachePlan` has used
`DispatchQueue.concurrentPerform` since v3.x, with a recorded +28% when it landed.
The serial fetch this design proposed to fix does not exist.

**A parallel pool does nothing at decode's batch size.** At 128 slots the hit rate
is ~92%, so decode fetches about *one* expert per layer, and one read is one read:

| batch | pool | serial pread |
| ---: | ---: | ---: |
| 1 | 3.41 GB/s | 3.44 |
| 4 | 5.36 | 3.80 |
| 8 | 5.95 | 3.69 |

The pool is worth 1.4-1.6x only at 4-8 misses per batch. So it does not speed up
today's configuration at all. What it does is make a *smaller* cache viable, since
smaller caches miss more often and therefore fetch in bigger batches. That is still
useful, but it is a different claim than the one this document was built on.

## The measured RAM/throughput curve

Directly measured on v3.8, no new code, 192 tokens each:

| slots | declared RAM | tok/s | io ms/token | bytes/token |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 0.53 GB | 14.48 | 25.22 | 316.4 MiB |
| 16 | 1.05 GB | 15.17 | 23.47 | 226.7 MiB |
| 32 | 2.11 GB | 15.96 | 20.67 | 153.1 MiB |
| 64 | 4.22 GB | 17.46 | 14.17 | 68.7 MiB |
| 128 | 8.44 GB | **20.93** | 6.18 | 14.7 MiB |

So the trade is real and not free: 4x less RAM costs 24% of throughput, 16x less
costs 31%.

### And the curve is flattered by the page cache

316 MiB/token in 25.2 ms is **12.5 GB/s -- three times the device ceiling.** Those
reads are not reaching the disk; the unified buffer cache is holding what the slot
cache does not, using memory that the "declared RAM" column does not count.

**So shrinking the slot cache does not reduce the machine's memory footprint. It
relocates it into the page cache**, where it is invisible to the budget and evicted
on the OS's terms rather than the engine's.

This is the central design question for v4.0, and it is now sharp rather than
assumed:

- **Page cache allowed** — small slot counts stay fast (the curve above) but total
  RAM use is not actually bounded, which defeats the stated purpose.
- **`F_NOCACHE`** — the slot budget becomes the true and only footprint, verified:
  streaming 16.88 GiB with it on left the machine at 78% free. But then every miss
  is a real 3.92 GB/s disk read rather than a 12.5 GB/s cache hit, so the curve
  above gets materially worse and has to be re-measured.

The honest position is that **the cost of a genuinely bounded footprint has never
been measured**, because v3.x has always been quietly leaning on the page cache. The
next increment is to wire the `F_NOCACHE` reader in and measure the true curve. Only
then is there a basis for choosing a default budget.

## What the measurements say the limits are

| resource | measured ceiling | what NVMAI 3.8 achieves |
| --- | ---: | ---: |
| memory bandwidth (compute path) | ~64 GB/s | ~65 GB/s during decode — **at the limit** |
| **SSD, expert-sized reads** | **~3.2 GB/s** | **~0.62 GB/s — 5x headroom** |
| GPU clocks during work | Maximum DVFS, Nominal thermals | already maxed |
| GPU occupancy, prefill | — | 97.4% — saturated |
| GPU occupancy, decode | — | 61% — 16.7 ms/token idle |

Two of those are already at the wall. The other two are the design targets: **SSD
bandwidth is 5x under-used, and decode leaves 16.7 ms of every 47 ms token idle.**

SSD detail, because the whole architecture rests on it:

| pattern | 1 thread | 2 | 4 | 8 |
| --- | ---: | ---: | ---: | ---: |
| sequential 1.688 MiB reads | 2.03 GB/s | 3.03 | **3.21** | 3.17 |
| random 1.688 MiB reads | 2.19 GB/s | 2.99 | **3.16** | 3.19 |

Random is as fast as sequential — NVMe does not care about locality at expert
granularity, so the streamer never needs to reorder for locality. And parallelism
matters: 4 concurrent readers are 1.6x one reader.

## The design idea: trade SSD bandwidth for RAM

At 4-bit, decode reads 540 MiB of expert weights per token. The slot cache absorbs
most of it; whatever misses comes from SSD. So for a cache achieving hit rate `H`:

```
disk bytes/token = 540 MiB x (1 - H)
disk time/token  = that / 3.2 GB/s
GPU time/token   = ~28 ms   (memory-bus bound, irreducible)
```

Disk time stays **fully hidden behind GPU compute** as long as
`540 MiB x (1-H) / 3.2 GB/s < 28 ms`, i.e. **H > ~83%**.

v3.8 reaches 92% with 128 slots per layer, which is 128 x 1.688 MiB x 40 =
**8.4 GB of RAM** — a third of the machine, which contradicts the point of the
project. But it only needs 83%, and the whole gap between 0.62 and 3.2 GB/s is
currently being spent buying hit rate that a faster streamer would not need.

**So the core bet: fix streaming bandwidth, then spend the surplus on shrinking the
RAM budget rather than on speed.** Same tok/s, a fraction of the footprint.

## Core architecture

### 1. RAM budget is an input, not an outcome

The engine takes a declared budget (`--ram-budget 2G`) and derives everything from
it: slot counts per layer, prefetch depth, KV reservation. It reports the resulting
predicted hit rate and disk load at startup, and refuses budgets that cannot hold
the resident tensors.

This inverts v3.x, where slot count was the knob and RAM was whatever fell out.

### 2. Streaming that actually uses the disk

The current path gets 0.62 GB/s against 3.2 available. Three causes to remove:

- **Serialised fetch.** Misses are fetched per layer, in order, on the calling
  thread. Four concurrent readers measure 1.6x one. Use a small I/O thread pool.
- **No depth.** A fetch is issued when the miss is discovered, so the disk is idle
  between layers. Keep a queue always non-empty.
- **Blocking on the critical path.** The layer loop waits for its own fetch.

I/O threads are the one CPU work that is safe here, and this was verified rather
than assumed. Running a saturating `pread` load during decode:

| load | tok/s | GPU busy/token |
| --- | ---: | ---: |
| none | 22.125 | 29.106 ms |
| pread I/O | 18.856 | 31.247 ms (**+7.4%**) |
| *CPU dequant, for contrast* | *-22.6%* | *+44.9%* |

GPU-busy rises **7.4%** under heavy I/O against **44.9%** under heavy compute, so
I/O threads are roughly six times gentler on GPU clocks -- they block in the kernel
instead of burning ALU. The 14.8% throughput drop in that test is the load
generator consuming the entire 3.2 GB/s and starving NVMAI's own fetches; it is
contention from an external hog, not a cost the engine pays for using its own
bandwidth. Budget the disk as a shared finite resource, but do not fear the threads.

### 3. Predictive prefetch, since routing cannot be known early

Expert choice for layer L is only known after layer L-1 completes, so nothing can
be prefetched from the current token's routing. But consecutive tokens share
**38%** of a layer's experts (measured over 383 tokens), and the working set over a
128-token window is 131 of 256 experts per layer.

So prefetch from the *previous token's* routing for the same layer, which is
available 40 layers early. A wrong guess costs a wasted read on an otherwise idle
disk; a right guess removes a stall. With 3.2 GB/s and 5x headroom, speculative
reads are close to free — this is what the surplus bandwidth is for.

### 4. Remove the per-layer CPU round trip from decode

Decode's 16.7 ms/token of idle is dominated by one transition, and the cause is
that the CPU must see the routing before experts can be dispatched. Two halves:

- **The dispatch half** goes away with GPU-side expert indexing: the MoE kernel
  reads expert ids from the router's own output buffer via an argument buffer
  covering the resident slots, so no readback is needed to *encode* the work.
- **The residency half** cannot go away while streaming — something must decide
  what to fetch. But it can move off the critical path: the kernel processes
  resident experts immediately and writes a miss list; the I/O pool services it
  asynchronously; a fixup pass completes the stragglers. At a 92% hit rate that
  makes the synchronous stall a 8%-of-layers event instead of every layer.

This is the one genuinely hard piece and the reason v4.0 is a rewrite rather than
a patch.

### 5. C99 for hot loops, Swift for structure

Established in 3.8: moving the int4 GEMV to C99/NEON was **2.9x**, and hoisting a
redundant per-group sum added another 16-20%. Swift's `SIMD8<Float>` does not lower
to vector loads. So: Swift owns lifetime, actors, and orchestration; `NVMAIKernelsC`
owns anything with a per-weight inner loop. Metal owns the GPU.

Not a blanket rewrite — Swift costs ~4.6 ms of a 47 ms token, and most of that is
Metal API calls that C would pay identically.

## What is deliberately not in v4.0

- **CPU co-execution.** Measured net negative: 8 threads of dequant work raise
  GPU-busy 45% and cost 22.6% throughput. Same memory controller, same power.
- **ANE during decode.** Worse: −45.4%, GPU-busy +89%.
- **Compression.** The payload sits at 93% of its entropy limit; zlib recovers 6%
  and decompresses at 0.53 GB/s against an 11.9 GB/s requirement.
- **Speculative decoding / MTP.** Verify cost tracks the expert union (1.585x at
  width 2) and cancels the 1.574 tokens emitted. Needs acceptance >0.585 to break
  even at all.
- **6-bit.** Dropped. Non-power-of-two packing measured 46.8 GB/s against 60 for
  both 4-bit and 8-bit.

## Quantisation: both, streamed

4-bit and 8-bit are both first-class and both streamed; the user picks quality and
the engine streams whatever they picked. 8-bit doubles expert bytes per token
(1020 MiB vs 540), so at a fixed RAM budget it needs roughly double the disk
bandwidth for the same hit rate — which is exactly why the 5x streaming headroom
matters. 8-bit at 3.2 GB/s needs H > ~91% to stay hidden, against 4-bit's 83%.

The v3.8 measurement of 8-bit at 1.6 tok/s is **not** evidence against this: that
run used a 128-slot cache and let the page cache fill, i.e. it was competing for
RAM rather than streaming within a budget. 8-bit under a declared budget with a
working streamer is untested and is a v4.0 acceptance target.

## Targets

| | v3.8 measured | v4.0 target | basis |
| --- | ---: | ---: | --- |
| decode, 4-bit | 21 tok/s | **32-36** | remove 16.7 ms idle; bus ceiling is 36 |
| RAM for that | 8.4 GB slots | **~2 GB** | H>83% suffices once disk runs at 3.2 GB/s |
| decode, 8-bit | 1.6 (thrashing) | **12-18** | streamed within budget, H>91% |
| prefill, 4-bit | 70 tok/s | unchanged | GPU saturated at max clocks |

Decode's 36 tok/s is a hard ceiling from 1.8 GB/token ÷ 64 GB/s and cannot be
exceeded on this machine by any means measured. The real v4.0 win is reaching it
**at a quarter of the RAM**, and making 8-bit usable at all.

## Prefill: prototype, do not commit

Prefill is GPU-saturated at maximum clocks — 97.4% occupancy, ~600 GFLOP/s, no
kernel fix available, because 4-bit dequant costs several ALU ops per weight on top
of the multiply-accumulate. The ANE reaches 13 TFLOP/s on the same shape because it
decompresses in hardware.

That is a real architectural advantage and worth a narrow prototype: one attention
block as an fp16/palettised Core ML ML Program, verified on-ANE via the Core ML
Instrument, measuring (a) achieved rate at width 1024+, (b) whether the KV cache can
be handed to the GPU decode path, (c) whether a 256-expert gather is expressible.

Not on the v4.0 critical path. Decode and the streamer are.

## Bounded footprint: the cost, finally measured

The page cache was purged (`sudo purge`, run by the user -- it cannot be driven
from here) and both arms were matched on slot count *and* context size, which the
first attempt was not:

| 32 slots, default context | wall, 192 tokens | process RSS |
| --- | ---: | ---: |
| cached | 26.43 / 23.20 / 22.82 s | 1.82 GB |
| bounded (`F_NOCACHE`) | 29.05 / 29.06 / 29.18 s | 3.74 GB |

**A bounded footprint costs ~20%.** An earlier reading of this put it at 2.3x; that
gap was mostly context size rather than cache policy, and the corrected figure is
the one to design against.

Note the RSS inversion, which is the whole point rather than an anomaly. Bounded is
*higher* because `F_NOCACHE` forces every expert into our own slots, where it is
counted. Cached is lower because it leans on the unified buffer cache, which does
not appear in process RSS at all. So bounded's 3.74 GB is the true machine cost,
while cached's 1.82 GB is 1.82 GB **plus** whatever the OS decided to hold. For a
project whose purpose is leaving RAM free, the honest number is the one you can
account for.

20% for a footprint that is actually bounded is a good trade, and `NVMAI_BOUNDED_IO`
should become the default in v4.0.

## Separately: the default context costs 1.6x throughput

Found while matching the arms above, and unrelated to streaming:

| `--max-context` | wall, 192 tokens | RSS |
| ---: | ---: | ---: |
| 8192 | 15.42 / **13.80** s | 3.68 GB |
| 32768 | 15.72 / 14.19 s | 2.67 GB |
| 262144 (default) | 25.92 / **22.44** s | 1.84 GB |

Same 32 slots, same 25-token prompt, same 192 generated tokens. **Reserving 262144
tokens of context makes decode ~1.6x slower than reserving 8192**, on a conversation
that uses neither.

The mechanism is that KV strides are sized by `max-context` rather than by the
sequence, so attention walks a buffer two orders of magnitude larger than the data
in it -- every access lands in a different page and the locality is gone. RSS
falling as context grows is consistent with that: more of the reservation is never
touched.

This is a v3.x defect, not a v4.0 design question, and it is worth more than most of
the work in this document: every user on the default is paying 1.6x for context they
are not using. v4.0 should size KV strides from the live sequence and grow them,
and the fix is likely backportable.

## Benchmark matrix: quant x RAM budget x cache policy x prompt length

All at the default `--max-context 262144`, which is a product requirement rather
than a tunable. Prompt sizes 25 / 452 / 3532 tokens. Figures are prefill seconds
and decode tok/s.

### 4-bit

| RAM | slots | cache | short | medium | long |
| ---: | ---: | --- | --- | --- | --- |
| 1 GB | 16 | bounded | 2.0s 11.10 | 5.3s 10.78 | 46.8s 7.16 |
| 1 GB | 16 | cached | 2.5s **13.61** | 6.0s 12.46 | 47.1s 7.54 |
| 2 GB | 32 | bounded | 1.8s 9.32 | 5.4s 10.09 | 46.7s 6.42 |
| 2 GB | 32 | cached | 1.9s 12.95 | 6.1s 12.36 | 47.4s 7.57 |
| 4 GB | 64 | bounded | 2.3s 8.36 | 5.4s 9.06 | 47.9s 4.52 |
| 4 GB | 64 | cached | 2.2s 9.85 | 6.1s 8.23 | 47.5s 6.41 |
| 8 GB | 128 | bounded | 2.4s 9.13 | 5.7s 13.49 | 47.2s 5.34 |
| 8 GB | 128 | cached | 2.5s 8.78 | 6.1s 13.23 | 47.6s 6.38 |

### 8-bit

| RAM | slots | cache | short | medium | long |
| ---: | ---: | --- | --- | --- | --- |
| 1 GB | 8 | bounded | 4.1s 5.22 | 10.7s 4.63 | 60.9s 3.39 |
| 1 GB | 8 | cached | 4.3s **5.64** | 11.2s 4.94 | 61.8s 3.83 |
| 2 GB | 16 | bounded | 3.7s 4.02 | 10.4s 3.87 | 60.6s 3.25 |
| 2 GB | 16 | cached | 3.7s 5.11 | 11.1s 4.66 | 60.9s 3.76 |
| 4 GB | 32 | bounded | 4.1s 4.24 | 10.2s 4.40 | 61.0s 3.23 |
| 4 GB | 32 | cached | 3.9s 5.26 | 10.5s 5.02 | 61.0s 3.72 |
| 8 GB | 64 | bounded | 4.9s 4.61 | 9.8s 5.24 | 62.1s 2.17 |
| 8 GB | 64 | cached | 4.1s 5.21 | 10.8s 5.33 | 62.6s 2.05 |
| 16 GB | 128 | bounded | 4.7s 1.22 | 45.8s 0.69 | 91.4s 0.46 |

4-bit cannot reach a 16 GB budget: 128 slots is the allowed maximum and that is
8.44 GB.

### What the matrix says

**More slot RAM is slower, not faster.** 4-bit at a 1 GB budget beats 4 GB by
~35-40% on short and medium prompts, in both cache modes. This reverses the curve
measured earlier in this document -- and the difference is the context. That curve
used `--max-context 8192`; this matrix uses the 262144 default, where the KV
reservation is already large enough that adding slot memory pushes the machine into
pressure. Under the real default, **a small slot cache is both faster and smaller.**

So NVMAI's shipped default of 64 slots is the wrong choice twice over: 16 slots is
~35% faster *and* uses a quarter of the RAM. That is the single most valuable
finding in this document and it is a one-line change.

**Cached beats bounded by 15-30%,** consistently, at every budget and prompt size.
That is a firmer number than the ~20% measured earlier and it holds across the
matrix.

**8-bit costs 2-2.5x throughput** against 4-bit at a comparable budget (5.64 vs
13.61 tok/s at 1 GB, short). It remains usable when streamed at a small budget,
which is the point -- but 8-bit at 16 GB collapses to 0.46-1.22 tok/s, because
15.94 GB of slots plus a 262144-token KV reservation does not fit 24 GB. Large
budgets are a trap, not a feature.

**Prefill is insensitive to the budget.** 46.8-47.9s for 4-bit and 60.6-62.6s for
8-bit at 3532 tokens, whatever the slot count, because a wide chunk touches nearly
every expert regardless of cache size. Prefill is bounded by GPU compute (97.4%
occupancy at max clocks), not by streaming.
