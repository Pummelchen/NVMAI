# Concept: what 20 tok/s would take for Qwen3.8-Flash-Next at 4-bit

---

# MEASURED, 2026-09-02: this document's inputs were wrong

Everything below this line predates the profile it asks for in "Step 0". That
profile has now been run on this model (`nvmai_overlap_measure`, 256-token warm
request, 4-bit, 24 GiB M3) and it falsifies three of the inputs above. The
analysis method was sound; the numbers fed into it were not.

## The token, measured

| component | ms/token | share |
| --- | ---: | ---: |
| GPU busy | 79 | 48.6% |
| exposed expert I/O | 42 | 26% |
| control plane / gaps | 41 | 25% |
| **total** | **162.5** | |

## What was wrong

**The hit rate.** The ceiling table used 78%, labelled "128 slots, measured
saturation". 78% is what **64** slots gives. Measured on this model:

| slots | hit rate | tok/s |
| ---: | ---: | ---: |
| 64 | 78.1% | 5.58 |
| 96 (shipped) | 85.4% | 5.71 |
| 128 | 89.8% | 1.80 (swaps: ~17 GB of cache on a 24 GB machine) |

So the curve does **not** saturate at 78%, and route A ("more resident memory")
is blocked by this machine's RAM rather than by a saturating curve. On a 64 GiB
machine it is ordinary work.

**The binding constraint.** At the measured 87.4% warm hit rate the I/O floor is
~48 ms, a ~19-21 tok/s ceiling -- not 12.3. The real floor is **GPU compute at
79 ms/token = 12.6 tok/s**. The document's headline number is roughly right by
coincidence and wrong in its reason.

**The omission.** `attn_norm_qkv` (29.8 ms/token) and `attn_tail_router` (15.8)
total **45.6 ms/token across 96 dispatches** -- larger than every MoE kernel
combined, and absent from this document entirely. At that weight volume they run
near unified-memory bandwidth, so there is no dispatch-fusion win hiding there.

## What was tried, and closed

A 15-arm sweep (`nvmai_knob_sweep.py`, baseline drift 2.9%) plus two defect
fixes. **Nothing improved throughput correctly.**

| lever | result |
| --- | --- |
| event I/O sync | **-15.5%** after fixing it; the +79% it first showed was a kernel returning early |
| immediate submission | -9.8% -- speculative reads steal service from demand reads |
| gpu-residency | -7.1% after fixing two defects; does strictly more work for the same answer |
| prefetch depth 2 | -6.5% (reproduces the recorded -6.0%) |
| cache policy lru / aging-lfu | -9.0% / -5.7% |
| slots 128 | -68.5% |
| layout pool, keep-wired, no-parallel-io | +1.1% to +4.6%, all inside drift |

Item 1 of the plan above -- "overlap I/O with compute" -- is now measured and
**backwards**. Three independent arms raised `io_hidden_pct` and every one of
them lowered throughput. Overlap is a diagnostic, not an objective: the device
is saturated, so an overlapped read is a read taken from someone else.

Item 4, "cut the control plane", is not reachable by removing host waits: that
is exactly what event sync does, and it loses 15.5%.

### Attention is dispatch-shaped, and that is not an inefficiency

`attn_norm_qkv` moves 745.8 MiB/token in 29.8 ms -- an effective **26.2 GB/s**
against ~100 GB/s peak -- while `head_logits` moves 644 MiB in a single
dispatch at **67.8 GB/s** on the same memory. That 2.6x gap looks like a
kernel problem and is not. Three attempts, all measured, all closed:

| attempt | premise | result |
| --- | --- | --- |
| `gdn_in_proj_gemv_simd_xsh8/_r16/_xsh16` | every row re-reads x, ~80 MB/layer | +0.3% / +1.6% / -1.0%, drift 3.2% |
| attn/tail command-buffer merge | ~0.15 ms/CB x 36 linear layers | **-1.8%**, reverted |
| (implied) fuse the 48 layers | one large dispatch is 2.6x more efficient | impossible: layer L+1 depends on L |

The x premise was simply wrong: x is **5 KiB**, so it sits in L1 and those
"device memory" re-reads were already cache hits. The traffic was nominal.

The command-buffer premise was right about the cost and wrong about where it
falls. Committing `attnCB` before `tailCB` is encoded lets the GPU execute
while the CPU is still encoding, so the buffer's overhead is already hidden
behind that overlap; removing the buffer removed the overlap too, and the two
cancelled with a slight loss.

**The conclusion is structural.** 48 dependent small GEMVs cannot be made into
one large one without changing the model's dependency chain, so 26 GB/s is what
this shape costs. Do not re-open it as a kernel-tuning problem.

## What is actually left

Only two things move this model on this hardware:

1. **A machine with more RAM.** 128 slots reach 89.8% and would fit.
2. **GPU-side fetch planning**, so the CPU never blocks on the router at all.
   Not a knob and not a fix -- an architecture change. `gpu-residency` is now
   correct and is the prerequisite for it, at the cost of 7%.

Falsifiers for *this* section: a profile showing GPU busy well under 79 ms/token,
or any arm that raises `io_hidden_pct` and throughput together.

---


Published decode is **6.82 tok/s** (147 ms/token) on the 24 GiB M3. The target
is 20 tok/s, 50 ms/token — a 2.9x cut.

This document leads with the constraint rather than a list of optimizations,
because the constraint decides which optimizations can matter. The short
version: **engineering can plausibly reach ~12 tok/s and cannot reach 20.**
Crossing 12 requires changing what is read, not how it is read.

## The governing constraint

Decode reads routed experts from SSD. The volume is fixed by the architecture:

| | |
| --- | ---: |
| experts per token | 10 of 512, in each of 48 layers = **480 reads** |
| per-expert record | 2.77 MB (gate + up + down, 4-bit, plus scales/biases) |
| all-miss traffic | **1.33 GB per token** |

The expert cache removes most of that. What is left is what the SSD must
deliver inside one token, and the measured effective read bandwidth is
**3.6 GB/s**:

| hit rate | bytes/token | I/O alone | ceiling |
| ---: | ---: | ---: | ---: |
| 65% (64 slots, measured) | 465 MB | 129 ms | 7.7 tok/s |
| 73% (~96 slots, shipped) | 359 MB | 100 ms | 10.0 tok/s |
| **78% (128 slots, measured saturation)** | **292 MB** | **81 ms** | **12.3 tok/s** |
| 86.5% | 179 MB | 50 ms | 20.1 tok/s |

The port measured the cache saturating at **78% around 128 slots** — 512
experts at top-10 spread far wider than Qwen 3.6's 256 at top-8, which reaches
93.6%. So 12.3 tok/s is the ceiling for perfect engineering at the shipped
geometry, and 20 tok/s sits ~1.6x beyond it.

**No amount of overlap, prefetch depth, dispatch fusion or kernel work crosses
that line.** Those hide latency; they do not reduce bytes, and the floor is
bytes ÷ bandwidth.

## What engineering can still take: 6.82 -> ~12 tok/s

The gap between 6.82 and the 10–12.3 floor is real and worth having. It is
~75% more throughput, and every part of it is ordinary work.

**1. Overlap I/O with compute.** Decode is I/O-compute *serialized* today
(measured). Serialized, the token is `io + compute`; overlapped it is
`max(io, compute)`. This is the single largest structural item and it is
worth roughly the whole compute term.

**2. Prefetch the misses instead of awaiting them.** The 78% ceiling is not
the problem — *demand-loading* the other 22% is. Prefetch depth 1 measured
+12.2%; depth 4 measured **-9.8%**, because the prefetch ring and the resident
cache compete for one fixed budget and a deeper ring starves the cache. This
is precisely the adaptive resident/ring split (tracker Item 11) borrowed from
the KV-streaming work: move the boundary under pressure instead of fixing it
at startup, with shrink hysteresis against thrash. Item 12 (churn-stable
residency priority) protects the same budget from oscillation.

**3. Saturate the device.** 3.6 GB/s effective is the number every row above
depends on. Item 7 concluded the SSD is already saturated; that conclusion
predates this model's access pattern and deserves one re-measurement at
realistic queue depth, because every 1 GB/s recovered moves the ceiling by
~3 tok/s.

**4. Cut the control plane.** Qwen 3.6 measured the GPU idle over half of
every token with 18.9 ms of inter-command-buffer gaps. Qwen 3.8 adds
hyper-connections in all 48 layers and has never been profiled this way. Only
matters once (1) lands and the token is `max(io, compute)`.

**Tuning all four is what Item 10 is for.** The optimum is machine-dependent
— our own numbers show depth 1 winning and depth 4 losing — and our benchmarks
carry +/-15% run-to-run spread. An in-process A/B that interleaves arms,
discards warmup and latches on a minimum gain is the only honest way to pick
these, and its 0.5% threshold must be re-derived against our noise floor.

## What 20 tok/s would actually require

Three routes cross the floor. All change the physics rather than the code.

**A. More resident memory.** At 3.6 GB/s the requirement is a **86.5%** hit
rate. 128 slots costs 15.8 GiB and measured 78%; peak footprint on this
machine is already 18 GiB of 24 GiB. A 64 GiB machine could hold several times
the cache. This is the only route that keeps the model's output identical.
Unknown: whether 86.5% is reachable at *any* cache size, since the curve was
described as saturating. Re-measuring the hit-rate curve past 128 slots is
cheap and decides this route.

**B. Faster storage.** At the measured 78% hit rate, 20 tok/s needs
**7.2 GB/s** — roughly twice the internal SSD. Thunderbolt/USB4 external NVMe
lands near 3–5 GB/s, so this route does not reach the target alone; it moves
the ceiling to ~17 tok/s at best and only combines usefully with A.

**C. Read fewer experts.** Dropping the two lowest-weighted of the ten cuts
traffic 20%; six-of-ten cuts it 40% and puts 20 tok/s inside the floor at the
current hit rate and bandwidth. The router's normalized weights make this
measurable rather than arbitrary — the bottom experts carry ~9% each. **This
changes the model's output** and needs a quality measurement before it is
anything but a benchmark trick. It is the only route reachable on this machine
today.

## Step 0, before any of it

**The decode split for Qwen3.8-Flash-Next has never been measured.** Every
number above except the cache curve and the 3.6 GB/s is derived from
architecture and from Qwen 3.6's profile. This project's history is emphatic
about what that is worth: the prefetch saving was predicted at 11.5 ms and
measured 0.59 ms, and dispatch-count arithmetic predicted 3.8 ms where the
real figure was 0.59. Both were wrong by more than an order of magnitude.

So the first work item is an instrumented token for *this* model: exposed
expert I/O, GPU busy, control-plane gaps, cache hit rate at the shipped 96
slots, and the hit-rate curve past 128. `benchmark/nvmai_profile.py` and
`nvmai_slots_ab.py` already exist for this.

If that profile shows exposed I/O far below the ~100 ms this arithmetic
predicts, the model is not I/O-bound in the way assumed here and this whole
document is wrong in a useful way.

## What would falsify this

- Exposed expert I/O measuring well under 100 ms/token at 96 slots.
- The hit-rate curve continuing to climb past 128 slots rather than saturating
  at 78% — which would make route A ordinary work rather than new hardware.
- Effective read bandwidth exceeding 3.6 GB/s at higher queue depth.

Any of the three changes the conclusion. None of them is expensive to check,
and all three should be checked before anyone writes a kernel.
