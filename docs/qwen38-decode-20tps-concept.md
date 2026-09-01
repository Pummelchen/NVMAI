# Concept: what 20 tok/s would take for Qwen3.8-Flash-Next at 4-bit

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
