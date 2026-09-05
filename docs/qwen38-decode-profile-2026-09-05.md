# Qwen3.8-Flash-Next 4-bit decode: measured profile and optimization plan

Date: 2026-09-05. Machine: base M3 MacBook Pro, 8 GPU cores, 24 GB, ~3.6 GB/s
SSD. Checkpoint: `models/qwen3.8-flash-next_125B_A6B_4Bit` (48 layers, 512
experts, top-10, expert stride 2,768,896 B). Shipped profile row: 12 GiB
expert budget = 96 slots per layer, prefetch depth 2 on the utility I/O
tier, chunk 4096, sampling 1.0 / 20 / 0.95. Every number below is from this
tree at tag `golden-code-2026-09-05` plus the diagnostics-only counters
added for this profile (`pre_*`, `loop_*`, `embed_ms`, `gather_ms`).

## 1. The decode pipeline as it runs today

Per token, `RealForwardRunner.produceToken`:

1. Preamble: ANE release (no-op), `setExpertCachePinned(true)`, KV reserve,
   QSA exactness gate.
2. Embed lookup (one command buffer, synchronous) and hyper-connection
   broadcast (one more).
3. N-gram gather: 16 `pread`s of 320 B from the 102 GB table, serial, cache
   bypassed, before layer 0.
4. Layers 0..47, each: attention command buffer (GDN or gated QSA attention
   plus PLE on its layers) and a tail buffer that ends in the router and the
   next-layer probe router; the CPU waits on the tail, reads the routed
   expert ids back, plans the cache (adopting any ready prefetch buffer by
   `memcpy` into its slot), pins the slots, issues the miss reads on the
   4-thread pread pool, commits a phase-1 command buffer for the resident
   experts, awaits the reads, then commits phase-1 for the misses plus the
   phase-2 reduce and the residual tail. The shared expert runs on its own
   buffer concurrently. After the routed encode it begins the next layer's
   prefetch reads from the probe's prediction, resident experts filtered
   out, first M in rank order.
5. Final norm, the 8-bit head GEMV (logits), then the sampler in the
   completion loop, then the server's stream callback.

## 2. Baseline

Three 512-token generations after a 200-word prompt, `decode_stats.py`
(kernel + runner stats), shipped defaults, one at a time:

| run | tok/s | body ms/token | GPU wait | expert I/O (hidden) | head | hit rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4.48 | 160 | 79 | 66 (19%) | 9.6 | 81.3% |
| 2 | 5.23 | 170 | 85 | 67 (18%) | 10.0 | 81.3% |
| 3 | 4.93 | 171 | 82 | 70 (18%) | 10.3 | 81.3% |

Median 4.93 tok/s. Story-prompt harness the same day: 4.73 (plain) with the
machine at 8.5 GB of a 9.2 GB swap limit and the kernel's memory-pressure
level at critical; 5.40 the day before on the same binary and prompt. The
server's footprint is 18.0 GB (11.9 GB malloc'd slot cache, 3.4 GB mapped
weights, 2 GB Metal). On this 24 GB machine the shipped budget leaves ~6 GB
for everything else, and run-to-run spread tracks how much of that has
been swapped.

Other baseline facts: TTFT 17.9 s for the 262-token prompt (prefill);
misses 52-95 per token depending on prompt (145-240 MB/token of expert
reads, 0.7-1.3 GB/s average, well under the SSD); load latency p50 4 ms,
p95 16 ms; router readback 0.02 ms/token; cache plan 0.9-1.1 ms/token
including all adoption memcpys; I/O queue 0.2 ms; I/O-completion-to-fixup
submit 1.1 ms; 57% of misses are reloads of experts the cache once held.

## 3. GPU occupancy: Case A

Merged busy / span over the whole request: 36.9% (run 1), 35.3% (later
run). Busy per token 103-120 ms of a 190-270 ms token. The GPU is not
saturated; the kernels are not the limit. Where the idle sits (gaps
between consecutive command buffers, per token):

| transition | ms/token | meaning |
| --- | ---: | --- |
| moe_phase1_hit -> moe_phase1_miss_fixup_phase2 | 69-76 | exposed expert I/O inside a layer |
| sample -> embed | 48-74 | the between-token gap (see 4) |
| shared_expert -> moe_phase1_hit | 11-14 | CPU turnaround per layer: readback, plan, pin, encode |
| shared_expert -> moe_phase1_2_routed | 8-9 | same, on all-hit layers |
| embed -> attn_norm_qkv | 3 | n-gram gather and layer-0 setup |

Busy roles per token: attention 36 ms, head 9-10, phase-1/2 routed 9,
tail+router 9, phase-1 hit 7.6, shared expert 2.9, sampler 0.17.

## 4. The between-token gap: one-time re-wiring, averaged

The segment counters place the sample->embed gap entirely in the produce
preamble, and inside it entirely in `setExpertCachePinned`: 73, 91, 125 and
170 ms per token across runs, tracking memory pressure, with sampler 0.4,
callback 0.05, loop remainder 0.6, embed 0.7 and gather 2.4 ms. Logging
every slow call showed it is one call per request: the first decode token
re-wires all 48 layers (4,716 ms in the logged run, 1.6-3 s in others), and
every later token returns early in under 2 ms. Prefill unpins the 12 GiB
cache at its start; under pressure the pages are swapped out while prefill
streams 76 GB of expert tiles past them, and the first decode token faults
them back. Amortized over 512 tokens that is 3-9 ms/token; over 48 tokens
it is 98 ms/token and 29% of decode time.

What was fixed: the per-layer `ProcessInfo.environment` read in the pin
path (the env-reads regression pattern, ~1.5 ms/token) is a static, the
48-layer walk returns early once complete, and `NVMAI_KEEP_WIRED=1` now
wires each layer as it opens instead of at model load before any layer
exists. Measured at 512 tokens, kept wired vs shipped: 5.75 / 5.52 vs 5.72
/ 5.88, a wash on throughput; prefill was not slower with the cache held
(17.3 / 17.7 s vs 17.9 / 17.6). The win from keeping the cache wired is
first-token latency and short generations, not steady-state tok/s.

## 5. Closed by measurement

- Prefetch -> cache zero-copy promotion: the adoption `memcpy` of one
  expert (2.64 MiB) costs 0.06 ms at 45 GB/s; the whole cache plan
  including every adoption is 0.9-1.1 ms/token (<0.5%). Not worth the
  ownership complexity. Closed.
- N-gram gather: 1.55 ms/token serial, 0.30 with all 16 reads in flight,
  measured on an idle SSD. Worth ~1 ms/token; below the noise floor. The
  reader's header is right that it could be issued off the critical path,
  but there is little to gain.
- MTP: 1.94 tok/s with the draft head vs 4.73 without on the same story
  prompt (this run); the earlier campaign found width-2 verification costs
  2.8x against a 2.0x ceiling. Closed as a net loss. MTP-guided expert
  prefetch cannot work for this model anyway: routing needs each layer's
  hidden state, which a draft token does not provide.
- Deeper or two-layer-ahead prefetch: a wash (5.34 / 5.17 and 5.41 / 5.24
  vs 5.21 / 5.33). The two-ahead probe is 58% precise as a set.
- Kernels: phase-1 layouts, HC gate fusion, GEMV variants, the head GEMV
  (89.9 GB/s, at the ceiling) all measured at or near bandwidth earlier.
- MTP fusion's commit-then-wait: real, but on a path that loses 2.4x for
  other reasons; not worth touching until MTP itself is viable.

## 6. Where the token goes, and the three opportunities

Steady-state token at 5.8 tok/s (~172 ms): GPU busy ~105, exposed expert
I/O ~55, per-layer CPU turnaround ~20, everything else ~5.

### A. Expert I/O: fewer wasted speculative bytes (Phase 5 lever, low risk)

The probe router's next-layer prediction is 91% precise at rank 1, but
what the ring actually issues, the first M predictions not already
resident, is 38% precise at the shipped depth: 1.37 reads per layer, 0.53
useful, 0.85 wasted = 112 MB/token of speculative bytes on an SSD that the
demand reads of the next layer are queued behind. The probe's normalized
weights separate the two: non-resident predictions with a margin of 0.02
over the 10th weight are 59% precise, 0.03 gives 67%. Simulated on the
weighted trace:

| rule | reads/layer | precision | useful/layer | wasted MB/token |
| --- | ---: | ---: | ---: | ---: |
| shipped (rank order, cap 2) | 1.37 | 0.38 | 0.53 | 112 |
| margin >= 0.02, cap 4 | 0.64 | 0.59 | 0.37 | 35 |
| margin >= 0.03, cap 4 | 0.44 | 0.67 | 0.30 | 20 |

Implemented behind `NVMAI_PREFETCH_MIN_MARGIN` (default 0 = shipped).
Files: `RealForwardRunner+Decode.swift` (prefetch site, weights readback),
`RealForwardRunner.swift` (static). Correctness risk: none to output; the
ring only changes which speculative reads it issues. Expected upside: the
SSD spends up to 30 ms/token less on wrong reads; how much of that reaches
the exposed-I/O line is what the A/B decides.

### B. Cache policy: forget the prefill (Phase 3 lever, low risk)

57% of misses are reloads. Replaying the model's own route traces through
policies at 96 slots per layer:

| policy | short prompt | 3000-word prompt |
| --- | ---: | ---: |
| LRU | 0.818 | 0.730 |
| LFU (shipped) | 0.845 | 0.703 |
| aging-LFU (existing, unselectable) | 0.831 | 0.750 |
| LRU-2 | 0.832 | 0.753 |
| decayed frequency, half-life 16 | 0.837 | 0.760 |
| Belady (oracle) | 0.897 | 0.863 |

LFU is best on a short prompt and worst on a long one, because prefill's
use counts outlive their relevance. A decayed count is within a point on
the short prompt and 5.7 points better on the long one, which is the
agentic case. Implemented as `ExpertCachePolicy.decayed` behind
`NVMAI_EXPERT_CACHE_POLICY` (with `aging-lfu` now selectable too). Files:
`PreadExpertStreamer.swift` (scoring, eviction), `RuntimeConfiguration.swift`
(override). Correctness risk: none to output; eviction order only. Expected
upside on a long prompt: ~28 fewer misses/token, ~20 ms of SSD time. The
oracle gap that remains (5-10 points) needs future knowledge the router
probe only partially provides.

### C. Per-layer CPU turnaround (medium risk, not attempted here)

~20 ms/token of GPU idle between the shared-expert buffer ending and
phase-1 starting: the tail wait, readback, plan, pin, argument build and
commit, ~0.4 ms per layer. The GPU-residency execution mode moves the
classification to the GPU but measured a loss on AgentWorld; a
double-buffered plan (encode layer L+1's hit phase-1 before layer L's fixup
completes) is the structural version. Files: `RealForwardRunner+Decode.swift`
(`encodeDecodeRoutedMoE`, `finishPendingRoutedCommand`). Expected upside
5-10%; risk: slot pin/lease lifetimes across two layers in flight.

## 7. What the A/Bs said, and what they exposed

Phase 3, 512-token story runs, two rounds interleaved (tok/s; expert hit rate):

| arm | round 1 | round 2 |
| --- | ---: | ---: |
| shipped | 5.62 (81.3%) | 5.85 (81.3%) |
| gate margin 0.02, cap 4 | 5.62 (81.3%) | 5.94 (81.3%) |
| gate margin 0.03, cap 4 | 5.56 (81.3%) | 5.87 (81.3%) |
| decayed policy, half-life 16 | 5.55 (80.0%) | 5.60 (80.0%) |
| keep cache wired through prefill | 5.75 (81.3%) | 5.52 (81.3%) |

The gate arms were identical to shipped to the fourth digit of the hit
rate, which is impossible if the gate had changed what the ring issues.
Per-token counters added to the ring settled it: the shipped ring issued
4.8 reads per token (0.1 per layer, against 1.37 the trace simulation
assumed) and adopted 1.45. At every `begin`, 1.78 of its 2 slots were
held by completed reads whose layer had not been planned. The reclaim
rule frees a terminal slot only when `slot.layer <= currentLayer`; a
prediction for layer 46 or 47 made at the end of one token carries that
layer index into the next token, where the current layer restarts at 0,
and a wrong prediction for layer 47 is never reclaimed at all. The ring
has been clogged since the first token of every generation, and the
prefetch depth, tier and lookahead experiments of the last week compared
variants of a mechanism that was effectively off (the 1.5 adoptions per
token were stale reads from the previous token that happened to match).

With a token-boundary reclaim the ring runs as designed: 62 reads issued
per token, 1.8 slots free at each begin, speculative queue wait 0.01 ms,
load 3.5 ms. And that loses: 4.9 adoptions per token (8% of issued reads),
expert I/O up from 75 to 102 ms/token, 3.8 tok/s. Reads take 3.5 ms
against a ~3 ms window before the next layer's plan, so half land late,
and every one takes SSD time the demand reads are waiting on. Break-even
needs roughly 50% precision and on-time completion; rank order gives 8%
in situ. Phase 5 measures the repaired ring against prefetch off and
against the margin gate at 0.03 and 0.06, which the trace puts at 67% and
~75% precision.

The decayed policy loses 1.3 points of hit rate and 4% on the short
prompt, as its replay predicted; its long-prompt case (+5.7 points in
replay) is not yet measured in situ.

Keeping the cache wired through prefill is a throughput wash and removes
the 1.6-4.7 s first-token re-wire; prefill was not slower with 12 GiB
held.

## 8. Phase 5: the repaired ring against prefetch off

512-token story runs, prefetch off in the control arm, two rounds:

| arm | tok/s | reads issued / adopted per token | expert I/O ms/token |
| --- | ---: | ---: | ---: |
| prefetch off | 5.81 / 5.83 | 0 / 0 | 64.5 / 64.8 |
| repaired ring, rank order, 2 slots | 4.70 / 4.66 | 57 / 4.2 | 85.4 / 86.3 |
| probe-margin gate 0.03 | 4.95 / 5.26 | 17 / 2.7 | 81.8 / 72.3 |
| probe-margin gate 0.06 | 5.10 / 5.41 | 7.8 / 1.4 | 78.4 / 71.4 |

Every prefetching arm loses, and the loss per speculative read is far
larger than the read's 0.75 ms of transfer: ~2.5 ms each at the 0.06
gate. A speculative batch holds the layer's reader while the demand batch
for the same layer queues behind it, and the read lands after the plan
that could have adopted it. One-layer-ahead prefetch is not viable on
this SSD in this I/O path, regardless of prediction quality.

The reclaim rule that clogged the ring dates from 2022f58 (2026-09-04),
which replaced "reclaim every terminal slot except the target layer's"
with "reclaim terminal slots whose layer <= the current layer" for the
two-layer-ahead experiment. Everything measured since -- the tier and
lookahead A/Bs, the 5.0.2 README table -- ran with prefetch effectively
off. The 4.7-era "+12% at depth 1" was on a working ring and an older I/O
path; it does not reproduce today.

## 9. Decisions and scorecard

| change | baseline | after | delta | correctness |
| --- | ---: | ---: | ---: | --- |
| pin walk: static env read, early return (per token) | ~1.5 ms/token | ~0 | +1% | golden identical |
| keep cache wired through prefill (profile row, Qwen 3.8) | first token +1.6-4.7 s | 0 | TTFT; decode wash (5.75 / 5.52 vs 5.72 / 5.88) | golden identical |
| prefetch ring: token-boundary reclaim (bug fix) | ring clogged | ring works | n/a alone | golden identical |
| prefetch off on every profile row | clogged ring (~off) | off | 0 vs measured state; +19-24% vs the repaired ring on | golden identical |
| probe-margin gate (opt-in, `NVMAI_PREFETCH_MIN_MARGIN`) | | | loses at 0.03 and 0.06 | off by default |
| decayed cache policy (opt-in, `NVMAI_EXPERT_CACHE_POLICY=decayed`) | 81.3% hit, 5.62 / 5.85 | 80.0%, 5.55 / 5.60 (short prompt) | -4% short; long prompt in section 10 | off by default |
| zero-copy prefetch adoption | memcpy 0.06 ms x ~1.5/token | not built | <0.1% | closed |
| n-gram gather parallel | 1.55 ms/token | 0.30 possible | ~+0.7% | not built |
| MTP | 4.73 | 1.94 | -59% | closed |

Steady-state decode on this machine and this day: 5.8 tok/s at the
shipped 12 GiB, 4.9-5.4 across the day's memory states. The limiting
resource is exposed expert I/O on a saturated SSD inside each layer, ~55
ms of a ~170 ms token, and it cannot be hidden by speculation on this
storage path. What remains above it is the GPU's own 105 ms and ~20 ms of
per-layer CPU turnaround; the 6+ tok/s target needs the turnaround (C in
section 6) or a cache policy that raises the hit rate on long prompts,
which is the last open measurement.

## 10. Cache policy on a long prompt

3000-word prompt (3,543 tokens of prefill, 145 s), 256 new tokens, prefetch
off, two rounds:

| policy | tok/s | hit rate | expert I/O ms/token | bytes read |
| --- | ---: | ---: | ---: | ---: |
| LFU (shipped) | 3.09 / 3.27 | 69.5% | 88.1 / 86.6 | 98.6 GB |
| decayed, half-life 16 | 3.38 / 3.23 | 70.5% | 79.5 / 80.6 | 95.4 GB |

+4% on the long prompt, -4% on the short one (section 7). The replay
overstated the long-prompt hit-rate gain (5.7 points predicted, 1.0
measured) because the runtime's per-layer plans and evictions differ
from a per-layer LRU replay in detail, but the I/O saving is consistent
in both rounds. Not shipped as the default: the sign depends on the
prompt. A longer half-life (32-64 replayed at 0.84 short / 0.73-0.75
long) is the tuning to try; a policy that switches from LFU to decayed
once decode has run for a few dozen tokens is the structural version.
Opt-in: `NVMAI_EXPERT_CACHE_POLICY=decayed`, `NVMAI_CACHE_DECAY_HALFLIFE`.

## 11. Retraction: multi-variable chain arms were the shipped configuration

The zsh chain scripts ran arms as `env $2 ...`; zsh does not word-split a
variable, so an arm string holding two assignments set one variable to
`"1 B=2"` and the arm ran the shipped configuration. Invalid, and to be
read as "shipped vs shipped": the 2026-09-04 Qwen 3.8 "two-layer-ahead"
and "utility tier depth 2 / 4" arms (so the "+3% utility tier" that went
into the profile row was noise), the throttle-tier arms with a depth, the
whole downstream "utility depth 2" pass on the 35B installs (washes,
trivially), and the phase-3 gate arms in section 7 (the four-digit
identity was the tell). Valid, single-variable: the throttle-alone loss,
every slot-count chain, keep-wired, phase 5, both policy chains, and the
early-hits chain after the fix (`env ${=2}`). A prefetch tier or depth
setting has therefore never been measured to help Qwen 3.8; prefetch is
off, which is consistent with everything valid.

## 12. Early hits: the per-layer turnaround lever, shipped

Opportunity C in section 6. Phase 1 for the resident experts no longer
waits for the CPU's cache plan: the residency classifier already runs on
the GPU behind the router, so its hit list, hit count and resolved slots
feed a new `moe_phase1_gate_up_act_pool_u16load` kernel that addresses the
pooled cache directly. The fixup pass then covers everything the
classifier called a miss, including any position the CPU plan later
adopts from a prefetch buffer, so the CPU plan remains the authority over
what is fetched and what is resident.

Where the buffer goes decides whether it helps:

| variant | tok/s | GPU wait ms/token | I/O hidden |
| --- | ---: | ---: | ---: |
| shipped (hit-fixup after the plan) | 5.82 / 5.96 | 75.6 / 74.9 | 17% / 16% |
| early hits inside the tail buffer | 5.08 / 5.09 | 99.3 / 99.1 | 0% |
| early hits in their own buffer | 6.02 / 6.10 | 72.4 / 70.9 | 10% / 6% |

Inside the tail buffer the CPU's wait for the router covers the hit work,
so the miss reads start only after it and nothing overlaps them. In its
own buffer, committed right behind the tail, the GPU starts the hits
while the CPU plans and issues reads: +3.4% and +2.4%, GPU wait -3.5
ms/token, both goldens byte-identical.

Smaller than the ~20 ms the gap table suggested, because the hits still
have to finish before the fixup can run; what is removed is the CPU round
trip ahead of them, not the hit work itself.

The story harness then disagreed: early hits on 5.07 / 5.43, off 5.34 /
5.23, a wash whose within-arm spread (0.36) is larger than the effect.
Two protocols, opposite signs, four runs each -- not enough to ship a
default on, and the same shape as the utility-tier result retracted in
section 11. `earlyExpertHits` therefore ships **off** on every row and is
opt-in as `NVMAI_EARLY_HITS=1`, which also selects the GPU residency
classifier and the pooled cache layout (both underlying switches still
override individually). A five-pair interleaved run is in section 13.

The 35B rows were never in scope: 40 layers of top-8 at a 94% hit rate is
a different turnaround share and needs its own A/B.

## 13. Early hits, settled: a wash

Five interleaved pairs, 512-token generations, off then on within each
pair so the machine's drift cancels:

| pair | off | on | diff |
| --- | ---: | ---: | ---: |
| 1 | 5.923 | 6.131 | +0.208 |
| 2 | 6.108 | 6.058 | -0.050 |
| 3 | 6.078 | 5.848 | -0.230 |
| 4 | 5.985 | 6.058 | +0.073 |
| 5 | 6.039 | 5.955 | -0.084 |

Mean difference **-0.017 tok/s (-0.3%)**, sd 0.166, 95% CI [-0.16, +0.13].
A wash. The +3.4% / +2.4% of section 12 was two unpaired rounds against a
machine drifting by more than the effect; the harness pair that
contradicted it was equally uninformative. Five pairs resolve it.

Why the gap table oversold it: the 8-10 ms/token charged to
`shared_expert -> moe_phase1_hit` is not all CPU turnaround. The shared
expert's buffer is short and finishes early, so the interval also
contains the miss reads that the hit buffer is deliberately waiting
behind. Removing the CPU round trip in front of the hits moves work
earlier in an interval that is still bounded by the SSD, which is the
same wall every other lever hit.

The code ships **off by default** on every row: `NVMAI_EARLY_HITS=1`
selects it (and with it the GPU residency classifier and the pooled cache
layout), output byte-identical either way. Kept rather than reverted
because it is the only implementation of "the GPU starts resident-expert
work before the CPU's plan", and its balance would change on faster
storage or a model whose hit rate leaves less I/O in the interval.

## 14. Conclusion

The limiting resource for Qwen3.8-Flash-Next 4-bit decode on this machine
is exposed expert I/O on a saturated SSD inside each layer, roughly 55 ms
of a 170 ms token, with the GPU busy 105 ms and idle the rest. Nothing
tried moves it: not prefetch (loses at every setting once the ring works),
not cache policy (sign depends on the prompt), not speculative decoding
(loses 59%), not kernel work (already at bandwidth), and not removing the
per-layer CPU round trip (a wash within +/-0.16 tok/s).

Measured decode today, shipped defaults: **5.8-6.1 tok/s** on a warm
machine, 4.5-5.4 when memory pressure has swapped part of the 12 GiB
cache out. The 6+ target is reached on a warm machine at the current
defaults; it is not reached by any of the levers in this document.

What did change: two real defects were found and fixed (the prefetch ring
had been clogged since 2026-09-04, and the expert cache was re-wired from
swap on the first decode token of every request), and a class of A/B
result was retracted (zsh word-splitting made every multi-variable arm run
the shipped configuration). What is left, in order of expected value:
more RAM (the whole spread between 4.5 and 6.1 is swap), faster storage,
or a model whose expert working set fits.
