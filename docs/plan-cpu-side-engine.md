# Plan: a CPU side-engine

NVMAI is the runtime. The memory work needs a second, much smaller model —
to extract facts from a session, to check a reply against what the store
holds, to notice a contradiction — and that model must not take the GPU,
which is where the answer the person is waiting for is being produced.

So: a dedicated CPU engine, running Qwen3.5-2B from a snapshot this
project's own converter produced, on the cores the main engine leaves idle.

## The premise, and the part of it that is wrong

The premise is that the CPU is free. Measured on this machine while a 35B
generates, NVMAIServer uses **0.20 of one core out of eight**. That is true,
and it is about *cores*.

Decode is not bound by cores. It is bound by memory, on both sides. The GPU
already reads at 74–88 GB/s during a 35B decode, at this machine's practical
ceiling, and a 2B at 8-bit reads about 1.9 GB for every token it produces —
the tied output head included, which is read in full each time. The two
engines compete for one memory system.

Measured with `NVMAIBench cpu` while a 35B was generating:

| threads | GB/s | implied 2B tok/s at 8-bit |
| --- | --- | --- |
| 1 | 11.9 | 6.3 |
| 2 | 26.5 | 14.0 |
| 4 | 44.0 | 23.2 |
| 8 | 45.0 | 23.7 |

Two things fall out immediately. Four threads is the whole win: the four
efficiency cores add 1 GB/s, which is why the kernel's default width is the
performance-core count rather than `activeProcessorCount`. And the headline
number is good — 23 tokens a second is a usable side model. Idle, the same
sweep reaches 52.6 GB/s at four threads, so the GPU's load costs the CPU
about a sixth of its bandwidth even before the CPU takes any back.

## What it costs the model the person is waiting for

Measured, not estimated. The same generation -- 292 tokens, fixed prompt,
temperature 0 -- run against Qwen 3.6 35B at 8-bit, alone and again with the
CPU kernel reading at a fixed width for the whole window.

| CPU threads | CPU GB/s | implied 2B tok/s | 35B generation | cost to the 35B |
| --- | --- | --- | --- | --- |
| none | — | — | 25.9 s | — |
| 1 | 13.9 | 7.3 | 26.8 s | 3% |
| 2 | 24.8 | 13.1 | 29.2 s | 13% |
| 3 | 31.1 | 16.4 | 33.7 s | 30% |
| 4 | 42.4 | 22.3 | 34.0 s | 31% |

**The side-engine is not free, and the knee is sharp.** One thread is
effectively invisible: 3% is inside this machine's own run-to-run spread,
and it still buys 7 tokens a second of 2B, which is enough to distil a
session or check a reply. Two threads costs 13% for nearly double that.
Three costs 30% and buys almost nothing over two -- past that point the two
engines are simply taking turns at the same memory controller.

The clean runs came in at 25.6, 25.6, 26.1 and 25.9 seconds, so the spread
here is far tighter than the ±15% this project usually sees, and a 30%
effect is nowhere near it.

**So width is a scheduling decision, not a constant.** The engine already
knows whether a client generation is in flight, which is the only input the
policy needs: one thread while the person is waiting, four in the gaps --
between requests, and during the idle window consolidation already waits
for. That is the design this measurement produced, and it is the opposite of
what "the CPU is idle, so it is free" would have produced.

## What exists

- `sources/NVMAIKernelsC/int8_affine_gemv.c` — the NEON decode primitive,
  four accumulators, group sums hoisted.
- `sources/NVMAI/Kernels/CPU/Int8AffineGEMV.swift` — the Swift wrapper, with
  row-split threading whose result is bit-identical to single-threaded, and
  a performance-core-count default.
- `.build/qwen35-2b-affine-8bit` (1.9 GB) and `-4bit` (1.3 GB, K/V promoted
  to 8-bit where the measurement said it was worth it).
- `tools/qwen35_reference.py` — a stateful numpy reference, and the oracle
  everything below is checked against.

## What does not

Everything else. In the order it has to be built:

1. **Position-0 parity.** Load the snapshot, run one token, agree with the
   reference on the logits. This is also the first thing that will have
   actually executed the converter's output, which until now has been
   checked only by inspection — and this project has already shipped one
   converter whose zero-centred norms were unfolded.
2. **The blocks**, each checked against the reference's own dump before the
   next is started: RMSNorm, SiLU, the gated MLP, full attention with its
   fused output gate and partial rotary, and the Gated DeltaNet recurrence
   with its convolution tail.
3. **Sequence parity.** The four carried states — the KV cache, the delta
   rule's recurrent state, the convolution tail — are where a wrong hand-off
   between tokens hides, and position 0 is blind to all of them. This is
   also the only thing that can check `rope_theta` and the partial rotary
   fraction, which the checkpoint does not state and the converter now
   writes explicitly from the family default.
4. **Tokenizer and sampler**, reusing what the engine already has.
5. **Residency and scheduling** — one thread while a client generation is in
   flight, four in the gaps, per the measurement above.

## Order of work

Parity first, always. This project debugs a new family with numpy parity and
an activation dump, not by reading code, and four of five bugs in the last
port were Qwen 3.6 constants silently reused for a different model. Qwen3.5
shares an architecture family with both models already ported here, which is
exactly the condition under which that mistake is made.
