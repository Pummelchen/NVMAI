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
number is good — 23 tokens a second is a usable side model.

**What it does not yet say is what those 44 GB/s cost the 35B.** That is the
measurement this plan is gated on, and it is cheap: the same generation, run
once alone and once with the probe hammering, comparing tokens per second.
If the cost is small, the side-engine runs whenever it likes. If it is
large, it runs in the gaps — between requests, during the idle window
consolidation already waits for — and the design changes accordingly. Either
way that number is known before the engine is built, not after.

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
5. **Residency and scheduling** — when it may run, decided by the bandwidth
   measurement above.

## Order of work

Parity first, always. This project debugs a new family with numpy parity and
an activation dump, not by reading code, and four of five bugs in the last
port were Qwen 3.6 constants silently reused for a different model. Qwen3.5
shares an architecture family with both models already ported here, which is
exactly the condition under which that mistake is made.
