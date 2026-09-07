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

## Where it stands

The engine runs, and it agrees with the oracle.

`NVMAIBench cpu35 <snapshot>` loads a snapshot and answers the same three
continuations the numpy reference checks itself with, at both widths. Against
the reference's own logits over the full 248,320-token vocabulary:

| prompt | max abs difference | cosine |
| --- | --- | --- |
| "Once upon a" | 0.00004 | 0.9999999 |
| "The capital of France is" | 0.00001 | 0.9999999 |
| "The quick brown fox jumps over the lazy" | 0.00003 | 0.9999999 |

That is float32 rounding order, not different arithmetic. And it is
*sequence* parity, not just position 0: the eight-token prompt threads the
KV cache, the delta rule's recurrent state and the convolution tail across
positions, which is where a wrong hand-off between tokens would hide.

Throughput, on the real model, tracks the synthetic bandwidth probe closely:

| threads | probe predicted | measured |
| --- | --- | --- |
| 1 | 7.5 tok/s | 6.2 |
| 2 | 14.8 | 12.2 |
| 4 | 27.7 | 20.0 |
| 8 | 31.6 | 19.1 |

The output is identical at every width, because the threading splits rows
and rows are independent — there is no reduction order to get wrong.

**The 4-bit snapshot is not worth using here.** It reads half the bytes and
runs at the same speed, 19.3 against 19.9 tokens a second, because unpacking
two lanes per byte costs what the saved reads buy back. At equal speed the
8-bit build is simply more accurate, so 4-bit is for a machine short of
memory and nothing else.

Loading is instant — 0.02 s — because the snapshot is memory-mapped rather
than read. The resident cost is the page cache's, which is the right owner
for pages read sequentially and never written.

## End to end

`NVMAIBench cpu35gen <snapshot> "<prompt>" [n]` takes text and returns text,
through the engine's own tokenizer loaded straight out of the snapshot:

```
prompt: "Write one sentence about a lighthouse." -> 8 tokens, threads 4
output: "A lighthouse stands as a silent sentinel on the cliffs, its beam
         cutting through the dark ocean to guide ships safely to shore."
8 prompt + 31 generated in 1.9s (20.0 tok/s)
```

It stops on the end-of-turn token rather than running to the limit.
Generation is greedy, deliberately: this engine distils sessions and checks
claims, both of which want the model's best answer and neither of which
wants variety — and a deterministic side-engine is one whose output can be
compared between runs.

| threads | end-to-end tok/s |
| --- | --- |
| 1 | 6.7 |
| 2 | 12.1 |
| 4 | 20.0 |

**One `madvise` hint was worth 2.6× of that.** The mapping was opened with
`MADV_SEQUENTIAL`, which is true of the access pattern and disastrous as a
promise: it also tells the kernel it may free pages once they are behind the
read point, and this file is read end to end again for the very next token,
fifty milliseconds later. The engine was re-faulting the whole model every
token and running at 7.6 tokens a second. `MADV_WILLNEED` — all of it will
be wanted — put it at 19.8. Worth remembering that a hint which describes
the access pattern correctly can still be the wrong thing to say.

## What does not exist yet

1. **A sampler.** Greedy is the only mode; temperature and top-p can be
   added when something needs them.
2. **Residency and scheduling** — one thread while a client generation is in
   flight, four in the gaps, per the measurement above. The knob exists
   (`CPUQwen35.threads`); the policy that sets it does not.
3. **A resident service** the memory subsystem can call, rather than a
   benchmark command.

## Order of work

Parity first, always. This project debugs a new family with numpy parity and
an activation dump, not by reading code, and four of five bugs in the last
port were Qwen 3.6 constants silently reused for a different model. Qwen3.5
shares an architecture family with both models already ported here, which is
exactly the condition under which that mistake is made.


## What the reference found on its first run

The converter was right and the reference was wrong, in one line.

Predictions were structurally healthy and semantically nonsense — a bare
space for "Once upon a", a colon for "The capital of France is" — with
logits far too flat. A logit lens over the layers put the failure at layer
4, right after the first attention layer, and an ablation settled it:
*zeroing the Gated DeltaNet blocks entirely made the model better*, which
only happens when those blocks are actively destroying the signal.

The cause: the GDN output gate is **SiLU** in the Qwen3-Next / 3.6 lineage
and **sigmoid** in Qwen3.8-Flash-Next, whose reference this file was ported
from. NVMAI's own kernel already carries the distinction as a function
constant, `FC_GDN_SIGMOID_GATE`, with a comment naming Qwen3.8 as the
sigmoid one. So the engine must select it per family as well — this is the
first constant to check when the Swift side disagrees.

That is the family-specialization hazard this project has been bitten by
before, in the other direction: four of five bugs in the last port were
Qwen 3.6 constants silently reused. Every baked function constant in a new
family is a suspect until a real continuation comes out right.

Worth keeping: the diagnosis cost four cheap steps and no ground truth at
all. Confirm the tensor set is complete against the source header; confirm
the tied head with `E @ E[t]`, which peaks on the token itself and its case
variants when the embedding and quantization are sound; logit-lens each
layer to find where meaning stops; then ablate whole block types. The
answer was the third-cheapest thing tried.
