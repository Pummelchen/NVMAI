# Why NVMAI can run models that "don't fit"

The first article said NVMAI runs a 125B model on a 24 GB Mac. This is the
article that explains how, and — just as importantly — what that costs.

If you only want to *use* NVMAI, you can skip to
[What NVMAI will not do](10-what-nvmai-will-not-do.md). This one is for the
curious, and for anyone deciding whether NVMAI is the right tool.

## The trick is that the model is mostly optional

A modern model like this is a **mixture of experts**, or MoE. That is not a
marketing phrase; it describes the machinery.

The model is divided into many small specialist sub-networks — *experts* —
plus a **router** that decides, for each token, which few experts should
handle it. A 35B model might have 256 experts and use only 8 of them for any
given token. The 125B model has 512 experts and uses 10.

So at any instant, **the overwhelming majority of the model is not needed.**
The weights exist, but the computation touches a small, changing slice.

This is the crack NVMAI drives a wedge into.

- The **dense** parts — attention, the router, the embeddings — stay resident
  in memory, because they are needed for every token.
- The **experts** stay on the SSD, and only the chosen few are read into
  memory as the model generates.

Your memory holds the part that is always used plus a bounded cache of
recently-used experts. The rest of the model is on disk, which is exactly
where it can afford to be.

## The consequence: your disk becomes the budget

This is the idea in one line:

> **Model size is bounded by your free disk space, not your memory.**

A 20 GB install runs on a 16 GB Mac. The 125B model's 174 GB install runs on
24 GB. That is not a compression trick — it is the same weights, read on
demand.

It also explains something that surprises people: **the 125B model is not
mostly "model" in the usual sense.** About 95 GiB of its install is a hashed
n-gram lookup table — by itself larger than every 35B model here. That is part
of how it achieves its quality, and it is why the disk figures are so large.

## The cost — and the part everyone gets wrong

Here is where most explanations of SSD streaming stop, and where NVMAI's own
measurements get more interesting.

The obvious story is: *every token reads experts from disk, so token speed is
limited by storage bandwidth.* The project believed this. It wrote it down,
did the arithmetic, and published a ceiling:

| | |
| --- | ---: |
| Experts read per token | 480 (10 of 512, across 48 layers) |
| Size of one expert record | 2.77 MB |
| Everything read if the cache never hit | ~1.33 GB per token |
| Measured effective read bandwidth | ~3.6 GB/s |
| **Expert-I/O floor** | **~81 ms/token ≈ 12.3 tokens/s** |

That predicts the 125B model can never exceed ~12.3 tokens/s, and that
crossing it needs more memory, faster storage, or reading fewer experts.

**Then the project profiled the model properly, and the story broke.** The
measured cost of one token at 4-bit, 256-token warm request, on a 24 GB M3:

| Component | ms/token | Share |
| --- | ---: | ---: |
| **GPU compute** | **79** | **48.6%** |
| Exposed expert I/O | 42 | 26% |
| Control plane / gaps | 41 | 25% |
| **Total** | **162.5** | |

The binding constraint is **GPU compute, at 79 ms/token = 12.6 tokens/s** —
not the SSD. The I/O floor is real, but at the measured warm hit rate it is
about 48 ms, which would allow 19–21 tokens/s. A worker waiting on GPU
compute is the limit; the disk is merely expensive.

The project's verdict on its own earlier document is blunt and worth
repeating: *"The document's headline number is roughly right by coincidence
and wrong in its reason."*

## What the team tried, and closed

This is the part that explains why the engine is worth building rather than
just tuning. Every intuitive optimisation was measured. Almost all of them
lost.

| Lever | Result |
| --- | --- |
| Overlap I/O with compute | **Backwards.** Three arms raised hidden-I/O and all three *lowered* throughput |
| Immediate submission of speculative reads | −9.8% (they steal service from demand reads) |
| Prefetch depth 2 | −6.5% |
| Alternative cache policies (LRU, ageing-LFU) | −9.0% / −5.7% |
| 128 expert slots | **−68.5%** (it swaps: ~17 GB of cache on a 24 GB Mac) |
| GPU residency | −7.1% after fixing two defects |
| Attention kernel rewrites | +0.3% / +1.6% / −1.0%, inside a 3.2% drift |

**Overlap is not an objective when the device is already saturated** — an
overlapped read is a read taken from someone else. That inverts the advice
you would get from first principles, and it is only visible because someone
measured it.

There is a second, structural finding. Attention moves 745 MiB per token at
an effective 26 GB/s, while another kernel moves 644 MiB at 68 GB/s on the
same memory. That 2.6× gap *looks* like a kernel bug. It is not: the 48
attention layers are **dependent** — layer 49 cannot start before layer 48
finishes — so they cannot be fused into one large, efficient dispatch. As the
project puts it, 26 GB/s is what that shape costs. They closed the
investigation rather than re-opening it forever.

## The honest bottom line on speed

Only two things actually move the 125B model on a 24 GB machine:

1. **A Mac with more RAM.** At 128 slots the cache reaches 89.8% and would
   fit comfortably — on a 64 GB machine this is ordinary work, not
   engineering.
2. **GPU-side fetch planning**, so the processor never blocks on the router.
   This is an architecture change, not a setting.

And the measured performance at the shipped configuration, for the record:

| Expert slots | Cache hit rate | Speed |
| ---: | ---: | ---: |
| 64 | 78.1% | 5.58 tok/s |
| **96 (shipped)** | **85.4%** | **5.71 tok/s** |
| 128 | 89.8% | 1.80 tok/s — swaps on a 24 GB Mac |

The lesson in that table is the one worth taking from this whole article:
**more cache is not better if it does not fit.** 96 slots is the shipped
default because 128 is actively worse on the machine most people have.

## So how does it actually go fast?

Not by beating the physics — by doing the controllable work well.

**Prefetch, where it measures well.** If you know which experts the next layer
will want, you can read them while the current layer computes. That is worth
a measured **+21.3%** on the 125B model (5.735 → 6.957 tokens/s) and roughly
+5.5% on the 8-bit 35B models. It is switched **off** for the 4-bit 35B
models, where measurement showed it *cost* about 4%. Defaults come from
measurement here, not intuition.

**Use hardware that is not already saturated.** Prompt processing (prefill)
runs on the **Apple Neural Engine** where an exported Core ML sidecar is
present — about **2.3× faster** than the GPU cores on a long prompt, with no
configuration needed and a quiet fallback to the GPU when the sidecar is
absent. Sampling runs on custom Metal kernels: a three-stage tiled reduction
cut per-token sampling from 15.5 ms to 1.4 ms with a token-for-token identical
output. Follow-up questions reuse prior conversation state instead of
re-reading history.

**Keep the working set bounded and honest.** Expert reads deliberately bypass
the OS page cache, so the configured RAM budget is real rather than
advisory — which is why the budget can be trusted to keep your Mac
responsive.

Add those up and you get the published numbers: around **21.7 tokens/s** for
the 35B models in 4-bit, and **5.46** for the 125B — on a base 8-core M3 with
24 GB.

## Why build a new engine at all?

Fair question, since MLX and llama.cpp exist and are excellent.

- **The streaming is the point.** Reading experts off an SSD with a bounded
  cache, and knowing exactly what that costs, is the core problem. It is the
  whole reason NVMAI exists.
- **It is built for this hardware specifically.** Custom Metal kernels, an
  opt-in Neural Engine path for prefill, and memory budgets derived from Apple
  Silicon's unified memory — rather than a portable layer that must also serve
  three other platforms.
- **The format is owned end to end.** A converter builds NVMAI's format
  straight from the original published weights, and every install is verified
  with a receipt bound to its path. A model that loads is provably the model
  that was promised.
- **The failures are published too.** The rejected optimisations above are
  documented so nobody re-derives them. That is rarer and more useful than a
  feature list.

NVMAI is a focused fork of
[turbo-fieldfare](https://github.com/drumih/turbo-fieldfare) — an honest
lineage for a project that would rather credit its parent than pretend to
have invented everything.

## What this design is not good at

- **It will not be as fast as a model that fits in memory.** Streaming from
  SSD is a compromise, and it costs speed.
- **First-token latency on huge prompts is real.** Prefill is fast for what it
  is, but the 125B model is not instant.
- **Disk matters.** An external drive that sleeps, or a nearly full SSD, will
  hurt you.
- **More cache is not always more speed** — see the 128-slot row above.
- **Some limits are structural.** Dependent attention layers cannot be fused;
  that is the model's shape, not an unfinished optimisation.

## Where to go next

One article left, and it is the useful one →
**[What NVMAI will not do (and how to get help)](10-what-nvmai-will-not-do.md)**

The full measurement record — including the retired conclusions —
is `docs/qwen38-decode-20tps-concept.md` in the repository, and
[System Design](https://github.com/Pummelchen/NVMAI/wiki/System-Design) on the
wiki.

*Token breakdown, hit rates, and the rejected-lever table are from the
project's measured profile of 2026-09-02 on a 24 GiB M3. Published speeds are
the project's benchmark figures at NVMAI 5.1 on a base 8-core M3 with 24 GB.
Some inputs to the original 12.3 tokens/s analysis were later falsified by
measurement, and this article reflects the corrected findings.*
