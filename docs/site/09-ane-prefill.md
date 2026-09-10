> **Category:** Guides
> **Status:** draft v1 — to be reviewed before posting
> **Wiki source:** v4.5-ane-prefill design note, Features, Runtime-Controls

# ANE prefill and the Metal engine

NVMAI's decode runs on the GPU over hand-written Metal kernels. But the two phases of generation — **prefill** (reading the prompt) and **decode** (producing tokens) — have different bottlenecks, so NVMAI gives each the hardware that fits it. This article is the second half of that story: running prefill's attention on the **Apple Neural Engine (ANE)**.

## Why prefill is different from decode

- **Decode** is one token at a time, memory-bandwidth bound, and the expert-streaming path is already tuned for it. It stays on the GPU.
- **Prefill** processes the whole prompt and is **quadratic in prompt length** — on a real 6,103-token 4-bit prefill, the ten full-attention layers cost **84.3 s of 133.2 s (63%)**, and that share grows as the prompt grows. That's a different shape of problem than decode, and it's the one that doesn't fit the SSD-streaming design.

The ANE runs those full-attention prefill blocks far faster than the GPU does, with the model's real weights. So NVMAI routes *exactly* the full-attention prefill blocks to the ANE, and leaves everything else — the gated-DeltaNet layers, the MoE, the format, the KV cache, the server API, and all of decode — untouched.

## How it works

1. **One-time export** of a Core ML sidecar covering all the full-attention layers (about 540 MB, fp16):

   ```bash
   python tools/export_ane_prefill.py \
     --model models/ornith-1.5_35B_A3B_4Bit --max-history 12288
   ```

2. **Run with the flag on** (it's on by default now):

   ```bash
   NVMAI_PREFILL_ANE=on .build/release/NVMAIServer --model ...
   ```

   `NVMAI_PREFILL_ANE` accepts `off|on` and **fails closed** on anything else. With `on` and **no sidecar present**, the runner fails at load and prints the export command — it won't silently pretend.

   > **Falls back to the GPU when there's no sidecar.** If you haven't exported one, ANE prefill simply isn't available and the GPU path runs. Nothing is lost; you just don't get the speedup. Short prompts and all of decode are on the GPU regardless.

## What it costs, one time

- **First request per machine:** a one-time ANE *specialization* per function (~130 s across all variants), cached by the OS afterward. This is Core ML JIT, not something NVMAI re-pays.
- **First request per process:** ~0.5 s per layer-chunk of model load.

After that, the steady state is the measurement below.

## Measured (M3, 24 GB, 4-bit, 6,103-token prompt, greedy, cache off)

Interleaved gpu/ane/ane/gpu, fresh server per run, one discarded warmup per arm (`benchmark/nvmai_ane_prefill_ab.py`):

| | prefill median | runs | decode after prefill |
| --- | ---: | --- | ---: |
| GPU path | 132.90 s | 132.85 / 132.95 | 8.70 tok/s |
| ANE path (warm OS cache) | **57.52 s** | 57.58 / 57.46 | 8.68 tok/s |
| | **2.31×** | | unchanged |

Two things in that table matter. The **2.31× is prefill only** — long prompt, time to first token. **Decode is unchanged** (8.70 → 8.68 tok/s), which is the design working: the ANE touches only the attention blocks and the GPU's decode path is untouched. One caveat the project is explicit about: **the ANE path is not byte-identical to the GPU path.** Different hardware, slightly different numerics. For production serving that's fine; for a golden baseline that has to be byte-for-byte, it isn't.

## A constraint worth knowing

**Keep one model resident when ANE prefill is on.** The Core ML arenas the ANE uses can evict the expert slot cache, so running ANE prefill against a second model in the same memory is how you'd break the budget the rest of the engine is trying to hold. One model, one process — the project's standing rule — is exactly the shape ANE prefill wants.

## Where to go next

- What prefill feeds and why the first token is slow: [Long context and KV cache](#08)
- The flag's place in the control set: [Runtime controls](#07)
- Measuring it against the GPU path: [Benchmarking](#11)

*Version at time of writing: NVMAI 5.1.*
