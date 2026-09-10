> **Category:** Guides
> **Status:** draft v1 — to be reviewed before posting
> **Wiki source:** System-Design (bounded expert streaming), v4.1 / v4.2 design notes

# SSD expert streaming: how NVMAI runs a model bigger than your RAM

This is the feature NVMAI is named for, and the one worth understanding before you tune anything else. Everything else — the RAM budget, the hit/fixup overlap, the prefetch — is a consequence of this.

## The problem

A 35B-A3B mixture-of-experts model has 256 routed experts per layer and picks **8 of them for every token**. The weights are 15–40 GB. A Mac is 16–64 GB. Load the whole model and it doesn't fit, or it fits but evicts your OS and you have a frozen Mac.

The naive options are all wrong:

- **Put it all in RAM.** Doesn't fit, or chokes the system.
- **Quantize harder.** You already are (4-bit); the model is still huge because it's *wide*, not because of precision.
- **Offload to a second GPU / cluster.** Not what a Mac is for.

## The idea: split "always needed" from "routed"

NVMAI separates the weights into two groups:

1. **Shared / always-needed tensors** — the attention heads, norms, embeddings, the router. These are read for *every* token, so they **stay resident in RAM**.
2. **Routed expert weights** — the 256 experts per layer. For a given token only 8 are used, and *which* 8 changes token to token. These are **not all kept in RAM.** They live on the NVMe drive, and NVMAI streams the ones the current token selected into a **bounded per-layer cache**.

So the model's resident footprint is: shared weights + a small sliding window of experts, not the full model. The rest sits on the SSD and is fetched on demand. Model size is now bounded by **disk** — which is the dimension where Macs actually have headroom.

## What "bounded" and "streaming" actually mean

- **Bounded.** The expert cache is a fixed size (see [The RAM budget](#04) for how it's set). It never grows to "all the experts." When a new expert is needed and the cache is full, one is evicted.
- **Streaming.** A cache **miss** is a read from the SSD. NVMAI issues these reads in **parallel**, and it does not wait for them: during decode it immediately runs the experts it already has (the **hits**), and does a short bounded **fixup** once the missed experts arrive. The GPU never idles waiting on the disk.
- **No page cache.** Expert reads bypass the OS page cache by default (`NVMAI_BOUNDED_IO` on), so the budget you configure is the *real* working set. Otherwise the kernel would quietly keep evicted experts around and your "8 GB budget" would be a lie.
- **Slot leases.** The GPU can't read a buffer that's mid-eviction. Each cache slot is *leased* until its Metal command completes, so a slot is never overwritten under the GPU's feet. This is a correctness guarantee, not a performance feature.

The data flow per token:

```
attention / router on the GPU
      ↓ decides which 8 experts this token needs
selected experts already cached?  ── hit ──────────────┐
      │                                                 │
      └─ miss → read from NVMe (parallel) → fixup  ────┤
                                                       ↓
        combine shared + routed branches → output head → sample token
```

## Why the SSD has to be fast

Decode is ultimately a **memory-bandwidth** game, not a compute one. Each token reads a large amount of weight bytes, and under SSD streaming some of that read comes over NVMe instead of from RAM. The expert pool that fits in RAM runs at full speed; the active pool that doesn't is limited by scattered SSD reads. That's why NVMAI is explicitly an **SSD** streamer — the whole trick only works if the drive can feed the GPU at the model's decode rate, and why NVMe (not a spinning disk, not a slow SATA SSD) is the floor.

The prefetch machinery (depth-1 speculative reads, on for the 8-bit families) is an optimization on top of this: it starts reading the *next* layer's predicted experts while the *current* layer is still running. It's a bandwidth decision — measured at +5.5% to +21% where it helps, and a regression where expert I/O is a small fraction of the token (see [Benchmarking](#11)).

## What this is *not*

- **Not offloading.** Offload moves whole layers or tensors between memory tiers and pays for the transfer every time. Streaming keeps the *shared* path hot and only moves the routed part, and overlaps the move with the compute.
- **Not a smaller model.** The weights are the real 35B/125B, at the installed 4- or 8-bit precision. You are not getting a distilled or pruned model; you are getting one that fits by design.
- **Not free bandwidth.** The SSD has a fixed read rate. If your active expert set doesn't fit in the cache, decode slows toward the drive's ceiling. That's the physical limit, and it's why bigger models (125B) run slower even though they "fit."

## The receipts, in one line

Each installed model is a `.gturbo` directory with aligned tensors, a manifest, tokenizer assets, and a `verified-install.json` receipt bound to its path. The converter (`NVMAIRepack`) builds that straight from the original weights — no MLX, no GGUF in the loop. (Full story in [Installs and verified receipts](#05).)

## Where to go next

- Why the cache is sized the way it is: [The RAM budget and bounded expert cache](#04)
- The install format in depth: [Installs and verified receipts](#05)
- The numbers behind the bandwidth claims: [Benchmarking](#11)

*Version at time of writing: NVMAI 5.1.*
