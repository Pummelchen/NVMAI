# Qwen 3.6 decode performance notes

Measured 2026-07-31 on an Apple M5 with 24 GB of memory, macOS 26.5, against
the installed `scratch/qwen36.gturbo` (19.55 GB) with the production runtime
profile: 16 expert slots, 4K context, RDADVISE off, greedy decode of 128
tokens from a 10-token prompt.

These are measurements from one host, not performance ceilings. Decode rate is
sensitive to OS page-cache state; see [Warming the page cache backfires]
(#warming-the-page-cache-backfires).

## Where decode time goes

`TURBO_FIELDFARE_PHASES=1` on the CLI reports the runner's phase counters:

| Phase | Time | Share |
| --- | ---: | ---: |
| Routed-expert I/O await | 3477 ms | 53% |
| GPU execution (waits) | 2976 ms | 45% |
| `cb1` encode + commit | 83 ms | 1% |
| `cb2` encode + commit | 52 ms | 1% |
| **Total decode** | **6588 ms** | 19.4 tok/s |

I/O and GPU are essentially serial: their sum is within 2% of the wall time.
That is inherent to the streaming design — a layer's routed experts cannot be
read until that layer's router has run, and the next layer cannot start until
this layer's MoE has finished.

Per token the runtime touches about 1.6 GB of weights, of which roughly
540 MiB is routed-expert data (40 layers x 8 experts x 1.69 MiB).

## What is not the bottleneck

Each of these was measured before being ruled out.

- **Kernel math.** The INT4 GEMV kernels reach 140-150 GB/s at Qwen's shapes,
  at or near this machine's memory bandwidth. Specializing the pipelines on
  the model's shapes (now derived from `ArchConfig`) lifted the narrow
  4096x2048 projection from 101.6 to 141.0 GB/s, but moved end-to-end decode
  only about 3% because GEMV work is a small share of the token.
- **Metal encoder overhead.** Creating one encoder per kernel costs ~1.3 us
  versus ~1.0 us for a shared encoder. Across ~900 dispatches that is a
  0.25 ms/token difference — not worth restructuring for.
- **Expert cache hit rate.** Raising slots from 16 to 24 to 32 moved decode
  from 19.0 to 19.6 to 19.8 tok/s. With 256 experts per layer the hit rate is
  low at any slot count the memory budget allows, and prior work already
  found LFU within ~8 percentage points of Belady's optimal. Slot counts
  above 32 are not offered: the 2 GB footprint is a product requirement.
- **RDADVISE read-ahead.** `--rdadvise default` cut the I/O await from 3474 to
  2931 ms, but the synchronous advice calls cost what the reads saved; total
  decode was unchanged (19.3 vs 19.3 tok/s). It stays off by default.

## Round-trip latency

Decode performs one `commit` + `waitUntilCompleted` per layer to read the
router's top-8 expert IDs back to the CPU. A measured round trip on this host
is ~196 us, so Qwen's 40 layers spend ~7.8 ms/token (15%) in submission and
completion latency alone — against Gemma 4's 30 layers at ~5.9 ms. This is
the structural cost of having 10 more layers, and it cannot be removed
without breaking the router-then-read dependency.

## Warming the page cache backfires

Reading all expert files into the OS page cache before a run
(`cat packed_experts/*.bin > /dev/null`) **halved** throughput, from 19.4 to
11.1 tok/s, and raised the I/O await from 3477 to 4500 ms. Indiscriminate
warming evicts the pages the runtime actually reuses — the mapped common
weights and the LFU-hot experts — and drives up memory pressure. Throughput
recovers over the next few runs as the cache re-warms naturally.

This reproduces, on much newer hardware, the original finding that the
virtual-memory system cannot be trusted to keep the expert working set warm.
See [`mmap` versus `pread`](experiments/summaries/01-model-install-and-expert-io.md#io-01).

Because decode rate depends on cache state, benchmark runs should be repeated
until the rate stabilizes, and the steady-state value reported.

## Measuring

```bash
TURBO_FIELDFARE_PHASES=1 .build/release/TurboFieldfareCLI \
  --model scratch/qwen36.gturbo \
  --prompt "Write a detailed essay about the history of computing." \
  --max-new 128 --temperature 0
```

`--expert-cache-slots` (8, 16, 24, 32) and `--rdadvise`
(off, default, bounded, adaptive) vary the two policies discussed above.
