# NVMAI all-six rerun and best-of-two selection

Measured at NVMAI commit `c74f11ffa6c6d72ad5b106f700d95bb6ed0fd622`
on an 8-core Apple M3 MacBook Pro with 24 GB RAM, macOS 26.6, and Apple
Swift 6.3.3.

## Protocol

- Six configurations: 4-bit, 6-bit, and 8-bit with cache off and on.
- 10 coding conversations, each with an initial request and two follow-ups
  (30 measured requests per configuration; 180 total).
- Explicit OpenCode `coding-lean` headers and the captured OpenCode 1.15.11
  system guidance.
- 4,096-token context, temperature 0.2, Top-K 64, Top-P 0.95, fixed
  per-conversation/per-turn seeds, and at most 128 generated tokens.
- One discarded warmup before each measured server process.
- Cache-on: `multi-prefix`, 64 entries, 512 MiB RAM, 4 GiB SSD.
- Run order: 4-off, 4-on, 6-off, 6-on, 8-on, 8-off.

The selection rule was fixed before execution: compare prior and rerun by
whole-run end-to-end output tokens per second, then retain every metric from
the winning complete run. Individual cells are never mixed across runs.

## Rerun results

| Quant | Cache | Prompt tokens | Cached tokens | Generated tokens | Decode tok/s | E2E output tok/s | Mean TTFT | Total wall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4-bit | Off | 6,337 | 0 | 2,669 | 10.36 | 6.60 | 4.91 s | 404.5 s |
| 4-bit | On | 6,382 | 4,827 | 2,521 | 9.89 | 7.44 | 2.71 s | 339.1 s |
| 6-bit | Off | 5,882 | 0 | 2,198 | 5.44 | 3.41 | 7.99 s | 643.9 s |
| 6-bit | On | 5,948 | 4,396 | 2,217 | 5.66 | 4.15 | 4.60 s | 534.0 s |
| 8-bit | Off | 6,217 | 0 | 2,474 | 4.04 | 2.60 | 11.15 s | 950.0 s |
| 8-bit | On | 6,321 | 4,769 | 2,491 | 3.92 | 2.92 | 7.03 s | 852.3 s |

## Prior-versus-rerun selection

| Quant | Cache | Prior E2E tok/s | Rerun E2E tok/s | Rerun change | Selected run |
| --- | --- | ---: | ---: | ---: | --- |
| 4-bit | Off | 7.574 | 6.597 | -12.9% | Prior |
| 4-bit | On | 7.396 | 7.435 | +0.5% | Rerun |
| 6-bit | Off | 4.106 | 3.414 | -16.9% | Prior |
| 6-bit | On | 4.092 | 4.152 | +1.5% | Rerun |
| 8-bit | Off | 3.352 | 2.604 | -22.3% | Prior |
| 8-bit | On | 3.803 | 2.923 | -23.2% | Prior |

## Selected best complete runs

| Quant | Cache | Prompt tokens | Cached tokens | Generated tokens | Decode tok/s | E2E output tok/s | Mean TTFT | Total wall | Source |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 4-bit | Off | 6,337 | 0 | 2,669 | 12.41 | 7.57 | 4.59 s | 352.4 s | Prior |
| 4-bit | On | 6,382 | 4,827 | 2,521 | 9.89 | 7.44 | 2.71 s | 339.1 s | Rerun |
| 6-bit | Off | 5,882 | 0 | 2,198 | 6.99 | 4.11 | 7.35 s | 535.3 s | Prior |
| 6-bit | On | 5,948 | 4,396 | 2,217 | 5.66 | 4.15 | 4.60 s | 534.0 s | Rerun |
| 8-bit | Off | 6,217 | 0 | 2,474 | 5.31 | 3.35 | 9.02 s | 738.0 s | Prior |
| 8-bit | On | 6,321 | 4,769 | 2,491 | 5.16 | 3.80 | 5.56 s | 655.0 s | Prior |

Using the selected complete runs, cache-on reduced follow-up computed prefill
by 88.0%, 87.0%, and 87.8% for 4-bit, 6-bit, and 8-bit. Follow-up mean TTFT
fell by 57.9%, 55.3%, and 55.7%, while follow-up wall time fell by 12.8%,
9.4%, and 19.7%, respectively.

## Validation and caveats

- 180/180 rerun requests succeeded; every response was non-empty and every
  finish reason was `stop` or `length`.
- Each cache-on configuration produced 20/20 intended follow-up hits and no
  initial-turn hit: 2 live, 5 RAM, and 13 SSD per configuration. There were no
  restore, snapshot, or disk-write failures.
- No model/test/MLX/QEMU process existed before the run, and no QEMU process
  appeared during any retained block. Memory pressure remained acceptable.
- Closing applications did not improve every configuration. The two cache-on
  rows improved slightly, while the other four reruns were slower. Expert
  routing, SSD/page-cache state, sustained-run order, and ordinary host
  variability can outweigh small reductions in background application load.
- `Decode tok/s` measures the interval from first to last visible streamed
  content token. `E2E output tok/s` divides generated tokens by complete request
  wall time, including prefill and cache publication.
- This custom server/cache stress protocol deliberately differs from
  `docs/COMMUNITY_BENCHMARKS.md`: it uses 10 coding conversations rather than
  three frozen community prompts, persistent servers rather than fresh CLI
  processes per case, and 128 rather than 1,024 maximum generated tokens.
- The multi-gigabyte derived cache directories are intentionally excluded from
  Git; raw requests, aggregates, warmups, server logs, harness, and system
  metadata are retained.
