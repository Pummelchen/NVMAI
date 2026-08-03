# NVMAI Qwen 3.6 35B 4-bit cache stress result

Measured at NVMAI commit `cae93751459c56060ac16905cb496c581cc7a935`
on an 8-core Apple M3 MacBook Pro with 24 GB RAM, macOS 26.6, and Apple
Swift 6.3.3. The inference source at that commit is equivalent to
`2ddf68e48ea29ef60a082abba309b37ef6a64506`; intervening commits added only
benchmark evidence and README changes.

## Workload

- 10 coding conversations, each with an initial request and two follow-ups
  (30 measured API requests per configuration; 60 total).
- Explicit OpenCode `coding-lean` headers and the captured OpenCode 1.15.11
  system guidance.
- 4,096-token context, temperature 0.2, Top-K 64, Top-P 0.95, fixed
  per-conversation/per-turn seeds, and at most 128 generated tokens.
- One discarded warmup before each measured server process.
- Cache-on: `multi-prefix`, 64 entries, 512 MiB RAM, 4 GiB SSD.
- Order: 4-bit off, then 4-bit on. Each retained block was QEMU-free.

`Decode tok/s` is an SSE estimate over the interval from first to last visible
content token. It excludes post-generation state capture and SSD persistence.
`E2E output tok/s` is generated tokens divided by complete request wall time,
including prefill and cache publication.

## Whole 30-request runs

| Cache | Prompt tokens | Cached tokens | Generated tokens | Decode tok/s | E2E output tok/s | Mean TTFT | Total wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Off | 6,337 | 0 | 2,669 | 12.41 | 7.57 | 4.59 s | 352.4 s |
| On | 6,382 | 4,827 | 2,521 | 9.84 | 7.40 | 2.72 s | 340.9 s |

Whole-run cache-on versus cache-off:

| TTFT reduction | Decode tok/s | E2E output tok/s | Total wall time |
| ---: | ---: | ---: | ---: |
| +40.8% | -20.7% | -2.3% | 3.3% faster |

## Follow-ups only (20 requests per configuration)

This is the meaningful cache comparison because initial turns cannot hit a
prefix.

| Cache | Prompt tokens | Computed prompt tokens | Cached share | Generated tokens | Decode tok/s | E2E output tok/s | Mean TTFT | Total wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Off | 5,434 | 5,434 | 0% | 1,744 | 12.18 | 7.17 | 5.02 s | 243.4 s |
| On | 5,479 | 652 | 88.1% | 1,596 | 9.46 | 7.48 | 2.13 s | 213.5 s |

Follow-up cache-on versus cache-off:

| Computed prefill reduction | TTFT reduction | Decode tok/s | E2E output tok/s | Total wall time |
| ---: | ---: | ---: | ---: | ---: |
| +88.0% | +57.5% | -22.3% | +4.3% | 12.3% faster |

## Validation and caveats

- 60/60 measured requests succeeded and every response was non-empty.
- Cache-on produced 20/20 intended follow-up hits: 2 live, 5 RAM, and 13 SSD.
  There were no restore, snapshot, or disk-write failures.
- Initial cache-on/off outputs were byte-identical for all 10 conversations.
  Four of 10 first follow-ups and none of the second follow-ups remained
  byte-identical.
- The divergence follows the current continuation contract: cache-on preserves
  the originally generated token IDs, while cache-off reconstructs and
  re-tokenizes assistant text. Qwen's decode/encode round trip can differ by a
  few tokens, and temperature 0.2 amplifies small differences. Decode-rate
  comparisons are therefore realistic workload results, not token-for-token
  controlled kernel comparisons.
- Both runs ended with 22 `stop` and 8 `length` finish reasons. No looping or
  empty response was found, but capped answers are incomplete by construction.
- The measured cache benefit is prefill latency: follow-up TTFT and wall time
  improved, while steady decode speed was lower in this sample. Cache state
  publication and different generated-token paths also affect the whole-run
  comparison.
- This custom server/cache stress protocol deliberately differs from
  `docs/COMMUNITY_BENCHMARKS.md`: it uses 10 coding conversations rather than
  the three frozen community prompts, a persistent server rather than fresh
  CLI processes per case, and 128 rather than 1,024 maximum generated tokens.
- Four attempts were stopped and excluded when an independent XAIOS task
  started QEMU. NVMAI processes were stopped without touching QEMU; the final
  cache-off and cache-on blocks were retained only after completing without
  QEMU overlap. Their raw contaminated directories and derived cache state are
  intentionally absent from this evidence set.
