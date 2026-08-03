# NVMAI Qwen 3.6 35B cache stress result

Measured at NVMAI commit `2ddf68e48ea29ef60a082abba309b37ef6a64506`
on an 8-core Apple M3 MacBook Pro with 24 GB RAM, macOS 26.6, and Apple
Swift 6.3.3.

## Workload

- 10 coding conversations, each with an initial request and two follow-ups
  (30 measured API requests per configuration; 120 total).
- Explicit OpenCode `coding-lean` headers and the captured OpenCode 1.15.11
  system guidance.
- 4,096-token context, temperature 0.2, Top-K 64, Top-P 0.95, fixed
  per-conversation/per-turn seeds, and at most 128 generated tokens.
- One discarded warmup before each measured server process.
- Cache-on: `multi-prefix`, 64 entries, 512 MiB RAM, 4 GiB SSD.
- Order: 6-bit off, 6-bit on, 8-bit on, 8-bit off.

`Decode tok/s` is an SSE estimate over the interval from first to last visible
content token. It excludes post-generation state capture and SSD persistence.
`E2E output tok/s` is generated tokens divided by complete request wall time,
including prefill and cache publication.

## Whole 30-request runs

| Quant | Cache | Prompt tokens | Cached tokens | Generated tokens | Decode tok/s | E2E output tok/s | Mean TTFT | Total wall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6-bit | Off | 5,882 | 0 | 2,198 | 6.99 | 4.11 | 7.35 s | 535.3 s |
| 6-bit | On | 5,948 | 4,396 | 2,217 | 5.61 | 4.09 | 4.69 s | 541.8 s |
| 8-bit | Off | 6,217 | 0 | 2,474 | 5.31 | 3.35 | 9.02 s | 738.0 s |
| 8-bit | On | 6,321 | 4,769 | 2,491 | 5.16 | 3.80 | 5.56 s | 655.0 s |

Whole-run cache-on versus cache-off:

| Quant | TTFT reduction | Decode tok/s | E2E output tok/s | Total wall time |
| --- | ---: | ---: | ---: | ---: |
| 6-bit | +36.2% | -19.8% | -0.3% | 1.2% slower |
| 8-bit | +38.4% | -2.9% | +13.5% | 11.3% faster |

## Follow-ups only (20 requests per configuration)

This is the meaningful cache comparison because initial turns cannot hit a
prefix.

| Quant | Cache | Cached share | Decode tok/s | E2E output tok/s | Mean TTFT | Total wall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 6-bit | Off | 0% | 6.85 | 3.88 | 8.04 s | 371.8 s |
| 6-bit | On | 87.1% | 5.40 | 4.17 | 3.79 s | 350.9 s |
| 8-bit | Off | 0% | 5.13 | 3.12 | 9.73 s | 501.4 s |
| 8-bit | On | 88.0% | 5.06 | 3.93 | 4.31 s | 402.5 s |

Follow-up cache-on versus cache-off:

| Quant | Computed prefill reduction | TTFT reduction | Decode tok/s | E2E output tok/s | Total wall time |
| --- | ---: | ---: | ---: | ---: | ---: |
| 6-bit | +87.0% | +52.8% | -21.2% | +7.4% | 5.6% faster |
| 8-bit | +87.8% | +55.7% | -1.2% | +25.9% | 19.7% faster |

Eight-bit versus six-bit on follow-ups:

| Cache | Decode tok/s | E2E output tok/s | Total wall time | Generated-token difference |
| --- | ---: | ---: | ---: | ---: |
| Off | -25.2% | -19.5% | 34.8% slower | +8.5% tokens |
| On | -6.2% | -5.6% | 14.7% slower | +8.3% tokens |

## Validation and caveats

- 120/120 measured requests succeeded and every response was non-empty.
- Cache-on produced 20/20 intended follow-up hits for each quantization:
  2 live, 5 RAM, and 13 SSD. There were no restore, snapshot, or disk-write
  failures.
- Initial cache-on/off outputs were byte-identical for all 10 conversations in
  both quantizations. Only 2/10 first follow-ups and 0/10 second follow-ups
  remained byte-identical.
- The divergence is expected from the current continuation contract: cache-on
  preserves the exact originally generated token IDs, while cache-off
  reconstructs and re-tokenizes assistant text. Qwen's decode/encode round trip
  differed by several tokens, and temperature 0.2 amplified small differences.
  Aggregate decode-rate comparisons are therefore realistic workload results,
  not token-for-token controlled kernel comparisons.
- Finish reasons were 27 stop / 3 length for 6-bit off, 6-bit on, and 8-bit
  off; 8-bit on was 26 stop / 4 length. No looping or empty response was found,
  but capped answers are incomplete by construction.
- This custom server/cache stress protocol deliberately differs from
  `docs/COMMUNITY_BENCHMARKS.md`: it uses 10 coding conversations rather than
  the three frozen community prompts, a persistent server rather than fresh
  CLI processes per case, and 128 rather than 1,024 maximum generated tokens.
- Two earlier attempts were excluded: one used an incorrect decode timer, and
  another was invalidated when XAI OS QEMU and Roblox consumed most CPU.
  Roblox was closed with user authorization; QEMU ended independently before
  the clean runs.
