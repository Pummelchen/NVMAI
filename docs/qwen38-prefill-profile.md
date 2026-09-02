# Qwen3.8-Flash-Next prefill, measured

Long-prompt work is prefill-dominated on this model, and prefill had never been
profiled. Measured on the 24 GiB M3, 4-bit, `NVMAI_KERNEL_STATS` +
`NVMAI_RUNNER_STATS`.

## Prefill is the cost for coding-shaped prompts

| prompt tokens | prefill |
| ---: | ---: |
| 320 | 19.4 s |
| 1,244 | 47.9 s |
| 5,006 | 298.1 s |
| 10,022 | **651.8 s** |

Eleven minutes before the first token on a 10k prompt, against ~42 s to decode
256 tokens. For anything long-context, decode throughput is not the number that
matters.

## Where an 8k-prompt prefill goes

| role | seconds | share |
| --- | ---: | ---: |
| `prefill_attn_router` (12 full-attention layers) | 271.6 | **41%** |
| `prefill_routed_tile` | 107.2 | 16% |
| `prefill_gdn_router` (36 linear layers) | 90.2 | 14% |
| `prefill_shared_expert` | 50.4 | 8% |
| `prefill_qsa_index` | 7.4 | 1% |
| non-GPU (exposed expert I/O) | ~137 | 21% |

## The expert cache does nothing in prefill

For one 8,192-token prompt:

| | |
| --- | ---: |
| expert hit rate | **0.6%** |
| expert bytes read | **168 GiB** |
| evictions / reloads | 60,665 / 47,423 |
| `io_hidden_pct` | **0.00** |
| load p50 / p95 / p99 | 8 / 128 / 256 ms |

A 2,048-token chunk routes essentially all 512 experts per layer; the cache
holds 96. So each chunk re-streams what the last one evicted, and one prompt
reads ~2.5x the entire 68 GiB expert corpus. Nothing overlaps.

**Untried:** processing the whole prompt layer-major rather than chunk-major
would read each layer's experts once. The activations for 8k tokens are ~170 MB
at the 4-stream residual width, so the memory is affordable. Worth ~12% on this
profile and nobody has attempted it.

## The QSA scan, found and fixed

QSA attention scaled with context when its 2,048-key budget should have bounded
it -- 1.42 s per layer-chunk at 2.4k context against 4.53 s at 10k, and 9x a
GDN layer whose projections are *larger*. The kernel skipped the arithmetic for
unselected keys but still visited every visible position to read a mask byte:
O(context) work per query per head to rediscover an O(budget) answer.

Fixed by publishing the selection compacted (ascending key ids + per-query
count) and iterating that. Measured at 8k, one build, one env var apart:

| | mask | compacted |
| --- | ---: | ---: |
| prefill | 653.9 s | **572.0 s** (-12.5%) |
| QSA per layer-chunk | 4.59 s | **3.31 s** (-28%) |
| GDN per layer-chunk | 0.49 s | 0.49 s (control) |

The saving grows with context, because the removed term was O(context).

**Still 6.8x a GDN layer.** The remaining cost is one threadgroup reduction per
key rather than per tile. That is the next item and a larger change.

## ANE: not yet, and for a different reason than first thought

The eligible share is not fixed. At short prompts the 12 full-attention layers
are ~20% of prefill GPU; at 10k they are 41% of prefill wall time, because
attention is quadratic where the 36 GDN layers are linear. So the port is worth
more than the first estimate, not less.

It is still second in line. ANE would have inherited the scan bug above, and
the cost that remains after fixing it -- per-key threadgroup reductions -- is
addressable in the same kernel by tiling, with no CoreML export, no sidecar, no
one-resident-model rule, and no `>= 2048` fused-SDPA NaN hazard sitting exactly
on QSA's budget.

If tiling stalls, ANE becomes attractive: the selection is now a fixed 2,048
keys, which is the dense shape ANE wants, and `export_ane_prefill.py` is
hardcoded to Qwen 3.6 geometry (`D = 2048`, 40 layers, `FULL_LAYERS =
range(3, 40, 4)`) and gated to `qwen36` in two places.
