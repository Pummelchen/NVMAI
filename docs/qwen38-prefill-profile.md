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

**Untried, and now the largest remaining prefill item.** Processing the whole
prompt layer-major rather than chunk-major
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

## Then the per-key reduction, tiled

The kernel gave each thread one head-dim element and reduced across the
threadgroup for *every* key -- ~2,048 reductions per query per head.
`attention_prefill_causal_qsa_tiled` gives a thread one key's whole dot product
instead, puts the tile's scores in threadgroup memory, and advances the running
softmax once per 128-key tile. Traffic is identical; only the barrier count
moves, O(keys) -> O(keys/128).

| | per-key | tiled |
| --- | ---: | ---: |
| prefill | 570.6 s | **541.2 s** (-5.2%) |
| QSA per layer-chunk | 3.25 s | **2.79 s** (-14%) |
| GDN per layer-chunk | 0.48 s | 0.47 s (control) |

**Cumulative: 653.9 s -> 541.2 s, -17.2%. QSA per layer-chunk 4.59 -> 2.79,
-39%.**

-14% is *less* than removing ~2,000 barriers per query suggested, and that is
the finding rather than a disappointment: **the reduction was not the dominant
cost.** What is left is the inner loop -- 128 serial iterations of dequantising
an 8-bit KV value and one FMA, unvectorised. Still 5.9x a GDN layer, and that
is where the rest sits. Vectorising the dequant is the next item, and it is a
smaller change than either of the two above.

**Metal caveat, learned the hard way:** these shaders compile at *runtime* from
`program_source`. `swift build` validated none of this and passed twice while
the kernel was malformed. A `[[kernel]]` attribute must stay adjacent to its
function, and every thread-position attribute in one signature must agree in
shape (`uint3 tid` with `uint threads` is rejected).

## ANE: not yet, and for a different reason than first thought

The eligible share is not fixed. At short prompts the 12 full-attention layers
are ~20% of prefill GPU; at 10k they are 41% of prefill wall time, because
attention is quadratic where the 36 GDN layers are linear. So the port is worth
more than the first estimate, not less.

It is still second in line, and **today's work lowered its value**: ANE targets
`prefill_attn_router`, which the two changes above cut by 39%, so the same
2.31x now applies to a much smaller base. Step 1 partly consumed step 3's
prize.

It would also have inherited both bugs above, with no CoreML export, no
sidecar, no one-resident-model rule, and no `>= 2048` fused-SDPA NaN hazard
sitting exactly on QSA's budget to reason about.

If tiling stalls, ANE becomes attractive: the selection is now a fixed 2,048
keys, which is the dense shape ANE wants, and `export_ane_prefill.py` is
hardcoded to Qwen 3.6 geometry (`D = 2048`, 40 layers, `FULL_LAYERS =
range(3, 40, 4)`) and gated to `qwen36` in two places.
