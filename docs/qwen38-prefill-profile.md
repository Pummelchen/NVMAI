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
| QSA per layer-chunk | 3.25 s | **2.79 s** (-14%) |
| GDN per layer-chunk | 0.48 s | 0.47 s (control) |

-14% is *less* than removing ~2,000 barriers per query suggested, and that was
the finding rather than a disappointment: **the reduction was not the dominant
cost.** What remained was the inner loop.

## Then the inner loop

`prefill_load_kv` costs two integer divisions and three loads per element, and
the scale/bias pair only changes every `kvGroupSize` elements -- two distinct
groups for a 128-wide head, so 126 of every 128 lookups recomputed the same
answer. `prefill_qsa_dot` walks the head group by group:

    sum_e q_e * (quant_e * s + b)  ==  s * sum_e q_e*quant_e + b * sum_e q_e

Scale and bias apply once per group, and `sum_e q_e` does not depend on the key
at all -- it is a property of the *query*, computed once into threadgroup
memory and reused for every key it attends to, which removes the bias term from
the inner loop entirely.

| | tiled | + hoisted |
| --- | ---: | ---: |
| QSA per layer-chunk | 2.79 s | **2.21 s** (-32% vs per-key) |
| GDN per layer-chunk | 0.47 s | 0.47 s (control) |

## Where the kernel ended up

| stage | QSA/layer-chunk | vs GDN |
| --- | ---: | ---: |
| original (mask scan + per-key reduction) | 4.59 s | 9.4x |
| + selection compaction | 3.25 s | 6.8x |
| + tile synchronisation | 2.79 s | 5.9x |
| **+ inner-loop hoist** | **2.21 s** | **4.7x** |

**QSA prefill attention is 52% faster and the gap to a GDN layer has halved.**
All three are default-on (`NVMAI_QSA_COMPACT` / `NVMAI_QSA_TILED` set to `0`
fall back for A/B on one build). Both goldens reproduce.

**Prefill wall is the noisy metric, not this one.** In the last A/B the
unchanged per-key arm moved 570.6 -> 505.6 s with its GPU roles identical: that
65 s is the expert-I/O half varying between runs. Quote seconds-per-dispatch;
treat prefill wall as carrying I/O noise of that order.

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

## The chunk trade, measured on both widths

Raising the chunk to 4,096 was measured on prefill; it also costs decode,
because the KV ring is sized from it.

| chunk | 4-bit decode | 8-bit decode |
| --- | ---: | ---: |
| 4096 | 6.97 | 2.09 |
| 2048 | **7.19** | **2.17** |
| 4096 (repeat) | 7.00 | 2.02 |
| drift | 0.4% | 3.4% |

Both widths pay ~3-5% of decode for the longer chunk. The expectation was that
8-bit would not -- it is SSD-bound and runs a smaller cache since the slot fix
-- so the cost is *not* memory headroom specific to 4-bit, and **the prefill
chunk does not want a per-width split**. `decodeTuning` already dispatches on
`(family, weightBits)`; there is simply no evidence for splitting this one.

4,096 stays for both: 56 s of prefill on a 10k prompt against ~2 s from 3% of a
512-token generation.

**Machine state dwarfs all of it.** The same 4-bit config measured 5.14 tok/s
during a story benchmark and 6.97-7.19 here -- a 36% swing, an order of
magnitude larger than any tuning decision on this page. Cross-session absolute
decode numbers are floors, not capabilities; only interleaved comparisons with
a drift check carry weight.

