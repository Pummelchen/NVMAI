# Design: layer-major prefill for Qwen3.8-Flash-Next

**Status: proposal. Not implemented.** Written because the measurement that
motivates it is solid and the restructure is not something to start
improvisationally.

## The problem, measured

Prefill's expert cache is inert. A chunk routes essentially every expert in a
layer against 96 slots, so:

| 8k prompt, chunk 4,096 | |
| --- | ---: |
| expert hit rate | **0.4%** |
| expert bytes read | **111 GiB** |
| whole expert corpus | 68 GiB |
| `io_hidden_pct` | 0.00 |

Each chunk re-streams what the previous one evicted, so cost scales with the
**chunk count**. Raising the chunk 2,048 -> 4,096 cut reads 167.5 -> 111.0 GiB
and prefill 506.4 -> 450.5 s, which confirms the mechanism. 4,096 is the
largest allowed chunk and scratch scales with it, so that lever is spent.

Layer-major takes the expert-visit count to **one per layer** regardless of
prompt length: read each layer's experts once, use them for every token.

## Why this is smaller than it first appears

Two things I assumed were blockers are not.

**The KV cache is already position-addressed.** `kSlot(layer:position:)` and
`vSlot(layer:position:)` take explicit positions, and prefill attention gets
`kvValidCount: startPosition + t` computed from the chunk rather than read from
the cursor. `KVCacheManager.position` is bookkeeping -- validation and where
decode resumes -- not a structural constraint. Writes already happen ahead of
the cursor within a chunk today; a band is the same pattern at larger scale.

**Only the residual has to persist between layers.** `hidden` is
`chunkTokens * hiddenSize * residualStreams` (2,560 x 4 = 10,240 per token,
20 KB at fp16). Everything else in `PrefillChunkScratch` -- `normed`, `q`,
`attn`, the GDN temporaries, `routedX`, `h1` -- is per-layer working space,
lives entirely inside one layer's pass over one chunk, and can stay
chunk-sized and be reused.

That is the difference between ~110 KB/token (all scratch) and 20 KB/token
(residual only): a 8k-token band costs **168 MB**, not 900 MB.

## The shape

```
for band in bands(prompt):              # bounded by residual memory
    for layer in 0..<numLayers:         # experts read ONCE per layer per band
        for chunk in band.chunks:       # order preserved for recurrent state
            run layer on chunk
    kv.advance(by: band.tokenCount)
```

- **Band size** from a residual budget: `band = budget / (residualWidth * 2)`.
  At a 256 MB budget that is ~12.8k tokens, so most coding prompts are one
  band and get the full effect; longer prompts degrade gracefully to today's
  behaviour rather than failing.
- **Chunks within a band stay in order**, so GDN's recurrent state, both conv
  histories and the PLE latch see the same token sequence they do now.
- **QSA selection** per (layer, chunk) still sees only preceding positions of
  its own layer, because chunks are processed in order within the layer.
- **KV** is written positionally per (layer, chunk) exactly as today; only the
  `advance` moves to the end of the band.

## Expected gain

Reads scale with band count rather than chunk count. An 8k prompt at chunk
4,096 is 3 chunks in 1 band: **111 GiB -> ~68 GiB, about -39%**. Expert I/O is
roughly half of prefill, so **~15-20% of prefill**, growing with prompt length
because the chunk count grows and the band count does not.

## What has to change

1. `PrefillChunkScratch`: split the residual out of the per-chunk layout and
   size it per band. Every `scratch.hidden` access becomes band-relative.
2. `executePrefillChunk` (261 lines): invert the loops, and take the layer as a
   parameter rather than iterating internally.
3. `prefillChunkState` / `markCommitted`: currently one commit per chunk; it
   becomes one per band.
4. `kv.advance`: once per band.
5. `reserve` / `validateRange`: must tolerate writes across a whole band ahead
   of the cursor.

## Test plan

The goldens **cannot gate this**. Their prompts are single-chunk and never
leave QSA's dense-exact window, so they would pass whatever it broke. Same
trap as the QSA selection work.

1. **Output identity, long prompt.** A multi-chunk, multi-layer prompt
   (>= 3,306 words, as used for the chunk change) must produce byte-identical
   text with layer-major on and off. Put it behind `NVMAI_PREFILL_LAYER_MAJOR`
   so both orders exist on one build.
2. **Recurrent state specifically.** GDN state and the conv histories are the
   parts most likely to break under reordering and least likely to show up in
   a short comparison. Compare `NVMAI_ACT_DUMP` activations at the last token
   of each chunk boundary against the chunk-major run.
3. **Both goldens**, to catch anything that leaked into the single-chunk path.
4. **Then** the 8k A/B for the number, reporting `expert_read_mib` alongside
   prefill so the mechanism is visible and not just the outcome.

## Risks

- **Highest-risk change in this runtime.** Recurrent state across a changed
  visit order, with no automated gate.
- **Decode blast radius** is small but non-zero: `reserve`/`validateRange` are
  shared, and the chunk change already showed that touching KV sizing costs
  ~3% of decode.
- **The prize is real but not transformative.** ~15-20% of prefill, against a
  10k prompt that currently costs 450 s. Worth doing, not worth rushing.

## Recommendation

Do it behind an environment flag, default off, and flip the default only after
test 1 and test 2 pass. That keeps a working prefill path throughout, which is
the property the QSA work relied on and the reason those three changes could be
landed one at a time.
