## NVMAI 5.0 — Qwen3.8-Flash-Next

NVMAI 5.0 adds a third model family, Qwen3.8-Flash-Next 125B-A6B, and runs
it on the same 24 GB machine as the 35B models: 512 routed experts at
top-10 streamed from SSD, hyper-connection residuals, gated-DeltaNet linear
attention, sparse full attention with the QSA indexer, and the PLE n-gram
block, at 4-bit and 8-bit. Bounded RAM use, SSD-resident routed experts and
deterministic greedy output are preserved for every family.

### Highlights

- **Qwen3.8-Flash-Next 125B-A6B** at 4-bit (162 GB) and 8-bit (220 GB),
  converted from Qwen's own bf16 release by `tools/prepare_qwen38.py`,
  installed through `tools/install_models.sh`. The 8-bit build promotes the
  router, indexer, scalar gates and five other families to bf16 where a
  ranking or a gate must match the reference exactly. Both quantizations
  are pinned by byte-identical goldens (`tools/golden-baseline.sh qwen38-4`,
  `qwen38-8`), which now also gate the server front end.
- **Long generation is faithful past 2,048 tokens.** The QSA indexer runs on
  decode and selects keys in prefill too, so long prompts are no longer
  refused; a simdgroup-per-key attention pass takes the sparse layers' decode
  attention from 31.6 to 3.45 ms/token at 3.7k context.
- **Prefill for long prompts.** Tiled sparse attention synchronises per tile
  instead of per key (-52% on the attention pass), the selection is
  compacted instead of scanned, and Qwen3.8's prefill chunk is 4,096.
- **Decode.** The router's top-k runs on one simdgroup instead of one thread
  (the router pair 11.0 → 3.8 ms/token, +5%), the expert-cache slot count
  no longer overshoots the RAM budget on wide-stride models (8-bit 2.8×), and
  the expert cache budget and prefetch depth are chosen per family and
  width. Qwen3.8 ships with its own sampling defaults (temperature 1.0,
  top-p 0.95).
- **Measured and closed, so nobody re-derives them:** the M3's practical GPU
  read ceiling is ~75–88 GB/s and the GEMV kernels sit within 15% of it;
  fusing the hyper-connection gates (5 dispatches → 2) is bit-exact and
  changes nothing; layer-major prefill, deeper prefetch, GPU-side fetch
  planning, CPU row-splitting and the async I/O paths all measured negative;
  the expert cache is not cold after a long prefill (decode-only 80% hit
  rate over the first 64 tokens). Each stays in the tree behind a flag or in
  `NVMAIBench` as a measured control.
- **Attribution tooling:** `NVMAI_KERNEL_SPLIT=1` times every decode kernel
  of the GDN chain, the residual glue, the router and the sparse-attention
  path as its own role; the route trace covers prefill positions; the
  prefetch trace gives decode-only hit rates.

### Performance

On the base 8-core M3 MacBook Pro with 24 GB, the 512-token story-generation
benchmark at the shipped defaults (fresh process, one discarded warmup, idle
machine; 4-bit the median of three runs, 8-bit a single run):

| Model | Quantization | Peak decode |
| --- | --- | ---: |
| Qwen3.8-Flash-Next 125B-A6B | 4-bit | **5.25 tok/s** |
| Qwen3.8-Flash-Next 125B-A6B | 8-bit | **2.03 tok/s** |

The 35B rows are unchanged from 4.6's publication and reproduce to within
0.5%. Qwen3.8 decode is bound by streaming 512 experts at top-10 from SSD,
where the 35B models stream 256 at top-8; see the README for the ceiling
analysis.

### Compatibility

- Requires macOS 26+, Apple Silicon. Binaries are not signed or notarized.
- The CLI now defaults to the native context (262,144 tokens), matching the
  server. `--max-context` still overrides it.
- An 8-bit install of a hyper-connection family from before the bf16
  promotions is refused at load with a message naming the fix (reinstall).
- The MTP draft head for Qwen3.8 installs and verifies but stays off: at
  4-bit the two-row verify costs more than its 71% acceptance returns.
