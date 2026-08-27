# Qwen3.8-Flash-Next port — verified design record

Requested in [issue #2](https://github.com/Pummelchen/NVMAI/issues/2). Every
number below was read from the official checkpoint's `config.json`, the
`transformers` `qwen4_exp` modeling source, or the quantized checkpoints'
tensor indexes — none is assumed. This document is the contract for the port;
later sessions implement against it.

## Sources — and the one thing the request cannot have

**No official MLX quantization exists.** `Qwen/Qwen3.8-Flash-Next` publishes
bf16 safetensors only (132 shards, ~360 GB — cannot be quantized locally on
this disk). Every MLX quantization is community-published. The uniform 4-bit
and 8-bit checkpoints are:

| pin | quant | total | notes |
| --- | --- | ---: | --- |
| `Vontra/Qwen3.8-Flash-Next-MLX-4bit` @ `de597762aa61` | affine 4-bit, **group 32** | 111.6 GB | no MTP, vision tower present in bf16 |
| `Vontra/Qwen3.8-Flash-Next-MLX-8bit-MTP` @ `545f9a172c6f` | affine 8-bit, **group 32** | 203.0 GB | MTP bundled |

**Decision 2026-08-27: wait for an mlx-community release** rather than pin
the Vontra checkpoints — the request was for official quantizations, and the
community table above is recorded so the pins can be wired quickly if the
decision changes. Re-check the group size when the mlx-community repos
appear; mlx_lm's default is group 64, which would make the group-32
parameterization below unnecessary.

Were community pins ever adopted, the trust model would match Ornith's
(repo + revision + source-index SHA) and must be stated in user-facing docs. There is **no
group-64 8-bit anywhere**, so the port must parameterize NVMAI's group size
(the manifest already records `groupSize` per slot; kernels and tools
hardcode 64 today). Affine group 32 costs ~6% more bytes per weight than
group 64 (5.0 vs 4.75 effective bits at 4-bit).

**Disk**: 284 GB free; 4-bit (111.6) + 8-bit (203.0) = 314.6 GB. Both cannot
be installed simultaneously without the user freeing ~35 GB or removing an
existing model (their call, never ours). Plan: 4-bit first.

## Architecture (`model_type: qwen4_exp`, text config)

The checkpoint is **multimodal** (`Qwen4ExpForConditionalGeneration`, vision
tower, interleaved mrope). This port is **text-only**, like the Ornith port;
with equal text positions the mrope collapses exactly to standard partial
RoPE, and `rotate_half` over the first 64 dims matches NVMAI's existing
NeoX-subdim convention.

Familiar (config-level deltas from Qwen3.5-MoE):

| | Qwen3.5-MoE (shipped) | Qwen3.8-Flash-Next |
| --- | --- | --- |
| layers | 40, (3×GDN → full)×10 | **48**, (3×GDN → QSA)×12 |
| hidden | 2048 | **2560** |
| experts | 256, top-8, ffn 512 | **512, top-10, ffn 640** |
| router | softmax→topk, no norm | softmax→topk, **norm_topk_prob=True** |
| attention | 16Q/2KV, dim 256, packed q+gate, per-head norms, sigmoid gate | **24Q**/2KV, dim 256, same structure |
| GDN | 16 QK / 32 V, dim 128, conv 4 | 16 QK / **48 V**, dim 128, conv 4 |
| RoPE | NeoX-subdim, factor 0.25, θ 1e7, 262K native | identical |
| vocab | 248,320 | identical |
| shared expert | 1, sigmoid-gated | identical (ffn 640) |
| MTP | 1 layer | 1 layer (hybrid full-attention; deferred) |

Genuinely new subsystems, in dependency order:

### 1. Hyper-connections (`hc_count=4`, `hc_lowrank=320`)

The residual stream is **4×2560 = 10,240 wide for the whole stack**. Entry:
`hidden = embeds.repeat(4)`. Every sublayer (attention and MLP separately)
runs a `GatedResidual`:

```
normed  = grouped_rmsnorm(streams)                    # per-2560 groups, one weight [10240]
mix     = sigmoid(W_up @ silu(W_down @ normed / 4))   # 10240 -> 320 -> 10240
input   = mean_over_streams(mix * normed_streams)     # -> 2560, feeds the block
inject  = 2 * sigmoid(W_inj @ normed / 4)             # -> 4 per-stream weights
streams = streams + block_output ⊗ inject             # outer product back into 10240
```

Final collapse is a model-level mixer (same, without inject) — **there is no
`model.norm`**; lm_head applies to the mixed 2560. KV cache, GDN state, and
expert flow are all per-2560-block; only the residual plumbing is 4-wide.
Cost: ~40 KB of extra weights per sublayer, elementwise + two skinny GEMVs.

### 2. Qwen Sparse Attention indexer (`self_attn.indexer.*`)

Per full-attention layer: `index_qk_proj` 2560 → (4+1)×128; q per-head
RMS-normed then RoPE'd; the single key stream is cached **raw** (pre-norm,
pre-RoPE). Selection per query:

- visible keys grouped in blocks of `compress_ratio=4`, block key = fp32 mean
  of 4 raw keys → RMS norm → RoPE at the block's first position;
- score = Σ over the 4 q-heads of relu(q·k) / √128;
- top `2048/4 = 512` blocks kept, plus the incomplete tail block always;
- result is a boolean mask ANDed into the causal mask of the main attention.

**Exactness window: whenever a query sees ≤ 2048+tail keys, everything is
selected and dense attention is bit-exact.** Beyond that the indexer is
mandatory for faithfulness. Runtime plan: dense path first (with a hard
context gate at the budget), indexer second. The indexer needs its own tiny
"raw key" cache (128 fp16/token/QSA-layer ≈ 3 MB per 12 layers at 4K).

### 3. PLE / n-gram embeddings (layer index 1 only, `ple_layer_ids=[2]` is 1-based)

16 hashed lookups per token (2 n-gram orders × 8 heads) into a ~320M-row
table (16 prime-sized head vocabularies just above 20M each; rows are
`2560/16 = 160` wide; the table is quantized and **51.2 B params ≈ 32 GB at
4-bit**). Hashing is exact integer math, CPU-side:

- multipliers: `splitmix64`-derived odd 63-bit constants per (layer, position)
  from `seed=1234` (shipped as a tensor — read it, do not re-derive);
- id = XOR of `token_shift_s * multiplier_s` for s in n-gram, mod the head's
  prime, plus the head's offset; token shifts reset at EOS boundaries
  (`_shift_right_ignore_eos` — a 2-token rolling context, cacheable like a
  conv state);
- gathered 16×160 → concat 2560 → `value_proj` (→2560) and `key_proj`
  (→10240); gate per stream = signed-sqrt of (key·query_normed)/√2560 through
  a sigmoid; plus a **dilated depthwise conv** (k=4, dilation=3, per-channel,
  silu) over the gated value, state length 9; result added to the hyper
  streams before the layer body.

Streaming: one token touches 16 rows × ~100 B (4-bit g32) — ideal SSD reads.
New `.gturbo` section: `ngram_table.bin` with a row-addressable layout, plus
resident buffers for multipliers/offsets/vocab-sizes.

### 4. Top-10 MoE and 3D expert storage

`MoE.maxStreamedExperts = 8` is load-bearing across argument buffers, the
K8-specialized phase-2 PSO, and the cache planner; top-10 needs those
parameterized. MLX stores experts stacked (`mlp.switch_mlp.gate_proj` etc.,
3D `[512, ...]` with per-row g32 scales); the repacker slices to per-expert
blobs exactly like today. Expert blob at 4-bit/g32 ≈ 3.07 MiB (vs 1.688
today), 512 per layer × 48 layers ≈ 75.5 GB of experts.

MLX tensor-name mapping (verified against the Vontra index):

| MLX (`language_model.model.*`) | HF (`model.language_model.*`) |
| --- | --- |
| `layers.N.mlp.switch_mlp.{gate,up,down}_proj` | `layers.N.mlp.experts.{gate_up,down}_proj` |
| `vision_tower.*` (bf16, unquantized) | `model.visual.*` — **skipped, text-only** |
| everything else | same suffixes; `A_log`/`dt_bias`/PLE buffers present ✓ |

## Performance expectations (projections, to be measured)

6 B active × 0.625 B/param (4-bit g32) ≈ 3.75 GB/token GPU reads → ~59 ms
floor on this M3 → **≤17 tok/s ceiling**; expert SSD traffic ~480
activations/token at a 3.07 MiB stride with an unknown hit rate against 512
experts/layer — the measured SSD saturation (~3.4 GB/s) makes **exposed
expert I/O the expected wall**, plausibly single-digit tok/s on 24 GB.
16–32 GB machines are exactly the audience; the memory story fits NVMAI's
thesis better than any model yet (only 6 B active, table lookups tiny).

## Phasing

- **P0 (foundations)**: this document; `qwen38flash` family + ArchConfig;
  group-size parameterization end-to-end; manifest v2 fields (ngram section,
  indexer/PLE/hyper geometry); repacker source pins + mapping + `ngram_table`
  writer. Installable artifact at the end of P0.
- **P1 (runtime)**: group-32 kernel support; hyper-connection kernels and the
  reworked layer loop; GDN at 16/48; dense QSA with the ≤2048 exactness gate;
  PLE path; top-10 MoE. CLI generation at the end of P1.
- **P2 (fidelity + scale)**: indexer for >2048; parity harness against
  `transformers` on short prefixes (the golden analog — greedy logit match on
  the exactness window); benchmarks; server enablement; 8-bit install once
  the disk decision is made; MTP sidecar last (MTP is measured-off anyway).

Deferred by decision: vision, MTP, YaRN >262K (native first).
