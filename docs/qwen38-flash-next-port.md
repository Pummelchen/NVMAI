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

## Cross-check against the official announcement (2026-08-27)

The launch blog (https://qwen.ai/blog?id=qwen3.8-flash-next) confirms the
verified geometry above and adds facts the config/source audit could not show:

- **This is the Qwen4 preview architecture.** Qwen states it plays the role
  Qwen3-Next played for Qwen3.5–3.8: the GDN+QSA / Gated-Residual / n-gram
  design will carry the whole Qwen4 family. The family-schema and ArchConfig
  work is therefore an investment in every upcoming Qwen model, not one port.
- **The n-gram offload strategy is official.** Qwen ships the 51 B-parameter
  table expecting it to live outside accelerator memory: "lookup locations
  can be determined in advance, these parameters can be stored in Host Memory
  and asynchronously prefetched in parallel with model computation." Our
  planned row-addressable `ngram_table.bin` on SSD is the same idea one tier
  down, with a stronger property than expert streaming: lookups depend only
  on token ids, never on router output, so decode can prefetch the next
  token's rows the moment it is sampled — the read is fully off the critical
  path. Prefill can batch all rows for a chunk up front.
- **"Gated Residual" is the official name** for what the config calls
  hyper-connections: 4 residual branches with element-wise, data-dependent
  read/write gates (the low-rank 320 projections). Qwen explicitly *removed*
  Hyper-Connection's branch-mixing matrices as unnecessary — the simplified
  form we mapped is the intended one. The residual state "supports FP8
  storage"; we start fp16 in Metal and note fp8-as-bytes as a later
  bandwidth lever (4 branches × 2560 quadruples residual traffic).
- **QSA indexing is per-layer by design** (independent sequence compression
  per layer, contrasted with cross-layer index reuse), so no shared index
  cache is needed. Claimed kernel speedups are 7.6× prefill / 4.9× decode at
  1M tokens — irrelevant below the 2048-token exactness gate, which supports
  dense-first phasing.
- **MTP is multi-step trained** for higher real acceptance, and its
  full-attention layers use QSA too. Still last in phasing; the 35 B verify
  economics don't transfer, so it would need fresh qualification.
- **Reasoning effort is not binary — verified and implemented.** The pinned
  upstream `chat_template.jinja` (fetched 2026-08-27) defines
  `reasoning_effort` `low|medium|xhigh` (default `xhigh`) while
  `enable_thinking` is on: `xhigh` and `low` inject an instruction sentence
  at the head of the system block, `medium` injects nothing, and invalid
  values raise. Thinking on also pre-opens `<think>\n` in the generation
  prompt (the off branch emits the closed block), and upstream
  `generation_config.json` defaults to temperature 1.0 / Top-K 20 /
  Top-P 0.95 — a per-family sampling-default question for P1. NVMAI now has
  the per-family control: `ModelFamily.reasoningControl` gates
  `--reasoning-effort` / `NVMAI_REASONING_EFFORT` / the API's
  `reasoning_effort` field (binary families reject them; effort is a
  load-time control validated against the manifest family), the tokenizer
  passes the effort into the bundled template and mirrors its system-block
  injection on the manual ChatML path, and the real template renders
  correctly under the Swift Jinja engine (fixture-tested, including the
  effort sentences). Remaining P1 wiring: `ManifestReader.peekIdentity`
  must learn to report `qwen38flash` (it derives qwen36/qwen36MTP from
  layer shape today), and the Mac app picker stays binary until the model
  is installable.
- **Totals as marketed**: 125 B backbone + 51 B n-gram + 6 B active/token;
  native 262,144 context, YaRN to 1M — matching the ArchConfig. Weights are
  live on HF/ModelScope (bf16 official); still no official MLX artifact, so
  the wait-for-mlx-community decision stands.

## Source found and verified against a real artifact (2026-08-28)

`RockTalk/Qwen3.8-Flash-Next-MLX-4bit` at revision
`478474da92599ad0cf9f8bd447e658b29cb8480a`
(`model.safetensors.index.json` sha256
`d7fe03ad2d1365e24ae2e305c829f600b80fac845d128383421a7be4adbdda1b`).
Not an mlx-community release — the wait-for-mlx-community decision is
superseded because this artifact answers every open question and mlx-community
has not shipped.

### The group-size question is answered: 64

`quantization: {group_size: 64, bits: 4, mode: affine}` — the format NVMAI
already uses. **The planned P1 group-32 kernel work is dropped.** Per-path
overrides are 8-bit g64 on `embed_tokens`, `lm_head`, `mlp.gate` (router) and
`shared_expert_gate`; everything else including all 512 routed experts is
4-bit g64. `manifest.quant` already carries independent `weightBits` per slot,
so this is expressible today.

### The design record is confirmed, not merely plausible

Every geometry figure matches: 48 layers, `full_attention_interval` 4 (36
`linear_attn` + 12 `self_attn` tensor sets, exactly the 3-GDN-to-1-attention
pattern), 512 experts top-10, ffn 640, hidden 2560, vocab 248,320, head_dim
256 with 24 Q / 2 KV, GDN 16 K-heads / 48 V-heads at dim 128, `hc_count` 4,
`hc_lowrank` 320, `ple_layer_ids` [2].

`ple_constants.json` confirms the PLE reverse-engineering independently:
`ple_n_heads` 16, `ple_head_dim` 160, 16 prime head vocabularies just above
20M, `ngram_size` 3, `heads_per_ngram` 8, eos 248044, and three layer
multipliers shipped as data (read them, as planned). Row count computes to
320,001,446 x 160 x fp16 = 102,400,491,520 bytes, matching `ngram_table.bin`
byte for byte.

QSA indexer tensors are present and separate per layer
(`self_attn.indexer.index_{q,k}_proj`, `{q,k}_layernorm`), and the upstream
card independently reports the mask is O(n^2) above 2051 tokens — the same
dense-exactness boundary the <=2048 gate was designed around.

There are **no layer norms**: `hc_norm` inside each hyper-connection block
replaces `input_layernorm` / `post_attention_layernorm` entirely.

### Sizes and the disk constraint

| part | bytes | |
| --- | ---: | ---: |
| backbone shards (49) | 71,425,417,784 | 71.4 GB |
| `mtp-weights.safetensors` | 1,467,252,412 | 1.5 GB |
| `ngram_table.bin` (fp16) | 102,400,491,520 | 102.4 GB |
| **total** | **175,293,161,716** | **175.3 GB** |

The n-gram table ships **fp16, not quantized** — 3.2x the ~32 GB the plan
assumed at 4-bit. With 288 GB free the full install fits **only because the
installer streams into `.gturbo` without staging a second checkpoint**:
downloading the repo first and then repacking would need ~352 GB and fail.
Quantizing the table during repack is possible but is a lossy change to a
component whose quality contribution is unmeasured — do not do it silently.

### What the repacker already handles, and what it does not

Qwen 3.6 shares the GDN+MoE lineage, so `RepackPlanner` already maps
`.mlp.switch_mlp.*` 3D fused experts, the whole `linear_attn` family
(`in_proj_qkv/z/a/b`, `conv1d`, `A_log`, `dt_bias`, `norm`, `out_proj`),
`shared_expert*` and `self_attn.{q,k,v,o}_proj`. New work:

- **tensor prefix differs**: `model.language_model.layers.N.*` here against
  qwen36's `language_model.model.layers.N.*`, and `lm_head.*` sits at the top
  level rather than under `language_model.`;
- **hyper-connections**: `attn_hyper_connection.*`, `mlp_hyper_connection.*`
  (`block_inject_weight`, `input_mix_weight_{up,down}`, `hc_norm`) plus a
  model-level `hyper_connection_mixer.*`;
- **QSA indexer** tensors per full-attention layer;
- **PLE**: `ple.conv1d` plus `ngram_table.bin` and `ple_constants.json` as
  passthrough sidecars, not safetensors tensors;
- **MTP**: `mtp.layers.0.*` is a complete layer with its own `self_attn`,
  indexer and **its own 512-expert `switch_mlp`** (hence 1.5 GB) — it should
  become an SSD-streamed sidecar like the existing Ornith/Qwen MTP path, not
  resident weights;
- **manifest v2**: ngram section, per-slot quant bits, indexer/PLE/hyper
  geometry.

### Trust

Created 2026-08-28, **0 downloads, 0 likes**, unknown author, no community
validation. The code is open (MIT, `Rocktalk-Holdings/mlx-qwen4exp`) and the
card reports measured M3 Ultra figures (26.4-26.6 tok/s greedy), so it reads
as serious work rather than a drive-by upload — but the quantization quality
is unverified by anyone. Pinning the revision plus the receipt's SHA256 covers
integrity, not quality.

### Quantization fidelity: VERIFIED SOUND (2026-08-28)

A greedy logit parity against `transformers` is not runnable here -- the bf16
checkpoint is ~360 GB against 288 GB free -- but the question underneath it is
answerable directly and cheaply. `benchmark/nvmai_quant_fidelity.py` pulls the
same tensors from both repos by HTTP range (no bulk download), dequantizes the
MLX affine blocks, and compares against the official bf16 values.

The meaningful test is not an absolute error threshold but whether the
quantizer reaches the floor the format allows, so each tensor is also compared
against an ideal affine round-trip of the official weights:

| tensor | bits | MAE ratio | ideal floor | excess |
| --- | ---: | ---: | ---: | ---: |
| L3 `self_attn.q_proj` | 4 | 0.1096 | 0.1074 | **1.02x** |
| L3 `self_attn.o_proj` | 4 | 0.1068 | 0.1047 | **1.02x** |
| L0 `linear_attn.out_proj` | 4 | 0.1158 | 0.1136 | **1.02x** |
| L3 `mlp.shared_expert.gate_proj` | 4 | 0.1120 | 0.1097 | **1.02x** |
| L19 `self_attn.q_proj` | 4 | 0.1058 | 0.1037 | **1.02x** |
| L3 `mlp.gate` (router) | 8 | 0.0096 | — | — |

**1.02x the floor everywhere.** The quantizer is as accurate as 4-bit affine
g64 permits; the residual 2% is consistent with quantizing from bf16. The
8-bit router lands at 0.96%, matching theory for 8-bit. Shapes, packing order
and the scale/bias convention all decode correctly against the official
tensors, which also validates the dequantization contract the repacker will
need. **Green light to build on this artifact.**

Note this measures weight fidelity, not end-to-end behaviour; a short greedy
logit comparison against the reference implementation is still worth running
once the runtime can execute the model.

## P0 implementation status (2026-08-28)

Landed, each gated by lint + the full serial suite and, where the runtime was
touched, both goldens byte-identical:

| piece | commit | what it does |
| --- | --- | --- |
| ArchInfo loader | `f6cd041` | reads the whole `qwen4_exp` config: layer mask, GDN, hyper-connection, indexer and PLE geometry; converts `ple_layer_ids` from 1-based and rejects out-of-range; reads the source group size instead of defaulting; contract-checks the production shape |
| tensor classification | `a11e17f` | the `model.language_model.*` prefix, top-level `lm_head.*`, and the MTP draft held out for its own sidecar; verified against all 3,164 real tensor names with zero unknowns |
| manifest geometry | `969225b` | writes and validates the new geometry, so a checkpoint whose hyper-connection rank, indexer budget, PLE layer set or group size differs is refused instead of silently mis-run |
| passthrough planning | `423c85d` | `ple_constants.json` (required) and `ngram_table.bin` (optional, 102 GB) as resumable chunked range copies |

Also corrected along the way: `quantGroupSize` was 32 in the runtime
`ArchConfig` from the Vontra-era assumption; the pinned artifact is 64.

### Still needed before an install runs

1. **Resolve passthrough sizes remotely and pass them to the planner.** The
   plumbing exists; the remote repacker does not call it yet. These are
   standalone files, not index entries, so their size comes from a
   `resolveFileInfo` before planning.
2. **`SupportedModelSource` pin.** Ready to write:
   repo `RockTalk/Qwen3.8-Flash-Next-MLX-4bit`, revision
   `478474da92599ad0cf9f8bd447e658b29cb8480a`, index sha256
   `d7fe03ad2d1365e24ae2e305c829f600b80fac845d128383421a7be4adbdda1b`,
   download 175,293,161,716 bytes.
3. **MTP sidecar install.** Its 81 tensors sit in their own
   `mtp-weights.safetensors`, so it wants a second install invocation
   producing its own directory, like the Ornith MTP path. It carries its own
   512-expert set, so those experts must be SSD-streamed too, not resident.

None of that makes the model *execute* -- that is P1 (hyper-connection
residual plumbing, QSA dense path with the <=2048 exactness gate, the PLE
block, top-10 MoE) and is the larger half of the work.

## P1 status (2026-08-29)

The blockers listed here previously are cleared. What follows is what exists,
what it was verified against, and what genuinely remains.

### Cleared

| piece | commit | verified against |
| --- | --- | --- |
| top-10 routing, 512 experts | `f705873`, `836074f` | both goldens byte-identical at k=8 |
| Gated Residual primitives | `94ef43f`, `80c37b0` | CPU references per kernel |
| Gated Residual composition | `9081afd` | CPU reference of the whole formula |
| PLE row hashing | `9c7d640` | golden vectors from the reference implementation |
| n-gram table gather | `a7267f1` | synthetic table encoding its own row indices |
| QSA dense-exactness window | `ba3e5f9` | derived, and equal to the card's 2,051 |

Notes worth keeping:

- **`kMaxStreamedExperts` is 16**, one bound for every family, because Metal
  array extents must be compile-time. Kernels read only `[0, top_k)`, so the
  eight unused pointers cost 64 bytes per argument buffer and nothing else.
- **`_kn` kernel variants** exist for router select and both phase-2 reduces,
  dispatched only when k != 8 so the shipping models stay on kernels whose
  output is byte-pinned. The launch width is the contract: exactly k
  simdgroups, so the kernels carry no `sg >= k` guard -- an early return there
  would skip a `threadgroup_barrier` every thread must reach.
- **The router already implements `norm_topk_prob=true`**: it softmaxes over
  the selected top-k, which equals softmax-over-all then renormalize. Wiring
  `ArchConfig.routerNormTopK` should preserve this, not change it.
- **Grouped RMSNorm is not the per-head kernel.** That one shares a single
  weight across groups; the hyper connection needs a distinct 2560 slice per
  stream out of its [10240] weight.
- **The write gate is `2 * sigmoid`.** Plain sigmoid caps every stream at 1.0
  and makes the residual strictly contractive -- plausible, and wrong.
- **PLE's context cut latches**, and a token that is itself EOS does not cut
  its own context. One mixed value serves all 8 heads of an n-gram order.

### What remains

1. **Layer-loop integration.** The structural change and where regression risk
   concentrates: the residual becomes 4 x 2560 = 10,240 wide through the whole
   stack, which touches buffer allocation, the prefill and decode paths, and
   the KV/GDN state plumbing in `RealForwardRunner`. The primitives above are
   built to make this two calls bracketing each block, but nothing calls them
   yet. **The model cannot execute until this lands.**
2. **PLE block body.** The gather exists; the value/key projections, the
   signed-sqrt gate and the dilated depthwise conv (k=4, dilation=3, state 9)
   do not.
3. **QSA indexer**, for contexts past 2,051 keys. Below that the gate says
   dense is exact; above it the indexer is mandatory for faithfulness.
4. **MTP sidecar install**, which carries its own 512-expert set and wants its
   own directory like the Ornith MTP path.

### How to verify, when it runs

Weight fidelity is settled (`benchmark/nvmai_quant_fidelity.py`); behaviour is
not. The reference implementation is public
(`Rocktalk-Holdings/mlx-qwen4exp`), so the honest check is a greedy logit
comparison against it on a short prefix -- the analogue of the golden baseline
the shipping models use. Nothing here should be called correct on the strength
of unit tests alone.

## PLE block: verified against the reference (2026-08-29)

Read from `mlx_qwen4exp/ple.py`, not inferred. The design record above is
wrong in two places, both of which would have produced plausible output with
no error:

1. **The block adds two terms, not one.** The result is
   `hidden_wide + gated + conv_out`. The record described only the convolution
   being added; dropping `gated` would silently remove the whole gated-value
   contribution.
2. **The gate's signed square root has a floor.**
   `sigmoid(sign(s) * sqrt(max(|s|, 1e-6)))` -- the record omitted the 1e-6,
   which matters exactly where `s` is near zero and the derivative blows up.

Also corrected: the convolution's **dilation is `ngram_size` (3)**, not an
independent constant, and its history is `(K-1) * dilation = 9`.

The full block, with shapes at production geometry:

```
emb        = table[rows]                      # [T,16,160] -> flat [T,2560]
                                              #   head axis is the slow one
key        = key_proj(emb)                    # [T,10240]
value      = value_proj(emb)                  # [T,2560]
key_w      = groupedNorm(key,  norm_key,  hc=4)
query      = groupedNorm(hidden_wide, norm_query, hc=4)
s          = sum over D of key_w * query / sqrt(2560)      # [T,4] per stream
gate       = sigmoid(sign(s) * sqrt(max(|s|, 1e-6)))       # [T,4]
gated      = value broadcast over streams * gate           # [T,4,2560]
normalized = groupedNorm(gated, norm_conv, hc=4)           # [T,10240]
conv[t,c]  = sum over k of w[c,k] * xpad[t + k*3, c]       # K=4, dilation=3
                                              #   tap k reads (K-1-k)*3 back
conv_out   = silu(conv)
result     = hidden_wide + gated + conv_out
state      = last 9 rows of xpad
```

Three norm weights, all `[hc_dim]`: `norm_key`, `norm_query`, `norm_conv`.
The two projections are ordinary INT4 GEMVs and the norms are the grouped
RMSNorm already built, so the kernels still missing are the per-stream dot,
the signed-sqrt gate, the broadcast scale, and the dilated depthwise causal
convolution with carried state.

The convolution's tap indexing is the part most worth pinning in a test: tap
`k` reads `(K-1-k) * dilation` positions back, so an off-by-one in either
direction still produces smooth, plausible output.

## It answers (2026-08-30)

```
$ NVMAICLI --model qwen3.8-flash-next_125B_A6B_4Bit \
    --prompt "The capital of France is" --max-new 20 --temperature 0
 Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The capital of Spain
```

### How it was found

Five defects stood between "runs" and "answers". Not one of them threw, and
not one was visible in the output as anything but fluent nonsense --- every
single one produced a smooth, confident, wrong distribution. They were found
by numerical parity against the reference implementation, stage by stage, and
they would not have been found any other way.

The harness is in `tools/`:

- `gturbo_reader.py` reads tensors straight out of the installed
  `model_weights.bin` and `packed_experts/`, dequantizing INT4 and INT8 affine
  groups. Both sides therefore run on the *same* weights, so quantization is
  common-mode and any disagreement is in the forward pass.
- `qwen38_reference.py` is the model in numpy, one token at a time, carrying
  all four states: the KV cache, the delta-rule state, the Gated DeltaNet
  convolution tail and the PLE convolution history.
- `qwen38_parity.py`, `qwen38_full_forward.py` and
  `qwen38_sequence_parity.py` compare it against a run's activation dumps.

The runtime side is `NVMAI_ACT_DUMP=<dir>`, with `NVMAI_ACT_DUMP_POSITIONS`
for how many positions to capture. It is off unless the variable is set.

Position 0 is the workhorse: every carried state is empty there, so the whole
stack reduces to closed-form algebra and a divergence is unambiguously one
layer's math. Once position 0 was exact end-to-end, running further positions
separated "wrong block" from "wrong hand-off between tokens".

### The five defects

1. **Attention split-KV scratch was sized for 16 query heads**; this model has
   24. The only one of the five that failed loudly (a precondition trap).
   `Attention.maxQHeads`.

2. **The Gated DeltaNet output gate was silu, and this family uses sigmoid**
   (`output_gate_type` in its config). Inherited from Qwen 3.6, where silu is
   right. Both are smooth and positive-ish, so the model kept generating.
   Fixed with a function constant (`FC_GDN_SIGMOID_GATE`) selected from
   `LinearAttentionConfig.outputGate`, so the choice is declared per family
   rather than defaulted from the other one.

3. **The PLE convolution applied silu twice** --- once in the kernel, once in
   the composing Swift --- and **read its BF16 tap weights as FP16**. The
   dtype error is the instructive one: reinterpreting BF16 bytes as FP16
   yields plausible small numbers, never a NaN or a crash.

4. **The MoE computed 8 of 10 experts.** The specialized decode pipelines bake
   D, F and k in as function constants; D and F came from the model but k was
   a hardcoded 8. This family shares Qwen 3.6's hidden dimensions and routes
   to ten experts, so it took the specialized path and phase 1 simply never
   wrote the last two activation slots, which the reduce then summed as zeros.
   The residual was the two lowest-weighted experts --- about 1% of the MLP
   output, which reads exactly like quantization noise. Pinned by
   `MoEFusedFFNTests.specializedPipelineComputesEveryExpertSlot`.

5. **The vocabulary head used the attention slot's bit width.** It is 8-bit
   here while attention is 4-bit --- a combination no earlier family had --- so
   the head read a packed-INT8 tensor as INT4. This one was total: the logit
   vector had cosine -0.18 against the reference while every one of the 48
   layers beneath it was exact. `RealForwardRunner.affineHead`.

The fused greedy head was also disabled for hyper-connection families: it
folds a plain RMSNorm into the vocabulary GEMV, and this stack ends in the
gated mixer that collapses the residual streams, not in an RMSNorm.

### What parity says now

At position 0, every layer entry and the stack output match the reference to
cosine 1.00000. Across a prompt the residual drifts to about 0.99 by position
five --- FP16 activations through 48 layers, entering the carried state and
compounding --- and the argmax tracks the reference through the tokens
checked. The per-block checks (`GDN`, `QSA`, `MoE`, `PLE`) hold at 1.00000 at
every position, which is what says the drift is precision rather than a bug.

Both shipping goldens (`tools/golden-baseline.sh --check 4|8`) are
byte-identical after all of this. The families differ from this one in every
place that was touched: silu output gate, top-k of 8, and a head that shares
the attention slot's width.

### Still open

- **Throughput is 3-5 tok/s** and has had no attention at all. Nothing here
  was chosen for speed: prefill for this family is a sequential one-token
  loop, and the n-gram gather sits inline on the decode path where it could
  be issued a token ahead.
- The **QSA indexer** is still unwritten. `QSAExactness` was defined for this
  and then never called, so any context past 2,051 keys ran dense with nothing
  reporting it --- the exact failure mode the type was written to name. The
  gate is now enforced in `produceToken` and the runtime *refuses* past the
  window (`QSAIndexerRequired`) rather than attending to keys the model's
  selection would have dropped. Implementing the indexer is what lifts it.
- The **MTP sidecar** is not installed.
- Long-context behaviour is unverified: parity has been run over a handful of
  positions, not over a long prompt.
