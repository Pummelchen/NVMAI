# Plan: the dense Qwen 3.5 models as `.gturbo` installs

The three small Qwen 3.5 models (2B, 4B, 9B) are the only installs in this
project that are not `.gturbo`. `tools/install_models.sh qwen35-2b` runs the
converter and writes an affine safetensors snapshot straight into `models/`,
because that is what the CPU engine reads. Every other install is a `.gturbo`
directory with a manifest, a layout, and a path-bound `verified-install.json`
receipt. This plan makes the dense models consistent with the rest.

Status: **design only, not implemented.** The reconnaissance below was done on
2026-09-11 against `main`; every claim carries the source it came from.

## Why it was not done at the time

`NVMAIRepack --input-snapshot` refuses a dense snapshot with
`config.json invalid: no text_config`, and `ArchInfo.load` accepts only
`qwen3_5_moe`, `qwen3_5_mtp` and `qwen4_exp` (`ArchInfo.swift:162-183`). At
that point the honest reading was "the repacker cannot express this
architecture", so the installer wrote the snapshot directly and the trade —
no receipt, `--verify-install` does not apply — was documented in the README
and the forum series. That trade is real and still stands until this lands.

## What the reconnaissance established

Feasible, and no architectural blocker. Each row is checked against source:

| Question | Finding | Evidence |
| --- | --- | --- |
| Can the planner hold a dense model? | **Yes.** Routed experts are optional: the plan writes `expertsPerLayer: 0, expertStride: 0` when a source has none, and only then errors if a layer *has* routed experts and `numExperts` is zero | `RepackPlanner.swift:301-307`, `:444-447` |
| Is the quantization the same? | **Yes, byte-for-byte.** Repacker group size is 64; the dense converter is group-64 affine. A repack copies bytes; it never re-quantizes | `ArchInfo.swift:109` (`quantGroupSize: Int = 64`), `prepare_qwen35.py:146` (`GROUP_SIZE = 64`) |
| Is the int4 layout the same? | **Yes.** A source `.weight` of dtype `u32` is treated as quantized-packed and mapped to resident `u32`; the snapshot stores four 8-bit levels per word | `RepackPlanner.swift:376`, `:521-523` |
| Can the CPU engine read a `.gturbo`? | **Yes, in principle.** `ResidentIndexEntry` carries `shape`, `dtype`, `fileOffset`, `scaleOffset`, `biasOffset` — exactly the `AffineSnapshot.Matrix` contract (`weights`, `scales`, `biases`, `rows`, `columns`, `bits`, `groupSize`) | `ResidentIndex.swift:15-25`, `AffineSnapshot.swift:24-32` |
| Does the config carry what `ArchInfo` needs? | **Yes.** All of it: `hidden_size`, `intermediate_size`, `num_attention_heads`, `num_key_value_heads`, `head_dim`, `vocab_size`, `num_hidden_layers`, `tie_word_embeddings`, `layer_types`, `hidden_act`, `attn_output_gate`, `rope_parameters.{rope_theta,partial_rotary_factor}` | `models/qwen3.5_2B_4Bit/config.json` |
| Is the downstream naming right already? | **Yes for the root head.** `rename()` writes `language_model.lm_head.weight` and `language_model.model.*`, which is the MLX spelling the repacker expects; `residentDestinationName` only special-cases the Qwen3.8 MTP draft | `prepare_qwen35.py` `rename()`, `RepackPlanner.swift:623-631` |

## The work

1. **`ArchInfo` gains the dense shape.** A `qwen3_5_dense` branch that accepts a
   flat config, derives the attention mask from `layer_types` the same way the
   MoE branch does (`linear_attention` → 2, `full_attention` → 1), reads the
   rope pair, and sets `numExperts = 0`, `topKExperts = 0`,
   `moeIntermediateSize = 0`, `intermediateSize = intermediate_size`.
   The sparse-indexer and hyper-connection fields stay at their defaults; a
   dense model has none.
2. **A family case.** `RepackModelFamily` needs a dense value mirrored into
   `manifest.json -> arch.family`, and the runtime's `ModelFamily` must accept
   it — or the manifest's family must be the existing `qwen36` with the CPU
   engine chosen by directory shape instead. **This is the one genuine design
   decision and it should be made deliberately**, because `ModelFamily` is what
   the GPU loader dispatches on.
3. **A `.gturbo`-backed weight source.** A second initializer on
   `AffineSnapshot` (or a protocol behind it) that reads `manifest.json` →
   `ArchInfo`-equivalent → `Configuration`, and `ResidentIndex` → `Matrix`.
   The resident file is memory-mapped and never written, so the existing
   `@unchecked Sendable` reasoning for `Matrix` carries over unchanged.
4. **Backend format detection and the catalog probe.** `CPUModelBackend.init`
   and `ModelCatalog.probe` currently recognise a snapshot by `config.json` plus
   the absence of `manifest.json`. They need to accept both shapes, and prefer
   `.gturbo` when a directory has one.
5. **Converter/installer wiring.** `convert_qwen35` in `install_models.sh`
   becomes a convert *and* repack, so the receipt path is the same one every
   other model uses.

## The risk, stated plainly

The failure mode here is **silent wrongness**, and it is the reason this plan
exists rather than a patch. If the resident index is mapped subtly wrong — a
name convention, the sign or stride of the scale/bias arrays, the per-row
dequantize layout — the model will **load, run, and produce fluent nonsense**
with no error raised. This project has already shipped that shape twice: the
AgentWorld norm fold (`1cc393a`) and the Qwen 3.5 9B untied head (`195eb3c`).

## The verification that must gate it

A repack is a byte copy, so a correct implementation is exactly equivalent to
the snapshot it came from. That makes the gate cheap to state and hard to fake:

> Load the same model twice — once through the existing snapshot path, once
> through the new `.gturbo` path — and require **token-for-token identical
> greedy output** on a fixed prompt at temperature 0, on the same machine and
> build.

Run it on the 2B first (1.3 GB, fast to repack and load), then the 4B, then the
9B. The 9B carries the extra value: it is the untied-head case, so it also
re-proves the `lm_head` mapping survives a repack.

Until that comparison passes, the snapshot path stays the shipped one and no
install is migrated. A `.gturbo` CPU model that has not been diffed against its
own snapshot is not a verified install; it is a second opinion.

## What "done" looks like

- `tools/install_models.sh qwen35-2b|qwen35-4b|qwen35-9b` produces
  `models/qwen3.5_*Bit/` with `manifest.json`, `packed_experts/` (empty or
  absent), and `verified-install.json`.
- `NVMAIRepack --verify-install --input-gturbo <dir>` passes on all three.
- The catalog lists them, the CPU engine serves them, and the equivalence check
  above holds for each.
- `README.md`, `docs/site/04-choosing-a-model.md` and the wiki drop the
  "snapshots, not `.gturbo` — no receipt" caveat, because it stops being true.
