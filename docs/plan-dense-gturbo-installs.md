# Plan: the dense Qwen 3.5 models as `.gturbo` installs

The three small Qwen 3.5 models (2B, 4B, 9B) are the only installs in this
project that are not `.gturbo`. `tools/install_models.sh qwen35-2b` runs the
converter and writes an affine safetensors snapshot straight into `models/`,
because that is what the CPU engine reads. Every other install is a `.gturbo`
directory with a manifest, a layout, and a path-bound `verified-install.json`
receipt. This plan makes the dense models consistent with the rest.

Status: **stage 1 done and verified; stage 2 blocked, and the reason is
recorded below.** The family decision is made (a new dense value, below) and
the repacker produces a correct dense `.gturbo`. The CPU engine reading it
does not yet, so the reader refuses rather than lies. The reconnaissance was
done on 2026-09-11 against `main`; every claim carries the source it came from.

## Where it actually got to (2026-09-11)

**Stage 1 -- the repacker -- is correct and checked.** `NVMAIRepack
--input-snapshot` accepts a dense snapshot; the 2B repacks into a `.gturbo`
that is byte-identical to its source. `tools/gturbo_diff_snapshot.py` proves
it rather than assuming it: it parses the resident index and the safetensors
shards directly and compared all 320 resident tensors, weight + scales +
biases, with zero mismatches. The two companion facts worth keeping:

  - `ArchInfo` accepts the converter's *flat* config, where the root is the
    text config. The MoE loader is reused for the shared DeltaNet and
    attention contract with the four MoE-only keys stubbed, so there is one
    reader for that contract rather than two that can drift.
  - `ManifestArch` gained the five gated-DeltaNet fields as optionals, and the
    wire codec now decodes the `linear_*` keys the writer had always emitted
    but the reader silently dropped. Optional, so existing installs are
    unaffected.

**Stage 2 -- the CPU reader -- is not correct, and is refused.** It loads, it
is fast, and it produces fluent nonsense. The blocker is precise:

    tensor            snapshot packed  snapshot logical  scales/row  wire shape[1]
    embed_tokens          [248320, 512]            2048         32          2048
    gate_proj               [6144, 256]            1024         32          2048

`shape[1]` is neither the packed nor the logical width, and for `gate_proj`
the scale span implies 32 groups/row (2048 logical columns at group 64) while
its weight bytes per row at 4 bits give 512. Those cannot both be true, so one
of the readings is wrong. `docs/gturbo-format.md` is referenced by
`GTurboEncoders.swift` and **does not exist**, and no dense `.gturbo` exists
in the wild, so the field is undocumented for a CPU reader. Reading
`ResidentBuffer`'s consumer is the fastest way to settle it.

Two further findings from building the reader, both worth not rediscovering:

  - The resident index stores the **logical** width in `shape[1]`, unlike a
    safetensors snapshot which stores the packed word count. Reading it the
    snapshot's way made every matrix 2-4x too wide, which showed up as a
    generation that never finished rather than as an error.
  - The scale and bias spans are **offsets on the weight's own entry**, not
    entries of their own. A snapshot names them as separate tensors.
  - Mapping `model_weights.bin` per tensor access pages the whole payload in
    repeatedly; mapping once and holding it took generation from never
    finishing to 2 seconds. `ResidentWeights` documents that invariant.

`AffineSnapshot.init(gturbo:)` therefore **throws** until the reader is
correct. The guard and the unreachable code beneath it must be removed in the
same commit that lands the fix. Nothing has been migrated: the shipped 2B, 4B
and 9B are still snapshots and still work.

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
2. **Two family cases**, decided above: `ModelFamily.qwen35Dense` mirrored
   into `manifest.json -> arch.family`, and the repacker's matching value.
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

## The family decision

**Decided: a new dense value, not a reuse of `qwen36`.**

`ModelFamily` (raw value mirrored into `manifest.json -> arch.family`) gains
`qwen35Dense = "qwen3_5_dense"`, matching the `model_type` the converter
already writes and `CPUModelFamily.qwen35Dense` already parses. `qwen36` would
have been the cheaper wiring, but it is wrong: `ModelFamily` is what the GPU
loader dispatches on, so a dense model wearing the MoE family's name would be
handed to `Model.load` and its `qwen36` schema validation — which requires
affine tensors at MoE shapes — before the CPU engine ever saw it.

Nine places ask about a family. Four must learn the new answer; the rest must
simply not mistake it for a GPU one:

| Site | Needs |
| --- | --- |
| `ModelTypes.swift` (`ModelFamily`) | the new case |
| `ArchInfo.swift` (`RepackModelFamily`) | the new case, `isDraftHead` false |
| `ReasoningControl.swift` | binary thinking, like `qwen36` |
| `Sampler.swift` (`forFamily`) | house defaults — 0.6 for Qwen 3.5 |
| `TensorSchema.swift` (`schema(for:)`) | not reached by the CPU path; must still compile, so return the dense schema or refuse explicitly |
| `ManifestReader.validateQuant` | dense has no router at a MoE width; reader must accept the embedded/router widths the converter writes |
| `Model.swift` (`validate*Schema`) | **not reached** — the CPU engine loads a dense `.gturbo` through its own path. If it were reached, the `qwen36` MoE checks would reject the model |
| `ModelSessionPlan` (display id) | a display name for the dense family |
| `AppModelInstallDescriptor` | `nil`, like the MTP heads — the app does not install these |
| `ModelCatalog.probeInstall` | **the routing change**: an install whose family is dense returns `.cpu(.qwen35Dense)`, not `.gpu(family)` |

`ModelCatalog.probeInstall` currently hard-codes `kind: .gpu(identity.family)`
(`ModelCatalog.swift:205`). That one line is what makes a `.gturbo` dense
install reach `CPUModelBackend`.

## Staged order, highest risk first

The risk here is byte-level, not architectural, so the stages are arranged to
test the bytes before anything is migrated.

1. **Repack without a reader.** Add the `ArchInfo` dense branch, the two family
   cases and the CLI path, then repack the 2B and **diff the produced
   resident bytes against the snapshot they came from**. A repack is a byte
   copy, so this is an exact comparison and it exercises the whole
   planner/writer path. If this stage is wrong, nothing downstream matters and
   nothing has been migrated.
2. **Teach the CPU engine to read it.** The `.gturbo`-backed weight source,
   then the equivalence check: same model, both paths, token-for-token greedy.
   Only now does a wrong mapping have anywhere to hide.
3. **Migrate.** `install_models.sh` converts *and* repacks; repack the 4B and
   the 9B and re-run both checks; drop the snapshot caveat from the docs.

Stage 1 is the cheap one and it is where a wrong `ArchInfo` — one field
misread, one mask entry wrong — shows up immediately as a byte difference or
a planner refusal rather than as fluent nonsense three stages later.

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
