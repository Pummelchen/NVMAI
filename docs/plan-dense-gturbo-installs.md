# Plan: the dense Qwen 3.5 models as `.gturbo` installs

The three small Qwen 3.5 models (2B, 4B, 9B) are the only installs in this
project that are not `.gturbo`. `tools/install_models.sh qwen35-2b` runs the
converter and writes an affine safetensors snapshot straight into `models/`,
because that is what the CPU engine reads. Every other install is a `.gturbo`
directory with a manifest, a layout, and a path-bound `verified-install.json`
receipt. This plan makes the dense models consistent with the rest.

Status: **done and verified on all three models.** The family decision is made
(a new dense value, below), the repacker produces a correct dense `.gturbo`, and
the CPU engine reads it. The blocker in the previous revision of this document
was not where it was recorded: the reader was right and the *writer* was
dropping data. That is corrected and explained below. Every claim carries the
source it came from.

## Where it actually got to (2026-09-11)

**Stages 1 and 2 are done, and the migration is done.** `NVMAIRepack
--input-snapshot` accepts a dense snapshot; all six dense installs (2B/4B/9B at
4-bit and 8-bit) repack into a `.gturbo` that is byte-identical to its source and
logit-identical through the CPU engine. `tools/repack_dense.sh` runs the whole
thing — stage, repack, byte-diff, re-issue the receipt, run the equivalence
gate — and is the reproducible way to redo it.

The verifications, in the order they ran:

    install                    residents  byte-diff   receipt  logits
    qwen3.5_2B_4Bit                  320   320/320      PASS    0 (exact)
    qwen3.5_4B_4Bit                  426   426/426      PASS    0 (exact)
    qwen3.5_9B_4Bit                  427   427/427      PASS    0 (exact)
    qwen3.5_2B_8Bit                  320   320/320      PASS    0 (exact)
    qwen3.5_4B_8Bit                  426   426/426      PASS    0 (exact)
    qwen3.5_9B_8Bit                  427   427/427      PASS    0 (exact)

The 9B carries the extra value: it is the untied-head case, so it also proves
the `lm_head` mapping survives a repack, at both widths.

Served, not just loaded: the catalog lists all six with distinct ids and the
`cpu` backend, and the server answers through the catalog path on both a 4-bit
and an 8-bit install, including unloading one to load the other.

    qwen3.5-2b_4-Bit  Qwen 3.5 2B  quant=4 backend=cpu
    qwen3.5-2b_8-Bit  Qwen 3.5 2B  quant=8 backend=cpu
    qwen3.5-4b_4-Bit  Qwen 3.5 4B  quant=4 backend=cpu
    qwen3.5-4b_8-Bit  Qwen 3.5 4B  quant=8 backend=cpu
    qwen3.5-9b_4-Bit  Qwen 3.5 9B  quant=4 backend=cpu
    qwen3.5-9b_8-Bit  Qwen 3.5 9B  quant=8 backend=cpu

**The blocker was the writer, not the reader.** The previous revision recorded
the failure as "the CPU reader produces fluent nonsense, so refuse it". The
reader was correct. The repacker was writing only the five **slots** into
`manifest.json -> quant` and dropping the source checkpoint's per-tensor widths,
because `GTurboJSON.encodeManifest` built `quantDict` from
`bitWidths.{embedding,attention,router,sharedExpert,routedExpert}` and nothing
else.

That matters because a 4-bit build is not uniformly 4-bit. The 2B keeps its
embedding and the K/V projections of its six full-attention layers at 8 bits —
13 tensors, declared in the source snapshot's own `quantization` block. The
slots say `attention: 4`, which is true of the other attention tensors. A reader
that trusts the slots unpacks those 8-bit tensors as 4-bit: the word count
changes, the strides still divide evenly, every shape check passes, and the
model answers fluently and wrongly.

The fix is small and the shape of it is the point: emit every quantified
resident tensor's real width beside the slots, so the manifest says what was
actually packed and no reader has to re-derive it. The contract already existed
— `GTurboManifestQuantV1` had hand-written `init(from:)`/`encode(to:)` to
preserve the open key set, and `ManifestReader.quantOverrides` already read it —
so only the writer was missing. The bug that made the overrides necessary in the
first place was a synthesised `Codable` silently dropping them; the same class
of bug, one layer up.

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

A third finding came out of the migration: `verify-install` required a
`packed_experts/layer_NN.bin` for every layer, and a dense install has none —
`expertsPerLayer` is 0, so each layer's expected size is 0 and its file is
never written. Writing 24 empty files to satisfy the check would have been
worse than the check, so the validator skips a layer whose expected size is
zero. It still requires the file when the size is non-zero, so no MoE install
lost a check.

A fourth came out of the 8-bit half, and it is the same disease one layer up:
`routedExpert` was initialised to a literal `4` in `writeManifest` and only
overwritten from the layer plan's sub-tensors. A dense model has no routed
experts, so the slot kept the literal and **every** dense install claimed to be
4-bit. That value is what `ManifestIdentity.weightBits` reads and what
`apiModelID` turns into the `_<bits>-Bit` suffix, so all three 8-bit installs
came back as `qwen3.5-2b_4-Bit` and the catalog skipped them as duplicates of
the 4-bit ones — an 8-bit install that installs, verifies, loads, and cannot be
selected. Initialising the slot from the source's base affine width fixes it,
and the 4-bit manifests are byte-unchanged, which is how the fix was confirmed
to be scoped. It was caught by reading `NVMAIServer --catalog` output rather
than by a test; nothing in the suite compares a manifest's slots to its
contents.

`docs/gturbo-format.md` now exists, and documents all of the above.

A fifth is a migration concern rather than a format one. `install_one` treats
"the directory exists" as "installed", so a user who installed the 2B before
this would be told it is installed and left without a manifest and receipt
forever — the inconsistency this work removed, kept alive for everyone who
installed first. A dense directory with no manifest is now recognised as a
legacy snapshot and moved into the converter's staging area instead of being
downloaded and quantized again, because it is already the same bytes. Verified
by planting a snapshot at `models/qwen3.5_2B_8Bit` with the staging directory
removed: the installer staged it, repacked, passed `--verify-install`, and
produced a `.gturbo` byte-identical across all 320 resident tensors; a second
run reports it as installed and touches nothing.

## What "done" looks like, checked

- `tools/install_models.sh qwen35-2b|qwen35-4b|qwen35-9b` (and `-8bit`)
  produces `models/qwen3.5_*Bit/` with `manifest.json`, `packed_experts/`, and
  `verified-install.json`. **Done.** The installer converts and repacks, drops
  the intermediate snapshot, and migrates a legacy snapshot in place.
- `NVMAIRepack --verify-install --input-gturbo <dir>` passes on all six.
  **Done.**
- The catalog lists them, the CPU engine serves them, and the equivalence check
  holds for each. **Done** — six distinct ids on the `cpu` backend, and the
  server answers through the catalog path on a 4-bit and an 8-bit install,
  including unloading one to load the other.
- `README.md`, `docs/site/04-choosing-a-model.md` and the wiki drop the
  "snapshots, not `.gturbo` — no receipt" caveat. **Done.**
- Not done, and worth knowing: the equivalence gate is not part of
  `swift test`. It loads real models, which the unit tests deliberately never
  do, so it is opt-in and read by a human. A regression in the resident-index
  mapping would be caught by `tools/repack_dense.sh` and by nothing automatic.

## Why it was not done at the time

Historical, and kept because the shape of the mistake is the useful part.

`NVMAIRepack --input-snapshot` refused a dense snapshot with
`config.json invalid: no text_config`, and `ArchInfo.load` accepted only
`qwen3_5_moe`, `qwen3_5_mtp` and `qwen4_exp` (`ArchInfo.swift:162-183`). At
that point the honest reading was "the repacker cannot express this
architecture", so the installer wrote the snapshot directly and the trade —
no receipt, `--verify-install` does not apply — was documented in the README
and the forum series.

That reading was right about `ArchInfo` and wrong about where the work was. The
dense shape was the easy half; the hard half was an undocumented contract, and
the bug hiding behind it was a writer dropping a field, not a reader misreading
one.

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
