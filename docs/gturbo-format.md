# The `.gturbo` install format

`GTurboEncoders.swift` has pointed at this file since the resident index was
first written. It did not exist. This is it.

A `.gturbo` install is a directory. It is the only install shape this project
ships: the runtime, the installer, `--verify-install`, and the Mac app all
assume it. The dense Qwen 3.5 2B/4B/9B installs used to be the exception — an
affine safetensors snapshot with no manifest and no receipt — and they are
`.gturbo` now too (`tools/repack_dense.sh`).

```
qwen3.5_2B_4Bit/
  manifest.json            the architecture, the widths, and a hash of every file
  model_weights.bin        the resident tensors: one index, then one payload
  packed_experts/
    layout.json            the per-layer expert blob layout
    layer_00.bin ...       one file per layer that has routed experts
  tokenizer/
    tokenizer.json
    tokenizer_config.json
    chat_template.jinja
  verified-install.json    the path-bound receipt
```

Everything is little-endian. Every field is read and written in one place,
`GTurboBinary` (`sources/NVMAIRepack/Core/Format/GTurboEncoders.swift`), so the
on-disk layout changes only there.

## Why the format exists

The models are 19-200 GB and the machines are 24-128 GB. A `.gturbo` install is
built so that **the runtime never has to hold the whole model**, and so that a
model can be streamed, laid out, and verified without a second copy anywhere on
disk:

- the resident weights live in one file with a binary index, so a tensor is one
  `mmap` and one offset, not a safetensors JSON parse;
- the routed experts are split into per-layer files so a decode step touches
  the experts it routed to and nothing else;
- every file is hashed into the manifest and into the receipt, so an install
  that is truncated, swapped, or edited is refused rather than run.

## `manifest.json`

Written by `GTurboJSON.encodeManifest`. The fields that matter:

| Field | Meaning |
| --- | --- |
| `magic`, `versionMajor`, `versionMinor` | `"GTURBO"`, 1, 0 |
| `modelID` | the model's id, e.g. `"qwen3.5-2b"`. **Not** the API id: the catalog appends the width, giving `qwen3.5-2b_4-Bit` |
| `sourceSnapshotHash` | the source checkpoint's index hash; how an install is traced to what it was built from |
| `arch` | the architecture the planner read. `arch.family` mirrors `ModelFamily` and is what loaders dispatch on |
| `quant` | the five width *slots*, plus one entry per quantified resident tensor (below) |
| `files` | every payload file with its size and sha256 |
| `expertsPerLayer`, `numLayers`, `expertStride` | cross-checked against `packed_experts/layout.json` |
| `bitWidthOverridesHonored` | how many per-tensor overrides the source checkpoint declared; an audit number, not an input |
| `flags` | `streamingPresent`, `turboQuantKV`, `aneSharedExpert` |

### `quant`: slots **and** per-tensor widths

This is the part that is easy to get wrong, and getting it wrong is silent.

The five slots are `embedding`, `attention`, `router`, `sharedExpert`, and
`routedExpert`, each `{weightBits, scheme, scaleType, biasType, groupSize}`.
They are what the **installer derived** from the tensor data, not what the
source checkpoint declared. A "4-bit" build is 4-bit for the routed experts —
which are almost the whole payload — and can be 8-bit for other tensors.

Concretely, the Qwen 3.5 2B 4-bit checkpoint keeps its embedding and the K/V
projections of its six full-attention layers at 8 bits, which is 13 tensors.
The slots say `attention: 4`. **Both** statements are true, and only the
per-tensor entries say which tensor is which:

```json
"quant": {
  "attention":  { "weightBits": 4, ... },
  "embedding":  { "weightBits": 8, ... },
  "language_model.model.layers.11.self_attn.k_proj": { "weightBits": 8, ... },
  "language_model.model.layers.11.self_attn.v_proj": { "weightBits": 8, ... }
}
```

`weightBits` is the only field a reader consumes per tensor; the rest are
repeated for shape and for a human reading the file. The key is the **stem** —
the resident index name with any `.weight` suffix removed. A reader that trusts
the slots alone unpacks those 8-bit tensors as 4-bit. The word count changes,
the strides still divide evenly, every shape check passes, and the model answers
fluently and wrongly.

Unquantized tensors (norms, scalars) have no entry: they carry no scales, they
are read as BF16 by `dtype`, and they are never dequantized. A stem that
collides with one of the five slot names would overwrite a slot, so the writer
skips it.

`routedExpert` is the slot the rest of the system treats as "the model's width":
`ManifestIdentity.weightBits` reads it, and `apiModelID` appends it as
`_<bits>-Bit`, so it is what `/v1/models` shows and what a request names. For a
model that has no routed experts — every dense install — it is set from the
source's base affine width. Left at a default, an 8-bit dense install advertises
itself as 4-bit and the catalog skips it as a duplicate of the real 4-bit one,
which is exactly what happened.

`GTurboManifestQuantV1` decodes this object by hand. A synthesised `Codable`
would drop the open set of per-tensor keys, because the fixed slots are a
`CodingKeys` enum and the overrides are not — which is exactly the bug that was
shipped and then found by comparing `.gturbo` logits against the snapshot's.

## `model_weights.bin`

One file, two regions. `indexSize` bytes of index, then `residentSize` bytes of
tensor payload. Both are page-aligned to 16 KB, which is the format's
alignment, not the kernel's.

```
offset 0                                    offset indexSize
+---------------------------+---------------+------------------------------+
|  index                    |  padding      |  resident tensor payload     |
|  header + entries + names |  to 16 KB     |  weights, scales, biases     |
+---------------------------+---------------+------------------------------+
```

`fileOffset`, `scaleOffset` and `biasOffset` in an entry are absolute file
offsets, so they already include `indexSize`. The payload is written in plan
order and packed back to back.

### The index header, 24 bytes

| Offset | Type | Field |
| --- | --- | --- |
| 0 | u64 | `indexSize` |
| 8 | u64 | `residentSize` |
| 16 | u64 | `entryCount` |

`indexSize` includes the header, the entry table, the string table, **and** the
padding to the 16 KB boundary. It is validated as a multiple of 16 KB.

### An index entry, 72 bytes

| Offset | Type | Field |
| --- | --- | --- |
| 0 | u32 | `nameOffset` — **absolute file offset** of the name's UTF-8 bytes |
| 4 | u16 | `nameLen` |
| 6 | u8 | `dtype` |
| 7 | u8 | reserved, 0 |
| 8 | u64 | `fileOffset` — packed weight bytes |
| 16 | u64 | `sizeBytes` |
| 24 | u32 ×4 | `shape[4]`, row-major, padded with zeros |
| 40 | u64 | `scaleOffset` |
| 48 | u64 | `scaleSize` |
| 56 | u64 | `biasOffset` |
| 64 | u64 | `biasSize` |

The string table follows the entry table; a name's offset is **not** relative to
the table. The writer computes `stringTableBase + stringTableOffsets[i]`
(`ResidentWriter.encodeIndex`), where `stringTableBase` is
`24 + entryCount * 72`.

`dtype`:

| Value | Type |
| --- | --- |
| 0 | u32 |
| 1 | bf16 |
| 2 | fp16 |
| 3 | fp32 |

A quantized weight is `dtype 0`: `sizeBytes` covers the packed words and the
group-64 scales and biases are **not entries of their own**. They are reached
through the weight's own `scaleOffset`/`scaleSize`/`biasOffset`/`biasSize`
fields, and they are BF16. This is the sharpest difference from a safetensors
snapshot, which names `foo.scales` and `foo.biases` as separate tensors. Reading
a `.gturbo` index by looking up `"foo.scales"` finds nothing.

### `shape[1]` is the logical width

For a quantized matrix, `shape[1]` in the index is the **logical (unpacked)**
width, not the packed word count that a safetensors header stores for a u32
tensor. A group-64 4-bit tensor with `shape[1] = 2048` holds 2048 logical
columns, `2048 / 64 = 32` scales per row, and `2048 / 8 = 256` packed words per
row. Reading it the snapshot's way makes every matrix two to four times too
wide, and because that only changes how much is read — never whether the read
succeeds — it shows up as a generation that never finishes rather than as an
error.

The scales/biases span is the consistency check that pins it down:
`scaleSize` must be `rows * (columns / groupSize) * 2`, and `biasSize` equal to
it.

### Mapping it

`model_weights.bin` is memory-mapped **once**, for the life of the weights
object, and every tensor is a pointer into that mapping. Mapping per tensor
access re-pages the whole payload on every dequantize; on the 2B that is the
difference between generation that never finishes and generation in about two
seconds. `ResidentWeights` carries the invariant.

## `packed_experts/`

Written by `GTurboJSON.encodeLayout` and the per-layer writers. `layout.json`
has `expertStride`, `numLayers`, `expertsPerLayer`, and a `layers` array; each
layer has `layer`, `file`, and an `experts` array; each expert has `offset`,
`size`, and a `tensors` object whose keys are `gate`/`up`/`down` with
`_scales`/`_biases` suffixes.

`layer_NN.bin` is a flat array of fixed-stride expert blobs:
`expertsPerLayer * expertStride` bytes, with `expertStride` a multiple of
16 KB. Every expert in a layer has the same stride, so expert `e` starts at
`e * expertStride` and the offsets inside an expert blob are what `layout.json`
records. A decode step reads the nine sub-tensor slices of the experts it
routed to and touches no other file.

### A layer with no routed experts

`expertsPerLayer` is 0 for a model with no MoE layers, and the dense Qwen 3.5
installs are exactly that. `layout.json` still lists a layer per layer, with an
empty `experts` array and a nominal `file` name, because the layer count must
match `manifest.numLayers`. **No `layer_NN.bin` is written**, because its size
would be `0 * expertStride = 0` — 24 empty files to satisfy a check would be
worse than the check. `VerifiedInstallTool.validatePackedExpertLayout` skips a
layer whose expected size is zero and still requires the file when it is not.
`tools/gturbo_diff_snapshot.py` reports this shape directly: 320 resident
tensors, no packed experts.

## `tokenizer/`

`tokenizer.json` is required, and so is `chat_template.jinja` when the model has
a chat template: the OpenAI-compatible server renders ChatML through the
template, and a tokenizer without one cannot serve a chat request. The
`tokenizer/` subdirectory is the canonical location and the only one
`GFTokenizer.tokenizerFolder` looks in. The CPU backend also accepts the files
at the install root and tries that first, because the older dense snapshots were
laid out that way; new installs get the sidecar.

The template, not the family, decides the prompt format, so an install carries
its own. A model with `reasoning` in its template gets thinking mode; effort
levels are resolved per request against the template that is loaded, because the
effort string is baked into the rendered prompt at load time.

## `verified-install.json`

The receipt. It holds `schemaVersion`, `toolVersion`, `verificationTimestamp`,
the source `sourceRevision`, `modelDirectoryPath`, the manifest's own hash, and
the sha256 and size of **every file the install declared** — including
`manifest.json` itself.

`modelDirectoryPath` is what makes the receipt **bound to the absolute path the
install was written to**. Moving or renaming the directory makes the install
fail to load with `trusted receipt invalid: model directory mismatch`. This is
the receipt doing its job: it is what detects a moved or swapped directory. It
is not corruption and does not need a re-download — re-issue it in place, which
re-hashes the payload against the manifest and rebinds it to the current path:

```bash
swift run -c release NVMAIRepack --verify-install --input-gturbo models/qwen3.5_2B_4Bit
```

Never hand-edit the receipt to match a new path. The binding is the check;
editing it forges the attestation instead of re-establishing it.

Because the receipt is path-bound, an install must be written to where it will
live. Repacking into a temporary directory and moving it into place produces an
install whose receipt is invalid — `tools/repack_dense.sh` repacks straight into
`models/` for this reason.

## Building one

```bash
# From a Hugging Face source, streamed, no full local checkpoint:
swift run -c release NVMAIRepack --output models/qwen3.6_35B_A3B_4Bit

# From a completed local affine snapshot (how the dense models are built):
swift run -c release NVMAIRepack --input-snapshot .build/qwen35-2b-affine-4bit \
    --model-id qwen3.5-2b --output models/qwen3.5_2B_4Bit
```

A repack is a **byte copy**, not a re-quantization: the repacker and the dense
converter both use affine group-64, so `RepackPlanner` copies a source `u32
.weight` into the resident payload unchanged. That is what makes the
verification exact:

- `tools/gturbo_diff_snapshot.py <install> <snapshot>` compares every resident
  tensor, weight plus scales plus biases, byte for byte;
- `tests/NVMAI/Core/CPUEngine/DenseGTurboEquivalenceTests.swift` loads both and
  requires **identical logits**.

The second is the one that matters. The first proves the bytes are right, which
a wrong per-tensor width would still pass — the bytes are right, they are just
read at the wrong width. `tools/repack_dense.sh` runs both, in that order, then
`--verify-install`.
