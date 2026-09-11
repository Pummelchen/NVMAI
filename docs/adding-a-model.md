# Adding a model to the family

This is the checklist for turning a Hugging Face checkpoint into a first-class
NVMAI member at 4-bit and 8-bit. It exists because the work is spread over
eight places in the tree and the order matters: the research decides whether it
is a wiring job or a runtime job, and nothing is called *supported* until a real
model has been run.

`KAT-Coder-V2.5-Dev` is the worked example throughout — a Qwen3.6-35B-A3B
fine-tune, which is the cheap case. A checkpoint whose architecture is not
already implemented is a different project; see the tracker's "planned work".

## 0. Research the checkpoint before touching the tree

Read it from the source, not from the model card's prose:

```bash
# Identity, gating, parameter count, per-file sizes, the pinned sha.
curl -sL 'https://huggingface.co/api/models/<org>/<name>?blobs=true' | python3 -m json.tool | head -40

# Geometry: hidden size, layers, experts, rope block, tie_word_embeddings.
curl -sL 'https://huggingface.co/<org>/<name>/raw/<sha>/config.json'

# Sampling the authors actually asked for (not the base model's).
curl -sL 'https://huggingface.co/<org>/<name>/raw/<sha>/generation_config.json'

# Tensor names, and how many there are.
curl -sL 'https://huggingface.co/<org>/<name>/resolve/<sha>/model.safetensors.index.json' \
  | python3 -c 'import json,sys; wm=json.load(sys.stdin)["weight_map"]; print(len(wm))'
```

Record, because each one has burned a release or a conversion before:

- **`gated`** — if true, stop and ask the human for a token; do not paste one
  into a script.
- **The pinned sha**, not `main`: the install receipt records the source, and a
  moved `main` must not silently change what "4-bit" means.
- **`tie_word_embeddings`** — decides whether the head exists. A vision-language
  wrapper often unties it.
- **The rope block.** A multimodal wrapper can carry `mrope_interleaved` /
  `mrope_section` even in text-only use. Compare it against a model the runtime
  already serves rather than reasoning about it: if it matches, the text path
  needs no new kernel.
- **The EOS ids.** `generation_config.json` may list two. Check they resolve as
  stop tokens by *string* (`<|endoftext|>`, `<|im_end|>`) rather than by number,
  because the numbers move between releases.
- **Recommended sampling**, including `presence_penalty` and any extra template
  kwargs (`preserve_thinking`). Note which of them this runtime cannot do — it
  supports presence penalty `0.0` only — and say so in the docs rather than
  quietly dropping it.

## 1. Decide: wiring job or runtime job

Compare the config's geometry and the index's tensor names against a family the
runtime already serves. For a converter built on `prepare_agentworld.py`, the
only namespaces it knows are `model.language_model.*`, `lm_head.weight` and
`mtp.*`; anything else is a new case. Verify rather than trust:

```bash
python3 - <<'PY'
import json
wm = json.load(open("/tmp/index.json"))["weight_map"]
known = ("model.language_model.", "model.visual.", "mtp.")
outside = [k for k in wm if k != "lm_head.weight" and not k.startswith(known)]
print(len(wm), "tensors;", len(outside), "outside the known namespaces")
print("visual:", sum(1 for k in wm if k.startswith("model.visual.")),
      "mtp:", sum(1 for k in wm if k.startswith("mtp.")),
      "head:", "lm_head.weight" in wm)
PY
```

A `vision_config` in `config.json` does **not** mean the checkpoint carries a
vision tower: KAT declares one and ships language weights only. Check the index.

## 2. The eight wiring points

| # | Where | What |
| --- | --- | --- |
| 1 | `tools/prepare_agentworld.py` — `MODELS` | The repo and the **pinned sha**, with a comment recording what was verified about the checkpoint |
| 2 | `tools/install_models.sh` — `CATALOGUE` | Two rows (`<key>`, `<key>-8bit`) and the preset→served-id `case` |
| 3 | `tools/nvmai_models.sh` | The key/stem/label `case`, the unknown-model help text, `NVMAI_ALL_MODELS` |
| 4 | `tools/server_launcher.sh` | The model-key list in the header comment and in the unknown-model error |
| 5 | `ModelProfile.swift` | One row per width. Sample from the **checkpoint's** config; say in the comment when cache/prefetch values are inherited from identical geometry rather than measured |
| 6 | `ModelCatalog.swift` — `displayNames` | The served id → human name (`/v1/models` and the app read this) |
| 7 | `AppModelInstallDescriptor.swift` | Two `converted(...)` descriptors, `all`, `installerTarget`, `selectedDescriptor` — **see §5 for the fingerprint** |
| 8 | `tests/` | `ModelProfileTests.shipped` (and the table count) and the app test's build table |

Validate 1 before any download:

```bash
python3.13 tools/prepare_agentworld.py --model <key> --plan
```

It fetches the config and the index only, and prints the shard count, the
tensor split per width, the bf16 keeps and the output size. `--plan` writing no
files is the point: it is the cheap place to discover that a last dimension is
not group-aligned.

## 3. Convert and install

```bash
tools/install_models.sh <key>            # 4-bit
tools/install_models.sh <key>-8bit       # 8-bit
```

The installer streams one source shard at a time (about 11 GB in flight for a
35B-A3B) and deletes each after use; it does **not** stage the full checkpoint.
Both widths come from one download when the converter is called with `--bits 4
8`, but the **snapshot and the install exist at the same time**, so budget:

| Build | Shards | Snapshot | Install | Peak |
| --- | ---: | ---: | ---: | ---: |
| 35B-A3B 4-bit | ~11 GB | ~20 GB | ~20 GB | ~50 GB |
| 35B-A3B 8-bit | ~11 GB | ~38 GB | ~35 GB | ~84 GB |

Delete the snapshot after the install (`rm -rf .build/<key>-affine-*`) and do
one width at a time on a full disk: 8-bit needs the space the 4-bit snapshot and
install are holding. Check `df -h .` against the table before starting, and say
so rather than starting a conversion that will die at 90%.

## 4. Verify before calling it supported

The project's bar, in this order:

1. **It loads and answers** through the CLI, and the continuations this project
   uses behave: `Once upon a` → " time", `The capital of France is` → " Paris",
   `The quick brown fox jumps over the lazy` → " dog".
2. **A golden baseline**, added to `tools/golden-baseline.sh` (target table) and
   to `release.sh`'s `check_golden` list, captured only for a deliberate
   numerics change — never re-captured to make a mismatch go away. Until it
   exists, `release.sh` will refuse the machine ("an installed model has no
   golden target"), which is the guard working.
3. **The receipt**: `NVMAIRepack --verify-install --input-gturbo <dir>` passes,
   and the manifest's `sourceSnapshotHash` matches the snapshot that produced
   it.
4. **The catalog is right**: `/v1/models` lists the id with the name from
   `displayNames` and the sampling from the profile row, and the launcher
   resolves the key, the stem and both widths.
5. **A first measured row** (TTFT, decode) from the model's own install, with
   the machine and commit stated.

## 5. The app's fingerprint (the piece that needs the conversion)

`AppModelInstallDescriptor.sourceIndexSHA256` is the **converted snapshot's**
`model.safetensors.index.json` hash — the same value NVMAIRepack records in the
install manifest as `sourceSnapshotHash`. It is *not* the source repository's
index, and it cannot be known before the conversion runs. Read it back:

```bash
python3 -c "import json;print(json.load(open('models/<dir>/manifest.json'))['sourceSnapshotHash'])"
```

Then add the two descriptors, the entries in `all`, the `installerTarget` cases
and the `selectedDescriptor` selectors, and extend the app test's table. The
test asserts the fingerprints are unique across `all`, so two widths of one
checkpoint must not share a value — which is exactly why this one is per-width.

## 6. Document it where a user will look

- `README.md` — the supported list, with the status stated honestly.
- The wiki: `Getting-Started` (the install table), `Features` (the model row),
  `Runtime-Controls` (thinking levels, and the temperature line if this
  checkpoint differs from its base's), `Project-Tracker` (what was checked,
  what landed, what is pending, and any deviation from the model's own
  recommendations), `Roadmap` (status on the entry that asked for it).
- Deviations are stated, not hidden: presence penalty, extra template kwargs,
  an unverified tool-call dialect, an inherited profile value.

Never mark a model "supported" before §4 passes. "Install path landed,
verification pending" is the honest state, and it is worth writing down.

## 7. After the checkout moves

An install receipt is bound to its absolute path. Moving or renaming the
checkout invalidates **every** install's receipt, with
`trusted receipt invalid: model directory mismatch` — not corruption, and no
re-download. Re-issue each in place:

```bash
for d in models/*/; do
  [ -f "$d/verified-install.json" ] || continue
  swift run -c release NVMAIRepack --verify-install --input-gturbo "$d"
done
```

Never hand-edit `verified-install.json`: the path binding is what detects a
moved or swapped directory.

## Checklist

- [ ] Gating, pinned sha, geometry, rope block, tie-embeddings and EOS ids read
      from the source
- [ ] The index's tensor names checked against the converter's namespaces
- [ ] `--plan` classifies every tensor and the output size fits the disk table
- [ ] All eight wiring points above
- [ ] Conversion and install, one width at a time, snapshots deleted after
- [ ] Continuations, golden target, receipt, catalog, launcher keys
- [ ] App descriptors with the fingerprints read back from the manifests
- [ ] README, wiki pages, tracker, roadmap — with deviations and status
- [ ] Committed, pushed, and the receipts re-issued if the checkout moved
