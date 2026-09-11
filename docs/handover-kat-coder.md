# Handover: finish adding KAT-Coder-V2.5-Dev

**Paste this into the next session:**

> Continue the NVMAI work in this checkout. Read `AGENTS.md`, then
> `docs/handover-kat-coder.md`, then the wiki's `Project-Tracker` entry
> "KAT-Coder-V2.5-Dev: prep landed, conversion and install pending". The
> checkout has been moved, so start by re-issuing every install receipt. Then
> convert and install KAT-Coder-V2.5-Dev at 4-bit, verify it to the project's
> standard, and take the 8-bit decision (it needs roughly 35 GB freed). Follow
> `docs/adding-a-model.md` for the verification bar and the app-fingerprint
> recipe. Report measurements, not assurances.

## Where the work stands at handover

Pushed: repo `main` at `065acc5`, wiki `master` at `3702725` (plus the figures
and mock-launcher path fixes that follow it). Nothing was left running: no
server, no conversion, no release job.

**Done — the whole wiring, verified:**

- `tools/prepare_agentworld.py` gained a pinned `katcoder` entry
  (`7be56fe773e72b6f5ca93c1ae45d828ddb893922`), validated with `--plan`:
  13 shards, 31,333 tensors, none outside the converter's namespaces, no
  `model.visual.*`, no `mtp.*`, `lm_head` present.
- **Output sizes measured**: 20.0 GB at 4-bit, 36.9 GB at 8-bit.
- `tools/install_models.sh` catalogue (`katcoder`, `katcoder-8bit`) and the
  preset→id mapping; `tools/nvmai_models.sh` key/stem/label, help and fallback
  list; `tools/server_launcher.sh` help and error text.
- `ModelProfile.swift` rows for both widths, carrying **KAT's own temperature
  1.0** (its `generation_config.json`), not Qwen 3.6's 0.6.
- `ModelCatalog.swift` display name; `ModelProfileTests` extended (table is now
  10 rows).
- **Golden targets declared for both widths** in `tools/golden-baseline.sh` and
  `release.sh` — inert without an install, fail-closed with one.
- Docs: README, wiki `Getting-Started`/`Features`/`Runtime-Controls`,
  `Project-Tracker`, `Roadmap`, and the new `docs/adding-a-model.md`.
- Gates at that state: lint clean, **1462 tests in 227 suites**, and the
  checkout contains no absolute paths outside the path-bound receipts.

**Pending:** the conversion, the two installs, the app descriptors, the
baselines, and real-model verification.

## Step 1 — after the move: re-issue every receipt

An install receipt is bound to its absolute path, so the move invalidates all
of them (`trusted receipt invalid: model directory mismatch` — not corruption,
no re-download). Re-issue in place:

```bash
cd <new checkout>
for d in models/*/; do
  [ -f "$d/verified-install.json" ] || continue
  swift run -c release NVMAIRepack --verify-install --input-gturbo "$d"
done
```

`.build/` is disposable: if the build misbehaves after the move,
`rm -rf .build` and rebuild. The wiki clone lives at `.qwen/wiki` and moves
with the checkout.

## Step 2 — 4-bit conversion and install

```bash
swift build -c release
tools/install_models.sh katcoder          # ~20 GB snapshot + ~20 GB install
```

Peak disk ~51 GB (two source shards ~11 GB, snapshot 20.0 GB, install ~20 GB).
Then delete the snapshot (`rm -rf .build/katcoder-affine-*`) and verify:

```bash
.build/release/NVMAICLI --model models/kat-coder-v2.5_35B_A3B_4Bit \
  --prompt "The capital of France is" --max-new 5 --temperature 0
tools/golden-baseline.sh katcoder-4        # capture (no --check)
tools/golden-baseline.sh --check katcoder-4
.build/release/NVMAIRepack --verify-install --input-gturbo models/kat-coder-v2.5_35B_A3B_4Bit
```

Apply the model-run preconditions first (macOS 26+, Swift 6.3+, disk,
`memory_pressure -Q`, and no process from the guard in `AGENTS.md`). Never
terminate a process this session did not start.

## Step 3 — the 8-bit decision

Peak ~83 GB (11 + 36.9 + ~35) against **69 GB free**: short by roughly 35 GB
after the 4-bit install. Free space (the 125B installs are 125-162 GB and
rebuild from the installer), or stop at 4-bit and say so. Do not delete an
existing install without asking.

## Step 4 — the app descriptors

`AppModelInstallDescriptor.sourceIndexSHA256` is the **converted snapshot's**
index hash — the install manifest's `sourceSnapshotHash`:

```bash
python3 -c "import json;print(json.load(open('models/kat-coder-v2.5_35B_A3B_4Bit/manifest.json'))['sourceSnapshotHash'])"
```

Add the two `converted(...)` entries, the `all` entries, the `installerTarget`
cases and the `selectedDescriptor` selectors, and extend
`AppModelInstallTests`' build table. The test asserts fingerprints are unique
across `all`, so the two widths must not share a value.

## Step 5 — status and documentation

Only after the verification passes: flip the README and wiki wording from
"verification pending" to supported, record the measured row, and note the two
deviations that are already documented and unchanged — the model card's
`presence_penalty: 1.5` (this runtime supports `0.0` only) and
`preserve_thinking` (no equivalent here). The `qwen3_coder` tool-call dialect
still needs checking against real output.

## Traps worth carrying forward

- `release.sh` reports a golden gate that *refused to start* as a "golden
  baseline mismatch". Read the line above it; the cause is usually a foreign
  `swiftpm-testing-helper` from another Dropbox project, which can run in a
  loop. Never kill it — wait, or ask the human. `docs/release-process.md` §5.
- `--publish` re-runs every gate, so publishing needs its own clean pass.
- The converter streams one shard at a time; a partial run resumes rather than
  restarting the download.
- `--plan` is free (config + index only). Use it before any conversion that
  would be expensive to discover a problem in.
