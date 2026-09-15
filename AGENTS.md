# TinyTitan

<!-- agent-harnesses:begin -->
> **One instruction file.** This is it. Codex, DeepSeek Harness, OpenCode,
> Qwen Code, Qoder and Zed read `AGENTS.md` directly, and Claude Code reads it
> through the committed `CLAUDE.md`, which contains nothing but `@AGENTS.md`.
> **Edit only this file** — do not add a second set of instructions anywhere.
>
> Do **not** add `.rules`, `.cursorrules`, `.windsurfrules`, `.clinerules`,
> `.github/copilot-instructions.md` or `AGENT.md`. Zed takes the *first match*
> from that list, **ahead of `AGENTS.md`**, so any one of them silently
> replaces this file for every Zed user.
<!-- agent-harnesses:end -->

Swift and Metal inference for Qwen-family MoE and dense text models on Apple
Silicon, streaming routed experts from SSD so a model larger than RAM still runs.
Supports 4-bit and 8-bit builds of Qwen3.8-Flash-Next 125B-A6B,
KAT-Coder-V2.5-Dev 35B-A3B, Qwen-AgentWorld 35B-A3B, Ornith 1.5 35B-A3B and
Qwen 3.6 35B-A3B, plus the dense Qwen 3.5 2B/4B/9B on either engine. **The README
is the authority on which checkpoints and widths are supported** — it changes more
often than this file, so do not treat any list here as the current matrix.
`Qwen 3.5 <size> <bits>` (keys `qwen35-2b/4b/9b`) are dense: no routed experts,
nothing streamed from SSD. Ornith 1.5 8-bit is the default install and the default
golden target; 6-bit is a withdrawn legacy format.

## Working here

This is the development repository — changing source is the point, subject to the
gates below, and a change is finished when it is tested, not when it compiles.
What needs a reason is *optimization*: do not start performance work, change
runtime defaults, or alter numerics unless the user asked. Report results as
measurements, not as ceilings, and treat an unexplained slowdown as a finding.

## Layout and commands

`sources/` holds one directory per SwiftPM target. `sources/TinyTitan/` is the
runtime; `sources/TinyTitanFormat/` plus `sources/TinyTitanKernelsC/` are its
format types and C kernels. `sources/TinyTitanRepack/`, `sources/TinyTitanCLI/`,
`sources/TinyTitanServer/` and `sources/TinyTitanApp/` hold the installer, CLI,
loopback server and Mac app; the app's `sources/TinyTitanDecodeService/` and
`sources/TinyTitanDecodeProtocol/` are the out-of-process decode helper and its IPC
contract. `sources/TinyTitanMemory/` and `sources/ContinuityCore/` are persistent
agent memory, `sources/TinyTitanMemoryTool/` inspects it, and
`sources/TinyTitanBench/` plus `sources/TinyTitanValidation/` are the benchmark
driver and the validation/reference target. An executable target keeps its
top-level or `@main` entry in `Command/`; `plugins/dsh-tinytitan/` is the DeepSeek
Harness bundle (route writer + quiet compaction). `docs/repository-layout.md` has
the conventions, `tests/` mirrors `sources/` path for path and never loads a model,
and user and engineering documentation lives in the
[GitHub Wiki](https://github.com/Pummelchen/TinyTitan/wiki).

```bash
swift build -c release
.build/release/TinyTitanMac
swift run -c release TinyTitanCLI \
  --model models/kat-coder-v2.5_35B_A3B_4Bit \
  --prompt "The capital of France is" \
  --max-new 64
```

## Models

**Installing a model is a separate, operator-requested job** — never run it to
satisfy a check, a gate, a benchmark or a release. The 4-bit download is about
19.5 GB and the 8-bit about 36.9 GB. The installer streams the pinned checkpoint
without staging the full source, so it needs `HF_TOKEN` only if one is requested;
cancellation preserves verified completed ranges, which `--resume` continues and
`--discard-partial --output <model.gturbo>` removes.

```bash
swift run -c release TinyTitanRepack --model ornith15-8bit --output models/ornith-1.5_35B_A3B_8Bit
swift run -c release TinyTitanRepack --model ornith15-8bit --output models/ornith-1.5_35B_A3B_8Bit --resume
```

An installed model's `verified-install.json` receipt is bound to the absolute path
it was installed to, so **moving or renaming a model directory makes it fail to
load** with `trusted receipt invalid: model directory mismatch`. This is not
corruption and does not need a re-download — re-issue the receipt in place
(re-hashes the payload against the manifest and rebinds it to the current path):

```bash
swift run -c release TinyTitanRepack --verify-install --input-gturbo models/kat-coder-v2.5_35B_A3B_4Bit
```

Never hand-edit the receipt to match the new path: the path binding is what detects
a moved or swapped directory, so editing it forges the attestation instead of
re-establishing it. Adding a model is the other runbook, `docs/adding-a-model.md`:
it lists the nine places a new checkpoint has to be wired — the last being its ANE
prefill sidecar — the disk each width needs, the verification bar before it may be
called supported, and how to re-issue install receipts after the checkout moves.

## Test rules

Before a model run, require macOS 26+, Swift 6.4+, enough disk, acceptable
`memory_pressure -Q`, a completed selected `.gturbo` installation, and no process
from `pgrep -fl 'TinyTitanServer|TinyTitanMac|TinyTitanDecodeService|TinyTitanCLI|TinyTitanPackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm'`.
If a check fails, inform the user and stop; do not terminate apps or delete or
reinstall the model.

Run package tests serially (`swift test --no-parallel`), passing extra arguments
like `--filter` through. Run only one app, CLI, or model-using test at a time.

`tools/lint.sh` runs the five gates CI enforces beyond the compiler: no `as!` /
`try!` under `sources/` without a `lint:allow-force <reason>` comment above it; no
function over 120 lines without an inline `lint:allow-long <reason>` — the ratchet
file `tools/func-length-baseline.txt` is currently **empty**, because every long
function carries its own justification, and the gate fails on a stale exemption row
as well as on a new offender; every `@unchecked Sendable` carrying an
`unchecked-invariant:` note; a `converter` probe that files routed experts by index
rather than arrival order; and no hardcoded SwiftPM target triple in a build path,
which points at nothing on a newer toolchain or at a stale binary on this one.
`tools/lint.sh <mode>` runs a single gate (`force-cast`, `func-length`, `sendable`,
`converter`, `arch-path`).

`tools/golden-baseline.sh --check <target>` compares greedy, fixed-seed generation
against `benchmark/golden/`. It is the only check that exercises real inference, so
run it for any change to the runtime or the model-load path — the unit tests never
load a model. Ornith 1.5 8-bit (`8`) is the default target; bare `4` still means
Ornith 1.5 4-bit. It counts as a model run: apply the preconditions above first. A
baseline is valid for one (machine, build, model) triple; re-capture only for a
deliberate numerics change, never to make a mismatch go away.

**Verification uses only the models already installed under `models/`.** `models/`
is deliberately kept smaller than the full supported set to save disk, so a golden
target with no install there is *reported as not checked* — by `release.sh` and in
the release notes — and never "fixed" by downloading, converting, repacking or
re-installing it. No gate, benchmark or release step may fetch a model to satisfy
itself. Do not download a full checkpoint, duplicate the `.gturbo` model, create a
worktree, or purge caches just to run tests, a gate or a release.

For performance results, build release once and follow the
[community benchmark guide](https://github.com/Pummelchen/TinyTitan/wiki/Benchmarking-Guide)
exactly. Do not enable experimental controls or profiling. Launch helpers live in
`benchmark/`; start the server before running any benchmark script.

Report the commit, hardware and RAM, macOS, Swift version, exact command, exit code,
complete timing footer or error, and every protocol deviation.

## Local server

Follow the [server guide](https://github.com/Pummelchen/TinyTitan/wiki/OpenAI-Compatible-Server)
for launch commands, health checks, client setup, prompt reuse, tool loops, and
supported API behavior. Apply the model-process checks above first; never start a
second model process or terminate an existing one.

Keep the server on `127.0.0.1`; it has no remote authentication or TLS, so do not
proxy, tunnel, or expose it. A tool call from the local model never bypasses the
client's normal permission policy. Keep the execution session alive while the
server is needed, and stop only a server you launched.

## App controls

The Mac app sends prompts through Qwen's ChatML format. It exposes context length,
temperature, Top-K, Top-P, expert-cache slots, prefill, and RDADVISE. Temperature
defaults to `0.6` for the Qwen 3.5/3.6 families, and `1.0` for
Qwen3.8-Flash-Next and for KAT-Coder-V2.5-Dev (whose own model card sets it, not
its Qwen 3.6 base); Top-K `20`, Top-P `0.95`, and presence penalty `0.0` (the only
currently supported presence-penalty value) are shared by all of them. Responses
can use the context space left after formatting the prompt, and FP16 is the runtime
KV format. The HUD shows generation rate, token count, and decode-service memory;
Last run also shows time to first token and I/O. Build the app with its sibling
`TinyTitanDecodeService`; it never loads a second in-process model. See
[README](README.md) and
[Runtime controls](https://github.com/Pummelchen/TinyTitan/wiki/Runtime-Controls).

## Releases and handover

Cutting a release is a runbook, not improvisation: `docs/release-process.md` holds
the order (notes, changelog, version, tag, dry run, publish), the machine
preconditions, and what `release.sh`'s failure messages actually mean — including a
golden gate that reports a *refused* start as a "mismatch". The cross-repository
standard is [`RELEASE.md`](RELEASE.md).

Work in flight is handed over in `docs/handover-<name>.md`;
`docs/handover-tinytitan.md` is the current one and starts with the prompt for the
next session. Read it before installing, converting or moving anything: it names
what is open, and the traps that have already cost a session.

<!-- release-rules:begin -->
## Releasing

**Read [`RELEASE.md`](RELEASE.md) before cutting a release.** It carries the
generic rules every Pummelchen repository follows, plus this repository's own
section. Do not improvise a release.

The non-negotiables:

- **Apple Silicon only** — build native `arm64` (M1–M6). Never `--arch x86_64`,
  never `ARCHS=arm64 x86_64`, and never `lipo -create`, which is how a universal
  binary gets made.
- **Assert it** — `lipo -archs <binary>` must report exactly `arm64`. A build that
  silently produced a fat binary is a release defect, not a build option.
- **Every release carries the artifacts.** A tag alone is not a release.
- **Identity is single-sourced and enforced** — never bump one declaration of the
  version or build number on its own; the build or CI must fail on a mismatch.
- **Dry run first**; publish only on an explicit flag.
- **Never fetch a model, dataset or dependency to make a gate pass.** A check that
  cannot run is reported *not checked*, and the release notes must name it.
<!-- release-rules:end -->
