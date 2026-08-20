# NVMAI

Swift and Metal inference for compatible Qwen3.5-MoE 35B-A3B text models on
Apple Silicon. Pinned installers support Qwen 3.6 and Ornith 1.5 in 4-bit and
8-bit; 6-bit is a withdrawn legacy format.

## Scope

This checkout is for running and reporting existing behavior. Do not edit source, change runtime defaults, or start optimization work unless the user asks.

## Layout and commands

`sources/NVMAI/` is the runtime; `sources/NVMAIRepack/`,
`sources/NVMAICLI/`, `sources/NVMAIServer/`, and
`sources/NVMAIApp/` contain the installer, CLI, loopback server, and
Mac app.
`tests/` contains focused public tests. User and engineering documentation lives in the
[GitHub Wiki](https://github.com/Pummelchen/NVMAI/wiki).

```bash
swift run -c release NVMAIRepack --model qwen36 --output models/qwen3.6_35B_A3B_4Bit
swift run -c release NVMAIRepack --model ornith15 --output models/ornith-1.5_35B_A3B_4Bit
swift run -c release NVMAIRepack --model qwen36 --output models/qwen3.6_35B_A3B_4Bit --resume
swift build -c release
.build/release/NVMAIMac
swift run -c release NVMAICLI \
  --model models/qwen3.6_35B_A3B_4Bit \
  --prompt "The capital of France is" \
  --max-new 64
```

The installer streams the pinned model without staging the full source checkpoint. Set `HF_TOKEN` only if requested. The 4-bit download is about 19.5 GB; 6-bit and 8-bit require more. Cancellation preserves verified completed ranges; continue them with `--resume` or remove them with `--discard-partial --output <model.gturbo>`.

An installed model's `verified-install.json` receipt is bound to the absolute
path it was installed to, so **moving or renaming a model directory makes it
fail to load** with `trusted receipt invalid: model directory mismatch`. This
is not corruption and does not need a re-download — re-issue the receipt in
place (re-hashes the payload against the manifest and rebinds it to the
current path):

```bash
swift run -c release NVMAIRepack --verify-install --input-gturbo models/qwen3.6_35B_A3B_4Bit
```

Never hand-edit the receipt to match the new path: the path binding is what
detects a moved or swapped directory, so editing it forges the attestation
instead of re-establishing it.

## Local server

Follow the [server guide](https://github.com/Pummelchen/NVMAI/wiki/OpenAI-Compatible-Server) for launch commands, health
checks, client setup, prompt reuse, tool loops, and supported API behavior.
Apply the model-process checks below first; never start a second model process
or terminate an existing one.

Keep the server on `127.0.0.1`; it has no remote authentication or TLS, so do
not proxy, tunnel, or expose it. A tool call from the local model never bypasses
the client's normal permission policy. Keep the execution session alive while
the server is needed, and stop only a server you launched.

## Test rules

Before a model run, require macOS 26+, Swift 6.3+, enough disk, acceptable `memory_pressure -Q`, a completed selected `.gturbo` installation, and no process from `pgrep -fl 'NVMAIServer|NVMAIMac|NVMAIDecodeService|NVMAICLI|NVMAIPackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm'`. If a check fails, inform the user and stop; do not terminate apps or delete or reinstall the model.

Run package tests serially (`swift test --no-parallel`), passing any extra
arguments like `--filter` through. Run only one app, CLI, or model-using test
at a time.

`tools/lint.sh` runs the two gates CI enforces beyond the compiler: no `as!` /
`try!` under `sources/` without a `lint:allow-force <reason>` comment above it,
and no new function over 120 lines (existing ones are listed in
`tools/func-length-baseline.txt`; drop a row when its function shrinks below
the limit, or the gate fails on the stale exemption).

`tools/golden-baseline.sh --check 4` compares greedy, fixed-seed generation
against `benchmark/golden/`. It is the only check that exercises real
inference, so run it for any change to the runtime or the model-load path —
the unit tests never load a model. It counts as a model run: apply the
preconditions above first. A baseline is valid for one (machine, build, model)
triple; re-capture only for a deliberate numerics change, never to make a
mismatch go away.

For performance results, build release once and follow the [community benchmark guide](https://github.com/Pummelchen/NVMAI/wiki/Benchmarking-Guide) exactly. Do not enable experimental controls or profiling.

Launch helpers live in `benchmark/`. Start the server before running any benchmark script.

Do not download a full checkpoint, duplicate the `.gturbo` model, create a worktree, or purge caches just to run tests.

Report the commit, hardware and RAM, macOS, Swift version, exact command, exit code, complete timing footer or error, and every protocol deviation. Treat results as measurements, not performance ceilings.

## App controls

The Mac app sends prompts through Qwen's ChatML format. It
exposes context length, temperature, Top-K, Top-P, expert-cache slots, prefill,
and RDADVISE. The defaults are temperature `0.6`, Top-K `20`, Top-P `0.95`,
and presence penalty `0.0` (the only currently supported presence-penalty value).
Responses can use the context space left after formatting the prompt, and FP16
is the runtime KV format. The HUD shows generation rate, token count, and
decode-service memory; Last run also shows time to first token and I/O. Build
the app with its sibling `NVMAIDecodeService`; it never loads a second
in-process model. See [README](README.md) and [Runtime controls](https://github.com/Pummelchen/NVMAI/wiki/Runtime-Controls).
