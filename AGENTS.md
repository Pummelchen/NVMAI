# NVMAI

Swift and Metal inference for Qwen 3.6 35B-A3B in 4-bit, 6-bit, and 8-bit
quantization on Apple Silicon.

## Scope

This checkout is for running and reporting existing behavior. Do not edit source, change runtime defaults, or start optimization work unless the user asks.

## Layout and commands

`Sources/NVMAI/` is the runtime; `Sources/NVMAIRepack/`,
`Sources/NVMAICLI/`, `Sources/NVMAIServer/`, and
`Sources/NVMAIApp/` contain the installer, CLI, loopback server, and
Mac app.
`Tests/` contains focused public tests. User and engineering documentation lives in the
[GitHub Wiki](https://github.com/Pummelchen/NVMAI/wiki).

```bash
swift run -c release NVMAIRepack --model qwen36 --output models/qwen36.gturbo
swift run -c release NVMAIRepack --model qwen36 --output models/qwen36.gturbo --resume
swift build -c release
.build/release/NVMAIMac
swift run -c release NVMAICLI \
  --model models/qwen36.gturbo \
  --prompt "The capital of France is" \
  --max-new 64
```

The installer streams the pinned model without staging the full source checkpoint. Set `HF_TOKEN` only if requested. The 4-bit download is about 19.5 GB; 6-bit and 8-bit require more. Cancellation preserves verified completed ranges; continue them with `--resume` or remove them with `--discard-partial --output <model.gturbo>`.

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

Before a model run, require macOS 26+, Swift 6.3+, enough disk, acceptable `memory_pressure -Q`, a completed selected Qwen `.gturbo` installation, and no process from `pgrep -fl 'NVMAIServer|NVMAIMac|NVMAIDecodeService|NVMAICLI|NVMAIPackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm'`. If a check fails, inform the user and stop; do not terminate apps or delete or reinstall the model.

Run package tests serially (`swift test --no-parallel`), passing any extra
arguments like `--filter` through. Run only one app, CLI, or model-using test
at a time.

For performance results, build release once and follow the [community benchmark guide](https://github.com/Pummelchen/NVMAI/wiki/Benchmarking-Guide) exactly. Do not enable experimental controls or profiling.

Do not download a full checkpoint, duplicate the `.gturbo` model, create a worktree, or purge caches just to run tests.

Report the commit, hardware and RAM, macOS, Swift version, exact command, exit code, complete timing footer or error, and every protocol deviation. Treat results as measurements, not performance ceilings.

## App controls

The Mac app sends prompts through Qwen's ChatML format. It
exposes context length, temperature, Top-K, Top-P, expert-cache slots, prefill,
and RDADVISE. The defaults are temperature `0.2`, Top-K `64`, and Top-P `0.95`.
Responses can use the context space left after formatting the prompt, and FP16
is the runtime KV format. The HUD shows generation rate, token count, and
decode-service memory; Last run also shows time to first token and I/O. Build
the app with its sibling `NVMAIDecodeService`; it never loads a second
in-process model. See [README](README.md) and [Runtime controls](https://github.com/Pummelchen/NVMAI/wiki/Runtime-Controls).
