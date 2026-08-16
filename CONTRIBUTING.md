# Contributing

NVMAI welcomes focused fixes, documentation improvements, and
benchmark reports from Apple Silicon Macs.

## Before opening a change

- Keep the package compatible with macOS 26, Swift 6.3, and Metal 4.
- Preserve the bounded-memory model path. Never load a complete checkpoint,
  shard, or large model tensor into Swift heap memory.
- Keep public runtime controls limited to those documented in
  [Runtime controls](https://github.com/Pummelchen/NVMAI/wiki/Runtime-Controls).
- Add or update a focused test for behavior changes.

Run the release build, the production gates, and the serial tests:

```bash
swift build -c release
tools/lint.sh
swift test --no-parallel
```

CI runs the same three, so a green run here is a green run there. `tools/lint.sh`
enforces two things the compiler cannot:

- **force-cast** — no `as!` / `try!` under `sources/`. To keep one, put
  `lint:allow-force <reason>` in the comment block directly above it; a marker
  without a reason fails exactly like no marker.
- **func-length** — a ratchet, not a limit. Functions already over 120 lines are
  listed in `tools/func-length-baseline.txt`; the gate fails on *new* ones. If
  you shrink a baselined function below the limit, drop its row — the gate
  fails on stale exemptions so they cannot be reused later.

These checks do not download or load the model. For a change to the runtime or
the model-load path, also run the golden baseline, which is the only check that
exercises real inference:

```bash
tools/golden-baseline.sh --check 4        # or: 4 6 8
```

It compares greedy, fixed-seed output against `benchmark/golden/`. A baseline is
valid for one (machine, build, model) triple — capture your own with
`tools/golden-baseline.sh 4` before making changes, and re-capture only for a
deliberate numerics change. For a real-model change also report the prompt,
generated token count, output, timing footer, Mac model, memory, macOS version,
Swift version, and any protocol change.

## Benchmark reports

Follow the [community benchmark protocol](https://github.com/Pummelchen/NVMAI/wiki/Benchmarking-Guide). Review
all captured files before sharing them, and remove personal paths or unrelated
process details.

## Pull requests

Keep each pull request narrow. Explain the behavior change, tests run, and any
remaining limitation. By contributing, you agree that your work is licensed
under the repository's [Apache License 2.0](LICENSE).
