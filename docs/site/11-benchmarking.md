> **Category:** Reference
> **Status:** draft v1 — to be reviewed before posting
> **Wiki source:** Benchmarking-Guide

# Benchmarking: how to measure NVMAI

NVMAI is a measurement project: every performance claim in this forum comes from a script in `benchmark/` with a fixed model, seed, and hardware. If you're going to quote a number — here, or in an issue — this is how the number is supposed to be made. The short version: **one build, one command, three fresh runs, the full footer, and the machine it ran on.**

## The rule that makes numbers mean anything

**Do not compare a result with another row unless the model, prompt, generated tokens, settings, and stop condition match.** A tok/s number with no machine, no build, and no prompt attached is an adjective, not a measurement. Results are *measurements of that machine and configuration*, not performance ceilings.

## 1. Prepare

- Laptop on **power**, **Low Power Mode off**.
- Quit unrelated heavy workloads.
- Use a **completed, verified** model installation.
- Check memory pressure and require an **empty model-process check**:

```bash
memory_pressure -Q
pgrep -fl 'NVMAIServer|NVMAIMac|NVMAIDecodeService|NVMAICLI|swiftpm-testing-helper|mlx_lm|mlx-lm'
```

Do **not** kill an existing process merely to run a benchmark. Build once:

```bash
swift build -c release
```

## 2. Record the environment

```bash
git rev-parse HEAD
git status --short
sw_vers
swift --version
system_profiler SPHardwareDataType | \
  awk -F': ' '/Model Name|Model Identifier|Chip|Total Number of Cores|Memory/ { print $1 ": " $2 }'
```

The filtered hardware command intentionally omits serial numbers and hardware UUIDs.

## 3. Run a fixed case

One **discarded warmup**, then **three fresh-process measurements** with the same command. Normal benchmark requests use the production policy (temperature 0.6, Top-P 0.95, Top-K 20, presence penalty 0.0); this example overrides temperature with greedy to remove sampling variation:

```bash
.build/release/NVMAICLI \
  --model models/ornith-1.5_35B_A3B_8Bit \
  --prompt "Explain what a mutex is and when you would use one." \
  --max-new 96 \
  --max-context 262144 \
  --rope-scaling none \
  --kv-bits 8 \
  --temperature 0 \
  --seed 1234
```

Run **only one model process at a time**. Preserve stdout and the **complete stderr footer** for every measured run — the footer is where the timing and token counts live.

The resumable coding-client and feature matrix:

```bash
python3 benchmark/coder_cli_benchmark.py
python3 benchmark/coder_cli_benchmark.py --round features
```

**Review every saved answer.** An exit code alone is not a quality result. Benchmarks select thinking explicitly and default it to off; to qualify the real reasoning branch, run the same command with `NVMAI_THINKING_MODE=on` (labels stay `off`/`on` — Ornith doesn't publish effort levels). For a server or client benchmark, start one server and **reuse it** for the whole configuration; don't restart between cache-warm requests unless startup is the behavior you're measuring.

## 4. Check the output

A usable result must:

- exit successfully;
- finish at a normal end-of-turn or the declared token limit;
- contain **coherent, complete text with no repetition loop**; and
- report prompt tokens, generated tokens, prefill/TTFT, and decode rate.

**Do not replace a failed result with a new baseline.** Record the failure and its exact command. Re-capturing a baseline to make a mismatch go away is how a real regression becomes invisible.

## 5. Report

Include all of it, or don't post the number:

- commit SHA and clean/dirty status;
- Mac model, chip, RAM, macOS, Swift version;
- quantization and model-installation path;
- exact command and exit code;
- power mode, memory pressure, and other active workloads;
- **the complete timing footer** and the output-quality result;
- **every deviation from this protocol.**

Use **medians only across truly identical successful runs.** A faster public-only result that fails hidden validation is not a valid winner.

## The golden baseline (the real-inference check)

The one check that exercises *actual* inference is the greedy, fixed-seed Ornith 1.5 4-bit baseline against `benchmark/golden/` (`tools/golden-baseline.sh --check 4`). A baseline is valid for **one (machine, build, model) triple** — re-capture only for a deliberate numerics change, never to make a mismatch disappear. This is why "byte-identical on all eight goldens" is the bar a change to the runtime or the model-load path has to meet.

## Where to go next

- The numbers the default profile produces: [The RAM budget](#04), [Long context and KV cache](#08)
- Why 8-bit is slower (more bytes per token, more fidelity): the FAQ in the wiki

*Version at time of writing: NVMAI 5.1.*
