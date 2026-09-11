#!/usr/bin/env bash
#
# The whole-model check for the dense CPU models on disk: `NVMAIBench cpu35`
# must print "all continuations correct" for each.
#
# The equivalence gate (tests/NVMAI/Core/CPUEngine/DenseGTurboEquivalenceTests.swift,
# driven by tools/repack_dense.sh) compares logits against the snapshot a repack
# came from. This asks the other question -- does the engine continue real text
# correctly end to end -- so the two are complements, not duplicates.
#
# Each model is faulted into memory (up to 4.5 GB for the 4B at 8 bits) and runs
# on the performance cores, so it refuses to run beside a live model server
# rather than evicting that server's pages or skewing a measurement it is part
# of. That is the guard AGENTS.md states for any model run.
#
# Usage:
#   tools/verify_cpu_models.sh [model ...]        # default: the four below
#
# Model names are directory names under models/.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS="$ROOT/models"
DEFAULT_MODELS=(qwen3.5_2B_4Bit qwen3.5_2B_8Bit qwen3.5_4B_4Bit qwen3.5_4B_8Bit)
GUARD='NVMAIServer|NVMAIMac|NVMAIDecodeService|NVMAICLI|NVMAIPackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm'

if [ "$#" -gt 0 ]; then
  targets=("$@")
else
  targets=("${DEFAULT_MODELS[@]}")
fi

if pgrep -fl "$GUARD" >/dev/null 2>&1; then
  echo "a model process is already running; not starting:"
  pgrep -fl "$GUARD" | sed 's/^/  /'
  exit 2
fi

cd "$ROOT" || exit 1
for model in "${targets[@]}"; do
  [ -f "$MODELS/$model/manifest.json" ] || {
    echo "no $MODELS/$model (an installed .gturbo)"; exit 1; }
done

swift build -c release --product NVMAIBench || exit 1
BENCH="$(swift build -c release --show-bin-path)/NVMAIBench"

status=0
for model in "${targets[@]}"; do
  echo "== $model"
  out="$(/usr/bin/time -l "$BENCH" cpu35 "$MODELS/$model" 2>&1)"
  # /usr/bin/time -l prints a page of counters per run; keep the verdict and
  # the peak footprint, which is what a reader compares between models.
  echo "$out" | grep -vE '^ +[0-9]+ +[a-z]' || true
  echo "$out" | grep 'maximum resident set size' || true
  grep -q "all continuations correct" <<<"$out" || status=1
done

if [ "$status" -eq 0 ]; then
  echo "VERIFIED: ${#targets[@]} model(s)"
else
  echo "FAILED: see above"
fi
exit "$status"
