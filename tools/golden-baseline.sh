#!/usr/bin/env bash
# Capture a deterministic generation baseline, so a refactor of the runtime
# can be checked against byte-identical output rather than "the tests still
# pass". Phase 2 of the production audit needs this before RealForwardRunner
# is decomposed.
#
#   tools/golden-baseline.sh [4|8 ...]       # default: Ornith 4-bit
#   tools/golden-baseline.sh --check [...]   # compare against the stored file
#
# Determinism comes from greedy decoding: --temperature 0 with a fixed seed and
# a fixed prompt. Greedy means the sampler never draws, so the only inputs are
# the weights and the kernels — exactly what a runtime refactor must not change.
#
# SCOPE: a baseline is valid for one (machine, build, model) triple. Metal
# reduction order is not guaranteed across GPU families, so a file captured on
# an M3 is not a reference for an M4. Re-capture after a deliberate numerics
# change; a diff at any other time is a regression.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CLI="$ROOT/.build/arm64-apple-macosx/release/NVMAICLI"
OUT_DIR="${OUT_DIR:-$ROOT/benchmark/golden}"

PROMPT="${PROMPT:-Explain what a mutex is and when you would use one.}"
MAX_NEW="${MAX_NEW:-96}"
SEED="${SEED:-1234}"

mode=capture
if [ "${1:-}" = "--check" ]; then mode=check; shift; fi
quants=("$@"); [ ${#quants[@]} -eq 0 ] && quants=(4)

if [ ! -x "$CLI" ]; then
  echo "missing $CLI — run: swift build -c release" >&2
  exit 2
fi

# AGENTS.md: never run alongside another model process, and never terminate one
# we did not start. Refuse rather than race.
if pgrep -f 'NVMAIServer|NVMAIMac|NVMAIDecodeService|NVMAICLI|NVMAIPackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm' >/dev/null 2>&1; then
  echo "a model process is already running; stop it yourself, then re-run" >&2
  exit 3
fi

mkdir -p "$OUT_DIR" "$ROOT/.build"
status=0

for q in "${quants[@]}"; do
  case "$q" in
    4|8) ;;
    *) echo "unsupported Ornith baseline quantization: $q (expected 4 or 8)" >&2
       status=1; continue ;;
  esac
  model="$ROOT/models/ornith-1.5_35B_A3B_${q}Bit"
  if [ ! -f "$model/verified-install.json" ]; then
    echo "missing ${q}-bit baseline model: no verified install at ${model#$ROOT/}" >&2
    status=1
    continue
  fi
  file="$OUT_DIR/ornith-1.5-35b-a3b-${q}bit.txt"
  work="$(mktemp "$ROOT/.build/golden-baseline.XXXXXX")"

  echo "== ${q}-bit =="
  # --quiet keeps the timing footer out of the compared text; only the
  # generated tokens are the contract. Timings vary run to run by design.
  "$CLI" --model "$model" --prompt "$PROMPT" --max-new "$MAX_NEW" \
         --temperature 0 --seed "$SEED" --quiet > "$work" 2>"$work.err"
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "  FAILED (exit $rc)"; sed 's/^/  /' "$work.err" | head -20
    rm -f "$work" "$work.err"; status=1; continue
  fi

  if [ "$mode" = capture ]; then
    {
      echo "# prompt:      $PROMPT"
      echo "# max-new:     $MAX_NEW"
      echo "# temperature: 0 (greedy)"
      echo "# seed:        $SEED"
      echo "# quant:       ${q}-bit"
      echo "# captured-on: $(sysctl -n hw.model), $(( $(sysctl -n hw.memsize) / 1073741824 )) GB, macOS $(sw_vers -productVersion)"
      echo "# commit:      $(git -C "$ROOT" rev-parse --short HEAD)"
      echo "---"
      cat "$work"
    } > "$file"
    echo "  captured -> ${file#$ROOT/} ($(wc -c < "$work" | tr -d ' ') bytes)"
  else
    if [ ! -f "$file" ]; then
      echo "  no baseline at ${file#$ROOT/}; run without --check first"; status=1
    elif diff -q <(sed '1,/^---$/d' "$file") "$work" >/dev/null; then
      echo "  ok — output identical to baseline"
    else
      echo "  MISMATCH against ${file#$ROOT/}:"
      diff <(sed '1,/^---$/d' "$file") "$work" | head -30 | sed 's/^/    /'
      status=1
    fi
  fi
  rm -f "$work" "$work.err"
done

exit $status
