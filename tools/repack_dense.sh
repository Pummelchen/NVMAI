#!/usr/bin/env bash
#
# Repack the dense Qwen 3.5 models as .gturbo installs, and prove the result.
#
# The three (six, with 8-bit) small Qwen 3.5 models were the only installs in
# this project that were affine safetensors snapshots rather than .gturbo
# directories. They are .gturbo now, built from the converted snapshot, so the
# whole pipeline is one shape: convert -> repack -> receipt -> verify-install.
#
# Usage:
#   tools/repack_dense.sh <2b|4b|9b|all> [4|8|both]
#
# What it does for each model, in this order and stopping at the first failure:
#
#   1. move the existing snapshot out of models/ and into .build/, which is
#      where the converter stages every other family too (see the ornith MTP
#      row in install_models.sh). It cannot stay in models/: the catalog scans
#      that directory, so a snapshot left beside its install is probed as a
#      second model and reported as a duplicate id;
#   2. repack .build/<key>-affine -> models/<dir>, which writes the receipt
#      bound to that final path -- a receipt is path-bound, so the install has
#      to be written where it will live, never moved into place afterwards;
#   3. byte-diff the residents against the snapshot with
#      tools/gturbo_diff_snapshot.py -- a repack is a byte copy, so this is an
#      exact comparison and not a tolerance;
#   4. re-issue and check the receipt (NVMAIRepack --verify-install);
#   5. run the CPU equivalence gate, which loads both and requires identical
#      logits (tests/NVMAI/CPUEngine/DenseGTurboEquivalenceTests.swift).
#
# Step 5 is the one that matters. Steps 3 and 4 prove the bytes are right and
# the receipt is right; only step 5 proves the reader interprets them right, and
# a wrong per-tensor width would pass both of the others.

set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
MODELS="$ROOT/models"
STAGE="$ROOT/.build"
BIN="$ROOT/.build/release/NVMAIRepack"

which="${1:-}"
widths="${2:-4}"

usage() {
  sed -n '3,32p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
}

[[ -n "$which" ]] || usage

case "$widths" in
  4)    want_bits=(4) ;;
  8)    want_bits=(8) ;;
  both) want_bits=(4 8) ;;
  *)    usage ;;
esac

# key:size_key:bits  ->  snapshot dir name is qwen3.5_<SIZE>_<BITS>Bit
keys=()
case "$which" in
  2b)  keys=(2b) ;;
  4b)  keys=(4b) ;;
  9b)  keys=(9b) ;;
  all) keys=(2b 4b 9b) ;;
  *)   usage ;;
esac

echo "building release (the repacker is the only binary this needs)"
swift build -c release
[[ -x "$BIN" ]] || { echo "no $BIN after build" >&2; exit 1; }

# One model at a time, on purpose: the snapshot and the install both exist at
# the same moment during verification, so the peak is two copies of the largest
# model, not two copies of all of them.
pairs=()
for key in "${keys[@]}"; do
  for bits in "${want_bits[@]}"; do
    dir="qwen3.5_${key^^}_${bits}Bit"
    snapshot="$STAGE/qwen35-${key}-affine-${bits}bit"

    echo
    echo "=============================================================="
    echo "$dir  ($bits-bit)"
    echo "=============================================================="

    # Stage the snapshot out of models/. A previous run has already done this
    # and left it here; the first run still finds it installed under the plain
    # name.
    if [[ ! -d "$snapshot" ]]; then
      if [[ -f "$MODELS/$dir/manifest.json" ]]; then
        echo "  $dir is already a .gturbo install and its snapshot is gone."
        echo "  Rebuild it with tools/prepare_qwen35.py --size ${key} --bits $bits"
        echo "  --output $snapshot first, then re-run this; skipping."
        continue
      fi
      [[ -d "$MODELS/$dir" ]] || { echo "  no $MODELS/$dir to repack" >&2; exit 1; }
      echo "  staging the snapshot at $snapshot"
      mv "$MODELS/$dir" "$snapshot"
    fi

    echo "  repacking -> $MODELS/$dir"
    rm -rf "${MODELS:?}/$dir"
    "$BIN" --input-snapshot "$snapshot" --model-id "qwen3.5-$key" \
        --output "$MODELS/$dir"

    echo "  byte-diffing every resident tensor against the snapshot"
    python3 tools/gturbo_diff_snapshot.py "$MODELS/$dir" "$snapshot"

    echo "  re-issuing and checking the receipt"
    "$BIN" --verify-install --input-gturbo "$MODELS/$dir"

    pairs+=("$snapshot:$MODELS/$dir")
  done
done

if [[ ${#pairs[@]} -eq 0 ]]; then
  echo "nothing to verify" >&2
  exit 1
fi

joined=$(IFS=,; echo "${pairs[*]}")
echo
echo "logit equivalence gate over ${#pairs[@]} model(s)"
NVMAI_DENSE_EQUIV=1 NVMAI_DENSE_EQUIV_PAIRS="$joined" \
    swift test --no-parallel --filter DenseGTurboEquivalenceTests

echo
echo "all checks passed. The snapshots are kept in .build/ so the equivalence"
echo "gate can be re-run without re-converting; delete them once you are done:"
echo "  rm -rf .build/qwen35-*-affine-*bit"
