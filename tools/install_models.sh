#!/usr/bin/env bash
# Install any supported model at any supported width.
#
#   tools/install_models.sh                 # what is installed, what is missing
#   tools/install_models.sh ornith15-8bit   # install one
#   tools/install_models.sh --all-4bit      # every 4-bit model
#   tools/install_models.sh --all-8bit      # every 8-bit model
#
# Most models install straight from a pinned Hugging Face release through
# NVMAIRepack, which streams and verifies in one pass. Qwen3.8-Flash-Next is
# the exception and is documented below, because the difference matters when
# choosing what to trust.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/.build/arm64-apple-macosx/release/NVMAIRepack"
MODELS="$ROOT/models"

# name|install directory|width|source
CATALOGUE=(
  "ornith15|ornith-1.5_35B_A3B_4Bit|4|repack"
  "ornith15-8bit|ornith-1.5_35B_A3B_8Bit|8|repack"
  "ornith15-mtp|ornith-1.5_35B_A3B_MTP_4Bit|4|prepare_ornith_mtp"
  "qwen36|qwen3.6_35B_A3B_4Bit|4|repack"
  "qwen36-8bit|qwen3.6_35B_A3B_8Bit|8|repack"
  "qwen36-mtp|qwen3.6_35B_A3B_MTP_4Bit|4|repack"
  "qwen38flash|qwen3.8-flash-next_125B_A6B_4Bit|4|convert"
  "qwen38flash-8bit|qwen3.8-flash-next_125B_A6B_8Bit|8|convert"
  "qwen38flash-mtp|qwen3.8-flash-next_125B_A6B_MTP_4Bit|4|repack"
  "agentworld|qwen-agentworld_35B_A3B_4Bit|4|convert_agentworld"
  "agentworld-8bit|qwen-agentworld_35B_A3B_8Bit|8|convert_agentworld"
)

usage() {
  cat <<'USAGE'
Coverage

  Ornith 1.5 35B-A3B      4-bit, 8-bit, MTP draft
  Qwen 3.6 35B-A3B        4-bit, 8-bit, MTP draft
  Qwen3.8-Flash-Next      4-bit, 8-bit, MTP draft

Sources

  repack     NVMAIRepack --model <name>, streaming a pinned Hugging Face
             release directly into a verified .gturbo install.

  convert    tools/prepare_qwen38.py, which quantizes Qwen's own bf16 release
             into an affine snapshot that NVMAIRepack then imports. Slower and
             far larger to fetch -- 360 GB against 63 GB -- but both widths are
             quantized once from the original weights.

             Quantized releases of this model do exist and are deliberately not
             used. Qwen ships an FP8 build at 186 GB, but its config excludes
             943 modules from conversion: everything except the routed experts
             stays bf16, so the only tensors it quantizes are exactly the ones
             NVMAI streams. Its [128, 128] blocks do not correspond to affine
             group-64 either, so sourcing from it means dequantize then
             requantize, and the result is 8 bits wide carrying E4M3's 3-bit
             mantissa -- full bandwidth cost on an I/O-bound decode for less
             than 8-bit quality. Third-party MLX 8-bit repacks have the same
             problem with less provenance.

8-bit and Qwen3.8-Flash-Next

  Executable since the hyper-connection, PLE and QSA-indexer kernels gained an
  affine path. Before that they took an INT4 GEMV unconditionally and read
  8-bit gates as nibbles: the install built, loaded, answered " Paris" and
  degenerated. If you are on a build older than that, 8-bit is refused at load
  rather than producing nonsense.

  Worth knowing before installing it: decode is bound by streaming routed
  experts from SSD, and 8-bit doubles the per-expert record from 2.77 MB to
  5.23 MB. It also halves how many experts fit the same cache budget, so
  expect substantially less than half the 4-bit throughput. The reason to run
  it is quality headroom, not speed.

Disk

  35B 4-bit   ~19 GB      35B 8-bit   ~35 GB
  125B 4-bit  ~157 GiB    125B 8-bit  ~219 GiB

  The 125B installs are dominated by the 95.4 GiB PLE n-gram table, which is
  identical in both widths. docs/ngram-table-sharing-plan.md covers sharing one
  copy between them.
USAGE
}

status() {
  printf '%-20s %-8s %-10s %s\n' MODEL WIDTH STATE SOURCE
  for row in "${CATALOGUE[@]}"; do
    IFS='|' read -r name dir width source <<<"$row"
    if [[ -d "$MODELS/$dir" ]]; then
      state="installed"
    else
      state="-"
    fi
    printf '%-20s %-8s %-10s %s\n' "$name" "${width}-bit" "$state" "$source"
  done
  echo
  echo "tools/install_models.sh <name>   to install one"
  echo "tools/install_models.sh --help   for sources and disk sizes"
}

install_one() {
  local want="$1" found=0
  for row in "${CATALOGUE[@]}"; do
    IFS='|' read -r name dir width source <<<"$row"
    [[ "$name" == "$want" ]] || continue
    found=1
    if [[ -d "$MODELS/$dir" ]]; then
      echo "$name is already installed at models/$dir"
      return 0
    fi
    case "$source" in
      repack)
        [[ -x "$BIN" ]] || { echo "build NVMAIRepack first: swift build -c release" >&2; return 1; }
        echo "installing $name -> models/$dir"
        "$BIN" --model "$name" --output "$MODELS/$dir" --resume
        ;;
      convert)
        cat <<EOF
$name is built from Qwen's own bf16 release rather than a third-party repack,
so it is two steps and a 360 GB fetch:

  python3.13 tools/prepare_qwen38.py --bits $width \\
      --output .build/qwen38-affine-${width}bit \\
      --work .build/qwen38-shards

  $BIN --input-snapshot .build/qwen38-affine-${width}bit \\
      --model-id qwen3.8-flash-next --output "$MODELS/$dir"

Check free space first: the snapshot and the install exist at the same time.
EOF
        ;;
      convert_agentworld)
        # Qwen's bf16 release, quantized one shard at a time by
        # tools/prepare_agentworld.py (about 70 GB fetched, at most two
        # shards on disk), then repacked. The snapshot and the install exist
        # at the same time: ~40 GB at 4-bit, ~75 GB at 8-bit.
        [[ -x "$BIN" ]] || { echo "build NVMAIRepack first: swift build -c release" >&2; return 1; }
        echo "converting $name -> .build/agentworld-affine-${width}bit"
        python3.13 tools/prepare_agentworld.py --bits "$width" \
            --output ".build/agentworld-affine-${width}bit" \
            --work .build/agentworld-shards || return 1
        echo "installing $name -> models/$dir"
        "$BIN" --input-snapshot ".build/agentworld-affine-${width}bit" \
            --model-id qwen-agentworld --output "$MODELS/$dir"
        ;;
      prepare_ornith_mtp)
        echo "$name is prepared from the pinned checkpoint; see tools/prepare_ornith_mtp.py"
        ;;
      unsupported)
        cat <<EOF
$name cannot be run.

Qwen3.8-Flash-Next drives its hyper-connection, PLE and QSA-indexer
projections through INT4-only kernels, so an 8-bit install builds correctly
and cannot execute -- the runtime refuses it at load. Use the 4-bit build.

See --help for what making 8-bit work would require.
EOF
        return 1
        ;;
    esac
    return 0
  done
  [[ "$found" == 1 ]] || { echo "unknown model: $want" >&2; status >&2; return 2; }
}

case "${1:-}" in
  "")            status ;;
  --help|-h)     usage ;;
  --all-4bit)    for row in "${CATALOGUE[@]}"; do IFS='|' read -r n _ w _ <<<"$row"
                   [[ "$w" == 4 ]] && install_one "$n"; done ;;
  --all-8bit)    for row in "${CATALOGUE[@]}"; do IFS='|' read -r n _ w _ <<<"$row"
                   [[ "$w" == 8 ]] && install_one "$n"; done ;;
  *)             install_one "$1" ;;
esac
