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
             far larger to fetch -- 360 GB against 63 GB -- but it quantizes
             once from the original weights instead of inheriting a third
             party's quantization, and no 8-bit release of this model exists
             that is worth depending on. See docs/qwen38-flash-next-port.md.

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
      prepare_ornith_mtp)
        echo "$name is prepared from the pinned checkpoint; see tools/prepare_ornith_mtp.py"
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
