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
  "ornith15|ornith-1.5_35B_A3B_4Bit|4|convert_qwen35moe"
  "ornith15-8bit|ornith-1.5_35B_A3B_8Bit|8|convert_qwen35moe"
  "ornith15-mtp|ornith-1.5_35B_A3B_MTP_4Bit|4|prepare_ornith_mtp"
  "qwen36|qwen3.6_35B_A3B_4Bit|4|convert_qwen35moe"
  "qwen36-8bit|qwen3.6_35B_A3B_8Bit|8|convert_qwen35moe"
  "qwen36-mtp|qwen3.6_35B_A3B_MTP_4Bit|4|convert_qwen36_mtp"
  "qwen38flash|qwen3.8-flash-next_125B_A6B_4Bit|4|convert"
  "qwen38flash-8bit|qwen3.8-flash-next_125B_A6B_8Bit|8|convert"
  "qwen38flash-mtp|qwen3.8-flash-next_125B_A6B_MTP_4Bit|4|convert_qwen38_mtp"
  "agentworld|qwen-agentworld_35B_A3B_4Bit|4|convert_qwen35moe"
  "agentworld-8bit|qwen-agentworld_35B_A3B_8Bit|8|convert_qwen35moe"
)

usage() {
  cat <<'USAGE'
Coverage

  Ornith 1.5 35B-A3B      4-bit, 8-bit, MTP draft
  Qwen 3.6 35B-A3B        4-bit, 8-bit, MTP draft
  Qwen3.8-Flash-Next      4-bit, 8-bit, MTP draft

Sources

  Every install is built from the model's own bf16 release, quantized here
  (group-64 affine) by the tools in tools/ and imported by NVMAIRepack.
  Third-party quantizations are deliberately not used: their group sizes,
  widths and norm conventions are theirs, and here the router, the
  shared-expert gate, the DeltaNet gating projections and every norm stay
  at bf16 in both widths.

  convert_qwen35moe   tools/prepare_agentworld.py --model {ornith15,qwen36,agentworld}
                      One ~70 GB download yields both widths.
  convert             tools/prepare_qwen38.py, one 360 GB fetch per width.
                      Qwen's own FP8 build is not used either: it quantizes
                      only the routed experts, in [128, 128] blocks that do
                      not map onto affine group-64.
  convert_qwen36_mtp  the same converter's --draft-head mode (two shards).
  convert_qwen38_mtp  tools/prepare_qwen38_mtp.py (31 tensors, range-fetched).
  prepare_ornith_mtp  tools/prepare_ornith_mtp.py (shard 16 of the original).

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
        # --resume is refused when there is nothing to resume; pass it only
        # when a previous attempt left its state behind.
        if [[ -f "$MODELS/$dir.resume.json" ]]; then
          "$BIN" --model "$name" --output "$MODELS/$dir" --resume
        else
          "$BIN" --model "$name" --output "$MODELS/$dir"
        fi
        ;;
      convert)
        # Qwen's own bf16 release, quantized one shard at a time by
        # tools/prepare_qwen38.py (a 360 GB fetch per width), then repacked.
        [[ -x "$BIN" ]] || { echo "build NVMAIRepack first: swift build -c release" >&2; return 1; }
        if [[ ! -f ".build/qwen38-affine-${width}bit/model.safetensors.index.json" ]]; then
          echo "converting $name -> .build/qwen38-affine-${width}bit"
          python3.13 tools/prepare_qwen38.py --bits "$width" \
              --output ".build/qwen38-affine-${width}bit" \
              --work .build/qwen38-shards || return 1
        fi
        echo "installing $name -> models/$dir"
        "$BIN" --input-snapshot ".build/qwen38-affine-${width}bit" \
            --model-id qwen3.8-flash-next --output "$MODELS/$dir"
        ;;
      convert_qwen38_mtp)
        # The draft head's 31 tensors, range-fetched from Qwen's original by
        # tools/prepare_qwen38_mtp.py, then imported as a draft-head sidecar.
        [[ -x "$BIN" ]] || { echo "build NVMAIRepack first: swift build -c release" >&2; return 1; }
        if [[ ! -f ".build/qwen38-mtp-affine/model.safetensors.index.json" ]]; then
          echo "converting $name -> .build/qwen38-mtp-affine"
          python3.13 tools/prepare_qwen38_mtp.py --bits "$width" \
              --output .build/qwen38-mtp-affine || return 1
        fi
        echo "installing $name -> models/$dir"
        "$BIN" --input-snapshot .build/qwen38-mtp-affine --draft-head \
            --model-id qwen3.8-flash-next-4bit --output "$MODELS/$dir"
        ;;
      convert_qwen36_mtp)
        # Qwen3.6's draft head: 19 tensors of the `mtp.*` namespace in two
        # shards of Qwen's original, converted as a qwen3_5_mtp sidecar.
        [[ -x "$BIN" ]] || { echo "build NVMAIRepack first: swift build -c release" >&2; return 1; }
        if [[ ! -f ".build/qwen36-mtp-affine/model.safetensors.index.json" ]]; then
          echo "converting $name -> .build/qwen36-mtp-affine"
          python3.13 tools/prepare_agentworld.py --model qwen36 --draft-head --bits "$width" \
              --output .build/qwen36-mtp-affine --work .build/qwen36-mtp-shards || return 1
        fi
        echo "installing $name -> models/$dir"
        "$BIN" --input-snapshot .build/qwen36-mtp-affine \
            --model-id qwen3.6-35b-a3b-mtp-4bit --output "$MODELS/$dir"
        ;;
      convert_qwen35moe)
        # Qwen's own bf16 release, quantized one shard at a time by
        # tools/prepare_agentworld.py (about 70 GB fetched, at most two
        # shards on disk), then repacked. Both widths come from one download,
        # so the other width installs without a second fetch. The snapshot
        # and the install exist at the same time: ~40 GB at 4-bit, ~75 GB at
        # 8-bit.
        [[ -x "$BIN" ]] || { echo "build NVMAIRepack first: swift build -c release" >&2; return 1; }
        local preset="${name%-8bit}" model_id
        case "$preset" in
          agentworld) model_id="qwen-agentworld" ;;
          qwen36)     model_id="qwen3.6-35b-a3b" ;;
          ornith15)   model_id="ornith-1.5-35b-a3b" ;;
        esac
        if [[ ! -f ".build/${preset}-affine-${width}bit/model.safetensors.index.json" ]]; then
          echo "converting $preset -> .build/${preset}-affine-{4,8}bit"
          python3.13 tools/prepare_agentworld.py --model "$preset" --bits 4 8 \
              --output ".build/${preset}-affine" \
              --work ".build/${preset}-shards" || return 1
        fi
        echo "installing $name -> models/$dir"
        "$BIN" --input-snapshot ".build/${preset}-affine-${width}bit" \
            --model-id "$model_id" --output "$MODELS/$dir"
        ;;
      prepare_ornith_mtp)
        # Ornith's draft head lives in shard 16 of its original checkpoint;
        # tools/prepare_ornith_mtp.py verifies that shard against its pinned
        # revision and converts it.
        [[ -x "$BIN" ]] || { echo "build NVMAIRepack first: swift build -c release" >&2; return 1; }
        local src=.build/ornith-mtp-src rev=e4dfb35a93d4b6822a811a7676f3488514abe7e2
        local base="https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B/resolve/$rev"
        mkdir -p "$src"
        for f in config.json model.safetensors.index.json model-00016-of-00016.safetensors; do
          [[ -f "$src/$f" ]] || curl -fL --retry 5 -C - -o "$src/$f" "$base/$f" || return 1
        done
        if [[ ! -f ".build/ornith-mtp-affine/model.safetensors.index.json" ]]; then
          echo "converting $name -> .build/ornith-mtp-affine"
          python3.13 tools/prepare_ornith_mtp.py --bits "$width" \
              --source-shard "$src/model-00016-of-00016.safetensors" \
              --source-config "$src/config.json" --source-index "$src/model.safetensors.index.json" \
              --output .build/ornith-mtp-affine || return 1
        fi
        echo "installing $name -> models/$dir"
        "$BIN" --input-snapshot .build/ornith-mtp-affine \
            --model-id ornith-1.5-35b-a3b-mtp-4bit --output "$MODELS/$dir"
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
