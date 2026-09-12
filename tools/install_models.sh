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

# The converters need a Python with numpy/ml_dtypes/safetensors at 3.10 or
# newer. Resolved by capability rather than by name: `python3` is 3.9 on a
# stock macOS and lacks the packages, while a pinned `python3.13` fails on a
# machine whose stack lives under another version. See tools/lib/python.sh.
# shellcheck source=tools/lib/python.sh
source "$ROOT/tools/lib/python.sh"

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
  "katcoder|kat-coder-v2.5_35B_A3B_4Bit|4|convert_qwen35moe"
  "katcoder-8bit|kat-coder-v2.5_35B_A3B_8Bit|8|convert_qwen35moe"
  "agentworld|qwen-agentworld_35B_A3B_4Bit|4|convert_qwen35moe"
  "agentworld-8bit|qwen-agentworld_35B_A3B_8Bit|8|convert_qwen35moe"
  # The dense Qwen 3.5 models. Small enough to run on the CPU, and the only
  # installs that do: the 2B beside a big GPU model, the 9B on its own.
  "qwen35-2b|qwen3.5_2B_4Bit|4|convert_qwen35"
  "qwen35-2b-8bit|qwen3.5_2B_8Bit|8|convert_qwen35"
  "qwen35-4b|qwen3.5_4B_4Bit|4|convert_qwen35"
  "qwen35-4b-8bit|qwen3.5_4B_8Bit|8|convert_qwen35"
  "qwen35-9b|qwen3.5_9B_4Bit|4|convert_qwen35"
  "qwen35-9b-8bit|qwen3.5_9B_8Bit|8|convert_qwen35"
)

usage() {
  cat <<'USAGE'
Coverage

  Ornith 1.5 35B-A3B        4-bit, 8-bit, MTP draft
  Qwen 3.6 35B-A3B          4-bit, 8-bit, MTP draft
  Qwen3.8-Flash-Next        4-bit, 8-bit, MTP draft
  Qwen-AgentWorld 35B-A3B   4-bit, 8-bit
  KAT-Coder-V2.5-Dev 35B-A3B 4-bit, 8-bit
  Qwen 3.5 2B / 4B / 9B     4-bit, 8-bit (CPU models)

Sources

  Every install is built from the model's own bf16 release, quantized here
  (group-64 affine) by the tools in tools/ and imported by NVMAIRepack.
  Third-party quantizations are deliberately not used: their group sizes,
  widths and norm conventions are theirs, and here the router, the
  shared-expert gate, the DeltaNet gating projections and every norm stay
  at bf16 in both widths.

  convert_qwen35moe   tools/prepare_agentworld.py --model {ornith15,qwen36,agentworld,katcoder}
                      One ~70 GB download yields both widths.
  convert_qwen35      tools/prepare_qwen35.py --size {2b,4b,9b}, then
                      NVMAIRepack --input-snapshot. One fetch yields both
                      widths. These are the dense models, and the only ones
                      the CPU engine runs; the 9B is the vision-language
                      build, converted text-only like the others.
                      tools/repack_dense.sh re-runs the repack and the
                      equivalence check against a retained snapshot.
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
  # Every conversion path runs a Python converter, so resolve the interpreter
  # once, here, rather than emitting a raw "command not found" per call.
  local python
  python="$(nvmai_resolve_python)" || return 1
  for row in "${CATALOGUE[@]}"; do
    IFS='|' read -r name dir width source <<<"$row"
    [[ "$name" == "$want" ]] || continue
    found=1
    if [[ -d "$MODELS/$dir" ]]; then
      # A dense install from before this project repacked them is an affine
      # snapshot: the directory exists but carries no manifest and no receipt.
      # Returning here would leave it that way forever, because "the directory
      # exists" is what this check means. Fall through and let the branch
      # repack it, which is the only way it becomes verifiable in place.
      if [[ "$source" == convert_qwen35 && ! -f "$MODELS/$dir/manifest.json" ]]; then
        echo "$name is an affine snapshot; repacking it as .gturbo"
      else
        echo "$name is already installed at models/$dir"
        return 0
      fi
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
          "$python" tools/prepare_qwen38.py --bits "$width" \
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
          "$python" tools/prepare_qwen38_mtp.py --bits "$width" \
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
          "$python" tools/prepare_agentworld.py --model qwen36 --draft-head --bits "$width" \
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
          katcoder)   model_id="kat-coder-v2.5" ;;
          qwen36)     model_id="qwen3.6-35b-a3b" ;;
          ornith15)   model_id="ornith-1.5-35b-a3b" ;;
        esac
        if [[ ! -f ".build/${preset}-affine-${width}bit/model.safetensors.index.json" ]]; then
          echo "converting $preset -> .build/${preset}-affine-{4,8}bit"
          "$python" tools/prepare_agentworld.py --model "$preset" --bits 4 8 \
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
        # A file is trusted only at the size the server reports; a partial
        # left by an interrupted run is resumed, not skipped.
        for f in config.json model.safetensors.index.json model-00016-of-00016.safetensors; do
          local want have
          want=$(curl -sIL --retry 5 "$base/$f" | grep -i '^content-length:' | tail -1 | tr -dc '0-9')
          have=$(stat -f %z "$src/$f" 2>/dev/null || echo 0)
          if [[ -z "$want" || "$have" != "$want" ]]; then
            curl -fL --retry 20 --retry-delay 15 --retry-all-errors -C - -o "$src/$f" "$base/$f" || return 1
          fi
        done
        if [[ ! -f ".build/ornith-mtp-affine/model.safetensors.index.json" ]]; then
          echo "converting $name -> .build/ornith-mtp-affine"
          "$python" tools/prepare_ornith_mtp.py --bits "$width" \
              --source-shard "$src/model-00016-of-00016.safetensors" \
              --source-config "$src/config.json" --source-index "$src/model.safetensors.index.json" \
              --output .build/ornith-mtp-affine || return 1
        fi
        echo "installing $name -> models/$dir"
        "$BIN" --input-snapshot .build/ornith-mtp-affine \
            --model-id ornith-1.5-35b-a3b-mtp-4bit --output "$MODELS/$dir"
        ;;
      convert_qwen35)
        # The dense Qwen 3.5 models, from Qwen's own bf16 release. One
        # download yields both widths, so the other width installs without a
        # second fetch: only the converted *staging* directory is per-width,
        # the source shards in .build/<preset>-shards are shared.
        #
        # Convert, then repack, then drop the staging directory. The snapshot
        # the converter writes is an intermediate, not the install: every
        # model this project serves is a .gturbo directory with a manifest and
        # a path-bound receipt, and a snapshot has neither. Keeping the
        # intermediate would double the disk for a 9B and buy nothing, since
        # it is reproducible from the cached shards.
        #
        # The receipt is bound to the absolute output path, so the repack must
        # write straight into models/. Nothing here may move the directory
        # afterwards.
        local preset="${name%-8bit}" size_key model_id
        case "$preset" in
          qwen35-2b) size_key=2b; model_id="qwen3.5-2b" ;;
          qwen35-4b) size_key=4b; model_id="qwen3.5-4b" ;;
          qwen35-9b) size_key=9b; model_id="qwen3.5-9b" ;;
          *) echo "unknown Qwen 3.5 size: $preset" >&2; return 2 ;;
        esac
        [[ -x "$BIN" ]] || { echo "build NVMAIRepack first: swift build -c release" >&2; return 1; }
        # The 9B checkpoint is the vision-language build; the converter drops
        # the model.visual.* tower and writes the text model, so the install
        # is text-only like every other model here.
        local stage=".build/qwen35-${size_key}-affine-${width}bit"
        if [[ ! -f "$stage/config.json" ]]; then
          if [[ -f "$MODELS/$dir/config.json" ]]; then
            # A legacy snapshot: move it into the converter's staging area
            # rather than converting it again. It is the same bytes the
            # converter would fetch and quantize, and it is already here.
            echo "staging the existing snapshot -> $stage"
            rm -rf "$stage"
            mv "$MODELS/$dir" "$stage"
          else
            echo "converting Qwen 3.5 ${size_key} ${width}-bit -> $stage"
            "$python" tools/prepare_qwen35.py --size "$size_key" --bits "$width" \
                --output "$stage" \
                --work ".build/${preset}-shards" || return 1
          fi
        fi
        # The receipt is bound to the absolute output path below, and it is
        # written by the repack, so the destination must be empty first and
        # must never be moved afterwards.
        echo "repacking $stage -> models/$dir"
        rm -rf "$MODELS/$dir"
        "$BIN" --input-snapshot "$stage" --model-id "$model_id" \
            --output "$MODELS/$dir" || return 1
        "$BIN" --verify-install --input-gturbo "$MODELS/$dir" || return 1
        # The staging snapshot is an intermediate and is reproducible from the
        # cached shards, so it does not outlive the install. Use
        # tools/repack_dense.sh instead if you want it kept for the
        # equivalence gate.
        rm -rf "$stage"
        echo "installed $name -> models/$dir (.gturbo)"
        ;;
      unsupported)
        # No catalogue row uses this today, and the message it used to carry was
        # false: it said 8-bit Qwen3.8-Flash-Next "cannot execute -- the runtime
        # refuses it at load", which predates `SlotGEMV` giving the
        # hyper-connection, PLE and QSA-indexer projections both a 4- and an
        # 8-bit path. `validateFamilyQuantSupport` now refuses only a width
        # neither GEMV implements, and the indexer's bf16 prefill branch exists.
        # Kept as a generic refusal rather than deleted, so wiring a genuinely
        # unsupported row here later fails loudly instead of falling through this
        # switch and reporting a successful install it never performed.
        cat <<EOF
$name cannot be run by this installer.

A row reaches this branch only when it is known to build an install the runtime
cannot execute. Check --help for what that model would require; if nothing
explains it, this message is stale and the row should be fixed rather than
shipped.
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
