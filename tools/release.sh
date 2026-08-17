#!/usr/bin/env bash
# Cut a release the way 3.6 was cut. Verifies, builds clean, packages binaries
# with a checksum, and publishes a GitHub Release from an existing tag.
#
#   tools/release.sh v3.7                  # dry run: verify, build, package, stop
#   tools/release.sh v3.7 --publish        # same, then create the Release
#   tools/release.sh v3.7 --publish --notes path/to/notes.md
#
# Dry run is the default on purpose: publishing is public and irreversible in
# the sense that watchers are notified immediately. Run it once without
# --publish, inspect the staged archive, then re-run with it.
#
# Two mistakes this script exists to prevent:
#
#   1. `gh` in a fork defaults to the PARENT repo. `gh release list` here shows
#      drumih/turbo-fieldfare, not this repo, and `gh release create` refuses
#      with a confusing message about an unpushed tag. Every gh call below pins
#      --repo.
#   2. An incremental `swift build` compiles nothing when the tree is unchanged,
#      so a warning gate over its output passes vacuously. The release build
#      always goes to a fresh scratch path.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO="${NVMAI_RELEASE_REPO:-Pummelchen/NVMAI}"
PRODUCTS=(NVMAIServer NVMAICLI NVMAIMac NVMAIDecodeService NVMAIRepack NVMAIBench)

die() { echo "error: $*" >&2; exit 1; }
step() { printf '\n== %s\n' "$*"; }

TAG="${1:-}"
[ -n "$TAG" ] || die "usage: tools/release.sh <tag> [--publish] [--notes <file>]"
shift
PUBLISH=0
NOTES=""
while [ $# -gt 0 ]; do
  case "$1" in
    --publish) PUBLISH=1; shift ;;
    --notes)   NOTES="${2:-}"; [ -n "$NOTES" ] || die "--notes needs a file"; shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done

VERSION="${TAG#v}"
STAGE_ROOT="${TMPDIR:-/tmp}/nvmai-release-$VERSION"
STAGE="$STAGE_ROOT/nvmai-$VERSION-macos-arm64"
ARCHIVE="$STAGE_ROOT/nvmai-$VERSION-macos-arm64.tar.gz"
SCRATCH="$STAGE_ROOT/build"

cd "$ROOT"

# --- preconditions ----------------------------------------------------------
step "preconditions"
[ -z "$(git status --porcelain)" ] || die "working tree is dirty; commit or stash first"
git rev-parse -q --verify "refs/tags/$TAG" >/dev/null || die "tag $TAG does not exist locally"
[ "$(git rev-parse "$TAG^{commit}")" = "$(git rev-parse HEAD)" ] \
  || die "HEAD is not $TAG; check out the tagged commit before releasing"
git ls-remote --tags origin 2>/dev/null | grep -q "refs/tags/$TAG$" \
  || die "$TAG is not pushed to origin; run: git push origin $TAG"
gh release view "$TAG" --repo "$REPO" >/dev/null 2>&1 \
  && die "a Release for $TAG already exists on $REPO"
echo "  tag $TAG at $(git rev-parse --short HEAD), tree clean, no existing Release"

# --- gates ------------------------------------------------------------------
step "gates"
"$SCRIPT_DIR/lint.sh" || die "tools/lint.sh failed"
swift test --no-parallel 2>&1 | tee "$STAGE_ROOT.testlog" 2>/dev/null | grep -E 'Test run with' \
  || true
grep -q 'Test run with .* passed' "$STAGE_ROOT.testlog" 2>/dev/null \
  || die "swift test did not report a passing run"

# The golden baseline is the only check that exercises real inference. Skip it
# only when no model is installed — never to make a mismatch go away.
if compgen -G "$ROOT/models/qwen3.6_35B_A3B_*Bit/verified-install.json" >/dev/null; then
  "$SCRIPT_DIR/golden-baseline.sh" --check 4 || die "golden baseline mismatch"
else
  echo "  no installed model; skipping golden baseline (state this in the notes)"
fi

# --- clean build ------------------------------------------------------------
step "clean release build"
rm -rf "$SCRATCH"
swift build -c release --scratch-path "$SCRATCH" 2>&1 | tee "$STAGE_ROOT.buildlog" | tail -1
grep -qE '^[^ ]+\.(swift|metal|c|h|m|mm):[0-9]+:[0-9]+: warning:' "$STAGE_ROOT.buildlog" \
  && die "release build emitted compiler warnings"
BIN="$SCRATCH/arm64-apple-macosx/release"
[ -x "$BIN/NVMAIServer" ] || die "build produced no NVMAIServer"

# --- stage ------------------------------------------------------------------
step "stage"
rm -rf "$STAGE" && mkdir -p "$STAGE"
for p in "${PRODUCTS[@]}"; do
  [ -x "$BIN/$p" ] || die "missing product: $p"
  cp "$BIN/$p" "$STAGE/"
done
# .bundle resources carry the Metal shader library; without them beside the
# executables the runtime cannot load its kernels.
find "$BIN" -maxdepth 1 -name '*.bundle' -exec cp -R {} "$STAGE/" \;
cp "$ROOT/LICENSE" "$ROOT/THIRD_PARTY_NOTICES.md" "$STAGE/"

cat > "$STAGE/README-binaries.txt" <<TXT
NVMAI $VERSION — prebuilt binaries (macOS, Apple Silicon / arm64)

Built from tag $TAG with: swift build -c release
Requires macOS 26+. Apple Silicon only; there is no x86_64 build.

Contents
  NVMAIServer          OpenAI-compatible local server (binds 127.0.0.1 only)
  NVMAICLI             one-shot prompt CLI
  NVMAIMac             Mac app
  NVMAIDecodeService   out-of-process decode service used by the Mac app
  NVMAIRepack          model installer / repacker
  NVMAIBench           benchmark driver
  *.bundle             Metal shader library and other runtime resources — keep
                       these next to the executables or the runtime cannot
                       load its kernels

These binaries are NOT code-signed or notarized. macOS Gatekeeper will refuse
them on first run. Either build from source, or clear the quarantine attribute
yourself after verifying the checksum published with this archive:

  xattr -dr com.apple.quarantine /path/to/nvmai-$VERSION-macos-arm64

No model weights are included. Install one with NVMAIRepack (the 4-bit download
is about 19.5 GB) as described in the README.
TXT

step "package"
( cd "$STAGE_ROOT" && tar czf "$ARCHIVE" "$(basename "$STAGE")" )
shasum -a 256 "$ARCHIVE" | sed "s|$STAGE_ROOT/||" > "$ARCHIVE.sha256"
SHA="$(awk '{print $1}' "$ARCHIVE.sha256")"
echo "  $(basename "$ARCHIVE")  $(wc -c < "$ARCHIVE" | tr -d ' ') bytes"
echo "  sha256 $SHA"

# --- publish ----------------------------------------------------------------
if [ "$PUBLISH" -ne 1 ]; then
  step "dry run complete"
  echo "  staged: $STAGE"
  echo "  re-run with --publish to create the Release on $REPO"
  exit 0
fi

[ -n "$NOTES" ] || die "--publish needs --notes <file> (see the previous release for the shape)"
[ -f "$NOTES" ] || die "notes file not found: $NOTES"
# Hard failure, not a warning. --publish rebuilds the archive, so a checksum
# copied from a dry run will not match what ships, and a release whose notes
# quote the wrong SHA-256 is worse than one quoting none: it tells a careful
# user their download is corrupt. 3.7 shipped this way for a few minutes.
if ! grep -q "$SHA" "$NOTES"; then
  die "the notes do not quote this archive's sha256 ($SHA); update them and re-run"
fi

step "publish"
gh release create "$TAG" "$ARCHIVE" "$ARCHIVE.sha256" \
  --repo "$REPO" \
  --title "NVMAI $VERSION" \
  --notes-file "$NOTES" \
  --latest || die "gh release create failed"
gh release view "$TAG" --repo "$REPO" --json url,assets \
  --jq '"  \(.url)\n  assets: \([.assets[].name] | join(", "))"'
