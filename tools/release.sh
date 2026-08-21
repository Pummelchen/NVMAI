#!/usr/bin/env bash
# Verify, clean-build, package, and optionally publish an NVMAI release.
# with a checksum, and publishes a GitHub Release from an existing tag.
#
#   tools/release.sh v4.0                  # dry run: verify, build, package, stop
#   tools/release.sh v4.0 --publish        # same, then create the Release
#   tools/release.sh v4.0 --publish --notes path/to/notes.md
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
STAGE_ROOT="$ROOT/.build/releases/nvmai-release-$VERSION"
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

rm -rf "$STAGE_ROOT"
mkdir -p "$STAGE_ROOT"

# --- gates ------------------------------------------------------------------
step "gates"
"$SCRIPT_DIR/lint.sh" || die "tools/lint.sh failed"
swift test --no-parallel 2>&1 | tee "$STAGE_ROOT.testlog" 2>/dev/null | grep -E 'Test run with' \
  || true
grep -q 'Test run with .* passed' "$STAGE_ROOT.testlog" 2>/dev/null \
  || die "swift test did not report a passing run"

# The golden baseline is the only check that exercises real inference. Skip it
# only when no model is installed — never to make a mismatch go away.
if [ -f "$ROOT/models/ornith-1.5_35B_A3B_8Bit/verified-install.json" ]; then
  "$SCRIPT_DIR/golden-baseline.sh" --check 8 || die "golden baseline mismatch"
elif compgen -G "$ROOT/models/*/verified-install.json" >/dev/null; then
  die "Ornith 1.5 8-bit baseline model is not installed"
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

No model weights are included. NVMAIRepack defaults to Ornith 1.5 8-bit (about
36.9 GB); 4-bit remains available explicitly. The runtime defaults to standard
answers with thinking off, as described in the README and Wiki.
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

# A release whose notes quote the wrong SHA-256 is worse than one quoting none:
# it tells a careful user their download is corrupt. 3.7 shipped that way for a
# few minutes, which is why this is enforced.
#
# But the notes cannot hard-code the digest either. --publish rebuilds from
# scratch, so the binaries carry fresh mtimes and the archive hashes differently
# than any dry run -- the value is unknowable when the notes are written. So the
# notes carry the literal SHA256_PENDING and it is filled in here, which makes
# the invariant hold by construction instead of by a check nothing can satisfy.
RENDERED_NOTES="$STAGE_ROOT/notes-rendered.md"
if grep -q 'SHA256_PENDING' "$NOTES"; then
  sed "s/SHA256_PENDING/$SHA/g" "$NOTES" > "$RENDERED_NOTES" \
    || die "failed to render notes"
  echo "  filled SHA256_PENDING with $SHA"
else
  cp "$NOTES" "$RENDERED_NOTES"
fi
if ! grep -q "$SHA" "$RENDERED_NOTES"; then
  die "the notes neither contain SHA256_PENDING nor quote this archive's sha256 ($SHA)"
fi

step "publish"
gh release create "$TAG" "$ARCHIVE" "$ARCHIVE.sha256" \
  --repo "$REPO" \
  --title "NVMAI $VERSION" \
  --notes-file "$RENDERED_NOTES" \
  --latest || die "gh release create failed"
gh release view "$TAG" --repo "$REPO" --json url,assets \
  --jq '"  \(.url)\n  assets: \([.assets[].name] | join(", "))"'
