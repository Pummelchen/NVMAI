#!/usr/bin/env bash
# Install NVMAI for someone who has never used a terminal.
#
# Two ways to run it:
#
#   curl -fsSL https://raw.githubusercontent.com/Pummelchen/NVMAI/main/tools/install_nvmai.sh | bash
#   tools/install_nvmai.sh                      # from a clone, installs that clone
#
# It checks the Mac, gets the source, builds it, optionally downloads a model,
# and wraps the Mac app into a real double-clickable NVMAI.app in
# ~/Applications. The app is ad-hoc signed (codesign -s -), which needs no
# Apple Developer account; it is not notarized, which is why it is built on
# your machine rather than downloaded.
#
# Nothing here is destructive. It never deletes a model, never removes a
# directory, and never touches anything outside its own folders:
#   ~/NVMAI                 the checkout (a clone started by this script)
#   ~/Applications/NVMAI.app the app bundle
#   ~/.local/bin/nvmai      a command-line launcher
# Re-running it is safe: it updates an existing checkout instead of cloning
# a second time.
#
# Flags:
#   --yes, -y        answer yes to every question (unattended install)
#   --model NAME     model to install; see tools/install_models.sh --help
#                    (default: ornith15-8bit)
#   --no-model       build only; download no model
#   --no-app         do not create the .app bundle
#   --dir PATH       where to clone when there is no checkout (default ~/NVMAI)
#   --help, -h       this text
set -euo pipefail

REPO_URL="https://github.com/Pummelchen/NVMAI.git"
DEFAULT_MODEL="ornith15-8bit"
DEFAULT_DIR="$HOME/NVMAI"
APP_NAME="NVMAI"

say()  { printf '\033[1m%s\033[0m\n' "$*"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*" >&2; }
die()  { printf '\n\033[31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# Ask a yes/no question. With --yes, or when nothing can answer, take the
# fallback rather than hanging a piped install.
ASSUME_YES=0
ask() {
  local prompt="$1" fallback="${2:-no}" reply
  if (( ASSUME_YES )); then
    [[ "$fallback" == "yes" ]]
    return
  fi
  if [[ ! -t 0 ]]; then
    [[ "$fallback" == "yes" ]]
    return
  fi
  printf '%s [%s] ' "$prompt" "$([[ "$fallback" == yes ]] && echo 'Y/n' || echo 'y/N')"
  read -r reply || reply=""
  reply="${reply:-$fallback}"
  [[ "$reply" =~ ^[Yy] ]]
}

# --- flags -----------------------------------------------------------------
MODEL="$DEFAULT_MODEL"
INSTALL_MODEL=1
MAKE_APP=1
TARGET_DIR="$DEFAULT_DIR"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --yes|-y)    ASSUME_YES=1 ;;
    --model)     MODEL="${2:?--model needs a name}"; shift ;;
    --no-model)  INSTALL_MODEL=0 ;;
    --no-app)    MAKE_APP=0 ;;
    --dir)       TARGET_DIR="${2:?--dir needs a path}"; shift ;;
    --help|-h)   sed -n '2,/^set -euo pipefail/p' "$0" | sed 's/^# \{0,1\}//' | sed '$d'; exit 0 ;;
    *)           die "unknown option: $1 (try --help)" ;;
  esac
  shift
done

say "NVMAI installer"
echo "  This takes a while and mostly waits. You can stop it with Ctrl-C at any"
echo "  point; run it again later and it continues where it can."
echo

# --- 1) the machine ---------------------------------------------------------
say "1/6  Checking this Mac"

os_version="$(sw_vers -productVersion 2>/dev/null || echo 0)"
os_major="${os_version%%.*}"
arch="$(uname -m)"
if [[ "$arch" != "arm64" ]]; then
  die "NVMAI needs Apple Silicon (M1 or newer). This Mac reports $arch."
fi
ok "Apple Silicon ($arch), macOS $os_version"
if [[ "${os_major:-0}" -lt 26 ]]; then
  warn "NVMAI targets macOS 26 or later; this is $os_version."
  warn "The build may fail. If it does, updating macOS fixes it."
fi

free_kb="$(df -Pk "$HOME" | awk 'NR==2 {print $4}')"
free_gb=$(( free_kb / 1048576 ))
if [[ "$free_gb" -lt 45 ]]; then
  warn "Only about ${free_gb} GB free on this volume."
  warn "The build needs a few GB, and a 35B 4-bit model about 20 GB."
  ask "Continue anyway?" no || die "Stopped at your request. Free up space and re-run."
else
  ok "About ${free_gb} GB free"
fi

if [[ "$(pgrep -fl 'NVMAIServer|NVMAIMac|NVMAIDecodeService|NVMAICLI' 2>/dev/null | wc -l | tr -d ' ')" != "0" ]]; then
  warn "An NVMAI process is already running. Quit it before using NVMAI."
fi

# --- 2) the source ----------------------------------------------------------
say "2/6  Getting the source"

# Are we inside a checkout? Walk up from both this script and the directory
# the user is standing in, so `cd ~/NVMAI && tools/install_nvmai.sh` and
# `bash tools/install_nvmai.sh` both work.
find_checkout() {
  local start="$1" probe
  probe="$start"
  while [[ -n "$probe" && "$probe" != "/" ]]; do
    if [[ -f "$probe/Package.swift" ]]; then printf '%s' "$probe"; return 0; fi
    probe="$(dirname "$probe")"
  done
  return 1
}

REPO_ROOT=""
if ! REPO_ROOT="$(find_checkout "$(cd "$(dirname "$0")" && pwd)")"; then
  REPO_ROOT="$(find_checkout "$PWD" || true)"
fi

if [[ -n "$REPO_ROOT" ]]; then
  ok "Using the checkout at $REPO_ROOT"
else
  command -v git >/dev/null 2>&1 \
    || die "git is missing. Install Xcode (App Store), open it once, then re-run."
  if [[ -d "$TARGET_DIR/.git" ]]; then
    ok "Updating the existing checkout at $TARGET_DIR"
    git -C "$TARGET_DIR" pull --ff-only || warn "Could not update; using what is there."
    REPO_ROOT="$TARGET_DIR"
  else
    [[ -e "$TARGET_DIR" ]] && die "$TARGET_DIR already exists and is not a checkout. Move it or use --dir."
    echo "  Downloading NVMAI into $TARGET_DIR ..."
    git clone --depth 1 "$REPO_URL" "$TARGET_DIR" || die "Could not download NVMAI. Check your connection."
    REPO_ROOT="$TARGET_DIR"
  fi
  ok "Source ready"
fi
cd "$REPO_ROOT"

# --- 3) the toolchain -------------------------------------------------------
say "3/6  Checking the Swift toolchain"

if ! command -v swift >/dev/null 2>&1; then
  warn "Swift is not installed yet."
  echo "  Pressing Enter opens the installer for Apple's command-line tools."
  if ask "Install them now?" yes; then
    xcode-select --install 2>/dev/null || true
    echo
    echo "  A dialog should appear. Accept it, wait for it to finish (it can take"
    echo "  several minutes), then run this installer again."
  fi
  exit 1
fi

swift_line="$(swift --version 2>&1 | head -1)"
swift_ver="$(printf '%s' "$swift_line" | grep -oE 'Swift version [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -1)"
if [[ -z "$swift_ver" ]]; then
  warn "Could not read a Swift version from: $swift_line"
  ask "Try the build anyway?" yes || exit 1
elif (( $(printf '%s' "$swift_ver" | cut -d. -f1) < 6 )) \
  || { [[ "$(printf '%s' "$swift_ver" | cut -d. -f1)" == "6" ]] \
       && (( $(printf '%s' "$swift_ver" | cut -d. -f2) < 3 )); }; then
  die "NVMAI needs Swift 6.3 or later; this Mac has $swift_ver.
     Update Xcode from the App Store (or set it with xcode-select), then re-run."
else
  ok "Swift $swift_ver"
fi

# --- 4) build ---------------------------------------------------------------
say "4/6  Building (this is the slow part)"

if ! swift build -c release; then
  die "The build failed. The last few lines explain why.
     Copy the whole message to https://nvmai.discourse.group/ and someone will help."
fi
ok "Build complete"

# --- 5) a model -------------------------------------------------------------
say "5/6  Model"

installed_any() {
  local d
  for d in "$REPO_ROOT"/models/*/manifest.json; do
    [[ -f "$d" ]] && return 0
  done
  return 1
}

if (( ! INSTALL_MODEL )); then
  ok "Skipped (--no-model)"
elif installed_any; then
  ok "A model is already installed under models/"
  echo "     Install another any time:  tools/install_models.sh $MODEL"
else
  echo "  No model is installed yet. NVMAI needs one to run."
  echo "  The recommended starting model is Ornith 1.5 35B-A3B at 8-bit,"
  echo "  about 37 GB installed. The 4-bit version is about 20 GB and faster"
  echo "  to download if that is a lot."
  if ask "Download the 37 GB model now?" yes; then
    if ! tools/install_models.sh "$MODEL"; then
      warn "The model download did not finish."
      echo "     Re-run this installer to continue, or start it directly:"
      echo "       tools/install_models.sh $MODEL"
    else
      ok "Model installed"
    fi
  else
    echo "  Fine — NVMAI is built but will have nothing to load until you run:"
    echo "       tools/install_models.sh $MODEL"
  fi
fi

# --- 6) the app -------------------------------------------------------------
say "6/6  Installing the Mac app"

if (( ! MAKE_APP )); then
  ok "Skipped (--no-app)"
else
  BIN_DIR="$REPO_ROOT/.build/release"
  APP_PATH="$HOME/Applications/$APP_NAME.app"

  if [[ ! -x "$BIN_DIR/NVMAIMac" ]]; then
    warn "NVMAIMac was not built; skipping the app bundle."
  else
    rm -rf "$APP_PATH"
    mkdir -p "$APP_PATH/Contents/MacOS" "$APP_PATH/Contents/Resources"

    # The app finds its model by walking up to the checkout, so the bundle
    # carries a launcher that says which checkout this bundle belongs to.
    # Without it, an app in ~/Applications would look in Application Support.
    cat > "$APP_PATH/Contents/MacOS/$APP_NAME" <<LAUNCHER
#!/bin/sh
# Installed by tools/install_nvmai.sh — runs NVMAI from its checkout so the
# app finds the model it was installed with.
NVMAI_ROOT="$REPO_ROOT"
BIN="\$NVMAI_ROOT/.build/release/NVMAIMac"
if [ ! -x "\$BIN" ]; then
  printf 'NVMAI is not built at %s.\\nRe-run the installer:\\n  %s/tools/install_nvmai.sh\\n' "\$NVMAI_ROOT" "\$NVMAI_ROOT"
  read -r _ || true
  exit 1
fi
export TURBO_FIELDFARE_MODEL="\${TURBO_FIELDFARE_MODEL:-ornith15-8bit}"
cd "\$NVMAI_ROOT" || exit 1
exec "\$BIN" "\$@"
LAUNCHER
    chmod +x "$APP_PATH/Contents/MacOS/$APP_NAME"

    # SwiftPM writes resources to sibling *.bundle directories; Bundle.module
    # finds them beside the executable or in Contents/Resources.
    shopt -s nullglob
    for bundle in "$BIN_DIR"/NVMAI_*.bundle; do
      cp -R "$bundle" "$APP_PATH/Contents/Resources/"
    done
    shopt -u nullglob

    # Turn the shipped PNG into an .icns so the Dock shows a real icon.
    icon_png=""
    for candidate in "$BIN_DIR"/NVMAI_NVMAIMac.bundle/nvmai-app-icon.png \
                     "$REPO_ROOT"/sources/NVMAIApp/Mac/Resources/nvmai-app-icon.png; do
      [[ -f "$candidate" ]] && { icon_png="$candidate"; break; }
    done
    if [[ -n "$icon_png" ]] && command -v sips >/dev/null 2>&1 \
       && command -v iconutil >/dev/null 2>&1; then
      iconset="$(mktemp -d)/icon.iconset"
      mkdir -p "$iconset"
      for size in 16 32 128 256 512; do
        sips -z $size $size "$icon_png" --out "$iconset/icon_${size}x${size}.png" >/dev/null 2>&1 || true
        sips -z $((size * 2)) $((size * 2)) "$icon_png" --out "$iconset/icon_${size}x${size}@2x.png" >/dev/null 2>&1 || true
      done
      iconutil -c icns "$iconset" -o "$APP_PATH/Contents/Resources/$APP_NAME.icns" >/dev/null 2>&1 || true
      rm -rf "$iconset"
    fi

    cat > "$APP_PATH/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>NVMAI</string>
  <key>CFBundleDisplayName</key><string>NVMAI</string>
  <key>CFBundleIdentifier</key><string>local.nvmai.app</string>
  <key>CFBundleVersion</key><string>5.1</string>
  <key>CFBundleShortVersionString</key><string>5.1</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleExecutable</key><string>NVMAI</string>
  <key>CFBundleIconFile</key><string>NVMAI</string>
  <key>LSMinimumSystemVersion</key><string>26.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>LSApplicationCategoryType</key><string>public.app-category.developer-tools</string>
</dict>
</plist>
PLIST

    # Ad-hoc signature: enough for macOS to run a locally built bundle, and it
    # needs no Apple Developer account. Failure is not fatal.
    if command -v codesign >/dev/null 2>&1; then
      codesign --force --deep --sign - "$APP_PATH" >/dev/null 2>&1 \
        && ok "App signed (ad-hoc, local)" \
        || warn "Could not sign the app; it should still run from your machine."
    fi

    ok "App installed at $APP_PATH"
    echo "     Open it from Applications, or:  open \"$APP_PATH\""
  fi

  # A PATH launcher for the terminal-minded, and for the app when it is not
  # wanted. It only stops a server that this launcher started.
  mkdir -p "$HOME/.local/bin"
  cat > "$HOME/.local/bin/nvmai" <<RUNNER
#!/bin/sh
# Installed by tools/install_nvmai.sh. Starts the NVMAI server from its checkout.
exec "$REPO_ROOT/tools/server_launcher.sh" "\$@"
RUNNER
  chmod +x "$HOME/.local/bin/nvmai"
  ok "Command-line launcher: ~/.local/bin/nvmai"
  case ":$PATH:" in
    *":$HOME/.local/bin:"*) ;;
    *) echo "     Add it to your PATH to use the 'nvmai' command:"
       echo "       echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.zshrc" ;;
  esac
fi

# --- done -------------------------------------------------------------------
echo
say "Done."
echo
echo "  Start NVMAI either way:"
echo "    • open the NVMAI app in Applications"
echo "    • or, in a terminal:  ~/.local/bin/nvmai"
echo
echo "  Then point a client at http://127.0.0.1:8080/v1 (any API key)."
echo "  Keep the window open while you use it; NVMAI runs one model at a time."
echo
echo "  New to this? Start here:"
echo "    https://github.com/Pummelchen/NVMAI/blob/main/docs/site/01-what-is-nvmai.md"
echo "  Questions: https://nvmai.discourse.group/"
