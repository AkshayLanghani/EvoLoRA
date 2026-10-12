#!/bin/bash
# EvoLoRA installer for macOS (Apple Silicon). Run it from Terminal:
#
#   curl -fsSL https://raw.githubusercontent.com/AkshayLanghani/EvoLoRA/main/public/download/install-mac.sh | bash
#
# Installs for the current user only, no sudo:
#   ~/Library/Application Support/EvoLoRA/app   EvoLoRA source (replaced on every run = update)
#   ~/Library/Application Support/EvoLoRA       models, tools, adapters, history (kept across updates)
# Sources: uv from astral.sh, EvoLoRA from GitHub. Models and training tools are fetched
# later from inside the app ("Install everything"), each from its official source.
#
# Options (environment variables):
#   EVOLORA_BRANCH=name     install a branch other than main
#   EVOLORA_HOME=/path      install somewhere else
#   EVOLORA_NO_START=1      install only; do not start EvoLoRA or add the desktop shortcut
set -euo pipefail

REPO="AkshayLanghani/EvoLoRA"
BRANCH="${EVOLORA_BRANCH:-main}"
ROOT="${EVOLORA_HOME:-$HOME/Library/Application Support/EvoLoRA}"
APP="$ROOT/app"

say() { printf '%s\n' "$*"; }
fail() { printf '\n  EvoLoRA setup failed: %s\n' "$*" >&2; exit 1; }

[ "$(uname -s)" = "Darwin" ] || fail "this installer is for macOS."
[ "$(uname -m)" = "arm64" ] || fail "EvoLoRA needs an Apple Silicon Mac (M1 or newer). Intel Macs are not supported."

say ""
say "  EvoLoRA setup"
say "  Installing into $ROOT"
say ""
mkdir -p "$ROOT"

# 1. uv: Astral's Python package manager (per-user install).
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if command -v uv >/dev/null 2>&1; then
  say "[1/4] uv is already installed"
else
  say "[1/4] Installing uv (astral.sh)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
fi
UV="$(command -v uv)" || fail "uv was installed but is not on PATH. Open a new Terminal window and run this again."

# 2. EvoLoRA source from GitHub.
say "[2/4] Downloading EvoLoRA..."
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
curl -fsSL "https://github.com/$REPO/archive/refs/heads/$BRANCH.tar.gz" -o "$TMP/src.tar.gz" \
  || fail "could not download EvoLoRA (branch $BRANCH). Check your internet connection."
mkdir "$TMP/src"
tar -xzf "$TMP/src.tar.gz" -C "$TMP/src" --strip-components 1
[ -f "$TMP/src/pyproject.toml" ] || fail "the download did not contain EvoLoRA."
# Stop a running copy before replacing its files.
pkill -f "evolora agent" >/dev/null 2>&1 || true
rm -rf "$APP"
mv "$TMP/src" "$APP"

# 3. Python environment (uv downloads Python 3.11 itself if needed).
say "[3/4] Setting up Python (the first time takes a few minutes)..."
"$UV" sync --project "$APP" --no-dev || fail "setting up the Python environment failed (uv sync)."

# 4. Launcher, then start EvoLoRA.
LAUNCHER="$ROOT/EvoLoRA.command"
cat > "$LAUNCHER" <<EOF
#!/bin/bash
# Starts EvoLoRA. Close this window (or press Ctrl+C) to stop it.
cd "$ROOT" || exit 1
export PATH="\$HOME/.local/bin:\$HOME/.cargo/bin:\$PATH"
exec "$UV" run --project "$APP" --no-dev evolora agent
EOF
chmod +x "$LAUNCHER"

if [ -n "${EVOLORA_NO_START:-}" ]; then
  say "[4/4] Installed. Start it with: \"$LAUNCHER\""
  exit 0
fi

# A shortcut made on this machine is not quarantined, so double-clicking it just works.
ln -sf "$LAUNCHER" "$HOME/Desktop/EvoLoRA.command" 2>/dev/null || true

say "[4/4] Starting EvoLoRA..."
open "$LAUNCHER"
say ""
say "  Done. EvoLoRA is starting in a new Terminal window and will open in your browser."
say "  Next time, double-click EvoLoRA on your desktop."
say "  On the EvoLoRA website, press Connect to use it from there (use Chrome if Safari cannot connect)."
