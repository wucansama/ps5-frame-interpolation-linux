#!/usr/bin/env bash
# Adds MoonGlide to the desktop's application menu (per user; no root needed).
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
apps="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
icons="${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor/scalable/apps"
mkdir -p "$apps" "$icons"
install -m 644 "$root/assets/moonglide.svg" "$icons/moonglide.svg"
cat > "$apps/moonglide.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=MoonGlide
GenericName=Capture enhancer
Comment=Denoise, upscale and frame-generate a capture card picture
Exec=$root/scripts/moonglide-app
Icon=moonglide
Terminal=false
Categories=AudioVideo;Video;Game;
Keywords=capture;upscale;denoise;frame generation;
StartupWMClass=moonglide
StartupNotify=false
DESKTOP
command -v update-desktop-database > /dev/null && update-desktop-database "$apps" 2> /dev/null || true
command -v gtk-update-icon-cache > /dev/null && gtk-update-icon-cache -q "${icons%/scalable/apps}" 2> /dev/null || true
echo "installed: $apps/moonglide.desktop"
