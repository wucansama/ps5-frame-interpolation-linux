#!/usr/bin/env bash
# Build MoonGlide-x86_64.AppImage from a clean checkout.
#
#   GLSLANG=/path/to/glslang      glslang standalone >= 16 (required; not in distro repos)
#   TAP=/path/to/moonglide-chiaki-tap   prebuilt tap (AGPL, built from its own repo), or
#   TAP_DIR=/path/to/moonglide-chiaki-tap checkout: its build.sh is run to produce it
#   CMAKE_EXTRA="-D..."           extra cmake options, e.g. -DMOONGLIDE_TRT_INCLUDE=DIR -DNVVFX_CUDA_INCLUDE=DIR to compile
#                                 the RIFE backend in (TensorRT 10.9 + CUDA headers; build time only, see CI)
#   WORK=dir                      scratch dir (default: ./appimage-work); OUT=dir for the result
# Needs: cmake ninja pkg-config libx11-dev libjpeg-dev libavcodec-dev libavutil-dev, curl, file.
# linuxdeploy and appimagetool are downloaded into $WORK (nothing is installed system-wide).
# SPDX-License-Identifier: MIT
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(cd "$HERE/../.." && pwd)"
WORK="${WORK:-$PWD/appimage-work}"
OUT="${OUT:-$PWD}"
GLSLANG="${GLSLANG:-glslang}"
mkdir -p "$WORK" "$OUT"
WORK="$(cd "$WORK" && pwd)"; OUT="$(cd "$OUT" && pwd)"
BUILD="$WORK/build"
APPDIR="$WORK/AppDir"

# --- the tap -------------------------------------------------------------------------------
if [ -z "${TAP:-}" ]; then
	[ -n "${TAP_DIR:-}" ] || { echo "set TAP=<binary> or TAP_DIR=<moonglide-chiaki-tap checkout>" >&2; exit 1; }
	MOONGLIDE_TAP_CACHE="${MOONGLIDE_TAP_CACHE:-$WORK/tap-cache}" MOONGLIDE_TAP_DEST="$WORK/tap-bin" \
		"$TAP_DIR/build.sh"
	TAP="$WORK/tap-bin/moonglide-chiaki-tap"
fi
[ -x "$TAP" ] || { echo "tap not found: $TAP" >&2; exit 1; }

# --- tools ----------------------------------------------------------------------------------
TOOLS="$WORK/tools"; mkdir -p "$TOOLS"
fetch() { [ -x "$TOOLS/$1" ] || { curl -fsSL -o "$TOOLS/$1" "$2" && chmod +x "$TOOLS/$1"; }; }
fetch linuxdeploy https://github.com/linuxdeploy/linuxdeploy/releases/download/1-alpha-20250213-2/linuxdeploy-x86_64.AppImage
fetch appimagetool https://github.com/AppImage/appimagetool/releases/download/1.9.0/appimagetool-x86_64.AppImage
export APPIMAGE_EXTRACT_AND_RUN=1   # no FUSE needed on build machines

# --- build the app ----------------------------------------------------------------------------
cmake -S "$SRC" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGLSLANG="$GLSLANG" ${CMAKE_EXTRA:-}
cmake --build "$BUILD"

# --- stage the AppDir ---------------------------------------------------------------------------
rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin" "$APPDIR/usr/share/moonglide/nr" "$APPDIR/usr/share/applications" \
	"$APPDIR/usr/share/icons/hicolor/scalable/apps"
install -m 755 "$BUILD/moonglide" "$BUILD/moonglide-rp-fake" "$APPDIR/usr/bin/"
install -m 755 "$TAP" "$APPDIR/usr/bin/moonglide-chiaki-tap"
cp -r "$BUILD/shaders" "$APPDIR/usr/share/moonglide/shaders"
cp -r "$BUILD/nr/shaders" "$BUILD/nr/ptx" "$APPDIR/usr/share/moonglide/nr/"
rm -f "$APPDIR/usr/share/moonglide/nr/ptx/.stamp"
# setup-rife (AppRun: `MoonGlide.AppImage setup-rife`): the script and the model exporter, no weights.
mkdir -p "$APPDIR/usr/share/moonglide/scripts" "$APPDIR/usr/share/moonglide/tools/rife"
install -m 755 "$SRC/scripts/setup-rife.sh" "$APPDIR/usr/share/moonglide/scripts/"
cp "$SRC"/tools/rife/*.py "$SRC/tools/rife/LICENSE" "$SRC/tools/rife/README.md" "$APPDIR/usr/share/moonglide/tools/rife/"
install -m 644 "$HERE/moonglide.desktop" "$APPDIR/usr/share/applications/moonglide.desktop"
install -m 644 "$SRC/assets/moonglide.svg" "$APPDIR/usr/share/icons/hicolor/scalable/apps/moonglide.svg"
# licences: the tap is AGPL and stays a separate program; its source is public.
mkdir -p "$APPDIR/usr/share/doc/moonglide"
cp "$SRC/LICENSE" "$SRC/NOTICE" "$APPDIR/usr/share/doc/moonglide/"
[ -f "${TAP_DIR:-/nonexistent}/LICENSE" ] && cp "$TAP_DIR/LICENSE" "$APPDIR/usr/share/doc/moonglide/LICENSE.moonglide-chiaki-tap"

# --- bundle shared libs (linuxdeploy), then drop what must come from the system -----------------------
"$TOOLS/linuxdeploy" --appdir "$APPDIR" \
	--executable "$APPDIR/usr/bin/moonglide" \
	--executable "$APPDIR/usr/bin/moonglide-rp-fake" \
	--executable "$APPDIR/usr/bin/moonglide-chiaki-tap" \
	--desktop-file "$HERE/moonglide.desktop" --icon-file "$SRC/assets/moonglide.svg" \
	--custom-apprun "$HERE/AppRun"
# Never ship the Vulkan loader, GPU drivers or anything from NVIDIA.
find "$APPDIR/usr/lib" \( -name 'libvulkan*' -o -name 'libcuda*' -o -name 'libnv*' -o -name 'libnpp*' \
	-o -name 'libGL.so*' -o -name 'libGLX*' -o -name 'libEGL*' -o -name 'libdrm*' \) -print -delete
if find "$APPDIR" -iname '*nvidia*' -o -iname 'libcuda*' -o -iname 'libnvinfer*' -o -iname 'libvulkan*' | grep -q .; then
	echo "forbidden library in the AppDir" >&2; exit 1
fi

# --- pack ---------------------------------------------------------------------------------------------
ARCH=x86_64 "$TOOLS/appimagetool" --no-appstream "$APPDIR" "$OUT/MoonGlide-x86_64.AppImage"
ls -la "$OUT/MoonGlide-x86_64.AppImage"
