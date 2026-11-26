#!/usr/bin/env bash
# One-shot, user-local setup of RIFE frame generation for MoonGlide (no root, safe to re-run).
#
#   scripts/setup-rife.sh                 # from a checkout
#   ./MoonGlide-x86_64.AppImage setup-rife   # from the AppImage
#
# What it does (everything under $MOONGLIDE_HOME, default ~/.local/share/moonglide):
#   1. python venv "rife-venv" with the pinned packages below
#        - tensorrt-cu12-libs  NVIDIA TensorRT 10 runtime + builder resources (NVIDIA licence;
#                              downloaded from PyPI by pip on YOUR machine, never redistributed here)
#        - torch (CPU) + onnx  only to export the model once
#   2. the RIFE v4.22-lite ONNX model (MIT weights, downloaded from the HolyWu/vs-rife release)
#        -> rife/rife_v4.22_lite_1920x1088.onnx   (more sizes: --size WxH, multiples of 32)
# MoonGlide finds both by itself on the next start; the TensorRT engine is built on first use
# (a minute or two) and cached in ~/.cache/moonglide/rife.
#
# Options: --size WxH (repeatable, default 1920x1088)   --python PYTHON   --keep-torch
# Environment: MOONGLIDE_HOME, PIP_INDEX_URL / TORCH_INDEX (CPU wheels, default download.pytorch.org/whl/cpu)
# SPDX-License-Identifier: MIT
set -euo pipefail

TRT_PIN="tensorrt-cu12-libs==10.9.0.34"   # TensorRT 10.9: what src/rife.cpp's headers are
TORCH_PIN="torch==2.7.1"                  # any recent CPU torch exports the model; pinned for repeatability
ONNX_PIN="onnx==1.18.0"
NUMPY_PIN="numpy<2.3"                     # keeps the pinned onnx / torch pair happy
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"

HOME_DIR="${MOONGLIDE_HOME:-$HOME/.local/share/moonglide}"   # where src/rife.cpp looks
VENV="$HOME_DIR/rife-venv"
MODEL_DIR="$HOME_DIR/rife"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS="$HERE/../tools/rife"                # checkout: scripts/../tools/rife; AppImage: share/moonglide/tools/rife

PY="${PYTHON:-python3}"
SIZES=()
KEEP_TORCH=0
while [ $# -gt 0 ]; do
	case "$1" in
	--size) SIZES+=("$2"); shift 2 ;;
	--python) PY="$2"; shift 2 ;;
	--keep-torch) KEEP_TORCH=1; shift ;;
	-h | --help) sed -n 2,19p "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
	*) echo "unknown option: $1" >&2; exit 2 ;;
	esac
done
[ ${#SIZES[@]} -gt 0 ] || SIZES=(1920x1088)

[ -f "$TOOLS/export_onnx.py" ] || { echo "export_onnx.py not found in $TOOLS" >&2; exit 1; }
command -v "$PY" > /dev/null || { echo "python3 not found (install python3 and python3-venv)" >&2; exit 1; }
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' || { echo "Python >= 3.9 needed" >&2; exit 1; }

echo "== MoonGlide RIFE setup in $HOME_DIR"
if [ ! -x "$VENV/bin/python" ]; then
	"$PY" -m venv "$VENV" || { echo "could not create a venv: on Debian/Ubuntu install python3-venv" >&2; exit 1; }
fi
PIP=("$VENV/bin/python" -m pip --disable-pip-version-check)

echo "== TensorRT ($TRT_PIN)"
"${PIP[@]}" install --quiet "$TRT_PIN"

# The model only needs exporting once; skip torch when every requested size exists already.
need=()
for s in "${SIZES[@]}"; do [ -f "$MODEL_DIR/rife_v4.22_lite_$s.onnx" ] || need+=("$s"); done
if [ ${#need[@]} -gt 0 ]; then
	echo "== exporting the RIFE v4.22-lite model (${need[*]})"
	"${PIP[@]}" install --quiet --index-url "$TORCH_INDEX" "$TORCH_PIN"
	"${PIP[@]}" install --quiet "$ONNX_PIN" "$NUMPY_PIN"
	for s in "${need[@]}"; do
		"$VENV/bin/python" "$TOOLS/export_onnx.py" --size "$s" --dir "$MODEL_DIR"
	done
	if [ "$KEEP_TORCH" = 0 ]; then
		echo "== removing torch (not needed at run time)"
		"${PIP[@]}" uninstall --quiet -y torch onnx numpy sympy networkx jinja2 mpmath fsspec filelock typing_extensions MarkupSafe 2> /dev/null || true
	fi
else
	echo "== model already there"
fi

lib="$(ls -d "$VENV"/lib/python3*/site-packages/tensorrt_libs 2> /dev/null | head -1)"
[ -f "$lib/libnvinfer.so.10" ] || { echo "libnvinfer.so.10 missing in $lib" >&2; exit 1; }
echo "== done"
echo "TensorRT: $lib"
for s in "${SIZES[@]}"; do ls -1 "$MODEL_DIR/rife_v4.22_lite_$s.onnx"; done
echo "Start MoonGlide; frame generation 'rife' is the default and builds its engine on first use."
