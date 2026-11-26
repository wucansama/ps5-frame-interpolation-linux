# RIFE (frame generation)

`IFNet_HDv3_v4_22_lite.py` and `warplayer.py` are the RIFE v4.22-lite network definition
(Practical-RIFE by hzwer, MIT; this copy via HolyWu/vs-rife, MIT, see `LICENSE`).
No weights are in this repository.

`export_onnx.py` downloads the v4.22-lite weights (HolyWu/vs-rife release asset, MIT) into
`~/.local/share/moonglide/rife/` and writes `rife_v4.22_lite_<W>x<H>.onnx` next to them:
inputs `img0`, `img1` (1x3xHxW, 0..1 RGB) and `t` (1x1x1x1), output `mid` (1x3xHxW).
`scripts/setup-rife.sh` (or `MoonGlide.AppImage setup-rife`) does all of this below in one go. By hand it needs PyTorch
(CPU is enough) and `onnx`, once:

```sh
python3 -m venv ~/.local/share/moonglide/rife-venv
~/.local/share/moonglide/rife-venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
~/.local/share/moonglide/rife-venv/bin/pip install onnx
~/.local/share/moonglide/rife-venv/bin/python tools/rife/export_onnx.py --size 1920x1088
```

TensorRT 10 is the user's install too (NVIDIA SLA; never in this repository). The pip
wheel carries the builder resources the engine build needs:

```sh
~/.local/share/moonglide/rife-venv/bin/pip install "tensorrt-cu12-libs==10.9.0.34"
```

moonglide (`--fg rife`, the default) looks for `libnvinfer.so.10` in `--rife-trt DIR`,
`$MOONGLIDE_TRT`, that venv's `tensorrt_libs/`, then `/usr/local/VideoFX/external/tensorrt/lib`
(runtime only: it can't build). The build headers are TensorRT 10.9's (`-DMOONGLIDE_TRT_INCLUDE`,
default the Video Effects SDK copy). It builds an fp16 engine from the ONNX on first use
(a minute or two) and caches it in `~/.cache/moonglide/rife/`. The work size is padded to
a multiple of 32 (1920x1080 -> the 1920x1088 model); `--rife-model FILE` for another export.
