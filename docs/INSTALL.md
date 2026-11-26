# Install

## AppImage

```sh
chmod +x MoonGlide-x86_64.AppImage
./MoonGlide-x86_64.AppImage --ui
```

- No FUSE 2: add `--appimage-extract-and-run`.
- Menu entry: put it in `~/Applications` with AppImageLauncher, or run `scripts/install-desktop.sh` from a checkout.
- Files: settings in `~/.config/moonglide/`, caches in `~/.cache/moonglide/`, optional components in
  `~/.local/share/moonglide/`.

Works out of the box: PS5 Remote Play, capture cards, FSR upscaling and the built-in frame generation.
Optional:

| Component | For |
|---|---|
| TensorRT + RIFE model | RIFE frame generation |
| NR weights | neural denoise |
| NVIDIA Video Effects SDK | Maxine effects (source build only) |
| `video` group | capture cards |

The neural denoise also needs an RTX 40 or 50 series GPU; everything else runs on RTX 20 and newer.

## RIFE

Press **Install RIFE** in the Processing tab (Frame generation card), or run:

```sh
./MoonGlide-x86_64.AppImage setup-rife     # or scripts/setup-rife.sh from a checkout
```

Needs `python3` with `venv`. Installs TensorRT 10.9 (~4.4 GB) into `~/.local/share/moonglide/rife-venv`, downloads the
RIFE 4.22-lite weights and exports the model to `~/.local/share/moonglide/rife/`. No sudo. The first start with RIFE
builds a TensorRT engine for your GPU (1-2 minutes, cached). Other sizes: `--size 2560x1440`.

Building from source with RIFE: pass `-DMOONGLIDE_TRT_INCLUDE=DIR` (TensorRT 10.9 and onnx-tensorrt headers) and
`-DNVVFX_CUDA_INCLUDE=DIR` (`cuda.h`). The CI workflow shows how to get them.

## Neural denoise weights

See [WEIGHTS.md](WEIGHTS.md). Put the model directory at `~/.local/share/moonglide/model` or pass `--model DIR`.

## NVIDIA Video Effects SDK (optional)

1. Install the SDK from <https://developer.nvidia.com/maxine>, e.g. to `/usr/local/VideoFX`.
2. Build from source with `-DMOONGLIDE_NVVFX_SDK=/usr/local/VideoFX`.
3. Run with `--nvvfx /usr/local/VideoFX` (or set the path in the app).

## Capture cards

```sh
sudo usermod -aG video "$USER"     # then log out and back in
```

Pick the device in the Input tab or pass `--source dongle --device /dev/videoN`.

MS2109 dongles (`534d:2109`): to let MoonGlide raise the JPEG quality, add a udev rule:

```sh
echo 'SUBSYSTEM=="hidraw", ATTRS{idVendor}=="534d", ATTRS{idProduct}=="2109", MODE="0660", GROUP="video"' \
  | sudo tee /etc/udev/rules.d/70-ms2109-hidraw.rules
sudo udevadm control --reload && sudo udevadm trigger
```

## PS5 helper (source builds)

The AppImage includes it. From source: build <https://github.com/lrfsh/moonglide-chiaki-tap> with `./build.sh`
(needs git, cmake, ninja, a C compiler, `python3-protobuf` and the openssl, opus and pulse dev packages).

## lsfg-vk (optional)

`--fg lsfg` uses the lsfg-vk layer; needs `Lossless.dll` from your own copy of Lossless Scaling. See
[TUNING.md](TUNING.md).

## Uninstall

Delete the AppImage and `~/.config/moonglide`, `~/.cache/moonglide`, `~/.local/share/moonglide`.
