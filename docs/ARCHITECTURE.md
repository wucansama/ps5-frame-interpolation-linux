# Architecture and internals

How MoonGlide is put together, what is implemented and how it was measured. For using it, see the [README](../README.md); for flags and settings, [TUNING.md](TUNING.md).

Pipeline: **capture card → neural denoise (OpenDLSS-NR, optional) → upscale → fullscreen Vulkan → frame generation**.
C++20 + Vulkan. No Python, no Wine. Built to play a 30 fps console game (PS5) at 60 fps with neural denoising.

```
v4l2 (MJPEG / YUYV / NV12)          capture thread, newest-frame-wins
  └─ libjpeg-turbo decode ──────────► GPU-visible slot (host-cached, mapped)
                                         │ copy / yuv_to_rgba.comp
                                         ▼
                                      RGBA8 (capture size) ─ resample.comp ─► work size
                                         │
                                         ▼  OpenDLSS-NR, embedded (same device, same command buffer)
                                      proxy → features (+ history) → network → composite
                                         │
                                         ▼  present.comp: upscale, intensity mix, A/B split, badge
                                      swapchain (FIFO, fullscreen, compositor bypassed)
                                         │
                                         ▼  lsfg-vk layer (profile "moonglide"): 30 → 60 fps
```

## Status

| Stage | State |
| --- | --- |
| v4l2 capture (MJPEG CPU decode, YUYV/NV12 GPU convert), drain-to-newest | working |
| Fullscreen Vulkan present + overlay (NR state, intensity, real fps, x2, latency) | working |
| OpenDLSS-NR embedded, temporal history | working |
| lsfg-vk 2x via profile | working |
| GPU optical flow (VK_NV_optical_flow) → motion vectors → reprojected history | working |
| FSR1 EASU + RCAS upscale (default; `--upscale bicubic` for plain scaling) | working |
| Strict 60→30 dedup (phase-locked, in the capture thread) | working on synthetic 60 fps; needs a 60 fps card |
| RGB range auto / limited / full (expanded to full before NR) | working |
| HUD / minimap exclusion zones (NR and sharpening bypassed) | working (lsfg can't honour them) |
| RIFE frame generation (RIFE 4.22-lite on TensorRT, `--fg rife`, default) | working (4.1 ms per generated 1080p frame on a 5090) |
| Own frame generation (flow-based, `--fg own`; or NVIDIA VFG via `--nvvfx`) | working, selectable (even pacing on the 240 Hz dummy) |
| MS2109 dongle JPEG quality (`--dongle-quality off/85/90/92`, default 90) | working (q90 30 fps clean on USB2) |

Measured on RTX 5090 (driver 595.99), MS2109 dongle MJPEG 1080p30, NR at 1080p, output 4K, lsfg x2:
30.0 real fps steady, GPU 7.4 ms per frame (NR 7.0), decode 3 ms, **dequeue → on screen 16 ms**
(from the driver's capture timestamp it's ~50 ms, because on USB2 most of that is the frame's own transfer).

RIFE FG (`--fg rife`, the default): Practical-RIFE 4.22-lite (MIT) interpolates the shown picture (NR output, work
size, before the upscale) at t = 0.5 on TensorRT, zero-copy through exportable Vulkan buffers and a shared timeline
semaphore; the HUD zones and the upscale apply to it like to the real frame, and it is presented through the same
two-image pacing as own FG. When the mean luma change between the two real frames is above `--rife-cut` (default
0.15, a scene cut) the real frame is held instead (`fg_held`). Status JSON: `rife_ms` (GPU time of the network),
`rife_gap_ms` (B end → C start, with the hand-offs). Setup (TensorRT and the model are the user's own install; nothing
NVIDIA is in this repository): see [tools/rife/README.md](../tools/rife/README.md). Without them it falls back to own FG.
`fg mode rife|own|off` switches live between the generators set up at start.


Own FG (`--fg own`): where the two warped pictures disagree the nearer one wins per pixel, and when more than
`--fg-hold` (default 0.2) of the picture disagrees (the flow failed on a fast move) the generated picture is the real
one held, because a per-pixel patchwork of two pictures reads as a semi-transparent ghost at 60 fps. Status JSON:
`fg_fail` (smoothed failing fraction), `fg_held`. Against single bright "fleas" in dark areas (default flags 232): the flow grid
gets a 3x3 vector median first (a vector more than 1.5 px from its neighbourhood's median is replaced, flag 128), and
a generated pixel whose luma is outside the 3x3 range of both real frames at that spot is replaced by their time
blend (flag 64).


### PS5 Remote Play source (optional)

Instead of an HDMI dongle, moonglide can take the PS5's own Remote Play stream: cleaner (no
MJPEG pass, no HDMI scaler), with exact repeat and loss information. `--source remoteplay` (or
Source → PS5 Remote Play in the app, then restart).

```
PS5 --HEVC over LAN--> moonglide-chiaki-tap --each compressed frame--> /dev/shm/moonglide-remoteplay
                       (separate process,                              |
                        chiaki-ng's libchiaki, AGPL)                   v
                       sound, held back to meet the picture <-- moonglide: NVDEC -> NR / clean / RIFE
                       controller (SDL, works in background)
```

- **The tap** is its own project ([moonglide-chiaki-tap](https://github.com/lrfsh/moonglide-chiaki-tap), AGPL-3.0 like
  chiaki-ng, never linked into MoonGlide). Its `build.sh` builds libchiaki without the Qt GUI and installs
  `~/.local/share/moonglide/bin/moonglide-chiaki-tap`. Pair the console from MoonGlide's Source tab (see the README);
  an existing chiaki-ng registration (`~/.config/Chiaki/Chiaki.conf`) is used too.
- **The protocol** is `src/rp_tap.h` (MIT): SmoovPlay's frame tap layout v1 (32 slots of 2 MiB,
  one compressed frame each, a header with the codec, keyframe requests and the sound delay),
  ported to POSIX shm and a futex. `moonglide-rp-fake clip.hevc --shm NAME` is a stand-in console
  for tests.
- **Decoding**: libavcodec with NVDEC (CUDA hwaccel), straight into the capture slot; software
  decoding as a fallback (`--rp-decoder sw`).
- **30 fps games**: the stream is 60 Hz with each picture twice; the repeats are told apart by
  their size (a repeat costs the encoder ~1/7 of a new picture), then RIFE makes 60 / 90 / 120
  (`--fg-mult 2|3|4`).
- **Latency vs cadence**: `--fg-real-delay F` (fraction of the real period, default even
  spacing 1/mult) shows the real picture sooner and interpolates the generated ones at the

## NVIDIA Video Effects backend (optional)

If you've installed NVIDIA's Video Effects SDK yourself (it's under NVIDIA's licence and is never
bundled), moonglide can use it for the whole chain:

- `--nvvfx-colour-fix on|off` (default on): the effects shift colour slightly (about +1/255 red and blue on game footage). Each effect's per-channel mean shift (output vs input) is measured on the live frames, smoothed, and undone in the shaders. A start-up test picture gives the wrong sign, because the shift depends on content.

Frames stay on the GPU throughout. The effects read and write Vulkan buffers exported as opaque fds and imported into CUDA. One exported timeline semaphore orders the Vulkan queue and the effects' CUDA stream. Status JSON reports `nv_denoise_ms`, `nv_sr_ms`, `nv_fg_ms` (CUDA events) and `vk_busy_ms`.

