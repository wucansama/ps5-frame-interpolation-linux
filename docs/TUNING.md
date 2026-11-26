# Tuning, options and settings reference

Everything configurable, with the command-line flag, config key and live control command. Most of it is also in the app UI (F1 / Tab).

## Running from a checkout

```sh
moonglide --model ~/.local/share/moonglide/model            # Remote Play + RIFE frame generation; add --nr for the denoiser, --source dongle --device /dev/video0 for a capture card
moonglide --device test --no-nr                               # synthetic source, no card needed
```

Options: `--device`, `--size WxH`, `--fps`, `--format auto|mjpeg|yuyv|nv12`, `--work WxH` (NR size), `--intensity`,
`--local-structure`, `--local-tone`, `--present fifo|mailbox|immediate`, `--dedup auto|on|off`,
`--range auto|limited|full`, `--upscale fsr|bicubic`, `--sharpen STOPS`, `--hud ZONE` (repeatable), `--hud-show`,
`--windowed WxH`, `--no-overlay`, `--overlay-pos tl|tr|bl|br`, `--overlay-size N` (0 auto), `--overlay-mode full|compact` (also in the Display tab; ctl `overlay pos|size|mode X`), `--frames N`.

HUD zones are fractions of the picture: `rect:x,y,w,h` or `ellipse:cx,cy,rx,ry`, optionally `:feather` in pixels,
e.g. a bottom-left minimap: `--hud ellipse:0.11,0.84,0.08,0.12`.

Keys: **N** toggle NR (A/B), **Up/Down** intensity, **[ ]** local structure, **S** split screen, **O** overlay (small, full, off),
**U** FSR/bicubic, **H** outline HUD zones, **R** reset history, **Ctrl+Q** quit (a plain Q or Esc, which Moonlight
forwards from game menus, no longer closes the stream).

### App UI and settings file

**F1** or **Tab** opens the app UI inside the same window: the picture moves into its own area, with a top bar
(source, format, frame generation, presets), a sidebar of cards (Quality, Frame gen, Source, HUD, Display, NVIDIA;
every control is one of the control-fifo commands below), and panels for the signal, GPU stage timings and frame
pacing. HUD zones are drawn by dragging on the picture. F1 / Tab / Esc closes it and the picture is fullscreen again.
`--ui` starts with it open, in a window (80% of the screen); **F11**, **Alt+Enter**, a double-click on the picture or the top bar's Fullscreen button
switch fullscreen on and off live (the picture then fills the screen, with no panel or stats badge; moving the mouse
shows an auto-hiding bar with "Settings" and "Exit fullscreen") (`--fullscreen` starts that way; ctl `fullscreen toggle|on|off`).

In the desktop app (`--ui`) the panel opens by itself, on the Source tab, while there is no picture (and closes
once one arrives, unless you opened it); while streaming, moving the mouse shows an auto-hiding "Settings" button.

When there is no picture (stream not up yet, PS5 busy with another Remote Play session, no HDMI signal) the window
still runs and says why, with the tap's retry countdown. If the start fails, the desktop app shows an error screen
instead of exiting. When the GPU is out of memory it lists the processes using it (from `nvidia-smi`, when
present) and offers Retry / Quit. Nothing is ever stopped for you. A feature that cannot get GPU memory (RIFE, noise reduction, Maxine effects) falls back (passthrough / own FG) and shows a dismissible "Not enough GPU memory for ..." notice in the UI. If the PS5 sits at its user-selection screen, Remote Play is refused until a
user is picked; turn on PS5 Settings > Users and Accounts > Login Settings > Log In to PS5 Automatically.

Settings live in `~/.config/moonglide/config.toml` (`--config PATH`, `MOONGLIDE_CONFIG`, `--no-config`): loaded at
start, saved on every change, with named presets (`[preset."Name"]`). Keys are the option names without `--`;
options given on the command line win over the file. Changes that need a restart (device, size, present mode, FG
backend, NVIDIA effects) show "Apply & restart", which re-executes in place.

Desktop menu: `scripts/install-desktop.sh` adds **MoonGlide** (icon, `StartupWMClass=moonglide`) to the
application menu. The launcher (`scripts/moonglide-app`) brings a running instance to the front with the UI open
(`ui show` on the fifo), or starts one as the user unit `moonglide` with `--ui`.

Remote control, for scripts or a second machine over ssh:

```sh
echo "nr off"        > $XDG_RUNTIME_DIR/moonglide.ctl
echo "intensity 0.7" > $XDG_RUNTIME_DIR/moonglide.ctl   # also: structure X, tone X, split on|off, overlay on|off, reset, quit
echo "hud add rect:0.02,0.80,0.15,0.18" > $XDG_RUNTIME_DIR/moonglide.ctl   # hud clear|pop|show on|off
echo "range auto"    > $XDG_RUNTIME_DIR/moonglide.ctl   # limited|full;  dedup auto|on|off;  upscale fsr|bicubic
echo "fg hold 0.2"   > $XDG_RUNTIME_DIR/moonglide.ctl   # own FG: hold the real frame when >20% of the flow fails (0 off)
echo "fg flags 232"  > $XDG_RUNTIME_DIR/moonglide.ctl   # own FG: 2 hard pick, 4 dilate, 8 clamp to source range, 32 static guard, 64 outlier rejection, 128 flow median
echo "ui show"       > $XDG_RUNTIME_DIR/moonglide.ctl   # ui hide|toggle; set KEY VALUE; restart; preset save|load|delete NAME
cat /dev/shm/moonglide.status                           # JSON, once a second
```

MCP server for Claude and other MCP clients (`moonglide mcp`): see [docs/MCP.md](MCP.md).

A second instance (tests, replays) that finds the control fifo in use switches to `<path>.<pid>` for both the fifo and
the status file, so it can't take over the live one; `--control PATH` / `--status PATH` (or `MOONGLIDE_CTL` /
`MOONGLIDE_STATUS`) set them explicitly. Every clean exit logs `quit: <reason>`.


MS2109 dongles (USB 534d:2109): the encoder's JPEG quality goes up from the firmware's q75 to `--dongle-quality`
(default 90; 92 is experimental) through the dongle's HID interface after every stream start (RAM only, never the
EEPROM; `dongle quality N|off` live). More than 2 corrupt frames in 5 s steps it down (92 → 90 → 85). It needs rw
on the dongle's hidraw node:

```sh
# /etc/udev/rules.d/70-ms2109-hidraw.rules
SUBSYSTEM=="hidraw", ATTRS{idVendor}=="534d", ATTRS{idProduct}=="2109", MODE="0660", GROUP="plugdev"
```


  matching times; x2 at 0.375 saves ~4 ms, at 0.25 ~8 ms, with an uneven cadence.
  `--deband F` (threshold in 1/255, ~2) smooths banded gradients for ~0.1 ms.
- **Network loss**: pictures after a loss are shown but never interpolated from (RIFE holds the
  real frame) until the PS5 repairs the picture; moonglide asks for a keyframe.
- **Sound**: the tap plays chiaki's sound and holds it back by moonglide's measured delay (packet
  arrival -> on screen) minus the sound device's own, so they stay in step.
- Only one Remote Play session per console: close chiaki-ng's own stream first.

| Setting | Flag | |
|---|---|---|
| `source` | `--source dongle\|remoteplay` | restart |
| `rp-host` | `--rp-host IP` | console address; empty: don't start a tap (one runs already) |
| `rp-bitrate` | `--rp-bitrate KBPS` | 0: chiaki-ng's setting |
| `rp-codec` | `--rp-codec h265\|h264` | |
| `rp-decoder` | `--rp-decoder cuda\|sw` | |
| `rp-res` | `--rp-res 720\|1080\|1440\|2160` | stream resolution (unset: from `size`) |
| `rp-fps` | `--rp-fps 30\|60` | stream frame rate (unset: from `fps`) |
| `rp-psn-account`, `rp-psn-online` | `--rp-psn-account B64`, `--rp-psn-online NAME` | PlayStation Network account used for pairing; written by the UI (online id is for display) |
| `rp-tap` | `--rp-tap PATH` | moonglide-chiaki-tap binary (default `~/.local/share/moonglide/bin/moonglide-chiaki-tap`) |
| `rp-wake` | `--rp-wake on\|off` | wake the console from rest mode at start (default off). Otherwise, and whenever the PS5 goes to rest later, MoonGlide waits and shows a "Wake PS5" button (ctl `rp wake`) |

#### Setting it up from the UI (Source tab)

Choose **PS5 Remote Play** as the source; the tab then shows only the Remote Play settings
(the capture card has its own keys: `device`, `size`, `fps`, `format`).

1. **Find the console**: *Scan* runs `moonglide-chiaki-tap --discover-all` in the background and lists
   name, address, PS5/PS4, ready/standby and whether it is already paired. Click a paired console
   to use it (sets `rp-host`); the address can also be typed.
2. **Pair an unpaired console** (the *Pair a console* card; needs a moonglide-chiaki-tap with the
   `--psn-login-url`, `--psn-account-id` and `--register` commands):
   - *Sign in to PlayStation Network* opens the sign-in page in the browser. When it ends on a blank
     or error page, copy that page's address into the field and press *Continue*; the account id is
     stored in `rp-psn-account` and the UI shows "signed in as <online id>".
   - On the console open Settings > System > Remote Play > Pair Device, type the 8-digit PIN and
     press *Pair*. The PIN is never stored or logged.
3. **Stream settings**: resolution, frame rate, codec, bitrate (5-100 Mbps or the console default),
   decoder, and *Wake the console when MoonGlide starts*. Changes apply after *Apply & restart*.

The **Capture card** view lists the video capture devices (name and path) and offers the formats,
sizes and frame rates the selected device reports (read in the background with read-only V4L2 queries,
so a device that is streaming is not disturbed); a custom device path can still be typed.


### Frame generation (lsfg-vk)

Add to `~/.config/lsfg-vk/conf.toml` (the badge shows `x2` only when the layer is loaded and this profile exists):

```toml
[[game]]
exe = "moonglide"
multiplier = 2
flow_scale = 1.0
performance_mode = false
hdr_mode = false
experimental_present_mode = "fifo"
```

lsfg-vk needs `Lossless.dll` from your own copy of Lossless Scaling. It's never bundled.

## NVIDIA Video Effects flags

    decode -> VSR denoise (q8) -> temporal clean -> NR -> VSR upscale to the display (q4) -> VFG x2 -> present (HUD zones)

```sh
cmake -S . -B build -DMOONGLIDE_NVVFX_SDK=/usr/local/VideoFX   # headers only; the libraries are dlopen()ed
MOONGLIDE_NVVFX=/usr/local/VideoFX build/moonglide            # or --nvvfx DIR
```

- `--nvvfx-denoise Q|off` (default 8, VSR denoise low): replaces the shader deblock and turns on `--clean`'s temporal pass.
- `--nvvfx-sr Q|off` (default 4, VSR ultra): replaces FSR. It's off when the display rect is the work size.
- `--nvvfx-fg MODE|off` (default 0, VFG low): replaces own FG and lsfg.
