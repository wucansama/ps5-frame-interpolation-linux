# Neural denoise (NR) weights

MoonGlide's neural denoise (`--nr`, the **N** key, "Neural denoise" in the app) runs the OpenDLSS-NR network
(`third_party/opendlss-nr`, MIT): a Vulkan implementation of a transformer-based neural-rendering denoiser. The code is in
this repository; **the network's weights are NVIDIA's and are not distributed here, in the releases or in the AppImage.**
Without them MoonGlide runs normally with the denoiser off (`--no-nr`, the default): frame generation, upscaling and
Remote Play do not need them.

## What the app expects

A model directory, by default `~/.local/share/moonglide/model` (`--model DIR` or `MOONGLIDE_MODEL=DIR` for another place):

```
<model dir>/manifest.json            stages[] (id, file, packedByteLength, sha256) and tensors[] (name, block, layer, ...)
<model dir>/model/stage00.bin ...    the packed FP8 (E4M3) weights, about 141 MiB in total
```

The network has exactly 71 blocks; a model with another block count is refused at load. Each stage file is sha256-checked
against the manifest at start-up. The layout is specified in OpenDLSS-NR's `docs/weights.md`
(<https://github.com/maanHimself/OpenDLSS-NR>).

## Getting the weights

This repository ships no extractor and no instructions for obtaining NVIDIA's files, like OpenDLSS-NR itself. The weights
are part of NVIDIA's proprietary software; you are responsible for the licence terms that apply to any copy you use, and
you may not redistribute them. If you have a model directory in the layout above, put it in place and run
`moonglide --model DIR --nr`.

Check it works without starting a stream: `moonglide --device test --nr --model DIR --frames 60` (synthetic source).
