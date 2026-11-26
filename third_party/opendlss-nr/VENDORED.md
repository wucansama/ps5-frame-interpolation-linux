# OpenDLSS-NR (vendored)

- Upstream: https://github.com/maanHimself/OpenDLSS-NR at `9d08f4184bbcb9d858e2fb7a7834ec0837a9d2f1` (MIT, see LICENSE / NOTICE).
- Plus a Linux port patch (patch id `58e98d824bce70856b56ae4c7f1456d53e1889e6`):
  the serve-path kernels (`composite_serve`, `proxy_expand`, `mid_warp`, `resize_rgba8`), `nr_schedule`, Linux
  build fixes.
- Only the library is vendored: `src/` minus `main.cpp`/`serve.cpp`/`verify.cpp`, `shaders/`, the PTX emitters
  (`scripts/ptx/`, Python, needed only to regenerate) and their generated output (`ptx/`, so the build needs no
  Python).
- Weights are **not** part of this; see docs/WEIGHTS.md.

Regenerating `ptx/` (optional): run the emitter invocations from upstream `scripts/build_linux.sh`.
