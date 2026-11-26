# MoonGlide 简体中文分支

这是 [lrfsh/moonglide](https://github.com/lrfsh/moonglide) 的简体中文本地化分支
（上游 MIT License，版权归原贡献者所有）。

## 改动

- `src/i18n.cpp` / `src/i18n.h`：中文翻译表（约 330 条 UI 文案；关键术语人工校对）。
- `third_party/imgui/imgui_draw.cpp`：在 ImGui 文本层替换文案（`CalcTextSizeA` / `AddText`），
  布局按中文宽度计算，不需要改几百处调用点。
- `src/ui.cpp`：使用系统中文字体（Noto Sans SC），并按翻译表动态生成字形范围，避免缺字；
  状态栏、GPU 预算等少量格式化文案改为中文。
- `CMakeLists.txt`：加入 `src/i18n.cpp`。

## 构建

依赖与上游相同（CMake、Ninja、GCC 13+、glslang 16+、X11、libjpeg、FFmpeg 开发包）：

```sh
cmake -S . -B build -G Ninja -DGLSLANG=/path/to/glslang
ninja -C build
```

运行（采集卡示例）：

```sh
./build/moonglide --source dongle --device /dev/video2 \
  --size 1920x1080 --fps 60 --format yuyv --fg own --fullscreen
```

## 中文字体

运行环境需要至少一种中文字体，优先查找：

- `/usr/share/fonts/truetype/winfonts/NotoSansSC-VF.ttf`
- `/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc`
- `/usr/share/fonts/truetype/noto/NotoSansSC-Regular.otf`

Debian/Ubuntu 可安装 `fonts-noto-cjk`；字体不随仓库分发。

## 说明

- 翻译为机器辅助 + 关键术语人工修正；GPU/VRAM/FPS/MJPG 等技术缩写与实时数值保留英文。
- RIFE / NVIDIA VFX 后端需要额外依赖，本分支未改动其上游行为。
- License：MIT（见 `LICENSE`，保留原版权声明）。
