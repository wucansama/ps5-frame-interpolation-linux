# Sony PlayStation 5 frame interpolation tool for Linux（简体中文分支）

> 非官方项目：与 Sony Interactive Entertainment 无隶属关系；PlayStation 是 Sony 的商标。
> 本项目基于 [lrfsh/moonglide](https://github.com/lrfsh/moonglide)（MIT），面向 PS5 采集卡 / Remote Play 做简体中文本地化。

## 功能

- **输入**：PS5 HDMI 采集卡（V4L2）或 PS5 Remote Play。
- **插帧**：内置光流 2×/3×/4×（无需 TensorRT）；可选 RIFE（TensorRT）后端。
- **超分/降噪**：可选 NVIDIA VFX / VSR（需额外安装 VFX SDK 与 TensorRT）。
- **低延迟**：RTX 4060 实测 1080p60 → 120fps，端到端约 **35ms**。
- **简体中文界面**：开始页、设置面板、统计浮层、状态栏均已中文化。
- **音频**：启动脚本自动建立 PipeWire 回环，退出时自动清理。

## 系统要求

- Linux（X11 / XWayland），NVIDIA RTX 20 系及以上。
- 构建依赖：CMake、Ninja、GCC 13+、glslang 16+、X11、libjpeg、FFmpeg 开发库。
- 中文字体：Noto Sans SC / Noto Sans CJK（Debian/Ubuntu 可安装 `fonts-noto-cjk`）。

## 构建

```sh
cmake -S . -B build -G Ninja -DGLSLANG=/path/to/glslang
ninja -C build
```

## 运行（采集卡）

```sh
./build/moonglide --source dongle --device /dev/video2 \
  --size 1920x1080 --fps 60 --format yuyv --fg own --fullscreen
```

操作键：**F1/Tab** 设置面板、**F11** 全屏、**O** 统计浮层、**S** 左右对比、**N** 降噪、**Ctrl+Q** 退出。

## 中文字体

运行环境需要至少一种中文字体，程序会优先查找：

- `/usr/share/fonts/truetype/winfonts/NotoSansSC-VF.ttf`
- `/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc`
- `/usr/share/fonts/truetype/noto/NotoSansSC-Regular.otf`

字体不随仓库分发。

## 说明

- 翻译由机器辅助 + 关键术语人工校对；GPU/VRAM/FPS/MJPG 等技术缩写与实时数值保留英文。
- 与上游 MoonGlide 的差异：ImGui 文本层 i18n、Noto Sans SC 字体、中文状态栏/预算文案。
- 上游英文说明见 [README.en.md](README.en.md)。
- RIFE / NVIDIA VFX 后端需要额外依赖，本分支未改动其上游行为。

## 许可证

MIT，见 [LICENSE](LICENSE)。上游版权归 MoonGlide 贡献者所有。
