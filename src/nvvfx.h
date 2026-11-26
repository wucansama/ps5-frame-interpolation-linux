// Optional NVIDIA Video Effects (Maxine VFX SDK) stages, loaded at run time from the user's own
// SDK install (MOONGLIDE_NVVFX=/usr/local/VideoFX or --nvvfx DIR). Nothing NVIDIA ships with
// moonglide: the SDK is under NVIDIA's licence and the user installs it.
//
// Zero-copy: the buffers the effects read and write are Vulkan buffers allocated exportable
// (opaque fd) and imported into CUDA; Vulkan and the effects' CUDA stream order each other
// through one exported timeline semaphore, so nothing goes through the host.
//
//   denoise   VideoSuperRes in a denoise mode (default quality 8), work size -> work size
//   sr        VideoSuperRes upscale (default quality 4, Ultra), work size -> display rect
//   fg        VideoFrameGeneration (default mode 0, low), between two sr outputs, t = 0.5
//
// The effects add a small, content-dependent colour cast (about +1/255 red and blue on game
// footage); the caller measures it on the live frames (mean out - mean in) and undoes it.
#pragma once
#include <memory>
#include <string>

#include "vk_context.h"

class Gpu;

struct NvvfxConfig {
  std::string sdk;          // SDK root
  int denoise = 8;          // VSR quality level, -1 off
  int sr = 4;               // VSR quality level, -1 off
  int fg = 0;               // VFG mode (0 low, 1 medium, 2 high), -1 off
  bool colourFix = true;
};

// Colour correction for a shader: corrected = m * c + t (rgb in 0..1), rows {m0, m1, m2, t}.
struct ColourFix {
  float row[3][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}};
};

class Nvvfx {
 public:
  // Throws std::runtime_error when the SDK, CUDA or an effect is unavailable. A stage that
  // can't run at these sizes (e.g. sr at an unsupported scale) is switched off with a message.
  Nvvfx(Gpu& gpu, const NvvfxConfig& config, uint32_t workW, uint32_t workH, uint32_t outW, uint32_t outH);
  ~Nvvfx();

  bool denoise() const { return denoise_; }
  bool sr() const { return sr_; }
  bool fg() const { return fg_; }
  uint32_t upW() const { return upW_; }   // sr output (= work size without sr)
  uint32_t upH() const { return upH_; }

  // Vulkan views of the shared buffers (RGBA8, tightly packed).
  const vk::Buffer& denoiseIn() const;
  const vk::Buffer& denoiseOut() const;
  const vk::Buffer& srIn() const;      // the shown picture at the work size (sr input)
  const vk::Buffer& up(int i) const;   // sr output ping-pong (or the shown picture without sr)
  const vk::Buffer& mid() const;       // generated picture between up(prev) and up(cur)

  VkSemaphore timeline() const;
  // Enqueue on the effects' stream: wait for the timeline to reach `wait`, run, signal `signal`.
  void runDenoise(uint64_t wait, uint64_t signal);
  // sr srIn -> up(cur) (when sr), then fg up(cur^1), up(cur) -> mid (when fg && havePrev).
  void runSrFg(uint64_t wait, uint64_t signal, int cur, bool havePrev);
  // GPU time of the effects in the last finished frame (CUDA events; call after its fence).
  struct Times { float denoise = 0, sr = 0, fg = 0; };
  Times lastTimes();

  struct Impl;

 private:
  std::unique_ptr<Impl> impl_;
  bool denoise_ = false, sr_ = false, fg_ = false;
  uint32_t upW_ = 0, upH_ = 0;
};

// If MOONGLIDE_NVVFX / --nvvfx is in use, the SDK's libraries must be on LD_LIBRARY_PATH before the
// process starts (their dependencies are resolved by the dynamic loader); re-exec once with it.
void nvvfxEnsureLibraryPath(const std::string& sdk, char** argv);
