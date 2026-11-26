// Optional RIFE frame generation (Practical-RIFE 4.22-lite, MIT) through TensorRT, loaded at run
// time from the user's own TensorRT install. Nothing NVIDIA ships with moonglide.
//
// The model is an ONNX export made by tools/rife/export_onnx.py (fixed size, padded to a
// multiple of 32): img0, img1 (1x3xHxW RGB 0..1), t (1x1x1x1) -> mid (1x3xHxW). TensorRT builds
// an fp16 engine from it once and caches it (~/.cache/moonglide/rife/).
//
// Zero-copy like nvvfx: the planar input / output tensors are Vulkan buffers allocated
// exportable (opaque fd) and imported into CUDA; Vulkan and the TensorRT stream order each
// other through one exported timeline semaphore.
#pragma once
#include <memory>
#include <string>

#include "vk_context.h"

class Gpu;

struct RifeConfig {
  std::string model;     // .onnx (empty: ~/.local/share/moonglide/rife/rife_v4.22_lite_<W>x<H>.onnx)
  std::string trtLib;    // dir with libnvinfer.so.10 (empty: search, see rife.cpp)
  bool fp16 = true;
};

class Rife {
 public:
  // Throws std::runtime_error when TensorRT, CUDA or the model is unavailable. The first run at a
  // size builds the engine (can take a minute or two).
  Rife(Gpu& gpu, const RifeConfig& config, uint32_t workW, uint32_t workH);
  ~Rife();

  uint32_t padW() const { return padW_; }   // tensor size (work size rounded up to 32)
  uint32_t padH() const { return padH_; }
  // Planar fp32 RGB tensors (3 * padW * padH floats).
  const vk::Buffer& in(int i) const;   // input ping-pong
  static constexpr int kMaxOut = 3;    // generated pictures per real frame (x4)
  void ensureOut(int n);                    // make the buffers of generated pictures 0..n-1 (x3 / x4)
  const vk::Buffer& out(int k = 0) const;   // generated picture k

  VkSemaphore timeline() const;
  // Enqueue on the TensorRT stream: wait for the timeline to reach `wait`, run
  // in(cur ^ 1), in(cur) -> out(k) at t[k] for k < n, signal `signal`.
  void run(uint64_t wait, uint64_t signal, int cur, const float* t, int n);
  // GPU time of the last finished run (all n pictures), ms (CUDA events; call after its fence).
  float lastMs();

  struct Impl;

 private:
  std::unique_ptr<Impl> impl_;
  uint32_t padW_ = 0, padH_ = 0;
};
