// OpenDLSS-NR embedded: the same record sequence as dlss5vk `serve` (src/serve.cpp
// upstream-patched), without the process boundary. Everything stays on the device:
//
//   colour RGBA8 (work size) -> proxy -> features (+ history proxy) -> network -> head
//   head + proxy + reprojected history -> composite -> image RGBA8 (also next history)
#pragma once
#include <memory>
#include <string>

#include "kernels.h"
#include "nr_graph.h"
#include "nr_model.h"
#include "vk_context.h"

struct NrParams {
  float localTone = 1.0f;
  float localStructure = 0.5f;
  float skinStructure = -1.0f;
  uint32_t style = 1;
  bool autoMask = false;
  uint32_t seed = 0;
};

class NrStage {
 public:
  NrStage(vk::Context& ctx, const std::string& modelDir, const std::string& shaderDir, uint32_t workWidth,
          uint32_t workHeight, const NrParams& params);
  ~NrStage();

  // Records the whole pass. `colour` is RGBA8 at the work size; `motion` is 2 x f16 per pixel
  // (current -> previous, work pixels) or nullptr for the identity. The result is image().
  void record(VkCommandBuffer cmd, const vk::Buffer& colour, const vk::Buffer* motion, bool reset);
  // After the frame's fence: surfaces a chained-wait timeout (the graph's watchdog).
  void afterFrame();

  const vk::Buffer& image() const { return image_; }
  // The history as an RGBA f32 proxy: what motion-compensated frame generation warps.
  const vk::Buffer& historyProxy() const { return historyProxy_; }
  const vk::Buffer& zeroMotion() const { return zeroMotion_; }
  uint32_t width() const { return width_; }
  uint32_t height() const { return height_; }
  float blendScale() const { return blendScale_; }
  void invalidateHistory() { haveHistory_ = false; }
  NrParams params;

 private:
  vk::Context& ctx_;
  uint32_t width_, height_, fullWidth_ = 0, fullHeight_ = 0;
  std::unique_ptr<nr::Model> model_;
  std::unique_ptr<nr::Kernels> kernels_;
  std::unique_ptr<nr::Graph> graph_;
  nr::Activation* features_ = nullptr;
  vk::Buffer proxy_{}, historyProxy_{}, image_{}, zeroMotion_{};
  // Lanes 7-9 want the REPROJECTED previous output (docs/frame.md); the composite reprojects
  // on its own, so the network's copy is warped separately.
  vk::Buffer warpedRgba_{}, warpedProxy_{};
  bool haveHistory_ = false;
  float blendScale_ = 1.0f;
};
