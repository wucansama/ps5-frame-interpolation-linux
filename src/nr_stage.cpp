#include "nr_stage.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "numeric.h"
#include "reference.h"

NrStage::NrStage(vk::Context& ctx, const std::string& modelDir, const std::string& shaderDir, uint32_t workWidth,
                 uint32_t workHeight, const NrParams& p)
    : params(p), ctx_(ctx), width_(workWidth), height_(workHeight) {
  const auto began = std::chrono::steady_clock::now();
  model_ = std::make_unique<nr::Model>(ctx_, modelDir, true);   // sha256-checked against the manifest

  // The temporal blend's learned cap: one f16 at the front of the last block's blend_scale.
  try {
    const uint32_t post = model_->blockCount() > 0 ? model_->blockCount() - 1 : 0;
    const nr::Tensor& blend = model_->tensor(post, 0, "blend_scale");
    if (blend.byteLength >= 2) blendScale_ = num::f16ToF32(nr::auxHalf(blend, 0, 0));
  } catch (...) {
  }

  kernels_ = std::make_unique<nr::Kernels>(ctx_, shaderDir);
  kernels_->setSiluTable(ref::siluTable());
  const nr::Geometry geometry = nr::Geometry::fromValid(width_, height_);
  fullWidth_ = geometry.fullWidth;
  fullHeight_ = geometry.fullHeight;
  const char* unfused = getenv("DLSS5VK_UNFUSED");
  graph_ = std::make_unique<nr::Graph>(ctx_, *model_, *kernels_, geometry,
                                       nr::Graph::Options{.fusedBlocks = !(unfused && !strcmp(unfused, "1"))});
  features_ = graph_->allocate("input features", fullWidth_ * fullHeight_, 16, nr::Format::F32);

  const VkDeviceSize pixels = (VkDeviceSize)width_ * height_;
  proxy_ = ctx_.createBuffer(pixels * 16, false, "nr proxy");
  historyProxy_ = ctx_.createBuffer(pixels * 16, false, "nr history proxy");
  image_ = ctx_.createBuffer(pixels * 4, false, "nr image");
  zeroMotion_ = ctx_.createBuffer(pixels * 4, false, "nr zero motion");
  warpedRgba_ = ctx_.createBuffer(pixels * 4, false, "nr warped history rgba8");
  warpedProxy_ = ctx_.createBuffer(pixels * 16, false, "nr warped history proxy");
  ctx_.fillZero(historyProxy_);
  ctx_.fillZero(zeroMotion_);
  ctx_.fillZero(proxy_);
  ctx_.fillZero(image_);

  // Warm-up: every PTX module JIT-compiles on first launch.
  VkCommandBuffer cmd = ctx_.beginCommands();
  kernels_->expandRgba8ToProxy(cmd, image_, proxy_, width_, height_);
  nr::Kernels::PreprocessArgs pre{};
  pre.fullWidth = fullWidth_;
  pre.fullHeight = fullHeight_;
  pre.validWidth = pre.sourceWidth = width_;
  pre.validHeight = pre.sourceHeight = height_;
  pre.style = (float)params.style;
  kernels_->preprocessFromProxy(cmd, proxy_, *features_, pre);
  graph_->record(cmd, *features_);
  ctx_.endAndSubmit(cmd, true);
  afterFrame();
  fprintf(stderr, "nr: %ux%u (padded %ux%u), blend scale %.4f, ready in %.0f ms\n", width_, height_, fullWidth_,
          fullHeight_, blendScale_,
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - began).count());
}

NrStage::~NrStage() {
  ctx_.waitIdle();
  graph_.reset();
  kernels_.reset();
  model_.reset();
  for (vk::Buffer* b : {&proxy_, &historyProxy_, &image_, &zeroMotion_, &warpedRgba_, &warpedProxy_}) ctx_.destroyBuffer(*b);
}

void NrStage::record(VkCommandBuffer cmd, const vk::Buffer& colour, const vk::Buffer* motion, bool reset) {
  if (reset) haveHistory_ = false;
  // The history first: image_ still holds the frame published last, and the composite below
  // overwrites it.
  if (haveHistory_) kernels_->expandRgba8ToProxy(cmd, image_, historyProxy_, width_, height_);
  const vk::Buffer* networkHistory = &historyProxy_;
  if (haveHistory_ && motion) {
    // The full frame's motion (offset 1.0) through the composite's own five-tap reconstruction.
    kernels_->midWarpToRgba8(cmd, historyProxy_, *motion, warpedRgba_, width_, width_, height_, 1.0f);
    kernels_->expandRgba8ToProxy(cmd, warpedRgba_, warpedProxy_, width_, height_);
    networkHistory = &warpedProxy_;
  }
  kernels_->expandRgba8ToProxy(cmd, colour, proxy_, width_, height_);

  nr::Kernels::PreprocessArgs pre{};
  pre.fullWidth = fullWidth_;
  pre.fullHeight = fullHeight_;
  pre.validWidth = pre.sourceWidth = width_;
  pre.validHeight = pre.sourceHeight = height_;
  pre.seed = params.seed;
  pre.autoMask = params.autoMask;
  pre.localTone = params.localTone;
  pre.localStructure = params.localStructure;
  pre.skinStructure = params.skinStructure;
  pre.style = (float)params.style;
  pre.history = haveHistory_ ? networkHistory : nullptr;
  kernels_->preprocessFromProxy(cmd, proxy_, *features_, pre);
  graph_->record(cmd, *features_);
  kernels_->compositeToRgba8(cmd, proxy_, graph_->head(), image_, width_, fullWidth_, width_, height_,
                             motion ? *motion : zeroMotion_, haveHistory_ ? &historyProxy_ : nullptr, blendScale_);
  haveHistory_ = true;
}

void NrStage::afterFrame() {
  const nr::Kernels::ChainTimeouts t = kernels_->chainTimeouts();
  if (t.waits)
    throw std::runtime_error(std::to_string(t.waits) + " NR chained wait(s) timed out (first on " + t.counter +
                             "); DLSS5VK_CHAIN=0 runs without chaining");
}
