// Motion vectors for the NR history from the GPU's optical-flow engine (VK_NV_optical_flow).
//
//   compute queue:  RGBA8 work frame -> luma (R8) -> image[cur]          recordLuma()
//   flow queue:     NVOF(input = cur, reference = prev) -> 4x4-grid flow  ofCommands()
//   compute queue:  grid flow (S10.5) -> per-pixel 2 x f16, work pixels   recordMotion()
//
// input = current and reference = previous makes each vector "where this pixel was in the
// previous frame", i.e. current -> previous, which is the NR composite's motion convention.
// One session, rebound every frame, so the engine's temporal hints keep working.
#pragma once
#include <volk.h>

#include "gpu.h"

class OpticalFlow {
 public:
  OpticalFlow(Gpu& gpu, uint32_t width, uint32_t height);
  ~OpticalFlow();

  // Compute queue: luma of `rgba` (work size) into the current input image.
  void recordLuma(VkCommandBuffer cmd, const vk::Buffer& rgba);
  // Flow queue: the command buffer for this frame (binds cur/prev first). Valid only when
  // havePrevious(); otherwise there is nothing to compare against.
  VkCommandBuffer ofCommands();
  // Compute queue, after the flow queue finished: per-pixel motion into motion().
  void recordMotion(VkCommandBuffer cmd);
  // After the frame: this frame's luma becomes the next frame's reference.
  void advance();
  void reset() { havePrevious_ = false; }
  // 3x3 vector median on the flow grid before the upsample: a vector more than 1.5 px from its
  // neighbourhood's median is replaced by the median (lone bad vectors near thin text).
  void setMedian(bool on) { median_ = on; }

  bool havePrevious() const { return havePrevious_; }
  const vk::Buffer& motion() const { return motion_; }
  uint32_t gridWidth() const { return gridW_; }
  uint32_t gridHeight() const { return gridH_; }

 private:
  struct Image {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
  };
  Image createImage(VkFormat format, uint32_t w, uint32_t h, VkImageUsageFlags usage, uint32_t ofUsage);
  void destroyImage(Image& img);

  Gpu& gpu_;
  uint32_t width_, height_, gridW_, gridH_;
  VkFormat lumaFormat_ = VK_FORMAT_R8_UNORM;
  Image luma_[2], flow_;
  int cur_ = 0;
  bool havePrevious_ = false;
  bool median_ = true;
  VkOpticalFlowSessionNV session_ = VK_NULL_HANDLE;
  VkCommandPool ofPool_ = VK_NULL_HANDLE;
  VkCommandBuffer ofCmd_ = VK_NULL_HANDLE;
  vk::Buffer lumaBuffer_{}, flowBuffer_{}, motion_{};
  vk::Pipeline lumaPipe_{}, upsamplePipe_{};
};
