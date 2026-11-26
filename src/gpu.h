// One Vulkan device for everything: capture upload, the NR network, the upscale and the
// swapchain. The NR library's vk::Context adopts this device (its "created elsewhere"
// constructor), so its buffers, pipelines and descriptor pools are ours to use too.
#pragma once
#include <X11/Xlib.h>
#include <volk.h>

#include <memory>
#include <string>
#include <vector>

#include "vk_context.h"

struct SwapchainImage {
  VkImage image = VK_NULL_HANDLE;
  VkSemaphore renderDone = VK_NULL_HANDLE;   // signalled by the frame's submit, waited by present
};

class Gpu {
 public:
  Gpu(Display* display, Window window);
  ~Gpu();

  vk::Context& ctx() { return *ctx_; }
  VkDevice device() const { return device_; }
  VkQueue queue() const { return ctx_->queue(); }
  VkPhysicalDevice physical() const { return physical_; }
  VkInstance instance() const { return instance_; }
  uint32_t queueFamily() const { return queueFamily_; }
  // VK_NV_optical_flow: its own queue family (NVOFA). Null queue when unavailable.
  bool opticalFlow() const { return ofQueue_ != VK_NULL_HANDLE; }
  uint32_t ofFamily() const { return ofFamily_; }
  VkQueue ofQueue() const { return ofQueue_; }
  uint32_t memoryType(uint32_t bits, VkMemoryPropertyFlags flags) const;
  // VK_KHR_external_memory_fd + external_semaphore_fd + timeline semaphores (CUDA interop).
  bool externalFd() const { return externalFd_; }
  // Neural denoise needs FP8 cooperative matrices (RTX 40/50); everything else runs without them.
  bool nrCapable() const { return nrCapable_; }

  // Swapchain -----------------------------------------------------------------
  void createSwapchain(uint32_t width, uint32_t height, VkPresentModeKHR mode, uint32_t imageCount);
  VkExtent2D extent() const { return extent_; }
  VkFormat format() const { return format_; }
  VkSwapchainKHR swapchain() const { return swapchain_; }
  const std::vector<SwapchainImage>& images() const { return images_; }
  bool presentWaitEnabled() const { return presentWait_; }
  bool formatIsBgra() const;

  // Compute helpers on the NR context's generic layout (12 storage buffers + 128B push).
  vk::Pipeline pipeline(const std::string& name, const vk::SpecConstants& spec = {});
  void dispatch(VkCommandBuffer cmd, const vk::Pipeline& pipe,
                std::initializer_list<const vk::Buffer*> bindings, const void* push, uint32_t pushBytes,
                uint32_t groupsX, uint32_t groupsY, uint32_t groupsZ = 1);

  std::string shaderDir = MOONGLIDE_SHADER_DIR;

 private:
  VkInstance instance_ = VK_NULL_HANDLE;
  VkPhysicalDevice physical_ = VK_NULL_HANDLE;
  VkDevice device_ = VK_NULL_HANDLE;
  VkSurfaceKHR surface_ = VK_NULL_HANDLE;
  uint32_t queueFamily_ = 0;
  std::unique_ptr<vk::Context> ctx_;
  VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
  VkFormat format_ = VK_FORMAT_B8G8R8A8_UNORM;
  VkExtent2D extent_{};
  std::vector<SwapchainImage> images_;
  bool presentWait_ = false;
  uint32_t ofFamily_ = 0;
  VkQueue ofQueue_ = VK_NULL_HANDLE;
  bool externalFd_ = false;
  bool nrCapable_ = false;
};
