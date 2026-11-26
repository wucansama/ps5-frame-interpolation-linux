#include "gpu.h"

#include <vulkan/vulkan_xlib.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace {

bool hasExtension(const std::vector<VkExtensionProperties>& list, const char* name) {
  for (const auto& e : list)
    if (!strcmp(e.extensionName, name)) return true;
  return false;
}

std::vector<VkExtensionProperties> deviceExtensions(VkPhysicalDevice device) {
  uint32_t count = 0;
  vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr);
  std::vector<VkExtensionProperties> list(count);
  vkEnumerateDeviceExtensionProperties(device, nullptr, &count, list.data());
  return list;
}

}  // namespace

Gpu::Gpu(Display* display, Window window) {
  if (volkInitialize() != VK_SUCCESS) throw std::runtime_error("libvulkan.so.1 not found");

  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "moonglide";
  app.pEngineName = "moonglide";
  app.apiVersion = VK_API_VERSION_1_3;
  std::vector<const char*> instanceExtensions = {VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_XLIB_SURFACE_EXTENSION_NAME};
  std::vector<const char*> layers;
  if (getenv("MOONGLIDE_VALIDATION")) {
    layers.push_back("VK_LAYER_KHRONOS_validation");
    instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }
  VkInstanceCreateInfo instanceInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  instanceInfo.pApplicationInfo = &app;
  instanceInfo.enabledExtensionCount = (uint32_t)instanceExtensions.size();
  instanceInfo.ppEnabledExtensionNames = instanceExtensions.data();
  instanceInfo.enabledLayerCount = (uint32_t)layers.size();
  instanceInfo.ppEnabledLayerNames = layers.data();
  VK_CHECK(vkCreateInstance(&instanceInfo, nullptr, &instance_));
  volkLoadInstance(instance_);

  VkXlibSurfaceCreateInfoKHR surfaceInfo{VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR};
  surfaceInfo.dpy = display;
  surfaceInfo.window = window;
  VK_CHECK(vkCreateXlibSurfaceKHR(instance_, &surfaceInfo, nullptr, &surface_));

  // Any GPU with a graphics+compute queue that can present. One that can also run the neural denoise
  // (FP8 cooperative matrices, RTX 40/50) is preferred when there are several.
  uint32_t count = 0;
  VK_CHECK(vkEnumeratePhysicalDevices(instance_, &count, nullptr));
  std::vector<VkPhysicalDevice> devices(count);
  VK_CHECK(vkEnumeratePhysicalDevices(instance_, &count, devices.data()));
  bool bestNr = false;
  for (VkPhysicalDevice candidate : devices) {
    auto ext = deviceExtensions(candidate);
    if (!hasExtension(ext, VK_KHR_SWAPCHAIN_EXTENSION_NAME)) continue;
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(candidate, &props);
    if (props.apiVersion < VK_API_VERSION_1_3) continue;
    const bool nrExt = hasExtension(ext, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) &&
                       hasExtension(ext, VK_EXT_SHADER_FLOAT8_EXTENSION_NAME);
    if (physical_ && (bestNr || !nrExt)) continue;
    uint32_t familyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(candidate, &familyCount, nullptr);
    std::vector<VkQueueFamilyProperties> families(familyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(candidate, &familyCount, families.data());
    for (uint32_t i = 0; i < familyCount; ++i) {
      VkBool32 present = VK_FALSE;
      vkGetPhysicalDeviceSurfaceSupportKHR(candidate, i, surface_, &present);
      const VkQueueFlags need = VK_QUEUE_COMPUTE_BIT | VK_QUEUE_GRAPHICS_BIT;
      if (present && (families[i].queueFlags & need) == need) {
        physical_ = candidate;
        queueFamily_ = i;
        bestNr = nrExt;
        break;
      }
    }
  }
  if (!physical_) throw std::runtime_error("no Vulkan 1.3 GPU that can show a picture on this display");

  const auto available = deviceExtensions(physical_);
  // The NR kernels' feature set, cut down to what this GPU has: requested features it lacks are
  // dropped, and the denoise is only offered when everything it needs is there.
  vk::DeviceRequirements req;
  {
    vk::DeviceRequirements sup;   // same chain, filled with what the GPU supports
    auto zero = [](auto& st) {
      auto* b = reinterpret_cast<VkBool32*>(reinterpret_cast<char*>(&st) + sizeof(VkBaseOutStructure));
      std::fill(b, b + (sizeof(st) - sizeof(VkBaseOutStructure)) / sizeof(VkBool32), VK_FALSE);
    };
    zero(sup.f11); zero(sup.f12); zero(sup.f13); zero(sup.coop); zero(sup.coop2); zero(sup.fp8);
    zero(sup.executable); zero(sup.clock); zero(sup.sm); zero(sup.features);
    vkGetPhysicalDeviceFeatures2(physical_, &sup.features);
    bool lost = false;   // a requested feature this GPU lacks
    auto meet = [&](auto& want, const auto& have) {
      constexpr size_t hdr = sizeof(VkBaseOutStructure);
      auto* w = reinterpret_cast<VkBool32*>(reinterpret_cast<char*>(&want) + hdr);
      auto* h = reinterpret_cast<const VkBool32*>(reinterpret_cast<const char*>(&have) + hdr);
      for (size_t i = 0; i < (sizeof(want) - hdr) / sizeof(VkBool32); ++i)
        if (w[i] && !h[i]) { w[i] = VK_FALSE; lost = true; }
      return !lost;
    };
    std::vector<const char*> ext;
    bool extOk = true;
    for (const char* e : req.extensions) {
      if (hasExtension(available, e)) ext.push_back(e);
      else extOk = false;
    }
    auto has = [&](const char* e) { return hasExtension(available, e); };
    meet(req.f11, sup.f11); meet(req.f12, sup.f12); meet(req.f13, sup.f13); meet(req.features, sup.features);
    const bool coreOk = !lost;
    const bool coopOk = has(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) && meet(req.coop, sup.coop);
    const bool coop2Ok = has(VK_NV_COOPERATIVE_MATRIX_2_EXTENSION_NAME) && meet(req.coop2, sup.coop2);
    const bool fp8Ok = has(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME) && meet(req.fp8, sup.fp8);
    meet(req.executable, sup.executable); meet(req.clock, sup.clock); meet(req.sm, sup.sm);
    nrCapable_ = coreOk && coopOk && coop2Ok && fp8Ok && extOk && !lost;
    if (getenv("MOONGLIDE_TEST_NO_NR_GPU")) nrCapable_ = false;   // tests: behave like an RTX 20/30
    // Rebuild the chain with only the structs whose extension is enabled.
    void* chain = nullptr;
    auto link = [&](auto& st, bool on) { if (on) { st.pNext = chain; chain = &st; } };
    req.f11.pNext = nullptr; chain = &req.f11;
    link(req.f12, true);
    link(req.f13, true);
    link(req.coop, nrCapable_ && has(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME));
    link(req.coop2, nrCapable_ && has(VK_NV_COOPERATIVE_MATRIX_2_EXTENSION_NAME));
    link(req.fp8, nrCapable_ && has(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME));
    link(req.executable, has(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME));
    link(req.clock, has(VK_KHR_SHADER_CLOCK_EXTENSION_NAME));
    link(req.sm, has(VK_NV_SHADER_SM_BUILTINS_EXTENSION_NAME));
    req.features.pNext = chain;
    if (!nrCapable_) {
      static const char* const nrOnly[] = {VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME, VK_EXT_SHADER_FLOAT8_EXTENSION_NAME,
                                           VK_NV_COOPERATIVE_MATRIX_2_EXTENSION_NAME, VK_NV_CUDA_KERNEL_LAUNCH_EXTENSION_NAME};
      ext.erase(std::remove_if(ext.begin(), ext.end(),
                               [](const char* e) {
                                 for (const char* n : nrOnly) if (!strcmp(e, n)) return true;
                                 return false;
                               }),
                ext.end());
    }
    req.extensions = ext;
  }
  std::vector<const char*> extensions = req.extensions;
  extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

  // present_id + present_wait: when a frame actually reached the screen, for the latency badge.
  VkPhysicalDevicePresentIdFeaturesKHR presentId{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_FEATURES_KHR};
  VkPhysicalDevicePresentWaitFeaturesKHR presentWait{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_WAIT_FEATURES_KHR};
  if (!getenv("MOONGLIDE_NO_PRESENT_WAIT") && hasExtension(available, VK_KHR_PRESENT_ID_EXTENSION_NAME) &&
      hasExtension(available, VK_KHR_PRESENT_WAIT_EXTENSION_NAME)) {
    VkPhysicalDevicePresentIdFeaturesKHR qId{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_FEATURES_KHR};
    VkPhysicalDevicePresentWaitFeaturesKHR qWait{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_WAIT_FEATURES_KHR};
    qWait.pNext = &qId;
    VkPhysicalDeviceFeatures2 q{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    q.pNext = &qWait;
    vkGetPhysicalDeviceFeatures2(physical_, &q);
    if (qId.presentId && qWait.presentWait) {
      presentId.presentId = VK_TRUE;
      presentWait.presentWait = VK_TRUE;
      presentId.pNext = req.features.pNext;
      presentWait.pNext = &presentId;
      req.features.pNext = &presentWait;
      extensions.push_back(VK_KHR_PRESENT_ID_EXTENSION_NAME);
      extensions.push_back(VK_KHR_PRESENT_WAIT_EXTENSION_NAME);
      presentWait_ = true;
    }
  }

  // Optical flow (NVOFA): a dedicated queue family on NVIDIA.
  VkPhysicalDeviceOpticalFlowFeaturesNV ofFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPTICAL_FLOW_FEATURES_NV};
  int ofFamily = -1;
  if (!getenv("MOONGLIDE_NO_OPTICAL_FLOW") && hasExtension(available, VK_NV_OPTICAL_FLOW_EXTENSION_NAME)) {
    VkPhysicalDeviceOpticalFlowFeaturesNV q{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPTICAL_FLOW_FEATURES_NV};
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.pNext = &q;
    vkGetPhysicalDeviceFeatures2(physical_, &f2);
    uint32_t familyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_, &familyCount, nullptr);
    std::vector<VkQueueFamilyProperties> families(familyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_, &familyCount, families.data());
    for (uint32_t i = 0; i < familyCount; ++i)
      if (families[i].queueFlags & VK_QUEUE_OPTICAL_FLOW_BIT_NV) { ofFamily = (int)i; break; }
    if (q.opticalFlow && ofFamily >= 0 && (uint32_t)ofFamily != queueFamily_) {
      ofFeatures.opticalFlow = VK_TRUE;
      ofFeatures.pNext = req.features.pNext;
      req.features.pNext = &ofFeatures;
      extensions.push_back(VK_NV_OPTICAL_FLOW_EXTENSION_NAME);
    } else {
      ofFamily = -1;
    }
  }

  // CUDA interop (optional NVIDIA Video Effects backend): exportable memory + timeline
  // semaphores as opaque fds.
  if (hasExtension(available, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME) &&
      hasExtension(available, VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME)) {
    extensions.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
    extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
    req.f12.timelineSemaphore = VK_TRUE;
    externalFd_ = true;
  }

  // Settings overlay (Dear ImGui) draws with dynamic rendering; the KHR name too, which the
  // ImGui backend looks up.
  req.f13.dynamicRendering = VK_TRUE;
  if (hasExtension(available, VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME))
    extensions.push_back(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);

  float priority = 1.0f;
  VkDeviceQueueCreateInfo queueInfos[2] = {{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO},
                                           {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}};
  queueInfos[0].queueFamilyIndex = queueFamily_;
  queueInfos[0].queueCount = 1;
  queueInfos[0].pQueuePriorities = &priority;
  queueInfos[1].queueFamilyIndex = (uint32_t)ofFamily;
  queueInfos[1].queueCount = 1;
  queueInfos[1].pQueuePriorities = &priority;
  VkDeviceCreateInfo deviceInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  deviceInfo.pNext = &req.features;
  deviceInfo.queueCreateInfoCount = ofFamily >= 0 ? 2 : 1;
  deviceInfo.pQueueCreateInfos = queueInfos;
  deviceInfo.enabledExtensionCount = (uint32_t)extensions.size();
  deviceInfo.ppEnabledExtensionNames = extensions.data();
  VK_CHECK(vkCreateDevice(physical_, &deviceInfo, nullptr, &device_));
  volkLoadDevice(device_);

  if (ofFamily >= 0) {
    ofFamily_ = (uint32_t)ofFamily;
    vkGetDeviceQueue(device_, ofFamily_, 0, &ofQueue_);
  }
  ctx_ = std::make_unique<vk::Context>(instance_, physical_, device_, queueFamily_, 0);
  fprintf(stderr, "gpu: %s, queue family %u, neural denoise %s, present_wait %s, optical flow %s\n",
          ctx_->deviceName().c_str(), queueFamily_, nrCapable_ ? "yes" : "no (needs RTX 40/50)", presentWait_ ? "yes" : "no",
          ofQueue_ ? ("family " + std::to_string(ofFamily_)).c_str() : "no");
}

Gpu::~Gpu() {
  if (device_) vkDeviceWaitIdle(device_);
  for (auto& img : images_)
    if (img.renderDone) vkDestroySemaphore(device_, img.renderDone, nullptr);
  if (swapchain_) vkDestroySwapchainKHR(device_, swapchain_, nullptr);
  ctx_.reset();
  if (device_) vkDestroyDevice(device_, nullptr);
  if (surface_) vkDestroySurfaceKHR(instance_, surface_, nullptr);
  if (instance_) vkDestroyInstance(instance_, nullptr);
}

bool Gpu::formatIsBgra() const {
  return format_ == VK_FORMAT_B8G8R8A8_UNORM || format_ == VK_FORMAT_B8G8R8A8_SRGB;
}

void Gpu::createSwapchain(uint32_t width, uint32_t height, VkPresentModeKHR mode, uint32_t imageCount) {
  VkSurfaceCapabilitiesKHR caps;
  VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_, surface_, &caps));
  uint32_t count = 0;
  vkGetPhysicalDeviceSurfaceFormatsKHR(physical_, surface_, &count, nullptr);
  std::vector<VkSurfaceFormatKHR> formats(count);
  vkGetPhysicalDeviceSurfaceFormatsKHR(physical_, surface_, &count, formats.data());
  // UNORM on purpose: the pixels are already display-referred sRGB code values.
  VkSurfaceFormatKHR chosen = formats[0];
  for (const auto& f : formats)
    if ((f.format == VK_FORMAT_B8G8R8A8_UNORM || f.format == VK_FORMAT_R8G8B8A8_UNORM) &&
        f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
      chosen = f;
      break;
    }
  format_ = chosen.format;

  vkGetPhysicalDeviceSurfacePresentModesKHR(physical_, surface_, &count, nullptr);
  std::vector<VkPresentModeKHR> modes(count);
  vkGetPhysicalDeviceSurfacePresentModesKHR(physical_, surface_, &count, modes.data());
  if (std::find(modes.begin(), modes.end(), mode) == modes.end()) mode = VK_PRESENT_MODE_FIFO_KHR;

  extent_ = caps.currentExtent.width != 0xFFFFFFFFu ? caps.currentExtent : VkExtent2D{width, height};
  imageCount = std::max(imageCount, caps.minImageCount);
  if (caps.maxImageCount) imageCount = std::min(imageCount, caps.maxImageCount);

  VkSwapchainCreateInfoKHR info{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
  info.surface = surface_;
  info.minImageCount = imageCount;
  info.imageFormat = chosen.format;
  info.imageColorSpace = chosen.colorSpace;
  info.imageExtent = extent_;
  info.imageArrayLayers = 1;
  info.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
  info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  info.preTransform = caps.currentTransform;
  info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  info.presentMode = mode;
  info.clipped = VK_TRUE;
  info.oldSwapchain = swapchain_;
  VkSwapchainKHR created;
  VK_CHECK(vkCreateSwapchainKHR(device_, &info, nullptr, &created));
  if (swapchain_) {
    vkDeviceWaitIdle(device_);
    for (auto& img : images_) vkDestroySemaphore(device_, img.renderDone, nullptr);
    vkDestroySwapchainKHR(device_, swapchain_, nullptr);
  }
  swapchain_ = created;

  vkGetSwapchainImagesKHR(device_, swapchain_, &count, nullptr);
  std::vector<VkImage> raw(count);
  vkGetSwapchainImagesKHR(device_, swapchain_, &count, raw.data());
  images_.assign(count, {});
  for (uint32_t i = 0; i < count; ++i) {
    images_[i].image = raw[i];
    VkSemaphoreCreateInfo s{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VK_CHECK(vkCreateSemaphore(device_, &s, nullptr, &images_[i].renderDone));
  }
  const char* modeName = mode == VK_PRESENT_MODE_FIFO_KHR        ? "fifo"
                         : mode == VK_PRESENT_MODE_MAILBOX_KHR   ? "mailbox"
                         : mode == VK_PRESENT_MODE_IMMEDIATE_KHR ? "immediate"
                                                                 : "other";
  fprintf(stderr, "swapchain: %ux%u, %u images, format %d, %s\n", extent_.width, extent_.height, count,
          (int)format_, modeName);
}

vk::Pipeline Gpu::pipeline(const std::string& name, const vk::SpecConstants& spec) {
  VkShaderModule module = ctx_->loadShaderModule(shaderDir + "/" + name + ".spv");
  return ctx_->createComputePipeline(module, spec, strdup(name.c_str()));
}

void Gpu::dispatch(VkCommandBuffer cmd, const vk::Pipeline& pipe, std::initializer_list<const vk::Buffer*> bindings,
                   const void* push, uint32_t pushBytes, uint32_t gx, uint32_t gy, uint32_t gz) {
  const vk::Buffer* slots[vk::kGenericBindings] = {};
  uint32_t i = 0;
  for (const vk::Buffer* b : bindings) {
    if (i >= vk::kGenericBindings) break;
    slots[i++] = b;
  }
  VkDescriptorSet set = ctx_->allocateSet(slots);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx_->pipelineLayout(), 0, 1, &set, 0, nullptr);
  if (pushBytes) vkCmdPushConstants(cmd, ctx_->pipelineLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, pushBytes, push);
  vkCmdDispatch(cmd, gx, gy, gz);
}

uint32_t Gpu::memoryType(uint32_t bits, VkMemoryPropertyFlags flags) const {
  VkPhysicalDeviceMemoryProperties props;
  vkGetPhysicalDeviceMemoryProperties(physical_, &props);
  for (uint32_t i = 0; i < props.memoryTypeCount; ++i)
    if ((bits & (1u << i)) && (props.memoryTypes[i].propertyFlags & flags) == flags) return i;
  throw std::runtime_error("no suitable memory type");
}
