#include "optical_flow.h"

#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {

std::vector<VkFormat> ofFormats(VkPhysicalDevice physical, VkOpticalFlowUsageFlagsNV usage) {
  VkOpticalFlowImageFormatInfoNV info{VK_STRUCTURE_TYPE_OPTICAL_FLOW_IMAGE_FORMAT_INFO_NV};
  info.usage = usage;
  uint32_t count = 0;
  vkGetPhysicalDeviceOpticalFlowImageFormatsNV(physical, &info, &count, nullptr);
  std::vector<VkOpticalFlowImageFormatPropertiesNV> props(count, {VK_STRUCTURE_TYPE_OPTICAL_FLOW_IMAGE_FORMAT_PROPERTIES_NV});
  vkGetPhysicalDeviceOpticalFlowImageFormatsNV(physical, &info, &count, props.data());
  std::vector<VkFormat> formats;
  for (auto& p : props) formats.push_back(p.format);
  return formats;
}

}  // namespace

OpticalFlow::OpticalFlow(Gpu& gpu, uint32_t width, uint32_t height) : gpu_(gpu), width_(width), height_(height) {
  if (!gpu.opticalFlow()) throw std::runtime_error("VK_NV_optical_flow not available");
  VkPhysicalDeviceOpticalFlowPropertiesNV props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPTICAL_FLOW_PROPERTIES_NV};
  VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  p2.pNext = &props;
  vkGetPhysicalDeviceProperties2(gpu.physical(), &p2);
  if (!(props.supportedOutputGridSizes & VK_OPTICAL_FLOW_GRID_SIZE_4X4_BIT_NV))
    throw std::runtime_error("optical flow: no 4x4 output grid");
  if (width < props.minWidth || height < props.minHeight || width > props.maxWidth || height > props.maxHeight)
    throw std::runtime_error("optical flow: size out of range");

  bool lumaOk = false;
  for (VkFormat f : ofFormats(gpu.physical(), VK_OPTICAL_FLOW_USAGE_INPUT_BIT_NV))
    if (f == VK_FORMAT_R8_UNORM) lumaOk = true;
  if (!lumaOk) throw std::runtime_error("optical flow: R8_UNORM input not supported");
  bool flowOk = false;
  for (VkFormat f : ofFormats(gpu.physical(), VK_OPTICAL_FLOW_USAGE_OUTPUT_BIT_NV))
    if (f == VK_FORMAT_R16G16_SFIXED5_NV) flowOk = true;
  if (!flowOk) throw std::runtime_error("optical flow: R16G16_S10_5 output not supported");

  gridW_ = (width + 3) / 4;
  gridH_ = (height + 3) / 4;
  for (Image& img : luma_)
    img = createImage(VK_FORMAT_R8_UNORM, width, height, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                      VK_OPTICAL_FLOW_USAGE_INPUT_BIT_NV);
  flow_ = createImage(VK_FORMAT_R16G16_SFIXED5_NV, gridW_, gridH_,
                      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_OPTICAL_FLOW_USAGE_OUTPUT_BIT_NV);

  VkOpticalFlowSessionCreateInfoNV info{VK_STRUCTURE_TYPE_OPTICAL_FLOW_SESSION_CREATE_INFO_NV};
  info.width = width;
  info.height = height;
  info.imageFormat = VK_FORMAT_R8_UNORM;
  info.flowVectorFormat = VK_FORMAT_R16G16_SFIXED5_NV;
  info.outputGridSize = VK_OPTICAL_FLOW_GRID_SIZE_4X4_BIT_NV;
  const char* level = getenv("MOONGLIDE_OF_LEVEL");
  info.performanceLevel = level && !strcmp(level, "slow")     ? VK_OPTICAL_FLOW_PERFORMANCE_LEVEL_SLOW_NV
                          : level && !strcmp(level, "medium") ? VK_OPTICAL_FLOW_PERFORMANCE_LEVEL_MEDIUM_NV
                                                              : VK_OPTICAL_FLOW_PERFORMANCE_LEVEL_FAST_NV;
  VK_CHECK(vkCreateOpticalFlowSessionNV(gpu.device(), &info, nullptr, &session_));

  VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  poolInfo.queueFamilyIndex = gpu.ofFamily();
  VK_CHECK(vkCreateCommandPool(gpu.device(), &poolInfo, nullptr, &ofPool_));
  VkCommandBufferAllocateInfo cb{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cb.commandPool = ofPool_;
  cb.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cb.commandBufferCount = 1;
  VK_CHECK(vkAllocateCommandBuffers(gpu.device(), &cb, &ofCmd_));

  vk::Context& ctx = gpu.ctx();
  lumaBuffer_ = ctx.createBuffer((VkDeviceSize)((width + 3) & ~3u) * height, false, "of luma");
  flowBuffer_ = ctx.createBuffer((VkDeviceSize)gridW_ * gridH_ * 4, false, "of grid flow");
  motion_ = ctx.createBuffer((VkDeviceSize)width * height * 4, false, "of motion f16");
  ctx.fillZero(motion_);
  lumaPipe_ = gpu.pipeline("luma");
  upsamplePipe_ = gpu.pipeline("flow_upsample");

  // Everything lives in GENERAL from here on.
  VkCommandBuffer cmd = ctx.beginCommands();
  for (Image* img : {&luma_[0], &luma_[1], &flow_}) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = img->image;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &b);
  }
  ctx.endAndSubmit(cmd, true);
  fprintf(stderr, "optical flow: %ux%u -> %ux%u grid (4x4), level %s\n", width, height, gridW_, gridH_,
          level ? level : "fast");
}

OpticalFlow::~OpticalFlow() {
  vkDeviceWaitIdle(gpu_.device());
  if (session_) vkDestroyOpticalFlowSessionNV(gpu_.device(), session_, nullptr);
  if (ofPool_) vkDestroyCommandPool(gpu_.device(), ofPool_, nullptr);
  for (Image* img : {&luma_[0], &luma_[1], &flow_}) destroyImage(*img);
  for (vk::Buffer* b : {&lumaBuffer_, &flowBuffer_, &motion_}) gpu_.ctx().destroyBuffer(*b);
}

OpticalFlow::Image OpticalFlow::createImage(VkFormat format, uint32_t w, uint32_t h, VkImageUsageFlags usage,
                                            uint32_t ofUsage) {
  Image img;
  VkOpticalFlowImageFormatInfoNV ofInfo{VK_STRUCTURE_TYPE_OPTICAL_FLOW_IMAGE_FORMAT_INFO_NV};
  ofInfo.usage = ofUsage;
  const uint32_t families[2] = {gpu_.ctx().queueFamily(), gpu_.ofFamily()};
  VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  info.pNext = &ofInfo;
  info.imageType = VK_IMAGE_TYPE_2D;
  info.format = format;
  info.extent = {w, h, 1};
  info.mipLevels = 1;
  info.arrayLayers = 1;
  info.samples = VK_SAMPLE_COUNT_1_BIT;
  info.tiling = VK_IMAGE_TILING_OPTIMAL;
  info.usage = usage;
  info.sharingMode = VK_SHARING_MODE_CONCURRENT;   // compute writes/reads it, the flow engine reads/writes it
  info.queueFamilyIndexCount = 2;
  info.pQueueFamilyIndices = families;
  info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  VK_CHECK(vkCreateImage(gpu_.device(), &info, nullptr, &img.image));
  VkMemoryRequirements req;
  vkGetImageMemoryRequirements(gpu_.device(), img.image, &req);
  VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  alloc.allocationSize = req.size;
  alloc.memoryTypeIndex = gpu_.memoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  VK_CHECK(vkAllocateMemory(gpu_.device(), &alloc, nullptr, &img.memory));
  VK_CHECK(vkBindImageMemory(gpu_.device(), img.image, img.memory, 0));
  VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  view.image = img.image;
  view.viewType = VK_IMAGE_VIEW_TYPE_2D;
  view.format = format;
  view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  VK_CHECK(vkCreateImageView(gpu_.device(), &view, nullptr, &img.view));
  return img;
}

void OpticalFlow::destroyImage(Image& img) {
  if (img.view) vkDestroyImageView(gpu_.device(), img.view, nullptr);
  if (img.image) vkDestroyImage(gpu_.device(), img.image, nullptr);
  if (img.memory) vkFreeMemory(gpu_.device(), img.memory, nullptr);
  img = {};
}

void OpticalFlow::recordLuma(VkCommandBuffer cmd, const vk::Buffer& rgba) {
  struct { uint32_t w, h; } push{width_, height_};
  gpu_.dispatch(cmd, lumaPipe_, {&rgba, &lumaBuffer_}, &push, sizeof(push), ((width_ + 3) / 4 + 15) / 16,
                (height_ + 15) / 16);
  VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr,
                       0, nullptr);
  VkBufferImageCopy copy{};
  copy.bufferRowLength = (width_ + 3) & ~3u;
  copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  copy.imageExtent = {width_, height_, 1};
  vkCmdCopyBufferToImage(cmd, lumaBuffer_.buffer, luma_[cur_].image, VK_IMAGE_LAYOUT_GENERAL, 1, &copy);
}

VkCommandBuffer OpticalFlow::ofCommands() {
  // Rebinding is host-side session state; the previous frame's execute has completed (one
  // frame in flight), so the session is idle here.
  VkDevice d = gpu_.device();
  VK_CHECK(vkBindOpticalFlowSessionImageNV(d, session_, VK_OPTICAL_FLOW_SESSION_BINDING_POINT_INPUT_NV,
                                           luma_[cur_].view, VK_IMAGE_LAYOUT_GENERAL));
  VK_CHECK(vkBindOpticalFlowSessionImageNV(d, session_, VK_OPTICAL_FLOW_SESSION_BINDING_POINT_REFERENCE_NV,
                                           luma_[cur_ ^ 1].view, VK_IMAGE_LAYOUT_GENERAL));
  VK_CHECK(vkBindOpticalFlowSessionImageNV(d, session_, VK_OPTICAL_FLOW_SESSION_BINDING_POINT_FLOW_VECTOR_NV,
                                           flow_.view, VK_IMAGE_LAYOUT_GENERAL));
  VK_CHECK(vkResetCommandBuffer(ofCmd_, 0));
  VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  VK_CHECK(vkBeginCommandBuffer(ofCmd_, &begin));
  VkOpticalFlowExecuteInfoNV exec{VK_STRUCTURE_TYPE_OPTICAL_FLOW_EXECUTE_INFO_NV};
  vkCmdOpticalFlowExecuteNV(ofCmd_, session_, &exec);
  VK_CHECK(vkEndCommandBuffer(ofCmd_));
  return ofCmd_;
}

void OpticalFlow::recordMotion(VkCommandBuffer cmd) {
  VkBufferImageCopy copy{};
  copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  copy.imageExtent = {gridW_, gridH_, 1};
  vkCmdCopyImageToBuffer(cmd, flow_.image, VK_IMAGE_LAYOUT_GENERAL, flowBuffer_.buffer, 1, &copy);
  VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr,
                       0, nullptr);
  // MOONGLIDE_FLOW_SCALE: debug multiplier on the vectors (-1 flips the convention).
  static const float scale = getenv("MOONGLIDE_FLOW_SCALE") ? (float)atof(getenv("MOONGLIDE_FLOW_SCALE")) : 1.0f;
  struct { uint32_t w, h, gw, gh; float scale; uint32_t median; } push{width_, height_, gridW_, gridH_, scale,
                                                                   median_ ? 1u : 0u};
  gpu_.dispatch(cmd, upsamplePipe_, {&flowBuffer_, &motion_}, &push, sizeof(push), (width_ + 15) / 16,
                (height_ + 15) / 16);
}

void OpticalFlow::advance() {
  cur_ ^= 1;
  havePrevious_ = true;
}
