#include "nvvfx.h"

#include <dlfcn.h>
#include <glob.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "gpu.h"

void nvvfxEnsureLibraryPath(const std::string& sdk, char** argv) {
  if (getenv("MOONGLIDE_NVVFX_EXEC")) return;
  std::string path = sdk + "/external/cuda/lib:" + sdk + "/external/tensorrt/lib:" + sdk + "/lib";
  glob_t g{};
  if (glob((sdk + "/features/*/lib").c_str(), 0, nullptr, &g) == 0)
    for (size_t i = 0; i < g.gl_pathc; ++i) path += std::string(":") + g.gl_pathv[i];
  globfree(&g);
  if (const char* old = getenv("LD_LIBRARY_PATH"); old && *old) path += std::string(":") + old;
  // The original, so a settings restart can put it back.
  setenv("MOONGLIDE_ORIG_LD_LIBRARY_PATH", getenv("LD_LIBRARY_PATH") ? getenv("LD_LIBRARY_PATH") : "", 1);
  setenv("LD_LIBRARY_PATH", path.c_str(), 1);
  setenv("MOONGLIDE_NVVFX_EXEC", "1", 1);
  execv("/proc/self/exe", argv);
  fprintf(stderr, "nvvfx: re-exec with the SDK library path failed (%s); loading may fail\n", strerror(errno));
}

#ifndef MOONGLIDE_HAVE_NVVFX

struct Nvvfx::Impl {};
Nvvfx::Nvvfx(Gpu&, const NvvfxConfig&, uint32_t, uint32_t, uint32_t, uint32_t) {
  throw std::runtime_error("built without the NVIDIA Video Effects SDK headers (cmake -DMOONGLIDE_NVVFX_SDK=DIR)");
}
Nvvfx::~Nvvfx() = default;
static const vk::Buffer kNone;
const vk::Buffer& Nvvfx::denoiseIn() const { return kNone; }
const vk::Buffer& Nvvfx::denoiseOut() const { return kNone; }
const vk::Buffer& Nvvfx::srIn() const { return kNone; }
const vk::Buffer& Nvvfx::up(int) const { return kNone; }
const vk::Buffer& Nvvfx::mid() const { return kNone; }
VkSemaphore Nvvfx::timeline() const { return VK_NULL_HANDLE; }
void Nvvfx::runDenoise(uint64_t, uint64_t) {}
void Nvvfx::runSrFg(uint64_t, uint64_t, int, bool) {}
Nvvfx::Times Nvvfx::lastTimes() { return {}; }

#else

#include <cuda.h>

#include "nvCVImage.h"
#include "nvVideoEffects.h"

// The SDK's NvCVImage C++ wrapper calls these from its inline constructor / destructor; the
// library is dlopen()ed, so forward them (before it is loaded: a zeroed, non-owning image).
static decltype(&NvCVImage_Alloc) gNvCVImageAlloc = nullptr;
static decltype(&NvCVImage_Dealloc) gNvCVImageDealloc = nullptr;
extern "C" NvCV_Status NvCVImage_Alloc(NvCVImage* im, unsigned width, unsigned height, NvCVImage_PixelFormat format,
                                       NvCVImage_ComponentType type, unsigned layout, unsigned memSpace,
                                       unsigned alignment) {
  if (gNvCVImageAlloc) return gNvCVImageAlloc(im, width, height, format, type, layout, memSpace, alignment);
  memset(static_cast<void*>(im), 0, sizeof(*im));
  return NVCV_SUCCESS;
}
extern "C" void NvCVImage_Dealloc(NvCVImage* im) {
  if (gNvCVImageDealloc) gNvCVImageDealloc(im);
}

namespace {

#define VX_STR2(x) #x
#define VX_STR(x) VX_STR2(x)

// CUDA driver API, resolved from libcuda.so.1 (names through cuda.h's versioning macros).
#define VX_CU_FUNCS(X)                                                                                    \
  X(cuInit) X(cuDeviceGetCount) X(cuDeviceGet) X(cuDeviceGetUuid) X(cuDevicePrimaryCtxRetain)            \
  X(cuDevicePrimaryCtxRelease) X(cuCtxSetCurrent) X(cuImportExternalMemory)                              \
  X(cuExternalMemoryGetMappedBuffer) X(cuDestroyExternalMemory) X(cuMemFree) X(cuImportExternalSemaphore) \
  X(cuDestroyExternalSemaphore) X(cuSignalExternalSemaphoresAsync) X(cuWaitExternalSemaphoresAsync)      \
  X(cuEventCreate) X(cuEventRecord) X(cuEventElapsedTime) X(cuEventDestroy) X(cuStreamSynchronize)
#define VX_VFX_FUNCS(X)                                                                                   \
  X(NvVFX_CreateEffect) X(NvVFX_DestroyEffect) X(NvVFX_SetU32) X(NvVFX_SetF32) X(NvVFX_SetString)        \
  X(NvVFX_SetImage) X(NvVFX_SetCudaStream) X(NvVFX_Load) X(NvVFX_Run) X(NvVFX_CudaStreamCreate)          \
  X(NvVFX_CudaStreamDestroy)
#define VX_CVI_FUNCS(X) X(NvCVImage_Init) X(NvCV_GetErrorStringFromCode)

struct Api {
#define VX_DECL(f) decltype(&f) p_##f = nullptr;
  VX_CU_FUNCS(VX_DECL)
  VX_VFX_FUNCS(VX_DECL)
  VX_CVI_FUNCS(VX_DECL)
#undef VX_DECL
  void* cuda = nullptr;
  void* vfx = nullptr;
  void* cvi = nullptr;

  void load(const std::string& sdk) {
    cuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (!cuda) throw std::runtime_error(std::string("libcuda.so.1: ") + dlerror());
    const std::string lib = sdk + "/lib/";
    cvi = dlopen((lib + "libNVCVImage.so").c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!cvi) throw std::runtime_error(std::string("libNVCVImage.so: ") + dlerror());
    vfx = dlopen((lib + "libVideoFX.so").c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!vfx) throw std::runtime_error(std::string("libVideoFX.so: ") + dlerror());
    auto sym = [](void* h1, void* h2, const char* name) {
      void* p = dlsym(h1, name);
      if (!p && h2) p = dlsym(h2, name);
      if (!p) throw std::runtime_error(std::string("missing symbol ") + name);
      return p;
    };
#define VX_LOAD_CU(f) p_##f = reinterpret_cast<decltype(p_##f)>(sym(cuda, nullptr, VX_STR(f)));
#define VX_LOAD_VFX(f) p_##f = reinterpret_cast<decltype(p_##f)>(sym(vfx, cvi, VX_STR(f)));
#define VX_LOAD_CVI(f) p_##f = reinterpret_cast<decltype(p_##f)>(sym(cvi, vfx, VX_STR(f)));
    VX_CU_FUNCS(VX_LOAD_CU)
    VX_VFX_FUNCS(VX_LOAD_VFX)
    VX_CVI_FUNCS(VX_LOAD_CVI)
    gNvCVImageAlloc = reinterpret_cast<decltype(gNvCVImageAlloc)>(sym(cvi, nullptr, "NvCVImage_Alloc"));
    gNvCVImageDealloc = reinterpret_cast<decltype(gNvCVImageDealloc)>(sym(cvi, nullptr, "NvCVImage_Dealloc"));
#undef VX_LOAD_CU
#undef VX_LOAD_VFX
#undef VX_LOAD_CVI
  }
};

}  // namespace

struct Nvvfx::Impl {
  Api api;
  Gpu* gpu = nullptr;
  CUcontext cuCtx = nullptr;
  CUdevice cuDev = 0;
  CUstream stream = nullptr;

  struct Shared {
    vk::Buffer vk;
    CUexternalMemory ext = nullptr;
    CUdeviceptr ptr = 0;
    NvCVImage img;
    uint32_t w = 0, h = 0;
  };
  Shared dnIn, dnOut, srIn, up[2], mid;
  VkSemaphore timeline = VK_NULL_HANDLE;
  CUexternalSemaphore cuTimeline = nullptr;

  NvVFX_Handle dn = nullptr, sr = nullptr, fg = nullptr;
  CUevent ev[6] = {};
  bool ranDn = false, ranSr = false, ranFg = false;

  void cu(CUresult r, const char* what) {
    if (r != CUDA_SUCCESS) throw std::runtime_error(std::string("nvvfx: ") + what + " failed: CUDA error " + std::to_string((int)r));
  }
  void vfx(NvCV_Status s, const char* what) {
    if (s != NVCV_SUCCESS)
      throw std::runtime_error(std::string("nvvfx: ") + what + ": " + api.p_NvCV_GetErrorStringFromCode(s));
  }

  void makeShared(Shared& s, uint32_t w, uint32_t h, const char* label) {
    VkDevice dev = gpu->device();
    s.w = w;
    s.h = h;
    const VkDeviceSize bytes = (VkDeviceSize)w * h * 4;
    VkExternalMemoryBufferCreateInfo extInfo{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    extInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.pNext = &extInfo;
    info.size = bytes;
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(dev, &info, nullptr, &s.vk.buffer));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev, s.vk.buffer, &req);
    VkMemoryDedicatedAllocateInfo dedicated{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
    dedicated.buffer = s.vk.buffer;
    VkExportMemoryAllocateInfo exportInfo{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
    exportInfo.pNext = &dedicated;
    exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    alloc.pNext = &exportInfo;
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = gpu->memoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(dev, &alloc, nullptr, &s.vk.memory));
    VK_CHECK(vkBindBufferMemory(dev, s.vk.buffer, s.vk.memory, 0));
    s.vk.size = bytes;
    s.vk.label = label;
    VkMemoryGetFdInfoKHR fdInfo{VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
    fdInfo.memory = s.vk.memory;
    fdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    int fd = -1;
    VK_CHECK(vkGetMemoryFdKHR(dev, &fdInfo, &fd));
    CUDA_EXTERNAL_MEMORY_HANDLE_DESC md{};
    md.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
    md.handle.fd = fd;   // CUDA owns the fd after a successful import
    md.size = req.size;
    md.flags = CUDA_EXTERNAL_MEMORY_DEDICATED;
    cu(api.p_cuImportExternalMemory(&s.ext, &md), "cuImportExternalMemory");
    CUDA_EXTERNAL_MEMORY_BUFFER_DESC bd{};
    bd.offset = 0;
    bd.size = bytes;
    cu(api.p_cuExternalMemoryGetMappedBuffer(&s.ptr, s.ext, &bd), "cuExternalMemoryGetMappedBuffer");
    vfx(api.p_NvCVImage_Init(&s.img, w, h, (int)(w * 4), reinterpret_cast<void*>(s.ptr), NVCV_RGBA, NVCV_U8,
                             NVCV_INTERLEAVED, NVCV_GPU),
        "NvCVImage_Init");
  }

  void freeShared(Shared& s) {
    if (s.ptr) api.p_cuMemFree(s.ptr);
    if (s.ext) api.p_cuDestroyExternalMemory(s.ext);
    if (s.vk.buffer) vkDestroyBuffer(gpu->device(), s.vk.buffer, nullptr);
    if (s.vk.memory) vkFreeMemory(gpu->device(), s.vk.memory, nullptr);
    s = Shared{};
  }

  void wait(uint64_t value) {
    CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS p{};
    p.params.fence.value = value;
    cu(api.p_cuWaitExternalSemaphoresAsync(&cuTimeline, &p, 1, stream), "cuWaitExternalSemaphoresAsync");
  }
  void signal(uint64_t value) {
    CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS p{};
    p.params.fence.value = value;
    cu(api.p_cuSignalExternalSemaphoresAsync(&cuTimeline, &p, 1, stream), "cuSignalExternalSemaphoresAsync");
  }

  NvVFX_Handle makeVsr(Shared& in, Shared& out, unsigned quality, const std::string& models) {
    NvVFX_Handle h = nullptr;
    vfx(api.p_NvVFX_CreateEffect("VideoSuperRes", &h), "create VideoSuperRes");
    api.p_NvVFX_SetString(h, NVVFX_MODEL_DIRECTORY, models.c_str());
    vfx(api.p_NvVFX_SetCudaStream(h, NVVFX_CUDA_STREAM, stream), "VSR stream");
    vfx(api.p_NvVFX_SetImage(h, NVVFX_INPUT_IMAGE, &in.img), "VSR input");
    vfx(api.p_NvVFX_SetImage(h, NVVFX_OUTPUT_IMAGE, &out.img), "VSR output");
    vfx(api.p_NvVFX_SetU32(h, "QualityLevel", quality), "VSR quality");
    NvCV_Status s = api.p_NvVFX_Load(h);
    if (s != NVCV_SUCCESS) {
      api.p_NvVFX_DestroyEffect(h);
      vfx(s, "VSR load");
    }
    return h;
  }
};

Nvvfx::Nvvfx(Gpu& gpu, const NvvfxConfig& config, uint32_t workW, uint32_t workH, uint32_t outW, uint32_t outH)
    : impl_(std::make_unique<Impl>()) {
  Impl& m = *impl_;
  m.gpu = &gpu;
  if (!gpu.externalFd()) throw std::runtime_error("nvvfx: the Vulkan device has no external memory/semaphore fd support");
  m.api.load(config.sdk);
  Api& a = m.api;

  // The CUDA device that is this Vulkan device.
  m.cu(a.p_cuInit(0), "cuInit");
  VkPhysicalDeviceIDProperties idProps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
  VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  props2.pNext = &idProps;
  vkGetPhysicalDeviceProperties2(gpu.physical(), &props2);
  int count = 0;
  m.cu(a.p_cuDeviceGetCount(&count), "cuDeviceGetCount");
  int found = -1;
  for (int i = 0; i < count && found < 0; ++i) {
    CUdevice d;
    CUuuid uuid;
    if (a.p_cuDeviceGet(&d, i) == CUDA_SUCCESS && a.p_cuDeviceGetUuid(&uuid, d) == CUDA_SUCCESS &&
        memcmp(uuid.bytes, idProps.deviceUUID, VK_UUID_SIZE) == 0) {
      found = i;
      m.cuDev = d;
    }
  }
  if (found < 0) throw std::runtime_error("nvvfx: no CUDA device matches the Vulkan GPU");
  if (found != 0) fprintf(stderr, "nvvfx: warning: the GPU is CUDA device %d; the effects use the current device\n", found);
  m.cu(a.p_cuDevicePrimaryCtxRetain(&m.cuCtx, m.cuDev), "cuDevicePrimaryCtxRetain");
  m.cu(a.p_cuCtxSetCurrent(m.cuCtx), "cuCtxSetCurrent");
  m.vfx(a.p_NvVFX_CudaStreamCreate(&m.stream), "NvVFX_CudaStreamCreate");
  for (CUevent& e : m.ev) m.cu(a.p_cuEventCreate(&e, CU_EVENT_DEFAULT), "cuEventCreate");

  // Timeline semaphore shared with CUDA.
  {
    VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkExportSemaphoreCreateInfo exp{VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO};
    exp.pNext = &type;
    exp.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
    VkSemaphoreCreateInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    info.pNext = &exp;
    VK_CHECK(vkCreateSemaphore(gpu.device(), &info, nullptr, &m.timeline));
    VkSemaphoreGetFdInfoKHR fdInfo{VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR};
    fdInfo.semaphore = m.timeline;
    fdInfo.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
    int fd = -1;
    VK_CHECK(vkGetSemaphoreFdKHR(gpu.device(), &fdInfo, &fd));
    CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC sd{};
    sd.type = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD;
    sd.handle.fd = fd;
    m.cu(a.p_cuImportExternalSemaphore(&m.cuTimeline, &sd), "cuImportExternalSemaphore");
  }

  const std::string models = config.sdk + "/lib/models";
  if (config.denoise >= 0) {
    m.makeShared(m.dnIn, workW, workH, "nvvfx denoise in");
    m.makeShared(m.dnOut, workW, workH, "nvvfx denoise out");
    m.dn = m.makeVsr(m.dnIn, m.dnOut, (unsigned)config.denoise, models);
    denoise_ = true;
  }
  if (config.sr >= 0 || config.fg >= 0) m.makeShared(m.srIn, workW, workH, "nvvfx shown");
  upW_ = workW;
  upH_ = workH;
  if (config.sr >= 0 && (outW != workW || outH != workH)) {
    m.makeShared(m.up[0], outW, outH, "nvvfx up 0");
    try {
      m.sr = m.makeVsr(m.srIn, m.up[0], (unsigned)config.sr, models);
      m.makeShared(m.up[1], outW, outH, "nvvfx up 1");
      sr_ = true;
      upW_ = outW;
      upH_ = outH;
    } catch (const std::exception& e) {
      fprintf(stderr, "%s (%ux%u -> %ux%u); sr off\n", e.what(), workW, workH, outW, outH);
      m.freeShared(m.up[0]);
    }
  }
  if (!sr_ && config.fg >= 0) {
    m.makeShared(m.up[0], workW, workH, "nvvfx up 0");
    m.makeShared(m.up[1], workW, workH, "nvvfx up 1");
  }
  if (config.fg >= 0) {
    m.makeShared(m.mid, upW_, upH_, "nvvfx mid");
    m.vfx(a.p_NvVFX_CreateEffect("VideoFrameGeneration", &m.fg), "create VideoFrameGeneration");
    m.vfx(a.p_NvVFX_SetCudaStream(m.fg, NVVFX_CUDA_STREAM, m.stream), "VFG stream");
    m.vfx(a.p_NvVFX_SetU32(m.fg, NVVFX_INPUT_WIDTH, upW_), "VFG width");
    m.vfx(a.p_NvVFX_SetU32(m.fg, NVVFX_INPUT_HEIGHT, upH_), "VFG height");
    m.vfx(a.p_NvVFX_SetU32(m.fg, "Mode", (unsigned)config.fg), "VFG mode");
    a.p_NvVFX_SetU32(m.fg, "AutomaticShotChangeDetectionEnabled", 1u);
    m.vfx(a.p_NvVFX_Load(m.fg), "VFG load");
    m.vfx(a.p_NvVFX_SetImage(m.fg, NVVFX_OUTPUT_IMAGE, &m.mid.img), "VFG output");
    m.vfx(a.p_NvVFX_SetU32(m.fg, "FrameMultiplier", 0u), "VFG multiplier");
    fg_ = true;
  }

  fprintf(stderr, "nvvfx: %s%s%s at %ux%u -> %ux%u\n", denoise_ ? "denoise " : "", sr_ ? "sr " : "",
          fg_ ? "fg " : "", workW, workH, upW_, upH_);
}

Nvvfx::~Nvvfx() {
  Impl& m = *impl_;
  if (m.stream) m.api.p_cuStreamSynchronize(m.stream);
  vkDeviceWaitIdle(m.gpu->device());
  for (NvVFX_Handle h : {m.dn, m.sr, m.fg})
    if (h) m.api.p_NvVFX_DestroyEffect(h);
  for (CUevent e : m.ev)
    if (e) m.api.p_cuEventDestroy(e);
  for (Impl::Shared* s : {&m.dnIn, &m.dnOut, &m.srIn, &m.up[0], &m.up[1], &m.mid}) m.freeShared(*s);
  if (m.cuTimeline) m.api.p_cuDestroyExternalSemaphore(m.cuTimeline);
  if (m.timeline) vkDestroySemaphore(m.gpu->device(), m.timeline, nullptr);
  if (m.stream) m.api.p_NvVFX_CudaStreamDestroy(m.stream);
  if (m.cuCtx) m.api.p_cuDevicePrimaryCtxRelease(m.cuDev);
}

const vk::Buffer& Nvvfx::denoiseIn() const { return impl_->dnIn.vk; }
const vk::Buffer& Nvvfx::denoiseOut() const { return impl_->dnOut.vk; }
const vk::Buffer& Nvvfx::srIn() const { return impl_->srIn.vk; }
const vk::Buffer& Nvvfx::up(int i) const { return impl_->up[i & 1].vk; }
const vk::Buffer& Nvvfx::mid() const { return impl_->mid.vk; }
VkSemaphore Nvvfx::timeline() const { return impl_->timeline; }

void Nvvfx::runDenoise(uint64_t wait, uint64_t signal) {
  Impl& m = *impl_;
  m.wait(wait);
  m.api.p_cuEventRecord(m.ev[0], m.stream);
  m.vfx(m.api.p_NvVFX_Run(m.dn, 0), "denoise");
  m.api.p_cuEventRecord(m.ev[1], m.stream);
  m.ranDn = true;
  m.signal(signal);
}

void Nvvfx::runSrFg(uint64_t wait, uint64_t signal, int cur, bool havePrev) {
  Impl& m = *impl_;
  Api& a = m.api;
  m.wait(wait);
  if (sr_) {
    m.vfx(a.p_NvVFX_SetImage(m.sr, NVVFX_OUTPUT_IMAGE, &m.up[cur & 1].img), "VSR output");
    a.p_cuEventRecord(m.ev[2], m.stream);
    m.vfx(a.p_NvVFX_Run(m.sr, 0), "sr");
    a.p_cuEventRecord(m.ev[3], m.stream);
    m.ranSr = true;
  }
  if (fg_ && havePrev) {
    m.vfx(a.p_NvVFX_SetImage(m.fg, NVVFX_INPUT_IMAGE_0, &m.up[(cur ^ 1) & 1].img), "VFG input 0");
    m.vfx(a.p_NvVFX_SetImage(m.fg, NVVFX_INPUT_IMAGE_1, &m.up[cur & 1].img), "VFG input 1");
    a.p_NvVFX_SetU32(m.fg, "ShotChange", 0u);
    a.p_NvVFX_SetF32(m.fg, "Timestep", 0.5f);
    a.p_cuEventRecord(m.ev[4], m.stream);
    m.vfx(a.p_NvVFX_Run(m.fg, 0), "fg");
    a.p_cuEventRecord(m.ev[5], m.stream);
    m.ranFg = true;
  }
  m.signal(signal);
}

Nvvfx::Times Nvvfx::lastTimes() {
  Impl& m = *impl_;
  Times t;
  auto span = [&](bool& ran, CUevent a, CUevent b) {
    float ms = 0;
    if (ran && m.api.p_cuEventElapsedTime(&ms, a, b) == CUDA_SUCCESS) {
      ran = false;
      return ms;
    }
    ran = false;
    return 0.f;
  };
  t.denoise = span(m.ranDn, m.ev[0], m.ev[1]);
  t.sr = span(m.ranSr, m.ev[2], m.ev[3]);
  t.fg = span(m.ranFg, m.ev[4], m.ev[5]);
  return t;
}

#endif
