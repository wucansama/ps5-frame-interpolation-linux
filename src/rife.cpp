#include "rife.h"

#include <dlfcn.h>
#include <glob.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <vector>

#include "gpu.h"

#ifndef MOONGLIDE_HAVE_RIFE_TRT

struct Rife::Impl {};
Rife::Rife(Gpu&, const RifeConfig&, uint32_t, uint32_t) {
  throw std::runtime_error("built without the TensorRT / CUDA headers (cmake -DMOONGLIDE_TRT_INCLUDE=DIR)");
}
Rife::~Rife() = default;
static const vk::Buffer kNone;
const vk::Buffer& Rife::in(int) const { return kNone; }
const vk::Buffer& Rife::out(int) const { return kNone; }
void Rife::ensureOut(int) {}
VkSemaphore Rife::timeline() const { return VK_NULL_HANDLE; }
void Rife::run(uint64_t, uint64_t, int, const float*, int) {}
float Rife::lastMs() { return 0; }

#else

#include <cuda.h>

#include "NvInfer.h"
#include "NvOnnxParser.h"

namespace {

#define RF_STR2(x) #x
#define RF_STR(x) RF_STR2(x)

#define RF_CU_FUNCS(X)                                                                                    \
  X(cuInit) X(cuDeviceGetCount) X(cuDeviceGet) X(cuDeviceGetUuid) X(cuDeviceGetName)                     \
  X(cuDevicePrimaryCtxRetain) X(cuDevicePrimaryCtxRelease) X(cuCtxSetCurrent) X(cuImportExternalMemory)  \
  X(cuExternalMemoryGetMappedBuffer) X(cuDestroyExternalMemory) X(cuMemFree) X(cuMemAlloc)               \
  X(cuMemcpyHtoD) X(cuImportExternalSemaphore) X(cuDestroyExternalSemaphore)                              \
  X(cuSignalExternalSemaphoresAsync) X(cuWaitExternalSemaphoresAsync) X(cuStreamCreate)                  \
  X(cuStreamDestroy) X(cuStreamSynchronize) X(cuEventCreate) X(cuEventRecord) X(cuEventElapsedTime)      \
  X(cuEventDestroy)

using CreateFn = void* (*)(void*, int32_t);
using ParserFn = void* (*)(void*, void*, int);
using VersionFn = int32_t (*)();

struct Api {
#define RF_DECL(f) decltype(&f) p_##f = nullptr;
  RF_CU_FUNCS(RF_DECL)
#undef RF_DECL
  CreateFn createRuntime = nullptr, createBuilder = nullptr;
  ParserFn createParser = nullptr;
  VersionFn version = nullptr;
  void* cuda = nullptr;
  void* infer = nullptr;
  void* onnx = nullptr;
  std::string libDir;

  static bool exists(const std::string& p) {
    struct stat st;
    return stat(p.c_str(), &st) == 0;
  }

  // TensorRT: --rife-trt / MOONGLIDE_TRT, the RIFE venv's pip TensorRT (tensorrt-cu12-libs, has the
  // builder resources), then the Video Effects SDK copy (runtime only: it can load a cached
  // engine of the same version but not build one).
  static std::string findLib(const std::string& asked) {
    std::vector<std::string> dirs;
    if (!asked.empty()) dirs.push_back(asked);
    if (const char* e = getenv("MOONGLIDE_TRT"); e && *e) dirs.push_back(e);
    if (const char* home = getenv("HOME")) {
      glob_t g{};
      const std::string pat = std::string(home) + "/.local/share/moonglide/rife-venv/lib/python3*/site-packages/tensorrt_libs";
      if (glob(pat.c_str(), 0, nullptr, &g) == 0)
        for (size_t i = 0; i < g.gl_pathc; ++i) dirs.push_back(g.gl_pathv[i]);
      globfree(&g);
    }
    dirs.push_back("/usr/local/VideoFX/external/tensorrt/lib");
    for (const std::string& d : dirs)
      if (exists(d + "/libnvinfer.so.10")) return d;
    throw std::runtime_error("rife: TensorRT 10 not found (run scripts/setup-rife.sh, or `MoonGlide.AppImage setup-rife`, "
                             "or point --rife-trt DIR at a TensorRT 10 lib directory)");
  }

  void load(const std::string& asked) {
    cuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (!cuda) throw std::runtime_error(std::string("libcuda.so.1: ") + dlerror());
    libDir = findLib(asked);
    // libnvinfer dlopen()s its builder resources by soname: load them first, by path.
    glob_t g{};
    if (glob((libDir + "/libnvinfer_builder_resource.so.*").c_str(), 0, nullptr, &g) == 0)
      for (size_t i = 0; i < g.gl_pathc; ++i)
        if (!strstr(g.gl_pathv[i], "_win")) dlopen(g.gl_pathv[i], RTLD_NOW | RTLD_GLOBAL);
    globfree(&g);
    infer = dlopen((libDir + "/libnvinfer.so.10").c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!infer) throw std::runtime_error(std::string("libnvinfer.so.10: ") + dlerror());
    onnx = dlopen((libDir + "/libnvonnxparser.so.10").c_str(), RTLD_NOW | RTLD_GLOBAL);
    auto sym = [](void* h, const char* name) {
      void* p = h ? dlsym(h, name) : nullptr;
      if (!p) throw std::runtime_error(std::string("rife: missing symbol ") + name);
      return p;
    };
#define RF_LOAD(f) p_##f = reinterpret_cast<decltype(p_##f)>(sym(cuda, RF_STR(f)));
    RF_CU_FUNCS(RF_LOAD)
#undef RF_LOAD
    createRuntime = reinterpret_cast<CreateFn>(sym(infer, "createInferRuntime_INTERNAL"));
    createBuilder = reinterpret_cast<CreateFn>(sym(infer, "createInferBuilder_INTERNAL"));
    version = reinterpret_cast<VersionFn>(sym(infer, "getInferLibVersion"));
    if (onnx) createParser = reinterpret_cast<ParserFn>(dlsym(onnx, "createNvOnnxParser_INTERNAL"));
  }
};

class Logger : public nvinfer1::ILogger {
 public:
  void log(Severity s, const char* msg) noexcept override {
    if (s <= Severity::kWARNING) fprintf(stderr, "rife/trt: %s\n", msg);
  }
};

std::string cacheDir() {
  std::string base;
  if (const char* x = getenv("XDG_CACHE_HOME"); x && *x) base = x;
  else base = std::string(getenv("HOME") ? getenv("HOME") : "/tmp") + "/.cache";
  const std::string d = base + "/moonglide/rife";
  std::string p;
  for (size_t i = 1; i <= d.size(); ++i)
    if (i == d.size() || d[i] == '/') mkdir(d.substr(0, i).c_str(), 0755);
  (void)p;
  return d;
}

}  // namespace

struct Rife::Impl {
  Api api;
  Logger logger;
  Gpu* gpu = nullptr;
  CUcontext cuCtx = nullptr;
  CUdevice cuDev = 0;
  CUstream stream = nullptr;

  struct Shared {
    vk::Buffer vk;
    CUexternalMemory ext = nullptr;
    CUdeviceptr ptr = 0;
  };
  Shared in[2], out[kMaxOut];
  CUdeviceptr tBuf[kMaxOut] = {};
  float tLast[kMaxOut] = {-1.0f, -1.0f, -1.0f};
  VkSemaphore timeline = VK_NULL_HANDLE;
  CUexternalSemaphore cuTimeline = nullptr;
  CUevent ev[2] = {};
  bool ran = false;

  nvinfer1::IRuntime* runtime = nullptr;
  nvinfer1::ICudaEngine* engine = nullptr;
  nvinfer1::IExecutionContext* exec = nullptr;

  void cu(CUresult r, const char* what) {
    if (r != CUDA_SUCCESS) throw std::runtime_error(std::string("rife: ") + what + " failed: CUDA error " + std::to_string((int)r));
  }

  void makeShared(Shared& s, VkDeviceSize bytes, const char* label) {
    VkDevice dev = gpu->device();
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
    md.handle.fd = fd;
    md.size = req.size;
    md.flags = CUDA_EXTERNAL_MEMORY_DEDICATED;
    cu(api.p_cuImportExternalMemory(&s.ext, &md), "cuImportExternalMemory");
    CUDA_EXTERNAL_MEMORY_BUFFER_DESC bd{};
    bd.size = bytes;
    cu(api.p_cuExternalMemoryGetMappedBuffer(&s.ptr, s.ext, &bd), "cuExternalMemoryGetMappedBuffer");
  }

  void freeShared(Shared& s) {
    if (s.ptr) api.p_cuMemFree(s.ptr);
    if (s.ext) api.p_cuDestroyExternalMemory(s.ext);
    if (s.vk.buffer) vkDestroyBuffer(gpu->device(), s.vk.buffer, nullptr);
    if (s.vk.memory) vkFreeMemory(gpu->device(), s.vk.memory, nullptr);
    s = Shared{};
  }

  std::vector<char> buildEngine(const std::string& model, bool fp16) {
    if (!api.createParser) throw std::runtime_error("rife: libnvonnxparser.so.10 not found next to libnvinfer; can't build the engine");
    fprintf(stderr, "rife: building the TensorRT engine from %s (once; a minute or two)...\n", model.c_str());
    auto* builder = static_cast<nvinfer1::IBuilder*>(api.createBuilder(&logger, NV_TENSORRT_VERSION));
    if (!builder) throw std::runtime_error("rife: createInferBuilder failed");
    nvinfer1::INetworkDefinition* net = builder->createNetworkV2(0);
    auto* parser = static_cast<nvonnxparser::IParser*>(api.createParser(net, &logger, NV_ONNX_PARSER_VERSION));
    if (!parser || !parser->parseFromFile(model.c_str(), (int)nvinfer1::ILogger::Severity::kWARNING)) {
      delete parser;
      delete net;
      delete builder;
      throw std::runtime_error("rife: parsing " + model + " failed");
    }
    nvinfer1::IBuilderConfig* cfg = builder->createBuilderConfig();
    cfg->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 2ull << 30);
    if (fp16) cfg->setFlag(nvinfer1::BuilderFlag::kFP16);
    cfg->setBuilderOptimizationLevel(4);
    nvinfer1::IHostMemory* plan = builder->buildSerializedNetwork(*net, *cfg);
    std::vector<char> data;
    if (plan) data.assign(static_cast<const char*>(plan->data()), static_cast<const char*>(plan->data()) + plan->size());
    delete plan;
    delete cfg;
    delete parser;
    delete net;
    delete builder;
    if (data.empty()) throw std::runtime_error("rife: building the TensorRT engine failed");
    return data;
  }
};

Rife::Rife(Gpu& gpu, const RifeConfig& config, uint32_t workW, uint32_t workH) : impl_(std::make_unique<Impl>()) {
  Impl& m = *impl_;
  m.gpu = &gpu;
  if (!gpu.externalFd()) throw std::runtime_error("rife: the Vulkan device has no external memory/semaphore fd support");
  padW_ = (workW + 31) / 32 * 32;
  padH_ = (workH + 31) / 32 * 32;
  std::string model = config.model;
  if (model.empty())
    model = std::string(getenv("HOME") ? getenv("HOME") : "") + "/.local/share/moonglide/rife/rife_v4.22_lite_" +
            std::to_string(padW_) + "x" + std::to_string(padH_) + ".onnx";
  struct stat ms;
  if (stat(model.c_str(), &ms) != 0)
    throw std::runtime_error("rife: no model " + model + " (run scripts/setup-rife.sh --size " +
                             std::to_string(padW_) + "x" + std::to_string(padH_) + ", or `MoonGlide.AppImage setup-rife`)");
  m.api.load(config.trtLib);
  Api& a = m.api;

  m.cu(a.p_cuInit(0), "cuInit");
  VkPhysicalDeviceIDProperties idProps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
  VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  props2.pNext = &idProps;
  vkGetPhysicalDeviceProperties2(gpu.physical(), &props2);
  int count = 0;
  m.cu(a.p_cuDeviceGetCount(&count), "cuDeviceGetCount");
  bool found = false;
  for (int i = 0; i < count && !found; ++i) {
    CUdevice d;
    CUuuid uuid;
    if (a.p_cuDeviceGet(&d, i) == CUDA_SUCCESS && a.p_cuDeviceGetUuid(&uuid, d) == CUDA_SUCCESS &&
        memcmp(uuid.bytes, idProps.deviceUUID, VK_UUID_SIZE) == 0) {
      found = true;
      m.cuDev = d;
    }
  }
  if (!found) throw std::runtime_error("rife: no CUDA device matches the Vulkan GPU");
  m.cu(a.p_cuDevicePrimaryCtxRetain(&m.cuCtx, m.cuDev), "cuDevicePrimaryCtxRetain");
  m.cu(a.p_cuCtxSetCurrent(m.cuCtx), "cuCtxSetCurrent");
  m.cu(a.p_cuStreamCreate(&m.stream, CU_STREAM_NON_BLOCKING), "cuStreamCreate");
  for (CUevent& e : m.ev) m.cu(a.p_cuEventCreate(&e, CU_EVENT_DEFAULT), "cuEventCreate");

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

  // Engine: cached per model, size, precision, TensorRT version and GPU.
  char gpuName[128] = {};
  a.p_cuDeviceGetName(gpuName, sizeof(gpuName) - 1, m.cuDev);
  std::string tag = gpuName;
  for (char& c : tag)
    if (!isalnum((unsigned char)c)) c = '_';
  std::string base = model.substr(model.rfind('/') + 1);
  if (base.size() > 5 && base.substr(base.size() - 5) == ".onnx") base.resize(base.size() - 5);
  const std::string enginePath = cacheDir() + "/" + base + (config.fp16 ? "_fp16" : "_fp32") + "_trt" +
                                 std::to_string(a.version()) + "_" + tag + "_" + std::to_string((long long)ms.st_mtime) +
                                 ".engine";
  std::vector<char> plan;
  {
    std::ifstream f(enginePath, std::ios::binary);
    if (f) plan.assign(std::istreambuf_iterator<char>(f), {});
  }
  m.runtime = static_cast<nvinfer1::IRuntime*>(a.createRuntime(&m.logger, NV_TENSORRT_VERSION));
  if (!m.runtime) throw std::runtime_error("rife: createInferRuntime failed");
  if (!plan.empty()) m.engine = m.runtime->deserializeCudaEngine(plan.data(), plan.size());
  if (!m.engine) {
    plan = m.buildEngine(model, config.fp16);
    const std::string tmp = enginePath + ".tmp";
    {
      std::ofstream f(tmp, std::ios::binary);
      f.write(plan.data(), (std::streamsize)plan.size());
    }
    rename(tmp.c_str(), enginePath.c_str());
    m.engine = m.runtime->deserializeCudaEngine(plan.data(), plan.size());
    if (!m.engine) throw std::runtime_error("rife: deserializing the engine failed");
  }
  fprintf(stderr, "rife: engine activation memory %.0f MiB\n", (double)m.engine->getDeviceMemorySizeV2() / 1048576.0);
  m.exec = m.engine->createExecutionContext();
  if (!m.exec) throw std::runtime_error("rife: createExecutionContext failed");

  const VkDeviceSize bytes = (VkDeviceSize)3 * padW_ * padH_ * sizeof(float);
  m.makeShared(m.in[0], bytes, "rife in 0");
  m.makeShared(m.in[1], bytes, "rife in 1");
  m.makeShared(m.out[0], bytes, "rife out");   // x3 / x4 pictures: ensureOut()
  for (CUdeviceptr& t : m.tBuf) m.cu(a.p_cuMemAlloc(&t, sizeof(float)), "cuMemAlloc");
  if (!m.exec->setTensorAddress("t", reinterpret_cast<void*>(m.tBuf[0])) ||
      !m.exec->setTensorAddress("mid", reinterpret_cast<void*>(m.out[0].ptr)))
    throw std::runtime_error("rife: the engine has no t / mid tensors");
  fprintf(stderr, "rife: RIFE 4.22-lite %ux%u (%s, TensorRT %d from %s)\n", padW_, padH_, config.fp16 ? "fp16" : "fp32",
          a.version(), a.libDir.c_str());
}

Rife::~Rife() {
  Impl& m = *impl_;
  if (m.stream) m.api.p_cuStreamSynchronize(m.stream);
  delete m.exec;
  delete m.engine;
  delete m.runtime;
  for (Impl::Shared* s : {&m.in[0], &m.in[1]}) m.freeShared(*s);
  for (Impl::Shared& s : m.out) m.freeShared(s);
  for (CUdeviceptr t : m.tBuf)
    if (t) m.api.p_cuMemFree(t);
  for (CUevent e : m.ev)
    if (e) m.api.p_cuEventDestroy(e);
  if (m.cuTimeline) m.api.p_cuDestroyExternalSemaphore(m.cuTimeline);
  if (m.timeline) vkDestroySemaphore(m.gpu->device(), m.timeline, nullptr);
  if (m.stream) m.api.p_cuStreamDestroy(m.stream);
  if (m.cuCtx) m.api.p_cuDevicePrimaryCtxRelease(m.cuDev);
}

void Rife::ensureOut(int n) {
  Impl& m = *impl_;
  const VkDeviceSize bytes = (VkDeviceSize)3 * padW_ * padH_ * sizeof(float);
  for (int k = 1; k < std::min(n, (int)kMaxOut); ++k)
    if (!m.out[k].ptr) {
      m.api.p_cuCtxSetCurrent(m.cuCtx);
      m.makeShared(m.out[k], bytes, "rife out");
    }
}

const vk::Buffer& Rife::in(int i) const { return impl_->in[i & 1].vk; }
const vk::Buffer& Rife::out(int k) const { return impl_->out[std::clamp(k, 0, kMaxOut - 1)].vk; }
VkSemaphore Rife::timeline() const { return impl_->timeline; }

void Rife::run(uint64_t wait, uint64_t signal, int cur, const float* t, int n) {
  Impl& m = *impl_;
  Api& a = m.api;
  a.p_cuCtxSetCurrent(m.cuCtx);
  CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS wp{};
  wp.params.fence.value = wait;
  m.cu(a.p_cuWaitExternalSemaphoresAsync(&m.cuTimeline, &wp, 1, m.stream), "cuWaitExternalSemaphoresAsync");
  n = std::clamp(n, 1, (int)kMaxOut);
  for (int k = 0; k < n; ++k)
    if (t[k] != m.tLast[k]) {
      // t[k] lives in its own device word, so it only changes when the multiplier does.
      m.api.p_cuStreamSynchronize(m.stream);
      m.cu(a.p_cuMemcpyHtoD(m.tBuf[k], &t[k], sizeof(float)), "cuMemcpyHtoD");
      m.tLast[k] = t[k];
    }
  m.exec->setTensorAddress("img0", reinterpret_cast<void*>(m.in[(cur ^ 1) & 1].ptr));
  m.exec->setTensorAddress("img1", reinterpret_cast<void*>(m.in[cur & 1].ptr));
  a.p_cuEventRecord(m.ev[0], m.stream);
  for (int k = 0; k < n; ++k) {
    m.exec->setTensorAddress("t", reinterpret_cast<void*>(m.tBuf[k]));
    m.exec->setTensorAddress("mid", reinterpret_cast<void*>(m.out[k].ptr));
    if (!m.exec->enqueueV3(reinterpret_cast<cudaStream_t>(m.stream))) fprintf(stderr, "rife: enqueueV3 failed\n");
  }
  a.p_cuEventRecord(m.ev[1], m.stream);
  m.ran = true;
  CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS sp{};
  sp.params.fence.value = signal;
  m.cu(a.p_cuSignalExternalSemaphoresAsync(&m.cuTimeline, &sp, 1, m.stream), "cuSignalExternalSemaphoresAsync");
}

float Rife::lastMs() {
  Impl& m = *impl_;
  if (!m.ran) return 0;
  float ms = 0;
  if (m.api.p_cuEventElapsedTime(&ms, m.ev[0], m.ev[1]) != CUDA_SUCCESS) return 0;
  return ms;
}

#endif
