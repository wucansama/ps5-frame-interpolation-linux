// Offline evaluation of NVIDIA Video Effects SDK features on a MOONGLIDE_REC recording.
// The SDK is the user's own install (NVIDIA SLA): this file only uses its public headers and
// libraries at build/run time; nothing NVIDIA is part of this repository.
//
//   nvvfx_bench <recording.vrec> <outprefix> <effect> <frames,...>
//   effect: vsr:<quality>:<scale>[:strength]   VideoSuperRes (quality 0-4 SR, 8-11 denoise,
//                                              12-15 deblur, 16-19 high-bitrate, 21/23 streaming)
//           denoise:<strength>                 Denoising (1080p only on most GPUs)
//           vfg:<mode>                         VideoFrameGeneration, t = 0.5 between frames
// Writes <outprefix>.<seq>.rgba (raw RGBA8 at the output size; for vfg the frame halfway
// between seq-1 and seq) and prints Run() times.
#include <cstddef>
#include <cstdio>
#include <jpeglib.h>

#include <algorithm>
#include <chrono>
#include <csetjmp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "nvCVImage.h"
#include "nvVideoEffects.h"

#define NVVFX_QUALITY_LEVEL "QualityLevel"
#define VFG_MODE "Mode"
#define VFG_MULT "FrameMultiplier"
#define VFG_TIMESTEP "Timestep"
#define VFG_SHOT "ShotChange"
#define VFG_AUTOSHOT "AutomaticShotChangeDetectionEnabled"

#define CHECK(x)                                                                          \
  do {                                                                                    \
    NvCV_Status s_ = (x);                                                                 \
    if (s_ != NVCV_SUCCESS) {                                                             \
      fprintf(stderr, "%s:%d %s -> %d (%s)\n", __FILE__, __LINE__, #x, (int)s_, NvCV_GetErrorStringFromCode(s_)); \
      exit(1);                                                                            \
    }                                                                                     \
  } while (0)

static std::vector<std::vector<unsigned char>> loadRecording(const char* path, unsigned& w, unsigned& h) {
  FILE* f = fopen(path, "rb");
  if (!f) { perror(path); exit(1); }
  char magic[8];
  unsigned hdr[3];
  if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "MOONGMJ1", 8) || fread(hdr, 4, 3, f) != 3) { fprintf(stderr, "bad recording\n"); exit(1); }
  w = hdr[0];
  h = hdr[1];
  std::vector<std::vector<unsigned char>> frames;
  unsigned len;
  while (fread(&len, 4, 1, f) == 1) {
    std::vector<unsigned char> d(len);
    if (fread(d.data(), 1, len, f) != len) break;
    frames.push_back(std::move(d));
  }
  fclose(f);
  return frames;
}

struct JErr { jpeg_error_mgr mgr; jmp_buf jump; };
static void jexit(j_common_ptr c) { longjmp(reinterpret_cast<JErr*>(c->err)->jump, 1); }

static bool decode(const std::vector<unsigned char>& d, unsigned char* rgba, unsigned w, unsigned h) {
  if (d.size() < 2 || d[0] != 0xFF || d[1] != 0xD8) return false;
  jpeg_decompress_struct c;
  JErr e;
  c.err = jpeg_std_error(&e.mgr);
  e.mgr.error_exit = jexit;
  e.mgr.emit_message = [](j_common_ptr, int) {};
  jpeg_create_decompress(&c);
  if (setjmp(e.jump)) { jpeg_destroy_decompress(&c); return false; }
  jpeg_mem_src(&c, d.data(), d.size());
  jpeg_read_header(&c, TRUE);
  c.out_color_space = JCS_EXT_RGBA;
  jpeg_start_decompress(&c);
  if (c.output_width != w || c.output_height != h) { jpeg_destroy_decompress(&c); return false; }
  while (c.output_scanline < h) {
    unsigned char* row = rgba + (size_t)c.output_scanline * w * 4;
    jpeg_read_scanlines(&c, &row, 1);
  }
  jpeg_finish_decompress(&c);
  jpeg_destroy_decompress(&c);
  return true;
}

int main(int argc, char** argv) {
  if (argc < 5) { fprintf(stderr, "usage: nvvfx_bench rec.vrec outprefix effect frames\n"); return 2; }
  unsigned w, h;
  auto frames = loadRecording(argv[1], w, h);
  const std::string prefix = argv[2], spec = argv[3];
  std::vector<unsigned> dumps;
  for (char* p = argv[4]; *p;) { dumps.push_back(strtoul(p, &p, 10)); if (*p == ',') ++p; }
  const unsigned last = *std::max_element(dumps.begin(), dumps.end());
  const char* sdk = getenv("MOONGLIDE_NVVFX") ? getenv("MOONGLIDE_NVVFX") : "/usr/local/VideoFX";
  const std::string models = std::string(sdk) + "/lib/models";

  std::string kind = spec.substr(0, spec.find(':'));
  std::vector<float> args;
  for (size_t p = spec.find(':'); p != std::string::npos; p = spec.find(':', p + 1)) args.push_back(strtof(spec.c_str() + p + 1, nullptr));

  CUstream stream;
  CHECK(NvVFX_CudaStreamCreate(&stream));
  NvVFX_Handle eff;
  unsigned ow = w, oh = h;
  const bool vfg = kind == "vfg";
  if (kind == "vsr") {
    CHECK(NvVFX_CreateEffect("VideoSuperRes", &eff));
    float scale = args.size() > 1 ? args[1] : 1.f;
    ow = (unsigned)(w * scale + 0.5f);
    oh = (unsigned)(h * scale + 0.5f);
  } else if (kind == "denoise") {
    CHECK(NvVFX_CreateEffect("Denoising", &eff));
  } else if (vfg) {
    CHECK(NvVFX_CreateEffect("VideoFrameGeneration", &eff));
  } else {
    fprintf(stderr, "unknown effect %s\n", kind.c_str());
    return 2;
  }
  NvVFX_SetString(eff, NVVFX_MODEL_DIRECTORY, models.c_str());

  // Host RGBA staging + GPU images. Denoising wants planar BGR f32 on the GPU; Transfer converts.
  std::vector<unsigned char> host((size_t)w * h * 4), outHost((size_t)ow * oh * 4);
  NvCVImage srcCpu, dstCpu, src, prev, dst, tmp;
  NvCVImage_Init(&srcCpu, w, h, w * 4, host.data(), NVCV_RGBA, NVCV_U8, NVCV_INTERLEAVED, NVCV_CPU);
  NvCVImage_Init(&dstCpu, ow, oh, ow * 4, outHost.data(), NVCV_RGBA, NVCV_U8, NVCV_INTERLEAVED, NVCV_CPU);
  memset(&tmp, 0, sizeof(tmp));
  float inScale = 1.f, outScale = 1.f;
  if (kind == "denoise") {
    CHECK(NvCVImage_Alloc(&src, w, h, NVCV_BGR, NVCV_F32, NVCV_PLANAR, NVCV_GPU, 1));
    CHECK(NvCVImage_Alloc(&dst, w, h, NVCV_BGR, NVCV_F32, NVCV_PLANAR, NVCV_GPU, 1));
    inScale = 1.f / 255.f;
    outScale = 255.f;
  } else {
    CHECK(NvCVImage_Alloc(&src, w, h, NVCV_RGBA, NVCV_U8, NVCV_INTERLEAVED, NVCV_GPU, 32));
    CHECK(NvCVImage_Alloc(&dst, ow, oh, NVCV_RGBA, NVCV_U8, NVCV_INTERLEAVED, NVCV_GPU, 32));
  }
  if (vfg) CHECK(NvCVImage_Alloc(&prev, w, h, NVCV_RGBA, NVCV_U8, NVCV_INTERLEAVED, NVCV_GPU, 32));

  CHECK(NvVFX_SetCudaStream(eff, NVVFX_CUDA_STREAM, stream));
  NvVFX_StateObjectHandle state = nullptr;
  if (kind == "vsr") {
    CHECK(NvVFX_SetImage(eff, NVVFX_INPUT_IMAGE, &src));
    CHECK(NvVFX_SetImage(eff, NVVFX_OUTPUT_IMAGE, &dst));
    CHECK(NvVFX_SetU32(eff, NVVFX_QUALITY_LEVEL, (unsigned)args[0]));
    if (args.size() > 2) NvVFX_SetF32(eff, NVVFX_STRENGTH, args[2]);
    CHECK(NvVFX_Load(eff));
  } else if (kind == "denoise") {
    CHECK(NvVFX_SetImage(eff, NVVFX_INPUT_IMAGE, &src));
    CHECK(NvVFX_SetImage(eff, NVVFX_OUTPUT_IMAGE, &dst));
    CHECK(NvVFX_SetF32(eff, NVVFX_STRENGTH, args.empty() ? 0.f : args[0]));
    CHECK(NvVFX_Load(eff));
    CHECK(NvVFX_AllocateState(eff, &state));
    CHECK(NvVFX_SetStateObjectHandleArray(eff, NVVFX_STATE, &state));
  } else {
    CHECK(NvVFX_SetU32(eff, NVVFX_INPUT_WIDTH, w));
    CHECK(NvVFX_SetU32(eff, NVVFX_INPUT_HEIGHT, h));
    CHECK(NvVFX_SetU32(eff, VFG_MODE, args.empty() ? 1u : (unsigned)args[0]));
    CHECK(NvVFX_SetU32(eff, VFG_AUTOSHOT, 1u));
    CHECK(NvVFX_Load(eff));
    CHECK(NvVFX_SetImage(eff, NVVFX_OUTPUT_IMAGE, &dst));
    CHECK(NvVFX_SetU32(eff, VFG_MULT, 0u));
  }

  std::vector<double> times;
  bool havePrev = false;
  // MOONGLIDE_VFG_STEP=2: feed only every 2nd frame, so the generated picture for i sits on the
  // skipped real frame i-1 (ground truth).
  const unsigned step = getenv("MOONGLIDE_VFG_STEP") ? std::max(1, atoi(getenv("MOONGLIDE_VFG_STEP"))) : 1;
  for (unsigned i = 0; i <= last && i < frames.size(); ++i) {
    if (i % step) continue;
    if (!decode(frames[i], host.data(), w, h)) continue;   // the recording's corrupt frames
    if (vfg && havePrev) CHECK(NvCVImage_Transfer(&src, &prev, 1.f, stream, nullptr));
    CHECK(NvCVImage_Transfer(&srcCpu, &src, inScale, stream, &tmp));
    CHECK(NvVFX_CudaStreamSynchronize(stream));
    if (vfg && !havePrev) { havePrev = true; continue; }
    auto t0 = std::chrono::steady_clock::now();
    if (vfg) {
      CHECK(NvVFX_SetImage(eff, NVVFX_INPUT_IMAGE_0, &prev));
      CHECK(NvVFX_SetImage(eff, NVVFX_INPUT_IMAGE_1, &src));
      CHECK(NvVFX_SetU32(eff, VFG_SHOT, 0u));
      CHECK(NvVFX_SetF32(eff, VFG_TIMESTEP, 0.5f));
    }
    CHECK(NvVFX_Run(eff, 0));
    CHECK(NvVFX_CudaStreamSynchronize(stream));
    times.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    if (std::find(dumps.begin(), dumps.end(), i) != dumps.end()) {
      CHECK(NvCVImage_Transfer(&dst, &dstCpu, outScale, stream, &tmp));
      CHECK(NvVFX_CudaStreamSynchronize(stream));
      if (kind == "denoise")
        for (size_t p = 0; p < outHost.size(); p += 4) outHost[p + 3] = 255;
      const std::string name = prefix + "." + std::to_string(i) + ".rgba";
      FILE* f = fopen(name.c_str(), "wb");
      fwrite(outHost.data(), 1, outHost.size(), f);
      fclose(f);
    }
  }
  std::vector<double> warm(times.begin() + std::min<size_t>(times.size(), 10), times.end());
  std::sort(warm.begin(), warm.end());
  double sum = 0;
  for (double t : warm) sum += t;
  if (!warm.empty())
    printf("%s %ux%u -> %ux%u: %zu runs, mean %.2f ms, p50 %.2f, p95 %.2f\n", spec.c_str(), w, h, ow, oh, warm.size(),
           sum / warm.size(), warm[warm.size() / 2], warm[warm.size() * 95 / 100]);
  if (state) NvVFX_DeallocateState(eff, state);
  NvVFX_DestroyEffect(eff);
  return 0;
}
