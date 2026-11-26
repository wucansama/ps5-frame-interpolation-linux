#include "remoteplay.h"

#include "paths.h"

#include <fcntl.h>
#include <linux/futex.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
#include <libavutil/imgutils.h>
}

namespace {

uint64_t nowNs() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

template <typename T> T load(const volatile T* p) { return __atomic_load_n(const_cast<const T*>(p), __ATOMIC_ACQUIRE); }
template <typename T> void store(volatile T* p, T v) { __atomic_store_n(const_cast<T*>(p), v, __ATOMIC_RELEASE); }

std::string avErr(int e) {
  char b[AV_ERROR_MAX_STRING_SIZE] = {};
  av_strerror(e, b, sizeof(b));
  return b;
}

}  // namespace

// ---- shared memory --------------------------------------------------------------------------

RpTapReader::RpTapReader(const std::string& name) {
  const std::string path = "/" + name;
  fd_ = shm_open(path.c_str(), O_RDWR | O_CREAT, 0600);
  if (fd_ < 0) throw std::runtime_error("remoteplay: shm_open " + path + ": " + strerror(errno));
  struct stat st {};
  if (fstat(fd_, &st) == 0 && (size_t)st.st_size < RP_TAP_SIZE && ftruncate(fd_, RP_TAP_SIZE) != 0)
    throw std::runtime_error("remoteplay: ftruncate " + path + ": " + strerror(errno));
  void* p = mmap(nullptr, RP_TAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
  if (p == MAP_FAILED) throw std::runtime_error("remoteplay: mmap " + path + ": " + strerror(errno));
  base_ = static_cast<uint8_t*>(p);
  hdr_ = reinterpret_cast<volatile RpTapHeader*>(base_);
}

RpTapReader::~RpTapReader() {
  if (base_) munmap(base_, RP_TAP_SIZE);
  if (fd_ >= 0) close(fd_);
}

bool RpTapReader::streaming() const {
  return load(&hdr_->magic) == RP_TAP_MAGIC && hdr_->version == RP_TAP_VERSION &&
         hdr_->slot_count == RP_TAP_SLOT_COUNT && hdr_->slot_size == RP_TAP_SLOT_SIZE;
}

uint64_t RpTapReader::writeSeq() const { return load(&hdr_->write_seq); }
uint32_t RpTapReader::futex() const { return load(&hdr_->futex); }

void RpTapReader::wait(uint32_t seen, int timeoutMs) const {
  timespec ts{timeoutMs / 1000, (long)(timeoutMs % 1000) * 1000000L};
  // Shared (not _PRIVATE): the writer is another process.
  syscall(SYS_futex, const_cast<uint32_t*>(&hdr_->futex), FUTEX_WAIT, seen, &ts, nullptr, 0);
}

bool RpTapReader::read(uint64_t n, Frame& out) const {
  const uint8_t* slot = base_ + RP_TAP_HEADER_SIZE + (size_t)(n % RP_TAP_SLOT_COUNT) * RP_TAP_SLOT_SIZE;
  const volatile RpTapSlot* s = reinterpret_cast<const volatile RpTapSlot*>(slot);
  if (load(&s->seq) != n) return false;
  const uint32_t size = std::min<uint32_t>((uint32_t)s->size, RP_TAP_SLOT_SIZE - RP_TAP_SLOT_HEADER);
  out.arrivalNs = s->arrival_ns;
  out.framesLost = s->frames_lost;
  out.recovered = s->recovered != 0;
  out.truncated = (s->flags & RP_TAP_SLOT_TRUNCATED) != 0;
  out.data.resize(size);
  memcpy(out.data.data(), slot + RP_TAP_SLOT_HEADER, size);
  __atomic_thread_fence(__ATOMIC_ACQUIRE);
  return load(&s->seq) == n;   // rewritten while copying: drop it
}

std::vector<uint8_t> RpTapReader::codecHeader(uint32_t session) const {
  const uint32_t size = load(&hdr_->codec_header_size);
  if (!size || size > RP_TAP_CODEC_HEADER_MAX || hdr_->codec_header_session != session) return {};
  std::vector<uint8_t> h(size);
  memcpy(h.data(), const_cast<const uint8_t*>(hdr_->codec_header), size);
  return h;
}

void RpTapReader::heartbeat(uint32_t audioDelayMs) {
  store(&hdr_->audio_delay_ms, std::min<uint32_t>(audioDelayMs, 500));
  store(&hdr_->reader_heartbeat, nowNs());
}

void RpTapReader::requestIdr() { store(&hdr_->idr_request, 1u); }
void RpTapReader::requestWake() { store(&hdr_->wake_request, 1u); }

// ---- decoder --------------------------------------------------------------------------------

struct RpDecoder::Impl {
  AVCodecContext* ctx = nullptr;
  AVBufferRef* device = nullptr;
  AVFrame* frame = nullptr;
  AVPacket* pkt = nullptr;
  std::vector<uint8_t> padded;
};

static AVPixelFormat pickCuda(AVCodecContext*, const AVPixelFormat* fmts) {
  for (const AVPixelFormat* p = fmts; *p != AV_PIX_FMT_NONE; ++p)
    if (*p == AV_PIX_FMT_CUDA) return *p;
  return fmts[0];
}

RpDecoder::RpDecoder(uint32_t codec, const std::string& preferred, uint32_t width, uint32_t height)
    : codec_(codec), width_(width), height_(height), preferred_(preferred) {
  open(preferred != "sw");
}

RpDecoder::~RpDecoder() { close(); }

const char* RpDecoder::name() const { return hw_ ? "NVDEC (CUDA)" : "software"; }

void RpDecoder::close() {
  if (!impl_) return;
  av_frame_free(&impl_->frame);
  av_packet_free(&impl_->pkt);
  avcodec_free_context(&impl_->ctx);
  av_buffer_unref(&impl_->device);
  delete impl_;
  impl_ = nullptr;
}

void RpDecoder::open(bool hw) {
  close();
  impl_ = new Impl;
  const AVCodec* c = avcodec_find_decoder(codec_ == RP_TAP_H264 ? AV_CODEC_ID_H264 : AV_CODEC_ID_HEVC);
  if (!c) throw std::runtime_error("remoteplay: libavcodec has no H.264/HEVC decoder");
  impl_->ctx = avcodec_alloc_context3(c);
  AVCodecContext* ctx = impl_->ctx;
  // No reordering in a Remote Play stream: every packet is one picture, out at once. The PS5 never
  // marks its intra refreshes as keyframes, so show everything (we judge corruption ourselves).
  ctx->flags |= AV_CODEC_FLAG_LOW_DELAY | AV_CODEC_FLAG_OUTPUT_CORRUPT;
  ctx->flags2 |= AV_CODEC_FLAG2_SHOW_ALL;
  hw_ = false;
  if (hw) {
    int e = av_hwdevice_ctx_create(&impl_->device, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0);
    if (e >= 0) {
      ctx->hw_device_ctx = av_buffer_ref(impl_->device);
      ctx->get_format = pickCuda;
      ctx->thread_count = 1;
      hw_ = true;
    } else {
      fprintf(stderr, "remoteplay: no CUDA decoder (%s); decoding in software\n", avErr(e).c_str());
    }
  }
  if (!hw_) {
    ctx->thread_count = 0;
    ctx->thread_type = FF_THREAD_SLICE;   // frame threads would add a frame of delay each
  }
  int e = avcodec_open2(ctx, c, nullptr);
  if (e < 0) throw std::runtime_error("remoteplay: avcodec_open2: " + avErr(e));
  impl_->frame = av_frame_alloc();
  impl_->pkt = av_packet_alloc();
}

static void noFree(void*, uint8_t*) {}

bool RpDecoder::decode(const uint8_t* data, size_t size, int64_t pts, const std::function<uint8_t*(int64_t)>& dst,
                       const std::function<void(const Picture&)>& done) {
  Impl& m = *impl_;
  m.padded.resize(size + AV_INPUT_BUFFER_PADDING_SIZE);
  memcpy(m.padded.data(), data, size);
  memset(m.padded.data() + size, 0, AV_INPUT_BUFFER_PADDING_SIZE);
  m.pkt->data = m.padded.data();
  m.pkt->size = (int)size;
  m.pkt->pts = pts;
  int e = avcodec_send_packet(m.ctx, m.pkt);
  av_packet_unref(m.pkt);
  bool decodeError = false;
  if (e < 0 && e != AVERROR(EAGAIN)) {
    if (e != AVERROR_INVALIDDATA) {
      error_ = "avcodec_send_packet: " + avErr(e);
      fprintf(stderr, "remoteplay: %s; reopening the decoder\n", error_.c_str());
      open(hw_ && preferred_ != "sw");
      return false;
    }
    decodeError = true;   // a damaged packet: the next pictures say whether it mattered
  }
  for (;;) {
    e = avcodec_receive_frame(m.ctx, m.frame);
    if (e == AVERROR(EAGAIN) || e == AVERROR_EOF) break;
    if (e < 0) {
      error_ = "avcodec_receive_frame: " + avErr(e);
      if (hw_) {
        fprintf(stderr, "remoteplay: %s; falling back to software\n", error_.c_str());
        preferred_ = "sw";
        open(false);
        return false;
      }
      break;
    }
    AVFrame* f = m.frame;
    Picture pic;
    pic.pts = f->pts;
    pic.corrupt = decodeError || (f->flags & AV_FRAME_FLAG_CORRUPT) || f->decode_error_flags;
    pic.intra = f->pict_type == AV_PICTURE_TYPE_I || (f->flags & AV_FRAME_FLAG_KEY);
    pic.range = f->color_range == AVCOL_RANGE_JPEG ? 1 : f->color_range == AVCOL_RANGE_MPEG ? 0 : -1;
    uint8_t* out = (uint32_t)f->width == width_ && (uint32_t)f->height == height_ ? dst(f->pts) : nullptr;
    if ((uint32_t)f->width != width_ || (uint32_t)f->height != height_) {
      static bool warned = false;
      if (!warned) fprintf(stderr, "remoteplay: stream is %dx%d, the source is set to %ux%u: set the size to match\n",
                           f->width, f->height, width_, height_);
      warned = true;
    }
    if (out) {
      const size_t lumaBytes = (size_t)width_ * height_;
      bool ok = true;
      if (f->format == AV_PIX_FMT_CUDA) {
        AVHWFramesContext* fc = reinterpret_cast<AVHWFramesContext*>(f->hw_frames_ctx->data);
        if (fc->sw_format != AV_PIX_FMT_NV12) {
          error_ = std::string("unsupported decoder output ") + av_get_pix_fmt_name(fc->sw_format) + " (HDR? set SDR)";
          ok = false;
        } else {
          // DMA straight from the decoder's surface into the GPU-visible capture slot.
          AVFrame* sw = av_frame_alloc();
          sw->format = AV_PIX_FMT_NV12;
          sw->width = (int)width_;
          sw->height = (int)height_;
          sw->data[0] = out;
          sw->data[1] = out + lumaBytes;
          sw->linesize[0] = sw->linesize[1] = (int)width_;
          sw->buf[0] = av_buffer_create(out, (int)(lumaBytes * 3 / 2), noFree, nullptr, 0);
          e = av_hwframe_transfer_data(sw, f, 0);
          av_frame_free(&sw);
          if (e < 0) { error_ = "av_hwframe_transfer_data: " + avErr(e); ok = false; }
        }
      } else if (f->format == AV_PIX_FMT_YUV420P || f->format == AV_PIX_FMT_YUVJ420P) {
        for (uint32_t y = 0; y < height_; ++y) memcpy(out + (size_t)y * width_, f->data[0] + (size_t)y * f->linesize[0], width_);
        uint8_t* uv = out + lumaBytes;
        for (uint32_t y = 0; y < height_ / 2; ++y) {
          const uint8_t* u = f->data[1] + (size_t)y * f->linesize[1];
          const uint8_t* v = f->data[2] + (size_t)y * f->linesize[2];
          uint8_t* d = uv + (size_t)y * width_;
          for (uint32_t x = 0; x < width_ / 2; ++x) { d[2 * x] = u[x]; d[2 * x + 1] = v[x]; }
        }
      } else if (f->format == AV_PIX_FMT_NV12) {
        for (uint32_t y = 0; y < height_ * 3 / 2; ++y) {
          const uint8_t* src = y < height_ ? f->data[0] + (size_t)y * f->linesize[0] : f->data[1] + (size_t)(y - height_) * f->linesize[1];
          memcpy(out + (size_t)y * width_, src, width_);
        }
      } else {
        error_ = std::string("unsupported decoder output ") + av_get_pix_fmt_name((AVPixelFormat)f->format);
        ok = false;
      }
      if (!ok) {
        static bool warned = false;
        if (!warned) fprintf(stderr, "remoteplay: %s\n", error_.c_str());
        warned = true;
        pic.corrupt = true;
      }
      done(pic);
    }
    av_frame_unref(f);
  }
  return true;
}

// ---- the tap process ------------------------------------------------------------------------

RpClient::RpClient(const RemotePlayConfig& c, uint32_t height, uint32_t fps) {
  binary_ = c.tap;
  if (binary_.empty()) binary_ = moonglide::defaultTap();
  if (access(binary_.c_str(), X_OK) != 0)
    throw std::runtime_error("remoteplay: " + binary_ + " not found: build it with tools/chiaki-tap/build.sh");
  const int res = c.res > 0 ? c.res : height >= 1080 ? 1080 : 720;
  const int rate = c.fps > 0 ? (c.fps >= 60 ? 60 : 30) : fps >= 60 ? 60 : 30;
  args_ = {binary_, "--host", c.host, "--shm", c.shm, "--res", std::to_string(res),
           "--fps", std::to_string(rate), "--codec", c.codec, "--reconnect"};
  if (c.bitrate > 0) { args_.push_back("--bitrate"); args_.push_back(std::to_string(c.bitrate)); }
  if (c.wake) args_.push_back("--wake");
  spawn();
}

RpClient::~RpClient() { stop(); }

void RpClient::spawn() {
  std::string line;
  for (const std::string& a : args_) line += " " + a;
  fprintf(stderr, "remoteplay: starting%s\n", line.c_str());
  const pid_t parent = getpid();
  pid_t pid = fork();
  if (pid < 0) throw std::runtime_error(std::string("remoteplay: fork: ") + strerror(errno));
  if (pid == 0) {
    prctl(PR_SET_PDEATHSIG, SIGTERM);
    if (getppid() != parent) _exit(1);
    std::vector<char*> argv;
    for (const std::string& a : args_) argv.push_back(const_cast<char*>(a.c_str()));
    argv.push_back(nullptr);
    execv(binary_.c_str(), argv.data());
    fprintf(stderr, "remoteplay: exec %s: %s\n", binary_.c_str(), strerror(errno));
    _exit(127);
  }
  pid_ = pid;
}

bool RpClient::running() {
  if (pid_ <= 0) return false;
  int status = 0;
  if (waitpid(pid_, &status, WNOHANG) == pid_) {
    fprintf(stderr, "remoteplay: moonglide-chiaki-tap exited (%s %d)\n", WIFEXITED(status) ? "code" : "signal",
            WIFEXITED(status) ? WEXITSTATUS(status) : WTERMSIG(status));
    pid_ = -1;
    return false;
  }
  return true;
}

void RpClient::restart() {
  stop();
  spawn();
}

void RpClient::stop() {
  if (pid_ <= 0) return;
  kill(pid_, SIGTERM);
  for (int i = 0; i < 300; ++i) {   // 3 s to end the session cleanly
    if (waitpid(pid_, nullptr, WNOHANG) == pid_) { pid_ = -1; return; }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  kill(pid_, SIGKILL);
  waitpid(pid_, nullptr, 0);
  pid_ = -1;
}
