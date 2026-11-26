#include "capture_v4l2.h"
#include "ms2109.h"

#include <fcntl.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cerrno>
#include <algorithm>
#include <deque>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <stdexcept>

#include "input_lag.h"
#include "jpeg_decode.h"

double monotonicMs() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

const char* pixelFormatName(PixelFormat f) {
  switch (f) {
    case PixelFormat::MJPEG: return "MJPEG";
    case PixelFormat::YUYV: return "YUYV";
    case PixelFormat::NV12: return "NV12";
    default: return "auto";
  }
}

PixelFormat parsePixelFormat(const std::string& s) {
  if (s == "mjpeg" || s == "MJPEG" || s == "mjpg") return PixelFormat::MJPEG;
  if (s == "yuyv" || s == "YUYV") return PixelFormat::YUYV;
  if (s == "nv12" || s == "NV12") return PixelFormat::NV12;
  return PixelFormat::Auto;
}

namespace {

int xioctl(int fd, unsigned long request, void* arg) {
  int r;
  do r = ioctl(fd, request, arg);
  while (r == -1 && errno == EINTR);
  return r;
}

uint32_t fourcc(PixelFormat f) {
  switch (f) {
    case PixelFormat::MJPEG: return V4L2_PIX_FMT_MJPEG;
    case PixelFormat::YUYV: return V4L2_PIX_FMT_YUYV;
    case PixelFormat::NV12: return V4L2_PIX_FMT_NV12;
    default: return 0;
  }
}

constexpr uint32_t kSigW = 64, kSigH = 36;

}  // namespace

V4l2Capture::V4l2Capture(const CaptureConfig& config) : config_(config) {
  if (config.device.rfind("replay:", 0) == 0) {
    // A recording made with MOONGLIDE_REC: "MOONGMJ1", u32 width, height, fps, then per frame
    // u32 length + the MJPEG bytes as the card sent them.
    const std::string path = config.device.substr(7);
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);
    char magic[8];
    uint32_t hdr[3];
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "MOONGMJ1", 8) || fread(hdr, 4, 3, f) != 3) {
      fclose(f);
      throw std::runtime_error(path + " is not a MOONGLIDE_REC recording");
    }
    uint32_t len;
    while (fread(&len, 4, 1, f) == 1) {
      std::vector<uint8_t> jpeg(len);
      if (fread(jpeg.data(), 1, len, f) != len) break;
      replay_.push_back(std::move(jpeg));
    }
    fclose(f);
    if (replay_.empty()) throw std::runtime_error(path + " has no frames");
    synthetic_ = true;
    width_ = hdr[0];
    height_ = hdr[1];
    format_ = PixelFormat::MJPEG;
    fps_ = config.fps != 30 ? config.fps : hdr[2];
    fprintf(stderr, "capture: replay %s, %zu frames %ux%u @ %.0f fps (looping)\n", path.c_str(), replay_.size(),
            width_, height_, fps_);
    return;
  }
  if (config.source == "remoteplay") {
    synthetic_ = true;
    rp_ = true;
    width_ = config.width;
    height_ = config.height;
    if (config.rp.res > 0) {   // the stream's own size: the capture size only matters for the card
      height_ = (uint32_t)config.rp.res;
      width_ = height_ * 16 / 9;
    }
    format_ = PixelFormat::NV12;
    const uint32_t rate = config.rp.fps > 0 ? (uint32_t)config.rp.fps : config.fps;
    fps_ = rate >= 60 ? 60 : 30;
    fprintf(stderr, "capture: PS5 Remote Play (%s, shm %s) %ux%u @ %.0f\n",
            config.rp.host.empty() ? "external tap" : config.rp.host.c_str(), config.rp.shm.c_str(), width_, height_, fps_);
    return;
  }
  if (config.device == "test") {
    synthetic_ = true;
    width_ = config.width;
    height_ = config.height;
    format_ = PixelFormat::MJPEG;   // slot layout: RGBA8
    fps_ = config.fps;
    fprintf(stderr, "capture: synthetic test source %ux%u @ %u fps\n", width_, height_, config.fps);
    return;
  }
  fd_ = open(config.device.c_str(), O_RDWR | O_NONBLOCK);
  if (fd_ < 0) throw std::runtime_error("cannot open " + config.device + ": " + strerror(errno));
  v4l2_capability cap{};
  if (xioctl(fd_, VIDIOC_QUERYCAP, &cap) < 0 || !(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE) ||
      !(cap.capabilities & V4L2_CAP_STREAMING))
    throw std::runtime_error(config.device + " is not a streaming capture device");

  // Auto: the first of NV12 > YUYV > MJPEG that holds the requested size AND rate (a USB2
  // dongle offers YUYV 1080p, but only at 5 fps).
  auto setRate = [&]() {
    v4l2_streamparm parm{};
    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    parm.parm.capture.timeperframe.numerator = 1;
    parm.parm.capture.timeperframe.denominator = config.fps;
    xioctl(fd_, VIDIOC_S_PARM, &parm);
    const auto& tpf = parm.parm.capture.timeperframe;
    return tpf.numerator ? (double)tpf.denominator / tpf.numerator : 0.0;
  };
  bool ok = false;
  double fps = 0;
  if (config.format != PixelFormat::Auto) {
    if ((ok = negotiate(config.format))) fps = setRate();
  } else {
    for (PixelFormat f : {PixelFormat::NV12, PixelFormat::YUYV, PixelFormat::MJPEG}) {
      if (!negotiate(f)) continue;
      fps = setRate();
      if (fps + 0.5 >= (double)config.fps) { ok = true; break; }
    }
  }
  if (!ok)
    throw std::runtime_error("no usable format at " + std::to_string(config.width) + "x" +
                             std::to_string(config.height) + "@" + std::to_string(config.fps) + " on " + config.device);

  fps_ = fps;
  v4l2_requestbuffers req{};
  req.count = 4;
  req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  req.memory = V4L2_MEMORY_MMAP;
  if (xioctl(fd_, VIDIOC_REQBUFS, &req) < 0 || req.count < 2)
    throw std::runtime_error(std::string("VIDIOC_REQBUFS: ") + strerror(errno));
  buffers_.resize(req.count);
  for (uint32_t i = 0; i < req.count; ++i) {
    v4l2_buffer buf{};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = i;
    if (xioctl(fd_, VIDIOC_QUERYBUF, &buf) < 0) throw std::runtime_error("VIDIOC_QUERYBUF failed");
    buffers_[i].length = buf.length;
    buffers_[i].data = mmap(nullptr, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, buf.m.offset);
    if (buffers_[i].data == MAP_FAILED) throw std::runtime_error("mmap of a capture buffer failed");
  }
  fprintf(stderr, "capture: %s %ux%u %s @ %.2f fps, %zu buffers\n", config.device.c_str(), width_, height_,
          pixelFormatName(format_), fps, buffers_.size());
  hidraw_ = format_ == PixelFormat::MJPEG ? ms2109Hidraw(config.device) : "";
  jpegWanted_ = config.jpegQuality;
  if (!hidraw_.empty()) fprintf(stderr, "capture: MS2109 dongle, HID %s\n", hidraw_.c_str());
}

bool V4l2Capture::negotiate(PixelFormat f) {
  v4l2_format fmt{};
  fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  fmt.fmt.pix.width = config_.width;
  fmt.fmt.pix.height = config_.height;
  fmt.fmt.pix.pixelformat = fourcc(f);
  fmt.fmt.pix.field = V4L2_FIELD_NONE;
  if (xioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) return false;
  if (fmt.fmt.pix.pixelformat != fourcc(f) || fmt.fmt.pix.width != config_.width ||
      fmt.fmt.pix.height != config_.height)
    return false;
  width_ = fmt.fmt.pix.width;
  height_ = fmt.fmt.pix.height;
  bytesPerLine_ = fmt.fmt.pix.bytesperline;
  format_ = f;
  return true;
}

V4l2Capture::~V4l2Capture() {
  stop();
  for (auto& b : buffers_)
    if (b.data && b.data != MAP_FAILED) munmap(b.data, b.length);
  if (fd_ >= 0) close(fd_);
}

size_t V4l2Capture::slotBytes() const {
  switch (format_) {
    case PixelFormat::YUYV: return (size_t)width_ * height_ * 2;
    case PixelFormat::NV12: return (size_t)width_ * height_ * 3 / 2;
    default: return (size_t)width_ * height_ * 4;
  }
}

void V4l2Capture::start(const std::vector<void*>& slots) {
  slots_ = slots;
  slotState_.assign(slots.size(), SlotState::Free);
  slotFrames_.assign(slots.size(), {});
  dedupMode_ = config_.dedup;
  if (synthetic_) {
    running_ = true;
    if (rp_ && !config_.rp.host.empty()) rpClient_ = std::make_unique<RpClient>(config_.rp, height_, (uint32_t)fps_);
    thread_ = std::thread([this] { if (rp_) runRemotePlay(); else if (replay_.empty()) runSynthetic(); else runReplay(); });
    return;
  }
  for (uint32_t i = 0; i < buffers_.size(); ++i) {
    v4l2_buffer buf{};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = i;
    if (xioctl(fd_, VIDIOC_QBUF, &buf) < 0) throw std::runtime_error("VIDIOC_QBUF failed");
  }
  v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (xioctl(fd_, VIDIOC_STREAMON, &type) < 0)
    throw std::runtime_error(std::string("VIDIOC_STREAMON: ") + strerror(errno));
  running_ = true;
  thread_ = std::thread([this] { run(); });
}

void V4l2Capture::setJpegQuality(int q) {
  jpegWanted_ = q;
  if (running_ && !hidraw_.empty()) applyJpegAsync();
}

std::string V4l2Capture::jpegQualityStatus() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return hidraw_.empty() ? "n/a" : jpegStatus_;
}

// Writes the wanted table on a helper thread (130 HID round trips); a change while it runs is
// picked up by the same thread.
void V4l2Capture::applyJpegAsync() {
  if (jpegBusy_.exchange(true)) return;
  if (jpegThread_.joinable()) jpegThread_.join();
  jpegThread_ = std::thread([this] {
    for (;;) {
      const int q = jpegWanted_;
      std::string status;
      if (q < 0 && jpegApplied_ < 0) {
        status = "off (factory q75)";
      } else {
        const double t0 = monotonicMs();
        try {
          ms2109SetJpegQuality(hidraw_, q < 0 ? 0 : q);
          jpegApplied_ = q;
          status = q < 0 ? "off (factory q75 restored)" : "q" + std::to_string(q);
          fprintf(stderr, "capture: MS2109 JPEG %s (%.0f ms)\n", status.c_str(), monotonicMs() - t0);
        } catch (const std::exception& e) {
          status = std::string("failed: ") + e.what();
          fprintf(stderr, "capture: %s\n", e.what());
        }
      }
      {
        std::lock_guard<std::mutex> lock(mutex_);
        jpegStatus_ = status;
      }
      jpegBusy_ = false;
      if (jpegWanted_ == q || jpegBusy_.exchange(true)) break;
    }
  });
}

void V4l2Capture::stop() {
  if (!running_.exchange(false)) return;
  if (thread_.joinable()) thread_.join();
  if (jpegThread_.joinable()) jpegThread_.join();
  rpClient_.reset();
  if (!synthetic_) {
    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    xioctl(fd_, VIDIOC_STREAMOFF, &type);
  }
  cv_.notify_all();
}

int V4l2Capture::freeSlotLocked() const {
  for (size_t i = 0; i < slotState_.size(); ++i)
    if (slotState_[i] == SlotState::Free) return (int)i;
  return -1;
}

void V4l2Capture::run() {
  JpegDecoder jpeg;
  bool dumped = false;
  uint64_t frameCounter = 0, lastAnalysed = 0;
  bool jpegStarted = false;
  double lastTs = 0;
  uint32_t dumpCount = 0;
  // MOONGLIDE_REC=path: record the raw MJPEG stream for offline replay (--device replay:path);
  // MOONGLIDE_REC_FRAMES caps it (default 900, 30 s).
  FILE* rec = nullptr;
  uint32_t recLeft = getenv("MOONGLIDE_REC_FRAMES") ? (uint32_t)atoi(getenv("MOONGLIDE_REC_FRAMES")) : 900;
  if (const char* path = getenv("MOONGLIDE_REC"); path && format_ == PixelFormat::MJPEG) {
    if ((rec = fopen(path, "wb"))) {
      const uint32_t hdr[3] = {width_, height_, (uint32_t)std::lround(fps_)};
      fwrite("MOONGMJ1", 1, 8, rec);
      fwrite(hdr, 4, 3, rec);
    } else {
      perror(path);
    }
  }
  FILE* capLog = getenv("MOONGLIDE_CAP_LOG") ? fopen(getenv("MOONGLIDE_CAP_LOG"), "a") : nullptr;
  signature_.assign(kSigW * kSigH, 0);
  previousSignature_.clear();
  while (running_) {
    pollfd p{fd_, POLLIN, 0};
    const int r = poll(&p, 1, 200);
    if (r <= 0) continue;

    // Drain: only the newest queued buffer is worth decoding.
    v4l2_buffer buf{}, newest{};
    bool have = false;
    for (;;) {
      buf = {};
      buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      buf.memory = V4L2_MEMORY_MMAP;
      if (xioctl(fd_, VIDIOC_DQBUF, &buf) < 0) break;
      // Capture index: one per frame on the wire, including drained ones and (from the
      // timestamp gap) ones the driver dropped, so 60->30 dedup keeps its phase.
      const double ts = buf.timestamp.tv_sec * 1e3 + buf.timestamp.tv_usec / 1e3;
      const double period = 1000.0 / std::max(1.0, fps_);
      const uint64_t step = lastTs > 0 ? (uint64_t)std::max(1.0, std::round((ts - lastTs) / period)) : 1;
      frameCounter += step;
      lastTs = ts;
      std::lock_guard<std::mutex> lock(mutex_);
      stats_.dequeued++;
      if (have) {
        stats_.dropped++;
        xioctl(fd_, VIDIOC_QBUF, &newest);
      }
      newest = buf;
      have = true;
    }
    if (!have) continue;
    // The firmware loads its own quant table on stream start: ours goes in once frames flow.
    if (!jpegStarted && !hidraw_.empty()) {
      jpegStarted = true;
      jpegApplied_ = -1;
      applyJpegAsync();
    }
    if (newest.flags & V4L2_BUF_FLAG_ERROR) {
      xioctl(fd_, VIDIOC_QBUF, &newest);
      continue;
    }

    CapturedFrame frame;
    frame.tDequeue = monotonicMs();
    timespec realtime;
    clock_gettime(CLOCK_REALTIME, &realtime);
    frame.tCapture = newest.timestamp.tv_sec * 1e3 + newest.timestamp.tv_usec / 1e3;
    frame.seq = frameCounter;   // not newest.sequence: the MS2109's driver doesn't count it reliably
    frame.format = format_;
    frame.width = width_;
    frame.height = height_;

    int slot;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      slot = freeSlotLocked();
      if (slot < 0 && latest_ >= 0 && slotState_[latest_] == SlotState::Ready) {
        // The consumer is slower than capture: overwrite the unconsumed frame.
        slot = latest_;
        latest_ = -1;
        stats_.dropped++;
      }
      if (slot >= 0) slotState_[slot] = SlotState::Writing;
    }
    if (slot < 0) {
      xioctl(fd_, VIDIOC_QBUF, &newest);
      continue;
    }

    uint8_t* dst = static_cast<uint8_t*>(slots_[slot]);
    const uint8_t* src = static_cast<const uint8_t*>(buffers_[newest.index].data);
    bool ok = true;
    if (rec && recLeft) {
      const uint32_t len = newest.bytesused;
      fwrite(&len, 4, 1, rec);
      fwrite(src, 1, len, rec);
      if (--recLeft == 0) { fclose(rec); rec = nullptr; fprintf(stderr, "capture: recording done\n"); }
    }
    if (format_ == PixelFormat::MJPEG) {
      // The MS2109 now and then hands over a buffer that isn't a JPEG at all (no SOI marker,
      // a torn USB transfer): skip it without bothering libjpeg.
      ok = newest.bytesused > 2 && src[0] == 0xFF && src[1] == 0xD8 &&
           jpeg.decode(src, newest.bytesused, dst, width_, height_);
    } else if (bytesPerLine_ == width_ * (format_ == PixelFormat::YUYV ? 2u : 1u)) {
      memcpy(dst, src, std::min<size_t>(slotBytes(), newest.bytesused));
    } else {
      // Padded rows: repack tightly (NV12 chroma plane follows the luma at the same stride).
      const uint32_t rowBytes = width_ * (format_ == PixelFormat::YUYV ? 2u : 1u);
      const uint32_t rows = format_ == PixelFormat::YUYV ? height_ : height_ * 3 / 2;
      for (uint32_t y = 0; y < rows; ++y) memcpy(dst + (size_t)y * rowBytes, src + (size_t)y * bytesPerLine_, rowBytes);
    }
    xioctl(fd_, VIDIOC_QBUF, &newest);
    frame.tReady = monotonicMs();
    frame.decodeMs = frame.tReady - frame.tDequeue;

    if (ok) {
      analyse(dst, frame);
      const bool diffValid = lastAnalysed + 1 == frame.seq;
      lastAnalysed = frame.seq;
      if (dedupDrop(frame.seq, frame, diffValid)) {
        std::lock_guard<std::mutex> lock(mutex_);
        slotState_[slot] = SlotState::Free;
        continue;
      }
      // MOONGLIDE_DUMP=path: the 90th decoded frame, raw slot bytes (debugging colour/range).
      if (const char* dump = getenv("MOONGLIDE_DUMP"); dump && ++dumpCount >= 90 && !dumped) {
        if (FILE* f = fopen(dump, "wb")) { fwrite(dst, 1, slotBytes(), f); fclose(f); fprintf(stderr, "dump: %s\n", dump); } else perror(dump);
        dumped = true;
      }
      // MOONGLIDE_CAP_LOG: "<realtime s at DQBUF> <mean luma>" per frame, for brightness-step
      // latency tests against a screen sampler (every 32nd pixel in x and y).
      if (capLog) {
        uint64_t sum = 0, n = 0;
        for (uint32_t y = 0; y < height_; y += 32)
          for (uint32_t x = 0; x < width_; x += 32) {
            if (format_ == PixelFormat::MJPEG) {
              const uint8_t* p = dst + ((size_t)y * width_ + x) * 4;
              sum += (uint32_t)(p[0] * 54 + p[1] * 183 + p[2] * 19) >> 8;
            } else if (format_ == PixelFormat::YUYV) {
              sum += dst[((size_t)y * width_ + x) * 2];
            } else {
              sum += dst[(size_t)y * width_ + x];
            }
            ++n;
          }
        fprintf(capLog, "%.4f %.2f\n", realtime.tv_sec + realtime.tv_nsec / 1e9, n ? (double)sum / n : 0.0);
        fflush(capLog);
      }
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (!ok) {
      stats_.decodeErrors++;
      if (error_.empty()) error_ = jpeg.error().empty() ? "no JPEG SOI marker" : jpeg.error();
      slotState_[slot] = SlotState::Free;
      continue;
    }
    frame.slot = slot;
    if (latest_ >= 0 && latest_ != slot && slotState_[latest_] == SlotState::Ready) {
      slotState_[latest_] = SlotState::Free;
      stats_.dropped++;
    }
    slotState_[slot] = SlotState::Ready;
    slotFrames_[slot] = frame;
    latest_ = slot;
    cv_.notify_all();
  }
}

bool V4l2Capture::acquire(CapturedFrame& out, int timeoutMs) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (!cv_.wait_for(lock, std::chrono::milliseconds(timeoutMs), [&] { return latest_ >= 0 || !running_; }))
    return false;
  if (latest_ < 0) return false;
  out = slotFrames_[latest_];
  slotState_[latest_] = SlotState::Held;
  latest_ = -1;
  cv_.notify_all();   // a lockstep replay waits for this
  return true;
}

void V4l2Capture::release(int slot) {
  if (slot < 0) return;
  std::lock_guard<std::mutex> lock(mutex_);
  if (slotState_[slot] == SlotState::Held) slotState_[slot] = SlotState::Free;
}

V4l2Capture::Stats V4l2Capture::stats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return stats_;
}

// Per-frame content signature (for dedup) and level counts (for range detection).
void V4l2Capture::analyse(const uint8_t* dst, CapturedFrame& frame) {
  // Content signature: a 64x36 grid of luma-ish samples (G for RGBA, Y for YUV).
  for (uint32_t sy = 0; sy < kSigH; ++sy) {
    const uint32_t y = (sy * height_ + height_ / 2) / kSigH;
    for (uint32_t sx = 0; sx < kSigW; ++sx) {
      const uint32_t x = (sx * width_ + width_ / 2) / kSigW;
      // Mean of an 8x8 patch: single pixels would let the card's (or the test's) noise
      // drown out real content changes.
      uint32_t sum = 0;
      for (uint32_t dy = 0; dy < 8; ++dy)
        for (uint32_t dx = 0; dx < 8; ++dx) {
          const size_t i = (size_t)std::min(y + dy, height_ - 1) * width_ + std::min(x + dx, width_ - 1);
          if (format_ == PixelFormat::MJPEG) sum += dst[i * 4 + 1];
          else if (format_ == PixelFormat::YUYV) sum += dst[i * 2];
          else sum += dst[i];
        }
      signature_[sy * kSigW + sx] = (uint8_t)((sum + 32) / 64);
    }
  }
  if (previousSignature_.size() == signature_.size()) {
    uint32_t sum = 0;
    for (size_t i = 0; i < signature_.size(); ++i) sum += (uint32_t)std::abs((int)signature_[i] - (int)previousSignature_[i]);
    frame.diff = (float)sum / (float)signature_.size();
  }
  previousSignature_ = signature_;
  // Levels for range detection.
  for (uint32_t y = 8; y < height_; y += 16)
    for (uint32_t x = 8; x < width_; x += 16) {
      uint32_t l;
      if (format_ == PixelFormat::MJPEG) {
        const uint8_t* p = dst + ((size_t)y * width_ + x) * 4;
        l = (uint32_t)(p[0] * 54 + p[1] * 183 + p[2] * 19) >> 8;
      } else if (format_ == PixelFormat::YUYV) {
        l = dst[((size_t)y * width_ + x) * 2];
      } else {
        l = dst[(size_t)y * width_ + x];
      }
      frame.levelSamples++;
      if (l < 8) frame.levelBelow8++;
      else if (l > 248) frame.levelAbove248++;
      else if (l >= 12 && l <= 20) frame.levelNear16++;
    }
}

bool PhaseDedup::feed(uint64_t idx, float diff, bool diffValid) {
  const int parity = (int)(idx & 1);
  if (diffValid) floor = samples == 0 || diff < floor ? diff : floor + (diff - floor) * 0.002f;
  if (diffValid) {
    // Fast enough to re-lock within ~half a second after a hitch shifts the phase.
    ema[parity] = samples < 4 ? diff : ema[parity] * 0.8f + diff * 0.2f;
    samples++;
  }
  if (samples < 8) return false;
  const int low = ema[0] <= ema[1] ? 0 : 1;
  const float ratio = ema[low] / std::max(1e-3f, ema[1 - low]);
  if (!locked && ratio < 0.5f) { locked = true; dupPhase = low; }
  else if (locked && low != dupPhase && ratio < 0.5f) dupPhase = low;   // a hitch moved the phase
  else if (locked && ratio > 0.7f && ema[dupPhase] > 2.f * floor + 0.1f) locked = false;   // real 60 fps
  // A static scene makes both phases look alike: stay locked (worst case the wrong phase
  // costs one frame of latency; a duplicate carries the same picture).
  return locked && parity == dupPhase;
}

bool V4l2Capture::dedupDrop(uint64_t idx, const CapturedFrame& frame, bool diffValid) {
  const int mode = dedupMode_.load();
  const bool active = mode == 1 || (mode == -1 && fps_ > 50.0);
  bool drop = false;
  if (active) drop = dedup_.feed(idx, frame.diff, diffValid);
  else dedup_ = {};
  std::lock_guard<std::mutex> lock(mutex_);
  stats_.dedupActive = active;
  stats_.dedupLocked = dedup_.locked;
  stats_.dedupPhase = dedup_.dupPhase;
  const int low = dedup_.ema[0] <= dedup_.ema[1] ? 0 : 1;
  stats_.dedupRatio = dedup_.samples >= 8 ? dedup_.ema[low] / std::max(1e-3f, dedup_.ema[1 - low]) : 1.f;
  if (drop) stats_.duplicates++;
  return drop;
}

std::string V4l2Capture::error() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return error_;
}

void V4l2Capture::runReplay() {
  JpegDecoder jpeg;
  const bool lockstep = getenv("MOONGLIDE_REPLAY_LOCKSTEP") != nullptr;
  signature_.assign(kSigW * kSigH, 0);
  previousSignature_.clear();
  uint64_t seq = 0;
  FILE* capLog = getenv("MOONGLIDE_CAP_LOG") ? fopen(getenv("MOONGLIDE_CAP_LOG"), "a") : nullptr;
  const double period = 1000.0 / std::max(1.0, fps_);
  // MOONGLIDE_REPLAY_STEP=2: only every 2nd recorded frame (seq keeps the recorded index), so a
  // generated in-between picture can be compared with the real frame that was skipped.
  const uint64_t step = getenv("MOONGLIDE_REPLAY_STEP") ? std::max(1ul, strtoul(getenv("MOONGLIDE_REPLAY_STEP"), nullptr, 10)) : 1;
  double next = monotonicMs();
  while (running_) {
    if (seq % step) { ++seq; continue; }
    next += period;
    const double wait = next - monotonicMs();
    if (wait > 0) std::this_thread::sleep_for(std::chrono::microseconds((int64_t)(wait * 1000)));
    CapturedFrame frame;
    frame.tCapture = frame.tDequeue = monotonicMs();
    frame.seq = seq;
    frame.format = format_;
    frame.width = width_;
    frame.height = height_;
    if (lockstep) {
      // Offline evaluation: every recorded frame reaches the consumer, in order.
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait_for(lock, std::chrono::milliseconds(2000), [&] { return latest_ < 0 || !running_; });
      next = monotonicMs();
    }
    const std::vector<uint8_t>& src = replay_[seq++ % replay_.size()];
    int slot;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stats_.dequeued++;
      slot = freeSlotLocked();
      if (slot < 0 && latest_ >= 0) { slot = latest_; latest_ = -1; stats_.dropped++; }
      if (slot < 0) continue;
      slotState_[slot] = SlotState::Writing;
    }
    uint8_t* dst = static_cast<uint8_t*>(slots_[slot]);
    const bool ok = src.size() > 2 && src[0] == 0xFF && src[1] == 0xD8 &&
                    jpeg.decode(src.data(), src.size(), dst, width_, height_);
    if (ok) analyse(dst, frame);
    if (ok && capLog) {
      // Same format as the device path: "<realtime s when the frame is published> <mean luma>".
      timespec realtime;
      clock_gettime(CLOCK_REALTIME, &realtime);
      uint64_t sum = 0, n = 0;
      for (uint32_t y = 0; y < height_; y += 32)
        for (uint32_t x = 0; x < width_; x += 32, ++n) {
          const uint8_t* q = dst + ((size_t)y * width_ + x) * 4;
          sum += (uint32_t)(q[0] * 54 + q[1] * 183 + q[2] * 19) >> 8;
        }
      fprintf(capLog, "%.4f %.2f\n", realtime.tv_sec + realtime.tv_nsec / 1e9, n ? (double)sum / n : 0.0);
      fflush(capLog);
    }
    frame.tReady = monotonicMs();
    frame.decodeMs = frame.tReady - frame.tDequeue;
    if (!ok || dedupDrop(frame.seq, frame, frame.seq > 0 && step == 1)) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!ok) stats_.decodeErrors++;
      slotState_[slot] = SlotState::Free;
      continue;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    frame.slot = slot;
    if (latest_ >= 0 && latest_ != slot && slotState_[latest_] == SlotState::Ready) slotState_[latest_] = SlotState::Free;
    slotState_[slot] = SlotState::Ready;
    slotFrames_[slot] = frame;
    latest_ = slot;
    cv_.notify_all();
  }
}

// A moving gradient + bars with film-grain noise: enough structure and noise to see NR work
// and to exercise the timing path without a device.
void V4l2Capture::runSynthetic() {
  uint32_t rng = 12345;
  const int noise = getenv("MOONGLIDE_TEST_NOISE") ? atoi(getenv("MOONGLIDE_TEST_NOISE")) : 64;
  // MOONGLIDE_TEST_REPEAT=N: each picture is shown N times (fresh noise each time), like a 30 fps
  // game on a 60 Hz card; MOONGLIDE_TEST_PHASE shifts where the repeats start.
  const uint64_t repeat = std::max(1, getenv("MOONGLIDE_TEST_REPEAT") ? atoi(getenv("MOONGLIDE_TEST_REPEAT")) : 1);
  const uint64_t phase = getenv("MOONGLIDE_TEST_PHASE") ? (uint64_t)atoi(getenv("MOONGLIDE_TEST_PHASE")) : 0;
  uint64_t seq = 0;
  signature_.assign(kSigW * kSigH, 0);
  previousSignature_.clear();
  const double period = 1000.0 / std::max(1u, config_.fps);
  double next = monotonicMs();
  while (running_) {
    next += period;
    const double wait = next - monotonicMs();
    if (wait > 0) std::this_thread::sleep_for(std::chrono::microseconds((int64_t)(wait * 1000)));
    CapturedFrame frame;
    frame.tCapture = frame.tDequeue = monotonicMs();
    frame.seq = seq++;
    frame.format = format_;
    frame.width = width_;
    frame.height = height_;
    int slot;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stats_.dequeued++;
      slot = freeSlotLocked();
      if (slot < 0 && latest_ >= 0) { slot = latest_; latest_ = -1; stats_.dropped++; }
      if (slot < 0) continue;
      slotState_[slot] = SlotState::Writing;
    }
    uint32_t* px = static_cast<uint32_t*>(slots_[slot]);
    const uint32_t shift = (uint32_t)((frame.seq + phase) / repeat * 4);
    for (uint32_t y = 0; y < height_; ++y)
      for (uint32_t x = 0; x < width_; ++x) {
        rng = rng * 1664525u + 1013904223u;
        const int n = noise > 0 ? (int)((rng >> 8) % (uint32_t)noise) - noise / 2 : 0;
        const uint32_t xs = x + shift;
        int r = (int)(x * 255 / width_), g = (int)(y * 255 / height_), b = ((xs / 64) & 1) ? 200 : 60;
        if (((y / 96) & 1) && ((xs / 24) % 5 == 0)) r = g = b = 240;
        r = std::clamp(r + n, 0, 255); g = std::clamp(g + n, 0, 255); b = std::clamp(b + n, 0, 255);
        px[(size_t)y * width_ + x] = (uint32_t)r | ((uint32_t)g << 8) | ((uint32_t)b << 16) | 0xFF000000u;
      }
    analyse(reinterpret_cast<const uint8_t*>(px), frame);
    if (dedupDrop(frame.seq, frame, frame.seq > 0)) {
      std::lock_guard<std::mutex> lock(mutex_);
      slotState_[slot] = SlotState::Free;
      continue;
    }
    frame.tReady = monotonicMs();
    frame.decodeMs = frame.tReady - frame.tDequeue;
    std::lock_guard<std::mutex> lock(mutex_);
    frame.slot = slot;
    if (latest_ >= 0 && latest_ != slot && slotState_[latest_] == SlotState::Ready) slotState_[latest_] = SlotState::Free;
    slotState_[slot] = SlotState::Ready;
    slotFrames_[slot] = frame;
    latest_ = slot;
    cv_.notify_all();
  }
}

V4l2Capture::RemotePlayStats V4l2Capture::remotePlayStats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return rpStats_;
}

// PS5 Remote Play through the frame tap (remoteplay.h): every compressed frame is decoded in
// order (a skipped one breaks the references), each picture goes into a slot as NV12. Repeats of
// a 30 fps game are told by their size (a repeat costs the encoder almost nothing), through the
// same phase dedup as the dongle's image diff. Pictures after a network loss are flagged corrupt
// until the PS5 repairs the picture, so the frame generator holds instead of blending them.
void V4l2Capture::runRemotePlay() {
  std::unique_ptr<RpTapReader> tap;
  try {
    tap = std::make_unique<RpTapReader>(config_.rp.shm);
  } catch (const std::exception& e) {
    std::lock_guard<std::mutex> lock(mutex_);
    error_ = e.what();
    rpStats_.state = e.what();
    return;
  }
  volatile RpTapHeader* hdr = tap->header();
  std::unique_ptr<RpDecoder> dec;
  signature_.assign(kSigW * kSigH, 0);
  previousSignature_.clear();
  uint32_t session = 0;
  bool haveSession = false, joined = false, damaged = true;
  uint64_t next = 0, seq = 0;
  double joinTime = 0, lastIdr = -1e9, lastStats = 0, kbStart = monotonicMs(), clientDied = 0;
  uint64_t kbBytes = 0;
  RpTapReader::Frame pkt;
  // Input lag (see RemotePlayStats): sizes of recent frames give the baseline of a still picture.
  uint32_t inputSeqSeen = hdr->input_seq;
  uint64_t pressNs = 0;
  double pressBaseline = 0;
  std::deque<uint32_t> recentSizes;
  std::deque<double> lagSamples, lagTimes;   // a sample counts for 2 minutes
  // Camera-pan method (input_lag.h): the tap publishes when the right stick left neutral.
  moonglide::CameraLag camLag;
  uint32_t stickSeqSeen = __atomic_load_n(&((volatile RpTapStick*)((uint8_t*)hdr + RP_TAP_STICK_OFFSET))->seq, __ATOMIC_ACQUIRE);
  int rejectRun = 0;
  const bool lagDbg = getenv("MOONGLIDE_LAGDBG") != nullptr;
  camLag.dbg = lagDbg;
  double camCostSum = 0;
  uint64_t camCostN = 0;
  auto addSample = [&](double dt, uint64_t arrivalNs, const char* method) {
    const double nowMs = arrivalNs / 1e6;
    if (lagDbg) fprintf(stderr, "inputlag: sample %.1f ms (%s)\n", dt, method);
    if (lagSamples.size() >= 5) {   // an outlier (a coincidence) must not move the median; three in a row are real
      std::vector<double> m(lagSamples.begin(), lagSamples.end());
      std::nth_element(m.begin(), m.begin() + m.size() / 2, m.end());
      const double med = m[m.size() / 2];
      if (std::fabs(dt - med) > std::max(80.0, 1.5 * med) && ++rejectRun < 3) return;
    }
    rejectRun = 0;
    lagSamples.push_back(dt);
    lagTimes.push_back(nowMs);
    while (!lagSamples.empty() && (lagSamples.size() > 15 || nowMs - lagTimes.front() > 120000)) {
      lagSamples.pop_front();
      lagTimes.pop_front();
    }
    std::vector<double> v(lagSamples.begin(), lagSamples.end());
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    std::lock_guard<std::mutex> lock(mutex_);
    rpStats_.inputLagMs = v[v.size() / 2];
    rpStats_.inputLagLastMs = dt;
    rpStats_.inputLagSamples = lagSamples.size() >= 2 ? (int)lagSamples.size() : 0;   // shown from 2 samples on
    rpStats_.inputLagMethod = method;
  };
  auto pollStick = [&]() {
    volatile RpTapStick* sk = (volatile RpTapStick*)((uint8_t*)hdr + RP_TAP_STICK_OFFSET);
    const uint32_t ss = __atomic_load_n(&sk->seq, __ATOMIC_ACQUIRE);
    if (ss != stickSeqSeen) {
      stickSeqSeen = ss;
      camLag.stick(sk->onset_ns, sk->x, sk->y);
    }
  };
  auto cameraLag = [&](const RpTapReader::Frame& f, const uint8_t* luma, bool intra) {
    pollStick();
    const double t0 = lagDbg ? monotonicMs() : 0;
    double dt = 0;
    if (camLag.frame(f.arrivalNs, luma, width_, height_, intra, dt)) addSample(dt, f.arrivalNs, "camera");
    if (lagDbg) {
      camCostSum += monotonicMs() - t0;
      if (++camCostN % 600 == 0) fprintf(stderr, "inputlag: camera detector %.3f ms/frame\n", camCostSum / 600), camCostSum = 0;
    }
  };
  auto inputLag = [&](const RpTapReader::Frame& f) {
    const uint32_t is = __atomic_load_n(&hdr->input_seq, __ATOMIC_ACQUIRE);
    if (is != inputSeqSeen) {
      inputSeqSeen = is;
      pressNs = __atomic_load_n(&hdr->input_ns, __ATOMIC_ACQUIRE);
      std::vector<uint32_t> v(recentSizes.begin(), recentSizes.end());
      pressBaseline = 0;
      if (v.size() >= 8) {
        // 80th percentile, not the median: a 30 fps game in a 60 fps stream alternates tiny repeats
        // with real frames, and a real frame must not count as "the press changed the picture".
        std::sort(v.begin(), v.end());
        pressBaseline = v[v.size() * 4 / 5];
        // Only a still picture can be measured: if the real frames vary a lot (gameplay), a big
        // frame after the press says nothing about the press. Real frames = the upper half.
        const double lo = v[v.size() / 2], hi = v[v.size() - 1];
        if (hi > 1.5 * std::max(lo, 1.0) && hi > lo + 2048.0) pressBaseline = 0;
        if (hi > 16384.0) pressBaseline = 0;   // a still picture codes to small frames; gameplay doesn't
      }
      if (pressBaseline <= 0) pressNs = 0;
    }
    const uint32_t size = (uint32_t)f.data.size();
    // Keyframes are big whatever happened on screen: never a sample, never in the baseline.
    bool key = false;
    for (size_t i = 0; i + 4 < f.data.size() && i < 256; ++i) {
      if (f.data[i] == 0 && f.data[i + 1] == 0 && f.data[i + 2] == 1) {
        const uint8_t h = f.data[i + 3];
        const bool hevc = hdr->codec != RP_TAP_H264;
        const int type = hevc ? (h >> 1) & 0x3f : h & 0x1f;
        if (hevc ? (type >= 16 && type <= 23) : type == 5) { key = true; break; }
        if (hevc ? type < 32 : (type >= 1 && type <= 5)) break;   // the first slice decides
      }
    }
    if (key) return;
    if (pressNs && f.arrivalNs > pressNs) {
      const double dt = (f.arrivalNs - pressNs) / 1e6;
      if (dt > 250) {
        pressNs = 0;   // nothing visibly changed (or the picture moves anyway)
      } else if (size > std::max(3.0 * pressBaseline, pressBaseline + 8192.0)) {
        pressNs = 0;
        addSample(dt, f.arrivalNs, "button");
      }
    }
    recentSizes.push_back(size);
    if (recentSizes.size() > 30) recentSizes.pop_front();
  };
  auto requestIdr = [&](double now) {
    if (now - lastIdr < 500) return;
    tap->requestIdr();
    lastIdr = now;
    std::lock_guard<std::mutex> lock(mutex_);
    rpStats_.idrRequests++;
  };
  while (running_) {
    const uint32_t fut = tap->futex();
    tap->heartbeat(rpAudioDelay_.load());
    if (rpWake_.exchange(false)) tap->requestWake();
    const double now = monotonicMs();
    if (now - lastStats > 250) {
      lastStats = now;
      static const char* kStates[] = {"tap idle", "waking the PS5", "connecting", "streaming", "session ended", "PS5 in rest mode"};
      const uint32_t st = hdr->state;
      char status[129] = {};
      memcpy(status, const_cast<const char*>(hdr->status), 128);
      const bool alive = !rpClient_ || rpClient_->running();
      std::lock_guard<std::mutex> lock(mutex_);
      rpStats_.tapState = !alive ? -1 : hdr->writer_pid == 0 ? -2 : (int)st;
      rpStats_.tapStatus = status;
      rpStats_.rttMs = hdr->rtt_us / 1000.0;
      rpStats_.state = !alive ? "moonglide-chiaki-tap not running"
                       : hdr->writer_pid == 0 ? "waiting for moonglide-chiaki-tap"
                                              : std::string(st < 6 ? kStates[st] : "?") + (status[0] ? ": " + std::string(status) : "");
      rpStats_.audioHeldMs = hdr->audio_held_ms;
      rpStats_.audioOutMs = hdr->audio_out_ms;
      rpStats_.bitrate = hdr->bitrate_kbps;
      rpStats_.decoder = dec ? dec->name() : "";
      if (now - kbStart > 2000) {
        rpStats_.kbps = kbBytes * 8.0 / (now - kbStart);
        kbBytes = 0;
        kbStart = now;
      }
      // The tap reconnects by itself; if the process itself died (bad config, crash), restart it.
      if (rpClient_ && !alive) {
        if (clientDied == 0) clientDied = now;
        else if (now - clientDied > 5000) { clientDied = 0; rpClient_->restart(); }
      }
    }
    if (!tap->streaming()) {
      haveSession = false;
      tap->wait(fut, 50);
      continue;
    }
    if (!haveSession || hdr->session != session) {
      session = hdr->session;
      haveSession = true;
      joined = false;
      damaged = true;
      joinTime = now;
      camLag.reset();
      try {
        dec = std::make_unique<RpDecoder>(hdr->codec, config_.rp.decoder, width_, height_);
      } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lock(mutex_);
        error_ = e.what();
        rpStats_.state = e.what();
        tap->wait(fut, 1000);
        haveSession = false;
        continue;
      }
      // Joining a running stream: its codec header was sent once, at its start.
      const uint64_t n = tap->writeSeq();
      next = n < RP_TAP_SLOT_COUNT / 2 ? 0 : n;
      if (next > 0) {
        const std::vector<uint8_t> h = tap->codecHeader(session);
        if (!h.empty()) dec->decode(h.data(), h.size(), -1, [](int64_t) { return (uint8_t*)nullptr; }, [](const RpDecoder::Picture&) {});
        requestIdr(now);
      }
      fprintf(stderr, "remoteplay: stream %ux%u %s @ %u, %s decoder, from frame %llu\n", hdr->width, hdr->height,
              hdr->codec == RP_TAP_H264 ? "H.264" : "HEVC", hdr->max_fps, dec->name(), (unsigned long long)next);
    }
    const uint64_t n = tap->writeSeq();
    if (next >= n) {
      tap->wait(fut, 20);
      continue;
    }
    if (n - next > RP_TAP_SLOT_COUNT - 4) {   // fell behind: the skipped frames' references are gone
      next = n - 1;
      damaged = true;
      requestIdr(now);
    }
    for (; next < n && running_; ++next) {
      const double tRead = monotonicMs();
      if (!tap->read(next, pkt)) { damaged = true; requestIdr(tRead); continue; }
      kbBytes += pkt.data.size();
      inputLag(pkt);
      if (pkt.framesLost > 0 || pkt.truncated) {
        damaged = true;
        std::lock_guard<std::mutex> lock(mutex_);
        rpStats_.lost += pkt.framesLost > 0 ? (uint64_t)pkt.framesLost : 1;
      }
      if (pkt.recovered) damaged = false;
      int slot = -1;
      auto dst = [&](int64_t) -> uint8_t* {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_.dequeued++;
        slot = freeSlotLocked();
        if (slot < 0 && latest_ >= 0) { slot = latest_; latest_ = -1; stats_.dropped++; }
        if (slot < 0) return nullptr;
        slotState_[slot] = SlotState::Writing;
        return static_cast<uint8_t*>(slots_[slot]);
      };
      auto done = [&](const RpDecoder::Picture& pic) {
        if (pic.intra && !pic.corrupt) damaged = false;
        const double tNow = monotonicMs();
        if (!joined && (!damaged || tNow - joinTime > 3000)) joined = true;
        if (damaged && tNow - lastIdr > 1000) requestIdr(tNow);
        CapturedFrame frame;
        frame.seq = seq++;
        frame.tCapture = pkt.arrivalNs / 1e6;
        frame.tDequeue = tRead;
        frame.format = format_;
        frame.width = width_;
        frame.height = height_;
        frame.corrupt = damaged || pic.corrupt;
        frame.range = pic.range;
        frame.packetBytes = (uint32_t)pkt.data.size();
        uint8_t* p = static_cast<uint8_t*>(slots_[slot]);
        analyse(p, frame);
        cameraLag(pkt, p, pic.intra);
        // Dedup by size, in KiB: a repeat is ~1/4 of a new frame.
        frame.diff = std::min(255.f, frame.packetBytes / 1024.f);
        frame.tReady = monotonicMs();
        frame.decodeMs = frame.tReady - frame.tDequeue;
        const bool drop = !joined || dedupDrop(frame.seq, frame, frame.seq > 0);
        std::lock_guard<std::mutex> lock(mutex_);
        rpStats_.frames++;
        if (frame.corrupt) rpStats_.corrupt++;
        if (drop) { slotState_[slot] = SlotState::Free; return; }
        frame.slot = slot;
        if (latest_ >= 0 && latest_ != slot && slotState_[latest_] == SlotState::Ready) {
          slotState_[latest_] = SlotState::Free;
          stats_.dropped++;
        }
        slotState_[slot] = SlotState::Ready;
        slotFrames_[slot] = frame;
        latest_ = slot;
        cv_.notify_all();
      };
      if (!dec->decode(pkt.data.data(), pkt.data.size(), (int64_t)next, dst, done)) {
        damaged = true;
        requestIdr(monotonicMs());
      }
      if (slot >= 0) {   // claimed but never finished (decode error): give it back
        std::lock_guard<std::mutex> lock(mutex_);
        if (slotState_[slot] == SlotState::Writing) slotState_[slot] = SlotState::Free;
      }
    }
  }
}
