// PS5 Remote Play as a capture source: the reader side of the frame tap (rp_tap.h), the HEVC /
// H.264 decoder (libavcodec, NVDEC through the CUDA hwaccel, software fallback) and the
// moonglide-chiaki-tap child process (tools/chiaki-tap/, AGPL, never linked into moonglide).
#pragma once
#include <sys/types.h>

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "rp_tap.h"

struct RemotePlayConfig {
  std::string host;             // console address, passed to the tap ("" = start no tap: one runs already)
  std::string tap;              // moonglide-chiaki-tap binary ("" = ~/.local/share/moonglide/bin/moonglide-chiaki-tap)
  std::string shm = RP_TAP_DEFAULT_NAME;
  int bitrate = 0;              // kbps, 0 = chiaki-ng's setting (Chiaki.conf)
  std::string codec = "h265";
  bool wake = false;            // wake the console at start if it is in rest mode (else: a "Wake PS5" button)
  std::string decoder = "cuda"; // cuda | sw
  int res = 0;                  // 720 | 1080 | 1440 | 2160; 0 = from the capture size
  int fps = 0;                  // 30 | 60; 0 = from the capture frame rate
  std::string psnAccount;       // PSN account id (base64), set by the pairing flow in the UI
  std::string psnOnline;        // its online id, for display only
};

// The shared memory, reader side. Created if the tap isn't running yet (either side may start first).
class RpTapReader {
 public:
  explicit RpTapReader(const std::string& name);
  ~RpTapReader();
  RpTapReader(const RpTapReader&) = delete;
  RpTapReader& operator=(const RpTapReader&) = delete;

  volatile RpTapHeader* header() const { return hdr_; }
  bool streaming() const;
  uint64_t writeSeq() const;
  // Waits until the futex moves away from `seen` (a frame or a state change) or timeoutMs.
  void wait(uint32_t seen, int timeoutMs) const;
  uint32_t futex() const;

  // Copies frame n out (false if it was overwritten while copying, or isn't there yet).
  struct Frame {
    std::vector<uint8_t> data;
    uint64_t arrivalNs = 0;
    int32_t framesLost = 0;
    bool recovered = false, truncated = false;
  };
  bool read(uint64_t n, Frame& out) const;
  std::vector<uint8_t> codecHeader(uint32_t session) const;

  void heartbeat(uint32_t audioDelayMs);
  void requestIdr();
  void requestWake();

 private:
  int fd_ = -1;
  uint8_t* base_ = nullptr;
  volatile RpTapHeader* hdr_ = nullptr;
};

// One decoded picture, NV12, written straight into the caller's buffer.
class RpDecoder {
 public:
  RpDecoder(uint32_t codec, const std::string& preferred, uint32_t width, uint32_t height);
  ~RpDecoder();
  RpDecoder(const RpDecoder&) = delete;
  RpDecoder& operator=(const RpDecoder&) = delete;

  struct Picture {
    bool corrupt = false;   // the decoder says so (missing references, bitstream errors)
    bool intra = false;     // an I picture (the PS5 answers a keyframe request with one not marked as IRAP)
    int range = -1;         // 0 limited, 1 full, -1 unknown
    int64_t pts = 0;
  };
  // Feeds one access unit. For every picture it produces, `dst(pts)` returns where to write it
  // (tight NV12, width * height * 3 / 2 bytes; nullptr skips it) and `done` gets its details.
  // False on a hard error (the decoder was reopened).
  bool decode(const uint8_t* data, size_t size, int64_t pts, const std::function<uint8_t*(int64_t)>& dst,
              const std::function<void(const Picture&)>& done);
  const char* name() const;
  bool hardware() const { return hw_; }
  std::string error() const { return error_; }

 private:
  struct Impl;
  Impl* impl_ = nullptr;
  bool hw_ = false;
  uint32_t codec_, width_, height_;
  std::string preferred_, error_;
  void open(bool hw);
  void close();
};

// The moonglide-chiaki-tap process: started with the source, stopped (SIGTERM) with it, and killed
// by the kernel if moonglide dies (PR_SET_PDEATHSIG).
class RpClient {
 public:
  RpClient(const RemotePlayConfig& config, uint32_t height, uint32_t fps);
  ~RpClient();
  bool running();   // reaps it if it exited
  std::string binary() const { return binary_; }
  void restart();

 private:
  std::vector<std::string> args_;
  std::string binary_;
  pid_t pid_ = -1;
  void spawn();
  void stop();
};
