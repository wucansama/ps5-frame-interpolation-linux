// V4L2 capture on its own thread: mmap buffers, newest-frame-wins, decode (MJPEG) or raw
// copy (YUYV / NV12) straight into GPU-visible slots, plus a cheap content signature for
// frame dedup. The consumer only ever sees the newest finished frame.
#pragma once
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "remoteplay.h"

enum class PixelFormat : uint32_t { Auto = 0, MJPEG = 1, YUYV = 2, NV12 = 3 };
const char* pixelFormatName(PixelFormat f);
PixelFormat parsePixelFormat(const std::string& s);

struct CaptureConfig {
  std::string device = "/dev/video0";
  uint32_t width = 1920, height = 1080, fps = 30;
  PixelFormat format = PixelFormat::Auto;   // Auto: NV12 > YUYV > MJPEG at the requested size
  int dedup = -1;                           // -1 auto (on above 50 fps), 0 off, 1 on
  int jpegQuality = 90;                     // MS2109 dongle encoder quality (ms2109.h), -1 off
  std::string source = "remoteplay";            // dongle (V4L2 `device`) | remoteplay (PS5 Remote Play, remoteplay.h)
  RemotePlayConfig rp;
};

// Strict 60->30 duplicate removal. A 30 fps game on a 60 Hz link repeats every picture
// exactly once, so the duplicates are every other frame: compare the mean diff of the even and
// odd capture frames and, when one parity is clearly the "no change" one, drop all of it.
// No per-frame threshold, so a near-static scene can't make real frames vanish and noise
// can't let duplicates through. Re-locks by itself if a hitch shifts the phase.
struct PhaseDedup {
  float ema[2] = {0, 0};
  float floor = 0;   // slow-rising minimum diff: the card's noise
  uint32_t samples = 0;
  bool locked = false;
  int dupPhase = 0;
  // Returns true if the frame at capture index idx is a duplicate to drop. diffValid: diff is
  // against the immediately preceding capture frame.
  bool feed(uint64_t idx, float diff, bool diffValid);
};

struct CapturedFrame {
  int slot = -1;
  uint64_t seq = 0;          // capture order, counts every dequeued frame
  double tCapture = 0;       // driver timestamp (CLOCK_MONOTONIC ms)
  double tDequeue = 0;       // when this process got the buffer
  double tReady = 0;         // decoded / copied into the slot
  double decodeMs = 0;
  float diff = 255.0f;
  // Black/white level evidence for range detection (luma-ish samples on a 16 px grid):
  uint32_t levelSamples = 0;
  uint32_t levelBelow8 = 0;      // only a full-range source goes here
  uint32_t levelAbove248 = 0;    // likewise
  uint32_t levelNear16 = 0;      // 12..20: where a limited-range source puts black       // mean abs difference to the previous frame's signature (0..255)
  PixelFormat format = PixelFormat::MJPEG;   // slot layout: MJPEG -> decoded RGBA8; else raw
  uint32_t width = 0, height = 0;
  // Remote Play: damaged by network loss (shown, but never interpolated from), the stream's own
  // range (0 limited, 1 full, -1 unknown) and the compressed size.
  bool corrupt = false;
  int range = -1;
  uint32_t packetBytes = 0;
};

class V4l2Capture {
 public:
  explicit V4l2Capture(const CaptureConfig& config);
  ~V4l2Capture();

  uint32_t width() const { return width_; }
  uint32_t height() const { return height_; }
  PixelFormat format() const { return format_; }
  double fps() const { return fps_; }
  // Bytes one slot needs for the negotiated format.
  size_t slotBytes() const;

  // Slots are host-visible, mapped GPU buffers owned by the caller.
  void start(const std::vector<void*>& slots);
  void stop();

  // Newest frame not yet handed out; the slot stays the caller's until release().
  bool acquire(CapturedFrame& out, int timeoutMs);
  void release(int slot);

  struct Stats {
    uint64_t dequeued = 0, dropped = 0, decodeErrors = 0, duplicates = 0;
    bool dedupActive = false, dedupLocked = false;
    int dedupPhase = 0;
    float dedupRatio = 1.f;   // diff(dup phase) / diff(real phase); < 0.5 locks
  };
  Stats stats() const;
  void setDedup(int mode) { dedupMode_ = mode; }
  // MS2109 only: set the encoder quality live (-1 off: the firmware's own table). Applied off
  // the capture thread; re-applied by every stream start.
  void setJpegQuality(int q);
  bool jpegQualitySupported() const { return !hidraw_.empty(); }
  std::string jpegQualityStatus() const;
  std::string error() const;
  bool remotePlay() const { return rp_; }
  // Remote Play: how long the tap should hold the sound back (the picture's delay), and a status line.
  void setAudioDelayMs(uint32_t ms) { rpAudioDelay_ = ms; }
  void requestWake() { rpWake_ = true; }   // wake the console from rest mode (RP_TAP_STANDBY)
  struct RemotePlayStats {
    std::string state, decoder;
    int tapState = -1;          // -1 tap not running, -2 no tap yet, else RpTapState
    std::string tapStatus;      // the tap's own line ("...; retry in 4s")
    uint64_t frames = 0, lost = 0, corrupt = 0, idrRequests = 0, truncated = 0;
    uint32_t audioHeldMs = 0, audioOutMs = 0, bitrate = 0;
    double kbps = 0;   // measured, compressed
    // Input lag: controller button press (tap, sent to the console) -> the first frame that visibly
    // changes after it has arrived here. Reliable in menus / still scenes; median of recent presses.
    double rttMs = 0;              // network round trip at session start (0 unknown)
    double inputLagMs = 0;         // median, press -> changed frame arrived (0: none measured yet)
    double inputLagLastMs = 0;
    int inputLagSamples = 0;
    std::string inputLagMethod;    // of the last sample: "button" (a menu) or "camera" (stick onset > pan)
  };
  RemotePlayStats remotePlayStats() const;

 private:
  void run();
  void runSynthetic();
  void runReplay();
  void runRemotePlay();
  bool rp_ = false;
  std::unique_ptr<RpClient> rpClient_;
  std::atomic<uint32_t> rpAudioDelay_{0};
  std::atomic<bool> rpWake_{false};
  RemotePlayStats rpStats_;   // under mutex_
  std::vector<std::vector<uint8_t>> replay_;
  void analyse(const uint8_t* dst, CapturedFrame& frame);
  bool synthetic_ = false;
  bool negotiate(PixelFormat f);
  int freeSlotLocked() const;

  CaptureConfig config_;
  int fd_ = -1;
  uint32_t width_ = 0, height_ = 0, bytesPerLine_ = 0;
  double fps_ = 0;
  PixelFormat format_ = PixelFormat::MJPEG;
  struct MappedBuffer { void* data = nullptr; size_t length = 0; };
  std::vector<MappedBuffer> buffers_;

  std::vector<void*> slots_;
  enum class SlotState { Free, Writing, Ready, Held };
  std::vector<SlotState> slotState_;
  int latest_ = -1;
  CapturedFrame latestFrame_;
  std::vector<CapturedFrame> slotFrames_;

  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::thread thread_;
  std::atomic<bool> running_{false};
  Stats stats_;
  std::string error_;
  std::vector<uint8_t> signature_, previousSignature_;
  std::atomic<int> dedupMode_{-1};
  PhaseDedup dedup_;
  std::string hidraw_;                 // MS2109 HID node ("" otherwise)
  std::atomic<int> jpegWanted_{-1};
  int jpegApplied_ = -1;               // jpeg thread only
  std::atomic<bool> jpegBusy_{false};
  std::thread jpegThread_;
  std::string jpegStatus_;             // under mutex_
  void applyJpegAsync();
  // Shared by both sources: true if the frame should be dropped as a 60->30 repeat.
  bool dedupDrop(uint64_t idx, const CapturedFrame& frame, bool diffValid);
};

double monotonicMs();
