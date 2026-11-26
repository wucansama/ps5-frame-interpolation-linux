// moonglide: capture card -> (NR) -> upscale -> fullscreen Vulkan, natively on Linux.
//
// One frame in flight, by design: a frame is ~10 ms of GPU work at 30 fps, so pipelining two
// would only add latency. The capture thread decodes the newest frame straight into a
// GPU-visible slot; this thread records everything for that frame into one command buffer
// and presents it the moment it is done.
#include "nvml.h"
#include <X11/keysym.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <condition_variable>
#include <cstdarg>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <climits>
#include <deque>
#include <functional>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <algorithm>
#include <vector>

#include <sys/wait.h>

#include "capture_v4l2.h"
#include "rp_tap.h"
#include "gpu.h"
#include "nr_stage.h"
#include "rife_install.h"
#include "optical_flow.h"
#include "config.h"
#include "hud_zones.h"
#include "mcp.h"
#include "ui.h"
#include "nvvfx.h"
#include "rife.h"
#include "overlay.h"
#include "stats.h"
#include "window_x11.h"

namespace {

std::atomic<bool> gQuit{false};
FILE* gPresentLog = getenv("MOONGLIDE_PRESENT_LOG") ? fopen(getenv("MOONGLIDE_PRESENT_LOG"), "w") : nullptr;
volatile sig_atomic_t gQuitSignal = 0;
std::atomic<bool> gRestart{false};
std::vector<std::string> gRestartArgs;   // filled when the settings overlay asks for a restart

// Who holds the video memory (nvidia-smi; empty when it isn't there), as text for the error banners.
std::string gpuUsersTextNow() {
  FILE* f = popen("nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null", "r");
  if (!f) return "";
  std::string out, line;
  char buf[512];
  int n = 0;
  while (fgets(buf, sizeof(buf), f)) {
    line = buf;
    while (!line.empty() && (line.back() == '\n' || line.back() == ' ')) line.pop_back();
    const size_t a = line.find(','), b = line.rfind(',');
    if (a == std::string::npos || a == b || atoi(line.c_str()) == getpid()) continue;
    std::string name = line.substr(a + 1, b - a - 1);
    while (!name.empty() && name.front() == ' ') name.erase(0, 1);
    if (const size_t sl = name.rfind('/'); sl != std::string::npos) name = name.substr(sl + 1);
    if (n++ < 4) out += "\n  " + name + " (pid " + std::to_string(atoi(line.c_str())) + "): " + std::to_string(atoi(line.c_str() + b + 1)) + " MiB";
  }
  pclose(f);
  return out.empty() ? "" : "\n\nUsing the GPU:" + out;
}
bool looksLikeGpuOom(std::string e) {
  for (char& ch : e) ch = (char)tolower((unsigned char)ch);
  for (const char* k : {"out of memory", "outofmemory", "out_of_memory", "failed: -2", "insufficient memory",
                        "cudaerrormemoryallocation"})
    if (e.find(k) != std::string::npos) return true;
  return false;
}
// Cached: the banners are rebuilt every frame and nvidia-smi is a process.
std::string gpuUsersText() {
  static std::string cached;
  static double at = -1e9;
  const double now = monotonicMs();
  if (now - at > 5000) { cached = gpuUsersTextNow(); at = now; }
  return cached;
}
// Features that failed for lack of GPU memory (shown as dismissible notices; nothing is stopped or blocked).
std::mutex gOomMutex;
std::vector<std::string> gOomFeatures;
bool gOomDismissed = false;
moonglide::RifeInstaller gRifeInstall;   // "Install RIFE" in the UI; outlives its worker thread
void noteGpuOom(const std::string& feature, const std::string& err) {
  if (!looksLikeGpuOom(err)) return;
  std::lock_guard<std::mutex> lock(gOomMutex);
  for (const std::string& f : gOomFeatures) if (f == feature) return;
  gOomFeatures.push_back(feature);
  gOomDismissed = false;
}
void onSignal(int sig) { gQuitSignal = sig; gQuit = true; }
// Every clean exit says why, so a live instance that quits can be traced.
void quitBecause(const char* why) {
  if (!gQuit) fprintf(stderr, "quit: %s\n", why);
  gQuit = true;
}

struct Options {
  CaptureConfig capture;
  uint32_t workW = 0, workH = 0;   // 0: the capture size
  std::string modelDir;
  bool nr = false;                 // off by default: streaming + frame generation only (--nr / settings to turn on)
  bool uiOpen = false;             // --ui: start with the app UI open (desktop launcher)
  float intensity = 0.6f;
  NrParams nrParams;
  bool fullscreen = true;
  bool fullscreenSet = false, windowedSet = false;
  uint32_t winW = 1920, winH = 1080;
  VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;
  uint32_t swapImages = 3;
  int range = -1;                  // -1 auto, 0 limited (16-235, expanded before NR), 1 full
  bool overlay = true;
  int overlayPos = 0;            // 0 top-left, 1 top-right, 2 bottom-left, 3 bottom-right
  int overlaySize = 0;           // pixel scale of the stats text, 0 = from the screen height
  bool wizardDone = false;       // the first-run "Connect your PlayStation" wizard was finished or dismissed
  unsigned overlayPc = 0;        // PC stat lines in the overlay: 1 GPU, 2 CPU, 4 RAM, 8 VRAM
  bool overlayCompact = false;   // a few lines (fps, latency) instead of everything
  bool clean = false;              // MJPEG deblock + flow-compensated temporal denoise before NR
  float cleanStrength = 0.5f;      // temporal: max history weight
  float deblockStrength = 1.0f;    // 0 = off
  float deband = 0.f;              // deband threshold in 1/255 per channel (0 off; ~2 for gradients)
  bool cleanDebug = false;
  int fg = 3;                      // 0 lsfg-vk (if its profile matches), 1 own (NVOF interpolation), 2 off,
                                   // 3 RIFE (TensorRT; falls back to own when unavailable)
  RifeConfig rife;
  float rifeCut = 0.15f;           // RIFE: hold the real frame above this mean luma change (scene cut)
  bool dongleAuto = true;          // MS2109: step the quality down on corrupt frames
  int fgMult = 2;                  // RIFE: pictures per real frame (2: 60 fps from 30, 3: 90, 4: 120)
  float fgRealDelay = 0.f;         // RIFE: picture spacing as a fraction of the real period (0: 1/mult, even).
                                   // Less brings the real picture forward (less latency, uneven cadence); the
                                   // generated ones are interpolated at the matching times
  float rifeOutlier = 0.f;         // RIFE: replace pixels outside both real frames' 3x3 luma range by this (0 off)
  float rifeStill = 0.02f;         // RIFE: static guard, 3x3 luma agreement of the real frames (0 off)
  bool fgDebug = false;            // tint where the interpolator fell back to the nearer frame
  uint32_t fgFlags = 232;          // fg_interp flags (2 hard pick, 4 dilate, 8 clamp to source range, 32 static guard,
                                   // 64 outlier rejection, 128 flow median, 256 static pass-through)
  float fgHold = 0.20f;            // hold the real frame when more than this fraction disagrees (0 off)
  std::vector<HudZone> hudZones;
  bool hudShow = false;
  bool flow = true;
  bool fsr = true;               // FSR 1 by default (under 1 ms); --upscale bicubic for plain scaling
  float sharpenStops = 0.2f;     // RCAS: 0 = max sharpening, 2 = little
  bool flowCheck = false;
  uint32_t frames = 0;             // exit after N presented frames (tests)
  std::string controlPath;
  std::string statusPath = "/dev/shm/moonglide.status";
  bool cleanSet = false, deblockSet = false, imagesSet = false;
  NvvfxConfig nvvfx;               // sdk empty: off
  int fgRequested = 0;             // fg as asked for (nvvfx FG turns `fg` off internally)
};

void usage() {
  fprintf(stderr,
          "usage: moonglide [options]\n"
          "  --device /dev/videoN      capture device (default /dev/video0)\n"
          "  --size WxH --fps N        capture mode (default 1920x1080 @ 30)\n"
          "  --format auto|mjpeg|yuyv|nv12\n"
          "  --work WxH                NR work size (default: capture size)\n"
          "  --model DIR               NR model dir (manifest.json + model/stageNN.bin)\n"
          "  --nr / --no-nr            start with NR on / off (default off)\n"
          "  --intensity F             0..1 mix of NR over the original (default 0.6)\n"
          "  --local-structure F --local-tone F --skin-structure F\n"
          "  --windowed WxH            not fullscreen  --fullscreen (with --ui the app starts windowed;\n"
          "                            F11 or the UI button switch)\n"
          "  --present fifo|mailbox|immediate   --images N\n"
          "  --dedup auto|on|off       strict 60->30 repeat removal (auto: on above 50 fps capture)\n"
          "  --range auto|limited|full  RGB/YCbCr input range (auto: detect from black/white levels)\n"
          "  --upscale fsr|bicubic (default fsr)  --sharpen STOPS (FSR RCAS, default 0.2; 0 = strongest)\n"
          "  --no-flow (zero-motion history)  --flow-check (log reprojection error)\n"
          "  --hud rect:x,y,w,h | ellipse:cx,cy,rx,ry[:feather]   HUD/minimap zone shown unprocessed\n"
          "                            (fractions 0..1 of the picture; repeatable)  --hud-show (outline)\n"
          "  --clean  --clean-strength F (0.5)  --deblock F (1.0)   pre-NR cleanup of the card's MJPEG\n"
          "  --deband F (0 off)        deband gradients, threshold in 1/255 (about 2)\n"
          "  --fg lsfg|own|off         frame generation (own: optical-flow interpolation, lsfg-vk disabled)\n"
          "  --fg-real-delay F (0 = 1/mult)  RIFE picture spacing, fraction of the real period; smaller\n"
          "                            shows the real frame sooner (less latency, uneven cadence)\n"
          "  --nvvfx DIR (or MOONGLIDE_NVVFX=DIR)   NVIDIA Video Effects SDK (user-installed): chain\n"
          "                            denoise -> clean -> NR -> VSR to the display -> VFG x2\n"
          "  --nvvfx-denoise Q|off (8)  --nvvfx-sr Q|off (4)  --nvvfx-fg MODE|off (0 low)\n"
          "  --nvvfx-colour-fix on|off (on: undo the effects' measured colour cast)\n"
          "  --no-overlay  --overlay-pos tl|tr|bl|br  --overlay-size N (0 auto)  --overlay-mode full|compact\n"
          "  --frames N\n"
          "  --config PATH (default ~/.config/moonglide/config.toml, or MOONGLIDE_CONFIG)  --no-config\n"
          "                            settings file, written by the overlay (F1 / Tab); options override it\n"
          "  --control PATH  --status PATH   control fifo / status file (default per user / /dev/shm)\n"
          "keys: N toggle NR | Up/Down intensity | [ ] local structure | S A/B split | O overlay\n"
          "      H outline HUD zones | R reset history | Ctrl+Q quit\n");
}

bool parseSize(const std::string& s, uint32_t& w, uint32_t& h) {
  return sscanf(s.c_str(), "%ux%u", &w, &h) == 2 && w && h;
}

Options parseArgs(int argc, char** argv) {
  Options o;
  const char* home = getenv("HOME");
  o.modelDir = getenv("MOONGLIDE_MODEL") ? getenv("MOONGLIDE_MODEL")
                                     : std::string(home ? home : "") + "/.local/share/moonglide/model";
  const char* runtime = getenv("XDG_RUNTIME_DIR");
  // Same path whether launched from a desktop session or a bare ssh/setsid shell.
  o.controlPath = (runtime ? std::string(runtime) : "/run/user/" + std::to_string(getuid())) + "/moonglide.ctl";
  if (const char* c = getenv("MOONGLIDE_CTL")) o.controlPath = c;
  if (const char* st = getenv("MOONGLIDE_STATUS")) o.statusPath = st;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        fprintf(stderr, "%s needs a value\n", a.c_str());
        exit(2);
      }
      return argv[++i];
    };
    if (a == "--device") o.capture.device = next();
    else if (a == "--source") { std::string v = next(); o.capture.source = v == "remoteplay" ? "remoteplay" : "dongle"; }
    else if (a == "--rp-host") o.capture.rp.host = next();
    else if (a == "--rp-bitrate") o.capture.rp.bitrate = std::max(0, atoi(next().c_str()));
    else if (a == "--rp-tap") o.capture.rp.tap = next();
    else if (a == "--rp-shm") o.capture.rp.shm = next();
    else if (a == "--rp-codec") o.capture.rp.codec = next() == "h264" ? "h264" : "h265";
    else if (a == "--rp-decoder") o.capture.rp.decoder = next() == "sw" ? "sw" : "cuda";
    else if (a == "--rp-wake") o.capture.rp.wake = next() != "off";
    else if (a == "--rp-res") { const int v = atoi(next().c_str()); o.capture.rp.res = v == 720 || v == 1080 || v == 1440 || v == 2160 ? v : 0; }
    else if (a == "--rp-fps") { const int v = atoi(next().c_str()); o.capture.rp.fps = v >= 60 ? 60 : v > 0 ? 30 : 0; }
    else if (a == "--rp-psn-account") o.capture.rp.psnAccount = next();
    else if (a == "--rp-psn-online") o.capture.rp.psnOnline = next();
    else if (a == "--size") { if (!parseSize(next(), o.capture.width, o.capture.height)) exit(2); }
    else if (a == "--fps") o.capture.fps = (uint32_t)atoi(next().c_str());
    else if (a == "--format") o.capture.format = parsePixelFormat(next());
    else if (a == "--work") { if (!parseSize(next(), o.workW, o.workH)) exit(2); }
    else if (a == "--model") o.modelDir = next();
    else if (a == "--no-nr") o.nr = false;
    else if (a == "--nr") o.nr = true;
    else if (a == "--intensity") o.intensity = (float)atof(next().c_str());
    else if (a == "--local-structure") o.nrParams.localStructure = (float)atof(next().c_str());
    else if (a == "--local-tone") o.nrParams.localTone = (float)atof(next().c_str());
    else if (a == "--skin-structure") o.nrParams.skinStructure = (float)atof(next().c_str());
    else if (a == "--windowed") { o.fullscreen = false; o.windowedSet = true; o.fullscreenSet = false; if (!parseSize(next(), o.winW, o.winH)) exit(2); }
    else if (a == "--fullscreen") { o.fullscreen = true; o.fullscreenSet = true; o.windowedSet = false; }
    else if (a == "--present") {
      std::string m = next();
      o.presentMode = m == "mailbox" ? VK_PRESENT_MODE_MAILBOX_KHR
                      : m == "immediate" ? VK_PRESENT_MODE_IMMEDIATE_KHR
                                         : VK_PRESENT_MODE_FIFO_KHR;
    } else if (a == "--images") { o.swapImages = (uint32_t)atoi(next().c_str()); o.imagesSet = true; }
    else if (a == "--dedup") { std::string v = next(); o.capture.dedup = v == "on" ? 1 : v == "off" ? 0 : -1; }
    else if (a == "--no-dedup") o.capture.dedup = 0;
    else if (a == "--dongle-auto") o.dongleAuto = next() != "off";
    else if (a == "--dongle-quality") { std::string v = next(); o.capture.jpegQuality = v == "off" ? -1 : std::clamp(atoi(v.c_str()), 1, 94); }
    else if (a == "--yuv-full-range") o.range = 1;
    else if (a == "--range") { std::string v = next(); o.range = v == "full" ? 1 : v == "limited" ? 0 : -1; }
    else if (a == "--no-overlay") o.overlay = false;
    else if (a == "--overlay-pos") { const std::string v = next(); o.overlayPos = v == "tr" ? 1 : v == "bl" ? 2 : v == "br" ? 3 : 0; }
    else if (a == "--overlay-size") o.overlaySize = std::clamp(atoi(next().c_str()), 0, 6);
    else if (a == "--overlay-pc") o.overlayPc = (unsigned)atoi(next().c_str()) & 15u;
    else if (a == "--overlay-mode") o.overlayCompact = next() == "compact";
    else if (a == "--wizard") o.wizardDone = next() == "done";
    else if (a == "--control") o.controlPath = next();
    else if (a == "--config") next();          // read in main() before the options
    else if (a == "--no-config") {}
    else if (a == "--ui") o.uiOpen = true;
    else if (a == "--status") o.statusPath = next();
    else if (a == "--hud") {
      HudZone z;
      std::string v = next();
      if (!parseHudZone(v, z)) { fprintf(stderr, "bad --hud zone '%s'\n", v.c_str()); exit(2); }
      o.hudZones.push_back(z);
    } else if (a == "--hud-show") o.hudShow = true;
    else if (a == "--fg") { std::string v = next(); o.fg = v == "own" ? 1 : v == "off" ? 2 : v == "rife" ? 3 : 0; }
    else if (a == "--rife-model") o.rife.model = next();
    else if (a == "--rife-trt") o.rife.trtLib = next();
    else if (a == "--rife-precision") o.rife.fp16 = next() != "fp32";
    else if (a == "--fg-mult") o.fgMult = std::clamp(atoi(next().c_str()), 2, 4);
    else if (a == "--fg-real-delay") o.fgRealDelay = std::clamp((float)atof(next().c_str()), 0.f, 0.5f);
    else if (a == "--rife-outlier") o.rifeOutlier = std::fmax(0.f, (float)atof(next().c_str()));
    else if (a == "--rife-still") o.rifeStill = std::fmax(0.f, (float)atof(next().c_str()));
    else if (a == "--rife-cut") o.rifeCut = std::fmax(0.f, (float)atof(next().c_str()));
    else if (a == "--fg-debug") o.fgDebug = true;
    else if (a == "--fg-flags") o.fgFlags = (uint32_t)strtoul(next().c_str(), nullptr, 0) & 494u;
    else if (a == "--fg-hold") o.fgHold = std::fmax(0.f, (float)atof(next().c_str()));
    else if (a == "--clean") { o.clean = true; o.cleanSet = true; }
    else if (a == "--no-clean") { o.clean = false; o.cleanSet = true; }
    else if (a == "--nvvfx") o.nvvfx.sdk = next();
    else if (a == "--nvvfx-denoise") { std::string v = next(); o.nvvfx.denoise = v == "off" ? -1 : atoi(v.c_str()); }
    else if (a == "--nvvfx-sr") { std::string v = next(); o.nvvfx.sr = v == "off" ? -1 : atoi(v.c_str()); }
    else if (a == "--nvvfx-fg") { std::string v = next(); o.nvvfx.fg = v == "off" ? -1 : atoi(v.c_str()); }
    else if (a == "--nvvfx-colour-fix" || a == "--nvvfx-color-fix") o.nvvfx.colourFix = next() != "off";
    else if (a == "--clean-strength") o.cleanStrength = (float)atof(next().c_str());
    else if (a == "--deband") o.deband = std::clamp((float)atof(next().c_str()), 0.f, 8.f);
    else if (a == "--deblock") { o.deblockStrength = (float)atof(next().c_str()); o.deblockSet = true; }
    else if (a == "--clean-debug") o.cleanDebug = true;
    else if (a == "--no-flow") o.flow = false;
    else if (a == "--upscale") o.fsr = next() != "bicubic";
    else if (a == "--sharpen") o.sharpenStops = (float)atof(next().c_str());
    else if (a == "--flow-check") o.flowCheck = true;
    else if (a == "--frames") o.frames = (uint32_t)atoi(next().c_str());
    else if (a == "-h" || a == "--help") { usage(); exit(0); }
    else { fprintf(stderr, "unknown option %s\n", a.c_str()); usage(); exit(2); }
  }
  return o;
}

// Line-oriented control channel: `echo "nr off" > $XDG_RUNTIME_DIR/moonglide.ctl`.
class ControlFifo {
 public:
  explicit ControlFifo(const std::string& path) : path_(path) {
    unlink(path.c_str());
    if (mkfifo(path.c_str(), 0600) == 0) {
      fd_ = open(path.c_str(), O_RDONLY | O_NONBLOCK);
      keep_ = open(path.c_str(), O_WRONLY | O_NONBLOCK);   // no EOF when writers come and go
      struct stat st;
      if (fd_ >= 0 && fstat(fd_, &st) == 0) { dev_ = st.st_dev; ino_ = st.st_ino; }
    }
  }
  ~ControlFifo() {
    // Only remove the path if it is still our fifo (another instance may have replaced it).
    struct stat st;
    if (fd_ >= 0 && stat(path_.c_str(), &st) == 0 && st.st_dev == dev_ && st.st_ino == ino_) unlink(path_.c_str());
    if (fd_ >= 0) close(fd_);
    if (keep_ >= 0) close(keep_);
  }
  std::vector<std::string> poll() {
    std::vector<std::string> lines;
    if (fd_ < 0) return lines;
    char chunk[512];
    ssize_t n;
    while ((n = read(fd_, chunk, sizeof(chunk))) > 0) buffer_.append(chunk, (size_t)n);
    size_t pos;
    while ((pos = buffer_.find('\n')) != std::string::npos) {
      lines.push_back(buffer_.substr(0, pos));
      buffer_.erase(0, pos + 1);
    }
    return lines;
  }

 private:
  std::string path_, buffer_;
  int fd_ = -1, keep_ = -1;
  dev_t dev_ = 0;
  ino_t ino_ = 0;
};

void fullBarrier(VkCommandBuffer cmd) {
  VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT |
                    VK_ACCESS_TRANSFER_WRITE_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &b, 0,
                       nullptr, 0, nullptr);
}

void imageBarrier(VkCommandBuffer cmd, VkImage image, VkImageLayout from, VkImageLayout to, VkAccessFlags srcAccess,
                  VkAccessFlags dstAccess, VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage) {
  VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  b.oldLayout = from;
  b.newLayout = to;
  b.srcAccessMask = srcAccess;
  b.dstAccessMask = dstAccess;
  b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.image = image;
  b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &b);
}

// lsfg-vk is an implicit layer: loaded into every Vulkan process, active only for a process
// its config names (or LSFG_PROCESS). "x2" on the badge means both.
bool lsfgActive() {
  bool loaded = false;
  {
    std::ifstream maps("/proc/self/maps");
    std::string line;
    while (std::getline(maps, line))
      if (line.find("lsfg") != std::string::npos) { loaded = true; break; }
  }
  if (!loaded) return false;
  if (const char* p = getenv("LSFG_PROCESS")) return std::string(p) == "moonglide";
  const char* home = getenv("HOME");
  std::ifstream conf(std::string(home ? home : "") + "/.config/lsfg-vk/conf.toml");
  std::string line;
  while (std::getline(conf, line))
    if (line.find("exe") != std::string::npos && line.find("\"moonglide\"") != std::string::npos &&
        line.find('#') == std::string::npos)
      return true;
  return false;
}

std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
std::string fmt(const char* f, ...) {
  char buf[1024];
  va_list ap;
  va_start(ap, f);
  vsnprintf(buf, sizeof(buf), f, ap);
  va_end(ap);
  return buf;
}

// When each present actually reached the screen (VK_KHR_present_wait), on its own thread.
class PresentWaiter {
 public:
  PresentWaiter(VkDevice device, VkSwapchainKHR swapchain) : device_(device), swapchain_(swapchain) {
    thread_ = std::thread([this] { run(); });
  }
  ~PresentWaiter() {
    {
      std::lock_guard<std::mutex> l(m_);
      stop_ = true;
    }
    cv_.notify_all();
    thread_.join();
  }
  void submitted(uint64_t id, double tCapture, double tDequeue) {
    std::lock_guard<std::mutex> l(m_);
    pending_.push_back({id, tCapture, tDequeue});
    cv_.notify_all();
  }
  // Drains (displayed - capture, displayed - dequeue) pairs measured since the last call.
  std::vector<std::pair<double, double>> take() {
    std::lock_guard<std::mutex> l(m_);
    auto r = std::move(done_);
    done_.clear();
    return r;
  }
  bool broken() const { return broken_; }

 private:
  struct Item { uint64_t id; double tCapture, tDequeue; };
  void run() {
    for (;;) {
      Item item;
      {
        std::unique_lock<std::mutex> l(m_);
        cv_.wait(l, [&] { return stop_ || !pending_.empty(); });
        if (stop_) return;
        item = pending_.front();
        pending_.pop_front();
      }
      if (broken_) continue;
      VkResult r = vkWaitForPresentKHR(device_, swapchain_, item.id, 500'000'000ull);
      const double now = monotonicMs();
      if (r == VK_SUCCESS) {
        std::lock_guard<std::mutex> l(m_);
        done_.push_back({now - item.tCapture, now - item.tDequeue});
      } else if (r != VK_TIMEOUT) {
        fprintf(stderr, "present-wait: %d, disabling the on-screen latency measurement\n", (int)r);
        broken_ = true;
      }
    }
  }
  VkDevice device_;
  VkSwapchainKHR swapchain_;
  std::thread thread_;
  std::mutex m_;
  std::condition_variable cv_;
  std::deque<Item> pending_;
  std::vector<std::pair<double, double>> done_;
  bool stop_ = false;
  std::atomic<bool> broken_{false};
};

// Frame generation: the real picture is presented half a real-frame period after the generated
// one. Doing that wait on the main thread delays the next frame's processing by it, so a chain
// slower than half a period (the nvvfx effects) starts every frame late and the pairs bunch up.
// This thread does the delayed present instead; queue use from both threads goes through
// `queueMutex`.
class DelayedPresenter {
 public:
  DelayedPresenter(VkQueue queue, std::mutex& queueMutex) : queue_(queue), queueMutex_(queueMutex) {
    thread_ = std::thread([this] { run(); });
  }
  ~DelayedPresenter() {
    {
      std::lock_guard<std::mutex> l(m_);
      stop_ = true;
    }
    cv_.notify_all();
    thread_.join();
  }
  // Queue a present for `due` (presents go out in posting order). `first`: wait for the
  // previously posted ones before queueing this one (a new real frame's set).
  void post(VkSwapchainKHR swapchain, uint32_t index, VkSemaphore wait, uint64_t presentId, bool useId, double due,
            bool first = true) {
    if (first) flush();
    std::lock_guard<std::mutex> l(m_);
    items_.push_back({swapchain, index, wait, presentId, useId, due});
    cv_.notify_all();
  }
  // Wait until the posted presents (if any) have been queued.
  void flush() {
    std::unique_lock<std::mutex> l(m_);
    cv_.wait(l, [&] { return items_.empty() && !busy_; });
  }
  VkResult lastResult() const { return result_; }

 private:
  struct Item {
    VkSwapchainKHR swapchain;
    uint32_t index;
    VkSemaphore wait;
    uint64_t presentId;
    bool useId;
    double due;
  };
  void run() {
    for (;;) {
      Item it;
      {
        std::unique_lock<std::mutex> l(m_);
        cv_.wait(l, [&] { return stop_ || !items_.empty(); });
        if (items_.empty()) return;
        it = items_.front();
        items_.pop_front();
        busy_ = true;
      }
      const double wait = it.due - monotonicMs();
      if (wait > 0) std::this_thread::sleep_for(std::chrono::microseconds((int64_t)(wait * 1000)));
      VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
      present.waitSemaphoreCount = 1;
      present.pWaitSemaphores = &it.wait;
      present.swapchainCount = 1;
      present.pSwapchains = &it.swapchain;
      present.pImageIndices = &it.index;
      VkPresentIdKHR pid{VK_STRUCTURE_TYPE_PRESENT_ID_KHR};
      if (it.useId) {
        pid.swapchainCount = 1;
        pid.pPresentIds = &it.presentId;
        present.pNext = &pid;
      }
      {
        std::lock_guard<std::mutex> q(queueMutex_);
        result_ = vkQueuePresentKHR(queue_, &present);
      }
      if (gPresentLog) fprintf(gPresentLog, "%.3f delayed\n", monotonicMs());
      if (onPresent) onPresent(monotonicMs());
      std::lock_guard<std::mutex> l(m_);
      busy_ = false;
      cv_.notify_all();
    }
  }
  VkQueue queue_;
  std::mutex& queueMutex_;
  std::thread thread_;
  std::mutex m_;
  std::condition_variable cv_;
  std::deque<Item> items_;
  bool busy_ = false, stop_ = false;

 public:
  std::function<void(double)> onPresent;   // presenter thread, after each present
  std::atomic<VkResult> result_{VK_SUCCESS};
};

}  // namespace

// The start-up (or the pipeline) failed: a window of its own with the reason and what can be
// done, instead of vanishing. 1: start again, 0: quit. Falls back to a zenity dialog when even
// this window can't be made (no Vulkan device at all, say).
int errorScreen(const std::string& what) {
  const bool oom = looksLikeGpuOom(what);
  try {
    X11Window window(false, 0, 0);
    Gpu gpu(window.display(), window.window());
    gpu.createSwapchain(window.width(), window.height(), VK_PRESENT_MODE_FIFO_KHR, 3);
    Ui ui(gpu, window.display(), window.window(), (uint32_t)gpu.images().size());
    ui.setVisible(true);
    VkDevice dev = gpu.device();
    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = gpu.queueFamily();
    VkCommandPool pool;
    VK_CHECK(vkCreateCommandPool(dev, &poolInfo, nullptr, &pool));
    VkCommandBufferAllocateInfo cbInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbInfo.commandPool = pool;
    cbInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbInfo.commandBufferCount = 1;
    VkCommandBuffer cmd;
    VK_CHECK(vkAllocateCommandBuffers(dev, &cbInfo, &cmd));
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    VK_CHECK(vkCreateFence(dev, &fi, nullptr, &fence));
    VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkSemaphore acquired;
    VK_CHECK(vkCreateSemaphore(dev, &si, nullptr, &acquired));
    int action = -1;
    uint32_t lastW = window.width(), lastH = window.height();
    fprintf(stderr, "error screen: %s\n", oom ? "GPU out of memory" : "start-up failed");
    while (action < 0 && !gQuit) {
      if (!window.poll(
              [&](unsigned long key) {
                if (key == XK_F11) window.setFullscreen(!window.fullscreen());
                if (key == (XK_q | X11Window::kKeyCtrl)) action = 0;
              },
              [&](XEvent& e) { return ui.handleEvent(e); }))
        action = 0;
      if ((window.width() != lastW || window.height() != lastH) && window.width() && window.height()) {
        vkDeviceWaitIdle(dev);
        ui.swapchainRecreated();
        gpu.createSwapchain(window.width(), window.height(), VK_PRESENT_MODE_FIFO_KHR, 3);
        lastW = window.width();
        lastH = window.height();
      }
      if (!ui.visible()) ui.setVisible(true);   // F1 / Esc would hide the only thing there is
      UiState s;
      s.fatal = true;
      s.rectW = gpu.extent().width;
      s.rectH = gpu.extent().height;
      s.fullscreen = window.fullscreen();
      UiBanner b;
      b.level = 2;
      if (oom) {
        b.title = "Not enough GPU memory";
        b.detail = "Another program is using the GPU. Close it or lower the settings, then press Retry." + gpuUsersText();
      } else {
        b.title = "MoonGlide could not start";
        b.detail = "The details are below. Fix the cause, then press Retry.";
      }
      b.detail += "\n\nError: " + what;
      b.actions.push_back({"Retry", "retry"});
      b.actions.push_back({"Quit", "quit"});
      s.banners.push_back(b);
      std::vector<std::string> cmds;
      ui.build(s, [&](const std::string& c) { cmds.push_back(c); });
      for (const std::string& c : cmds) {
        if (c == "retry") action = 1;
        else if (c == "quit") action = 0;
        else if (c == "fullscreen toggle") window.setFullscreen(!window.fullscreen());
      }
      uint32_t index = 0;
      VkResult acq = vkAcquireNextImageKHR(dev, gpu.swapchain(), 200000000ull, acquired, VK_NULL_HANDLE, &index);
      if (acq == VK_ERROR_OUT_OF_DATE_KHR) { lastW = 0; continue; }
      if (acq == VK_TIMEOUT || acq == VK_NOT_READY) continue;
      if (acq != VK_SUCCESS && acq != VK_SUBOPTIMAL_KHR) VK_CHECK(acq);
      const VkImage image = gpu.images()[index].image;
      VK_CHECK(vkResetCommandBuffer(cmd, 0));
      VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
      VkImageMemoryBarrier ib{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
      ib.srcQueueFamilyIndex = ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      ib.image = image;
      ib.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      ib.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      ib.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      ib.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &ib);
      VkClearColorValue dark{};
      dark.float32[0] = dark.float32[1] = dark.float32[2] = 13.f / 255.f;
      dark.float32[3] = 1.f;
      vkCmdClearColorImage(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &dark, 1, &ib.subresourceRange);
      ui.record(cmd, image, gpu.extent().width, gpu.extent().height);
      VK_CHECK(vkEndCommandBuffer(cmd));
      const VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
      VkSubmitInfo sub{VK_STRUCTURE_TYPE_SUBMIT_INFO};
      sub.waitSemaphoreCount = 1;
      sub.pWaitSemaphores = &acquired;
      sub.pWaitDstStageMask = &stage;
      sub.commandBufferCount = 1;
      sub.pCommandBuffers = &cmd;
      sub.signalSemaphoreCount = 1;
      sub.pSignalSemaphores = &gpu.images()[index].renderDone;
      VK_CHECK(vkQueueSubmit(gpu.queue(), 1, &sub, fence));
      VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
      pi.waitSemaphoreCount = 1;
      pi.pWaitSemaphores = &gpu.images()[index].renderDone;
      pi.swapchainCount = 1;
      VkSwapchainKHR sc = gpu.swapchain();
      pi.pSwapchains = &sc;
      pi.pImageIndices = &index;
      const VkResult pr = vkQueuePresentKHR(gpu.queue(), &pi);
      if (pr == VK_ERROR_OUT_OF_DATE_KHR) lastW = 0;
      VK_CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
      VK_CHECK(vkResetFences(dev, 1, &fence));
    }
    vkDeviceWaitIdle(dev);
    vkDestroySemaphore(dev, acquired, nullptr);
    vkDestroyFence(dev, fence, nullptr);
    vkDestroyCommandPool(dev, pool, nullptr);
    fprintf(stderr, "error screen: %s\n", action == 1 ? "start again" : "quit");
    return action == 1 ? 1 : 0;
  } catch (const std::exception& e2) {
    fprintf(stderr, "error screen unavailable (%s); trying zenity\n", e2.what());
  }
  // Last resort: a desktop dialog.
  if (access("/usr/bin/zenity", X_OK) != 0) return 0;
  const std::string text = std::string(oom ? "Not enough GPU memory: another program is using the GPU." + gpuUsersText() + "\n\n" : "MoonGlide could not start.\n\n") + what;
  auto dialog = [&](std::vector<std::string> a) {
    const pid_t pid = fork();
    if (pid == 0) {
      std::vector<char*> av;
      for (std::string& x : a) av.push_back(x.data());
      av.push_back(nullptr);
      execv("/usr/bin/zenity", av.data());
      _exit(127);
    }
    int st = 0;
    if (pid < 0 || waitpid(pid, &st, 0) < 0) return -1;
    return WIFEXITED(st) ? WEXITSTATUS(st) : -1;
  };
  return dialog({"zenity", "--question", "--title=MoonGlide", "--ok-label=Retry", "--cancel-label=Quit",
                 "--no-markup", "--text=" + text}) == 0 ? 1 : 0;
}

int main(int argc, char** argv) {
  if (argc > 1 && !strcmp(argv[1], "mcp")) return runMcp(argc, argv);   // MCP server: no GPU, no window
  // Environment as started, for a settings restart (this process sets DISABLE_LSFG itself).
  const char* disableLsfgEnv = getenv("DISABLE_LSFG");
  const bool hadDisableLsfg = disableLsfgEnv != nullptr;
  const std::string origDisableLsfg = disableLsfgEnv ? disableLsfgEnv : "";
  // Settings file first, then the command line on top of it.
  std::string configPath = defaultConfigPath();
  bool configOn = true, configGiven = false;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--config") && i + 1 < argc) { configPath = argv[i + 1]; configGiven = true; }
    if (!strcmp(argv[i], "--no-config")) configOn = false;
  }
  ConfigFile config;
  if (configOn && loadConfig(configPath, config)) fprintf(stderr, "config: %s\n", configPath.c_str());
  config.path = configPath;
  std::vector<std::string> cliKeys;
  for (int i = 1; i < argc; ++i) {
    const std::string k = settingKeyOfFlag(argv[i]);
    if (!k.empty()) cliKeys.push_back(k);
  }
  std::vector<std::string> fileArgs = settingsToArgs(config.active, cliKeys);
  std::vector<char*> mergedArgs{argv[0]};
  for (std::string& a : fileArgs) mergedArgs.push_back(a.data());
  for (int i = 1; i < argc; ++i) mergedArgs.push_back(argv[i]);
  Options opt = parseArgs((int)mergedArgs.size(), mergedArgs.data());
  opt.fgRequested = opt.fg;
  // The desktop app opens as a window with its UI; fullscreen is a button / F11 away.
  if (opt.uiOpen && !opt.fullscreenSet && !opt.windowedSet) {
    opt.fullscreen = false;
    opt.winW = opt.winH = 0;   // 80% of the screen
  }
  if (opt.nvvfx.sdk.empty() && getenv("MOONGLIDE_NVVFX") && *getenv("MOONGLIDE_NVVFX")) opt.nvvfx.sdk = getenv("MOONGLIDE_NVVFX");
  if (!opt.nvvfx.sdk.empty()) {
    nvvfxEnsureLibraryPath(opt.nvvfx.sdk, argv);   // re-execs once
    if (opt.nvvfx.fg >= 0) opt.fg = 2;             // VFG is the frame generator
  }
  // lsfg-vk is an implicit layer: keep it out of the process unless it is the frame generator.
  if (opt.fg != 0) setenv("DISABLE_LSFG", "1", 1);
  // Own / nvvfx FG holds two images per real frame, one of them until the delayed present: room
  // for the next frame's pair without blocking on that.
  if (!opt.imagesSet && (opt.fg == 1 || opt.fg == 3 || (!opt.nvvfx.sdk.empty() && opt.nvvfx.fg >= 0)))
    opt.swapImages = opt.fg == 3 ? (uint32_t)(2 * opt.fgMult + 1) : 5;   // RIFE holds fgMult per real frame; fewer images, less VRAM
  signal(SIGINT, onSignal);
  signal(SIGTERM, onSignal);

  try {
    V4l2Capture capture(opt.capture);
    const uint32_t capW = capture.width(), capH = capture.height();
    const uint32_t workW = opt.workW ? opt.workW : capW, workH = opt.workH ? opt.workH : capH;
    const bool workScaled = workW != capW || workH != capH;

    X11Window window(opt.fullscreen, opt.winW, opt.winH);
    Gpu gpu(window.display(), window.window());
    vk::Context& ctx = gpu.ctx();
    if (getenv("MOONGLIDE_TEST_FAIL") && !strcmp(getenv("MOONGLIDE_TEST_FAIL"), "oom"))
      throw std::runtime_error("vkAllocateMemory(device, &ai, nullptr, &memory) failed: -2 (test)");
    gpu.createSwapchain(window.width(), window.height(), opt.presentMode, opt.swapImages);
    VkExtent2D ext = gpu.extent();   // follows the window (fullscreen toggle)
    // Output-size buffers are made for the whole screen, so a bigger window needs no new ones.
    const uint32_t screenW = std::max(window.screenWidth(), ext.width), screenH = std::max(window.screenHeight(), ext.height);
    const bool lsfg = lsfgActive();
    // Our frame generators: RIFE (TensorRT, the default) and own (optical-flow interpolation).
    // Both set up at start can be switched live; RIFE falls back to own.
    std::unique_ptr<Rife> rife;
    std::string rifeError;   // why RIFE did not start (shown in the UI)
    if (opt.fg == 3) {
      try {
        if (getenv("MOONGLIDE_TEST_FAIL") && !strcmp(getenv("MOONGLIDE_TEST_FAIL"), "rife-oom"))
          throw std::runtime_error("TensorRT: OutOfMemory (test)");
        rife = std::make_unique<Rife>(gpu, opt.rife, workW, workH);
      } catch (const std::exception& e) {
        fprintf(stderr, "rife: unavailable (%s); own FG instead\n", e.what());
        rifeError = e.what();
      }
    }
    const bool fgFlowAvailable = (opt.fg == 1 || opt.fg == 3) && gpu.opticalFlow();
    if (opt.fg == 1 && !fgFlowAvailable) fprintf(stderr, "fg: own frame generation needs VK_NV_optical_flow; off\n");
    const bool fgOwnAvailable = fgFlowAvailable || rife;
    bool fgRife = rife != nullptr;   // which of ours generates (fgOwn: whether one does)
    bool fgOwn = fgOwnAvailable && (fgRife || opt.flow);
    fprintf(stderr, "frame generation: %s\n", rife ? "RIFE x2 (TensorRT)" : fgOwnAvailable ? "own x2 (optical-flow interpolation)"
                                               : lsfg ? "lsfg-vk x2 (profile moonglide)"
                                                      : "none");

    // Pipelines and buffers.
    vk::Pipeline yuvPipe = gpu.pipeline("yuv_to_rgba");
    vk::Pipeline resamplePipe = gpu.pipeline("resample");
    vk::Pipeline levelsPipe = gpu.pipeline("levels");
    vk::Pipeline presentPipe = gpu.pipeline("present");
    vk::Pipeline easuPipe = gpu.pipeline("fsr_easu");
    vk::Pipeline rcasPipe = gpu.pipeline("fsr_rcas");
    vk::Pipeline deblockPipe = gpu.pipeline("deblock");
    vk::Pipeline debandPipe = gpu.pipeline("deband");
    vk::Pipeline tdnPipe = gpu.pipeline("temporal_denoise");
    vk::Pipeline fgMixPipe = fgOwnAvailable ? gpu.pipeline("fg_mix") : vk::Pipeline{};
    vk::Pipeline fgInterpPipe = fgOwnAvailable ? gpu.pipeline("fg_interp") : vk::Pipeline{};
    vk::Pipeline rifePackPipe = rife ? gpu.pipeline("rife_pack") : vk::Pipeline{};
    vk::Pipeline rifeUnpackPipe = rife ? gpu.pipeline("rife_unpack") : vk::Pipeline{};

    std::vector<vk::Buffer> slots;
    std::vector<void*> slotPtrs;
    for (int i = 0; i < 3; ++i) {
      slots.push_back(ctx.createBuffer(capture.slotBytes(), true, "capture slot"));
      slotPtrs.push_back(slots.back().mapped);
    }
    vk::Buffer colour = ctx.createBuffer((VkDeviceSize)capW * capH * 4, false, "capture rgba8");
    vk::Buffer workColour = workScaled ? ctx.createBuffer((VkDeviceSize)workW * workH * 4, false, "work rgba8") : vk::Buffer{};
    const vk::Buffer& workSrc = workScaled ? workColour : colour;
    vk::Buffer outBuffer = ctx.createBuffer((VkDeviceSize)screenW * screenH * 4, false, "present out");
    std::vector<uint32_t> font = overlayFont();
    vk::Buffer fontBuffer = ctx.createBuffer(font.size() * 4, false, "font");
    ctx.upload(fontBuffer, font.data(), font.size() * 4);
    vk::Buffer textBuffer = ctx.createBuffer(kOverlayLines * kOverlayColumns * 4, true, "overlay text");
    vk::Buffer zoneBuffer = ctx.createBuffer(kHudZoneBufferBytes, true, "hud zones");
    // Own FG: the shown picture of the last two real frames (ping-pong) and the generated one.
    const VkDeviceSize workBytes = (VkDeviceSize)workW * workH * 4;
    vk::Buffer fgShown[2], fgMid, fgMidMore[Rife::kMaxOut - 1], fgStats;
    bool fgStatsPending = false;
    double fgFail = 0;            // smoothed fraction of the picture where the flow failed
    uint64_t fgHeld = 0;          // generated pictures replaced by the held real one
    if (fgOwnAvailable) {
      fgShown[0] = ctx.createBuffer(workBytes, false, "fg shown 0");
      fgShown[1] = ctx.createBuffer(workBytes, false, "fg shown 1");
      fgMid = ctx.createBuffer(workBytes, false, "fg mid");
      fgStats = ctx.createBuffer(16, true, "fg stats");
    }
    // RIFE: scene-cut stats (luma diff sum, pixels), its timeline and GPU time.
    vk::Buffer rifeStats = rife ? ctx.createBuffer(16, true, "rife stats") : vk::Buffer{};
    int rifeCur = 0;
    uint64_t rifeBase = 0;
    bool rifeStatsPending = false;
    Samples rifeMs(90), rifeGapMs(90);
    // Made on first use: deblock / deband scratch and the clean history (off by default).
    vk::Buffer deblockTmp, cleanBuf[2];
    auto ensureBuf = [&](vk::Buffer& b, VkDeviceSize bytes, const char* label) {
      if (!b.buffer) b = ctx.createBuffer(bytes, false, label);
    };
    // MOONGLIDE_DUMP_WORK=prefix:seq -> prefix.work.rgba (the frame NR gets, after cleanup) and
    // prefix.nr.rgba (NR output) of capture frame `seq`, raw RGBA8 at the work size. For A/B.
    // Several frames: prefix:s1,s2,... -> prefix.<seq>.work.rgba / .nr.rgba.
    std::string dumpPrefix;
    std::vector<uint64_t> dumpSeqs;
    uint64_t dumpSeq = UINT64_MAX;
    bool dumpPending = false;
    if (const char* d = getenv("MOONGLIDE_DUMP_WORK")) {
      std::string v = d;
      const size_t colon = v.rfind(':');
      dumpPrefix = v.substr(0, colon);
      std::string list = colon == std::string::npos ? "60" : v.substr(colon + 1);
      for (size_t pos = 0; pos <= list.size();) {
        size_t comma = list.find(',', pos);
        if (comma == std::string::npos) comma = list.size();
        dumpSeqs.push_back(strtoull(list.substr(pos, comma - pos).c_str(), nullptr, 10));
        pos = comma + 1;
      }
      std::sort(dumpSeqs.begin(), dumpSeqs.end());
    }
    vk::Buffer dumpWork = dumpPrefix.empty() ? vk::Buffer{} : ctx.createBuffer(workBytes, true, "dump work");
    vk::Buffer dumpNr = dumpPrefix.empty() ? vk::Buffer{} : ctx.createBuffer(workBytes, true, "dump nr");
    // Also the final output picture (.out.rgba, swapchain size) and, with own FG, the generated
    // picture between seq-1 and seq (.mid.rgba, work size).
    const VkDeviceSize outBytes = (VkDeviceSize)ext.width * ext.height * 4;
    vk::Buffer dumpOut = dumpPrefix.empty() ? vk::Buffer{} : ctx.createBuffer(outBytes, true, "dump out");
    vk::Buffer dumpMid = dumpPrefix.empty() ? vk::Buffer{} : ctx.createBuffer(workBytes, true, "dump mid");
    bool dumpHasMid = false;
    bool dumpHasNr = false;
    int cleanCur = 0;
    bool frameUsedClean = false;
    bool cleanHave = false;
    int fgCur = 0;
    bool fgHavePrev = false;
    bool prevCorrupt = false;     // Remote Play: the previous real frame was damaged by network loss
    uint64_t generated = 0;
    // 0 start, 4 end of A, 5 start of B, 1 after NR, 2 after present, 3 end, 6/7 deblock, 8/9 temporal
    constexpr uint32_t kStamps = 12;   // 10 end of B / 11 start of C with nvvfx
    VkQueryPool timestamps = ctx.createTimestampPool(kStamps);

    std::unique_ptr<NrStage> nr;
    auto ensureNr = [&]() -> bool {
      if (nr) return true;
      if (!gpu.nrCapable()) {
        fprintf(stderr, "nr: this GPU has no FP8 cooperative matrices (needs RTX 40/50); denoise stays off\n");
        opt.nr = false;
        return false;
      }
      try {
        nr = std::make_unique<NrStage>(ctx, opt.modelDir, MOONGLIDE_NR_SHADER_DIR, workW, workH, opt.nrParams);
        return true;
      } catch (const std::exception& e) {
        fprintf(stderr, "nr: cannot start (%s); running passthrough\n", e.what());
        noteGpuOom("noise reduction", e.what());
        opt.nr = false;
        return false;
      }
    };
    if (opt.nr) ensureNr();

    // Frame resources: one in flight.
    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = ctx.queueFamily();
    VkCommandPool pool;
    VK_CHECK(vkCreateCommandPool(gpu.device(), &poolInfo, nullptr, &pool));
    VkCommandBufferAllocateInfo cbInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbInfo.commandPool = pool;
    cbInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbInfo.commandBufferCount = 4;
    VkCommandBuffer cmds[4];
    VK_CHECK(vkAllocateCommandBuffers(gpu.device(), &cbInfo, cmds));
    VkCommandBuffer cmd = cmds[0], cmdB = cmds[1], cmdC = cmds[2], idleCmd = cmds[3];
    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    VkFence fence, idleFence;
    VK_CHECK(vkCreateFence(gpu.device(), &fenceInfo, nullptr, &fence));
    VK_CHECK(vkCreateFence(gpu.device(), &fenceInfo, nullptr, &idleFence));
    VkSemaphoreCreateInfo semInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkSemaphore acquired, acquiredMid, acquiredMore[Rife::kMaxOut - 1], lumaReady, flowReady;
    VK_CHECK(vkCreateSemaphore(gpu.device(), &semInfo, nullptr, &acquired));
    VK_CHECK(vkCreateSemaphore(gpu.device(), &semInfo, nullptr, &acquiredMid));
    for (VkSemaphore& s : acquiredMore) VK_CHECK(vkCreateSemaphore(gpu.device(), &semInfo, nullptr, &s));
    VK_CHECK(vkCreateSemaphore(gpu.device(), &semInfo, nullptr, &lumaReady));
    VK_CHECK(vkCreateSemaphore(gpu.device(), &semInfo, nullptr, &flowReady));
    VkSemaphore idleAcquired;
    VK_CHECK(vkCreateSemaphore(gpu.device(), &semInfo, nullptr, &idleAcquired));

    // Motion: the GPU's optical-flow engine on the work-size frames.
    // Made on first use (NR, clean or own FG running): RIFE alone does not need it.
    std::unique_ptr<OpticalFlow> flow;
    auto ensureFlow = [&]() {
      if (flow || !opt.flow) return;
      try {
        flow = std::make_unique<OpticalFlow>(gpu, workW, workH);
      } catch (const std::exception& e) {
        fprintf(stderr, "optical flow: unavailable (%s); NR history uses zero motion\n", e.what());
        noteGpuOom("optical flow", e.what());
        opt.flow = false;
      }
    };
    if (opt.flowCheck) ensureFlow();
    // --flow-check: reprojection error of the previous frame with vs without the flow.
    const bool flowCheck = opt.flowCheck && flow;
    vk::Buffer prevColour = flowCheck ? ctx.createBuffer((VkDeviceSize)workW * workH * 4, false, "prev colour") : vk::Buffer{};
    vk::Buffer checkBuffer = ctx.createBuffer(16, true, "flow check sums");
    vk::Pipeline checkPipe = flowCheck ? gpu.pipeline("flow_check") : vk::Pipeline{};
    bool havePrevColour = false, frameUsedFlow = false;
    Samples flowErr(90), zeroErr(90), flowMs(90);

    std::unique_ptr<PresentWaiter> waiter;
    if (gpu.presentWaitEnabled()) waiter = std::make_unique<PresentWaiter>(gpu.device(), gpu.swapchain());

    // Letterbox rect preserving the capture aspect.
    uint32_t rectW = 0, rectH = 0, rectX = 0, rectY = 0;
    auto computeRect = [&]() {
      rectW = ext.width;
      rectH = (uint32_t)((uint64_t)ext.width * capH / capW);
      if (rectH > ext.height) {
        rectH = ext.height;
        rectW = (uint32_t)((uint64_t)ext.height * capW / capH);
      }
      rectX = (ext.width - rectW) / 2;
      rectY = (ext.height - rectH) / 2;
    };
    computeRect();
    vk::Buffer easuBuffer, rcasBuffer;   // made when FSR first runs

    // Optional NVIDIA Video Effects chain (user-installed SDK).
    std::unique_ptr<Nvvfx> nv;
    if (!opt.nvvfx.sdk.empty()) {
      try {
        nv = std::make_unique<Nvvfx>(gpu, opt.nvvfx, workW, workH, rectW, rectH);
      } catch (const std::exception& e) {
        fprintf(stderr, "nvvfx: unavailable (%s)\n", e.what());
        noteGpuOom("NVIDIA Maxine effects", e.what());
      }
      if (nv && nv->denoise()) {
        // Its denoise replaces the shader deblock; the temporal clean runs on its output.
        if (!opt.cleanSet) opt.clean = true;
        if (!opt.deblockSet) opt.deblockStrength = 0.f;
      }
    }
    const bool nvDn = nv && nv->denoise();
    const bool nvChain = nv && (nv->sr() || nv->fg());   // B ends after NR; C presents the effects' output
    const bool nvFg = nv && nv->fg();
    if (nvChain && rife) {
      fprintf(stderr, "rife: off (the NVIDIA effects chain presents)\n");
      fgRife = false;
    }
    vk::Pipeline colourFixPipe = nvDn ? gpu.pipeline("colour_fix") : vk::Pipeline{};
    if (nvChain && !fgMixPipe.pipeline) fgMixPipe = gpu.pipeline("fg_mix");
    // The effects' colour cast, measured on the live frames: per-channel sums of each effect's
    // input and output (slots: 0 dn in, 1 dn out, 2 sr in, 3 sr out, 4 fg out, 5 fg other input),
    // smoothed, undone as an offset in the shaders that read the effects' output.
    const bool nvMeasure = nv && opt.nvvfx.colourFix;
    vk::Pipeline sumsPipe = nvMeasure ? gpu.pipeline("channel_sums") : vk::Pipeline{};
    vk::Buffer sumsBuffer = ctx.createBuffer(18 * 4, true, "nvvfx channel sums");
    bool measuredDn = false, measuredSr = false, measuredFg = false;
    float castDn[3] = {0, 0, 0}, castSr[3] = {0, 0, 0}, castFg[3] = {0, 0, 0};
    ColourFix fixDn, fixUp, fixMid;
    auto channelSums = [&](VkCommandBuffer c, const vk::Buffer& b, uint32_t pixels, uint32_t slot) {
      struct { uint32_t count, slot; } sp{pixels, slot};
      gpu.dispatch(c, sumsPipe, {&b, &sumsBuffer}, &sp, sizeof(sp), 256, 1);
    };
    int nvCur = 0;
    bool nvHavePrev = false;
    // Default on (MOONGLIDE_NVVFX_HOSTWAIT=0 to queue the effects behind the semaphore right away;
    // measured: that time-slices NR 8.3 -> 17.9 ms).
    const bool nvHostWait = !getenv("MOONGLIDE_NVVFX_HOSTWAIT") || atoi(getenv("MOONGLIDE_NVVFX_HOSTWAIT"));
    uint64_t nvBase = 0;
    Samples nvDnMs(90), nvSrMs(90), nvFgMs(90), nvGapMs(90), vkBusyMs(90);

    capture.start(slotPtrs);
    // A second instance (tests, replays) must not take over the live one's control fifo or
    // status file: if someone already reads the fifo, both get a per-pid path.
    bool secondInstance = false;
    {
      // Only a fifo with a reader counts: a leftover regular file (or anything else) at the path
      // is replaced by ControlFifo.
      struct stat cst;
      const bool isFifo = stat(opt.controlPath.c_str(), &cst) == 0 && S_ISFIFO(cst.st_mode);
      const int probe = isFifo ? open(opt.controlPath.c_str(), O_WRONLY | O_NONBLOCK) : -1;
      if (probe >= 0) {
        close(probe);
        secondInstance = true;
        const std::string pid = "." + std::to_string(getpid());
        fprintf(stderr, "control: %s is in use by another instance; using %s%s", opt.controlPath.c_str(),
                opt.controlPath.c_str(), pid.c_str());
        opt.controlPath += pid;
        if (opt.statusPath == "/dev/shm/moonglide.status") opt.statusPath += pid;
        fprintf(stderr, " and %s\n", opt.statusPath.c_str());
      }
    }
    ControlFifo control(opt.controlPath);
    // Only the main instance writes the settings file.
    const bool configWritable = configOn && !secondInstance;
    if (configOn && secondInstance) fprintf(stderr, "config: another instance is running; not saving settings\n");
    fprintf(stderr, "ready: %ux%u %s -> work %ux%u -> %ux%u; control: %s\n", capW, capH,
            pixelFormatName(capture.format()), workW, workH, ext.width, ext.height, opt.controlPath.c_str());

    bool split = false, resetHistory = true, inFlight = false;
    int heldSlot = -1;
    uint64_t presentId = 0, presented = 0;
    double lastPresent = 0, lastLog = monotonicMs();
    RateMeter realFps(1000.0), presentFps(1000.0);
    Samples e2eCapture(90), e2eDequeue(90), gpuMs(90), nrMs(90), cleanMs(90), decodeMs(90), submitLatency(90);
    bool frameUsedNr = false;

    auto adjust = [](float& v, float d, float lo, float hi) { v = std::fmin(hi, std::fmax(lo, v + d)); };
    std::mutex queueMutex;
    DelayedPresenter delayed(gpu.queue(), queueMutex);
    std::unique_ptr<Ui> ui;
    if (!getenv("MOONGLIDE_NO_UI")) {
      try {
        ui = std::make_unique<Ui>(gpu, window.display(), window.window(), (uint32_t)gpu.images().size());
      } catch (const std::exception& e) {
        fprintf(stderr, "ui: settings overlay unavailable: %s\n", e.what());
      }
      if (ui) ui->setDesktop(opt.uiOpen);
      if (ui && opt.uiOpen) ui->setVisible(true);
    }
    // Where the picture goes this frame: the letterbox rect, or the app layout's picture area
    // while the UI is open. The effects' upscale output is rect-sized, so it stays put then.
    struct ViewRect { uint32_t x, y, w, h; };
    auto viewRect = [&]() -> ViewRect {
      if (!ui || !ui->visible() || (nv && nv->sr())) return {rectX, rectY, rectW, rectH};
      float ax, ay, aw, ah;
      ui->pictureArea(ext.width, ext.height, ax, ay, aw, ah);
      uint32_t w = (uint32_t)aw, h = (uint32_t)((uint64_t)w * capH / capW);
      if (h > (uint32_t)ah) { h = (uint32_t)ah; w = (uint32_t)((uint64_t)h * capW / capH); }
      w = std::min(w & ~1u, rectW);
      h = std::min(h & ~1u, rectH);
      return {(uint32_t)ax + ((uint32_t)aw - w) / 2, (uint32_t)ay + ((uint32_t)ah - h) / 2, w, h};
    };
    bool rangeLimitedNow = false;   // the last frame's range decision (for the UI)
    std::deque<float> realIntervals;   // ms between unique frames, for the 1% low
    double lastRealTick = 0;
    auto setNr = [&](bool on) {
      delayed.flush();   // NR creation uses the queue
      if (on && !ensureNr()) return;
      opt.nr = on;
      resetHistory = true;
      fprintf(stderr, "nr: %s\n", on ? "ON" : "OFF");
    };
    auto onKey = [&](unsigned long key) {
      switch (key) {
        case XK_n: case XK_F1: setNr(!opt.nr); break;
        case XK_Up: case XK_plus: case XK_equal: case XK_KP_Add: adjust(opt.intensity, 0.05f, 0.f, 1.f); break;
        case XK_Down: case XK_minus: case XK_KP_Subtract: adjust(opt.intensity, -0.05f, 0.f, 1.f); break;
        case XK_bracketright: if (nr) adjust(nr->params.localStructure, 0.05f, 0.f, 2.f); break;
        case XK_bracketleft: if (nr) adjust(nr->params.localStructure, -0.05f, 0.f, 2.f); break;
        case XK_s: split = !split; break;
        case XK_u: opt.fsr = !opt.fsr; break;
        // O cycles the stats overlay: small > full > off.
        case XK_o:
          if (!opt.overlay) { opt.overlay = true; opt.overlayCompact = true; }
          else if (opt.overlayCompact) opt.overlayCompact = false;
          else opt.overlay = false;
          break;
        case XK_h: opt.hudShow = !opt.hudShow; break;
        case XK_c: opt.clean = !opt.clean; break;
        case XK_g: fgOwn = fgOwnAvailable && !fgOwn; break;
        case XK_r: resetHistory = true; break;
        // Only Ctrl+Q quits: Moonlight forwards every key, and a stray Esc or Q from a game
        // menu used to close the stream.
        case XK_q | X11Window::kKeyCtrl: quitBecause("Ctrl+Q"); break;
        case XK_F11: window.setFullscreen(!window.fullscreen()); break;
        default: break;
      }
    };
    // --- settings (file, overlay, presets) ------------------------------------------
    // Keys that only take effect when the process starts.
    const std::vector<std::string> restartKeys = {"source", "rp-host", "rp-bitrate", "rp-codec", "rp-decoder", "rp-wake", "rp-res", "rp-fps",
                                                  "device", "size", "fps", "format", "work", "windowed", "present",
                                                  "images", "fg", "nvvfx", "nvvfx-denoise", "nvvfx-sr",
                                                  "nvvfx-fg", "nvvfx-colour-fix"};
    Settings pending;   // restart-only settings changed since start ("set KEY VALUE")
    auto num = [](double v) {
      char b[32];
      snprintf(b, sizeof(b), "%.3f", v);
      std::string r = b;
      while (r.size() > 1 && r.back() == '0') r.pop_back();
      if (r.back() == '.') r.pop_back();
      return r;
    };
    auto fgModeName = [](int fg) { return std::string(fg == 1 ? "own" : fg == 2 ? "off" : fg == 3 ? "rife" : "lsfg"); };
    // FG modes that switch live (set up at start).
    auto fgLive = [&](const std::string& m) {
      return m == "off" ? fgOwnAvailable : m == "own" ? fgFlowAvailable : m == "rife" ? rife && !nvChain : false;
    };
    // Every setting as it is now: live values, then pending restart-only ones on top.
    auto currentSettings = [&]() {
      Settings st;
      const NrParams& np = nr ? nr->params : opt.nrParams;
      st["device"] = opt.capture.device;
      st["source"] = opt.capture.source;
      st["rp-host"] = opt.capture.rp.host;
      st["rp-bitrate"] = std::to_string(opt.capture.rp.bitrate);
      st["rp-codec"] = opt.capture.rp.codec;
      st["rp-decoder"] = opt.capture.rp.decoder;
      st["rp-wake"] = opt.capture.rp.wake ? "on" : "off";
      if (opt.capture.rp.res > 0) st["rp-res"] = std::to_string(opt.capture.rp.res);
      if (opt.capture.rp.fps > 0) st["rp-fps"] = std::to_string(opt.capture.rp.fps);
      if (!opt.capture.rp.psnAccount.empty()) {
        st["rp-psn-account"] = opt.capture.rp.psnAccount;
        st["rp-psn-online"] = opt.capture.rp.psnOnline;
      }
      if (!opt.capture.rp.tap.empty()) st["rp-tap"] = opt.capture.rp.tap;
      if (opt.capture.rp.shm != RP_TAP_DEFAULT_NAME) st["rp-shm"] = opt.capture.rp.shm;
      st["size"] = fmt("%ux%u", opt.capture.width, opt.capture.height);
      st["fps"] = std::to_string(opt.capture.fps);
      std::string f = pixelFormatName(opt.capture.format);
      for (char& ch : f) ch = (char)tolower(ch);
      st["format"] = f;
      if (opt.workW) st["work"] = fmt("%ux%u", opt.workW, opt.workH);
      if (opt.windowedSet) st["windowed"] = fmt("%ux%u", opt.winW, opt.winH);
      st["present"] = opt.presentMode == VK_PRESENT_MODE_MAILBOX_KHR ? "mailbox"
                      : opt.presentMode == VK_PRESENT_MODE_IMMEDIATE_KHR ? "immediate" : "fifo";
      if (opt.imagesSet) st["images"] = std::to_string(opt.swapImages);
      st["nr"] = opt.nr ? "true" : "false";
      st["intensity"] = num(opt.intensity);
      st["local-structure"] = num(np.localStructure);
      st["local-tone"] = num(np.localTone);
      st["skin-structure"] = num(np.skinStructure);
      st["dedup"] = opt.capture.dedup == 1 ? "on" : opt.capture.dedup == 0 ? "off" : "auto";
      st["dongle-quality"] = opt.capture.jpegQuality < 0 ? "off" : std::to_string(opt.capture.jpegQuality);
      st["dongle-auto"] = opt.dongleAuto ? "on" : "off";
      st["range"] = opt.range == 1 ? "full" : opt.range == 0 ? "limited" : "auto";
      st["overlay"] = opt.overlay ? "true" : "false";
      st["overlay-pos"] = std::vector<std::string>{"tl", "tr", "bl", "br"}[opt.overlayPos & 3];
      st["overlay-size"] = std::to_string(opt.overlaySize);
      st["overlay-pc"] = std::to_string(opt.overlayPc);
      st["overlay-mode"] = opt.overlayCompact ? "compact" : "full";
      if (opt.wizardDone) st["wizard"] = "done";
      std::string hud;
      for (const HudZone& z : opt.hudZones)
        hud += (hud.empty() ? "" : "\n") + hudZoneString(z) + (z.feather > 0 ? ":" + num(z.feather) : "");
      st["hud"] = hud;
      st["hud-show"] = opt.hudShow ? "true" : "false";
      st["fg"] = fgOwnAvailable ? (fgOwn ? (fgRife ? "rife" : "own") : "off") : fgModeName(opt.fgRequested);
      st["fg-flags"] = std::to_string(opt.fgFlags);
      st["fg-hold"] = num(opt.fgHold);
      st["fg-mult"] = std::to_string(opt.fgMult);
      if (opt.fgRealDelay > 0.f) st["fg-real-delay"] = num(opt.fgRealDelay);
      st["rife-still"] = num(opt.rifeStill);
      st["rife-outlier"] = num(opt.rifeOutlier);
      st["rife-cut"] = num(opt.rifeCut);
      if (opt.cleanSet) st["clean"] = opt.clean ? "true" : "false";
      st["clean-strength"] = num(opt.cleanStrength);
      if (opt.deblockSet) st["deblock"] = num(opt.deblockStrength);
      if (opt.deband > 0.f) st["deband"] = num(opt.deband);
      st["flow"] = opt.flow ? "true" : "false";
      st["upscale"] = opt.fsr ? "fsr" : "bicubic";
      st["sharpen"] = num(opt.sharpenStops);
      st["nvvfx"] = opt.nvvfx.sdk;
      auto q = [](int v) { return v < 0 ? std::string("off") : std::to_string(v); };
      st["nvvfx-denoise"] = q(opt.nvvfx.denoise);
      st["nvvfx-sr"] = q(opt.nvvfx.sr);
      st["nvvfx-fg"] = q(opt.nvvfx.fg);
      st["nvvfx-colour-fix"] = opt.nvvfx.colourFix ? "on" : "off";
      for (const auto& [k, v] : pending) st[k] = v;
      return st;
    };
    Settings runningSettings;
    {
      const Settings now = currentSettings();
      for (const std::string& k : restartKeys)
        if (auto it = now.find(k); it != now.end()) runningSettings[k] = it->second;
    }
    auto saveSettings = [&]() {
      if (!configWritable) return;
      const Settings now = currentSettings();
      if (now == config.active) return;
      config.active = now;
      if (!saveConfig(config)) fprintf(stderr, "config: cannot write %s\n", config.path.c_str());
    };
    // What the picture area says when something is wrong (and why it is black).
    std::string rpStatusSeen;
    double rpStatusSince = 0;
    bool rifeNoticeDismissed = false;
    auto banners = [&]() {
      std::vector<UiBanner> b;
      const double now = monotonicMs();
      const bool noPicture = lastRealTick == 0 || now - lastRealTick > 1000;
      if (!rifeError.empty() && !rifeNoticeDismissed) {
        UiBanner r;
        r.level = 1;
        if (looksLikeGpuOom(rifeError)) {
          r.title = "Not enough GPU memory for RIFE";
          r.detail = std::string("Another program is using the GPU; close it or lower the settings, then restart. ") +
                     (fgOwnAvailable ? "Generated frames now come from the built-in optical flow." : "Frame generation is off.") +
                     gpuUsersText();
        } else if (moonglide::RifeInstaller::missingError(rifeError)) {
          r.title = "RIFE is not installed";
          r.detail = std::string("RIFE gives the best generated frames. It needs NVIDIA TensorRT and the RIFE model "
                                 "(about 4.5 GB download, installed into your home folder). ") +
                     (fgOwnAvailable ? "Until then the built-in optical flow generates the frames." : "Frame generation is off.");
          if (gRifeInstall.state() == moonglide::RifeInstaller::Idle) r.actions.push_back({"Install RIFE", "rife install"});
        } else {
          r.title = "RIFE unavailable";
          r.detail = rifeError;
        }
        r.actions.push_back({"Dismiss", "notice dismiss"});
        b.push_back(r);
      }
      {
        std::lock_guard<std::mutex> lock(gOomMutex);
        if (!gOomDismissed)
          for (const std::string& f : gOomFeatures)
            b.push_back({"Not enough GPU memory for " + f,
                         "Another program is using the GPU; close it or lower the settings, then restart. " + f +
                             " is off for now." + gpuUsersText(),
                         1, {{"Dismiss", "notice dismiss"}}});
      }
      if (noPicture && capture.remotePlay()) {
        const auto rs = capture.remotePlayStats();
        const std::string host = opt.capture.rp.host.empty() ? std::string("PS5") : "PS5 " + opt.capture.rp.host;
        if (rs.tapStatus != rpStatusSeen) { rpStatusSeen = rs.tapStatus; rpStatusSince = now; }
        int retryIn = -1;
        if (const size_t k = rs.tapStatus.find("retry in "); k != std::string::npos)
          retryIn = std::max(0, atoi(rs.tapStatus.c_str() + k + 9) - (int)((now - rpStatusSince) / 1000));
        const std::string retry = retryIn > 0 ? fmt(" Retrying in %d s.", retryIn) : retryIn == 0 ? " Retrying now…" : "";
        switch (rs.tapState) {
          case -1: b.push_back({"Remote Play stopped", "moonglide-chiaki-tap is not running; starting it again in a few seconds.", 2, {}}); break;
          case -2: case RP_TAP_IDLE: b.push_back({"Starting Remote Play…", host, 0, {}}); break;
          case RP_TAP_WAKING: b.push_back({"Waking the PS5…", host, 0, {}}); break;
          case RP_TAP_STANDBY: b.push_back({"PS5 is in rest mode", "Wake it from here, or turn it on with the controller: MoonGlide connects once it is on.", 0, {{"Wake PS5", "rp wake"}}}); break;
          case RP_TAP_CONNECTING: b.push_back({"Connecting to the PS5…", host, 0, {}}); break;
          case RP_TAP_STREAMING: b.push_back({"Waiting for the picture from the PS5…", host, 0, {}}); break;
          default: {
            std::string st = rs.tapStatus, low = st;
            for (char& ch : low) ch = (char)tolower((unsigned char)ch);
            const bool busy = low.find("busy") != std::string::npos || low.find("in use") != std::string::npos;
            if (const size_t k = st.find("; retry in"); k != std::string::npos) st.resize(k);
            if (busy)
              b.push_back({"PS5 busy (another Remote Play session)",
                           "Close the other session (phone, another PC, chiaki-ng)." + retry, 1, {}});
            else if (low.find("no user signed in") != std::string::npos)
              b.push_back({"No user signed in on the PS5",
                           "Pick a user on the console screen. To skip this next time: PS5 Settings > Users and Accounts > "
                           "Login Settings > Log In to PS5 Automatically." + retry, 1, {}});
            else
              b.push_back({"Remote Play interrupted", (st.empty() ? std::string("The session ended.") : st) + retry, 1, {}});
          }
        }
      } else if (noPicture) {
        b.push_back({"No signal from the capture card", "Check the HDMI source and cable (" + opt.capture.device + ").", 1, {}});
      }
      return b;
    };
    // The in-picture stats card's data (drawn by Ui over every presented picture).
    auto fillHud = [&](UiHud& hd) {
      hd.on = opt.overlay;
      if (!hd.on) return;
      const double t = monotonicMs();
      const auto cs = capture.stats();
      hd.compact = opt.overlayCompact;
      hd.pos = opt.overlayPos & 3;
      hd.size = opt.overlaySize;
      hd.pc = opt.overlayPc;
      hd.fps = (float)presentFps.rate(t);
      hd.real = (float)realFps.rate(t);
      hd.budget = capture.fps() > 0 ? (float)(1000.0 / (cs.dedupActive && cs.dedupLocked ? capture.fps() / 2 : capture.fps())) : 33.3f;
      hd.fg = nvFg ? "NVIDIA" : (fgOwn && fgRife && rife) ? fmt("RIFE x%d", std::clamp(opt.fgMult, 2, Rife::kMaxOut + 1)) : fgOwn && opt.flow ? "own x2" : lsfg ? "x2" : "";
      hd.e2e = e2eCapture.size() ? (float)e2eCapture.percentile(0.5) : -1.f;
      hd.gpu = (float)gpuMs.mean();
      hd.nr = (float)nrMs.mean();
      hd.flow = (float)flowMs.mean();
      hd.decode = (float)decodeMs.mean();
      hd.dropped = cs.dropped;
      hd.dup = cs.duplicates;
      hd.denoise = opt.nr ? fmt("on, %.2f", opt.intensity) : nv && nv->denoise() ? std::string("NVIDIA") : std::string("off");
      hd.upscale = fmt("%s %ux%u > %ux%u", opt.fsr ? "FSR" : "Bicubic", workW, workH, rectW, rectH);
      hd.capSize = fmt("%ux%u %s", capW, capH, pixelFormatName(capture.format()));
      hd.rp = capture.remotePlay();
      if (hd.rp) {
        const auto rs = capture.remotePlayStats();
        hd.rtt = (float)rs.rttMs;
        hd.kbps = (float)rs.kbps;
        hd.lost = rs.lost;
        hd.corrupt = rs.corrupt;
        hd.codec = rs.decoder;
        hd.state = rs.state;
        hd.inputLag = rs.inputLagSamples ? (float)(rs.inputLagMs + (e2eCapture.size() ? e2eCapture.percentile(0.5) : 0.0)) : -1.f;
        hd.inputSamples = rs.inputLagSamples;
        hd.inputMethod = rs.inputLagMethod;
      }
      if (opt.overlayPc) hd.sys = moonglide::sysProbe().get();
      hd.frameMs.assign(realIntervals.begin() + (realIntervals.size() > 60 ? realIntervals.size() - 60 : 0), realIntervals.end());
    };
    auto uiState = [&]() {
      UiState u;
      fillHud(u.hud);
      u.current = currentSettings();
      u.running = runningSettings;
      // Own FG on/off switches live when it was set up at start.
      if (fgLive(u.current["fg"])) u.running["fg"] = u.current["fg"];
      u.restartKeys = restartKeys;
      for (const auto& [name, p] : config.presets) u.presets.push_back(name);
      u.zones = opt.hudZones;
      const ViewRect vr = viewRect();
      u.rectX = vr.x; u.rectY = vr.y; u.rectW = vr.w; u.rectH = vr.h;
      u.fgActive = nvFg ? "VFG (NVIDIA)" : fgOwn ? (fgRife ? (opt.fgMult > 2 ? fmt("RIFE x%d", opt.fgMult) : std::string("RIFE")) : std::string("own")) : lsfg ? "lsfg" : "off";
      u.ownAvailable = fgOwnAvailable;
      u.nrCapable = gpu.nrCapable();
      u.rifeMissing = !rife && (moonglide::RifeInstaller::missingError(rifeError) ||
                                (rifeError.empty() && !moonglide::RifeInstaller::filesPresent()));
      u.rifeInstall = (int)gRifeInstall.state();
      u.rifeInstallLine = gRifeInstall.line();
      u.lsfgLoaded = lsfg;
      u.nvvfxStages = !nv ? "" : fmt("%s%s%s", nv->denoise() ? "denoise " : "", nv->sr() ? "upscale " : "", nv->fg() ? "framegen" : "");
      u.srcW = capW; u.srcH = capH; u.srcFps = (float)capture.fps(); u.outW = ext.width; u.outH = ext.height;
      u.sys = moonglide::sysProbe().get();
      u.noPicture = lastRealTick == 0 || monotonicMs() - lastRealTick > 1000;
      u.configPath = config.path;
      u.configWritable = configWritable;
      u.split = split;
      UiStats& st = u.stats;
      const double t = monotonicMs();
      st.realFps = (float)realFps.rate(t);
      st.presentFps = (float)presentFps.rate(t);
      if (!realIntervals.empty()) {
        std::vector<float> iv(realIntervals.begin(), realIntervals.end());
        std::sort(iv.begin(), iv.end());
        const float p99 = iv[std::min(iv.size() - 1, (size_t)(iv.size() * 0.99))];
        st.low1Fps = p99 > 0 ? 1000.f / p99 : 0.f;
        st.frameMs.assign(realIntervals.begin() + (realIntervals.size() > 200 ? realIntervals.size() - 200 : 0),
                          realIntervals.end());
      }
      st.e2e = (float)e2eCapture.percentile(0.5);
      st.e2e90 = (float)e2eCapture.percentile(0.9);
      st.dq = (float)e2eDequeue.percentile(0.5);
      st.gpu = (float)gpuMs.mean();
      st.nr = (float)nrMs.mean();
      st.flow = (float)flowMs.mean();
      st.clean = (float)cleanMs.mean();
      st.decode = (float)decodeMs.mean();
      st.nvDenoise = (float)nvDnMs.mean();
      st.nvSr = (float)nvSrMs.mean();
      st.nvFg = (float)nvFgMs.mean();
      st.vkBusy = (float)vkBusyMs.mean();
      st.fgFail = (float)fgFail;
      st.fgHeld = fgHeld;
      st.generated = generated;
      const auto cs = capture.stats();
      st.budget = capture.fps() > 0 ? (float)(1000.0 / (cs.dedupActive && cs.dedupLocked ? capture.fps() / 2 : capture.fps())) : 33.3f;
      {
        const std::string dev = opt.capture.device;
        u.dongleJpeg = capture.jpegQualitySupported() ? capture.jpegQualityStatus() : "";
        u.sourceChip = capture.remotePlay() ? "PS5 Remote Play"
                       : dev.rfind("replay:", 0) == 0 ? "replay  " + dev.substr(dev.rfind('/') + 1) : dev;
        if (capture.remotePlay()) {
          const auto rs = capture.remotePlayStats();
          u.remotePlay = fmt("%s  ·  %.1f Mbps  ·  lost %llu  ·  %s  ·  sound held %u ms", rs.state.c_str(), rs.kbps / 1000.0,
                             (unsigned long long)rs.lost, rs.decoder.empty() ? "no decoder" : rs.decoder.c_str(), rs.audioHeldMs);
          if (rs.rttMs > 0) u.remotePlay += fmt("  ·  RTT %.1f ms", rs.rttMs);
        }
        const bool limited = rangeLimitedNow;
        u.formatChip = fmt("%up%.0f  ·  %s  ·  SDR %s", capH, capture.fps(), pixelFormatName(capture.format()),
                           limited ? "limited" : "full");
        u.signal = {
            {"Input", fmt("%ux%u %s @ %.2f", capW, capH, pixelFormatName(capture.format()), capture.fps())},
            {"Range", fmt("%s%s", limited ? "limited -> full" : "full", opt.range == -1 ? "  (auto)" : "")},
            {"Dedup", !cs.dedupActive ? "off" : cs.dedupLocked ? fmt("60 -> 30, phase %d", cs.dedupPhase) : "60 -> 30, locking"},
            {"Work", fmt("%ux%u%s", workW, workH, opt.nr ? "  NR" : "")},
            {"Flow", flow && opt.flow ? fmt("NVOF %ux%u grid", flow->gridWidth(), flow->gridHeight()) : "off"},
            {"Output", fmt("%ux%u  %s  %s", ext.width, ext.height, opt.fsr ? "FSR 1" : "bicubic",
                           u.current["present"].c_str())},
            {"Dropped", fmt("%llu  ·  dup %llu", (unsigned long long)cs.dropped, (unsigned long long)cs.duplicates)},
        };
      }
      st.dropped = cs.dropped;
      st.duplicates = cs.duplicates;
      u.banners = banners();
      u.fullscreen = window.fullscreen();
      return u;
    };
    std::function<void(const std::string&)> applySetting;   // one key of a preset, via the commands

    bool swapImagesGrew = false;   // a higher RIFE multiplier needs more swapchain images: rebuild it
    auto onCommand = [&](const std::string& line) {
      std::istringstream in(line);
      std::string c, v;
      in >> c >> v;
      if (c == "nr") setNr(v == "toggle" ? !opt.nr : v == "on");
      else if (c == "intensity") opt.intensity = std::fmin(1.f, std::fmax(0.f, (float)atof(v.c_str())));
      else if (c == "structure") (nr ? nr->params : opt.nrParams).localStructure = (float)atof(v.c_str());
      else if (c == "tone") (nr ? nr->params : opt.nrParams).localTone = (float)atof(v.c_str());
      else if (c == "split") split = v == "on" || (v == "toggle" && !split);
      else if (c == "ui") {
        // "ui show" (the desktop launcher when already running), "ui hide", "ui toggle".
        if (ui) {
          const bool show = v == "show" || (v == "toggle" && !ui->visible());
          ui->setVisible(show);
          if (show) window.raise();
        }
      }
      else if (c == "fullscreen") window.setFullscreen(v == "on" || ((v == "toggle" || v.empty()) && !window.fullscreen()));
      else if (c == "notice" && v == "dismiss") { rifeNoticeDismissed = true; std::lock_guard<std::mutex> lock(gOomMutex); gOomDismissed = true; }
      else if (c == "rife" && v == "install") {
        gRifeInstall.start(std::string(MOONGLIDE_SCRIPTS_DIR) + "/setup-rife.sh");
        rifeNoticeDismissed = true;
      }
      else if (c == "rp" && v == "wake") capture.requestWake();
      else if (c == "rp" && v == "psn") {
        // rp psn ACCOUNT ONLINE_ID | rp psn -   (the PSN sign-in of the pairing flow; no restart needed)
        std::string acc, online;
        in >> acc >> online;
        opt.capture.rp.psnAccount = acc == "-" ? "" : acc;
        opt.capture.rp.psnOnline = acc == "-" ? "" : online;
      }
      else if (c == "overlay" && (v == "pos" || v == "size" || v == "mode" || v == "pc")) {
        std::string arg;
        in >> arg;
        if (v == "pos") opt.overlayPos = arg == "tr" ? 1 : arg == "bl" ? 2 : arg == "br" ? 3 : 0;
        else if (v == "pc") opt.overlayPc = (unsigned)atoi(arg.c_str()) & 15u;
        else if (v == "size") opt.overlaySize = std::clamp(atoi(arg.c_str()), 0, 6);
        else opt.overlayCompact = arg == "compact";
      }
      else if (c == "wizard") opt.wizardDone = v == "done";
      else if (c == "overlay") opt.overlay = v == "on" || (v == "toggle" && !opt.overlay);
      else if (c == "dongle" && v == "auto") { std::string arg; in >> arg; opt.dongleAuto = arg != "off"; }
      else if (c == "dongle" && v == "quality") {
        std::string arg;
        in >> arg;
        opt.capture.jpegQuality = arg == "off" ? -1 : std::clamp(atoi(arg.c_str()), 1, 94);
        capture.setJpegQuality(opt.capture.jpegQuality);
      } else if (c == "dedup") { opt.capture.dedup = v == "on" ? 1 : v == "off" ? 0 : -1; capture.setDedup(opt.capture.dedup); }
      else if (c == "hud") {
        // hud add <zone> | hud clear | hud pop | hud show on|off|toggle
        std::string arg;
        in >> arg;
        HudZone z;
        if (v == "add" && parseHudZone(arg, z) && opt.hudZones.size() < kMaxHudZones) opt.hudZones.push_back(z);
        else if (v == "clear") opt.hudZones.clear();
        else if (v == "pop" && !opt.hudZones.empty()) opt.hudZones.pop_back();
        else if (v == "set") {
          // hud set INDEX ZONE
          std::string zs;
          in >> zs;
          const size_t i = (size_t)atoi(arg.c_str());
          if (i < opt.hudZones.size() && parseHudZone(zs, z)) opt.hudZones[i] = z;
        } else if (v == "del") {
          const size_t i = (size_t)atoi(arg.c_str());
          if (i < opt.hudZones.size()) opt.hudZones.erase(opt.hudZones.begin() + (long)i);
        }
        else if (v == "show") opt.hudShow = arg == "on" || (arg == "toggle" && !opt.hudShow);
        else fprintf(stderr, "control: bad hud command '%s'\n", line.c_str());
        for (const HudZone& hz : opt.hudZones) fprintf(stderr, "hud: %s\n", hudZoneString(hz).c_str());
      } else if (c == "range") opt.range = v == "full" ? 1 : v == "limited" ? 0 : -1;
      else if (c == "reset") resetHistory = true;
      else if (c == "upscale") opt.fsr = v != "bicubic";
      else if (c == "sharpen") opt.sharpenStops = std::fmax(0.f, (float)atof(v.c_str()));
      else if (c == "flow") { opt.flow = v == "on" && (flow || gpu.opticalFlow()); resetHistory = true; }
      else if (c == "clean" && v == "strength") { std::string arg; in >> arg; opt.cleanStrength = (float)atof(arg.c_str()); }
      else if (c == "clean") { opt.clean = v == "on" || (v == "toggle" && !opt.clean); opt.cleanSet = true; }
      else if (c == "deband") opt.deband = std::clamp((float)atof(v.c_str()), 0.f, 8.f);
      else if (c == "deblock") { opt.deblockStrength = std::fmax(0.f, (float)atof(v.c_str())); opt.deblockSet = true; }
      else if (c == "fg" && v == "debug") { std::string arg; in >> arg; opt.fgDebug = arg == "on"; }
      else if (c == "fg" && v == "flags") { std::string arg; in >> arg; opt.fgFlags = (uint32_t)strtoul(arg.c_str(), nullptr, 0) & 494u; }
      else if (c == "fg" && v == "mult") { std::string arg; in >> arg; opt.fgMult = std::clamp(atoi(arg.c_str()), 2, 4);
        if (!opt.imagesSet && opt.fg == 3 && opt.swapImages < (uint32_t)(2 * opt.fgMult + 1)) {
          opt.swapImages = (uint32_t)(2 * opt.fgMult + 1);
          swapImagesGrew = true;
        } }
      else if (c == "fg" && v == "real-delay") { std::string arg; in >> arg; opt.fgRealDelay = std::clamp((float)atof(arg.c_str()), 0.f, 0.5f); }
      else if (c == "fg" && v == "hold") { std::string arg; in >> arg; opt.fgHold = std::fmax(0.f, (float)atof(arg.c_str())); }
      else if (c == "rife" && v == "still") { std::string arg; in >> arg; opt.rifeStill = std::fmax(0.f, (float)atof(arg.c_str())); }
      else if (c == "rife" && v == "outlier") { std::string arg; in >> arg; opt.rifeOutlier = std::fmax(0.f, (float)atof(arg.c_str())); }
      else if (c == "rife" && v == "cut") { std::string arg; in >> arg; opt.rifeCut = std::fmax(0.f, (float)atof(arg.c_str())); }
      else if (c == "fg" && (v == "on" || v == "off" || v == "toggle"))
        fgOwn = fgOwnAvailable && (v == "on" || (v == "toggle" && !fgOwn));
      else if (c == "quit") quitBecause("control fifo \"quit\"");
      else if (c == "skin") (nr ? nr->params : opt.nrParams).skinStructure = (float)atof(v.c_str());
      else if (c == "fg" && v == "mode") {
        // own <-> off switch live when own FG was set up at start; everything else restarts.
        std::string arg;
        in >> arg;
        if (arg == "lsfg" || arg == "own" || arg == "off" || arg == "rife") {
          if (fgLive(arg)) {
            fgOwn = arg != "off";
            if (arg != "off") fgRife = arg == "rife";
            pending.erase("fg");
          }
          else if (arg == runningSettings["fg"]) pending.erase("fg");
          else pending["fg"] = arg;
        }
      } else if (c == "set") {
        // set KEY VALUE: a restart-only setting (applied by "restart").
        std::string rest;
        std::getline(in, rest);
        rest = rest.substr(std::min(rest.size(), rest.find_first_not_of(' ')));
        if (rest == "\"\"") rest.clear();
        bool known = false;
        for (const std::string& k : restartKeys) known |= k == v;
        if (!known) fprintf(stderr, "control: '%s' is not a restart setting\n", v.c_str());
        else if (runningSettings[v] == rest) pending.erase(v);
        else pending[v] = rest;
      } else if (c == "restart") {
        // Everything as it is now goes on the new command line (the file may be read-only).
        saveSettings();
        gRestartArgs = settingsToArgs(currentSettings());
        // The window as it is now, and the UI if it is open.
        if (window.fullscreen()) gRestartArgs.push_back("--fullscreen");
        else { gRestartArgs.push_back("--windowed"); gRestartArgs.push_back(fmt("%ux%u", window.width(), window.height())); }
        if (ui && ui->visible()) gRestartArgs.push_back("--ui");
        gRestart = true;
        quitBecause("settings restart");
      } else if (c == "preset") {
        std::string name;
        std::getline(in, name);
        name = name.substr(std::min(name.size(), name.find_first_not_of(' ')));
        if (v == "save" && !name.empty()) {
          Settings cur = currentSettings();
          config.presets[name] = cur;
        }
        else if (v == "delete") config.presets.erase(name);
        else if (v == "load" && config.presets.count(name)) {
          const Settings p = config.presets[name];
          for (const auto& [k, val] : p) applySetting(k + "\t" + val);
          fprintf(stderr, "preset: %s loaded\n", name.c_str());
        } else fprintf(stderr, "control: bad preset command '%s'\n", line.c_str());
        if (configWritable && !saveConfig(config)) fprintf(stderr, "config: cannot write %s\n", config.path.c_str());
      }
      else fprintf(stderr, "control: unknown command '%s'\n", line.c_str());
    };
    applySetting = [&](const std::string& kv) {
      const size_t tab = kv.find('\t');
      const std::string k = kv.substr(0, tab), val = tab == std::string::npos ? "" : kv.substr(tab + 1);
      const auto onOff = [&](const char* cmd) { onCommand(std::string(cmd) + (val == "true" ? " on" : " off")); };
      bool restartKey = false;
      for (const std::string& r : restartKeys) restartKey |= r == k;
      if (k == "fg") onCommand("fg mode " + val);
      else if (k == "fg-mult") onCommand("fg mult " + val);
      else if (k == "fg-real-delay") onCommand("fg real-delay " + val);
      else if (restartKey) onCommand("set " + k + " " + (val.empty() ? std::string("\"\"") : val));
      else if (k == "nr") onOff("nr");
      else if (k == "intensity") onCommand("intensity " + val);
      else if (k == "local-structure") onCommand("structure " + val);
      else if (k == "local-tone") onCommand("tone " + val);
      else if (k == "skin-structure") onCommand("skin " + val);
      else if (k == "dongle-quality") onCommand("dongle quality " + val);
      else if (k == "dongle-auto") onCommand("dongle auto " + val);
      else if (k == "dedup" || k == "range" || k == "upscale" || k == "sharpen" || k == "deblock" || k == "deband") onCommand(k + " " + val);
      else if (k == "overlay") onOff("overlay");
      else if (k == "overlay-pos") onCommand("overlay pos " + val);
      else if (k == "overlay-size") onCommand("overlay size " + val);
      else if (k == "overlay-mode") onCommand("overlay mode " + val);
      else if (k == "overlay-pc") onCommand("overlay pc " + val);
      else if (k == "wizard") onCommand("wizard " + val);
      else if (k == "flow") onOff("flow");
      else if (k == "clean") onOff("clean");
      else if (k == "clean-strength") onCommand("clean strength " + val);
      else if (k == "fg-flags") onCommand("fg flags " + val);
      else if (k == "fg-hold") onCommand("fg hold " + val);
      else if (k == "rife-still" || k == "rife-outlier" || k == "rife-cut") onCommand("rife " + k.substr(5) + " " + val);
      else if (k == "hud-show") onOff("hud show");
      else if (k == "hud") {
        opt.hudZones.clear();
        std::istringstream zs(val);
        std::string z;
        while (std::getline(zs, z))
          if (!z.empty()) onCommand("hud add " + z);
      }
    };

    // Range auto-detect. Only a full-range source has real content below 8 or above 248, so
    // that flips to full at once; limited needs ~10 s of nothing outside 16-235 *and* blacks
    // sitting near 16. MJPEG cards decode to full-range RGB (the MS2109 measured black ~6,
    // white 255), so MJPEG starts at full; raw YCbCr starts at the HDMI default, limited.
    struct RangeDetector {
      bool limited = false;
      bool sawFull = false;   // levels outside 16-235 seen: full range for the rest of the session
      uint32_t cleanFrames = 0, blackFrames = 0;
      void add(const CapturedFrame& f) {
        if (!f.levelSamples) return;
        const uint32_t n = f.levelSamples;
        if (f.levelBelow8 + f.levelAbove248 > n / 500) {
          if (limited) fprintf(stderr, "range: full (levels outside 16-235)\n");
          limited = false;
          sawFull = true;
          cleanFrames = blackFrames = 0;
          return;
        }
        // A full-range source can go a long while without pure black / white (a dark menu); a
        // limited one never produces them. So the switch to limited only happens before any
        // full-range evidence, or it flips the contrast back and forth.
        if (sawFull) return;
        cleanFrames++;
        if (f.levelNear16 > n / 100) blackFrames++;
        if (!limited && cleanFrames >= 300 && blackFrames >= 30) {
          limited = true;
          fprintf(stderr, "range: limited (no levels outside 16-235 for %u frames, black at 16)\n", cleanFrames);
        }
      }
    } rangeDetector;
    rangeDetector.limited = capture.format() != PixelFormat::MJPEG;

    CapturedFrame frame;
    double submitTime = 0;
    if (capture.fps() > 100.0)
      fprintf(stderr, "WARNING: %.0f fps capture: a 120 Hz source repeats a 30 fps game 4x and the 60->30 dedup "
              "only removes every other frame. Set the console to 60 Hz output.\n", capture.fps());
    FILE* diffLog = getenv("MOONGLIDE_DIFF_LOG") ? fopen(getenv("MOONGLIDE_DIFF_LOG"), "w") : nullptr;
    // New swapchain for the window as it is now (fullscreen toggle, out of date).
    uint32_t lastWinW = window.width(), lastWinH = window.height();
    auto recreateSwapchain = [&]() {
      delayed.flush();
      waiter.reset();
      vkDeviceWaitIdle(gpu.device());
      if (ui) ui->swapchainRecreated();
      gpu.createSwapchain(window.width(), window.height(), opt.presentMode, opt.swapImages);
      lastWinW = window.width();
      lastWinH = window.height();
      const VkExtent2D ne = gpu.extent();
      if (ne.width > screenW || ne.height > screenH) throw std::runtime_error("window larger than the screen");
      if (ne.width != ext.width || ne.height != ext.height) {
        ext = ne;
        computeRect();
        fprintf(stderr, "swapchain: %ux%u, picture %ux%u at %u,%u\n", ext.width, ext.height, rectW, rectH, rectX, rectY);
        // The effects' upscale was made for the old size.
        if (nv && nv->sr()) onCommand("restart");
      }
      if (gpu.presentWaitEnabled()) waiter = std::make_unique<PresentWaiter>(gpu.device(), gpu.swapchain());
    };
    // No picture coming (no stream yet, PS5 busy, no signal): keep the window alive anyway, a
    // dark frame with the UI or the banner saying why, ~15 times a second.
    double lastIdle = 0;
    auto idleFrame = [&]() {
      const double now = monotonicMs();
      if (lastRealTick > 0 && now - lastRealTick < 1000) return;   // a short stall: keep the last picture
      if (now - lastIdle < 14) return;   // ~60 Hz: the UI follows the mouse and keys
      lastIdle = now;
      if (gPresentLog) { fprintf(gPresentLog, "%.3f idle\n", now); fflush(gPresentLog); }
      delayed.flush();
      VK_CHECK(vkWaitForFences(gpu.device(), 1, &fence, VK_TRUE, UINT64_MAX));
      VK_CHECK(vkWaitForFences(gpu.device(), 1, &idleFence, VK_TRUE, UINT64_MAX));
      if (ui) {
        std::vector<std::string> uiCommands;
        ui->build(uiState(), [&](const std::string& c) { uiCommands.push_back(c); });
        for (const std::string& c : uiCommands) onCommand(c);
      }
      uint32_t index = 0;
      const VkResult acq = vkAcquireNextImageKHR(gpu.device(), gpu.swapchain(), 100000000ull, idleAcquired, VK_NULL_HANDLE, &index);
      if (acq == VK_ERROR_OUT_OF_DATE_KHR) { recreateSwapchain(); return; }
      if (acq == VK_TIMEOUT || acq == VK_NOT_READY) return;
      if (acq != VK_SUCCESS && acq != VK_SUBOPTIMAL_KHR) VK_CHECK(acq);
      const VkImage image = gpu.images()[index].image;
      VK_CHECK(vkResetFences(gpu.device(), 1, &idleFence));
      VK_CHECK(vkResetCommandBuffer(idleCmd, 0));
      VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      VK_CHECK(vkBeginCommandBuffer(idleCmd, &bi));
      VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
      b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      b.image = image;
      b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      vkCmdPipelineBarrier(idleCmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
      VkClearColorValue dark{};
      dark.float32[0] = dark.float32[1] = dark.float32[2] = 13.f / 255.f;
      dark.float32[3] = 1.f;
      vkCmdClearColorImage(idleCmd, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &dark, 1, &b.subresourceRange);
      if (!(ui && ui->record(idleCmd, image, ext.width, ext.height))) {
        b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.dstAccessMask = 0;
        vkCmdPipelineBarrier(idleCmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
      }
      VK_CHECK(vkEndCommandBuffer(idleCmd));
      const VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
      VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
      si.waitSemaphoreCount = 1;
      si.pWaitSemaphores = &idleAcquired;
      si.pWaitDstStageMask = &stage;
      si.commandBufferCount = 1;
      si.pCommandBuffers = &idleCmd;
      si.signalSemaphoreCount = 1;
      si.pSignalSemaphores = &gpu.images()[index].renderDone;
      VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
      pi.waitSemaphoreCount = 1;
      pi.pWaitSemaphores = &gpu.images()[index].renderDone;
      pi.swapchainCount = 1;
      VkSwapchainKHR sc = gpu.swapchain();
      pi.pSwapchains = &sc;
      pi.pImageIndices = &index;
      VkResult pr;
      {
        std::lock_guard<std::mutex> q(queueMutex);
        VK_CHECK(vkQueueSubmit(gpu.queue(), 1, &si, idleFence));
        pr = vkQueuePresentKHR(gpu.queue(), &pi);
      }
      if (pr == VK_ERROR_OUT_OF_DATE_KHR) recreateSwapchain();
      else if (pr != VK_SUCCESS && pr != VK_SUBOPTIMAL_KHR) VK_CHECK(pr);
    };

    while (!gQuit) {
      if (!window.poll(onKey, [&](XEvent& e) { return (ui && ui->handleEvent(e)); })) {
        quitBecause("window closed (WM_DELETE_WINDOW)");
        break;
      }
      for (const std::string& line : control.poll()) onCommand(line);
      if (swapImagesGrew || ((window.width() != lastWinW || window.height() != lastWinH) && window.width() && window.height())) {
        swapImagesGrew = false;
        recreateSwapchain();
        continue;
      }

      if (!capture.acquire(frame, 6)) {   // short, so X events and the idle UI are served promptly
        idleFrame();
        continue;
      }
      const double now = monotonicMs();
      if (diffLog) fprintf(diffLog, "%llu %.3f %.2f\n", (unsigned long long)frame.seq, frame.diff, frame.tCapture);
      // 60->30 duplicates are removed in the capture thread (PhaseDedup), which sees every frame.
      realFps.tick(now);   // unique frames only
      if (lastRealTick > 0) {
        realIntervals.push_back((float)(now - lastRealTick));
        if (realIntervals.size() > 600) realIntervals.pop_front();
      }
      lastRealTick = now;
      if (ui) {   // hidden: an empty state, so no banner from an idle frame stays drawn
        std::vector<std::string> uiCommands;
        ui->build(ui->visible() ? uiState() : [&] { UiState h; h.fullscreen = window.fullscreen(); fillHud(h.hud); return h; }(), [&](const std::string& c) { uiCommands.push_back(c); });
        for (const std::string& c : uiCommands) onCommand(c);
      }
      decodeMs.add(frame.decodeMs);
      rangeDetector.add(frame);
      // Remote Play says its range in the bitstream; the setting is the dongle's (HDMI guesswork).
      const bool limitedRange = capture.remotePlay() ? frame.range != 1
                                                     : opt.range == 0 || (opt.range == -1 && rangeDetector.limited);
      rangeLimitedNow = limitedRange;

      // Retire the previous frame.
      VK_CHECK(vkWaitForFences(gpu.device(), 1, &fence, VK_TRUE, UINT64_MAX));
      if (inFlight) {
        const double done = monotonicMs();
        submitLatency.add(done - submitTime);
        std::vector<double> ts = ctx.readTimestampsMs(timestamps, kStamps);
        gpuMs.add(ts[3] - ts[0]);
        if (frameUsedNr) nrMs.add(ts[1] - ts[9]);
        if (frameUsedClean) cleanMs.add((ts[7] - ts[6]) + (ts[9] - ts[8]));
        if (nvChain) nvGapMs.add(ts[11] - ts[10]);   // B end -> C start: the effects + hand-offs
        if (rifeStatsPending) {
          rifeGapMs.add(ts[11] - ts[10]);
          if (const float ms = rife->lastMs(); ms > 0) rifeMs.add(ms);
          const uint32_t* rs = static_cast<const uint32_t*>(rifeStats.mapped);
          const double d = rs[1] ? rs[0] / 255.0 / rs[1] : 0.0;
          if (opt.rifeCut > 0 && d > opt.rifeCut) fgHeld++;
          if (getenv("MOONGLIDE_FG_LOG")) fprintf(stderr, "rife diff %.3f%s ms %.2f\n", d, opt.rifeCut > 0 && d > opt.rifeCut ? " held" : "", rife->lastMs());
          rifeStatsPending = false;
        }
        // Vulkan queue busy time (A + B + C segments), without the idle gaps between submits.
        vkBusyMs.add((ts[4] - ts[0]) + (ts[10] - ts[5]) + (ts[3] - ts[11]));
        if (frameUsedFlow) flowMs.add(ts[5] - ts[4]);
        if (flowCheck && frameUsedFlow) {
          const uint32_t* sums = static_cast<const uint32_t*>(checkBuffer.mapped);
          const double pixels = (double)workW * workH;
          flowErr.add(sums[0] / 16.0 / pixels);
          zeroErr.add(sums[1] / 16.0 / pixels);
        }
        if (nr && frameUsedNr) nr->afterFrame();
        if (nvMeasure && (measuredDn || measuredSr || measuredFg)) {
          const uint32_t* sum = static_cast<const uint32_t*>(sumsBuffer.mapped);
          const double workPx = (double)workW * workH, upPx = (double)nv->upW() * nv->upH();
          auto update = [&](float* cast, uint32_t outSlot, double outPx, auto inMean) {
            float d[3];
            for (int c = 0; c < 3; ++c) d[c] = (float)(sum[outSlot * 3 + c] / outPx - inMean(c));
            if (std::fabs(d[0]) > 6 || std::fabs(d[1]) > 6 || std::fabs(d[2]) > 6) return;   // cut / outlier
            for (int c = 0; c < 3; ++c) cast[c] += 0.05f * (d[c] - cast[c]);
          };
          if (measuredDn) update(castDn, 1, workPx, [&](int c) { return sum[0 * 3 + c] / workPx; });
          if (measuredSr) update(castSr, 3, upPx, [&](int c) { return sum[2 * 3 + c] / workPx; });
          if (measuredFg)
            update(castFg, 4, upPx, [&](int c) { return 0.5 * (sum[3 * 3 + c] + sum[5 * 3 + c]) / upPx; });
          for (int c = 0; c < 3; ++c) {
            fixDn.row[c][3] = -castDn[c] / 255.f;
            fixUp.row[c][3] = -castSr[c] / 255.f;
            fixMid.row[c][3] = -(castSr[c] + castFg[c]) / 255.f;
          }
        }
        if (nv) {
          const Nvvfx::Times nt = nv->lastTimes();
          if (nt.denoise > 0) nvDnMs.add(nt.denoise);
          if (nt.sr > 0) nvSrMs.add(nt.sr);
          if (nt.fg > 0) nvFgMs.add(nt.fg);
        }
        if (dumpPending && !dumpPrefix.empty()) {
          auto save = [&](const vk::Buffer& b, const char* suffix) {
            if (FILE* f = fopen((dumpPrefix + suffix).c_str(), "wb")) { fwrite(b.mapped, 1, workBytes, f); fclose(f); }
          };
          const std::string tag = "." + std::to_string(dumpSeq);
          save(dumpWork, (tag + ".work.rgba").c_str());
          if (dumpHasNr) save(dumpNr, (tag + ".nr.rgba").c_str());
          if (dumpHasMid) save(dumpMid, (tag + ".mid.rgba").c_str());
          if (FILE* f = fopen((dumpPrefix + tag + ".out.rgba").c_str(), "wb")) { fwrite(dumpOut.mapped, 1, outBytes, f); fclose(f); }
          fprintf(stderr, "dumped frame %llu\n", (unsigned long long)dumpSeq);
          dumpPending = false;
          if (dumpSeq == dumpSeqs.back()) { fprintf(stderr, "dumps done\n"); quitBecause("dumps done"); }
        }
        capture.release(heldSlot);
        inFlight = false;
      }
      heldSlot = frame.slot;

      uint32_t imageIndex = 0;
      VkResult acq = vkAcquireNextImageKHR(gpu.device(), gpu.swapchain(), UINT64_MAX, acquired, VK_NULL_HANDLE, &imageIndex);
      if (acq == VK_ERROR_OUT_OF_DATE_KHR) {
        fprintf(stderr, "swapchain out of date; recreating\n");
        recreateSwapchain();
        capture.release(heldSlot);
        heldSlot = -1;
        continue;
      }
      if (acq != VK_SUCCESS && acq != VK_SUBOPTIMAL_KHR) VK_CHECK(acq);
      // Own FG presents two images per real frame: the generated one, then the real one.
      const bool rifeFrame = fgOwn && fgRife && rife && !nvChain;
      const bool fgThisFrame = (fgOwn && (rifeFrame || (!fgRife && opt.flow && (flow || gpu.opticalFlow())))) || nvFg;
      // RIFE x3 / x4: two or three generated pictures (t = k / mult) per real frame.
      const int fgMore = rifeFrame ? std::clamp(opt.fgMult, 2, Rife::kMaxOut + 1) - 2 : 0;   // beyond the first
      const int fgParts = fgMore + 2;                                                        // pictures per real frame
      if (rifeFrame && fgMore > 0) {   // x3 / x4: the extra pictures' buffers, made when first asked for
        rife->ensureOut(fgMore + 1);
        for (int k = 0; k < fgMore; ++k) ensureBuf(fgMidMore[k], workBytes, "fg mid n");
      }
      // Spacing of the pictures, a fraction of the real period: even by default; fg-real-delay
      // shortens it so the real picture comes sooner. Generated picture k (0 = first) is shown
      // (fgParts - 1 - k) steps before the real one, so it is interpolated at t = 1 - that * step.
      const float fgStep = opt.fgRealDelay > 0.f ? std::min(opt.fgRealDelay, 1.0f / (float)fgParts) : 1.0f / (float)fgParts;
      const auto fgT = [&](int k) { return 1.0f - (float)(fgParts - 1 - k) * fgStep; };
      uint32_t midIndex = 0, moreIndex[Rife::kMaxOut - 1] = {};
      if (fgThisFrame) {
        VkResult acq2 = vkAcquireNextImageKHR(gpu.device(), gpu.swapchain(), UINT64_MAX, acquiredMid, VK_NULL_HANDLE, &midIndex);
        if (acq2 != VK_SUCCESS && acq2 != VK_SUBOPTIMAL_KHR) VK_CHECK(acq2);
      }
      for (int k = 0; k < fgMore; ++k) {
        VkResult acq3 = vkAcquireNextImageKHR(gpu.device(), gpu.swapchain(), UINT64_MAX, acquiredMore[k], VK_NULL_HANDLE, &moreIndex[k]);
        if (acq3 != VK_SUCCESS && acq3 != VK_SUBOPTIMAL_KHR) VK_CHECK(acq3);
      }
      if (!fgThisFrame) fgHavePrev = false;

      // Overlay text for this frame (host-visible, read by the present pass).
      const auto cs = capture.stats();
      std::vector<std::string> lines;   // bitmap overlay text (the stats card is the UI's); currently none
      uint32_t textCols = 0;
      if (lines.size() > kOverlayLines) lines.resize(kOverlayLines);
      for (auto& l : lines) textCols = std::max<uint32_t>(textCols, (uint32_t)l.size());
      textCols = std::min(textCols, kOverlayColumns);   // the text buffer's row stride
      overlayPack(lines, static_cast<uint32_t*>(textBuffer.mapped));

      ctx.resetDescriptorPool();
      VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

      // ---- A (compute queue): capture -> RGBA8 at the work size, luma for the flow engine.
      VK_CHECK(vkResetCommandBuffer(cmd, 0));
      VK_CHECK(vkBeginCommandBuffer(cmd, &begin));
      vkCmdResetQueryPool(cmd, timestamps, 0, kStamps);
      vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timestamps, 0);
      const vk::Buffer& slot = slots[frame.slot];
      if (frame.format == PixelFormat::MJPEG && !limitedRange) {
        VkBufferCopy region{0, 0, (VkDeviceSize)capW * capH * 4};
        vkCmdCopyBuffer(cmd, slot.buffer, colour.buffer, 1, &region);
      } else if (frame.format == PixelFormat::MJPEG) {
        struct { uint32_t count, expand; } push{capW * capH, 1u};
        gpu.dispatch(cmd, levelsPipe, {&slot, &colour}, &push, sizeof(push), (capW * capH + 255) / 256, 1);
      } else {
        struct { uint32_t w, h, format, fullRange; } push{capW, capH, (uint32_t)frame.format, limitedRange ? 0u : 1u};
        gpu.dispatch(cmd, yuvPipe, {&slot, &colour}, &push, sizeof(push), (capW + 15) / 16, (capH + 15) / 16);
      }
      fullBarrier(cmd);
      const bool deblockThisFrame = opt.clean && opt.deblockStrength > 0.f && frame.format == PixelFormat::MJPEG;
      if (deblockThisFrame || opt.deband > 0.f) ensureBuf(deblockTmp, (VkDeviceSize)capW * capH * 4, "deblock tmp");
      vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timestamps, 6);
      if (deblockThisFrame) {
        struct { uint32_t w, h, vertical; float strength; } dp{capW, capH, 0u, opt.deblockStrength};
        gpu.dispatch(cmd, deblockPipe, {&colour, &deblockTmp}, &dp, sizeof(dp), (capW + 15) / 16, (capH + 15) / 16);
        fullBarrier(cmd);
        dp.vertical = 1u;
        gpu.dispatch(cmd, deblockPipe, {&deblockTmp, &colour}, &dp, sizeof(dp), (capW + 15) / 16, (capH + 15) / 16);
        fullBarrier(cmd);
      }
      if (opt.deband > 0.f) {
        struct { uint32_t w, h; float threshold, radius; uint32_t frame; } bp{capW, capH, opt.deband / 255.f, 16.f, (uint32_t)frame.seq};
        gpu.dispatch(cmd, debandPipe, {&colour, &deblockTmp}, &bp, sizeof(bp), (capW + 15) / 16, (capH + 15) / 16);
        fullBarrier(cmd);
        VkBufferCopy back{0, 0, (VkDeviceSize)capW * capH * 4};
        vkCmdCopyBuffer(cmd, deblockTmp.buffer, colour.buffer, 1, &back);
        fullBarrier(cmd);
      }
      vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timestamps, 7);
      if (workScaled) {
        struct { uint32_t sw, sh, dw, dh; } push{capW, capH, workW, workH};
        gpu.dispatch(cmd, resamplePipe, {&colour, &workColour}, &push, sizeof(push), (workW + 15) / 16, (workH + 15) / 16);
        fullBarrier(cmd);
      }
      frameUsedNr = opt.nr && nr;
      const bool flowWanted = frameUsedNr || (fgThisFrame && !nvFg && !rifeFrame) || opt.clean;   // RIFE does not use the flow
      if (flowWanted) ensureFlow();
      const bool flowThisFrame = flowWanted && flow && opt.flow;
      if (!flowThisFrame && flow) flow->reset();
      const bool flowRuns = flowThisFrame && flow->havePrevious();
      if (flowThisFrame) flow->recordLuma(cmd, workSrc);
      if (nvDn) {
        VkBufferCopy region{0, 0, workBytes};
        vkCmdCopyBuffer(cmd, workSrc.buffer, nv->denoiseIn().buffer, 1, &region);
      }
      vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timestamps, 4);
      VK_CHECK(vkEndCommandBuffer(cmd));

      // Timeline values of this frame on the shared Vulkan/CUDA semaphore: +1 A done,
      // +2 denoise done, +3 B done (shown picture ready), +4 sr/fg done.
      nvBase += 4;
      VK_CHECK(vkResetFences(gpu.device(), 1, &fence));
      VkSubmitInfo submitA{VK_STRUCTURE_TYPE_SUBMIT_INFO};
      submitA.commandBufferCount = 1;
      submitA.pCommandBuffers = &cmd;
      VkSemaphore signalsA[2];
      uint64_t signalValuesA[2] = {0, 0};
      uint32_t signalCountA = 0;
      if (flowRuns) signalsA[signalCountA++] = lumaReady;
      VkTimelineSemaphoreSubmitInfo timelineA{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
      if (nvDn) {
        signalValuesA[signalCountA] = nvBase + 1;
        signalsA[signalCountA++] = nv->timeline();
        timelineA.signalSemaphoreValueCount = signalCountA;
        timelineA.pSignalSemaphoreValues = signalValuesA;
        submitA.pNext = &timelineA;
      }
      submitA.signalSemaphoreCount = signalCountA;
      submitA.pSignalSemaphores = signalsA;
      {
        std::lock_guard<std::mutex> q(queueMutex);
        VK_CHECK(vkQueueSubmit(gpu.queue(), 1, &submitA, VK_NULL_HANDLE));
      }
      if (nvDn) nv->runDenoise(nvBase + 1, nvBase + 2);
      if (flowRuns) {
        // ---- the flow engine's queue: current vs previous luma.
        VkCommandBuffer ofCmd = flow->ofCommands();
        VkPipelineStageFlags ofWait = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        VkSubmitInfo submitOf{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submitOf.waitSemaphoreCount = 1;
        submitOf.pWaitSemaphores = &lumaReady;
        submitOf.pWaitDstStageMask = &ofWait;
        submitOf.commandBufferCount = 1;
        submitOf.pCommandBuffers = &ofCmd;
        submitOf.signalSemaphoreCount = 1;
        submitOf.pSignalSemaphores = &flowReady;
        VK_CHECK(vkQueueSubmit(gpu.ofQueue(), 1, &submitOf, VK_NULL_HANDLE));
      }

      // ---- B (compute queue): motion, NR, present pass, swapchain copy.
      VK_CHECK(vkResetCommandBuffer(cmdB, 0));
      VK_CHECK(vkBeginCommandBuffer(cmdB, &begin));
      vkCmdWriteTimestamp(cmdB, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timestamps, 5);
      measuredDn = nvMeasure && nvDn;
      if (nvDn) {
        struct { uint32_t count, pad[3]; float row[3][4]; } cp{workW * workH, {0, 0, 0}, {}};
        memcpy(cp.row, fixDn.row, sizeof(cp.row));
        if (nvMeasure) {
          vkCmdFillBuffer(cmdB, sumsBuffer.buffer, 0, VK_WHOLE_SIZE, 0);
          fullBarrier(cmdB);
          channelSums(cmdB, nv->denoiseIn(), workW * workH, 0);
          channelSums(cmdB, nv->denoiseOut(), workW * workH, 1);
        }
        gpu.dispatch(cmdB, colourFixPipe, {&nv->denoiseOut(), &workSrc}, &cp, sizeof(cp), (workW * workH + 255) / 256, 1);
        fullBarrier(cmdB);
      }
      const vk::Buffer* motion = nullptr;
      if (flowRuns) {
        flow->setMedian((opt.fgFlags & 128u) != 0u);
        flow->recordMotion(cmdB);
        fullBarrier(cmdB);
        motion = &flow->motion();
      }
      if (flowCheck && flowRuns && havePrevColour) {
        struct { uint32_t w, h; } push{workW, workH};
        vkCmdFillBuffer(cmdB, checkBuffer.buffer, 0, VK_WHOLE_SIZE, 0);
        fullBarrier(cmdB);
        gpu.dispatch(cmdB, checkPipe, {&workSrc, &prevColour, motion, &checkBuffer}, &push, sizeof(push),
                     (workW + 15) / 16, (workH + 15) / 16);
        fullBarrier(cmdB);
      }
      vkCmdWriteTimestamp(cmdB, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timestamps, 8);
      if (opt.clean) {
        // The clean frame replaces the capture for everything downstream (NR, intensity mix,
        // FG, HUD zones); its own history is the previous clean frame.
        ensureBuf(cleanBuf[0], workBytes, "clean 0");
        ensureBuf(cleanBuf[1], workBytes, "clean 1");
        vk::Buffer& out = cleanBuf[cleanCur];
        if (motion && cleanHave && !resetHistory) {
          struct { uint32_t w, h; float strength; uint32_t flags; } tp{workW, workH, opt.cleanStrength,
                                                                      opt.cleanDebug ? 1u : 0u};
          gpu.dispatch(cmdB, tdnPipe, {&workSrc, &cleanBuf[cleanCur ^ 1], motion, &out}, &tp, sizeof(tp),
                       (workW + 15) / 16, (workH + 15) / 16);
        } else {
          VkBufferCopy region{0, 0, workBytes};
          vkCmdCopyBuffer(cmdB, workSrc.buffer, out.buffer, 1, &region);
        }
        fullBarrier(cmdB);
        VkBufferCopy region{0, 0, workBytes};
        vkCmdCopyBuffer(cmdB, out.buffer, workSrc.buffer, 1, &region);
        fullBarrier(cmdB);
        cleanCur ^= 1;
        cleanHave = true;
        frameUsedClean = true;
      } else {
        frameUsedClean = false;
        cleanHave = false;
      }
      vkCmdWriteTimestamp(cmdB, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timestamps, 9);
      if (frameUsedNr) {
        nr->record(cmdB, workSrc, motion, resetHistory);
        fullBarrier(cmdB);
      } else if (nr) {
        nr->invalidateHistory();
      }
      resetHistory = false;   // consumed by every history user above (clean, NR)
      vkCmdWriteTimestamp(cmdB, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timestamps, 1);
      if (!dumpPrefix.empty() && !dumpPending && std::find(dumpSeqs.begin(), dumpSeqs.end(), frame.seq) != dumpSeqs.end()) {
        dumpSeq = frame.seq;
        VkBufferCopy region{0, 0, workBytes};
        vkCmdCopyBuffer(cmdB, workSrc.buffer, dumpWork.buffer, 1, &region);
        dumpHasNr = frameUsedNr;
        if (frameUsedNr) vkCmdCopyBuffer(cmdB, nr->image().buffer, dumpNr.buffer, 1, &region);
        fullBarrier(cmdB);
        dumpPending = true;
      }

      packHudZones(opt.hudZones, viewRect().h, zoneBuffer.mapped);   // also with NR off: same HUD in A/B
      if (flowCheck) {
        VkBufferCopy region{0, 0, workBytes};
        vkCmdCopyBuffer(cmdB, workSrc.buffer, prevColour.buffer, 1, &region);
        havePrevColour = flowThisFrame;
      }
      // Waits of the submit that ends this frame (it signals the swapchain images + fence).
      VkSemaphore waits[8];
      VkPipelineStageFlags waitStages[8];
      uint64_t waitValues[8] = {};
      uint32_t waitCount = 0;
      bool waitTimeline = false;
      VkCommandBuffer cmdP = cmdB;   // where the present passes go
      if (nvChain) {
        // The shown picture (NR mixed over the clean frame by intensity) goes to the effects;
        // B ends here, C presents their output.
        const vk::Buffer& shownDst = nv->sr() ? nv->srIn() : nv->up(nvCur);
        struct { uint32_t count; float mixAmount; uint32_t origIsSrc; } mp{workW * workH, opt.intensity,
                                                                           frameUsedNr ? 0u : 1u};
        gpu.dispatch(cmdB, fgMixPipe, {frameUsedNr ? &nr->image() : &workSrc, &workSrc, &shownDst}, &mp, sizeof(mp),
                     (workW * workH + 255) / 256, 1);
        fullBarrier(cmdB);
        vkCmdWriteTimestamp(cmdB, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timestamps, 10);
        VK_CHECK(vkEndCommandBuffer(cmdB));
        VkSemaphore bWaits[2];
        VkPipelineStageFlags bStages[2] = {VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT};
        uint64_t bWaitValues[2] = {0, 0};
        uint32_t bWaitCount = 0;
        if (flowRuns) bWaits[bWaitCount++] = flowReady;
        if (nvDn) {
          bWaitValues[bWaitCount] = nvBase + 2;
          bWaits[bWaitCount++] = nv->timeline();
        }
        VkSemaphore bSignal = nv->timeline();
        uint64_t bSignalValue = nvBase + 3;
        VkTimelineSemaphoreSubmitInfo tl{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
        tl.waitSemaphoreValueCount = bWaitCount;
        tl.pWaitSemaphoreValues = bWaitValues;
        tl.signalSemaphoreValueCount = 1;
        tl.pSignalSemaphoreValues = &bSignalValue;
        VkSubmitInfo sb{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        sb.pNext = &tl;
        sb.waitSemaphoreCount = bWaitCount;
        sb.pWaitSemaphores = bWaits;
        sb.pWaitDstStageMask = bStages;
        sb.commandBufferCount = 1;
        sb.pCommandBuffers = &cmdB;
        sb.signalSemaphoreCount = 1;
        sb.pSignalSemaphores = &bSignal;
        {
          std::lock_guard<std::mutex> q(queueMutex);
          VK_CHECK(vkQueueSubmit(gpu.queue(), 1, &sb, VK_NULL_HANDLE));
        }
        if (nvHostWait) {
          // Enqueue the effects only once B is done: a CUDA stream parked on the semaphore
          // while B runs time-slices the GPU against it.
          const uint64_t v = nvBase + 3;
          VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
          VkSemaphore tsem = nv->timeline();
          wi.semaphoreCount = 1;
          wi.pSemaphores = &tsem;
          wi.pValues = &v;
          VK_CHECK(vkWaitSemaphores(gpu.device(), &wi, UINT64_MAX));
        }
          nv->runSrFg(nvBase + 3, nvBase + 4, nvCur, nvHavePrev);
        VK_CHECK(vkResetCommandBuffer(cmdC, 0));
        VK_CHECK(vkBeginCommandBuffer(cmdC, &begin));
        vkCmdWriteTimestamp(cmdC, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timestamps, 11);
        cmdP = cmdC;
        measuredSr = nvMeasure && nv->sr();
        measuredFg = nvMeasure && nv->fg() && nvHavePrev;
        if (measuredSr || measuredFg) {
          const uint32_t upPx = nv->upW() * nv->upH();
          if (!nvDn) {
            vkCmdFillBuffer(cmdC, sumsBuffer.buffer, 0, VK_WHOLE_SIZE, 0);
            fullBarrier(cmdC);
          }
          if (measuredSr) channelSums(cmdC, nv->srIn(), workW * workH, 2);
          channelSums(cmdC, nv->up(nvCur), upPx, 3);
          if (measuredFg) {
            channelSums(cmdC, nv->mid(), upPx, 4);
            channelSums(cmdC, nv->up(nvCur ^ 1), upPx, 5);
          }
          fullBarrier(cmdC);
        }
        waitValues[waitCount] = nvBase + 4;
        waits[waitCount] = nv->timeline();
        waitStages[waitCount++] = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        waitTimeline = true;
      } else if (rifeFrame) {
        // RIFE: B ends with the shown picture packed into the TensorRT input; the engine runs on
        // its CUDA stream between B and C; C unpacks (or holds the real frame on a cut) and presents.
        rifeBase += 2;
        const vk::Buffer& shown = fgShown[fgCur];
        struct { uint32_t count; float mixAmount; uint32_t origIsSrc; } mp{workW * workH, opt.intensity,
                                                                           frameUsedNr ? 0u : 1u};
        gpu.dispatch(cmdB, fgMixPipe, {frameUsedNr ? &nr->image() : &workSrc, &workSrc, &shown}, &mp, sizeof(mp),
                     (workW * workH + 255) / 256, 1);
        vkCmdFillBuffer(cmdB, rifeStats.buffer, 0, VK_WHOLE_SIZE, 0);
        fullBarrier(cmdB);
        struct { uint32_t w, h, pw, ph; } pp{workW, workH, rife->padW(), rife->padH()};
        gpu.dispatch(cmdB, rifePackPipe, {&shown, fgHavePrev ? &fgShown[fgCur ^ 1] : &shown, &rife->in(rifeCur), &rifeStats},
                     &pp, sizeof(pp), (rife->padW() + 15) / 16, (rife->padH() + 15) / 16);
        fullBarrier(cmdB);
        vkCmdWriteTimestamp(cmdB, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timestamps, 10);
        VK_CHECK(vkEndCommandBuffer(cmdB));
        VkSemaphore bWaits[2];
        VkPipelineStageFlags bStages[2] = {VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT};
        uint64_t bWaitValues[2] = {0, 0};
        uint32_t bWaitCount = 0;
        if (flowRuns) bWaits[bWaitCount++] = flowReady;
        if (nvDn) {
          bWaitValues[bWaitCount] = nvBase + 2;
          bWaits[bWaitCount++] = nv->timeline();
        }
        VkSemaphore bSignal = rife->timeline();
        uint64_t bSignalValue = rifeBase + 1;
        VkTimelineSemaphoreSubmitInfo tl{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
        tl.waitSemaphoreValueCount = bWaitCount;
        tl.pWaitSemaphoreValues = bWaitValues;
        tl.signalSemaphoreValueCount = 1;
        tl.pSignalSemaphoreValues = &bSignalValue;
        VkSubmitInfo sb{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        sb.pNext = &tl;
        sb.waitSemaphoreCount = bWaitCount;
        sb.pWaitSemaphores = bWaits;
        sb.pWaitDstStageMask = bStages;
        sb.commandBufferCount = 1;
        sb.pCommandBuffers = &cmdB;
        sb.signalSemaphoreCount = 1;
        sb.pSignalSemaphores = &bSignal;
        {
          std::lock_guard<std::mutex> q(queueMutex);
          VK_CHECK(vkQueueSubmit(gpu.queue(), 1, &sb, VK_NULL_HANDLE));
        }
        if (nvHostWait) {
          // As for the effects: a CUDA stream parked on the semaphore time-slices against B.
          VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
          VkSemaphore tsem = rife->timeline();
          wi.semaphoreCount = 1;
          wi.pSemaphores = &tsem;
          wi.pValues = &bSignalValue;
          VK_CHECK(vkWaitSemaphores(gpu.device(), &wi, UINT64_MAX));
        }
        {
          float tv[Rife::kMaxOut];
          for (int k = 0; k < fgParts - 1; ++k) tv[k] = fgT(k);
          rife->run(rifeBase + 1, rifeBase + 2, rifeCur, tv, fgParts - 1);
        }
        VK_CHECK(vkResetCommandBuffer(cmdC, 0));
        VK_CHECK(vkBeginCommandBuffer(cmdC, &begin));
        vkCmdWriteTimestamp(cmdC, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timestamps, 11);
        cmdP = cmdC;
        waitValues[waitCount] = rifeBase + 2;
        waits[waitCount] = rife->timeline();
        waitStages[waitCount++] = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        waitTimeline = true;
      } else {
        // Every timestamp is written each frame (the readback waits on all of them).
        vkCmdWriteTimestamp(cmdB, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timestamps, 10);
        vkCmdWriteTimestamp(cmdB, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timestamps, 11);
        if (flowRuns) { waits[waitCount] = flowReady; waitStages[waitCount++] = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT; }
        if (nvDn) {
          waitValues[waitCount] = nvBase + 2;
          waits[waitCount] = nv->timeline();
          waitStages[waitCount++] = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
          waitTimeline = true;
        }
      }

      // Upscale + present pass + swapchain copy of one picture. `mixed`: src already has the
      // intensity mix baked in (FG pictures), so EASU reads it alone and the mix is 1.
      // `fix`: colour correction for the effects' output; `preScaled`: src is already rect-sized.
      auto renderTo = [&](const vk::Buffer* src, bool mixed, uint32_t index, const ColourFix* fix = nullptr,
                          bool preScaled = false) {
        const ViewRect vr = viewRect();
        struct {
          uint32_t srcW, srcH, dstW, dstH, rectX, rectY, rectW, rectH;
          float mixAmount;
          uint32_t flags, scale, textCols, textLines;
          float splitX;
          uint32_t pad0, pad1;
          float fix[3][4];
        } push{workW, workH, ext.width, ext.height, vr.x, vr.y, vr.w, vr.h, mixed ? 1.f : opt.intensity,
               0u, opt.overlaySize > 0 ? (uint32_t)opt.overlaySize : std::max(1u, ext.height / 540u), textCols,
               (uint32_t)lines.size(), 0.5f, (uint32_t)(opt.overlayPos & 3), 0u, {}};
        if (fix) {
          memcpy(push.fix, fix->row, sizeof(push.fix));
          push.flags |= 64u;
        }
        if (gpu.formatIsBgra()) push.flags |= 2u;
        if (split && frameUsedNr) push.flags |= 4u;
        if (!frameUsedNr && !mixed) push.flags |= 16u;
        if (opt.hudShow) push.flags |= 32u;
        const vk::Buffer* procPtr = src;
        if (preScaled) {
          push.flags |= 8u;
        } else if (opt.fsr) {
          ensureBuf(easuBuffer, (VkDeviceSize)screenW * screenH * 4, "fsr easu");
          ensureBuf(rcasBuffer, (VkDeviceSize)screenW * screenH * 4, "fsr rcas");
          struct { uint32_t sw, sh, dw, dh; float mixAmount; uint32_t origIsSrc; } ep{
              workW, workH, vr.w, vr.h, opt.intensity, (mixed || !frameUsedNr) ? 1u : 0u};
          gpu.dispatch(cmdP, easuPipe, {procPtr, &workSrc, &easuBuffer}, &ep, sizeof(ep), (vr.w + 15) / 16,
                       (vr.h + 15) / 16);
          fullBarrier(cmdP);
          struct { uint32_t w, h; float sharpness; } rp{vr.w, vr.h, std::exp2(-opt.sharpenStops)};
          gpu.dispatch(cmdP, rcasPipe, {&easuBuffer, &rcasBuffer}, &rp, sizeof(rp), (vr.w + 15) / 16, (vr.h + 15) / 16);
          fullBarrier(cmdP);
          push.flags |= 8u;
          procPtr = &rcasBuffer;
        }
        gpu.dispatch(cmdP, presentPipe, {procPtr, &workSrc, &fontBuffer, &textBuffer, &outBuffer, &zoneBuffer}, &push,
                     sizeof(push), (ext.width + 15) / 16, (ext.height + 15) / 16);
        VkImage image = gpu.images()[index].image;
        fullBarrier(cmdP);
        imageBarrier(cmdP, image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
                     VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {ext.width, ext.height, 1};
        vkCmdCopyBufferToImage(cmdP, outBuffer.buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        // The settings overlay goes on last, over the finished picture (never into NR / FG).
        if (!(ui && ui->record(cmdP, image, ext.width, ext.height)))
          imageBarrier(cmdP, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                       VK_ACCESS_TRANSFER_WRITE_BIT, 0, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
        fullBarrier(cmdP);   // outBuffer / easu / rcas are reused by the next picture
      };

      if (nvChain) {
        // The effects' pictures: sr output of this frame and (fg) the one halfway back.
        const vk::Buffer& real = nv->up(nvCur);
        const bool pre = nv->sr();
        if (fgThisFrame) {
          renderTo(nvHavePrev ? &nv->mid() : &real, true, midIndex, nvHavePrev ? &fixMid : &fixUp, pre);
          vkCmdWriteTimestamp(cmdP, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timestamps, 2);
          renderTo(&real, true, imageIndex, &fixUp, pre);
        } else {
          renderTo(&real, true, imageIndex, &fixUp, pre);
          vkCmdWriteTimestamp(cmdP, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timestamps, 2);
        }
        nvCur ^= 1;
        nvHavePrev = true;
      } else if (rifeFrame) {
        const vk::Buffer& shown = fgShown[fgCur];
        struct { uint32_t w, h, pw, ph; float cut; uint32_t hold; float still, outlier, t; } up{
            workW, workH, rife->padW(), rife->padH(), opt.rifeCut, fgHavePrev && !frame.corrupt && !prevCorrupt ? 0u : 1u, opt.rifeStill, opt.rifeOutlier,
            fgT(0)};
        const vk::Buffer* prevShown = fgHavePrev ? &fgShown[fgCur ^ 1] : &shown;
        gpu.dispatch(cmdP, rifeUnpackPipe, {&rife->out(0), &shown, &rifeStats, &fgMid, prevShown}, &up, sizeof(up),
                     (workW + 15) / 16, (workH + 15) / 16);
        for (int k = 0; k < fgMore; ++k) {
          up.t = fgT(k + 1);
          gpu.dispatch(cmdP, rifeUnpackPipe, {&rife->out(k + 1), &shown, &rifeStats, &fgMidMore[k], prevShown}, &up,
                       sizeof(up), (workW + 15) / 16, (workH + 15) / 16);
        }
        fullBarrier(cmdP);
        renderTo(&fgMid, true, midIndex);
        vkCmdWriteTimestamp(cmdP, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timestamps, 2);
        for (int k = 0; k < fgMore; ++k) renderTo(&fgMidMore[k], true, moreIndex[k]);
        renderTo(&shown, true, imageIndex);
        rifeStatsPending = fgHavePrev;
        if (fgHavePrev && (frame.corrupt || prevCorrupt)) fgHeld++;
        fgCur ^= 1;
        rifeCur ^= 1;
        fgHavePrev = true;
        prevCorrupt = frame.corrupt;
      } else if (fgThisFrame) {
        // The shown picture of this real frame, then the one halfway back to the previous.
        const vk::Buffer& shown = fgShown[fgCur];
        struct { uint32_t count; float mixAmount; uint32_t origIsSrc; } mp{workW * workH, opt.intensity,
                                                                           frameUsedNr ? 0u : 1u};
        gpu.dispatch(cmdB, fgMixPipe, {frameUsedNr ? &nr->image() : &workSrc, &workSrc, &shown}, &mp, sizeof(mp),
                     (workW * workH + 255) / 256, 1);
        fullBarrier(cmdB);
        if (fgHavePrev && motion) {
          struct { uint32_t w, h; float t; uint32_t flags; float hold; } ip{workW, workH, 0.5f, 16u, opt.fgHold};
          vkCmdFillBuffer(cmdB, fgStats.buffer, 0, VK_WHOLE_SIZE, 0);
          fullBarrier(cmdB);
          gpu.dispatch(cmdB, fgInterpPipe, {&fgShown[fgCur ^ 1], &shown, motion, &fgMid, &fgStats}, &ip, sizeof(ip),
                       (workW + 15) / 16, (workH + 15) / 16);
          fullBarrier(cmdB);
          ip.flags = (opt.fgDebug ? 1u : 0u) | opt.fgFlags;
          gpu.dispatch(cmdB, fgInterpPipe, {&fgShown[fgCur ^ 1], &shown, motion, &fgMid, &fgStats}, &ip, sizeof(ip),
                       (workW + 15) / 16, (workH + 15) / 16);
          fgStatsPending = true;
        } else {
          VkBufferCopy region{0, 0, workBytes};
          vkCmdCopyBuffer(cmdB, shown.buffer, fgMid.buffer, 1, &region);
        }
        fullBarrier(cmdB);
        renderTo(&fgMid, true, midIndex);
        vkCmdWriteTimestamp(cmdB, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timestamps, 2);
        renderTo(&shown, true, imageIndex);
        fgCur ^= 1;
        fgHavePrev = true;
      } else {
        renderTo(frameUsedNr ? &nr->image() : &workSrc, false, imageIndex);
        vkCmdWriteTimestamp(cmdB, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timestamps, 2);
      }
      if (dumpPending && frame.seq == dumpSeq) {
        VkBufferCopy oc{0, 0, outBytes};
        vkCmdCopyBuffer(cmdP, outBuffer.buffer, dumpOut.buffer, 1, &oc);
        dumpHasMid = !nvChain && fgThisFrame && fgHavePrev;
        if (dumpHasMid) {
          VkBufferCopy mc{0, 0, workBytes};
          vkCmdCopyBuffer(cmdP, fgMid.buffer, dumpMid.buffer, 1, &mc);
        }
        fullBarrier(cmdP);
      }
      vkCmdWriteTimestamp(cmdP, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timestamps, 3);
      VK_CHECK(vkEndCommandBuffer(cmdP));

      waits[waitCount] = acquired; waitStages[waitCount++] = VK_PIPELINE_STAGE_TRANSFER_BIT;
      if (fgThisFrame) { waits[waitCount] = acquiredMid; waitStages[waitCount++] = VK_PIPELINE_STAGE_TRANSFER_BIT; }
      for (int k = 0; k < fgMore; ++k) { waits[waitCount] = acquiredMore[k]; waitStages[waitCount++] = VK_PIPELINE_STAGE_TRANSFER_BIT; }
      VkSemaphore signals[Rife::kMaxOut + 1] = {gpu.images()[imageIndex].renderDone, gpu.images()[midIndex].renderDone};
      for (int k = 0; k < fgMore; ++k) signals[2 + k] = gpu.images()[moreIndex[k]].renderDone;
      VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
      VkTimelineSemaphoreSubmitInfo finalTimeline{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
      if (waitTimeline) {
        finalTimeline.waitSemaphoreValueCount = waitCount;
        finalTimeline.pWaitSemaphoreValues = waitValues;
        submit.pNext = &finalTimeline;
      }
      submit.waitSemaphoreCount = waitCount;
      submit.pWaitSemaphores = waits;
      submit.pWaitDstStageMask = waitStages;
      submit.commandBufferCount = 1;
      submit.pCommandBuffers = &cmdP;
      submit.signalSemaphoreCount = fgThisFrame ? 2 + fgMore : 1;
      submit.pSignalSemaphores = signals;
      {
        std::lock_guard<std::mutex> q(queueMutex);
        VK_CHECK(vkQueueSubmit(gpu.queue(), 1, &submit, fence));
      }
      if (flowThisFrame) flow->advance();
      frameUsedFlow = flowRuns;
      submitTime = monotonicMs();
      inFlight = true;

      VkSwapchainKHR sc = gpu.swapchain();
      if (fgThisFrame) {
        // Generated picture as soon as the GPU is done, the real one half a real-frame period
        // later (CPU-timed; the 240 Hz dummy latches within 4.2 ms of either).
        VK_CHECK(vkWaitForFences(gpu.device(), 1, &fence, VK_TRUE, UINT64_MAX));
        if (fgStatsPending) {
          const uint32_t* st = static_cast<const uint32_t*>(fgStats.mapped);
          const double f = st[1] ? (double)st[0] / st[1] : 0.0;
          fgFail += 0.1 * (f - fgFail);
          if (opt.fgHold > 0 && f > opt.fgHold) fgHeld++;
          if (getenv("MOONGLIDE_FG_LOG")) fprintf(stderr, "fg seq %llu fail %.3f%s\n", (unsigned long long)frame.seq, f,
                                              opt.fgHold > 0 && f > opt.fgHold ? " held" : "");
          fgStatsPending = false;
        }
        const double rate = realFps.rate(monotonicMs());
        const double period = 1000.0 / (rate > 20.0 && rate < 70.0 ? rate : 30.0);
        const double half = period * (rifeFrame ? fgStep : 1.0f / fgParts);   // spacing of the pictures
        VkPresentInfoKHR mid{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        mid.waitSemaphoreCount = 1;
        mid.pWaitSemaphores = &gpu.images()[midIndex].renderDone;
        mid.swapchainCount = 1;
        mid.pSwapchains = &sc;
        mid.pImageIndices = &midIndex;
        VkPresentIdKHR midPid{VK_STRUCTURE_TYPE_PRESENT_ID_KHR};
        const uint64_t midId = ++presentId;
        if (waiter) {
          midPid.swapchainCount = 1;
          midPid.pPresentIds = &midId;
          mid.pNext = &midPid;
        }
        delayed.flush();   // the previous real picture first
        VkResult mr;
        {
          std::lock_guard<std::mutex> q(queueMutex);
          mr = vkQueuePresentKHR(gpu.queue(), &mid);
        }
        if (gPresentLog) fprintf(gPresentLog, "%.3f mid %llu\n", monotonicMs(), (unsigned long long)frame.seq);
        if (mr != VK_SUCCESS && mr != VK_SUBOPTIMAL_KHR && mr != VK_ERROR_OUT_OF_DATE_KHR) VK_CHECK(mr);
        const double tMid = monotonicMs();
        presentFps.tick(tMid);
        generated++;
        // x3: the second generated picture a third of a period later; then the real one (half a
        // period after the first at x2, two thirds at x3), from the presenter thread.
        for (int k = 0; k < fgMore; ++k) {
          delayed.post(sc, moreIndex[k], gpu.images()[moreIndex[k]].renderDone, ++presentId, waiter != nullptr,
                       tMid + (k + 1) * half);
          presentFps.tick(tMid);   // a count: the rate meter wants monotonic times
          generated++;
        }
        const uint64_t realId = ++presentId;
        delayed.post(sc, imageIndex, gpu.images()[imageIndex].renderDone, realId, waiter != nullptr,
                     tMid + (fgMore + 1) * half, fgMore == 0);
        const VkResult dr = delayed.lastResult();
        if (dr != VK_SUCCESS && dr != VK_SUBOPTIMAL_KHR && dr != VK_ERROR_OUT_OF_DATE_KHR) VK_CHECK(dr);
        if (waiter) waiter->submitted(realId, frame.tCapture, frame.tDequeue);
      }
      if (!fgThisFrame) {
      delayed.flush();
      VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
      present.waitSemaphoreCount = 1;
      present.pWaitSemaphores = &gpu.images()[imageIndex].renderDone;
      present.swapchainCount = 1;
      present.pSwapchains = &sc;
      present.pImageIndices = &imageIndex;
      VkPresentIdKHR pid{VK_STRUCTURE_TYPE_PRESENT_ID_KHR};
      const uint64_t thisId = ++presentId;
      if (waiter) {
        pid.swapchainCount = 1;
        pid.pPresentIds = &thisId;
        present.pNext = &pid;
      }
      VkResult pr;
      {
        std::lock_guard<std::mutex> q(queueMutex);
        pr = vkQueuePresentKHR(gpu.queue(), &present);
      }
      if (pr != VK_SUCCESS && pr != VK_SUBOPTIMAL_KHR && pr != VK_ERROR_OUT_OF_DATE_KHR) VK_CHECK(pr);
      if (waiter) waiter->submitted(thisId, frame.tCapture, frame.tDequeue);
      }
      lastPresent = monotonicMs();
      presentFps.tick(lastPresent);
      presented++;

      if (waiter)
        for (auto& [fromCapture, fromDequeue] : waiter->take()) {
          e2eCapture.add(fromCapture);
          e2eDequeue.add(fromDequeue);
        }
      // Remote Play: chiaki's sound waits for the picture (arrival -> on screen).
      if (capture.remotePlay() && e2eCapture.size() && presented % 30 == 0)
        capture.setAudioDelayMs((uint32_t)std::lround(e2eCapture.percentile(0.5)));

      if (lastPresent - lastLog >= 1000.0) {
        lastLog = lastPresent;
        {
          // Dongle quality auto-fallback: more than 2 corrupt frames in 5 s at q > 85 steps down
          // (92 -> 90 -> 85); the link can't carry it with this content.
          static std::deque<uint64_t> corrupt;
          const auto cst = capture.stats();
          corrupt.push_back(cst.decodeErrors);
          if (corrupt.size() > 6) corrupt.pop_front();
          const int q = opt.capture.jpegQuality;
          if (opt.dongleAuto && capture.jpegQualitySupported() && q > 85 && corrupt.size() == 6 && corrupt.back() - corrupt.front() > 2) {
            const int down = q > 90 ? 90 : 85;
            fprintf(stderr, "capture: %llu corrupt frames in 5 s at dongle q%d; down to q%d\n",
                    (unsigned long long)(corrupt.back() - corrupt.front()), q, down);
            opt.capture.jpegQuality = down;
            capture.setJpegQuality(down);
            corrupt.clear();
          }
        }
        saveSettings();   // settings changed by the overlay, keys or the fifo
        const moonglide::SysStats gmem = moonglide::sysProbe().get();
        std::string status = fmt(
            "{\"nr\":%s,\"intensity\":%.2f,\"real_fps\":%.2f,\"present_fps\":%.2f,\"dup\":%llu,\"drop\":%llu,"
            "\"e2e_ms\":%.2f,\"e2e_p90_ms\":%.2f,\"dq_ms\":%.2f,\"gpu_ms\":%.2f,\"nr_ms\":%.2f,\"decode_ms\":%.2f,"
            "\"flow\":%s,\"flow_ms\":%.2f,\"reproj_err_flow\":%.3f,\"reproj_err_zero\":%.3f,\"lsfg\":%s,"
            "\"range\":\"%s\",\"range_auto\":%s,\"dedup\":%s,\"dedup_locked\":%s,\"dedup_phase\":%d,"
            "\"dedup_ratio\":%.2f,\"hud_zones\":%zu,\"fg\":\"%s\",\"generated\":%llu,\"fg_fail\":%.3f,\"fg_held\":%llu,\"corrupt_frames\":%llu,\"clean\":%s,\"clean_ms\":%.2f,"
            "\"rife_ms\":%.2f,\"rife_gap_ms\":%.2f,\"nvvfx\":\"%s\",\"nv_denoise_ms\":%.2f,\"nv_sr_ms\":%.2f,\"nv_fg_ms\":%.2f,\"nv_gap_ms\":%.2f,\"vk_busy_ms\":%.2f,\"nv_cast\":\"dn %+.2f/%+.2f/%+.2f sr %+.2f/%+.2f/%+.2f fg %+.2f/%+.2f/%+.2f\",\"gpu_util\":%d,\"gpu_temp_c\":%d,\"gpu_clock_mhz\":%d,\"gpu_power_w\":%d,\"vram_app_mb\":%ld,\"vram_free_mb\":%ld,\"vram_used_mb\":%ld,\"vram_total_mb\":%ld,\"cpu_pct\":%.1f,\"app_cpu_pct\":%.1f,\"ram_used_mb\":%ld,\"ram_total_mb\":%ld,\"app_rss_mb\":%ld}",
            opt.nr ? "true" : "false", opt.intensity, realFps.rate(lastPresent), presentFps.rate(lastPresent),
            (unsigned long long)cs.duplicates, (unsigned long long)cs.dropped, e2eCapture.percentile(0.5),
            e2eCapture.percentile(0.9), e2eDequeue.percentile(0.5), gpuMs.mean(), nrMs.mean(), decodeMs.mean(),
            opt.flow && flow ? "true" : "false", flowMs.mean(), flowErr.mean(), zeroErr.mean(), lsfg ? "true" : "false",
            limitedRange ? "limited" : "full", opt.range == -1 ? "true" : "false", cs.dedupActive ? "true" : "false",
            cs.dedupLocked ? "true" : "false", cs.dedupPhase, cs.dedupRatio, opt.hudZones.size(),
            nvFg ? "nvvfx" : fgOwn ? (fgRife ? "rife" : "own") : lsfg ? "lsfg" : "off", (unsigned long long)generated, fgFail, (unsigned long long)fgHeld, (unsigned long long)cs.decodeErrors, opt.clean ? "true" : "false", cleanMs.mean(),
            rifeMs.mean(), rifeGapMs.mean(),
            !nv ? "off" : fmt("%s%s%s", nv->denoise() ? "dn" : "", nv->sr() ? "+sr" : "", nv->fg() ? "+fg" : "").c_str(),
            nvDnMs.mean(), nvSrMs.mean(), nvFgMs.mean(), nvGapMs.mean(), vkBusyMs.mean(),
            castDn[0], castDn[1], castDn[2], castSr[0], castSr[1], castSr[2], castFg[0], castFg[1], castFg[2],
            gmem.gpuUtil, gmem.gpuTempC, gmem.gpuClockMHz, gmem.gpuPowerW, gmem.vramAppMB, gmem.vramFreeMB, gmem.vramUsedMB,
            gmem.vramTotalMB, gmem.cpuPct, gmem.appCpuPct, gmem.ramUsedMB, gmem.ramTotalMB, gmem.appRssMB);
        if (capture.remotePlay()) {
          const auto rs = capture.remotePlayStats();
          status.pop_back();
          std::string state = rs.state;
          for (char& ch : state) if (ch == '"' || ch == '\\' || (unsigned char)ch < 0x20) ch = ' ';
          status += fmt(",\"rp_state\":\"%s\",\"rp_kbps\":%.0f,\"rp_frames\":%llu,\"rp_lost\":%llu,\"rp_corrupt\":%llu,"
                        "\"rp_idr\":%llu,\"rp_audio_held_ms\":%u,\"rp_audio_out_ms\":%u,\"rp_decoder\":\"%s\","
                        "\"rp_rtt_ms\":%.1f,\"rp_input_ms\":%.1f,\"rp_input_total_ms\":%.1f,\"rp_input_samples\":%d,\"rp_input_method\":\"%s\"}",
                        state.c_str(), rs.kbps, (unsigned long long)rs.frames, (unsigned long long)rs.lost,
                        (unsigned long long)rs.corrupt, (unsigned long long)rs.idrRequests, rs.audioHeldMs, rs.audioOutMs,
                        rs.decoder.c_str(), rs.rttMs, rs.inputLagMs,
                        rs.inputLagSamples ? rs.inputLagMs + (e2eCapture.size() ? e2eCapture.percentile(0.5) : 0.0) : 0.0,
                        rs.inputLagSamples, rs.inputLagSamples ? rs.inputLagMethod.c_str() : "");
        }
        fprintf(stderr, "%s\n", status.c_str());
        std::ofstream(opt.statusPath) << status << "\n";
      }
      if (opt.frames && presented >= opt.frames) break;
    }

    saveSettings();
    delayed.flush();
    vkDeviceWaitIdle(gpu.device());
    ui.reset();
    waiter.reset();
    capture.stop();
    const std::string err = capture.error();
    if (!err.empty())
      fprintf(stderr, "capture: %llu corrupt frames skipped (first: %s)\n",
              (unsigned long long)capture.stats().decodeErrors, err.c_str());
    nr.reset();
    flow.reset();
    for (VkSemaphore sem : {acquired, acquiredMid, lumaReady, flowReady}) vkDestroySemaphore(gpu.device(), sem, nullptr);
    for (VkSemaphore sem : acquiredMore) vkDestroySemaphore(gpu.device(), sem, nullptr);
    vkDestroyFence(gpu.device(), fence, nullptr);
    vkDestroyFence(gpu.device(), idleFence, nullptr);
    vkDestroySemaphore(gpu.device(), idleAcquired, nullptr);
    vkDestroyCommandPool(gpu.device(), pool, nullptr);
    vkDestroyQueryPool(gpu.device(), timestamps, nullptr);
    for (auto& s : slots) ctx.destroyBuffer(s);
    for (vk::Buffer* b : {&colour, &workColour, &outBuffer, &fontBuffer, &textBuffer, &zoneBuffer, &easuBuffer, &rcasBuffer,
                          &prevColour, &checkBuffer})
      if (b->buffer) ctx.destroyBuffer(*b);
  } catch (const std::exception& e) {
    fprintf(stderr, "fatal: %s\n", e.what());
    fflush(stderr);
    // The desktop app never just disappears: say what happened and offer what can be done.
    if (!opt.uiOpen && !getenv("MOONGLIDE_ERROR_SCREEN")) _exit(1);
    if (errorScreen(e.what()) != 1) {
      _exit(1);
    }
    // Again, as started .
    gRestart = true;
    gRestartArgs.clear();
    for (int i = 1; i < argc; ++i)
      if (strcmp(argv[i], "--no-config") && strcmp(argv[i], "--config") && (i == 1 || strcmp(argv[i - 1], "--config")))
        gRestartArgs.push_back(argv[i]);
  }
  if (gQuitSignal) fprintf(stderr, "quit: signal %d\n", (int)gQuitSignal);
  if (gRestart) {
    // Back to the environment this process started with, then the same binary again.
    if (getenv("MOONGLIDE_NVVFX_EXEC")) {
      const char* ld = getenv("MOONGLIDE_ORIG_LD_LIBRARY_PATH");
      if (ld && *ld) setenv("LD_LIBRARY_PATH", ld, 1);
      else unsetenv("LD_LIBRARY_PATH");
      unsetenv("MOONGLIDE_NVVFX_EXEC");
      unsetenv("MOONGLIDE_ORIG_LD_LIBRARY_PATH");
    }
    if (hadDisableLsfg) setenv("DISABLE_LSFG", origDisableLsfg.c_str(), 1);
    else unsetenv("DISABLE_LSFG");
    std::vector<char*> args{argv[0]};
    if (!configOn) args.push_back((char*)"--no-config");
    else if (configGiven) { args.push_back((char*)"--config"); args.push_back(configPath.data()); }
    for (std::string& a : gRestartArgs) args.push_back(a.data());
    // A custom control fifo / status file (as given on the command line) must survive the restart.
    for (int i = 1; i + 1 < argc; ++i)
      if (!strcmp(argv[i], "--control") || !strcmp(argv[i], "--status")) { args.push_back(argv[i]); args.push_back(argv[i + 1]); }
    args.push_back(nullptr);
    fprintf(stderr, "restart:");
    for (char* a : args) if (a) fprintf(stderr, " %s", a);
    fprintf(stderr, "\n");
    fflush(stderr);
    // The file on disk, not /proc/self/exe: after a rebuild that is the old, deleted binary.
    char self[PATH_MAX] = {};
    std::string exe = readlink("/proc/self/exe", self, sizeof(self) - 1) > 0 ? self : "/proc/self/exe";
    if (exe.size() > 10 && exe.compare(exe.size() - 10, 10, " (deleted)") == 0) exe.resize(exe.size() - 10);
    execv(exe.c_str(), args.data());
    fprintf(stderr, "restart of %s failed: %s; trying the running binary\n", exe.c_str(), strerror(errno));
    execv("/proc/self/exe", args.data());
    fprintf(stderr, "restart failed: %s\n", strerror(errno));
  }
  // _exit, not return: an implicit layer (lsfg-vk) registers an atexit handler and is then
  // unloaded with the instance, so a normal exit jumps into unmapped code.
  fflush(stdout);
  fflush(stderr);
  _exit(0);
}