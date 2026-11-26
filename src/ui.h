// Settings overlay (Dear ImGui), drawn into the swapchain image after the picture is copied in,
// so NR, own FG and the HUD mask never see it. F1 or Tab opens and closes it; while it is open
// it takes the X11 pointer and keyboard (Moonlight forwards both to the X session).
//
// The panel reads one snapshot of the settings and only emits control-channel command lines
// ("intensity 0.7", "hud add rect:...", "set present mailbox", "restart"), so the overlay, the
// fifo and the keys share one code path.
#pragma once
#include <X11/Xlib.h>
#include <volk.h>

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "config.h"
#include "hud_zones.h"
#include "nvml.h"

class Gpu;
class RpSetup;
class CaptureEnum;

struct UiStats {
  float realFps = 0, presentFps = 0, low1Fps = 0;
  float e2e = 0, e2e90 = 0, dq = 0;
  float gpu = 0, nr = 0, flow = 0, clean = 0, decode = 0;
  float nvDenoise = 0, nvSr = 0, nvFg = 0, vkBusy = 0;
  float fgFail = 0;
  float budget = 33.3f;         // ms per real frame at the capture rate
  unsigned long long fgHeld = 0, generated = 0, dropped = 0, duplicates = 0;
  std::vector<float> frameMs;   // recent present intervals, oldest first
};

// A message over the picture (or the whole window while nothing runs): what is wrong and
// what to do about it. Buttons emit ctl command lines like the rest of the panel.
struct UiAction {
  std::string label, command;
};
struct UiBanner {
  std::string title, detail;
  int level = 0;   // 0 info, 1 warning, 2 error
  std::vector<UiAction> actions;
};

// The in-picture stats card (HUD): drawn by Ui with ImGui draw lists over every presented picture.
struct UiHud {
  bool on = false, compact = false;
  int pos = 0, size = 0;         // corner 0 tl 1 tr 2 bl 3 br; size 0 auto, 1..6
  unsigned pc = 0;               // PC rows: 1 GPU, 2 CPU, 4 RAM, 8 VRAM
  float fps = 0, real = 0, budget = 33.3f;
  std::string fg;                // "RIFE x2", "NVIDIA", "" = off
  float e2e = -1, gpu = 0, nr = 0, flow = 0, decode = 0;
  unsigned long long dropped = 0, dup = 0;
  std::string denoise, upscale, capSize, outSize;   // readable pipeline summary
  bool rp = false;
  float rtt = 0, kbps = 0, inputLag = -1;
  int inputSamples = 0;          // samples behind inputLag
  std::string inputMethod;       // "camera" or "button"
  unsigned long long lost = 0, corrupt = 0;
  std::string codec, state;
  moonglide::SysStats sys;
  std::vector<float> frameMs;    // recent real-frame intervals
};

struct UiState {
  UiHud hud;
  Settings current;     // every setting as it is now (live values + pending restart-only ones)
  Settings running;     // the restart-only settings this process started with
  std::vector<std::string> restartKeys;   // keys that only take effect after a restart
  std::vector<std::string> presets;
  std::vector<HudZone> zones;
  uint32_t rectX = 0, rectY = 0, rectW = 1, rectH = 1;   // the picture on screen
  std::string fgActive;          // "lsfg", "own", "nvvfx", "off"
  bool ownAvailable = false, lsfgLoaded = false;
  bool nrCapable = true;   // the GPU can run the neural denoise (RTX 40/50)
  bool rifeMissing = false;      // TensorRT / the RIFE model are not installed
  int rifeInstall = 0;           // RifeInstaller::State: 0 idle, 1 running, 2 done, 3 failed
  std::string rifeInstallLine;
  std::string nvvfxStages;       // "" when the backend isn't running
  std::string configPath;
  bool configWritable = true;
  bool split = false;
  uint32_t srcW = 0, srcH = 0, outW = 0, outH = 0;   // capture and output size
  float srcFps = 0;
  moonglide::SysStats sys;                           // PC stats (-1: not available)
  // Top bar chips and the Signal panel.
  std::string sourceChip, formatChip;
  std::string dongleJpeg;        // MS2109 encoder quality status ("" when not an MS2109)
  std::string remotePlay;        // PS5 Remote Play status ("" when the source is the dongle)
  std::vector<std::pair<std::string, std::string>> signal;
  UiStats stats;
  std::vector<UiBanner> banners;
  bool noPicture = false;        // nothing to show yet / any more (desktop app: opens the panel)
  bool fatal = false;            // nothing else to show: the start-up failed (error screen)
  bool fullscreen = false;
};

class Ui {
 public:
  Ui(Gpu& gpu, Display* display, Window window, uint32_t imageCount);
  ~Ui();

  bool visible() const { return visible_; }
  bool playMode() const { return desktop_ && fullscreen_; }   // desktop app, fullscreen: picture only
  // The desktop app (--ui): settings reachable without F1 (auto-open without a picture, a Settings button on mouse move).
  void setDesktop(bool on) { desktop_ = on; }
  void setVisible(bool v);
  // An X event; true when the overlay took it (toggle key, or anything while it is open).
  bool handleEvent(XEvent& e);
  // Builds this frame's panel (while visible; while hidden, only the banners). `command` gets
  // ctl command lines.
  void build(const UiState& s, const std::function<void(const std::string&)>& command);
  // Draws the panel onto a swapchain image that is in TRANSFER_DST_OPTIMAL and leaves it in
  // PRESENT_SRC. Returns false (nothing recorded) while hidden.
  bool record(VkCommandBuffer cmd, VkImage image, uint32_t width, uint32_t height);
  // Where the picture goes while the app UI is open (left of the sidebar, between the top bar
  // and the bottom panels), in swapchain pixels.
  void pictureArea(uint32_t width, uint32_t height, float& x, float& y, float& w, float& h) const;
  // Image views belong to the swapchain images: drop them when it is recreated (device idle).
  void swapchainRecreated();

 private:
  using Cmd = std::function<void(const std::string&)>;
  void panel(const UiState& s, const Cmd& cmd);
  void topBar(const UiState& s, const Cmd& cmd);
  void sidebar(const UiState& s, const Cmd& cmd);
  void bottomPanels(const UiState& s);
  void statusLine(const UiState& s, const Cmd& cmd);
  void hudOverlay(const UiState& s, const Cmd& cmd);
  void statsCard(const UiHud& h);
  void bannerWindows(const UiState& s, const Cmd& cmd, bool interactive);
  void tabSource(const UiState& s, const Cmd& cmd);        // Input
  void tabProcessing(const UiState& s, const Cmd& cmd);
  void tabDisplay(const UiState& s, const Cmd& cmd);
  void tabNvidia(const UiState& s, const Cmd& cmd);
  void settingsBar(float alpha, const Cmd& cmd);
  bool doubleClick(int x, int y);
  // The Static zones card is open on the Processing tab: drawing / outlining zones, no fullscreen double-click.
  bool editingZones() const {
    auto it = expanded_.find("zones");
    return tab_ == 1 && visible_ && it != expanded_.end() && it->second;
  }
  void setCursorShown(bool on);
  VkImageView viewOf(VkImage image);
  void loadFontsAndStyle();

  Gpu& gpu_;
  Display* display_;
  Window window_;
  bool visible_ = false;
  bool built_ = false;
  double lastTime_ = 0;
  float scale_ = 1;
  VkDescriptorPool pool_ = VK_NULL_HANDLE;
  std::map<VkImage, VkImageView> views_;
  // HUD drawing on the picture.
  int drawKind_ = -1;   // -1 none, 0 rect, 1 ellipse
  bool dragging_ = false;
  float dragX0_ = 0, dragY0_ = 0;
  int tab_ = 0;
  // VRAM cost measured when a feature was switched on (MiB), and the pending measurements.
  std::map<std::string, float> liveCost_;
  struct CostWatch { float before; double t; };
  std::map<std::string, CostWatch> costWatch_;
  std::map<std::string, bool> featPrev_;
  float vramPrev_ = -1;
  bool desktop_ = false, autoOpened_ = false, suppressAuto_ = false, autoChange_ = false, firstBuild_ = true;
  bool noPicture_ = false, barShown_ = false, cursorShown_ = true;
  double lastMove_ = 0, lastClickT_ = 0;
  int lastClickX_ = 0, lastClickY_ = 0;
  bool fullscreen_ = false, wasVisible_ = false;
  std::vector<std::string> pendingCmds_;
  float barX_ = 0, barY_ = 0, barW_ = 0, barH_ = 0;
  std::map<std::string, bool> expanded_;   // card id -> params open
  struct Fonts { void *regular = nullptr, *semibold = nullptr, *small = nullptr, *big = nullptr, *title = nullptr; } fonts_;
  char presetName_[64] = "";
  char nvvfxPath_[256] = "";
  // Source tab: background helpers and the pairing form.
  std::unique_ptr<RpSetup> rpSetup_;
  std::unique_ptr<CaptureEnum> capEnum_;
  void sourceRemotePlay(const UiState& s, const Cmd& cmd);
  void tabSignal(const UiState& s, const Cmd& cmd);
  void sourceCapture(const UiState& s, const Cmd& cmd);
  char pairHost_[64] = "";
  char psnUrl_[1024] = "";
  char pin_[16] = "";
  std::string loginLink_;
  bool pairPs4_ = false, pairShow_ = false;
  bool nvvfxPathInit_ = false;
  // First-run "Connect your PlayStation" wizard (drawn over the panel; the manual cards stay as they are).
  void connectWizard(const UiState& s, const Cmd& cmd);
  void requestClipboard();
  bool wizOpen_ = false, wizChecked_ = false, wizPopup_ = false, wizScanStarted_ = false, wizPairing_ = false;
  bool wizPs4_ = false, wizClipNew_ = false;
  int wizStep_ = 0;                  // 0 find console, 1 PSN sign-in, 2 PIN, 3 done
  double wizClipReq_ = 0;
  std::string wizClip_, wizClipTried_;
  char wizHost_[64] = "", wizUrl_[1024] = "", wizPin_[16] = "";
};
