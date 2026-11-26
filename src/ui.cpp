#include "ui.h"
#include "i18n.h"

#include <X11/Xatom.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "backends/imgui_impl_vulkan.h"
#include "gpu.h"
#include "imgui.h"
#include "input_setup.h"

namespace {

double nowSeconds() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

void check(VkResult r) {
  if (r != VK_SUCCESS) fprintf(stderr, "ui: vulkan error %d\n", (int)r);
}

ImGuiKey keyOf(KeySym k) {
  switch (k) {
    case XK_Tab: return ImGuiKey_Tab;
    case XK_Left: return ImGuiKey_LeftArrow;
    case XK_Right: return ImGuiKey_RightArrow;
    case XK_Up: return ImGuiKey_UpArrow;
    case XK_Down: return ImGuiKey_DownArrow;
    case XK_Page_Up: return ImGuiKey_PageUp;
    case XK_Page_Down: return ImGuiKey_PageDown;
    case XK_Home: return ImGuiKey_Home;
    case XK_End: return ImGuiKey_End;
    case XK_Insert: return ImGuiKey_Insert;
    case XK_Delete: return ImGuiKey_Delete;
    case XK_BackSpace: return ImGuiKey_Backspace;
    case XK_space: return ImGuiKey_Space;
    case XK_Return: case XK_KP_Enter: return ImGuiKey_Enter;
    case XK_Escape: return ImGuiKey_Escape;
    case XK_Control_L: return ImGuiKey_LeftCtrl;
    case XK_Control_R: return ImGuiKey_RightCtrl;
    case XK_Shift_L: return ImGuiKey_LeftShift;
    case XK_Shift_R: return ImGuiKey_RightShift;
    case XK_Alt_L: return ImGuiKey_LeftAlt;
    case XK_Alt_R: return ImGuiKey_RightAlt;
    case XK_a: return ImGuiKey_A;
    case XK_c: return ImGuiKey_C;
    case XK_v: return ImGuiKey_V;
    case XK_x: return ImGuiKey_X;
    case XK_y: return ImGuiKey_Y;
    case XK_z: return ImGuiKey_Z;
    default: return ImGuiKey_None;
  }
}

std::string get(const Settings& s, const char* key, const char* fallback = "") {
  auto it = s.find(key);
  return it == s.end() ? fallback : it->second;
}
float getf(const Settings& s, const char* key, float fallback) {
  auto it = s.find(key);
  return it == s.end() || it->second.empty() ? fallback : (float)atof(it->second.c_str());
}
bool getb(const Settings& s, const char* key, bool fallback) {
  auto it = s.find(key);
  return it == s.end() ? fallback : it->second == "true";
}

std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
std::string fmt(const char* f, ...) {
  char buf[512];
  va_list ap;
  va_start(ap, f);
  vsnprintf(buf, sizeof(buf), f, ap);
  va_end(ap);
  return buf;
}

}  // namespace

Ui::Ui(Gpu& gpu, Display* display, Window window, uint32_t imageCount) : gpu_(gpu), display_(display), window_(window) {
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  io.IniFilename = nullptr;   // our settings live in config.toml
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
  io.BackendPlatformName = "moonglide-x11";
  // From the screen, not the window: the window starts smaller and goes fullscreen later.
  scale_ = std::max(1.f, (float)DisplayHeight(display, DefaultScreen(display)) / 1080.f);
  loadFontsAndStyle();

  VkDescriptorPoolSize sizes[] = {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 16}};
  VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  pi.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  pi.maxSets = 16;
  pi.poolSizeCount = 1;
  pi.pPoolSizes = sizes;
  check(vkCreateDescriptorPool(gpu.device(), &pi, nullptr, &pool_));

  static VkFormat format;
  format = gpu.format();
  ImGui_ImplVulkan_InitInfo info{};
  info.Instance = gpu.instance();
  info.PhysicalDevice = gpu.physical();
  info.Device = gpu.device();
  info.QueueFamily = gpu.queueFamily();
  info.Queue = gpu.queue();
  info.DescriptorPool = pool_;
  info.MinImageCount = 2;
  info.ImageCount = std::max(4u, imageCount * 2);   // two pictures per frame with FG
  info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
  info.UseDynamicRendering = true;
  info.PipelineRenderingCreateInfo = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR};
  info.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
  info.PipelineRenderingCreateInfo.pColorAttachmentFormats = &format;
  info.CheckVkResultFn = check;
  if (!ImGui_ImplVulkan_Init(&info)) throw std::runtime_error("ImGui Vulkan init failed");
  // Upload the font now, before other threads use the queue.
  ImGui_ImplVulkan_CreateFontsTexture();

  // The window needs pointer events and key releases too.
  XWindowAttributes wa;
  XGetWindowAttributes(display_, window_, &wa);
  XSelectInput(display_, window_,
               wa.your_event_mask | KeyReleaseMask | ButtonPressMask | ButtonReleaseMask | PointerMotionMask);
  lastTime_ = nowSeconds();
}

Ui::~Ui() {
  vkDeviceWaitIdle(gpu_.device());
  swapchainRecreated();
  ImGui_ImplVulkan_Shutdown();
  ImGui::DestroyContext();
  if (pool_) vkDestroyDescriptorPool(gpu_.device(), pool_, nullptr);
}

void Ui::setCursorShown(bool on) {
  if (on) {
    XUndefineCursor(display_, window_);
  } else {
    static char empty[8] = {};
    Pixmap blank = XCreateBitmapFromData(display_, window_, empty, 8, 8);
    XColor black{};
    Cursor c = XCreatePixmapCursor(display_, blank, blank, &black, &black, 0, 0);
    XDefineCursor(display_, window_, c);
    XFreeCursor(display_, c);
    XFreePixmap(display_, blank);
  }
  XFlush(display_);
  cursorShown_ = on;
}

void Ui::setVisible(bool v) {
  if (!autoChange_) {   // the user (or the launcher): an auto-opened panel is theirs now
    if (v) autoOpened_ = false;
    else suppressAuto_ = true;
  }
  visible_ = v;
  barShown_ = false;
  // The real X cursor (hardware plane, moves at the mouse rate) while the menu is open; the
  // ImGui-drawn one only moved when a frame was presented (15 fps idle, the stream rate otherwise).
  ImGui::GetIO().MouseDrawCursor = false;
  setCursorShown(v);
  if (!v) {
    drawKind_ = -1;
    dragging_ = false;
    built_ = false;
  }
}

bool Ui::handleEvent(XEvent& e) {
  ImGuiIO& io = ImGui::GetIO();
  if (e.type == SelectionNotify && e.xselection.requestor == window_) {
    // The clipboard text we asked for (wizard): only kept when it is plain text and not huge.
    if (e.xselection.property != None) {
      Atom type = 0;
      int fmtBits = 0;
      unsigned long n = 0, rest = 0;
      unsigned char* data = nullptr;
      if (XGetWindowProperty(display_, window_, e.xselection.property, 0, 1024, True, AnyPropertyType, &type, &fmtBits, &n, &rest,
                             &data) == Success && data) {
        if (fmtBits == 8 && rest == 0) wizClip_.assign((const char*)data, n);
        else wizClip_.clear();
        wizClipNew_ = true;
        XFree(data);
      }
    }
    return true;
  }
  if (e.type == KeyPress || e.type == KeyRelease) {
    const bool down = e.type == KeyPress;
    char text[32] = {};
    KeySym sym = 0;
    const int n = XLookupString(&e.xkey, text, sizeof(text) - 1, &sym, nullptr);
    if (sym == XK_F11) return false;   // fullscreen: the app's key handler, UI open or not
    if (down && (sym == XK_Return || sym == XK_KP_Enter) && (e.xkey.state & Mod1Mask)) {
      pendingCmds_.push_back("fullscreen toggle");
      return true;
    }
    if (down && (sym == XK_F1 || sym == XK_Tab)) {
      setVisible(!visible_);
      return true;
    }
    if (!visible_) return false;
    if (down && sym == XK_Escape && !io.WantTextInput && drawKind_ < 0) {
      setVisible(false);
      return true;
    }
    if (down && sym == XK_Escape) { drawKind_ = -1; dragging_ = false; }
    io.AddKeyEvent(ImGuiMod_Ctrl, (e.xkey.state & ControlMask) != 0);
    io.AddKeyEvent(ImGuiMod_Shift, (e.xkey.state & ShiftMask) != 0);
    io.AddKeyEvent(ImGuiMod_Alt, (e.xkey.state & Mod1Mask) != 0);
    const ImGuiKey k = keyOf(XLookupKeysym(&e.xkey, 0));
    if (k != ImGuiKey_None) io.AddKeyEvent(k, down);
    if (down && n > 0 && !(e.xkey.state & ControlMask))
      for (int i = 0; i < n; ++i)
        if ((unsigned char)text[i] >= 32 && text[i] != 127) io.AddInputCharacter((unsigned char)text[i]);
    return true;
  }
  if (!visible_) {
    if (!desktop_) return false;
    // Desktop app: a mouse move shows the small Settings bar for a few seconds.
    if (e.type == MotionNotify) {
      io.AddMousePosEvent((float)e.xmotion.x, (float)e.xmotion.y);
      lastMove_ = nowSeconds();
    } else if (e.type == ButtonPress && e.xbutton.button == 1 &&
               !(barShown_ && e.xbutton.x >= barX_ && e.xbutton.x <= barX_ + barW_ && e.xbutton.y >= barY_ && e.xbutton.y <= barY_ + barH_)) {
      if (doubleClick(e.xbutton.x, e.xbutton.y)) { pendingCmds_.push_back("fullscreen toggle"); return true; }
    } else if ((e.type == ButtonPress || e.type == ButtonRelease) && barShown_ && e.xbutton.button == 1 &&
               e.xbutton.x >= barX_ && e.xbutton.x <= barX_ + barW_ && e.xbutton.y >= barY_ && e.xbutton.y <= barY_ + barH_) {
      io.AddMousePosEvent((float)e.xbutton.x, (float)e.xbutton.y);
      io.AddMouseButtonEvent(0, e.type == ButtonPress);
      return true;
    }
    return false;
  }
  if (desktop_ && e.type == ButtonPress && e.xbutton.button == 1) {
    float ax, ay, aw, ah;
    pictureArea(gpu_.extent().width, gpu_.extent().height, ax, ay, aw, ah);
    if (e.xbutton.x >= ax && e.xbutton.x <= ax + aw && e.xbutton.y >= ay && e.xbutton.y <= ay + ah) {
      if (!editingZones() && doubleClick(e.xbutton.x, e.xbutton.y)) { pendingCmds_.push_back("fullscreen toggle"); return true; }   // not while editing static zones
      // A click on the picture closes the panel.
      if (!noPicture_ && !editingZones() && drawKind_ < 0) {
        setVisible(false);
        return true;
      }
    }
  }
  switch (e.type) {
    case MotionNotify:
      io.AddMousePosEvent((float)e.xmotion.x, (float)e.xmotion.y);
      return true;
    case ButtonPress:
    case ButtonRelease: {
      const bool down = e.type == ButtonPress;
      io.AddMousePosEvent((float)e.xbutton.x, (float)e.xbutton.y);
      const unsigned b = e.xbutton.button;
      if (b >= 1 && b <= 3) io.AddMouseButtonEvent(b == 1 ? 0 : b == 3 ? 1 : 2, down);
      else if (down && (b == 4 || b == 5)) io.AddMouseWheelEvent(0, b == 4 ? 1.f : -1.f);
      else if (down && (b == 6 || b == 7)) io.AddMouseWheelEvent(b == 6 ? 1.f : -1.f, 0);
      return true;
    }
    default:
      return false;
  }
}

bool Ui::doubleClick(int x, int y) {
  const double t = nowSeconds();
  const bool dbl = t - lastClickT_ < 0.35 && std::abs(x - lastClickX_) < 8 && std::abs(y - lastClickY_) < 8;
  lastClickT_ = dbl ? 0 : t;
  lastClickX_ = x;
  lastClickY_ = y;
  return dbl;
}

void Ui::build(const UiState& s, const std::function<void(const std::string&)>& command) {
  built_ = false;
  const double now0 = nowSeconds();
  for (const std::string& c : pendingCmds_) command(c);
  pendingCmds_.clear();
  if (desktop_ && s.fullscreen != fullscreen_) {
    // Fullscreen is play mode: the picture fills the screen, no panel; leaving brings the layout back.
    fullscreen_ = s.fullscreen;
    autoChange_ = true;
    if (fullscreen_) {
      wasVisible_ = visible_;
      if (visible_) setVisible(false);
      autoOpened_ = false;
    } else if (wasVisible_) {
      setVisible(true);
    }
    autoChange_ = false;
  }
  if (desktop_) {
    noPicture_ = s.noPicture;
    if (firstBuild_) {
      firstBuild_ = false;
      if (visible_ && s.noPicture) { autoOpened_ = true; tab_ = 0; }
    }
    if (!s.noPicture) suppressAuto_ = false;
    if (s.noPicture && !visible_ && !suppressAuto_ && !s.fatal && !fullscreen_) {
      autoChange_ = true;
      setVisible(true);
      autoChange_ = false;
      autoOpened_ = true;
      tab_ = 0;
    } else if (!s.noPicture && visible_ && autoOpened_) {
      autoChange_ = true;
      setVisible(false);
      autoChange_ = false;
      autoOpened_ = false;
    }
  }
  const double idleFor = now0 - lastMove_;
  const bool wantBar = desktop_ && !visible_ && lastMove_ > 0 && idleFor < 3.0;
  if (desktop_ && !visible_ && wantBar != cursorShown_) setCursorShown(wantBar);
  barShown_ = false;
  const bool hudNow = s.hud.on && !visible_ && !s.noPicture && !s.fatal;
  if (!visible_ && s.banners.empty() && !wantBar && !hudNow) return;
  ImGuiIO& io = ImGui::GetIO();
  io.DisplaySize = ImVec2((float)gpu_.extent().width, (float)gpu_.extent().height);
  const double t = nowSeconds();
  io.DeltaTime = (float)std::max(1e-4, t - lastTime_);
  lastTime_ = t;
  ImGui_ImplVulkan_NewFrame();
  ImGui::NewFrame();
  if (!visible_) {
    bannerWindows(s, command, false);   // hidden: just say what is wrong
    if (wantBar) settingsBar(idleFor < 2.5 ? 1.f : (float)((3.0 - idleFor) / 0.5), command);
  } else {
    panel(s, command);
  }
  if (hudNow) statsCard(s.hud);
  ImGui::Render();
  built_ = true;
}

VkImageView Ui::viewOf(VkImage image) {
  auto it = views_.find(image);
  if (it != views_.end()) return it->second;
  VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  vi.image = image;
  vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
  vi.format = gpu_.format();
  vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  VkImageView view = VK_NULL_HANDLE;
  check(vkCreateImageView(gpu_.device(), &vi, nullptr, &view));
  views_[image] = view;
  return view;
}

void Ui::swapchainRecreated() {
  for (auto& [image, view] : views_) vkDestroyImageView(gpu_.device(), view, nullptr);
  views_.clear();
}

bool Ui::record(VkCommandBuffer cmd, VkImage image, uint32_t width, uint32_t height) {
  if (!built_) return false;
  ImDrawData* dd = ImGui::GetDrawData();
  if (!dd) return false;
  VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.image = image;
  b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  b.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  b.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0,
                       nullptr, 0, nullptr, 1, &b);
  VkRenderingAttachmentInfo att{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
  att.imageView = viewOf(image);
  att.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  att.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
  att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
  ri.renderArea = {{0, 0}, {width, height}};
  ri.layerCount = 1;
  ri.colorAttachmentCount = 1;
  ri.pColorAttachments = &att;
  vkCmdBeginRendering(cmd, &ri);
  ImGui_ImplVulkan_RenderDrawData(dd, cmd);
  vkCmdEndRendering(cmd);
  b.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  b.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
  b.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  b.dstAccessMask = 0;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0,
                       0, nullptr, 0, nullptr, 1, &b);
  return true;
}

// ---------------------------------------------------------------------------------------------
// The app layout: top bar, the picture on the left, a sidebar of cards on the right, three
// panels under the picture (signal, GPU stage timings, frame rate and pacing) and a status line.
// Sizes are in 1080p units times scale_.

namespace {

constexpr float kPi = 3.14159265f;
constexpr float kTop = 52, kSide = 452, kBottom = 236, kStatus = 30, kGap = 10;

// Palette (MoonGlide design tokens).
const ImU32 cBg = IM_COL32(11, 20, 23, 255);
const ImU32 cPanel = IM_COL32(16, 28, 32, 255);
const ImU32 cCard = IM_COL32(21, 37, 42, 255);
const ImU32 cCardHi = IM_COL32(27, 47, 53, 255);
const ImU32 cBorder = IM_COL32(38, 59, 65, 255);
const ImU32 cPicBg = IM_COL32(7, 13, 15, 255);           // picture background
const ImU32 cPearl = IM_COL32(237, 230, 211, 255);       // moon / primary button / big numbers
const ImU32 cPearlHi = IM_COL32(255, 249, 232, 255);
const ImU32 cAccent = IM_COL32(95, 208, 196, 255);       // teal: active / data
const ImU32 cAccentHi = IM_COL32(131, 224, 213, 255);    // hovered / pressed accent
const ImU32 cAccentSoft = IM_COL32(22, 54, 58, 255);
const ImU32 cAccentText = IM_COL32(95, 208, 196, 255);   // accent-coloured text on dark
const ImU32 cInk = IM_COL32(11, 20, 23, 255);            // text on the accent
const ImU32 cAlt = IM_COL32(127, 168, 224, 255);         // second series (CPU bars)
const ImU32 cText = IM_COL32(230, 237, 236, 255);
const ImU32 cDim = IM_COL32(143, 163, 166, 255);
const ImU32 cTrack = IM_COL32(30, 48, 54, 255);
const ImU32 cGood = IM_COL32(91, 208, 138, 255);
const ImU32 cWarn = IM_COL32(240, 185, 74, 255);
const ImU32 cBad = IM_COL32(242, 109, 109, 255);

ImVec4 v4(ImU32 c) { return ImGui::ColorConvertU32ToFloat4(c); }

// Mix of two colours (t = share of b), opaque.
ImU32 mix(ImU32 a, ImU32 b, float t) {
  auto ch = [&](int sh) { return (int)(((a >> sh) & 255) * (1 - t) + ((b >> sh) & 255) * t); };
  return IM_COL32(ch(IM_COL32_R_SHIFT), ch(IM_COL32_G_SHIFT), ch(IM_COL32_B_SHIFT), 255);
}

// Pearl primary button colours for the scope of the object.
struct PrimaryButton {
  PrimaryButton() {
    ImGui::PushStyleColor(ImGuiCol_Button, v4(cPearl));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, v4(cPearlHi));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, v4(cPearlHi));
    ImGui::PushStyleColor(ImGuiCol_Text, v4(cInk));
  }
  ~PrimaryButton() { ImGui::PopStyleColor(4); }
};

// The MoonGlide mark: a pearl crescent with two teal echoes, on a dark rounded tile. Same geometry as
// assets/moonglide.svg (256 units); `size` is the tile's side in pixels.
void drawMoonMark(ImDrawList* dl, ImVec2 p, float size) {
  const float k = size / 256.f;
  auto X = [&](float x, float off) { return p.x + (128.f + 1.15f * (x + off + 8.f - 128.f)) * k; };
  auto Y = [&](float y) { return p.y + (128.f + 1.15f * (y - 128.f)) * k; };
  dl->AddRectFilled(p, ImVec2(p.x + size, p.y + size), cBg, size * 56.f / 256.f);
  // Outer circle (112,128) r64 from P1 (angle -85.6 deg) the long way to P2 (-330.5 deg); inner
  // circle (142.31,111.83) r54 from P2 (62 deg) to P1 (242 deg).
  auto crescent = [&](float off, ImU32 col, float grow) {
    const int N = 28;
    ImVec2 o[N + 1], in[N + 1];
    for (int i = 0; i <= N; ++i) {
      const float t = (float)i / N;
      const float ao = (-85.6f + (-330.5f + 85.6f) * t) * kPi / 180.f;
      const float ai = (242.f + (62.f - 242.f) * t) * kPi / 180.f;
      o[i] = ImVec2(X(112.f + (64.f + grow) * std::cos(ao), off), Y(128.f + (64.f + grow) * std::sin(ao)));
      in[i] = ImVec2(X(142.31f + (54.f - grow) * std::cos(ai), off), Y(111.83f + (54.f - grow) * std::sin(ai)));
    }
    for (int i = 0; i < N; ++i) {
      const ImVec2 q[4] = {o[i], o[i + 1], in[i + 1], in[i]};
      dl->AddConvexPolyFilled(q, 4, col);
    }
  };
  const ImU32 teal = cAccent;
  crescent(-6.f, teal, 0.f);
  crescent(12.f, cBg, 5.f);
  crescent(12.f, teal, 0.f);
  crescent(30.f, cBg, 5.f);
  crescent(30.f, cPearl, 0.f);
}

ImFont* F(void* f) { return f ? static_cast<ImFont*>(f) : ImGui::GetFont(); }

// Pill switch; true when clicked.
bool toggleSwitch(const char* id, bool on, float sc) {
  const ImVec2 size(42 * sc, 24 * sc);
  const ImVec2 p = ImGui::GetCursorScreenPos();
  const bool clicked = ImGui::InvisibleButton(id, size);
  const bool hover = ImGui::IsItemHovered();
  ImDrawList* dl = ImGui::GetWindowDrawList();
  const float r = size.y * 0.5f;
  dl->AddRectFilled(p, ImVec2(p.x + size.x, p.y + size.y), on ? cAccent : hover ? cCardHi : cTrack, r);
  const float kx = on ? p.x + size.x - r : p.x + r;
  dl->AddCircleFilled(ImVec2(kx, p.y + r), r - 3 * sc, on ? cInk : cDim, 24);
  return clicked;
}

// Rounded label; a `border` without `bg` draws it hollow (inactive stages).
void chip(const std::string& text, ImU32 bg, ImU32 fg, float sc, ImU32 border = 0) {
  const ImVec2 ts = ImGui::CalcTextSize(text.c_str());
  const ImVec2 pad(10 * sc, 4 * sc);
  const ImVec2 p = ImGui::GetCursorScreenPos();
  const ImVec2 q(p.x + ts.x + 2 * pad.x, p.y + ts.y + 2 * pad.y);
  ImDrawList* dl = ImGui::GetWindowDrawList();
  if (bg) dl->AddRectFilled(p, q, bg, 6 * sc);
  if (border) dl->AddRect(p, q, border, 6 * sc, 0, 1.2f * sc);
  dl->AddText(ImVec2(p.x + pad.x, p.y + pad.y), fg, text.c_str());
  ImGui::Dummy(ImVec2(q.x - p.x, q.y - p.y));
}

// Row of joined buttons; the clicked index or -1.
int segmented(const char* id, const std::vector<std::string>& labels, int current, float sc) {
  ImGui::PushID(id);
  const float w = ImGui::GetContentRegionAvail().x, h = 32 * sc;
  const ImVec2 p = ImGui::GetCursorScreenPos();
  ImDrawList* dl = ImGui::GetWindowDrawList();
  dl->AddRectFilled(p, ImVec2(p.x + w, p.y + h), cBg, 8 * sc);
  dl->AddRect(p, ImVec2(p.x + w, p.y + h), cBorder, 7 * sc);
  int clicked = -1;
  const float cw = w / (float)labels.size();
  for (size_t i = 0; i < labels.size(); ++i) {
    const ImVec2 a(p.x + cw * i, p.y), b(p.x + cw * (i + 1), p.y + h);
    ImGui::SetCursorScreenPos(a);
    ImGui::PushID((int)i);
    if (ImGui::InvisibleButton("seg", ImVec2(cw, h))) clicked = (int)i;
    const bool hover = ImGui::IsItemHovered();
    ImGui::PopID();
    const bool on = (int)i == current;
    if (on) {
      dl->AddRectFilled(ImVec2(a.x + 3 * sc, a.y + 3 * sc), ImVec2(b.x - 3 * sc, b.y - 3 * sc), cAccentSoft, 6 * sc);
      dl->AddRect(ImVec2(a.x + 3 * sc, a.y + 3 * sc), ImVec2(b.x - 3 * sc, b.y - 3 * sc), cAccent, 6 * sc);
    }
    else if (hover) dl->AddRectFilled(ImVec2(a.x + 3 * sc, a.y + 3 * sc), ImVec2(b.x - 3 * sc, b.y - 3 * sc), cCardHi, 5 * sc);
    const ImVec2 ts = ImGui::CalcTextSize(labels[i].c_str());
    dl->AddText(ImVec2(a.x + (cw - ts.x) / 2, a.y + (h - ts.y) / 2), on ? cAccent : cText,
                labels[i].c_str());
  }
  ImGui::SetCursorScreenPos(ImVec2(p.x, p.y + h));
  ImGui::Dummy(ImVec2(w, 4 * sc));
  ImGui::PopID();
  return clicked;
}

// Label on the left, value on the right, slider under them.
bool sliderRow(const char* label, float* v, float lo, float hi, const char* format, const char* hint = nullptr) {
  ImGui::PushStyleColor(ImGuiCol_Text, v4(cDim));
  ImGui::TextUnformatted(label);
  ImGui::PopStyleColor();
  char val[32];
  snprintf(val, sizeof(val), format, *v);
  ImGui::SameLine(ImGui::GetContentRegionAvail().x + ImGui::GetCursorStartPos().x - ImGui::CalcTextSize(val).x);
  ImGui::TextUnformatted(val);
  ImGui::SetNextItemWidth(-FLT_MIN);
  ImGui::PushID(label);
  const bool ch = ImGui::SliderFloat("##s", v, lo, hi, "");
  ImGui::PopID();
  if (hint && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", hint);
  return ch;
}

// A labelled switch inside a card.
bool switchRow(const char* label, bool on, float sc, const char* hint = nullptr) {
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted(label);
  if (hint && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", hint);
  ImGui::SameLine(ImGui::GetContentRegionAvail().x + ImGui::GetCursorStartPos().x - 42 * sc);
  return toggleSwitch(label, on, sc);
}

void sectionLabel(const char* text) {
  ImGui::Spacing();
  ImGui::PushStyleColor(ImGuiCol_Text, v4(cDim));
  ImGui::TextUnformatted(text);
  ImGui::PopStyleColor();
}

// Ring gauge centred in a w x h cell.
void ring(ImDrawList* dl, ImVec2 c, float r, float frac, ImU32 col, const char* value, const char* label, float sc,
          ImFont* valueFont, ImFont* labelFont) {
  const float th = 7 * sc;
  dl->PathArcTo(c, r, 0, 2 * kPi, 64);
  dl->PathStroke(cTrack, 0, th);
  frac = std::clamp(frac, 0.f, 1.f);
  if (frac > 0.001f) {
    const float a0 = -kPi / 2;
    dl->PathArcTo(c, r, a0, a0 + frac * 2 * kPi, 64);
    dl->PathStroke(col, 0, th);
  }
  ImVec2 ts = valueFont->CalcTextSizeA(valueFont->FontSize, FLT_MAX, 0, value);
  dl->AddText(valueFont, valueFont->FontSize, ImVec2(c.x - ts.x / 2, c.y - ts.y / 2), cText, value);
  ts = labelFont->CalcTextSizeA(labelFont->FontSize, FLT_MAX, 0, label);
  dl->AddText(labelFont, labelFont->FontSize, ImVec2(c.x - ts.x / 2, c.y + r + 8 * sc), cDim, label);
}

}  // namespace

void Ui::loadFontsAndStyle() {
  ImGuiIO& io = ImGui::GetIO();
  const float sc = scale_;
  auto pick = [](std::initializer_list<const char*> paths) -> const char* {
    for (const char* p : paths)
      if (access(p, R_OK) == 0) return p;
    return nullptr;
  };
  // Inter when all three static weights are installed; otherwise Noto Sans, then DejaVu.
  const char* reg = nullptr; const char* semi = nullptr; const char* bold = nullptr;
  for (const char* dir : {"/usr/share/fonts/truetype/inter/", "/usr/share/fonts/opentype/inter/", "/usr/share/fonts/inter/",
                          "/usr/share/fonts/truetype/inter-fonts/"}) {
    static std::string r, sb, b;
    r = std::string(dir) + "Inter-Regular.ttf"; sb = std::string(dir) + "Inter-SemiBold.ttf"; b = std::string(dir) + "Inter-Bold.ttf";
    if (access(r.c_str(), R_OK) == 0 && access(sb.c_str(), R_OK) == 0 && access(b.c_str(), R_OK) == 0) {
      reg = r.c_str(); semi = sb.c_str(); bold = b.c_str();
      break;
    }
  }
  const char* cjk = pick({"/usr/share/fonts/truetype/winfonts/NotoSansSC-VF.ttf",
                          "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                          "/usr/share/fonts/truetype/noto/NotoSansSC-Regular.otf"});
  if (cjk) { reg = cjk; semi = cjk; bold = cjk; }
  if (!reg) {
    reg = pick({"/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"});
    semi = pick({"/usr/share/fonts/truetype/noto/NotoSans-SemiBold.ttf", "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"});
    bold = pick({"/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"});
  }
  // Latin-1, Romanian (ă â î ș ț, also the cedilla forms), dashes, bullet, ellipsis.
  static const ImWchar latin_ranges[] = {0x0020, 0x00FF, 0x0100, 0x017F, 0x0218, 0x021B, 0x2013, 0x2014, 0x201C, 0x201E, 0x2022, 0x2022, 0x2026, 0x2026, 0};
  const ImWchar* ranges = cjk ? (const ImWchar*)I18nGlyphRanges() : latin_ranges;
  ImFontConfig cfg;
  cfg.OversampleH = 2;
  if (reg) {
    fonts_.regular = io.Fonts->AddFontFromFileTTF(reg, 16.5f * sc, &cfg, ranges);
    fonts_.small = io.Fonts->AddFontFromFileTTF(reg, 14.f * sc, &cfg, ranges);
  } else {
    fonts_.regular = io.Fonts->AddFontDefault();
    io.FontGlobalScale = sc;
  }
  if (semi) fonts_.semibold = io.Fonts->AddFontFromFileTTF(semi, 16.5f * sc, &cfg, ranges);
  if (semi) fonts_.title = io.Fonts->AddFontFromFileTTF(semi, 19.f * sc, &cfg, ranges);
  if (bold) fonts_.big = io.Fonts->AddFontFromFileTTF(bold, 44.f * sc, &cfg, ranges);
  io.FontDefault = F(fonts_.regular);

  ImGui::StyleColorsDark();
  ImGuiStyle& st = ImGui::GetStyle();
  st.WindowRounding = 0;
  st.ChildRounding = 12;
  st.FrameRounding = 8;
  st.PopupRounding = 8;
  st.GrabRounding = 8;
  st.ScrollbarRounding = 6;
  st.WindowBorderSize = 0;
  st.ChildBorderSize = 1;
  st.FrameBorderSize = 0;
  st.WindowPadding = ImVec2(14, 12);
  st.FramePadding = ImVec2(10, 6);
  st.ItemSpacing = ImVec2(10, 8);
  st.ItemInnerSpacing = ImVec2(8, 6);
  st.ScrollbarSize = 10;
  st.GrabMinSize = 14;
  ImVec4* c = st.Colors;
  c[ImGuiCol_Text] = v4(cText);
  c[ImGuiCol_TextDisabled] = v4(cDim);
  c[ImGuiCol_WindowBg] = v4(cPanel);
  c[ImGuiCol_ChildBg] = v4(cCard);
  c[ImGuiCol_PopupBg] = v4(IM_COL32(21, 37, 42, 250));
  c[ImGuiCol_Border] = v4(cBorder);
  c[ImGuiCol_FrameBg] = v4(IM_COL32(27, 47, 53, 255));
  c[ImGuiCol_FrameBgHovered] = v4(IM_COL32(33, 56, 63, 255));
  c[ImGuiCol_FrameBgActive] = v4(IM_COL32(38, 59, 65, 255));
  c[ImGuiCol_SliderGrab] = v4(cAccent);
  c[ImGuiCol_SliderGrabActive] = v4(cAccentHi);
  c[ImGuiCol_CheckMark] = v4(cAccent);
  c[ImGuiCol_Button] = v4(IM_COL32(27, 47, 53, 255));
  c[ImGuiCol_ButtonHovered] = v4(IM_COL32(33, 56, 63, 255));
  c[ImGuiCol_ButtonActive] = v4(IM_COL32(41, 66, 73, 255));
  c[ImGuiCol_Header] = v4(IM_COL32(27, 47, 53, 255));
  c[ImGuiCol_HeaderHovered] = v4(IM_COL32(38, 59, 65, 255));
  c[ImGuiCol_HeaderActive] = v4(cAccentSoft);
  c[ImGuiCol_Separator] = v4(cBorder);
  c[ImGuiCol_ScrollbarBg] = v4(IM_COL32(0, 0, 0, 0));
  c[ImGuiCol_ScrollbarGrab] = v4(IM_COL32(38, 59, 65, 255));
  c[ImGuiCol_PlotLines] = v4(cAccent);
  c[ImGuiCol_TextSelectedBg] = v4(cAccentSoft);
  st.ScaleAllSizes(sc);
}

void Ui::pictureArea(uint32_t width, uint32_t height, float& x, float& y, float& w, float& h) const {
  const float sc = scale_;
  x = kGap * sc;
  y = (kTop + kGap) * sc;
  w = std::max(64.f, (float)width - (kSide + 2 * kGap) * sc);
  h = std::max(36.f, (float)height - (kTop + kBottom + kStatus + 2 * kGap) * sc);
}

void Ui::panel(const UiState& s, const Cmd& cmd) {
  if (s.fatal) {   // the error screen: nothing but the message
    ImGui::GetBackgroundDrawList()->AddRectFilled(ImVec2(0, 0), ImGui::GetIO().DisplaySize, cBg);
    bannerWindows(s, cmd, true);
    return;
  }
  // Backdrop around the picture (the picture itself is never covered).
  {
    ImDrawList* bg = ImGui::GetBackgroundDrawList();
    const ImVec2 ds = ImGui::GetIO().DisplaySize;
    const float rx = (float)s.rectX, ry = (float)s.rectY, rw = (float)s.rectW, rh = (float)s.rectH;
    bg->AddRectFilled(ImVec2(0, 0), ImVec2(ds.x, ry), cBg);
    bg->AddRectFilled(ImVec2(0, ry + rh), ImVec2(ds.x, ds.y), cBg);
    bg->AddRectFilled(ImVec2(0, ry), ImVec2(rx, ry + rh), cBg);
    bg->AddRectFilled(ImVec2(rx + rw, ry), ImVec2(ds.x, ry + rh), cBg);
    bg->AddRect(ImVec2(rx - 1, ry - 1), ImVec2(rx + rw + 1, ry + rh + 1), cBorder, 0, 0, 1.f);
  }
  topBar(s, cmd);
  sidebar(s, cmd);
  bottomPanels(s);
  statusLine(s, cmd);
  hudOverlay(s, cmd);
  bannerWindows(s, cmd, true);
  connectWizard(s, cmd);
}

// The in-picture stats card (design: Fullscreen play mode, corner overlay). Drawn with draw lists on
// the foreground list, so it is part of the same draw data that goes onto every presented picture
// (real and generated), and costs a handful of triangles.
void Ui::statsCard(const UiHud& h) {
  const ImVec2 ds = ImGui::GetIO().DisplaySize;
  static const float kSize[] = {1.f, 0.8f, 1.f, 1.3f, 1.6f, 2.f, 2.4f};
  const float k = scale_ * kSize[std::clamp(h.size, 0, 6)];
  ImDrawList* dl = ImGui::GetForegroundDrawList();
  ImFont* fReg = F(fonts_.regular);
  ImFont* fSemi = F(fonts_.title);
  auto textW = [&](ImFont* f, float sz, const std::string& t) { return f->CalcTextSizeA(sz, 1e9f, 0.f, t.c_str()).x; };
  auto sev = [&](int s) { return s == 2 ? cBad : s == 1 ? cWarn : s == 0 ? cGood : cText; };
  struct Row { int kind; std::string k, v; int sev; };   // kind 0 section, 1 key/value
  std::vector<Row> rows;
  auto kv = [&](const std::string& a, const std::string& b, int sv = -1) { rows.push_back({1, a, b, sv}); };
  const bool haveE2e = h.e2e > 0;
  kv("Real fps", fmt("%.1f", h.real));
  kv("GPU", fmt("%.1f ms", h.gpu), h.gpu > h.budget ? 2 : h.gpu > h.budget * 0.8f ? 1 : -1);
  kv("Latency", haveE2e ? fmt("%.0f ms", h.e2e) : std::string("-"), !haveE2e ? -1 : h.e2e > 120 ? 2 : h.e2e > 70 ? 1 : 0);
  kv("Dropped", fmt("%llu", h.dropped), h.dropped > 30 ? 2 : h.dropped > 0 ? 1 : -1);
  if (!h.compact) {
    rows.push_back({0, "Pipeline", "", -1});
    kv("Denoise", h.denoise);
    kv("Upscale", h.upscale);
    kv("Frame gen", h.fg.empty() ? std::string("off") : h.fg);
    kv("Duplicates", fmt("%llu", h.dup));
    if (h.nr > 0.05f) kv("Denoise time", fmt("%.1f ms", h.nr));
    if (h.flow > 0.05f) kv("Motion", fmt("%.1f ms", h.flow));
    kv("Decode", fmt("%.1f ms", h.decode));
    rows.push_back({0, "Stream", "", -1});
    kv("Source", h.capSize);
    if (h.rp) {
      kv("Bitrate", fmt("%.1f Mbps", h.kbps / 1000.0));
      kv("Loss", fmt("%llu lost", h.lost + h.corrupt), h.lost + h.corrupt > 50 ? 2 : h.lost + h.corrupt > 0 ? 1 : 0);
      if (h.rtt > 0) kv("RTT", fmt("%.0f ms", h.rtt), h.rtt > 40 ? 2 : h.rtt > 15 ? 1 : 0);
      if (!h.codec.empty()) kv("Codec", h.codec);
    }
  }
  if (h.pc) {
    const moonglide::SysStats& p = h.sys;
    bool head = false;
    auto pcRow = [&](const std::string& a, const std::string& b, int sv = -1) {
      if (!head) { rows.push_back({0, "PC", "", -1}); head = true; }
      kv(a, b, sv);
    };
    auto gb = [](long mb) { return mb / 1024.0; };
    if ((h.pc & 1u) && p.gpuUtil >= 0) {
      std::string v = fmt("%d%%", p.gpuUtil);
      if (p.gpuTempC >= 0) v += fmt("  %d\xC2\xB0""C", p.gpuTempC);
      if (p.gpuPowerW >= 0) v += fmt("  %d W", p.gpuPowerW);
      pcRow("GPU load", v, p.gpuTempC > 85 ? 2 : p.gpuTempC > 75 ? 1 : -1);
    }
    if ((h.pc & 2u) && p.cpuPct >= 0) pcRow("CPU", fmt("%.0f%%  (app %.0f%%)", p.cpuPct, p.appCpuPct));
    if ((h.pc & 4u) && p.ramTotalMB > 0) pcRow("RAM", fmt("%.1f / %.1f GB", gb(p.ramUsedMB), gb(p.ramTotalMB)));
    if ((h.pc & 8u) && p.vramTotalMB > 0)
      pcRow("VRAM", fmt("%.1f / %.1f GB", gb(p.vramUsedMB), gb(p.vramTotalMB)), p.vramUsedMB > p.vramTotalMB * 0.95 ? 1 : -1);
  }

  const float m = 20 * scale_;
  auto height = [&](float kk) {
    float hh = 10 * kk + 30 * kk + 6 * kk;
    for (const Row& r : rows) hh += (r.kind == 2 ? 15 * (1 + (int)std::count(r.k.begin(), r.k.end(), '\n')) + 4 : r.kind ? 22 : 24) * kk;
    return hh + 8 * kk + 30 * kk + 12 * kk;
  };
  const float avail = ds.y - 2 * m - (h.pos == 1 ? 38 * scale_ : 0);   // fit the screen (a big size on a small window)
  const float fit = height(k) > avail ? avail / height(k) : 1.f;
  const float kf = k * fit;
  const float rowH = 22 * kf, secH = 24 * kf, fsz = 12.5f * kf, padX = 14 * kf, padT = 10 * kf, padB = 12 * kf;
  const float W = (h.compact ? 236.f : 290.f) * kf;
  const float graphH = 30 * kf;
  float H = padT + 30 * kf + 6 * kf;
  for (const Row& r : rows) H += r.kind == 2 ? 15 * kf * (1 + (float)std::count(r.k.begin(), r.k.end(), '\n')) + 4 * kf : r.kind ? rowH : secH;
  H += 8 * kf + graphH + padB;
  // Top right: below the Settings / Exit fullscreen buttons, which show while the mouse moves.
  const float topY = h.pos == 1 ? 58 * scale_ : m;
  const float x = (h.pos & 1) ? ds.x - m - W : m;
  const float y = (h.pos & 2) ? ds.y - m - H : topY;
  const float rad = 12 * kf;
  dl->AddRectFilled(ImVec2(x, y), ImVec2(x + W, y + H), IM_COL32(16, 28, 32, 246), rad);
  dl->AddRect(ImVec2(x, y), ImVec2(x + W, y + H), cBorder, rad, 0, std::max(1.f, kf));
  // Header: big fps + "fps" and the FG chip.
  float cy = y + padT;
  const std::string big = fmt("%.1f", h.fps);
  const float bigSz = 24 * kf;
  dl->AddText(fSemi, bigSz, ImVec2(x + padX, cy), cText, big.c_str());
  dl->AddText(fReg, 12 * kf, ImVec2(x + padX + textW(fSemi, bigSz, big) + 5 * kf, cy + bigSz - 12 * kf - 4 * kf), cDim, "fps");
  {
    const bool on = !h.fg.empty();
    const std::string chip = on ? "FG " + h.fg : "FG off";
    const float cs = 11 * kf, cw = textW(fReg, cs, chip) + 18 * kf, ch = 20 * kf;
    const ImVec2 a(x + W - padX - cw, cy + (bigSz - ch) / 2 + 1 * kf);
    if (on) dl->AddRectFilled(a, ImVec2(a.x + cw, a.y + ch), cAccentSoft, 6 * kf);
    dl->AddRect(a, ImVec2(a.x + cw, a.y + ch), on ? cAccent : cBorder, 6 * kf, 0, std::max(1.f, kf));
    dl->AddText(fReg, cs, ImVec2(a.x + 9 * kf, a.y + (ch - cs) / 2 - 1 * kf), on ? cAccent : cDim, chip.c_str());
  }
  cy += 30 * kf + 6 * kf;
  for (size_t i = 0; i < rows.size(); ++i) {
    const Row& r = rows[i];
    if (r.kind == 2) {   // dim hint under a row
      float ly = cy;
      size_t from = 0;
      for (;;) {
        const size_t nl = r.k.find('\n', from);
        const std::string line = r.k.substr(from, nl == std::string::npos ? std::string::npos : nl - from);
        dl->AddText(fReg, 11 * kf, ImVec2(x + padX, ly - 4 * kf), cDim, line.c_str());
        ly += 15 * kf;
        if (nl == std::string::npos) break;
        from = nl + 1;
      }
      cy = ly + 4 * kf;
      continue;
    }
    if (!r.kind) {
      dl->AddText(fReg, 11 * kf, ImVec2(x + padX, cy + 8 * kf), cDim, r.k.c_str());
      dl->AddLine(ImVec2(x + padX + textW(fReg, 11 * kf, r.k) + 8 * kf, cy + 15 * kf), ImVec2(x + W - padX, cy + 15 * kf), IM_COL32(24, 42, 47, 255), 1.f);
      cy += secH;
      continue;
    }
    dl->AddText(fReg, fsz, ImVec2(x + padX, cy + (rowH - fsz) / 2 - 1 * kf), cDim, r.k.c_str());
    dl->AddText(fReg, fsz, ImVec2(x + W - padX - textW(fReg, fsz, r.v), cy + (rowH - fsz) / 2 - 1 * kf), sev(r.sev), r.v.c_str());
    cy += rowH;
  }
  // Frame-interval graph: 0..40 ms, dashed line at the frame budget.
  cy += 8 * kf;
  const float gx = x + padX, gw = W - 2 * padX;
  float gmax = 40.f;   // ms at the top; grows when the intervals are long
  for (size_t i = h.frameMs.size() > 60 ? h.frameMs.size() - 60 : 0; i < h.frameMs.size(); ++i) gmax = std::max(gmax, h.frameMs[i] * 1.15f);
  auto gy = [&](float ms) { return cy + graphH - 3 * kf - std::min(ms, gmax) / gmax * (graphH - 6 * kf); };
  for (float dx = 0; dx < gw; dx += 8 * kf)
    dl->AddLine(ImVec2(gx + dx, gy(h.budget)), ImVec2(gx + std::min(dx + 4 * kf, gw), gy(h.budget)), cBorder, 1.f);
  const size_t n = std::min<size_t>(h.frameMs.size(), 60);
  if (n >= 2) {
    std::vector<ImVec2> pts(n);
    for (size_t i = 0; i < n; ++i)
      pts[i] = ImVec2(gx + 1 + (float)i * (gw - 2) / (float)(n - 1), gy(h.frameMs[h.frameMs.size() - n + i]));
    dl->AddPolyline(pts.data(), (int)n, cAccent, 0, std::max(1.5f, 1.5f * kf));
  }
}

// The auto-hiding "Settings" button of the desktop app (shown while the mouse moves).
void Ui::settingsBar(float alpha, const Cmd& cmd) {
  const float sc = scale_;
  const ImVec2 ds = ImGui::GetIO().DisplaySize;
  const float bw = 138 * sc, ew = 170 * sc, gap = 8 * sc, h = 36 * sc;
  const float w = (fullscreen_ ? bw + gap + ew : bw);
  const ImVec2 pos(ds.x - w - 16 * sc, 12 * sc);
  ImGui::SetNextWindowPos(pos);
  ImGui::SetNextWindowSize(ImVec2(w, h));
  ImGui::PushStyleVar(ImGuiStyleVar_Alpha, std::clamp(alpha, 0.f, 1.f));
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
  ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.f);
  ImGui::PushStyleColor(ImGuiCol_WindowBg, v4(IM_COL32(0, 0, 0, 0)));
  ImGui::Begin("##settingsbar", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings |
                                             ImGuiWindowFlags_NoScrollbar);
  ImGui::PopStyleColor();
  ImGui::PopStyleVar(2);
  ImGui::PushStyleColor(ImGuiCol_Button, v4(IM_COL32(16, 28, 32, 235)));
  ImGui::PushStyleColor(ImGuiCol_ButtonHovered, v4(IM_COL32(27, 47, 53, 245)));
  ImGui::PushStyleColor(ImGuiCol_Border, v4(cAccent));
  ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 9 * sc);
  ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1.f);
  const bool hit = ImGui::Button("      Settings", ImVec2(bw, h));
  bool exitFs = false;
  if (ImGui::IsItemHovered()) lastMove_ = nowSeconds();
  if (fullscreen_) {
    ImGui::SameLine(0, gap);
    exitFs = ImGui::Button("Exit fullscreen", ImVec2(ew, h));
  }
  ImGui::PopStyleVar(2);
  ImGui::PopStyleColor(3);
  if (ImGui::IsWindowHovered()) lastMove_ = nowSeconds();
  // The gear.
  ImDrawList* dl = ImGui::GetWindowDrawList();
  const ImVec2 c(pos.x + 22 * sc, pos.y + h / 2);
  const float r = 7 * sc;
  const ImU32 col = IM_COL32(95, 208, 196, (int)(255 * std::clamp(alpha, 0.f, 1.f)));
  dl->AddCircle(c, r * 0.62f, col, 20, 2.f * sc);
  for (int i = 0; i < 8; ++i) {
    const float a = (float)i * 3.14159265f / 4.f;
    dl->AddLine(ImVec2(c.x + std::cos(a) * r * 0.85f, c.y + std::sin(a) * r * 0.85f),
                ImVec2(c.x + std::cos(a) * r * 1.35f, c.y + std::sin(a) * r * 1.35f), col, 2.6f * sc);
  }
  barX_ = pos.x; barY_ = pos.y; barW_ = w; barH_ = h;
  barShown_ = true;
  ImGui::End();
  ImGui::PopStyleVar();
  if (exitFs) cmd("fullscreen off");
  if (hit) setVisible(true);
}

void Ui::bannerWindows(const UiState& s, const Cmd& cmd, bool interactive) {
  const float sc = scale_;
  const ImVec2 ds = ImGui::GetIO().DisplaySize;
  // Over the picture (the whole window on the error screen).
  float ax = (float)s.rectX, ay = (float)s.rectY, aw = (float)s.rectW, ah = (float)s.rectH;
  if (s.fatal || aw < 64 || ah < 36) { ax = 0; ay = 0; aw = ds.x; ah = ds.y; }
  const float w = std::min(aw - 32 * sc, (s.fatal ? 820.f : 720.f) * sc);
  float y = ay + (s.fatal ? ah * 0.3f : std::max(24.f * sc, ah * 0.12f));
  for (size_t i = 0; i < s.banners.size(); ++i) {
    const UiBanner& b = s.banners[i];
    const ImU32 col = b.level >= 2 ? cBad : b.level == 1 ? cWarn : cAccent;
    ImGui::SetNextWindowPos(ImVec2(ax + (aw - w) / 2, y));
    ImGui::SetNextWindowSize(ImVec2(w, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(22 * sc, 16 * sc));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 12 * sc);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, v4(mix(cPanel, col, 0.14f)));
    ImGui::PushStyleColor(ImGuiCol_Border, v4(mix(cPanel, col, 0.55f)));
    ImGui::Begin(fmt("##banner%zu", i).c_str(), nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings |
                     ImGuiWindowFlags_AlwaysAutoResize | (interactive ? 0 : ImGuiWindowFlags_NoInputs));
    ImGui::PopStyleColor(2);
    ImGui::PopStyleVar(3);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 p = ImGui::GetCursorScreenPos();
    const float lh = ImGui::GetFontSize();
    {  // Left icon: a filled disc with a dark exclamation mark.
      const ImVec2 ic(p.x + 9 * sc, p.y + lh * 0.7f);
      dl->AddCircleFilled(ic, 9 * sc, col, 24);
      dl->AddLine(ImVec2(ic.x, ic.y - 5 * sc), ImVec2(ic.x, ic.y + 1.5f * sc), cInk, 2.f * sc);
      dl->AddCircleFilled(ImVec2(ic.x, ic.y + 5 * sc), 1.3f * sc, cInk, 8);
    }
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 28 * sc);
    ImGui::PushFont(F(fonts_.title));
    ImGui::PushTextWrapPos(w - 22 * sc);
    ImGui::TextUnformatted(b.title.c_str());
    ImGui::PopFont();
    if (!b.detail.empty()) {
      ImGui::PushStyleColor(ImGuiCol_Text, v4(cDim));
      ImGui::TextUnformatted(b.detail.c_str());
      ImGui::PopStyleColor();
    }
    ImGui::PopTextWrapPos();
    if (!b.actions.empty()) {
      ImGui::Dummy(ImVec2(0, 4 * sc));
      if (interactive) {
        for (size_t k = 0; k < b.actions.size(); ++k) {
          if (k) ImGui::SameLine();
          const bool primary = k == 0;
          if (primary) {
            PrimaryButton pb;
            if (ImGui::Button(b.actions[k].label.c_str())) cmd(b.actions[k].command);
          } else if (ImGui::Button(b.actions[k].label.c_str())) cmd(b.actions[k].command);
        }
      } else {
        ImGui::PushStyleColor(ImGuiCol_Text, v4(cDim));
        ImGui::TextUnformatted("Press F1 for the menu.");
        ImGui::PopStyleColor();
      }
    }
    y += ImGui::GetWindowHeight() + 10 * sc;
    ImGui::End();
  }
}

namespace {
constexpr ImGuiWindowFlags kFixed = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                                    ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus;
}

void Ui::topBar(const UiState& s, const Cmd& cmd) {
  const float sc = scale_;
  const ImVec2 ds = ImGui::GetIO().DisplaySize;
  ImGui::SetNextWindowPos(ImVec2(0, 0));
  ImGui::SetNextWindowSize(ImVec2(ds.x, kTop * sc));
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(16 * sc, 10 * sc));
  ImGui::Begin("##top", nullptr, kFixed);
  ImGui::PopStyleVar();
  ImDrawList* dl = ImGui::GetWindowDrawList();
  dl->AddLine(ImVec2(0, kTop * sc - 1), ImVec2(ds.x, kTop * sc - 1), cBorder);
  // Logo mark + name.
  {
    const ImVec2 p = ImGui::GetCursorScreenPos();
    const float h = 30 * sc;
    drawMoonMark(dl, p, h);
    ImGui::Dummy(ImVec2(h, h));
    ImGui::SameLine();
    ImGui::PushFont(F(fonts_.title));
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 1 * sc);
    ImGui::TextUnformatted("MoonGlide");
    ImGui::PopFont();
  }
  ImGui::SameLine(0, 28 * sc);
  ImGui::SetCursorPosY(13 * sc);
  chip(s.sourceChip, cCardHi, cText, sc, cBorder);
  ImGui::SameLine();
  ImGui::SetCursorPosY(13 * sc);
  chip(s.formatChip, cCardHi, cText, sc, cBorder);
  ImGui::SameLine();
  ImGui::SetCursorPosY(13 * sc);
  chip("FG  " + s.fgActive, cAccentSoft, cAccentText, sc);

  // Right side: preset picker, save, hide.
  const float right = ds.x - 16 * sc;
  const float wCombo = 260 * sc, wSave = 120 * sc, wHide = 110 * sc, wFull = 190 * sc;
  ImGui::SameLine(right - wCombo - wSave - wHide - wFull - 3 * 10 * sc);
  ImGui::SetCursorPosY(10 * sc);
  if (ImGui::Button(s.fullscreen ? "Window  (F11)" : "Fullscreen  (F11)", ImVec2(wFull, 0))) cmd("fullscreen toggle");
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Fullscreen (F11 / Alt+Enter / double-click the picture)\nThe picture fills the screen; move the mouse for the Settings bar.");
  ImGui::SameLine();
  ImGui::SetCursorPosY(10 * sc);
  ImGui::SetNextItemWidth(wCombo);
  if (ImGui::BeginCombo("##preset", "Presets", ImGuiComboFlags_HeightLarge)) {
    if (s.presets.empty()) ImGui::TextDisabled("No presets yet");
    for (const std::string& p : s.presets) {
      ImGui::PushID(p.c_str());
      if (ImGui::Selectable(p.c_str(), false, 0, ImVec2(wCombo - 90 * sc, 0))) cmd(fmt("preset load %s", p.c_str()));
      ImGui::SameLine(wCombo - 70 * sc);
      if (ImGui::SmallButton("delete")) cmd(fmt("preset delete %s", p.c_str()));
      ImGui::PopID();
    }
    ImGui::EndCombo();
  }
  ImGui::SameLine();
  if (ImGui::Button("Save preset", ImVec2(wSave, 0))) ImGui::OpenPopup("savepreset");
  if (ImGui::BeginPopup("savepreset")) {
    ImGui::TextUnformatted("Save the current settings as");
    ImGui::SetNextItemWidth(280 * sc);
    if (ImGui::IsWindowAppearing()) ImGui::SetKeyboardFocusHere();
    const bool enter = ImGui::InputText("##name", presetName_, sizeof(presetName_), ImGuiInputTextFlags_EnterReturnsTrue);
    if ((ImGui::Button("Save") || enter) && presetName_[0]) {
      cmd(fmt("preset save %s", presetName_));
      presetName_[0] = 0;
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  }
  ImGui::SameLine();
  ImGui::PushStyleColor(ImGuiCol_Button, v4(cAccentSoft));
  if (ImGui::Button("Hide  (F1)", ImVec2(wHide, 0))) setVisible(false);
  ImGui::PopStyleColor();
  ImGui::End();
}

void Ui::sidebar(const UiState& s, const Cmd& cmd) {
  const float sc = scale_;
  const ImVec2 ds = ImGui::GetIO().DisplaySize;
  ImGui::SetNextWindowPos(ImVec2(ds.x - kSide * sc, kTop * sc));
  ImGui::SetNextWindowSize(ImVec2(kSide * sc, ds.y - (kTop + kStatus) * sc));
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(16 * sc, 14 * sc));
  ImGui::Begin("##side", nullptr, kFixed);
  ImGui::PopStyleVar();
  ImDrawList* dl = ImGui::GetWindowDrawList();
  const ImVec2 wp = ImGui::GetWindowPos();
  dl->AddLine(wp, ImVec2(wp.x, wp.y + ImGui::GetWindowHeight()), cBorder);

  // Tabs.
  static const char* kTabs[] = {"Input", "Processing", "Output"};
  {
    ImGui::PushFont(F(fonts_.small));
    const float w = ImGui::GetContentRegionAvail().x;
    const ImVec2 p = ImGui::GetCursorScreenPos();
    const float h = 34 * sc, cw = w / 3;
    for (int i = 0; i < 3; ++i) {
      ImGui::SetCursorScreenPos(ImVec2(p.x + cw * i, p.y));
      ImGui::PushID(i);
      if (ImGui::InvisibleButton("tab", ImVec2(cw, h))) { tab_ = i; drawKind_ = -1; }
      const bool hover = ImGui::IsItemHovered();
      ImGui::PopID();
      const ImVec2 ts = ImGui::CalcTextSize(kTabs[i]);
      const bool on = tab_ == i;
      dl->AddText(ImVec2(p.x + cw * i + (cw - ts.x) / 2, p.y + (h - ts.y) / 2 - 2 * sc), on ? cText : hover ? cText : cDim,
                  kTabs[i]);
      if (on) dl->AddRectFilled(ImVec2(p.x + cw * i + 8 * sc, p.y + h - 3 * sc), ImVec2(p.x + cw * (i + 1) - 8 * sc, p.y + h),
                                cAccent, 2 * sc);
    }
    dl->AddLine(ImVec2(p.x, p.y + h), ImVec2(p.x + w, p.y + h), cBorder);
    ImGui::SetCursorScreenPos(ImVec2(p.x, p.y + h + 10 * sc));
    ImGui::PopFont();
  }

  ImGui::PushStyleColor(ImGuiCol_ChildBg, v4(cPanel));
  ImGui::BeginChild("##cards", ImVec2(0, 0), 0, 0);
  ImGui::PopStyleColor();
  switch (tab_) {
    case 0: tabSource(s, cmd); break;
    case 1: tabProcessing(s, cmd); break;
    default: tabDisplay(s, cmd); break;
  }
  ImGui::EndChild();
  ImGui::End();
}

// ---- cards ----------------------------------------------------------------------------------

namespace {

// Card header: icon tile, title, subtitle, optional switch (on != nullptr), chevron. Returns
// whether the body is open; `toggled` is set when the switch was clicked.
bool cardBegin(std::map<std::string, bool>& expanded, const char* id, const char* icon, const char* title,
               const std::string& subtitle, const bool* on, bool* toggled, float sc, ImFont* semibold, ImFont* small,
               bool defaultOpen = false) {
  ImGui::PushID(id);
  auto it = expanded.find(id);
  if (it == expanded.end()) it = expanded.emplace(id, defaultOpen).first;
  bool& open = it->second;
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(14 * sc, 12 * sc));
  ImGui::PushStyleColor(ImGuiCol_ChildBg, v4(cCard));
  ImGui::BeginChild("card", ImVec2(-FLT_MIN, 0), ImGuiChildFlags_Borders | ImGuiChildFlags_AutoResizeY |
                                                     ImGuiChildFlags_AlwaysUseWindowPadding);
  ImGui::PopStyleColor();
  ImGui::PopStyleVar();
  ImDrawList* dl = ImGui::GetWindowDrawList();
  const ImVec2 p = ImGui::GetCursorScreenPos();
  const float w = ImGui::GetContentRegionAvail().x;
  const float ih = 38 * sc;
  // Header click area (everything left of the switch).
  const float switchW = on ? 52 * sc : 0;
  if (ImGui::InvisibleButton("hdr", ImVec2(w - switchW, ih))) open = !open;
  const bool hover = ImGui::IsItemHovered();
  const bool active = on ? *on : true;
  dl->AddRectFilled(p, ImVec2(p.x + ih, p.y + ih), active ? cAccentSoft : IM_COL32(27, 47, 53, 255), 9 * sc);
  {
    const ImVec2 ts = semibold->CalcTextSizeA(semibold->FontSize, FLT_MAX, 0, icon);
    dl->AddText(semibold, semibold->FontSize, ImVec2(p.x + (ih - ts.x) / 2, p.y + (ih - ts.y) / 2),
                active ? cAccent : cDim, icon);
  }
  dl->AddText(semibold, semibold->FontSize, ImVec2(p.x + ih + 12 * sc, p.y + 1 * sc), hover ? IM_COL32(255, 255, 255, 255) : cText, title);
  dl->AddText(small, small->FontSize, ImVec2(p.x + ih + 12 * sc, p.y + ih - small->FontSize - 1 * sc), cDim, subtitle.c_str());
  // Chevron.
  {
    const float cx = p.x + w - switchW - 14 * sc, cy = p.y + ih / 2, r = 5 * sc;
    if (open) dl->AddTriangleFilled(ImVec2(cx - r, cy - r / 2), ImVec2(cx + r, cy - r / 2), ImVec2(cx, cy + r / 2), cDim);
    else dl->AddTriangleFilled(ImVec2(cx - r / 2, cy - r), ImVec2(cx - r / 2, cy + r), ImVec2(cx + r / 2, cy), cDim);
  }
  if (on) {
    ImGui::SameLine(0, 0);
    ImGui::SetCursorScreenPos(ImVec2(p.x + w - 42 * sc, p.y + (ih - 24 * sc) / 2));
    if (toggleSwitch("on", *on, sc) && toggled) *toggled = true;
    ImGui::SetCursorScreenPos(ImVec2(p.x, p.y + ih));
  }
  if (open) {
    ImGui::Dummy(ImVec2(1, 2 * sc));
    dl->AddLine(ImVec2(p.x, p.y + ih + 6 * sc), ImVec2(p.x + w, p.y + ih + 6 * sc), cBorder);
    ImGui::Dummy(ImVec2(1, 4 * sc));
  }
  return open;
}

void cardEnd(float sc) {
  ImGui::EndChild();
  ImGui::PopID();
  ImGui::Dummy(ImVec2(1, 4 * sc));
}

void restartNote(const UiState& s, const char* key, bool sameLine = true) {
  if (get(s.current, key) != get(s.running, key)) {
    if (sameLine) ImGui::SameLine();
    ImGui::TextColored(v4(cWarn), "applies after restart");
  }
}

// One "applies after restart" note for a card: its own key changed, or the NVIDIA SDK switch did while the card uses it.
void restartIf(const UiState& s, const char* key, bool usesNvidia) {
  const bool keyDiff = get(s.current, key) != get(s.running, key), sdkDiff = get(s.current, "nvvfx") != get(s.running, "nvvfx");
  if (usesNvidia ? keyDiff || sdkDiff : keyDiff && !get(s.running, "nvvfx").empty())
    ImGui::TextColored(v4(cWarn), "applies after restart");
}

}  // namespace

#define CARD(id, icon, title, sub, onp, tog, defOpen) \
  cardBegin(expanded_, id, icon, title, sub, onp, tog, scale_, F(fonts_.semibold), F(fonts_.small), defOpen)

namespace { void dimText(const std::string& text); }

void Ui::tabProcessing(const UiState& s, const Cmd& cmd) {
  const float sc = scale_;
  const Settings& c = s.current;
  const UiStats& st = s.stats;
  const moonglide::SysStats& sy = s.sys;
  const bool rpSrc = get(c, "source", "remoteplay") == "remoteplay";
  // NVIDIA choices exist only with the (optional, user-installed) NVIDIA Video Effects SDK.
  const bool nvOn = !get(c, "nvvfx").empty();
  const bool nvPresent = nvOn || !s.nvvfxStages.empty() || access("/usr/local/VideoFX/lib", F_OK) == 0;
  auto nvStage = [&](const char* key) { const std::string v = get(c, key, "off"); return nvOn && v != "off" && !v.empty(); };
  const bool nvDn = nvStage("nvvfx-denoise"), nvSr = nvStage("nvvfx-sr"), nvFg = nvStage("nvvfx-fg");
  // Turning the SDK on enables every NVIDIA stage by default; keep only the chosen one.
  auto ensureSdk = [&](const char* keep) {
    if (nvOn) return;
    cmd("set nvvfx /usr/local/VideoFX");
    for (const char* k : {"nvvfx-denoise", "nvvfx-sr", "nvvfx-fg"})
      if (strcmp(k, keep) != 0) cmd(std::string("set ") + k + " off");
  };
  // Leaving the last NVIDIA stage switches the SDK off again.
  auto dropSdk = [&](bool thisOn) { if (thisOn && (int)nvDn + (int)nvSr + (int)nvFg == 1) cmd("set nvvfx \"\""); };
  const bool nrOn = getb(c, "nr", false);
  const bool fsr = get(c, "upscale", "fsr") == "fsr";
  const std::string fgMode = get(c, "fg", "rife");
  const int mult = std::clamp(atoi(get(c, "fg-mult", "2").c_str()), 2, 4);
  const float intensity = getf(c, "intensity", 0.6f);
  const float deband = getf(c, "deband", 0.f);
  const bool cleanOn = getb(c, "clean", false);

  // VRAM cost of a feature: measured here when it was switched on, else the table (MiB, 1080p: streaming alone
  // 639, +RIFE 1685, NR 2221, NR+RIFE 3267, own FG 850).
  const bool rifeOn = fgMode == "rife" && !nvFg, ownOn = fgMode == "own" && !nvFg;
  {
    auto track = [&](const char* f, bool on) {
      auto it = featPrev_.find(f);
      if (it == featPrev_.end()) { featPrev_[f] = on; return; }
      if (on && !it->second && sy.vramAppMB >= 0) {
        const bool alone = costWatch_.empty();
        costWatch_.clear();   // two features switched on together can't be told apart
        if (alone) costWatch_[f] = {(float)sy.vramAppMB, ImGui::GetTime()};
      }
      if (!on) costWatch_.erase(f);
      it->second = on;
      auto w = costWatch_.find(f);
      if (w != costWatch_.end() && ImGui::GetTime() - w->second.t > 6 && sy.vramAppMB >= 0) {
        const float d = (float)sy.vramAppMB - w->second.before;
        if (d > 50) liveCost_[f] = d;
        costWatch_.erase(w);
      }
    };
    track("nr", nrOn);
    track("rife", rifeOn);
    track("own", ownOn);
  }
  auto costMB = [&](const char* f, float table) { auto it = liveCost_.find(f); return it != liveCost_.end() ? it->second : table; };
  auto costStr = [&](const char* f, float table) {
    const float mb = costMB(f, table);
    return mb < 100 ? std::string("~0") : fmt("+%.1f GB", mb / 1024);
  };
  const float nrCost = costMB("nr", 1582), rifeCost = costMB("rife", 1046), ownCost = costMB("own", 211);
  // Dim line of what each choice costs; a warning (not a block) when the free VRAM is below a cost.
  auto memLine = [&](const std::string& text) { dimText(text); };
  auto lowVram = [&](const char* name, float mb) {
    if (sy.vramFreeMB >= 0 && (float)sy.vramFreeMB < mb)
      ImGui::TextColored(v4(cWarn), "%s needs ~%.1f GB, %.1f GB free", name, mb / 1024, sy.vramFreeMB / 1024.0);
  };

  // ---- pipeline strip: what is really running, from the live status ----
  const std::string upName = nvSr ? "NVIDIA VSR" : fsr ? "FSR 1" : "Bicubic";
  const std::string dnName = nvDn ? "NVIDIA denoise" : nrOn ? "MoonGlide NR" : "";
  const std::string fgName = s.fgActive;
  {
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(14 * sc, 12 * sc));
    ImGui::PushStyleColor(ImGuiCol_ChildBg, v4(cCard));
    ImGui::BeginChild("pipe", ImVec2(-FLT_MIN, 0), ImGuiChildFlags_Borders | ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding);
    ImGui::PopStyleColor();
    ImGui::PopStyleVar();
    ImGui::PushFont(F(fonts_.semibold));
    ImGui::TextUnformatted("Processing");
    ImGui::PopFont();
    std::vector<std::string> parts;
    if (s.srcW) parts.push_back(fmt("%s %up%s", rpSrc ? "PS5" : "Capture", s.srcH, s.srcFps > 0 ? fmt("%.0f", s.srcFps).c_str() : ""));
    if (cleanOn && !rpSrc) parts.push_back("Clean-up");
    if (!dnName.empty()) parts.push_back(dnName);
    parts.push_back(s.outW ? fmt("%s -> %ux%u", upName.c_str(), s.outW, s.outH) : upName);
    if (fgName != "off") parts.push_back(st.presentFps > 1 ? fmt("%s -> %.0f fps", fgName.c_str(), st.presentFps) : fgName);
    else if (st.presentFps > 1) parts.push_back(fmt("%.0f fps", st.presentFps));
    ImGui::PushFont(F(fonts_.small));
    const float maxX = ImGui::GetWindowPos().x + ImGui::GetWindowWidth() - 14 * sc;
    for (size_t i = 0; i < parts.size(); ++i) {
      const ImVec2 ts = ImGui::CalcTextSize(parts[i].c_str());
      if (i > 0) {
        ImGui::SameLine(0, 3 * sc);
        if (ImGui::GetCursorScreenPos().x + ts.x + 40 * sc > maxX) ImGui::NewLine();
        else {
          ImGui::TextColored(v4(cDim), ">");
          ImGui::SameLine(0, 3 * sc);
        }
      }
      chip(parts[i], i == 0 ? 0 : cAccentSoft, i == 0 ? cText : cAccentText, sc, i == 0 ? cBorder : 0);
    }
    std::string line;
    if (st.gpu > 0) line = fmt("GPU %.1f ms", st.gpu);
    if (st.e2e > 0) line += fmt("%slatency %.0f ms", line.empty() ? "" : " · ", st.e2e);
    if (!line.empty()) ImGui::TextColored(v4(cDim), "%s", line.c_str());
    std::string res;
    if (sy.gpuUtil >= 0) res = fmt("GPU %d%%", sy.gpuUtil);
    if (sy.vramAppMB >= 0) res += fmt("%sMoonGlide %.1f GB VRAM", res.empty() ? "" : " · ", sy.vramAppMB / 1024.0);
    if (sy.vramFreeMB >= 0) res += fmt("%s%.1f GB free", res.empty() ? "" : " · ", sy.vramFreeMB / 1024.0);
    if (!res.empty()) ImGui::TextColored(v4(cDim), "%s", res.c_str());
    ImGui::PopFont();
    sectionLabel("PRESET");
    // Processing presets: the same commands the cards send.
    const bool base = !nvDn && !nvSr && !nvFg && fsr && fgMode == "rife" && mult == 2 && !cleanOn;
    auto near = [](float a, float b) { return std::fabs(a - b) < 0.03f; };
    const int match = !base ? -1 : (!nrOn && deband <= 0.f) ? 0 : (nrOn && near(intensity, 0.4f) && deband <= 0.f) ? 1
                      : (nrOn && near(intensity, 0.6f) && (!rpSrc || near(deband, 2.f))) ? 2 : -1;
    const int pk = segmented("procpreset", {"Lowest latency", "Balanced", "Best quality"}, match, sc);
    if (pk >= 0) {
      if (nvDn) cmd("set nvvfx-denoise off");
      if (nvSr) cmd("set nvvfx-sr off");
      if (nvFg) cmd("set nvvfx-fg off");
      cmd("clean off");
      cmd("upscale fsr");
      cmd("fg mode rife");
      cmd("fg mult 2");
      cmd(pk == 0 ? "nr off" : "nr on");
      if (pk == 1) cmd("intensity 0.4");
      if (pk == 2) cmd("intensity 0.6");
      if (rpSrc) cmd(pk == 2 ? "deband 2" : "deband 0");
    }
    if (match < 0) dimText("Custom settings");
    ImGui::EndChild();
    ImGui::Dummy(ImVec2(1, 4 * sc));
  }

  // ---- 0: clean-up (capture card only) ----
  if (!rpSrc) {
    bool on = cleanOn, tog = false;
    const bool open = CARD("clean", "0", "Clean-up (MJPEG capture)", fmt("deblock %.2f · temporal %.2f", getf(c, "deblock", 1.f),
                                                           getf(c, "clean-strength", 0.5f)), &on, &tog, false);
    if (tog) cmd(on ? "clean off" : "clean on");
    if (open) {
      dimText("Removes JPEG block edges and noise from a compressed capture, before denoise. Not needed for a clean source.");
      float v = getf(c, "clean-strength", 0.5f);
      if (sliderRow("Temporal strength", &v, 0.f, 1.f, "%.2f", "Flow-compensated temporal denoise of the capture.")) cmd(fmt("clean strength %.3f", v));
      v = getf(c, "deblock", 1.f);
      if (sliderRow("Deblock", &v, 0.f, 2.f, "%.2f", "Removes 8x8 JPEG block edges.")) cmd(fmt("deblock %.3f", v));
    }
    cardEnd(sc);
  }

  // ---- 1: denoise ----
  {
    const int cur = nvDn ? 2 : nrOn ? 1 : 0;
    const std::string sub = cur == 2 ? "NVIDIA" + (st.nvDenoise > 0 ? fmt(" · %.1f ms", st.nvDenoise) : std::string())
                          : cur == 1 ? fmt("MoonGlide NR · strength %.2f", intensity) + (st.nr > 0 ? fmt(" · %.1f ms", st.nr) : std::string()) + " · " + costStr("nr", 1582)
                          : "Off";
    const bool open = CARD("nr", "1", "Denoise", sub, nullptr, nullptr, true);
    if (open) {
      std::vector<std::string> labels = {"Off", "MoonGlide NR", "NVIDIA"};
      if (!nvPresent) labels.pop_back();
      int k = segmented("dn", labels, cur, sc);
      if (k == 1 && !s.nrCapable) k = -1;
      if (!s.nrCapable) dimText("MoonGlide NR needs an RTX 40 or 50 series GPU.");
      if (k >= 0 && k != cur) {
        if (k == 2) { ensureSdk("nvvfx-denoise"); cmd("nr off"); cmd("set nvvfx-denoise 8"); }
        else { if (nvDn) { cmd("set nvvfx-denoise off"); dropSdk(true); } cmd(k == 1 ? "nr on" : "nr off"); }
      }
      restartIf(s, "nvvfx-denoise", nvDn || cur == 2);
      memLine(std::string("Video memory (approx., 1080p): MoonGlide NR ") + costStr("nr", 1582) +
              (cur == 1 ? std::string() : std::string(" if selected")));
      if (cur != 1) lowVram("MoonGlide NR", nrCost);
      if (cur == 0) dimText("Off is recommended for a clean source.");
      if (cur == 1) {
        float v = intensity;
        if (sliderRow("Strength", &v, 0.f, 1.f, "%.2f", "How much of the denoised picture is shown (0 = original).")) cmd(fmt("intensity %.3f", v));
      }
      bool& adv = expanded_["nr-adv"];
      ImGui::PushID("nradv");
      if (ImGui::SmallButton(adv ? "v Advanced" : "> Advanced")) adv = !adv;
      ImGui::PopID();
      if (adv) {
        if (cur == 1) {
          float v = getf(c, "local-structure", 0.5f);
          if (sliderRow("Keep fine texture", &v, 0.f, 2.f, "%.2f", "Fine detail kept from the original (local structure).")) cmd(fmt("structure %.3f", v));
          v = getf(c, "local-tone", 1.f);
          if (sliderRow("Local contrast", &v, 0.f, 2.f, "%.2f", "Local tone mapping strength of the denoiser.")) cmd(fmt("tone %.3f", v));
          v = getf(c, "skin-structure", -1.f);
          if (v < 0) v = getf(c, "local-structure", 0.5f);
          if (sliderRow("Skin detail", &v, 0.f, 2.f, "%.2f", "Detail on skin tones (faces).")) cmd(fmt("skin %.3f", v));
          sectionLabel("WORK SIZE");
          const std::vector<std::string> ws = {"1280x720", "1920x1080", "2560x1440"};
          const int kw = segmented("work", {"720p", "1080p", "1440p"},
                                   (int)(std::find(ws.begin(), ws.end(), get(c, "work", "1920x1080")) - ws.begin()), sc);
          if (kw >= 0) cmd("set work " + ws[kw]);
          restartNote(s, "work", false);
          if (switchRow("A/B split screen", s.split, sc, "Left half original, right half processed.")) cmd(s.split ? "split off" : "split on");
          if (ImGui::Button("Reset NR history", ImVec2(-FLT_MIN, 0))) cmd("reset");
        }
        const bool flowOn = getb(c, "flow", true);
        if (switchRow("Motion history (optical flow)", flowOn, sc,
                      "The denoise history, the clean-up and Optical flow frame generation follow motion. Needs NVIDIA optical flow."))
          cmd(flowOn ? "flow off" : "flow on");
        if (st.flow > 0 && flowOn) dimText(fmt("Optical flow %.1f ms", st.flow));
        if (rpSrc) {
          float v = deband;
          if (sliderRow("Remove banding (Remote Play)", &v, 0.f, 8.f, "%.1f", "Smooths colour steps in gradients from the stream's compression. 0 = off, about 2 is typical.")) cmd(fmt("deband %.2f", v));
        }
      }
    }
    cardEnd(sc);
  }

  // ---- 2: upscale ----
  {
    const int cur = nvSr ? 2 : fsr ? 1 : 0;
    const std::string sub = cur == 2 ? "NVIDIA VSR" + (st.nvSr > 0 ? fmt(" · %.1f ms", st.nvSr) : std::string()) : cur == 1 ? "FSR 1" : "Bicubic";
    const bool open = CARD("up", "2", "Upscale", sub, nullptr, nullptr, true);
    if (open) {
      std::vector<std::string> labels = {"Bicubic", "FSR 1", "NVIDIA VSR"};
      if (!nvPresent) labels.pop_back();
      const int k = segmented("upscale", labels, cur, sc);
      if (k >= 0 && k != cur) {
        if (k == 2) { ensureSdk("nvvfx-sr"); cmd("upscale fsr"); cmd("set nvvfx-sr 4"); }
        else { if (nvSr) { cmd("set nvvfx-sr off"); dropSdk(true); } cmd(k == 1 ? "upscale fsr" : "upscale bicubic"); }
      }
      restartIf(s, "nvvfx-sr", nvSr || cur == 2);
      memLine("Video memory: Bicubic ~0 · FSR 1 ~0   ·   recommended: FSR 1 (under 1 ms)");
      if (s.srcW && s.outW) dimText(fmt("%ux%u -> %ux%u (window size)", s.srcW, s.srcH, s.outW, s.outH));
      if (cur == 1) {
        // The sharpen key is RCAS stops (0 = strongest); the slider shows detail, more to the right.
        float v = 1.f - std::clamp(getf(c, "sharpen", 0.2f), 0.f, 2.f) / 2.f;
        if (sliderRow("Detail / sharpness", &v, 0.f, 1.f, "%.2f", "How much FSR sharpens the upscaled picture.")) cmd(fmt("sharpen %.3f", (1.f - v) * 2.f));
      }
    }
    cardEnd(sc);
  }

  // ---- 3: frame generation ----
  {
    // Option ids: 0 off, 1 RIFE, 2 optical flow (own), 3 LSFG, 4 NVIDIA VFG.
    const int cur = nvFg ? 4 : fgMode == "rife" ? 1 : fgMode == "own" ? 2 : fgMode == "lsfg" ? 3 : 0;
    std::vector<int> ids = {0, 1, 2};
    std::vector<std::string> labels = {"Off", "RIFE", "Optical flow"};
    if (s.lsfgLoaded || cur == 3) { ids.push_back(3); labels.push_back("LSFG"); }
    if (nvPresent) { ids.push_back(4); labels.push_back("NVIDIA"); }
    const std::string sub = s.fgActive + (cur == 4 && st.nvFg > 0 ? fmt(" · %.1f ms", st.nvFg) : std::string()) +
                            (cur == 1 ? " · " + costStr("rife", 1046) : cur == 2 ? " · " + costStr("own", 211) : std::string());
    const bool open = CARD("fg", "3", "Frame generation", sub, nullptr, nullptr, true);
    if (open) {
      const int shown = (int)(std::find(ids.begin(), ids.end(), cur) - ids.begin());
      const int k = segmented("fgmode", labels, shown, sc);
      if (k >= 0 && ids[k] != cur) {
        const int id = ids[k];
        if (id == 4) { ensureSdk("nvvfx-fg"); cmd("set nvvfx-fg 0"); }
        else {
          if (nvFg) { cmd("set nvvfx-fg off"); dropSdk(true); }
          static const char* kModes[] = {"off", "rife", "own", "lsfg"};
          cmd(fmt("fg mode %s", kModes[id]));
        }
      }
      restartNote(s, "fg", false);
      restartIf(s, "nvvfx-fg", nvFg || cur == 4);
      memLine(std::string("Video memory (approx., 1080p): RIFE ") + costStr("rife", 1046) + " · Optical flow " + costStr("own", 211) + "   ·   recommended: RIFE x2");
      if (cur != 1) lowVram("RIFE", rifeCost);
      if (cur != 2) lowVram("Optical flow", ownCost);
      if (s.rifeInstall == 1) {
        ImGui::TextColored(v4(cAccentText), "Installing RIFE: %s", s.rifeInstallLine.c_str());
        dimText("This downloads about 4.5 GB and can take several minutes. You can keep playing.");
      } else if (s.rifeInstall == 2) {
        ImGui::TextColored(v4(cAccentText), "RIFE is installed.");
        if (ImGui::Button("Restart to use RIFE")) { if (cur != 1) cmd("fg mode rife"); cmd("restart"); }
      } else if (s.rifeMissing) {
        if (s.rifeInstall == 3) dimText("RIFE install failed: " + s.rifeInstallLine + " (log: ~/.cache/moonglide/setup-rife.log)");
        else dimText("RIFE is not installed. It needs NVIDIA TensorRT and the RIFE model (about 4.5 GB download, "
                     "installed into your home folder, no admin rights). Until then Optical flow is used.");
        if (ImGui::Button(s.rifeInstall == 3 ? "Retry RIFE install" : "Install RIFE")) cmd("rife install");
      }
      if (cur == 1) {
        sectionLabel("MULTIPLIER");
        const int km = segmented("fgmult", {"x2", "x3", "x4"}, mult - 2, sc);
        if (km >= 0 && km != mult - 2) cmd(fmt("fg mult %d", km + 2));
      }
      if (cur != 0 && cur != 4) {
        bool& adv = expanded_["fg-adv"];
        ImGui::PushID("fgadv");
        if (ImGui::SmallButton(adv ? "v Advanced" : "> Advanced")) adv = !adv;
        ImGui::PopID();
        if (adv) {
          if (cur == 1) {
            float v = getf(c, "rife-cut", 0.15f);
            if (sliderRow("Scene cut: hold above", &v, 0.f, 1.f, "%.2f", "Show the real frame instead of a blend when the picture changes by more than this. 0 = never hold.")) cmd(fmt("rife cut %.3f", v));
            v = getf(c, "rife-outlier", 0.f);
            if (sliderRow("Outlier guard", &v, 0.f, 1.f, "%.2f", "Replaces generated pixels outside both real frames' local range (specks). 0 = off.")) cmd(fmt("rife outlier %.3f", v));
            v = getf(c, "rife-still", 0.02f);
            if (sliderRow("Still-area guard", &v, 0.f, 0.2f, "%.3f", "Where both real frames agree, show them and ignore the motion estimate. 0 = off.")) cmd(fmt("rife still %.3f", v));
            v = getf(c, "fg-real-delay", 0.f);
            if (sliderRow("Real-frame delay (0 = even)", &v, 0.f, 0.5f, "%.2f", "Picture spacing as a fraction of the real frame period. 0 spaces them evenly.")) cmd(fmt("fg real-delay %.3f", v));
          }
          if (cur == 2) {
            const unsigned flags = (unsigned)strtoul(get(c, "fg-flags", "232").c_str(), nullptr, 0);
            float hold = getf(c, "fg-hold", 0.2f);
            if (sliderRow("Hold when the flow fails above", &hold, 0.f, 1.f, "%.2f",
                          "Lower: fewer ghosts on fast pans, more judder. 0 = never hold."))
              cmd(fmt("fg hold %.3f", hold));
            sectionLabel("ARTEFACT GUARDS");
            struct Bit { unsigned bit; const char* name; const char* hint; };
            const Bit bits[] = {
                {256, "Static pass-through", "Where both real frames agree around a pixel, show their blend and ignore the flow."},
                {128, "Flow median", "3x3 median on the flow field: lone bad vectors are replaced."},
                {64, "Outlier rejection", "A pixel brighter/darker than both real frames around it is replaced by their blend (fleas)."},
                {32, "Static guard", "Static areas stay inside their own colour range (text over moving backgrounds)."},
                {8, "Clamp to source range", "Never leave the 3x3 colour range of the nearer frame."},
                {4, "Dilate disagreement", "Grow the fallback area by 2 px."},
                {2, "Hard pick", "Pick one frame where they disagree instead of a soft blend."},
            };
            for (const Bit& b : bits)
              if (switchRow(b.name, (flags & b.bit) != 0, sc, b.hint)) cmd(fmt("fg flags %u", flags ^ b.bit));
            const bool dbg = getb(c, "fg-debug", false);
            if (switchRow("Tint fallback pixels (debug)", dbg, sc)) cmd(dbg ? "fg debug off" : "fg debug on");
          }
        }
      }
    }
    cardEnd(sc);
  }

  // ---- 4: static zones ----
  {
    const bool show = getb(c, "hud-show", false);
    const bool open = CARD("zones", "4", "Static zones", s.zones.empty() ? std::string("none · whole picture processed")
                           : fmt("%zu zone%s · kept unprocessed", s.zones.size(), s.zones.size() == 1 ? "" : "s"),
                           nullptr, nullptr, false);
    if (open) {
      dimText("Areas like a minimap, score or on-screen text that should stay sharp. Inside a zone the picture is the captured "
              "frame: not denoised, not upscaled or sharpened (plain resize only) and not interpolated by frame "
              "generation. Pick a shape, then drag on the picture to draw it.");
      if (switchRow("Outline zones on the picture (H)", show, sc)) cmd(show ? "hud show off" : "hud show on");
      const float bw = (ImGui::GetContentRegionAvail().x - 2 * 8 * sc) / 3;
      auto drawBtn = [&](const char* label, int kind) {
        const bool on = drawKind_ == kind;
        if (on) ImGui::PushStyleColor(ImGuiCol_Button, v4(cAccent));
        if (on) ImGui::PushStyleColor(ImGuiCol_Text, v4(cInk));
        if (ImGui::Button(on ? "Drawing... Esc" : label, ImVec2(bw, 0))) drawKind_ = on ? -1 : kind;
        if (on) ImGui::PopStyleColor(2);
      };
      drawBtn("+ Rectangle", 0);
      ImGui::SameLine(0, 8 * sc);
      drawBtn("+ Ellipse", 1);
      ImGui::SameLine(0, 8 * sc);
      if (ImGui::Button("Clear all", ImVec2(bw, 0))) cmd("hud clear");
    }
    cardEnd(sc);
  }
  for (size_t i = 0; i < s.zones.size(); ++i) {
    HudZone z = s.zones[i];
    const std::string id = fmt("zone%zu", i);
    const bool open = CARD(id.c_str(), z.ellipse ? "()" : "[]", fmt("Zone %zu", i + 1).c_str(),
                           fmt("%s %.3f, %.3f · %.3f x %.3f · feather %.0f", z.ellipse ? "ellipse" : "rect", z.a, z.b, z.c, z.d, z.feather),
                           nullptr, nullptr, false);
    if (open) {
      bool changed = false;
      int kind = z.ellipse ? 1 : 0;
      const int k = segmented("kind", {"Rectangle", "Ellipse"}, kind, sc);
      if (k >= 0 && k != kind) { kind = k; changed = true; }
      const char* names[4] = {z.ellipse ? "Centre x" : "Left", z.ellipse ? "Centre y" : "Top", z.ellipse ? "Radius x" : "Width",
                              z.ellipse ? "Radius y" : "Height"};
      float* vals[4] = {&z.a, &z.b, &z.c, &z.d};
      for (int j = 0; j < 4; ++j) changed |= sliderRow(names[j], vals[j], 0.f, 1.f, "%.3f");
      changed |= sliderRow("Feather (px)", &z.feather, 0.f, 40.f, "%.0f");
      if (changed) {
        z.ellipse = kind == 1;
        z.c = std::max(0.002f, z.c);
        z.d = std::max(0.002f, z.d);
        cmd(fmt("hud set %zu %s:%.0f", i, hudZoneString(z).c_str(), z.feather));
      }
      ImGui::PushStyleColor(ImGuiCol_Button, v4(IM_COL32(74, 32, 36, 255)));
      if (ImGui::Button("Delete zone", ImVec2(-FLT_MIN, 0))) cmd(fmt("hud del %zu", i));
      ImGui::PopStyleColor();
    }
    cardEnd(sc);
  }
  tabNvidia(s, cmd);
}

namespace {

// A drop-down; returns the picked index or -1.
int comboRow(const char* id, const std::string& preview, const std::vector<std::string>& labels, float width = -FLT_MIN) {
  int pick = -1;
  ImGui::SetNextItemWidth(width);
  if (ImGui::BeginCombo(id, preview.c_str())) {
    for (size_t i = 0; i < labels.size(); ++i) {
      ImGui::PushID((int)i);
      if (ImGui::Selectable(labels[i].c_str(), labels[i] == preview)) pick = (int)i;
      ImGui::PopID();
    }
    ImGui::EndCombo();
  }
  return pick;
}

std::string sizeLabel(const std::string& wxh) {
  static const std::map<std::string, const char*> names = {{"1280x720", "720p"}, {"1920x1080", "1080p"},
                                                           {"2560x1440", "1440p"}, {"3840x2160", "2160p"}};
  auto it = names.find(wxh);
  return it == names.end() ? wxh : std::string(it->second) + "  (" + wxh + ")";
}

void dimText(const std::string& text) {
  ImGui::PushStyleColor(ImGuiCol_Text, v4(cDim));
  ImGui::TextWrapped("%s", text.c_str());
  ImGui::PopStyleColor();
}

}  // namespace

// ---- "Connect your PlayStation" wizard -------------------------------------------------------------

// Asks the X11 CLIPBOARD owner for its text; the answer arrives as a SelectionNotify (handleEvent).
void Ui::requestClipboard() {
  XConvertSelection(display_, XInternAtom(display_, "CLIPBOARD", False), XInternAtom(display_, "UTF8_STRING", False),
                    XInternAtom(display_, "MOONGLIDE_CLIP", False), window_, CurrentTime);
  XFlush(display_);
}

namespace {
// The address PlayStation Network's sign-in ends on (the page itself is blank or an error).
bool looksLikePsnCallback(const std::string& t) {
  if (t.size() < 20 || t.size() > 2000 || t.find_first_of(" \n\r\t") != std::string::npos) return false;
  if (t.rfind("https://", 0) != 0 && t.rfind("http://", 0) != 0) return false;
  return t.find("code=") != std::string::npos && (t.find("playstation") != std::string::npos || t.find("redirect") != std::string::npos);
}
}  // namespace

void Ui::connectWizard(const UiState& s, const Cmd& cmd) {
  const float sc = scale_;
  const Settings& c = s.current;
  if (!rpSetup_) rpSetup_ = std::make_unique<RpSetup>();
  // Auto-show once per run: Remote Play is the source, no console is set up yet, and it was never dismissed.
  if (!wizChecked_) {
    wizChecked_ = true;
    if (get(c, "source", "remoteplay") == "remoteplay" && get(c, "rp-host").empty() && get(c, "wizard") != "done") wizOpen_ = true;
  }
  if (!wizOpen_) {
    wizPopup_ = false;
    return;
  }
  const std::string tap = tapBinary(get(c, "rp-tap"));
  const RpSetup::Snapshot rs = rpSetup_->snapshot();
  const bool signedIn = !get(c, "rp-psn-account").empty();
  const double now = nowSeconds();
  if (!wizScanStarted_) {
    wizScanStarted_ = true;
    wizStep_ = 0;
    rpSetup_->scan(tap);
  }
  // The account arrives from a worker thread.
  {
    std::string acc, online;
    if (rpSetup_->takeAccount(acc, online)) {
      cmd("rp psn " + acc + " " + (online.empty() ? std::string("-") : online));
      wizUrl_[0] = 0;
      if (wizStep_ == 1) wizStep_ = 2;
    }
    const std::string url = rpSetup_->takeLoginUrl();
    if (!url.empty()) {
      loginLink_ = url;
      openInBrowser(url);
    }
  }
  // Step 2: watch the clipboard for the address the sign-in ends on.
  if (wizStep_ == 1 && !signedIn) {
    if (now - wizClipReq_ > 0.8) {
      wizClipReq_ = now;
      requestClipboard();
    }
    if (wizClipNew_) {
      wizClipNew_ = false;
      if (looksLikePsnCallback(wizClip_) && wizClip_ != wizClipTried_ && !rs.accountBusy) {
        wizClipTried_ = wizClip_;
        snprintf(wizUrl_, sizeof(wizUrl_), "%s", wizClip_.c_str());
        rpSetup_->accountFromUrl(tap, wizClip_);
      }
    }
  }

  if (!wizPopup_) {
    ImGui::OpenPopup("##connectwizard");
    wizPopup_ = true;
  }
  const ImVec2 ds = ImGui::GetIO().DisplaySize;
  const float w = std::min(ds.x - 32 * sc, 700 * sc), h = std::min(ds.y - 32 * sc, 600 * sc);
  ImGui::SetNextWindowPos(ImVec2(ds.x / 2, ds.y / 2), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
  ImGui::SetNextWindowSize(ImVec2(w, h));
  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 14 * sc);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(26 * sc, 22 * sc));
  ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.f);
  ImGui::PushStyleColor(ImGuiCol_PopupBg, v4(cBg));
  ImGui::PushStyleColor(ImGuiCol_Border, v4(cBorder));
  const bool open = ImGui::BeginPopupModal("##connectwizard", nullptr,
                                           ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                                               ImGuiWindowFlags_NoSavedSettings);
  ImGui::PopStyleColor(2);
  ImGui::PopStyleVar(3);
  if (!open) {
    wizPopup_ = false;
    return;
  }
  bool close = false, finished = false;
  ImGui::PushFont(F(fonts_.title));
  ImGui::TextUnformatted("Connect your PlayStation");
  ImGui::PopFont();
  ImGui::SameLine(w - 26 * sc - ImGui::CalcTextSize("Set up manually").x - 2 * ImGui::GetStyle().FramePadding.x);
  if (ImGui::SmallButton("Set up manually")) close = true;
  // Progress.
  {
    static const char* names[4] = {"Find console", "Sign in", "Pair", "Done"};
    ImGui::Spacing();
    for (int i = 0; i < 4; ++i) {
      if (i) { ImGui::SameLine(); ImGui::PushStyleColor(ImGuiCol_Text, v4(cDim)); ImGui::TextUnformatted(">"); ImGui::PopStyleColor(); ImGui::SameLine(); }
      ImGui::PushStyleColor(ImGuiCol_Text, v4(i == wizStep_ ? cAccentText : cDim));
      ImGui::Text("%d  %s", i + 1, names[i]);
      ImGui::PopStyleColor();
    }
    ImGui::Separator();
  }
  const float footer = ImGui::GetFrameHeight() + 14 * sc;
  ImGui::BeginChild("##wizbody", ImVec2(0, -footer), false);
  const std::string host = get(c, "rp-host");
  if (wizStep_ == 0) {
    ImGui::Spacing();
    dimText("Turn your PlayStation on (or into rest mode) and connect it to the same network as this computer. "
            "MoonGlide looks for it automatically.");
    if (rs.scanning) ImGui::BeginDisabled();
    if (ImGui::Button(rs.scanning ? "Scanning…" : "Scan again", ImVec2(140 * sc, 0))) rpSetup_->scan(tap);
    if (rs.scanning) ImGui::EndDisabled();
    if (rs.scanned && rs.consoles.empty() && !rs.scanning) {
      ImGui::SameLine();
      dimText(rs.scanError.empty() ? "No console found. Is it on or in rest mode, and on this network?" : rs.scanError);
    }
    for (size_t i = 0; i < rs.consoles.size(); ++i) {
      const RpSetup::Console& con = rs.consoles[i];
      const bool ready = con.state == "ready" || con.state == "READY";
      const std::string label = fmt("%s   %s   %s   %s   %s", con.name.empty() ? "(unnamed)" : con.name.c_str(), con.host.c_str(),
                                    con.type.c_str(), ready ? "ready" : "standby", con.registered ? "paired" : "not paired");
      ImGui::PushID((int)i);
      if (ImGui::Selectable(label.c_str(), false, 0, ImVec2(0, 30 * sc))) {
        if (con.registered) {
          cmd("set rp-host " + con.host);
          wizStep_ = 3;
        } else {
          snprintf(wizHost_, sizeof(wizHost_), "%s", con.host.c_str());
          wizPs4_ = con.type == "PS4";
          wizStep_ = signedIn ? 2 : 1;
          wizPairing_ = false;
        }
      }
      ImGui::PopID();
    }
    if (!rs.consoles.empty()) dimText("Click your console. A paired one is used right away; a new one is paired in the next steps.");
    sectionLabel("OR ENTER THE ADDRESS YOURSELF");
    ImGui::SetNextItemWidth(-FLT_MIN);
    ImGui::InputTextWithHint("##wizhost", "Console IP address, e.g. 192.168.1.20", wizHost_, sizeof(wizHost_));
    ImGui::Checkbox("This is a PS4 (not a PS5)", &wizPs4_);
    if (!wizHost_[0]) ImGui::BeginDisabled();
    bool go;
    { PrimaryButton pb; go = ImGui::Button("Pair this address", ImVec2(180 * sc, 0)); }
    if (!wizHost_[0]) ImGui::EndDisabled();
    if (go) { wizStep_ = signedIn ? 2 : 1; wizPairing_ = false; }
  } else if (wizStep_ == 1) {
    ImGui::Spacing();
    dimText("To pair, MoonGlide needs to know your PlayStation Network account (the one you use on the console). "
            "Your password is only typed on Sony's own page in your browser.");
    if (rs.loginBusy) ImGui::BeginDisabled();
    bool open2;
    { PrimaryButton pb; open2 = ImGui::Button(rs.loginBusy ? "Opening…" : "Open PlayStation sign-in in my browser", ImVec2(-FLT_MIN, 36 * sc)); }
    if (open2) rpSetup_->loginUrl(tap);
    if (rs.loginBusy) ImGui::EndDisabled();
    if (!rs.loginError.empty()) ImGui::TextColored(v4(cWarn), "%s", rs.loginError.c_str());
    sectionLabel("WHAT TO DO IN THE BROWSER");
    dimText("1. Sign in to PlayStation Network.\n"
            "2. The browser then ends on a blank page or an error page. That is expected.\n"
            "3. Copy that page's address: click the address bar at the top of the browser, then press Ctrl+C.\n"
            "MoonGlide notices the copied address by itself. You can also paste it below.");
    if (!loginLink_.empty()) {
      dimText("If no browser opened, open this link yourself:");
      char link[1024];
      snprintf(link, sizeof(link), "%s", loginLink_.c_str());
      ImGui::SetNextItemWidth(-FLT_MIN);
      ImGui::InputText("##wizlink", link, sizeof(link), ImGuiInputTextFlags_ReadOnly);
    }
    ImGui::Spacing();
    ImGui::SetNextItemWidth(-90 * sc);
    ImGui::InputTextWithHint("##wizurl", "Address the browser ended on", wizUrl_, sizeof(wizUrl_));
    ImGui::SameLine();
    const bool noGo = rs.accountBusy || !wizUrl_[0];
    if (noGo) ImGui::BeginDisabled();
    if (ImGui::Button("Continue", ImVec2(-FLT_MIN, 0))) rpSetup_->accountFromUrl(tap, wizUrl_);
    if (noGo) ImGui::EndDisabled();
    if (rs.accountBusy) dimText("Checking the address…");
    else if (!rs.accountError.empty()) ImGui::TextColored(v4(cWarn), "%s", rs.accountError.c_str());
    else if (!wizClipTried_.empty() && !wizUrl_[0]) dimText("Address read.");
  } else if (wizStep_ == 2) {
    ImGui::Spacing();
    ImGui::Text("Pairing with %s", wizHost_);
    if (signedIn) {
      ImGui::PushStyleColor(ImGuiCol_Text, v4(cDim));
      ImGui::Text("Signed in as %s", get(c, "rp-psn-online").empty() ? "(unknown)" : get(c, "rp-psn-online").c_str());
      ImGui::PopStyleColor();
    }
    sectionLabel("ON YOUR PLAYSTATION");
    dimText("Open Settings > System > Remote Play > Pair Device. The console shows an 8-digit PIN for a few minutes. "
            "Type it here.");
    ImGui::Spacing();
    ImGui::SetNextItemWidth(-130 * sc);
    ImGui::InputTextWithHint("##wizpin", "8-digit PIN", wizPin_, sizeof(wizPin_),
                             ImGuiInputTextFlags_CharsDecimal | ImGuiInputTextFlags_Password);
    ImGui::SameLine();
    const bool can = signedIn && wizHost_[0] && strlen(wizPin_) == 8 && !rs.pairBusy;
    if (!can) ImGui::BeginDisabled();
    bool pairHit;
    { PrimaryButton pb; pairHit = ImGui::Button(rs.pairBusy ? "Pairing…" : "Pair", ImVec2(-FLT_MIN, 0)); }
    if (pairHit) {
      rpSetup_->pair(tap, wizHost_, get(c, "rp-psn-account"), wizPin_, wizPs4_);
      memset(wizPin_, 0, sizeof(wizPin_));
      wizPairing_ = true;
    }
    if (!can) ImGui::EndDisabled();
    if (wizPairing_ && rs.pairDone) {
      if (rs.pairOk) {
        cmd("set rp-host " + rs.pairedHost);
        wizPairing_ = false;
        wizStep_ = 3;
      } else {
        ImGui::Spacing();
        ImGui::TextColored(v4(cWarn), "%s", rs.pairMessage.c_str());
        dimText("Check the PIN and try again. A PIN stops working after a few minutes: generate a new one on the console if needed.");
      }
    }
  } else {
    ImGui::Spacing();
    ImGui::TextColored(v4(cGood), "Your console is set up.");
    dimText(host.empty() ? "Pick a console to finish." : "MoonGlide will connect to " + host + ". Choose the picture quality, then restart to connect.");
    std::string curRes = get(c, "rp-res");
    if (curRes.empty()) {
      const std::string sz = get(c, "size", "1920x1080");
      curRes = sz.substr(sz.find('x') == std::string::npos ? 0 : sz.find('x') + 1) == "720" ? "720" : "1080";
    }
    sectionLabel("STREAM RESOLUTION");
    std::vector<std::string> res = {"720", "1080"}, resLabels = {"720p", "1080p"};
    if (curRes != "720" && curRes != "1080") { res.push_back(curRes); resLabels.push_back(curRes + "p"); }
    const int k = segmented("wizres", resLabels, (int)(std::find(res.begin(), res.end(), curRes) - res.begin()), sc);
    if (k >= 0) cmd("set rp-res " + res[k]);
    dimText("1080p looks sharper; 720p is lighter on the network. You can change this later in the Stream card.");
    finished = true;
  }
  ImGui::EndChild();
  // Footer.
  ImGui::SetCursorPosY(ImGui::GetWindowHeight() - 22 * sc - ImGui::GetFrameHeight());
  if (wizStep_ > 0) {
    if (ImGui::Button("Back", ImVec2(100 * sc, 0))) { wizStep_ = wizStep_ == 3 ? 0 : wizStep_ == 2 && signedIn ? 0 : wizStep_ - 1; wizPairing_ = false; }
  }
  if (finished) {
    const bool needRestart = [&]() { auto it = s.running.find("rp-host"); return it == s.running.end() || it->second != host; }();
    ImGui::SameLine();
    ImGui::SetCursorPosX(w - 26 * sc - (needRestart ? 312 : 150) * sc);
    if (needRestart) {
      bool rst;
      { PrimaryButton pb; rst = ImGui::Button("Save and restart", ImVec2(150 * sc, 0)); }
      if (rst) { pendingCmds_.push_back("restart"); close = true; }
      ImGui::SameLine();
    }
    bool dn;
    { PrimaryButton pb; dn = ImGui::Button(needRestart ? "Later" : "Done", ImVec2(150 * sc, 0)); }
    if (dn) close = true;
  }
  if (close) {
    wizOpen_ = false;
    wizPopup_ = false;
    wizScanStarted_ = false;
    memset(wizPin_, 0, sizeof(wizPin_));
    cmd("wizard done");
    ImGui::CloseCurrentPopup();
  }
  ImGui::EndPopup();
}

void Ui::tabSource(const UiState& s, const Cmd& cmd) {
  const float sc = scale_;
  const Settings& c = s.current;
  if (!rpSetup_) rpSetup_ = std::make_unique<RpSetup>();
  if (!capEnum_) capEnum_ = std::make_unique<CaptureEnum>();
  const bool rp = get(c, "source", "remoteplay") == "remoteplay";
  {
    const bool open = CARD("src", "PS", "Source", rp ? "PS5 Remote Play" : "Capture card", nullptr, nullptr, true);
    if (open) {
      int k = segmented("source", {"Capture card", "PS5 Remote Play"}, rp ? 1 : 0, sc);
      if (k >= 0) cmd(k == 1 ? "set source remoteplay" : "set source dongle");
      restartNote(s, "source");
      if (!rp) dimText("An HDMI capture card (or any V4L2 video capture device).");
    }
    cardEnd(sc);
  }
  if (rp) sourceRemotePlay(s, cmd);
  else sourceCapture(s, cmd);
  tabSignal(s, cmd);
}

void Ui::sourceRemotePlay(const UiState& s, const Cmd& cmd) {
  const float sc = scale_;
  const Settings& c = s.current;
  const std::string tap = tapBinary(get(c, "rp-tap"));
  const RpSetup::Snapshot rs = rpSetup_->snapshot();
  const std::string host = get(c, "rp-host");
  {
    const bool open = CARD("rpcon", "PS", "Console", host.empty() ? "no console selected" : host, nullptr, nullptr, true);
    if (open) {
      if (ImGui::Button("Connection wizard", ImVec2(-FLT_MIN, 0))) { wizOpen_ = true; wizScanStarted_ = false; }
      sectionLabel("CONSOLES ON YOUR NETWORK");
      if (rs.scanning) ImGui::BeginDisabled();
      bool scanHit;
      { PrimaryButton pb; scanHit = ImGui::Button(rs.scanning ? "Scanning…" : "Scan", ImVec2(120 * sc, 0)); }
      if (scanHit) rpSetup_->scan(tap);
      if (rs.scanning) ImGui::EndDisabled();
      if (rs.scanned && rs.consoles.empty()) {
        ImGui::SameLine();
        dimText(rs.scanError.empty() ? "No console found. Is it on or in rest mode, and on this network?" : rs.scanError);
      }
      for (size_t i = 0; i < rs.consoles.size(); ++i) {
        const RpSetup::Console& con = rs.consoles[i];
        const bool ready = con.state == "ready" || con.state == "READY";
        const std::string label = fmt("%s   %s   %s   %s   %s", con.name.empty() ? "(unnamed)" : con.name.c_str(), con.host.c_str(),
                                      con.type.c_str(), ready ? "ready" : "standby", con.registered ? "paired" : "not paired");
        ImGui::PushID((int)i);
        if (ImGui::Selectable(label.c_str(), con.host == host, 0, ImVec2(0, 26 * sc))) {
          if (con.registered) {
            cmd("set rp-host " + con.host);
          } else {
            snprintf(pairHost_, sizeof(pairHost_), "%s", con.host.c_str());
            pairPs4_ = con.type == "PS4";
            expanded_["rppair"] = true;
          }
        }
        ImGui::PopID();
      }
      if (!rs.consoles.empty()) dimText("Click a paired console to use it, or an unpaired one to pair it.");
      sectionLabel("CONSOLE ADDRESS"); restartNote(s, "rp-host");
      char hostBuf[80];
      snprintf(hostBuf, sizeof(hostBuf), "%s", host.c_str());
      ImGui::SetNextItemWidth(-FLT_MIN);
      if (ImGui::InputText("##rphost", hostBuf, sizeof(hostBuf), ImGuiInputTextFlags_EnterReturnsTrue)) cmd(fmt("set rp-host %s", hostBuf));
      dimText(s.remotePlay.empty() ? "Restart to switch. One Remote Play session per console: close any other Remote Play app first."
                                   : s.remotePlay);
    }
    cardEnd(sc);
  }
  {
    const bool open = CARD("rppair", "PA", "Pair a console",
                           get(c, "rp-psn-online").empty() ? std::string("needs a PlayStation Network sign-in")
                                                           : "signed in as " + get(c, "rp-psn-online"),
                           nullptr, nullptr, false);
    if (open) {
      std::string acc, online;
      if (rpSetup_->takeAccount(acc, online)) {
        cmd("rp psn " + acc + " " + (online.empty() ? std::string("-") : online));
        psnUrl_[0] = 0;
      }
      const std::string url = rpSetup_->takeLoginUrl();
      if (!url.empty()) {
        loginLink_ = url;
        openInBrowser(url);
      }
      const bool signedIn = !get(c, "rp-psn-account").empty();
      sectionLabel("1  SIGN IN TO PLAYSTATION NETWORK");
      if (signedIn) {
        ImGui::Text("Signed in as %s", get(c, "rp-psn-online").empty() ? "(unknown)" : get(c, "rp-psn-online").c_str());
        ImGui::SameLine();
        if (ImGui::SmallButton("Sign out")) cmd("rp psn -");
      } else {
        if (rs.loginBusy) ImGui::BeginDisabled();
        if (ImGui::Button("Sign in to PlayStation Network", ImVec2(-FLT_MIN, 0))) rpSetup_->loginUrl(tap);
        if (rs.loginBusy) ImGui::EndDisabled();
        if (!rs.loginError.empty()) ImGui::TextColored(v4(cWarn), "%s", rs.loginError.c_str());
        if (!loginLink_.empty()) {
          dimText("Sign in in the browser. When it ends on a blank or error page, copy that page's address and paste it here. "
                  "If no browser opened, use this link:");
          char link[1024];
          snprintf(link, sizeof(link), "%s", loginLink_.c_str());
          ImGui::SetNextItemWidth(-FLT_MIN);
          ImGui::InputText("##psnlink", link, sizeof(link), ImGuiInputTextFlags_ReadOnly);
          ImGui::SetNextItemWidth(-90 * sc);
          ImGui::InputTextWithHint("##psnurl", "Address the browser ended on", psnUrl_, sizeof(psnUrl_));
          ImGui::SameLine();
          const bool noGo = rs.accountBusy || !psnUrl_[0];
          if (noGo) ImGui::BeginDisabled();
          if (ImGui::Button("Continue", ImVec2(-FLT_MIN, 0))) rpSetup_->accountFromUrl(tap, psnUrl_);
          if (noGo) ImGui::EndDisabled();
          if (rs.accountBusy) dimText("Checking…");
          if (!rs.accountError.empty()) ImGui::TextColored(v4(cWarn), "%s", rs.accountError.c_str());
        }
      }
      sectionLabel("2  PAIR WITH THE CONSOLE");
      dimText("On the console: Settings > System > Remote Play > Pair Device. It shows an 8-digit PIN for a few minutes.");
      ImGui::SetNextItemWidth(-FLT_MIN);
      ImGui::InputTextWithHint("##pairhost", "Console address (or click an unpaired console above)", pairHost_, sizeof(pairHost_));
      ImGui::Checkbox("PS4 (not PS5)", &pairPs4_);
      ImGui::SetNextItemWidth(-130 * sc);
      ImGui::InputTextWithHint("##pin", "PIN", pin_, sizeof(pin_), ImGuiInputTextFlags_CharsDecimal | ImGuiInputTextFlags_Password);
      ImGui::SameLine();
      const bool can = signedIn && pairHost_[0] && strlen(pin_) == 8 && !rs.pairBusy;
      if (!can) ImGui::BeginDisabled();
      bool pairHit;
      { PrimaryButton pb; pairHit = ImGui::Button(rs.pairBusy ? "Pairing…" : "Pair", ImVec2(-FLT_MIN, 0)); }
      if (pairHit) {
        rpSetup_->pair(tap, pairHost_, get(c, "rp-psn-account"), pin_, pairPs4_);
        memset(pin_, 0, sizeof(pin_));
      }
      if (!can) ImGui::EndDisabled();
      if (!signedIn) dimText("Sign in first.");
      if (rs.pairDone) {
        ImGui::TextColored(rs.pairOk ? v4(cAccent) : v4(cWarn), "%s", rs.pairMessage.c_str());
        if (rs.pairOk && get(c, "rp-host") != rs.pairedHost) cmd("set rp-host " + rs.pairedHost);
      }
    }
    cardEnd(sc);
  }
  {
    std::string curRes = get(c, "rp-res");
    if (curRes.empty()) {
      const std::string sz = get(c, "size", "1920x1080");
      curRes = sz.substr(sz.find('x') == std::string::npos ? 0 : sz.find('x') + 1) == "720" ? "720" : "1080";
    }
    std::string curFps = get(c, "rp-fps");
    if (curFps.empty()) curFps = atoi(get(c, "fps", "30").c_str()) >= 60 ? "60" : "30";
    const bool open = CARD("rpstream", "ST", "Stream", fmt("%sp · %s fps · %s", curRes.c_str(), curFps.c_str(), get(c, "rp-codec", "h265").c_str()),
                           nullptr, nullptr, true);
    if (open) {
      sectionLabel("RESOLUTION"); restartNote(s, "rp-res");
      std::vector<std::string> res = {"720", "1080"};   // libchiaki has no real 1440p / 2160p preset
      std::vector<std::string> resLabels = {"720p", "1080p"};
      if (curRes != "720" && curRes != "1080") { res.push_back(curRes); resLabels.push_back(curRes + "p"); }   // from the config
      int k = segmented("rpres", resLabels, (int)(std::find(res.begin(), res.end(), curRes) - res.begin()), sc);
      if (k >= 0) cmd("set rp-res " + res[k]);
      sectionLabel("FRAME RATE"); restartNote(s, "rp-fps");
      k = segmented("rpfps", {"30", "60"}, curFps == "60" ? 1 : 0, sc);
      if (k >= 0) cmd(k == 1 ? "set rp-fps 60" : "set rp-fps 30");
      sectionLabel("VIDEO CODEC"); restartNote(s, "rp-codec");
      k = segmented("rpcodec", {"H.264", "H.265 (HEVC)"}, get(c, "rp-codec", "h265") == "h264" ? 0 : 1, sc);
      if (k >= 0) cmd(k == 0 ? "set rp-codec h264" : "set rp-codec h265");
      sectionLabel("BITRATE"); restartNote(s, "rp-bitrate");
      const int kbps = atoi(get(c, "rp-bitrate", "0").c_str());
      if (switchRow("Console default", kbps == 0, sc)) cmd(kbps == 0 ? "set rp-bitrate 25000" : "set rp-bitrate 0");
      if (kbps > 0) {
        int mbps = std::clamp((kbps + 500) / 1000, 5, 100);
        ImGui::SetNextItemWidth(-FLT_MIN);
        ImGui::SliderInt("##rpbr", &mbps, 5, 100, "%d Mbps");
        if (ImGui::IsItemDeactivatedAfterEdit()) cmd(fmt("set rp-bitrate %d", mbps * 1000));
      }
      dimText("This is only a request: the PlayStation decides the real bitrate itself (it adapts to the network and "
              "usually stays well below this). The actual rate is shown in the Console card and the stats overlay.");
      sectionLabel("DECODER"); restartNote(s, "rp-decoder");
      k = segmented("rpdec", {"NVIDIA (CUDA)", "Software"}, get(c, "rp-decoder", "cuda") == "sw" ? 1 : 0, sc);
      if (k >= 0) cmd(k == 1 ? "set rp-decoder sw" : "set rp-decoder cuda");
      ImGui::Spacing();
      const bool wake = get(c, "rp-wake", "off") == "on";
      if (switchRow("Wake the console when MoonGlide starts", wake, sc)) cmd(wake ? "set rp-wake off" : "set rp-wake on");
    }
    cardEnd(sc);
  }
}

void Ui::sourceCapture(const UiState& s, const Cmd& cmd) {
  const float sc = scale_;
  const Settings& c = s.current;
  if (!capEnum_->devicesReady()) capEnum_->refreshDevices();
  const bool open = CARD("dev", "IN", "Capture device", get(c, "device", "/dev/video0"), nullptr, nullptr, true);
  if (open) {
    const std::string dev = get(c, "device", "/dev/video0");
    const auto devices = capEnum_->devices();
    std::vector<std::string> labels;
    std::string preview = dev;
    for (const auto& d : devices) {
      labels.push_back(d.card + "  (" + d.path + ")");
      if (d.path == dev) preview = labels.back();
    }
    sectionLabel("DEVICE"); restartNote(s, "device");
    ImGui::SameLine();
    if (ImGui::SmallButton("Refresh")) { capEnum_->refreshDevices(); capEnum_->forget(dev); }
    if (labels.empty() && capEnum_->devicesReady()) dimText("No video capture device found. Plug in the card, or type a path below.");
    if (!labels.empty()) {
      const int pick = comboRow("##devpick", preview, labels);
      if (pick >= 0) cmd("set device " + devices[pick].path);
    }
    sectionLabel("CUSTOM PATH");
    char devb[160];
    snprintf(devb, sizeof(devb), "%s", dev.c_str());
    ImGui::SetNextItemWidth(-FLT_MIN);
    if (ImGui::InputText("##dev", devb, sizeof(devb), ImGuiInputTextFlags_EnterReturnsTrue)) cmd(fmt("set device %s", devb));

    const CaptureEnum::Modes modes = capEnum_->modes(dev);
    const bool known = modes.done && modes.ok;
    if (!modes.done) dimText("Reading the device's modes…");
    else if (!modes.ok) dimText("Could not read this device's modes: showing the usual ones.");
    auto upper = [](std::string v) { for (char& ch : v) ch = (char)toupper((unsigned char)ch); return v; };

    const std::string curFmt = get(c, "format", "auto");
    std::vector<std::string> fmtKeys = {"auto"}, fmtLabels = {"Auto"};
    if (known) {
      for (const auto& f : modes.formats) { fmtKeys.push_back(f.key); fmtLabels.push_back(upper(f.key)); }
    } else {
      fmtKeys = {"auto", "mjpeg", "yuyv", "nv12"};
      fmtLabels = {"Auto", "MJPEG", "YUYV", "NV12"};
    }
    if (std::find(fmtKeys.begin(), fmtKeys.end(), curFmt) == fmtKeys.end()) { fmtKeys.push_back(curFmt); fmtLabels.push_back(upper(curFmt) + " (not offered)"); }
    sectionLabel("FORMAT"); restartNote(s, "format");
    int k = comboRow("##fmt", fmtLabels[std::find(fmtKeys.begin(), fmtKeys.end(), curFmt) - fmtKeys.begin()], fmtLabels);
    if (k >= 0) cmd("set format " + fmtKeys[k]);

    std::vector<std::pair<uint32_t, uint32_t>> sizes;
    if (known) {
      for (const auto& f : modes.formats)
        if (curFmt == "auto" || curFmt == f.key)
          for (const auto& sz : f.sizes)
            if (std::find(sizes.begin(), sizes.end(), sz) == sizes.end()) sizes.push_back(sz);
      std::sort(sizes.begin(), sizes.end(), [](auto a, auto b) { return (uint64_t)a.first * a.second > (uint64_t)b.first * b.second; });
    } else {
      sizes = {{3840, 2160}, {2560, 1440}, {1920, 1080}, {1280, 720}};
    }
    const std::string curSize = get(c, "size", "1920x1080");
    std::vector<std::string> sizeKeys, sizeLabels;
    for (const auto& sz : sizes) { sizeKeys.push_back(fmt("%ux%u", sz.first, sz.second)); sizeLabels.push_back(sizeLabel(sizeKeys.back())); }
    if (std::find(sizeKeys.begin(), sizeKeys.end(), curSize) == sizeKeys.end()) {
      sizeKeys.push_back(curSize);
      sizeLabels.push_back(sizeLabel(curSize) + (known ? " (not offered)" : ""));
    }
    sectionLabel("SIZE"); restartNote(s, "size");
    k = comboRow("##size", sizeLabels[std::find(sizeKeys.begin(), sizeKeys.end(), curSize) - sizeKeys.begin()], sizeLabels);
    if (k >= 0) cmd("set size " + sizeKeys[k]);

    std::vector<int> rates;
    if (known) {
      uint32_t w = 0, h = 0;
      if (sscanf(curSize.c_str(), "%ux%u", &w, &h) != 2) w = h = 0;
      for (const auto& f : modes.formats) {
        if (curFmt != "auto" && curFmt != f.key) continue;
        auto it = f.fps.find({w, h});
        if (it != f.fps.end())
          for (int r : it->second)
            if (std::find(rates.begin(), rates.end(), r) == rates.end()) rates.push_back(r);
      }
      std::sort(rates.begin(), rates.end());
    }
    if (rates.empty()) rates = {30, 60, 120};
    std::vector<std::string> rateLabels;
    for (int r : rates) rateLabels.push_back(std::to_string(r) + " fps");
    std::string ratePrev = get(c, "fps", "30") + " fps";
    if (std::find(rateLabels.begin(), rateLabels.end(), ratePrev) == rateLabels.end()) {
      rateLabels.push_back(ratePrev + (known ? " (not offered)" : ""));
      ratePrev = rateLabels.back();
    }
    sectionLabel("FRAME RATE"); restartNote(s, "fps");
    k = comboRow("##rate", ratePrev, rateLabels);
    if (k >= 0) cmd("set fps " + std::to_string(atoi(rateLabels[k].c_str())));

    sectionLabel("DONGLE JPEG QUALITY");
    const std::vector<std::string> qs = {"off", "85", "90", "92"};
    k = segmented("dongleq", {"Off", "85", "90", "92 (exp.)"},
                  (int)(std::find(qs.begin(), qs.end(), get(c, "dongle-quality", "90")) - qs.begin()), sc);
    if (k >= 0) cmd("dongle quality " + qs[k]);
    sectionLabel("STEP DOWN ON CORRUPT FRAMES");
    const std::vector<std::string> au = {"on", "off"};
    k = segmented("dongleauto", {"On (92 > 90 > 85)", "Off (keep it)"}, get(c, "dongle-auto", "on") == "off" ? 1 : 0, sc);
    if (k >= 0) cmd("dongle auto " + au[k]);
    dimText(s.dongleJpeg.empty() ? "MS2109 dongles only (q95 overflows USB2)"
                                 : "MS2109: " + s.dongleJpeg + "  ·  92 is experimental; steps down on corrupt frames");
  }
  cardEnd(sc);
}

void Ui::tabSignal(const UiState& s, const Cmd& cmd) {
  const float sc = scale_;
  const Settings& c = s.current;
  {
    const bool open = CARD("sig", "YC", "Signal", fmt("range %s · dedup %s", get(c, "range", "auto").c_str(), get(c, "dedup", "auto").c_str()),
                           nullptr, nullptr, true);
    if (open) {
      sectionLabel("COLOUR RANGE");
      const std::vector<std::string> ranges = {"auto", "limited", "full"};
      int k = segmented("range", {"Auto", "Limited", "Full"},
                        (int)(std::find(ranges.begin(), ranges.end(), get(c, "range", "auto")) - ranges.begin()), sc);
      if (k >= 0) cmd("range " + ranges[k]);
      sectionLabel("DUPLICATE FRAMES (60 -> 30 CAPTURE)");
      const std::vector<std::string> dd = {"auto", "on", "off"};
      k = segmented("dedup", {"Auto", "On", "Off"},
                    (int)(std::find(dd.begin(), dd.end(), get(c, "dedup", "auto")) - dd.begin()), sc);
      if (k >= 0) cmd("dedup " + dd[k]);
    }
    cardEnd(sc);
  }
}

void Ui::tabDisplay(const UiState& s, const Cmd& cmd) {
  const float sc = scale_;
  const Settings& c = s.current;
  {
    const bool open = CARD("present", "PR", "Presentation", "present mode " + get(c, "present", "fifo"), nullptr, nullptr, true);
    if (open) {
      sectionLabel("PRESENT MODE");
      const std::vector<std::string> modes = {"fifo", "mailbox", "immediate"};
      const int k = segmented("present", {"FIFO (vsync)", "Mailbox", "Immediate"},
                              (int)(std::find(modes.begin(), modes.end(), get(c, "present", "fifo")) - modes.begin()), sc);
      if (k >= 0) cmd("set present " + modes[k]);
      restartNote(s, "present");
      dimText("Fullscreen: Alt+Enter or double-click the picture.");
    }
    cardEnd(sc);
  }
  {
    bool ov = getb(c, "overlay", true), tog = false;
    const bool compact = get(c, "overlay-mode", "full") == "compact";
    const bool open = CARD("hud", "HU", "HUD (performance overlay)",
                           ov ? std::string(compact ? "compact" : "full") + " · " + get(c, "overlay-pos", "tl") : "off", &ov, &tog, true);
    if (tog) cmd(ov ? "overlay off" : "overlay on");
    if (open) {
      dimText("Frame rate, latency and timings drawn over the picture while you play (O toggles it).");
      sectionLabel("POSITION");
      const std::vector<std::string> pos = {"tl", "tr", "bl", "br"};
      int k2 = segmented("ovpos", {"Top left", "Top right", "Bottom left", "Bottom right"},
                         (int)(std::find(pos.begin(), pos.end(), get(c, "overlay-pos", "tl")) - pos.begin()), sc);
      if (k2 >= 0) cmd("overlay pos " + pos[k2]);
      sectionLabel("CONTENT");
      k2 = segmented("ovmode", {"Compact (fps, latency)", "Full"}, compact ? 0 : 1, sc);
      if (k2 >= 0) cmd(k2 == 0 ? "overlay mode compact" : "overlay mode full");
      sectionLabel("SIZE");
      const std::vector<std::string> sz = {"0", "1", "2", "3", "4"};
      k2 = segmented("ovsize", {"Auto", "S", "M", "L", "XL"},
                     (int)(std::find(sz.begin(), sz.end(), get(c, "overlay-size", "0")) - sz.begin()), sc);
      if (k2 >= 0) cmd("overlay size " + sz[k2]);
      sectionLabel("PC STATS (EXTRA LINES)");
      {
        const unsigned mask = (unsigned)atoi(get(c, "overlay-pc", "0").c_str()) & 15u;
        struct Pc { unsigned bit; const char* name; };
        for (const Pc& p : {Pc{1, "GPU: load, temperature, power"}, Pc{2, "CPU: system and MoonGlide"}, Pc{4, "RAM"}, Pc{8, "VRAM"}})
          if (switchRow(p.name, (mask & p.bit) != 0, sc)) cmd(fmt("overlay pc %u", mask ^ p.bit));
      }
      // Preview: the metrics the overlay shows, with live values.
      sectionLabel("SHOWN");
      const UiStats& st = s.stats;
      const bool rpSrc = get(c, "source", "remoteplay") == "remoteplay";
      std::vector<std::string> lines;
      if (compact) {
        lines.push_back(fmt("FPS %.0f   REAL %.0f", st.presentFps, st.realFps));
        lines.push_back(fmt("LATENCY %.1f MS   GPU %.1f", st.e2e, st.gpu));
        if (rpSrc) lines.push_back("INPUT LAG   RTT   BITRATE");
      } else {
        lines.push_back("MOONGLIDE   processing on/off, intensity");
        lines.push_back(fmt("OUT %.1f FPS   REAL %.1f   DUP %llu   DROP %llu", st.presentFps, st.realFps, st.duplicates, st.dropped));
        lines.push_back(fmt("E2E %.1f MS   GPU %.1f   NR %.1f   OF %.1f   DEC %.1f", st.e2e, st.gpu, st.nr, st.flow, st.decode));
        if (rpSrc) { lines.push_back("INPUT LAG   RTT"); lines.push_back("Remote Play bitrate, loss, state"); }
        lines.push_back("capture size -> work size -> output size");
      }
      ImGui::PushFont(F(fonts_.small));
      ImDrawList* dl = ImGui::GetWindowDrawList();
      const float lh = ImGui::GetFontSize() + 3 * sc, pad = 8 * sc;
      const ImVec2 p0 = ImGui::GetCursorScreenPos();
      const float w = ImGui::GetContentRegionAvail().x;
      dl->AddRectFilled(p0, ImVec2(p0.x + w, p0.y + lines.size() * lh + 2 * pad), cPicBg, 6 * sc);
      for (size_t i = 0; i < lines.size(); ++i)
        dl->AddText(ImVec2(p0.x + pad, p0.y + pad + i * lh), cText, lines[i].c_str());
      ImGui::Dummy(ImVec2(w, lines.size() * lh + 2 * pad));
      ImGui::PopFont();
    }
    cardEnd(sc);
  }
  {
    const bool open = CARD("cfg", "CF", "Settings file", s.configWritable ? "saved on every change" : "read-only (another instance)",
                           nullptr, nullptr, true);
    if (open) {
      ImGui::PushStyleColor(ImGuiCol_Text, v4(cDim));
      ImGui::TextWrapped("%s", s.configPath.c_str());
      ImGui::TextWrapped("Options given on the command line override the file.");
      ImGui::PopStyleColor();
    }
    cardEnd(sc);
  }
}

void Ui::tabNvidia(const UiState& s, const Cmd& cmd) {
  const float sc = scale_;
  const Settings& c = s.current;
  if (!nvvfxPathInit_) {
    snprintf(nvvfxPath_, sizeof(nvvfxPath_), "%s", get(c, "nvvfx").empty() ? "/usr/local/VideoFX" : get(c, "nvvfx").c_str());
    nvvfxPathInit_ = true;
  }
  bool on = !get(c, "nvvfx").empty(), tog = false;
  // Optional: only offered when the SDK is configured, running or installed at the default path.
  if (!on && s.nvvfxStages.empty() && access((std::string(nvvfxPath_) + "/lib").c_str(), F_OK) != 0) return;
  const bool open = CARD("nv", "NV", "NVIDIA Video Effects: SDK and quality", s.nvvfxStages.empty() ? "not running" : "running: " + s.nvvfxStages,
                         &on, &tog, false);
  if (tog) cmd(!on ? fmt("set nvvfx %s", nvvfxPath_) : std::string("set nvvfx \"\""));
  if (open) {
    ImGui::PushStyleColor(ImGuiCol_Text, v4(cDim));
    ImGui::TextWrapped("Installed by you (NVIDIA licence), never bundled.");
    ImGui::PopStyleColor();
    restartNote(s, "nvvfx");
    sectionLabel("SDK PATH");
    ImGui::SetNextItemWidth(-FLT_MIN);
    if (ImGui::InputText("##sdk", nvvfxPath_, sizeof(nvvfxPath_), ImGuiInputTextFlags_EnterReturnsTrue) && on)
      cmd(fmt("set nvvfx %s", nvvfxPath_));
    auto seg = [&](const char* title, const char* key, std::vector<std::string> values, std::vector<std::string> labels) {
      sectionLabel(title);
      const int k = segmented(key, labels, (int)(std::find(values.begin(), values.end(), get(c, key)) - values.begin()), sc);
      if (k >= 0) cmd(fmt("set %s %s", key, values[k].c_str()));
      restartNote(s, key);
    };
    seg("DENOISE (VSR QUALITY)", "nvvfx-denoise", {"off", "8", "9", "10", "11"}, {"Off", "8", "9", "10", "11"});
    seg("SUPER RESOLUTION", "nvvfx-sr", {"off", "1", "2", "3", "4"}, {"Off", "1", "2", "3", "4"});
    seg("FRAME GENERATION (VFG)", "nvvfx-fg", {"off", "0", "1", "2"}, {"Off", "0", "1", "2"});
    seg("COLOUR FIX", "nvvfx-colour-fix", {"on", "off"}, {"On", "Off"});
  }
  cardEnd(sc);
}

// ---- bottom panels --------------------------------------------------------------------------

void Ui::bottomPanels(const UiState& s) {
  const float sc = scale_;
  const ImVec2 ds = ImGui::GetIO().DisplaySize;
  const float y0 = ds.y - (kBottom + kStatus) * sc;
  const float totalW = ds.x - kSide * sc - 2 * kGap * sc;
  const float w1 = totalW * 0.21f, wPc = totalW * 0.20f, w2 = totalW * 0.27f, w3 = totalW - w1 - wPc - w2 - 3 * kGap * sc;
  const float h = (kBottom - kGap) * sc;
  const UiStats& st = s.stats;
  auto begin = [&](const char* id, float x, float w, const char* title, const std::string& right) {
    ImGui::SetNextWindowPos(ImVec2(x, y0));
    ImGui::SetNextWindowSize(ImVec2(w, h));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(16 * sc, 12 * sc));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 10 * sc);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.f);
    ImGui::Begin(id, nullptr, kFixed);
    ImGui::PopStyleVar(3);
    ImGui::PushFont(F(fonts_.semibold));
    ImGui::TextUnformatted(title);
    ImGui::PopFont();
    if (!right.empty()) {
      ImGui::PushFont(F(fonts_.small));
      ImGui::SameLine(ImGui::GetContentRegionAvail().x + ImGui::GetCursorStartPos().x - ImGui::CalcTextSize(right.c_str()).x);
      ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 2 * sc);
      ImGui::TextColored(v4(cDim), "%s", right.c_str());
      ImGui::PopFont();
    }
    ImGui::Dummy(ImVec2(1, 2 * sc));
  };

  // Signal.
  float x = kGap * sc;
  begin("##signal", x, w1, "Signal", "");
  ImGui::PushFont(F(fonts_.small));
  if (ImGui::BeginTable("sig", 2, ImGuiTableFlags_SizingStretchProp)) {
    ImGui::TableSetupColumn("k", ImGuiTableColumnFlags_WidthFixed, 72 * sc);
    ImGui::TableSetupColumn("v", ImGuiTableColumnFlags_WidthStretch);
    for (const auto& [k, v] : s.signal) {
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::TextColored(v4(cDim), "%s", k.c_str());
      ImGui::TableNextColumn();
      ImGui::TextUnformatted(v.c_str());
    }
    ImGui::EndTable();
  }
  ImGui::PopFont();
  ImGui::End();

  // PC stats (NVML / /proc, sampled once a second in the background).
  x += w1 + kGap * sc;
  begin("##pc", x, wPc, "PC", "");
  {
    const moonglide::SysStats& y = s.sys;
    std::vector<std::pair<std::string, std::string>> rows;
    if (y.gpuUtil >= 0) {
      std::string g = fmt("%d%%", y.gpuUtil);
      if (y.gpuTempC >= 0) g += fmt("  %d C", y.gpuTempC);
      if (y.gpuPowerW >= 0) g += fmt("  %d W", y.gpuPowerW);
      rows.push_back({"GPU", g});
    }
    if (y.gpuClockMHz >= 0) rows.push_back({"GPU clock", fmt("%d MHz", y.gpuClockMHz)});
    if (y.vramTotalMB > 0) rows.push_back({"VRAM", fmt("%.1f / %.1f GB", y.vramUsedMB / 1024.0, y.vramTotalMB / 1024.0)});
    if (y.vramAppMB >= 0) rows.push_back({"MoonGlide", fmt("%.1f GB VRAM", y.vramAppMB / 1024.0)});
    if (y.cpuPct >= 0) rows.push_back({"CPU", fmt("%.0f%%  (MoonGlide %.0f%%)", y.cpuPct, y.appCpuPct)});
    if (y.ramTotalMB > 0) rows.push_back({"RAM", fmt("%.1f / %.1f GB", y.ramUsedMB / 1024.0, y.ramTotalMB / 1024.0)});
    if (y.appRssMB >= 0) rows.push_back({"MoonGlide", fmt("%.1f GB RAM", y.appRssMB / 1024.0)});
    ImGui::PushFont(F(fonts_.small));
    if (!rows.empty() && ImGui::BeginTable("pcs", 2, ImGuiTableFlags_SizingStretchProp)) {
      ImGui::TableSetupColumn("k", ImGuiTableColumnFlags_WidthFixed, 72 * sc);
      ImGui::TableSetupColumn("v", ImGuiTableColumnFlags_WidthStretch);
      for (const auto& [k, v] : rows) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextColored(v4(cDim), "%s", k.c_str());
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(v.c_str());
      }
      ImGui::EndTable();
    }
    ImGui::PopFont();
  }
  ImGui::End();

  // GPU stage timings.
  x += wPc + kGap * sc;
  begin("##gpu", x, w2, "GPU stage timings", fmt("预算 %.1f 毫秒", st.budget));
  {
    struct Row { const char* name; float ms; bool gpu; };
    std::vector<Row> rows = {{"Decode (CPU)", st.decode, false}, {"Clean", st.clean, true}, {"Denoise (NR)", st.nr, true},
                             {"Optical flow", st.flow, true}};
    if (!s.nvvfxStages.empty()) {
      rows.push_back({"NV denoise", st.nvDenoise, true});
      rows.push_back({"NV upscale", st.nvSr, true});
      rows.push_back({"NV frame gen", st.nvFg, true});
    }
    const float other = std::max(0.f, st.gpu - st.clean - st.nr - st.flow);
    if (s.nvvfxStages.empty()) rows.push_back({"FG + upscale + present", other, true});
    ImGui::PushFont(F(fonts_.small));
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const float labelW = std::min(170 * sc, ImGui::GetContentRegionAvail().x * 0.48f), valW = 70 * sc;
    const float rowH = std::min(22 * sc, (ImGui::GetContentRegionAvail().y - 30 * sc) / (float)(rows.size() + 1));
    auto bar = [&](const char* name, float ms, ImU32 col, bool total) {
      const ImVec2 p = ImGui::GetCursorScreenPos();
      const float w = ImGui::GetContentRegionAvail().x;
      dl->PushClipRect(p, ImVec2(p.x + labelW - 4 * sc, p.y + rowH), true);
      dl->AddText(ImVec2(p.x, p.y + (rowH - ImGui::GetFontSize()) / 2), total ? cText : cDim, name);
      dl->PopClipRect();
      const float bx = p.x + labelW, bw = std::max(8 * sc, w - labelW - valW), bh = 8 * sc, by = p.y + (rowH - bh) / 2;
      dl->AddRectFilled(ImVec2(bx, by), ImVec2(bx + bw, by + bh), cTrack, bh / 2);
      const float f = std::clamp(ms / std::max(1.f, st.budget), 0.f, 1.f);
      if (f > 0) dl->AddRectFilled(ImVec2(bx, by), ImVec2(bx + std::max(bh, bw * f), by + bh), col, bh / 2);
      const std::string v = fmt("%.2f ms", ms);
      const float tw = ImGui::CalcTextSize(v.c_str()).x;
      dl->AddText(ImVec2(p.x + w - tw, p.y + (rowH - ImGui::GetFontSize()) / 2), total ? cText : cDim, v.c_str());
      ImGui::Dummy(ImVec2(w, rowH));
    };
    for (const Row& r : rows) bar(r.name, r.ms, r.gpu ? cAccent : cAlt, false);
    const float frac = st.gpu / std::max(1.f, st.budget);
    bar("GPU total", st.gpu, frac < 0.6f ? cGood : frac < 0.9f ? cWarn : cBad, true);
    ImGui::PopFont();
  }
  ImGui::End();

  // Frame rate and pacing.
  x += w2 + kGap * sc;
  begin("##pace", x, w3, "Frame rate and pacing", fmt("1%% low %.1f", st.low1Fps));
  {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 p = ImGui::GetCursorScreenPos();
    const float w = ImGui::GetContentRegionAvail().x, hh = ImGui::GetContentRegionAvail().y;
    ImFont* big = F(fonts_.big);
    ImFont* small = F(fonts_.small);
    const std::string fps = fmt("%.1f", st.presentFps);
    // Rings: columns as wide as their labels need; the sparkline takes the rest (at least a third of the card).
    const char* ringLabels[3][2] = {{"GPU budget", "GPU"}, {"latency ms", "ms"}, {"flow fail", "fail"}};
    float need = 0;
    for (auto& l : ringLabels) need = std::max(need, small->CalcTextSizeA(small->FontSize, FLT_MAX, 0, l[0]).x + 8 * sc);
    const float rwWant = 3 * need;
    const float sw = std::clamp(w - rwWant - 10 * sc, w * 0.33f, w * 0.52f);
    dl->PushClipRect(p, ImVec2(p.x + w, p.y + hh), true);
    dl->AddText(big, big->FontSize, p, cPearl, fps.c_str());
    const float fw = big->CalcTextSizeA(big->FontSize, FLT_MAX, 0, fps.c_str()).x;
    const std::string realTxt = fmt("%.1f real", st.realFps);
    const float lw = std::max(small->CalcTextSizeA(small->FontSize, FLT_MAX, 0, "fps shown").x,
                              small->CalcTextSizeA(small->FontSize, FLT_MAX, 0, realTxt.c_str()).x);
    if (fw + 8 * sc + lw < sw) {   // only when it doesn't run into the rings
      dl->AddText(small, small->FontSize, ImVec2(p.x + fw + 8 * sc, p.y + big->FontSize * 0.25f), cDim, "fps shown");
      dl->AddText(small, small->FontSize, ImVec2(p.x + fw + 8 * sc, p.y + big->FontSize * 0.25f + small->FontSize + 2 * sc),
                  cDim, realTxt.c_str());
    }
    // Sparkline of real frame intervals.
    const float sx = p.x, sy = p.y + big->FontSize + 14 * sc, sh = std::max(10 * sc, hh - big->FontSize - 22 * sc);
    dl->AddRectFilled(ImVec2(sx, sy), ImVec2(sx + sw, sy + sh), cPicBg, 6 * sc);
    if (st.frameMs.size() > 1) {
      float hi = std::max(st.budget * 2.f, 1.f);
      for (float v : st.frameMs) hi = std::max(hi, v * 1.1f);
      const float by = sy + sh - st.budget / hi * sh;
      dl->AddLine(ImVec2(sx, by), ImVec2(sx + sw, by), IM_COL32(255, 255, 255, 40), 1.f);
      std::vector<ImVec2> pts;
      const size_t n = st.frameMs.size();
      for (size_t i = 0; i < n; ++i)
        pts.push_back(ImVec2(sx + sw * i / (float)(n - 1), sy + sh - std::clamp(st.frameMs[i] / hi, 0.f, 1.f) * (sh - 2 * sc) - sc));
      dl->AddPolyline(pts.data(), (int)pts.size(), cAccent, 0, 1.6f * sc);
    }
    dl->AddText(small, small->FontSize, ImVec2(sx + 6 * sc, sy + 4 * sc), cDim, "frame interval");
    // Rings.
    const float rx0 = sx + sw + 10 * sc, rw = std::max(30 * sc, w - sw - 10 * sc);
    const float cw = rw / 3;
    const float r = std::max(8 * sc, std::min(cw / 2 - 6 * sc, (hh - small->FontSize - 20 * sc) / 2.f - 4 * sc));
    const float cy = p.y + r + 8 * sc;
    auto pick = [&](int i) {   // the full label when it fits its column, else the short one
      return small->CalcTextSizeA(small->FontSize, FLT_MAX, 0, ringLabels[i][0]).x <= cw - 2 * sc ? ringLabels[i][0] : ringLabels[i][1];
    };
    const float gpuFrac = st.gpu / std::max(1.f, st.budget);
    ring(dl, ImVec2(rx0 + cw * 0.5f, cy), r, gpuFrac, gpuFrac < 0.6f ? cGood : gpuFrac < 0.9f ? cWarn : cBad,
         fmt("%.0f%%", gpuFrac * 100).c_str(), pick(0), sc, F(fonts_.semibold), small);
    ring(dl, ImVec2(rx0 + cw * 1.5f, cy), r, st.e2e / 120.f, st.e2e < 60 ? cGood : st.e2e < 90 ? cWarn : cBad,
         fmt("%.0f", st.e2e).c_str(), pick(1), sc, F(fonts_.semibold), small);
    ring(dl, ImVec2(rx0 + cw * 2.5f, cy), r, st.fgFail * 4.f, st.fgFail < 0.05f ? cGood : st.fgFail < 0.2f ? cWarn : cBad,
         fmt("%.1f%%", st.fgFail * 100).c_str(), pick(2), sc, F(fonts_.semibold), small);
    dl->PopClipRect();
  }
  ImGui::End();
}

void Ui::statusLine(const UiState& s, const Cmd& cmd) {
  const float sc = scale_;
  const ImVec2 ds = ImGui::GetIO().DisplaySize;
  ImGui::SetNextWindowPos(ImVec2(0, ds.y - kStatus * sc));
  ImGui::SetNextWindowSize(ImVec2(ds.x, kStatus * sc));
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(16 * sc, 4 * sc));
  ImGui::PushStyleColor(ImGuiCol_WindowBg, v4(cPicBg));
  ImGui::Begin("##status", nullptr, kFixed);
  ImGui::PopStyleColor();
  ImGui::PopStyleVar();
  ImGui::PushFont(F(fonts_.small));
  ImDrawList* dl = ImGui::GetWindowDrawList();
  dl->AddLine(ImGui::GetWindowPos(), ImVec2(ds.x, ImGui::GetWindowPos().y), cBorder);
  const ImVec2 p = ImGui::GetCursorScreenPos();
  const float lh = ImGui::GetFontSize();
  const bool healthy = s.stats.realFps > 1.f;
  dl->AddCircleFilled(ImVec2(p.x + 5 * sc, p.y + lh / 2 + 2 * sc), 4.5f * sc, healthy ? cGood : cBad);
  ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 16 * sc);
  ImGui::AlignTextToFramePadding();
  const UiStats& st = s.stats;
  ImGui::Text("%s  ·  实际 %.1f / 输出 %.1f fps  ·  延迟 %.0f ms（p90 %.0f）  ·  丢帧 %llu  ·  保持 %llu",
              healthy ? "运行中" : "无信号", st.realFps, st.presentFps, st.e2e, st.e2e90, st.dropped, st.fgHeld);
  bool pending = false;
  for (const std::string& k : s.restartKeys) pending |= get(s.current, k.c_str()) != get(s.running, k.c_str());
  const char* hint = "F1 / Tab hide  ·  Ctrl+Q quit";
  float rightW = ImGui::CalcTextSize(hint).x;
  if (pending) rightW += 360 * sc;
  ImGui::SameLine(ds.x - 16 * sc - rightW);
  if (pending) {
    ImGui::TextColored(v4(cWarn), "Some changes apply after a restart");
    ImGui::SameLine();
    ImGui::PushStyleColor(ImGuiCol_Button, v4(cAccent));
    ImGui::PushStyleColor(ImGuiCol_Text, v4(cInk));
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(10 * sc, 1 * sc));
    if (ImGui::SmallButton("Apply & restart")) cmd("restart");
    ImGui::PopStyleVar();
    ImGui::PopStyleColor(2);
    ImGui::SameLine(0, 18 * sc);
  }
  ImGui::TextColored(v4(cDim), "%s", hint);
  ImGui::PopFont();
  ImGui::End();
}

void Ui::hudOverlay(const UiState& s, const Cmd& cmd) {
  const float sc = scale_;
  if (!editingZones()) return;
  ImDrawList* dl = ImGui::GetForegroundDrawList();
  const float rx = (float)s.rectX, ry = (float)s.rectY, rw = (float)s.rectW, rh = (float)s.rectH;
  for (size_t i = 0; i < s.zones.size(); ++i) {
    const HudZone& z = s.zones[i];
    if (z.ellipse) {
      dl->AddEllipse(ImVec2(rx + z.a * rw, ry + z.b * rh), ImVec2(z.c * rw, z.d * rh), cAccent, 0, 64, 2 * sc);
    } else {
      dl->AddRect(ImVec2(rx + z.a * rw, ry + z.b * rh), ImVec2(rx + (z.a + z.c) * rw, ry + (z.b + z.d) * rh), cAccent, 4 * sc, 0, 2 * sc);
    }
    const ImVec2 tl(rx + (z.ellipse ? z.a - z.c : z.a) * rw, ry + (z.ellipse ? z.b - z.d : z.b) * rh);
    const std::string n = fmt("%zu", i + 1);
    dl->AddRectFilled(ImVec2(tl.x, tl.y - 22 * sc), ImVec2(tl.x + 22 * sc, tl.y), cAccent, 4 * sc);
    dl->AddText(ImVec2(tl.x + 7 * sc, tl.y - 20 * sc), cInk, n.c_str());
  }
  ImGuiIO& io = ImGui::GetIO();
  if (drawKind_ < 0) return;
  const bool overPanel = ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow);
  const auto inPicture = [&](ImVec2 m) { return m.x >= rx && m.y >= ry && m.x < rx + rw && m.y < ry + rh; };
  if (!dragging_ && ImGui::IsMouseClicked(0) && !overPanel && inPicture(io.MousePos)) {
    dragging_ = true;
    dragX0_ = io.MousePos.x;
    dragY0_ = io.MousePos.y;
  }
  if (dragging_) {
    const float mx = std::clamp(io.MousePos.x, rx, rx + rw), my = std::clamp(io.MousePos.y, ry, ry + rh);
    const ImVec2 a(std::min(dragX0_, mx), std::min(dragY0_, my)), b(std::max(dragX0_, mx), std::max(dragY0_, my));
    const ImU32 dc = cAccent;
    if (drawKind_ == 1) dl->AddEllipse(ImVec2((a.x + b.x) / 2, (a.y + b.y) / 2), ImVec2((b.x - a.x) / 2, (b.y - a.y) / 2), dc, 0, 64, 2 * sc);
    else dl->AddRect(a, b, dc, 0, 0, 2 * sc);
    if (!ImGui::IsMouseDown(0)) {
      dragging_ = false;
      const float x0 = (a.x - rx) / rw, y0 = (a.y - ry) / rh, x1 = (b.x - rx) / rw, y1 = (b.y - ry) / rh;
      if (x1 - x0 > 0.004f && y1 - y0 > 0.004f) {
        HudZone z;
        z.ellipse = drawKind_ == 1;
        if (z.ellipse) { z.a = (x0 + x1) / 2; z.b = (y0 + y1) / 2; z.c = (x1 - x0) / 2; z.d = (y1 - y0) / 2; }
        else { z.a = x0; z.b = y0; z.c = x1 - x0; z.d = y1 - y0; }
        cmd("hud add " + hudZoneString(z));
      }
      drawKind_ = -1;
    }
  } else if (inPicture(io.MousePos) && !overPanel) {
    dl->AddText(ImVec2(io.MousePos.x + 18 * sc, io.MousePos.y + 18 * sc), cAccent,
                drawKind_ == 1 ? "drag an ellipse" : "drag a rectangle");
  }
}
