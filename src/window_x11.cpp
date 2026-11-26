#include "window_x11.h"

#include <X11/Xatom.h>
#include <X11/Xutil.h>

#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {

// The app icon (assets/moonglide.svg) drawn at `n` x `n`: a pearl crescent with two teal echoes on a
// dark rounded tile, as _NET_WM_ICON ARGB (one long per pixel). Coverage by 3x3 supersampling.
void appIcon(int n, std::vector<long>& out) {
  out.push_back(n);
  out.push_back(n);
  const float s = n / 256.f;
  auto rbox = [](float px, float py, float cx, float cy, float hx, float hy, float r) {
    const float qx = std::fabs(px - cx) - (hx - r), qy = std::fabs(py - cy) - (hy - r);
    return std::hypot(std::fmax(qx, 0.f), std::fmax(qy, 0.f)) + std::fmin(std::fmax(qx, qy), 0.f) - r;
  };
  // Inside the crescent (256 units) moved right by `off`, grown by `grow`.
  auto inside = [](float x, float y, float off, float grow) {
    x = (x - 128.f) / 1.15f + 128.f - 8.f - off;     // undo the group transform and the copy offset
    y = (y - 128.f) / 1.15f + 128.f;
    const float R = 64.f + grow, r = 54.f - grow;
    const bool outer = std::hypot(x - 112.f, y - 128.f) <= R;
    const bool in = std::hypot(x - 142.31f, y - 111.83f) <= r;
    const bool side = 50.82f * (y - 64.18f) - 95.3f * (x - 116.9f) >= 0;   // centre side of the chord
    return outer && !in && (side || grow > 0);
  };
  for (int y = 0; y < n; ++y)
    for (int x = 0; x < n; ++x) {
      float ar = 0, ag = 0, ab = 0, aa = 0;
      for (int sy = 0; sy < 3; ++sy)
        for (int sx = 0; sx < 3; ++sx) {
          const float px = (x + (sx + 0.5f) / 3) / s, py = (y + (sy + 0.5f) / 3) / s;
          if (rbox(px, py, 128, 128, 128, 128, 56) > 0) continue;
          float r = 11, g = 20, b = 23;
          const float teal[3] = {95, 208, 196}, pearl[3] = {237, 230, 211};
          auto paint = [&](float off, const float* c) {
            if (inside(px, py, off, 5.f)) r = 11, g = 20, b = 23;   // bg outline
            if (inside(px, py, off, 0.f)) r = c[0], g = c[1], b = c[2];
          };
          if (inside(px, py, -6.f, 0.f)) r = teal[0], g = teal[1], b = teal[2];
          paint(12.f, teal);
          paint(30.f, pearl);
          ar += r, ag += g, ab += b, aa += 1;
        }
      const float k = aa > 0 ? 1.f / aa : 0.f;
      out.push_back(((long)(aa / 9.f * 255) << 24) | ((long)(ar * k) << 16) | ((long)(ag * k) << 8) | (long)(ab * k));
    }
}

}  // namespace

X11Window::X11Window(bool fullscreen, uint32_t width, uint32_t height) {
  display_ = XOpenDisplay(nullptr);
  if (!display_) throw std::runtime_error("cannot open X display (DISPLAY / XAUTHORITY?)");
  const int screen = DefaultScreen(display_);
  Window root = RootWindow(display_, screen);
  screenW_ = (uint32_t)DisplayWidth(display_, screen);
  screenH_ = (uint32_t)DisplayHeight(display_, screen);
  fullscreen_ = fullscreen;
  if (fullscreen) {
    width = screenW_;
    height = screenH_;
  } else if (!width || !height) {   // windowed, no size given: 80% of the screen
    width = screenW_ * 4 / 5 & ~1u;
    height = screenH_ * 4 / 5 & ~1u;
  }
  width_ = width;
  height_ = height;

  XSetWindowAttributes attrs{};
  attrs.background_pixel = BlackPixel(display_, screen);
  attrs.event_mask = KeyPressMask | StructureNotifyMask | ExposureMask;
  window_ = XCreateWindow(display_, root, 0, 0, width, height, 0, CopyFromParent, InputOutput, CopyFromParent,
                          CWBackPixel | CWEventMask, &attrs);
  XStoreName(display_, window_, "MoonGlide");
  // Taskbar / task switcher: UTF-8 title, pid, and the icon.
  {
    const char* title = "MoonGlide";
    XChangeProperty(display_, window_, XInternAtom(display_, "_NET_WM_NAME", False),
                    XInternAtom(display_, "UTF8_STRING", False), 8, PropModeReplace, (const unsigned char*)title, 11);
    long pid = getpid();
    XChangeProperty(display_, window_, XInternAtom(display_, "_NET_WM_PID", False), XA_CARDINAL, 32, PropModeReplace,
                    (unsigned char*)&pid, 1);
    std::vector<long> icon;
    for (int n : {16, 32, 48, 64, 128}) appIcon(n, icon);
    XChangeProperty(display_, window_, XInternAtom(display_, "_NET_WM_ICON", False), XA_CARDINAL, 32, PropModeReplace,
                    (unsigned char*)icon.data(), (int)icon.size());
  }
  XClassHint hint{(char*)"moonglide", (char*)"moonglide"};
  XSetClassHint(display_, window_, &hint);

  deleteAtom_ = XInternAtom(display_, "WM_DELETE_WINDOW", False);
  XSetWMProtocols(display_, window_, &deleteAtom_, 1);

  if (fullscreen) {
    Atom state = XInternAtom(display_, "_NET_WM_STATE", False);
    Atom fs = XInternAtom(display_, "_NET_WM_STATE_FULLSCREEN", False);
    XChangeProperty(display_, window_, state, XA_ATOM, 32, PropModeReplace, (unsigned char*)&fs, 1);
  }
  // Unredirect: a composited window costs a frame of latency on KWin/X11.
  Atom bypass = XInternAtom(display_, "_NET_WM_BYPASS_COMPOSITOR", False);
  long one = 1;
  XChangeProperty(display_, window_, bypass, XA_CARDINAL, 32, PropModeReplace, (unsigned char*)&one, 1);

  // Invisible cursor.
  static char empty[8] = {};
  Pixmap blank = XCreateBitmapFromData(display_, window_, empty, 8, 8);
  XColor black{};
  Cursor cursor = XCreatePixmapCursor(display_, blank, blank, &black, &black, 0, 0);
  XDefineCursor(display_, window_, cursor);
  XFreePixmap(display_, blank);

  XMapRaised(display_, window_);
  XFlush(display_);
  // Wait for the map so the surface has its final size before the swapchain is made.
  for (;;) {
    XEvent e;
    XNextEvent(display_, &e);
    if (e.type == MapNotify) break;
  }
  XSetInputFocus(display_, window_, RevertToParent, CurrentTime);
  XWindowAttributes wa;
  XGetWindowAttributes(display_, window_, &wa);
  width_ = (uint32_t)wa.width;
  height_ = (uint32_t)wa.height;
  fprintf(stderr, "window: %ux%u%s\n", width_, height_, fullscreen ? " fullscreen" : "");
}

void X11Window::setFullscreen(bool on) {
  if (on == fullscreen_) return;
  fullscreen_ = on;
  XEvent e{};
  e.xclient.type = ClientMessage;
  e.xclient.window = window_;
  e.xclient.message_type = XInternAtom(display_, "_NET_WM_STATE", False);
  e.xclient.format = 32;
  e.xclient.data.l[0] = on ? 1 : 0;   // _NET_WM_STATE_ADD / _REMOVE
  e.xclient.data.l[1] = (long)XInternAtom(display_, "_NET_WM_STATE_FULLSCREEN", False);
  e.xclient.data.l[3] = 1;            // from a normal application
  // No window manager (Xvfb tests): nobody acts on the message, so resize the window ourselves.
  Atom check = XInternAtom(display_, "_NET_SUPPORTING_WM_CHECK", True), type = 0;
  int fmt = 0;
  unsigned long n = 0, rest = 0;
  unsigned char* data = nullptr;
  const bool wm = check && XGetWindowProperty(display_, DefaultRootWindow(display_), check, 0, 1, False, XA_WINDOW, &type,
                                              &fmt, &n, &rest, &data) == Success && n == 1;
  if (data) XFree(data);
  if (wm) {
    XSendEvent(display_, DefaultRootWindow(display_), False, SubstructureRedirectMask | SubstructureNotifyMask, &e);
  } else if (on) {
    windowedW_ = width_;
    windowedH_ = height_;
    XMoveResizeWindow(display_, window_, 0, 0, screenW_, screenH_);
  } else {
    XResizeWindow(display_, window_, windowedW_ ? windowedW_ : screenW_ * 4 / 5, windowedH_ ? windowedH_ : screenH_ * 4 / 5);
  }
  XFlush(display_);
  fprintf(stderr, "window: fullscreen %s%s\n", on ? "on" : "off", wm ? "" : " (no window manager: resized)");
}

void X11Window::raise() {
  // _NET_ACTIVE_WINDOW asks the window manager (KWin) to activate it, also across desktops.
  XEvent e{};
  e.xclient.type = ClientMessage;
  e.xclient.window = window_;
  e.xclient.message_type = XInternAtom(display_, "_NET_ACTIVE_WINDOW", False);
  e.xclient.format = 32;
  e.xclient.data.l[0] = 2;   // from a pager / tool: honoured even without a user timestamp
  e.xclient.data.l[1] = CurrentTime;
  XSendEvent(display_, DefaultRootWindow(display_), False, SubstructureRedirectMask | SubstructureNotifyMask, &e);
  XMapRaised(display_, window_);
  XFlush(display_);
}

X11Window::~X11Window() {
  if (window_) XDestroyWindow(display_, window_);
  // No XCloseDisplay: the Vulkan driver registered display-close hooks that are gone by now
  // (segfault on exit). The process is exiting; the server reclaims everything.
}

bool X11Window::poll(const std::function<void(unsigned long)>& onKey, const std::function<bool(XEvent&)>& filter) {
  while (XPending(display_)) {
    XEvent e;
    XNextEvent(display_, &e);
    if (filter && filter(e)) continue;
    switch (e.type) {
      case KeyPress:
        onKey(XLookupKeysym(&e.xkey, 0) | ((e.xkey.state & ControlMask) ? X11Window::kKeyCtrl : 0ul));
        break;
      case ConfigureNotify:
        width_ = (uint32_t)e.xconfigure.width;
        height_ = (uint32_t)e.xconfigure.height;
        break;
      case ClientMessage:
        if ((Atom)e.xclient.data.l[0] == deleteAtom_) return false;
        break;
      default:
        break;
    }
  }
  return true;
}
