// A borderless fullscreen X11 window that asks the compositor to get out of the way.
#pragma once
#include <X11/Xlib.h>

#include <cstdint>
#include <functional>

class X11Window {
 public:
  X11Window(bool fullscreen, uint32_t width, uint32_t height);
  ~X11Window();
  Display* display() const { return display_; }
  Window window() const { return window_; }
  uint32_t width() const { return width_; }
  uint32_t height() const { return height_; }
  // Brings the window to the front and gives it the keyboard (desktop launcher re-open).
  void raise();
  // Fullscreen on / off through the window manager (_NET_WM_STATE); the size follows in a
  // ConfigureNotify, so width() / height() change a moment later.
  void setFullscreen(bool on);
  bool fullscreen() const { return fullscreen_; }
  uint32_t screenWidth() const { return screenW_; }
  uint32_t screenHeight() const { return screenH_; }
  // Added to the keysym passed to onKey while Ctrl is held.
  static constexpr unsigned long kKeyCtrl = 1ul << 31;
  // Drains pending events; returns false once the window was closed.
  // `filter` sees every event first; returning true consumes it (the settings overlay).
  bool poll(const std::function<void(unsigned long keysym)>& onKey,
            const std::function<bool(XEvent&)>& filter = nullptr);

 private:
  Display* display_ = nullptr;
  Window window_ = 0;
  Atom deleteAtom_ = 0;
  uint32_t width_ = 0, height_ = 0, screenW_ = 0, screenH_ = 0;
  bool fullscreen_ = false;
  uint32_t windowedW_ = 0, windowedH_ = 0;
};
