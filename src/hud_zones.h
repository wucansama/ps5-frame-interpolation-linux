// HUD / minimap exclusion zones: parts of the picture shown as captured (NR and sharpening
// bypassed). Coordinates are fractions of the picture, so one set works at any resolution.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

struct HudZone {
  bool ellipse = false;
  float a = 0, b = 0, c = 0, d = 0;   // rect: x, y, w, h; ellipse: cx, cy, rx, ry
  float feather = 0;                  // output pixels; 0: rect height / 360
};

// "rect:x,y,w,h" or "ellipse:cx,cy,rx,ry" (fractions 0..1), optional ":feather" in pixels.
bool parseHudZone(const std::string& s, HudZone& out);
std::string hudZoneString(const HudZone& z);

constexpr uint32_t kMaxHudZones = 16;
constexpr uint32_t kHudZoneBufferBytes = 16 + kMaxHudZones * 32;
// Layout matches present.comp's Zones buffer.
void packHudZones(const std::vector<HudZone>& zones, uint32_t rectH, void* out);
