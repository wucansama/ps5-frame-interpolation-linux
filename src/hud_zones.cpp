#include "hud_zones.h"

#include <cstdio>
#include <cstring>

bool parseHudZone(const std::string& s, HudZone& out) {
  HudZone z;
  char kind[16] = {0};
  float f = 0;
  const int n = sscanf(s.c_str(), "%15[a-z]:%f,%f,%f,%f:%f", kind, &z.a, &z.b, &z.c, &z.d, &f);
  if (n < 5) return false;
  if (!strcmp(kind, "ellipse")) z.ellipse = true;
  else if (strcmp(kind, "rect")) return false;
  if (z.c <= 0 || z.d <= 0) return false;
  z.feather = n == 6 ? f : 0;
  out = z;
  return true;
}

std::string hudZoneString(const HudZone& z) {
  char buf[96];
  snprintf(buf, sizeof(buf), "%s:%.3f,%.3f,%.3f,%.3f", z.ellipse ? "ellipse" : "rect", z.a, z.b, z.c, z.d);
  return buf;
}

void packHudZones(const std::vector<HudZone>& zones, uint32_t rectH, void* out) {
  auto* u = static_cast<uint32_t*>(out);
  const uint32_t count = zones.size() < kMaxHudZones ? (uint32_t)zones.size() : kMaxHudZones;
  u[0] = count;
  u[1] = u[2] = u[3] = 0;
  auto* f = reinterpret_cast<float*>(u + 4);
  for (uint32_t i = 0; i < count; ++i) {
    const HudZone& z = zones[i];
    float* q = f + i * 8;
    q[0] = z.ellipse ? 1.f : 0.f;
    q[1] = z.feather > 0 ? z.feather : (float)rectH / 360.f;
    q[2] = q[3] = 0;
    q[4] = z.a; q[5] = z.b; q[6] = z.c; q[7] = z.d;
  }
}
