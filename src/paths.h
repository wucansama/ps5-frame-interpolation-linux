// Where bundled files live. In an AppImage (or any prefix install) the executable sits in
// <prefix>/bin and its data in <prefix>/share/moonglide; the tap is a separate executable next to it.
// Everything falls back to the compile-time / ~/.local locations, so a plain build is unchanged.
// SPDX-License-Identifier: MIT
#pragma once
#include <limits.h>
#include <stdlib.h>
#include <unistd.h>
#include <string>

namespace moonglide {

inline std::string exeDir() {
  char buf[PATH_MAX];
  const ssize_t n = readlink("/proc/self/exe", buf, sizeof buf - 1);
  if (n <= 0) return {};
  buf[n] = 0;
  std::string p = buf;
  const size_t s = p.rfind('/');
  return s == std::string::npos ? std::string() : p.substr(0, s);
}

// <exe dir>/../share/moonglide/<sub> when it exists, else the build-time directory.
inline std::string dataDir(const char* buildTime, const char* sub) {
  const std::string d = exeDir();
  if (!d.empty()) {
    const std::string c = d + "/../share/moonglide/" + sub;
    if (access(c.c_str(), R_OK | X_OK) == 0) return c;
  }
  return buildTime;
}

// The tap: next to our own executable first, then ~/.local/share/moonglide/bin.
inline std::string defaultTap() {
  const std::string d = exeDir();
  if (!d.empty()) {
    const std::string c = d + "/moonglide-chiaki-tap";
    if (access(c.c_str(), X_OK) == 0) return c;
  }
  const char* home = getenv("HOME");
  return std::string(home ? home : "") + "/.local/share/moonglide/bin/moonglide-chiaki-tap";
}

}  // namespace moonglide
