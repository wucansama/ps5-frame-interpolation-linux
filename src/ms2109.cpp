#include "ms2109.h"

#include <dirent.h>
#include <fcntl.h>
#include <linux/hidraw.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace {

std::string real(const std::string& p) {
  char buf[PATH_MAX];
  return realpath(p.c_str(), buf) ? std::string(buf) : std::string();
}
std::string parent(const std::string& p) { return p.substr(0, p.rfind('/')); }
std::string readLine(const std::string& p) {
  std::ifstream f(p);
  std::string s;
  std::getline(f, s);
  return s;
}

const uint8_t kLum[64] = {16, 11, 10, 16, 24,  40,  51,  61,  12, 12, 14, 19, 26,  58,  60,  55,
                          14, 13, 16, 24, 40,  57,  69,  56,  14, 17, 22, 29, 51,  87,  80,  62,
                          18, 22, 37, 56, 68,  109, 103, 77,  24, 35, 55, 64, 81,  104, 113, 92,
                          49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99};
const uint8_t kZigzag[64] = {0,  1,  8,  16, 9,  2,  3,  10, 17, 24, 32, 25, 18, 11, 4,  5,
                             12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6,  7,  14, 21, 28,
                             35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
                             58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63};
// The firmware's own table (ROM 0x796B), restored by quality 0.
const uint8_t kRom75[128] = {
    0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08, 0x07, 0x07, 0x07, 0x09, 0x09, 0x08, 0x0a, 0x0c, 0x14,
    0x0d, 0x0c, 0x0b, 0x0b, 0x0c, 0x19, 0x12, 0x13, 0x0f, 0x14, 0x1d, 0x1a, 0x1f, 0x1e, 0x1d, 0x1a,
    0x1c, 0x1c, 0x20, 0x24, 0x2e, 0x27, 0x20, 0x22, 0x2c, 0x23, 0x1c, 0x1c, 0x28, 0x37, 0x29, 0x2c,
    0x30, 0x31, 0x34, 0x34, 0x34, 0x1f, 0x27, 0x39, 0x3d, 0x38, 0x32, 0x3c, 0x2e, 0x33, 0x34, 0x32,
    0x01, 0x09, 0x09, 0x09, 0x0c, 0x0b, 0x0c, 0x18, 0x0d, 0x0d, 0x18, 0x32, 0x21, 0x1c, 0x21, 0x32,
    0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32,
    0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32,
    0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32,
};

void table(int q, uint8_t out[128]) {
  if (q <= 0) {
    memcpy(out, kRom75, 128);
    return;
  }
  uint8_t chr[64];
  for (int i = 0; i < 64; ++i) chr[i] = 99;
  const uint8_t c0[26] = {17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99, 47, 66};
  memcpy(chr, c0, sizeof(c0));
  q = std::clamp(q, 1, 100);
  const int s = q < 50 ? 5000 / q : 200 - 2 * q;
  auto f = [&](int v) { return (uint8_t)std::clamp((v * s + 50) / 100, 1, 255); };
  for (int i = 0; i < 64; ++i) out[i] = f(kLum[kZigzag[i]]);
  // ROM chroma layout quirk: [1] + IJG chroma zigzag[0..62].
  out[64] = 1;
  for (int i = 0; i < 63; ++i) out[65 + i] = f(chr[kZigzag[i]]);
}

void xdataWrite(int fd, uint16_t addr, uint8_t value) {
  uint8_t out[9] = {0, 0xb6, (uint8_t)(addr >> 8), (uint8_t)addr, value, 0, 0, 0, 0};
  if (ioctl(fd, HIDIOCSFEATURE(9), out) < 0) throw std::runtime_error(std::string("ms2109: set feature: ") + strerror(errno));
  uint8_t in[9] = {0};
  if (ioctl(fd, HIDIOCGFEATURE(9), in) < 0) throw std::runtime_error(std::string("ms2109: get feature: ") + strerror(errno));
  if (memcmp(out, in, 4) != 0) throw std::runtime_error("ms2109: unexpected reply to an XDATA write");
}

}  // namespace

std::string ms2109Hidraw(const std::string& videoDevice) {
  const std::string node = videoDevice.substr(videoDevice.rfind('/') + 1);
  const std::string iface = real("/sys/class/video4linux/" + node + "/device");
  if (iface.empty()) return "";
  const std::string usb = parent(iface);
  if (readLine(usb + "/idVendor") != "534d" || readLine(usb + "/idProduct") != "2109") return "";
  DIR* d = opendir("/sys/class/hidraw");
  if (!d) return "";
  std::string found;
  while (dirent* e = readdir(d)) {
    if (e->d_name[0] == '.') continue;
    const std::string dev = real(std::string("/sys/class/hidraw/") + e->d_name + "/device");
    if (!dev.empty() && parent(parent(dev)) == usb) { found = std::string("/dev/") + e->d_name; break; }
  }
  closedir(d);
  return found;
}

void ms2109SetJpegQuality(const std::string& hidraw, int quality) {
  uint8_t t[128];
  table(quality, t);
  const int fd = open(hidraw.c_str(), O_RDWR);
  if (fd < 0) throw std::runtime_error("ms2109: " + hidraw + ": " + strerror(errno) + " (needs the udev rule, see README)");
  try {
    xdataWrite(fd, 0xF526, 0);
    for (uint8_t b : t) xdataWrite(fd, 0xF527, b);
  } catch (...) {
    close(fd);
    throw;
  }
  close(fd);
}
