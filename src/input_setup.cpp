#include "input_setup.h"

#include "paths.h"

#include <dirent.h>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <sys/ioctl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <set>
#include <thread>

bool parseJsonFlat(const std::string& s, JsonFlat& out) {
  out.clear();
  size_t i = 0;
  auto ws = [&] { while (i < s.size() && isspace((unsigned char)s[i])) ++i; };
  auto str = [&](std::string& r) {
    if (i >= s.size() || s[i] != '"') return false;
    ++i;
    r.clear();
    while (i < s.size() && s[i] != '"') {
      if (s[i] == '\\' && i + 1 < s.size()) {
        ++i;
        switch (s[i]) {
          case 'n': r += '\n'; break;
          case 't': r += '\t'; break;
          case 'r': r += '\r'; break;
          case 'b': r += '\b'; break;
          case 'f': r += '\f'; break;
          case 'u':
            if (i + 4 < s.size()) {
              unsigned cp = (unsigned)strtoul(s.substr(i + 1, 4).c_str(), nullptr, 16);
              i += 4;
              if (cp < 0x80) r += (char)cp;
              else if (cp < 0x800) { r += (char)(0xC0 | cp >> 6); r += (char)(0x80 | (cp & 63)); }
              else { r += (char)(0xE0 | cp >> 12); r += (char)(0x80 | ((cp >> 6) & 63)); r += (char)(0x80 | (cp & 63)); }
            }
            break;
          default: r += s[i];
        }
      } else {
        r += s[i];
      }
      ++i;
    }
    if (i >= s.size()) return false;
    ++i;
    return true;
  };
  ws();
  if (i >= s.size() || s[i] != '{') return false;
  ++i;
  ws();
  if (i < s.size() && s[i] == '}') return true;
  while (i < s.size()) {
    ws();
    std::string key, val;
    if (!str(key)) return false;
    ws();
    if (i >= s.size() || s[i] != ':') return false;
    ++i;
    ws();
    if (i < s.size() && s[i] == '"') {
      if (!str(val)) return false;
    } else if (i < s.size() && (s[i] == '{' || s[i] == '[')) {
      int depth = 0;
      bool inStr = false;
      for (; i < s.size(); ++i) {
        if (inStr) { if (s[i] == '\\') ++i; else if (s[i] == '"') inStr = false; continue; }
        if (s[i] == '"') inStr = true;
        else if (s[i] == '{' || s[i] == '[') ++depth;
        else if (s[i] == '}' || s[i] == ']') { if (--depth == 0) { ++i; break; } }
      }
    } else {
      const size_t a = i;
      while (i < s.size() && s[i] != ',' && s[i] != '}') ++i;
      val = s.substr(a, i - a);
      while (!val.empty() && isspace((unsigned char)val.back())) val.pop_back();
      if (val == "null") val.clear();
    }
    out[key] = val;
    ws();
    if (i < s.size() && s[i] == ',') { ++i; continue; }
    if (i < s.size() && s[i] == '}') return true;
    return false;
  }
  return false;
}

std::vector<std::string> runForLines(const std::vector<std::string>& args, int timeoutMs, int* exitCode,
                                     const std::string* stdinData) {
  std::vector<std::string> lines;
  if (exitCode) *exitCode = -1;
  int fds[2], in[2] = {-1, -1};
  if (args.empty() || pipe(fds) != 0) return lines;
  if (stdinData && pipe(in) != 0) { close(fds[0]); close(fds[1]); return lines; }
  const pid_t pid = fork();
  if (pid < 0) { close(fds[0]); close(fds[1]); if (in[0] >= 0) { close(in[0]); close(in[1]); } return lines; }
  if (pid == 0) {
    dup2(fds[1], 1);
    const int nul = open("/dev/null", O_RDWR);
    if (nul >= 0) { dup2(nul, 0); dup2(nul, 2); }
    if (in[0] >= 0) dup2(in[0], 0);
    for (int fd = 3; fd < 256; ++fd) close(fd);
    std::vector<char*> av;
    for (const std::string& a : args) av.push_back(const_cast<char*>(a.c_str()));
    av.push_back(nullptr);
    execvp(av[0], av.data());
    _exit(127);
  }
  close(fds[1]);
  if (in[0] >= 0) {
    close(in[0]);
    sigset_t blocked, old;   // a child that exited early must not kill us with SIGPIPE
    sigemptyset(&blocked);
    sigaddset(&blocked, SIGPIPE);
    pthread_sigmask(SIG_BLOCK, &blocked, &old);
    const ssize_t w = write(in[1], stdinData->data(), stdinData->size());
    (void)w;
    close(in[1]);
    pthread_sigmask(SIG_SETMASK, &old, nullptr);
  }
  std::string buf;
  const auto t0 = std::chrono::steady_clock::now();
  bool timedOut = false;
  for (;;) {
    const int left = timeoutMs - (int)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    if (left <= 0) { timedOut = true; break; }
    pollfd p{fds[0], POLLIN, 0};
    const int r = poll(&p, 1, std::min(left, 200));
    if (r < 0 && errno != EINTR) break;
    if (r > 0) {
      char tmp[4096];
      const ssize_t n = read(fds[0], tmp, sizeof(tmp));
      if (n <= 0) break;
      buf.append(tmp, (size_t)n);
    }
  }
  close(fds[0]);
  if (timedOut) kill(pid, SIGKILL);
  int status = 0;
  waitpid(pid, &status, 0);
  if (!timedOut && exitCode) *exitCode = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
  size_t a = 0;
  while (a < buf.size()) {
    size_t b = buf.find('\n', a);
    if (b == std::string::npos) b = buf.size();
    std::string l = buf.substr(a, b - a);
    while (!l.empty() && (l.back() == '\r' || isspace((unsigned char)l.back()))) l.pop_back();
    if (!l.empty()) lines.push_back(l);
    a = b + 1;
  }
  return lines;
}

void openInBrowser(const std::string& url) {
  if (url.rfind("http", 0) != 0) return;
  const pid_t pid = fork();
  if (pid != 0) {
    if (pid > 0) waitpid(pid, nullptr, 0);   // the intermediate child exits at once
    return;
  }
  setsid();
  if (fork() != 0) _exit(0);
  const int nul = open("/dev/null", O_RDWR);
  if (nul >= 0) { dup2(nul, 0); dup2(nul, 1); dup2(nul, 2); }
  execlp("xdg-open", "xdg-open", url.c_str(), (char*)nullptr);
  _exit(127);
}

std::string tapBinary(const std::string& configured) {
  if (!configured.empty()) return configured;
  return moonglide::defaultTap();
}

// ---- Remote Play setup -------------------------------------------------------------------------

struct RpSetup::State {
  mutable std::mutex m;
  Snapshot s;
  std::string account, online;
  bool accountReady = false;
};

RpSetup::RpSetup() : st_(std::make_shared<State>()) {}
RpSetup::~RpSetup() = default;

RpSetup::Snapshot RpSetup::snapshot() const {
  std::lock_guard<std::mutex> l(st_->m);
  return st_->s;
}

void RpSetup::scan(const std::string& tap) {
  auto st = st_;
  {
    std::lock_guard<std::mutex> l(st->m);
    if (st->s.scanning) return;
    st->s.scanning = true;
    st->s.scanError.clear();
  }
  std::thread([st, tap] {
    int code = 0;
    const auto lines = runForLines({tap, "--discover-all"}, 20000, &code);
    std::vector<Console> found;
    for (const std::string& l : lines) {
      JsonFlat j;
      if (!parseJsonFlat(l, j) || j["host"].empty()) continue;
      Console c;
      c.host = j["host"];
      c.name = j["name"];
      c.type = j["type"];
      c.state = j["state"];
      c.version = j["system_version"];
      c.hostId = j["host_id"];
      c.registered = j["registered"] == "true";
      found.push_back(c);
    }
    std::lock_guard<std::mutex> l(st->m);
    st->s.scanning = false;
    st->s.scanned = true;
    st->s.consoles = found;
    if (found.empty())
      st->s.scanError = code == 127 ? "moonglide-chiaki-tap not found (see the Remote Play section of the README)"
                                    : code != 0 ? "The scan failed (is moonglide-chiaki-tap up to date?)" : "";
  }).detach();
}

void RpSetup::loginUrl(const std::string& tap) {
  auto st = st_;
  {
    std::lock_guard<std::mutex> l(st->m);
    if (st->s.loginBusy) return;
    st->s.loginBusy = true;
    st->s.loginError.clear();
    st->s.loginUrl.clear();
  }
  std::thread([st, tap] {
    int code = 0;
    const auto lines = runForLines({tap, "--psn-login-url"}, 15000, &code);
    std::string url;
    for (const std::string& l : lines) {
      JsonFlat j;
      if (parseJsonFlat(l, j) && !j["url"].empty()) url = j["url"];
    }
    std::lock_guard<std::mutex> l(st->m);
    st->s.loginBusy = false;
    st->s.loginUrl = url;
    if (url.empty()) st->s.loginError = code == 127 ? "moonglide-chiaki-tap not found" : "Could not get the sign-in address";
  }).detach();
}

std::string RpSetup::takeLoginUrl() {
  std::lock_guard<std::mutex> l(st_->m);
  std::string u;
  u.swap(st_->s.loginUrl);
  return u;
}

void RpSetup::accountFromUrl(const std::string& tap, const std::string& url) {
  auto st = st_;
  {
    std::lock_guard<std::mutex> l(st->m);
    if (st->s.accountBusy) return;
    st->s.accountBusy = true;
    st->s.accountError.clear();
  }
  std::thread([st, tap, url] {
    int code = 0;
    const auto lines = runForLines({tap, "--psn-account-id", url}, 30000, &code);
    std::string acc, online, err;
    for (const std::string& l : lines) {
      JsonFlat j;
      if (!parseJsonFlat(l, j)) continue;
      if (j["ok"] == "true" && !j["account_id"].empty()) { acc = j["account_id"]; online = j["online_id"]; }
      else if (j.count("error")) err = j["error"];
    }
    std::lock_guard<std::mutex> l(st->m);
    st->s.accountBusy = false;
    if (acc.empty()) {
      st->s.accountError = !err.empty() ? err : "Could not read the account from that address. Paste the full address the browser ended on.";
    } else {
      st->account = acc;
      st->online = online;
      st->accountReady = true;
    }
  }).detach();
}

bool RpSetup::takeAccount(std::string& accountId, std::string& onlineId) {
  std::lock_guard<std::mutex> l(st_->m);
  if (!st_->accountReady) return false;
  accountId = st_->account;
  onlineId = st_->online;
  st_->accountReady = false;
  return true;
}

void RpSetup::pair(const std::string& tap, const std::string& host, const std::string& accountId, const std::string& pin, bool ps4) {
  auto st = st_;
  {
    std::lock_guard<std::mutex> l(st->m);
    if (st->s.pairBusy) return;
    st->s.pairBusy = true;
    st->s.pairDone = false;
    st->s.pairMessage.clear();
  }
  std::thread([st, tap, host, accountId, pin, ps4] {
    std::vector<std::string> a = {tap, "--register", "--host", host, "--psn-account-id", accountId, "--pin", "-"};
    if (ps4) a.push_back("--ps4");
    int code = 0;
    const std::string pinLine = pin + "\n";   // on stdin: never in the process list
    const auto lines = runForLines(a, 60000, &code, &pinLine);
    bool ok = false;
    std::string msg, err;
    for (const std::string& l : lines) {
      JsonFlat j;
      if (!parseJsonFlat(l, j)) continue;
      if (j["ok"] == "true") { ok = true; msg = "Paired with " + (j["name"].empty() ? host : j["name"]); }
      else if (j.count("error")) err = j["error"];
    }
    std::lock_guard<std::mutex> l(st->m);
    st->s.pairBusy = false;
    st->s.pairDone = true;
    st->s.pairOk = ok;
    st->s.pairedHost = ok ? host : "";
    st->s.pairMessage = ok ? msg : !err.empty() ? err : "Pairing failed (wrong PIN, or the PIN expired: generate a new one on the console)";
  }).detach();
}

// ---- V4L2 enumeration --------------------------------------------------------------------------

struct CaptureEnum::State {
  mutable std::mutex m;
  bool devicesReady = false, devicesBusy = false;
  std::vector<Device> devices;
  std::map<std::string, Modes> modes;
  std::set<std::string> running;
};

CaptureEnum::CaptureEnum() : st_(std::make_shared<State>()) {}
CaptureEnum::~CaptureEnum() = default;

bool CaptureEnum::devicesReady() const {
  std::lock_guard<std::mutex> l(st_->m);
  return st_->devicesReady;
}
std::vector<CaptureEnum::Device> CaptureEnum::devices() const {
  std::lock_guard<std::mutex> l(st_->m);
  return st_->devices;
}
void CaptureEnum::forget(const std::string& path) {
  std::lock_guard<std::mutex> l(st_->m);
  if (!st_->running.count(path)) st_->modes.erase(path);
}

static int ioctlRetry(int fd, unsigned long req, void* arg) {
  int r;
  do r = ioctl(fd, req, arg); while (r < 0 && errno == EINTR);
  return r;
}

void CaptureEnum::refreshDevices() {
  auto st = st_;
  {
    std::lock_guard<std::mutex> l(st->m);
    if (st->devicesBusy) return;
    st->devicesBusy = true;
  }
  std::thread([st] {
    std::vector<Device> out;
    if (DIR* d = opendir("/dev")) {
      std::vector<std::string> names;
      while (dirent* e = readdir(d))
        if (strncmp(e->d_name, "video", 5) == 0 && e->d_name[5] >= '0' && e->d_name[5] <= '9') names.push_back(e->d_name);
      closedir(d);
      std::sort(names.begin(), names.end(), [](const std::string& a, const std::string& b) {
        return a.size() != b.size() ? a.size() < b.size() : a < b;
      });
      for (const std::string& n : names) {
        const std::string path = "/dev/" + n;
        const int fd = open(path.c_str(), O_RDONLY | O_NONBLOCK);
        if (fd < 0) continue;
        v4l2_capability cap{};
        if (ioctlRetry(fd, VIDIOC_QUERYCAP, &cap) == 0) {
          const uint32_t caps = (cap.capabilities & V4L2_CAP_DEVICE_CAPS) ? cap.device_caps : cap.capabilities;
          if (caps & (V4L2_CAP_VIDEO_CAPTURE | V4L2_CAP_VIDEO_CAPTURE_MPLANE))
            out.push_back({path, std::string((const char*)cap.card, strnlen((const char*)cap.card, sizeof(cap.card)))});
        }
        close(fd);
      }
    }
    std::lock_guard<std::mutex> l(st->m);
    st->devices = out;
    st->devicesReady = true;
    st->devicesBusy = false;
  }).detach();
}

static void addStandardSizes(std::vector<std::pair<uint32_t, uint32_t>>& v, const v4l2_frmsizeenum& fs) {
  static const std::pair<uint32_t, uint32_t> std_[] = {{1280, 720}, {1920, 1080}, {2560, 1440}, {3840, 2160}, {640, 480},
                                                       {1024, 768}, {1280, 1024}, {1600, 900}, {1920, 1200}, {800, 600}};
  for (const auto& s : std_)
    if (s.first >= fs.stepwise.min_width && s.first <= fs.stepwise.max_width && s.second >= fs.stepwise.min_height &&
        s.second <= fs.stepwise.max_height)
      v.push_back(s);
}

static CaptureEnum::Modes enumerate(const std::string& path) {
  CaptureEnum::Modes m;
  m.done = true;
  const int fd = open(path.c_str(), O_RDONLY | O_NONBLOCK);
  if (fd < 0) return m;
  v4l2_capability cap{};
  if (ioctlRetry(fd, VIDIOC_QUERYCAP, &cap) == 0) {
    const uint32_t caps = (cap.capabilities & V4L2_CAP_DEVICE_CAPS) ? cap.device_caps : cap.capabilities;
    const uint32_t type = (caps & V4L2_CAP_VIDEO_CAPTURE) ? V4L2_BUF_TYPE_VIDEO_CAPTURE : V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    if (caps & (V4L2_CAP_VIDEO_CAPTURE | V4L2_CAP_VIDEO_CAPTURE_MPLANE)) {
      for (uint32_t i = 0;; ++i) {
        v4l2_fmtdesc fd_{};
        fd_.index = i;
        fd_.type = type;
        if (ioctlRetry(fd, VIDIOC_ENUM_FMT, &fd_) < 0) break;
        const char* key = fd_.pixelformat == V4L2_PIX_FMT_MJPEG ? "mjpeg" : fd_.pixelformat == V4L2_PIX_FMT_YUYV ? "yuyv"
                          : fd_.pixelformat == V4L2_PIX_FMT_NV12 ? "nv12" : nullptr;
        if (!key) continue;   // formats the capture path can't take
        CaptureEnum::Format f;
        f.key = key;
        for (uint32_t j = 0;; ++j) {
          v4l2_frmsizeenum fs{};
          fs.index = j;
          fs.pixel_format = fd_.pixelformat;
          if (ioctlRetry(fd, VIDIOC_ENUM_FRAMESIZES, &fs) < 0) break;
          if (fs.type == V4L2_FRMSIZE_TYPE_DISCRETE) f.sizes.push_back({fs.discrete.width, fs.discrete.height});
          else { addStandardSizes(f.sizes, fs); break; }
        }
        std::sort(f.sizes.begin(), f.sizes.end(), [](auto a, auto b) { return (uint64_t)a.first * a.second > (uint64_t)b.first * b.second; });
        f.sizes.erase(std::unique(f.sizes.begin(), f.sizes.end()), f.sizes.end());
        for (const auto& sz : f.sizes) {
          std::vector<int> rates;
          for (uint32_t j = 0;; ++j) {
            v4l2_frmivalenum fi{};
            fi.index = j;
            fi.pixel_format = fd_.pixelformat;
            fi.width = sz.first;
            fi.height = sz.second;
            if (ioctlRetry(fd, VIDIOC_ENUM_FRAMEINTERVALS, &fi) < 0) break;
            if (fi.type == V4L2_FRMIVAL_TYPE_DISCRETE) {
              if (fi.discrete.numerator) rates.push_back((int)((fi.discrete.denominator + fi.discrete.numerator / 2) / fi.discrete.numerator));
            } else {
              const double lo = fi.stepwise.max.numerator ? (double)fi.stepwise.max.denominator / fi.stepwise.max.numerator : 1;
              const double hi = fi.stepwise.min.numerator ? (double)fi.stepwise.min.denominator / fi.stepwise.min.numerator : 240;
              for (int r : {24, 25, 30, 50, 60, 75, 90, 100, 120, 144, 240})
                if (r >= lo - 0.5 && r <= hi + 0.5) rates.push_back(r);
              break;
            }
          }
          std::sort(rates.begin(), rates.end(), std::greater<int>());
          rates.erase(std::unique(rates.begin(), rates.end()), rates.end());
          if (!rates.empty()) f.fps[sz] = rates;
        }
        if (!f.sizes.empty()) m.formats.push_back(f);
      }
    }
  }
  close(fd);
  m.ok = !m.formats.empty();
  return m;
}

CaptureEnum::Modes CaptureEnum::modes(const std::string& path) {
  auto st = st_;
  {
    std::lock_guard<std::mutex> l(st->m);
    if (auto it = st->modes.find(path); it != st->modes.end()) return it->second;
    if (st->running.count(path)) return {};
    st->running.insert(path);
  }
  std::thread([st, path] {
    Modes m = enumerate(path);
    std::lock_guard<std::mutex> l(st->m);
    st->modes[path] = m;
    st->running.erase(path);
  }).detach();
  return {};
}
