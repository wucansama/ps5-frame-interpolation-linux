// MCP server for MoonGlide. See docs/MCP.md. Talks to a running instance only through the control fifo
// (line commands, see main.cpp "onCommand") and the status JSON; links nothing heavy.
#include "mcp.h"

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

// ---- minimal JSON ------------------------------------------------------------------------
struct J {
  enum T { Null, Bool, Num, Str, Arr, Obj } t = Null;
  bool b = false;
  double n = 0;
  std::string s;
  std::vector<J> a;
  std::vector<std::pair<std::string, J>> o;

  static J str(std::string v) { J j; j.t = Str; j.s = std::move(v); return j; }
  static J num(double v) { J j; j.t = Num; j.n = v; return j; }
  static J boolean(bool v) { J j; j.t = Bool; j.b = v; return j; }
  static J arr() { J j; j.t = Arr; return j; }
  static J obj() { J j; j.t = Obj; return j; }
  J& set(const std::string& k, J v) {
    for (auto& kv : o) if (kv.first == k) { kv.second = std::move(v); return *this; }
    o.emplace_back(k, std::move(v));
    return *this;
  }
  J& push(J v) { a.push_back(std::move(v)); return *this; }
  const J* get(const std::string& k) const {
    if (t != Obj) return nullptr;
    for (const auto& kv : o) if (kv.first == k) return &kv.second;
    return nullptr;
  }
};

struct Parser {
  const std::string& s;
  size_t i = 0;
  int depth = 0;
  bool ok = true;
  explicit Parser(const std::string& str) : s(str) {}
  void ws() { while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) i++; }
  static void utf8(std::string& out, unsigned cp) {
    if (cp < 0x80) out += (char)cp;
    else if (cp < 0x800) { out += (char)(0xC0 | cp >> 6); out += (char)(0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) { out += (char)(0xE0 | cp >> 12); out += (char)(0x80 | ((cp >> 6) & 0x3F)); out += (char)(0x80 | (cp & 0x3F)); }
    else { out += (char)(0xF0 | cp >> 18); out += (char)(0x80 | ((cp >> 12) & 0x3F)); out += (char)(0x80 | ((cp >> 6) & 0x3F)); out += (char)(0x80 | (cp & 0x3F)); }
  }
  bool hex4(unsigned& v) {
    if (i + 4 > s.size()) return false;
    v = 0;
    for (int k = 0; k < 4; k++) {
      const char c = s[i++];
      v <<= 4;
      if (c >= '0' && c <= '9') v |= c - '0';
      else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10;
      else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10;
      else return false;
    }
    return true;
  }
  bool parseString(std::string& out) {
    if (i >= s.size() || s[i] != '"') return false;
    i++;
    while (i < s.size()) {
      const char c = s[i++];
      if (c == '"') return true;
      if (c != '\\') { out += c; continue; }
      if (i >= s.size()) return false;
      const char e = s[i++];
      switch (e) {
        case 'n': out += '\n'; break;
        case 't': out += '\t'; break;
        case 'r': out += '\r'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case '/': case '\\': case '"': out += e; break;
        case 'u': {
          unsigned cp;
          if (!hex4(cp)) return false;
          if (cp >= 0xD800 && cp < 0xDC00 && i + 1 < s.size() && s[i] == '\\' && s[i + 1] == 'u') {
            i += 2;
            unsigned lo;
            if (!hex4(lo)) return false;
            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
          }
          utf8(out, cp);
          break;
        }
        default: return false;
      }
    }
    return false;
  }
  J value() {
    J j;
    ws();
    if (i >= s.size() || ++depth > 64) { ok = false; return j; }
    const char c = s[i];
    if (c == '{') {
      i++;
      j.t = J::Obj;
      ws();
      if (i < s.size() && s[i] == '}') i++;
      else for (;;) {
        ws();
        std::string k;
        if (!parseString(k)) { ok = false; break; }
        ws();
        if (i >= s.size() || s[i++] != ':') { ok = false; break; }
        j.o.emplace_back(k, value());
        if (!ok) break;
        ws();
        if (i < s.size() && s[i] == ',') { i++; continue; }
        if (i < s.size() && s[i] == '}') { i++; break; }
        ok = false;
        break;
      }
    } else if (c == '[') {
      i++;
      j.t = J::Arr;
      ws();
      if (i < s.size() && s[i] == ']') i++;
      else for (;;) {
        j.a.push_back(value());
        if (!ok) break;
        ws();
        if (i < s.size() && s[i] == ',') { i++; continue; }
        if (i < s.size() && s[i] == ']') { i++; break; }
        ok = false;
        break;
      }
    } else if (c == '"') {
      j.t = J::Str;
      if (!parseString(j.s)) ok = false;
    } else if (s.compare(i, 4, "true") == 0) { i += 4; j = J::boolean(true); }
    else if (s.compare(i, 5, "false") == 0) { i += 5; j = J::boolean(false); }
    else if (s.compare(i, 4, "null") == 0) { i += 4; }
    else {
      char* end = nullptr;
      const double v = strtod(s.c_str() + i, &end);
      if (end == s.c_str() + i) ok = false;
      else { i = (size_t)(end - s.c_str()); j = J::num(v); }
    }
    depth--;
    return j;
  }
};

bool parseJson(const std::string& text, J& out) {
  Parser p(text);
  out = p.value();
  p.ws();
  return p.ok && p.i == text.size();
}

void dumpStr(std::string& out, const std::string& v) {
  out += '"';
  for (const unsigned char c : v) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) { char b[8]; snprintf(b, sizeof(b), "\\u%04x", c); out += b; }
        else out += (char)c;
    }
  }
  out += '"';
}

void dump(std::string& out, const J& j) {
  switch (j.t) {
    case J::Null: out += "null"; break;
    case J::Bool: out += j.b ? "true" : "false"; break;
    case J::Num: {
      char b[48];
      if (std::isfinite(j.n) && j.n == std::floor(j.n) && std::fabs(j.n) < 1e15) snprintf(b, sizeof(b), "%.0f", j.n);
      else if (std::isfinite(j.n)) snprintf(b, sizeof(b), "%.6g", j.n);
      else snprintf(b, sizeof(b), "null");
      out += b;
      break;
    }
    case J::Str: dumpStr(out, j.s); break;
    case J::Arr:
      out += '[';
      for (size_t k = 0; k < j.a.size(); k++) { if (k) out += ','; dump(out, j.a[k]); }
      out += ']';
      break;
    case J::Obj:
      out += '{';
      for (size_t k = 0; k < j.o.size(); k++) {
        if (k) out += ',';
        dumpStr(out, j.o[k].first);
        out += ':';
        dump(out, j.o[k].second);
      }
      out += '}';
      break;
  }
}

std::string toString(const J& j) { std::string s; dump(s, j); return s; }

// ---- secrets ------------------------------------------------------------------------------
// Never hand out anything that looks like a credential: Remote Play registration, PSN account, keys.
bool secretKey(std::string k) {
  for (char& c : k) c = (char)tolower((unsigned char)c);
  static const char* bad[] = {"psn", "regist", "key", "token", "secret", "passw", "auth", "credential", "account", "morning",
                              "rp-tap", "rp-shm", "cookie", "online_id", "online-id"};
  for (const char* b : bad) if (k.find(b) != std::string::npos) return true;
  return false;
}

void scrub(J& j) {
  if (j.t == J::Obj) {
    std::vector<std::pair<std::string, J>> keep;
    for (auto& kv : j.o) {
      if (secretKey(kv.first)) continue;
      scrub(kv.second);
      keep.push_back(std::move(kv));
    }
    j.o = std::move(keep);
  } else if (j.t == J::Arr) for (J& x : j.a) scrub(x);
}

// ---- state --------------------------------------------------------------------------------
std::string gCtl, gStatus;

struct Err { std::string msg; };

std::string defaultCtl() {
  if (const char* c = getenv("MOONGLIDE_CTL")) return c;
  const char* rt = getenv("XDG_RUNTIME_DIR");
  return (rt && *rt ? std::string(rt) : "/run/user/" + std::to_string(getuid())) + "/moonglide.ctl";
}
std::string defaultStatus() {
  if (const char* c = getenv("MOONGLIDE_STATUS")) return c;
  return "/dev/shm/moonglide.status";
}

// Writes one line to the fifo without ever blocking. Returns "" on success, else the error text.
std::string sendLine(const std::string& line) {
  if (line.find('\n') != std::string::npos || line.find('\r') != std::string::npos) return "command must be one line";
  struct stat st;
  if (stat(gCtl.c_str(), &st) != 0)
    return "MoonGlide is not running (no control fifo at " + gCtl + "). Start it, or pass --control PATH.";
  if (!S_ISFIFO(st.st_mode)) return gCtl + " is not a fifo";
  const int fd = open(gCtl.c_str(), O_WRONLY | O_NONBLOCK);
  if (fd < 0) {
    if (errno == ENXIO) return "MoonGlide is not running (nothing is reading " + gCtl + ").";
    return std::string("cannot open control fifo: ") + strerror(errno);
  }
  const std::string data = line + "\n";
  ssize_t w = write(fd, data.data(), data.size());
  std::string err;
  if (w < 0) err = errno == EAGAIN ? "control fifo is full (MoonGlide is not reading)" : std::string("write failed: ") + strerror(errno);
  else if ((size_t)w != data.size()) err = "short write to the control fifo";
  close(fd);
  return err;
}

double nowSec() {
  timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Reads and parses the status file. age_s = seconds since it was written.
bool readStatus(J& out, double& age, std::string& err) {
  struct stat st;
  if (stat(gStatus.c_str(), &st) != 0) { err = "MoonGlide is not running (no status file at " + gStatus + ")."; return false; }
  age = nowSec() - ((double)st.st_mtim.tv_sec + st.st_mtim.tv_nsec * 1e-9);
  std::ifstream f(gStatus);
  std::stringstream ss;
  ss << f.rdbuf();
  if (!parseJson(ss.str(), out) || out.t != J::Obj) { err = "status file is not valid JSON yet; try again in a second"; return false; }
  scrub(out);
  return true;
}

double numOf(const J& st, const char* k, double def = 0) {
  const J* v = st.get(k);
  return v && v->t == J::Num ? v->n : def;
}
std::string strOf(const J& st, const char* k) {
  const J* v = st.get(k);
  return v && v->t == J::Str ? v->s : "";
}

// ---- settings -----------------------------------------------------------------------------
// key -> how it is applied. kind: 'b' bool (on/off), 'v' "cmd value", 'S' restart-only (set KEY VALUE).
struct Key { const char* name; char kind; const char* cmd; const char* help; };
const Key kKeys[] = {
    {"nr", 'b', "nr", "denoise on/off (true|false)"},
    {"intensity", 'v', "intensity", "denoise intensity 0..1"},
    {"local-structure", 'v', "structure", "NR local structure"},
    {"local-tone", 'v', "tone", "NR local tone"},
    {"skin-structure", 'v', "skin", "NR skin structure"},
    {"clean", 'b', "clean", "cleanup pass on/off"},
    {"clean-strength", 'v', "clean strength", "cleanup strength"},
    {"deblock", 'v', "deblock", "deblock strength >= 0"},
    {"deband", 'v', "deband", "deband 0..8"},
    {"upscale", 'v', "upscale", "fsr | bicubic"},
    {"sharpen", 'v', "sharpen", "sharpen stops >= 0"},
    {"flow", 'b', "flow", "optical flow on/off"},
    {"range", 'v', "range", "auto | limited | full"},
    {"dedup", 'v', "dedup", "auto | on | off"},
    {"dongle-quality", 'v', "dongle quality", "off | 1..94"},
    {"dongle-auto", 'v', "dongle auto", "on | off"},
    {"overlay", 'b', "overlay", "performance overlay on/off"},
    {"overlay-pos", 'v', "overlay pos", "tl | tr | bl | br"},
    {"overlay-size", 'v', "overlay size", "0..6"},
    {"overlay-mode", 'v', "overlay mode", "full | compact"},
    {"overlay-pc", 'v', "overlay pc", "PC stat lines in the overlay, bit mask: 1 GPU, 2 CPU, 4 RAM, 8 VRAM (0..15)"},
    {"wizard", 'v', "wizard", "done | reset (first-run console wizard)"},
    {"hud-show", 'b', "hud show", "show static zones on/off"},
    {"fg", 'v', "fg mode", "off | own | rife | lsfg (live between off/own/rife when set up at start, else restart)"},
    {"fg-mult", 'v', "fg mult", "2..4"},
    {"fg-real-delay", 'v', "fg real-delay", "0..0.5"},
    {"fg-flags", 'v', "fg flags", "own FG flag bits (integer)"},
    {"fg-hold", 'v', "fg hold", ">= 0"},
    // restart-only
    {"source", 'S', "", "v4l2 | remoteplay (restart)"},
    {"rp-host", 'S', "", "Remote Play console host (restart)"},
    {"rp-bitrate", 'S', "", "Remote Play bitrate (restart)"},
    {"rp-codec", 'S', "", "Remote Play codec (restart)"},
    {"rp-decoder", 'S', "", "Remote Play decoder (restart)"},
    {"rp-wake", 'S', "", "on | off (restart)"},
    {"rp-res", 'S', "", "Remote Play resolution (restart)"},
    {"rp-fps", 'S', "", "Remote Play fps (restart)"},
    {"device", 'S', "", "capture device (restart)"},
    {"size", 'S', "", "WxH (restart)"},
    {"fps", 'S', "", "capture fps (restart)"},
    {"format", 'S', "", "capture pixel format (restart)"},
    {"work", 'S', "", "WxH work size (restart)"},
    {"windowed", 'S', "", "WxH (restart)"},
    {"present", 'S', "", "fifo | mailbox | immediate (restart)"},
    {"images", 'S', "", "swapchain images (restart)"},
    {"nvvfx", 'S', "", "NVIDIA Video Effects SDK path (restart)"},
    {"nvvfx-denoise", 'S', "", "off | 0..n (restart)"},
    {"nvvfx-sr", 'S', "", "off | 0..n (restart)"},
    {"nvvfx-fg", 'S', "", "off | 0..n (restart)"},
    {"nvvfx-colour-fix", 'S', "", "on | off (restart)"},
};
const std::set<std::string> kRestart = {"source", "rp-host", "rp-bitrate", "rp-codec", "rp-decoder", "rp-wake", "rp-res", "rp-fps",
                                        "device", "size", "fps", "format", "work", "windowed", "present", "images", "fg",
                                        "nvvfx", "nvvfx-denoise", "nvvfx-sr", "nvvfx-fg", "nvvfx-colour-fix"};

const Key* findKey(const std::string& k) {
  for (const Key& e : kKeys) if (k == e.name) return &e;
  return nullptr;
}

bool safeToken(const std::string& v, bool allowSpace = false) {
  if (v.empty() || v.size() > 200) return false;
  for (const unsigned char c : v)
    if (!(isalnum(c) || c == '.' || c == '_' || c == ':' || c == ',' || c == '/' || c == '=' || c == '@' || c == '+' || c == '-' ||
          (allowSpace && c == ' ')))
      return false;
  return true;
}

// ---- config file (fallback for get_settings) ----------------------------------------------
std::string gConfig;
using Kv = std::vector<std::pair<std::string, std::string>>;

std::string trim(std::string s) {
  const size_t a = s.find_first_not_of(" \t\r\n");
  if (a == std::string::npos) return "";
  s = s.substr(a);
  s.erase(s.find_last_not_of(" \t\r\n") + 1);
  return s;
}

std::string defaultConfig() {
  if (const char* c = getenv("MOONGLIDE_CONFIG")) return c;
  if (const char* x = getenv("XDG_CONFIG_HOME")) if (*x) return std::string(x) + "/moonglide/config.toml";
  const char* h = getenv("HOME");
  return std::string(h ? h : "") + "/.config/moonglide/config.toml";
}

// Top-level `key = value` lines (the active settings) and the names of the [preset."Name"] sections.
// Secret-looking keys are dropped. Returns false when there is no readable file.
bool readConfig(Kv& active, std::vector<std::string>& presets) {
  std::ifstream f(gConfig);
  if (!f) return false;
  std::string line;
  bool top = true;
  while (std::getline(f, line)) {
    line = trim(line);
    if (line.empty() || line[0] == '#') continue;
    if (line[0] == '[') {
      top = false;
      const size_t q1 = line.find('"'), q2 = line.rfind('"');
      if (line.rfind("[preset.", 0) == 0 && q1 != std::string::npos && q2 > q1) presets.push_back(line.substr(q1 + 1, q2 - q1 - 1));
      continue;
    }
    if (!top) continue;
    const size_t eq = line.find('=');
    if (eq == std::string::npos) continue;
    const std::string k = trim(line.substr(0, eq));
    std::string v = trim(line.substr(eq + 1));
    if (v.size() >= 2 && v.front() == '"' && v.back() == '"') v = v.substr(1, v.size() - 2);
    if (!secretKey(k)) active.emplace_back(k, v);
  }
  return true;
}

// ---- tools --------------------------------------------------------------------------------
struct Res {
  bool err = false;
  J body = J::obj();
};
Res fail(const std::string& m) {
  Res r;
  r.err = true;
  r.body.set("error", J::str(m));
  return r;
}

using Args = J;
bool argStr(const Args& a, const char* k, std::string& out) {
  const J* v = a.get(k);
  if (!v) return false;
  if (v->t == J::Str) out = v->s;
  else if (v->t == J::Bool) out = v->b ? "true" : "false";
  else if (v->t == J::Num) { char b[48]; snprintf(b, sizeof(b), "%.6g", v->n); out = b; }
  else return false;
  return true;
}
bool argNum(const Args& a, const char* k, double& out) {
  const J* v = a.get(k);
  if (!v) return false;
  if (v->t == J::Num) { out = v->n; return std::isfinite(out); }
  if (v->t == J::Str) {
    char* e = nullptr;
    out = strtod(v->s.c_str(), &e);
    return e && *e == 0 && !v->s.empty() && std::isfinite(out);
  }
  return false;
}
bool argBool(const Args& a, const char* k, bool& out) {
  const J* v = a.get(k);
  if (!v) return false;
  if (v->t == J::Bool) { out = v->b; return true; }
  if (v->t == J::Str) {
    if (v->s == "true" || v->s == "on" || v->s == "1") { out = true; return true; }
    if (v->s == "false" || v->s == "off" || v->s == "0") { out = false; return true; }
  }
  if (v->t == J::Num) { out = v->n != 0; return true; }
  return false;
}

std::string fnum(double v) { char b[48]; snprintf(b, sizeof(b), "%.4g", v); return b; }

// A list of commands, sent in order; stops at the first error.
Res sendAll(const std::vector<std::string>& cmds, const char* note = nullptr) {
  Res r;
  J sent = J::arr();
  for (const std::string& c : cmds) {
    const std::string e = sendLine(c);
    if (!e.empty()) return fail(e);
    sent.push(J::str(c));
  }
  r.body.set("ok", J::boolean(true));
  r.body.set("sent", sent);
  r.body.set("note", J::str(note ? note : "Applied asynchronously; get_status refreshes about once a second."));
  return r;
}

// Validation of one live setting value; returns "" when fine, else the problem. `v` may be normalised.
std::string checkValue(const std::string& key, std::string& v) {
  const Key* k = findKey(key);
  if (!k) return "unknown key";
  const auto oneOf = [&](std::initializer_list<const char*> l) -> std::string {
    std::string all;
    for (const char* x : l) { if (v == x) return ""; all += std::string(all.empty() ? "" : ", ") + x; }
    return "value must be one of: " + all;
  };
  const auto range = [&](double lo, double hi) -> std::string {
    char* e = nullptr;
    const double d = strtod(v.c_str(), &e);
    if (v.empty() || *e || !std::isfinite(d)) return "value must be a number";
    if (d < lo || d > hi) return "value must be between " + fnum(lo) + " and " + fnum(hi);
    return "";
  };
  if (k->kind == 'b') {
    if (v == "true" || v == "on" || v == "1") v = "true";
    else if (v == "false" || v == "off" || v == "0") v = "false";
    else return "value must be true or false";
    return "";
  }
  if (key == "upscale") return oneOf({"fsr", "bicubic"});
  if (key == "range") return oneOf({"auto", "limited", "full"});
  if (key == "dedup") return oneOf({"auto", "on", "off"});
  if (key == "dongle-auto" || key == "rp-wake" || key == "nvvfx-colour-fix") return oneOf({"on", "off"});
  if (key == "overlay-pos") return oneOf({"tl", "tr", "bl", "br"});
  if (key == "overlay-mode") return oneOf({"full", "compact"});
  if (key == "overlay-pc") return range(0, 15);
  if (key == "fg") return oneOf({"off", "own", "rife", "lsfg"});
  if (key == "source") return oneOf({"v4l2", "remoteplay"});
  if (key == "present") return oneOf({"fifo", "mailbox", "immediate"});
  if (key == "intensity") return range(0, 1);
  if (key == "deband") return range(0, 8);
  if (key == "overlay-size") return range(0, 6);
  if (key == "fg-mult") return range(2, 4);
  if (key == "fg-real-delay") return range(0, 0.5);
  if (key == "dongle-quality") return v == "off" ? "" : range(1, 94);
  if (key == "sharpen" || key == "deblock" || key == "fg-hold" || key == "clean-strength" || key == "local-structure" ||
      key == "local-tone" || key == "skin-structure")
    return range(0, 1000);
  if (key == "fg-flags") return range(0, 4294967295.0);
  if (!safeToken(v) && !(k->kind == 'S' && v.empty())) return "value has characters that are not allowed (letters, digits and . _ : , / = @ + - only)";
  return "";
}

Res setSetting(const std::string& key, std::string v) {
  const Key* k = findKey(key);
  if (!k) {
    std::string all;
    for (const Key& e : kKeys) all += std::string(all.empty() ? "" : ", ") + e.name;
    return fail("unknown setting '" + key + "'. Known keys: " + all);
  }
  const std::string bad = checkValue(key, v);
  if (!bad.empty()) return fail(key + ": " + bad);
  std::string cmd;
  if (k->kind == 'S') cmd = "set " + key + " " + (v.empty() ? "\"\"" : v);
  else if (k->kind == 'b') cmd = std::string(k->cmd) + (v == "true" ? " on" : " off");
  else cmd = std::string(k->cmd) + " " + v;
  const bool restart = kRestart.count(key) > 0;
  Res r = sendAll({cmd}, restart ? (key == "fg" ? "fg switches live between off/own/rife only when own frame generation was set up at start; otherwise it is pending until a restart (command 'restart')."
                                                : "This setting only applies after a restart: it is pending now; apply it with the 'command' tool ('restart').")
                            : nullptr);
  if (!r.err) {
    r.body.set("key", J::str(key));
    r.body.set("value", J::str(v));
    r.body.set("needs_restart", J::boolean(restart));
  }
  return r;
}

// Allowed `command` shapes. Alternatives with |, <num> = a number, <zone> = rect:/ellipse: zone, <name> = a safe word.
const std::vector<std::vector<std::string>> kPatterns = {
    {"nr", "on|off|toggle"}, {"intensity", "<num>"}, {"structure", "<num>"}, {"tone", "<num>"}, {"skin", "<num>"},
    {"split", "on|off|toggle"}, {"ui", "show|hide|toggle"}, {"fullscreen", "on|off|toggle"}, {"fullscreen"},
    {"overlay", "on|off|toggle"}, {"overlay", "pos", "tl|tr|bl|br"}, {"overlay", "size", "<num>"},
    {"overlay", "mode", "full|compact"}, {"overlay", "pc", "<num>"}, {"reset"}, {"flow", "on|off"}, {"range", "auto|limited|full"},
    {"dedup", "auto|on|off"}, {"upscale", "fsr|bicubic"}, {"sharpen", "<num>"}, {"deband", "<num>"}, {"deblock", "<num>"},
    {"clean", "on|off|toggle"}, {"clean", "strength", "<num>"},
    {"fg", "on|off|toggle"}, {"fg", "mode", "off|own|rife|lsfg"}, {"fg", "mult", "<num>"}, {"fg", "flags", "<num>"},
    {"fg", "hold", "<num>"}, {"fg", "real-delay", "<num>"}, {"fg", "debug", "on|off"},
    {"rife", "still|outlier|cut", "<num>"},
    {"hud", "show", "on|off|toggle"}, {"hud", "clear"}, {"hud", "pop"}, {"hud", "add", "<zone>"}, {"hud", "del", "<num>"},
    {"hud", "set", "<num>", "<zone>"},
    {"dongle", "auto", "on|off"}, {"dongle", "quality", "off|<num>"},
    {"restart"}, {"quit"},
};

bool isNum(const std::string& s) {
  if (s.empty()) return false;
  char* e = nullptr;
  const double d = strtod(s.c_str(), &e);
  return !*e && std::isfinite(d);
}
bool isZone(const std::string& s) {
  if (s.rfind("rect:", 0) != 0 && s.rfind("ellipse:", 0) != 0) return false;
  for (const unsigned char c : s) if (!(isalnum(c) || c == ':' || c == ',' || c == '.' || c == '-')) return false;
  return true;
}
bool isWord(const std::string& s) {   // a key / action name: letters, digits, underscore
  if (s.empty() || s.size() > 40) return false;
  for (const unsigned char c : s) if (!(isalnum(c) || c == '_')) return false;
  return true;
}
bool tokenMatches(const std::string& pat, const std::string& tok) {
  size_t pos = 0;
  while (pos <= pat.size()) {
    size_t bar = pat.find('|', pos);
    if (bar == std::string::npos) bar = pat.size();
    const std::string alt = pat.substr(pos, bar - pos);
    if (alt == "<num>" ? isNum(tok) : alt == "<zone>" ? isZone(tok) : alt == "<word>" ? isWord(tok) : alt == tok) return true;
    pos = bar + 1;
  }
  return false;
}
bool commandAllowed(const std::string& line) {
  std::vector<std::string> t;
  std::istringstream in(line);
  for (std::string w; in >> w;) t.push_back(w);
  if (t.empty()) return false;
  for (const auto& p : kPatterns) {
    if (p.size() != t.size()) continue;
    bool ok = true;
    for (size_t i = 0; i < p.size() && ok; i++) ok = tokenMatches(p[i], t[i]);
    if (ok) return true;
  }
  return false;
}

std::string statusSummary(const J& st, double age) {
  char b[512];
  const std::string rp = strOf(st, "rp_state");
  snprintf(b, sizeof(b), "shown %.1f fps (real %.1f), e2e %.1f ms (p90 %.1f), source %s, FG %s, NR %s, range %s; dup %.0f drop %.0f corrupt %.0f%s",
           numOf(st, "present_fps"), numOf(st, "real_fps"), numOf(st, "e2e_ms"), numOf(st, "e2e_p90_ms"),
           rp.empty() ? "capture card" : ("remote play (" + rp + ")").c_str(), strOf(st, "fg").c_str(),
           st.get("nr") && st.get("nr")->b ? "on" : "off", strOf(st, "range").c_str(), numOf(st, "dup"), numOf(st, "drop"),
           numOf(st, "corrupt_frames"), age > 5 ? "; WARNING: status is stale, MoonGlide may have stopped" : "");
  std::string s = b;
  if (st.get("rp_lost") && (numOf(st, "rp_lost") > 0 || numOf(st, "rp_corrupt") > 0))
    s += "; remote play lost " + fnum(numOf(st, "rp_lost")) + " corrupt " + fnum(numOf(st, "rp_corrupt"));
  if (st.get("rp_input_total_ms") && numOf(st, "rp_input_samples") > 0) s += "; input lag " + fnum(numOf(st, "rp_input_total_ms")) + " ms";
  return s;
}

Res getStatus(const Args&) {
  J st;
  double age = 0;
  std::string err;
  if (!readStatus(st, age, err)) return fail(err);
  Res r;
  r.body.set("summary", J::str(statusSummary(st, age)));
  r.body.set("age_s", J::num(std::round(age * 10) / 10));
  r.body.set("stale", J::boolean(age > 5));
  r.body.set("status", st);
  return r;
}

Res getSettings(const Args&) {
  Res r;
  J live = J::obj();
  J st;
  double age = 0;
  std::string err;
  if (readStatus(st, age, err)) {
    for (const char* k : {"nr", "intensity", "flow", "range", "dedup", "fg", "clean", "hud_zones", "nvvfx"})
      if (const J* v = st.get(k)) live.set(k, *v);
    r.body.set("running", J::boolean(age <= 5));
  } else r.body.set("running", J::boolean(false));
  r.body.set("live_from_status", live);
  Kv cfg;
  std::vector<std::string> presets;
  if (readConfig(cfg, presets)) {
    J c = J::obj();
    for (const auto& kv : cfg) c.set(kv.first, J::str(kv.second));
    r.body.set("config_file", J::str(gConfig));
    r.body.set("config", c);
    r.body.set("config_note", J::str("The config file holds the settings MoonGlide last saved (it saves changes within a second or two). A running instance started with --no-config or --config elsewhere may differ; live_from_status is authoritative for what it shows."));
  } else r.body.set("config_file", J::str("(not found: " + gConfig + ")"));
  J rk = J::arr(), ak = J::arr();
  for (const Key& e : kKeys) {
    (kRestart.count(e.name) ? rk : ak).push(J::str(e.name));
  }
  r.body.set("restart_keys", rk);
  r.body.set("live_keys", ak);
  r.body.set("key_help", [] { J h = J::obj(); for (const Key& e : kKeys) h.set(e.name, J::str(e.help)); return h; }());
  return r;
}

bool safeName(const std::string& n) {
  if (n.empty() || n.size() > 60) return false;
  for (const unsigned char c : n) if (!(isalnum(c) || c == ' ' || c == '_' || c == '.' || c == '-')) return false;
  return n.front() != ' ' && n.back() != ' ';
}

Res presetsTool(const Args& a) {
  std::string action, name;
  argStr(a, "action", action);
  argStr(a, "name", name);
  Kv cfg;
  std::vector<std::string> presets;
  const bool haveCfg = readConfig(cfg, presets);
  if (action == "list") {
    Res r;
    J l = J::arr();
    for (const auto& p : presets) l.push(J::str(p));
    r.body.set("presets", l);
    if (!haveCfg) r.body.set("note", J::str("config file not found: " + gConfig));
    return r;
  }
  if (action != "load" && action != "save") return fail("action must be list, load or save");
  if (!safeName(name)) return fail("name is required: letters, digits, space, . _ - only (max 60)");
  if (action == "load" && haveCfg && std::find(presets.begin(), presets.end(), name) == presets.end())
    return fail("no preset named '" + name + "'. Existing: " + [&] { std::string s; for (auto& p : presets) s += (s.empty() ? "" : ", ") + p; return s; }());
  return sendAll({"preset " + action + " " + name},
                 action == "load" ? "Preset loaded live; restart-only keys inside it become pending (use command 'restart')."
                                  : "Saved the current settings as this preset (written to the config file if it is writable).");
}

Res setFg(const Args& a) {
  std::vector<std::string> cmds;
  std::string s;
  double d;
  if (argStr(a, "mode", s)) {
    std::string bad = checkValue("fg", s);
    if (!bad.empty()) return fail("mode: " + bad);
    cmds.push_back("fg mode " + s);
  }
  const struct { const char* arg; const char* key; const char* cmd; } nums[] = {
      {"multiplier", "fg-mult", "fg mult"}, {"flags", "fg-flags", "fg flags"}, {"hold", "fg-hold", "fg hold"},
      {"real_delay", "fg-real-delay", "fg real-delay"}};
  for (const auto& n : nums)
    if (argNum(a, n.arg, d)) {
      std::string v = fnum(d);
      if (!strcmp(n.arg, "multiplier") || !strcmp(n.arg, "flags")) { char b[32]; snprintf(b, sizeof(b), "%.0f", d); v = b; }
      const std::string bad = checkValue(n.key, v);
      if (!bad.empty()) return fail(std::string(n.arg) + ": " + bad);
      cmds.push_back(std::string(n.cmd) + " " + v);
    }
  if (cmds.empty()) return fail("give at least one of mode, multiplier, flags, hold, real_delay");
  return sendAll(cmds, "Frame generation mode switches live only when own FG was set up at start; otherwise it is pending until 'restart'.");
}

Res setDenoise(const Args& a) {
  std::vector<std::string> cmds;
  bool b;
  double d;
  if (argBool(a, "enabled", b)) cmds.push_back(std::string("nr ") + (b ? "on" : "off"));
  if (argBool(a, "clean", b)) cmds.push_back(std::string("clean ") + (b ? "on" : "off"));
  const struct { const char* arg; const char* key; const char* cmd; } nums[] = {
      {"intensity", "intensity", "intensity"}, {"local_structure", "local-structure", "structure"},
      {"local_tone", "local-tone", "tone"}, {"skin_structure", "skin-structure", "skin"},
      {"clean_strength", "clean-strength", "clean strength"}, {"deband", "deband", "deband"}, {"deblock", "deblock", "deblock"}};
  for (const auto& n : nums)
    if (argNum(a, n.arg, d)) {
      std::string v = fnum(d);
      const std::string bad = checkValue(n.key, v);
      if (!bad.empty()) return fail(std::string(n.arg) + ": " + bad);
      cmds.push_back(std::string(n.cmd) + " " + v);
    }
  if (cmds.empty()) return fail("give at least one parameter");
  return sendAll(cmds);
}

Res setUpscale(const Args& a) {
  std::vector<std::string> cmds;
  std::string s;
  double d;
  if (argStr(a, "mode", s)) {
    const std::string bad = checkValue("upscale", s);
    if (!bad.empty()) return fail("mode: " + bad);
    cmds.push_back("upscale " + s);
  }
  if (argNum(a, "sharpen", d)) {
    std::string v = fnum(d);
    const std::string bad = checkValue("sharpen", v);
    if (!bad.empty()) return fail("sharpen: " + bad);
    cmds.push_back("sharpen " + v);
  }
  if (cmds.empty()) return fail("give mode and/or sharpen");
  return sendAll(cmds);
}

std::vector<std::string> gZones;   // zones added through this server (the status only has a count)

Res staticZones(const Args& a) {
  std::string action;
  argStr(a, "action", action);
  if (action == "list") {
    Res r;
    J st;
    double age = 0;
    std::string err;
    if (readStatus(st, age, err)) r.body.set("zones_in_moonglide", J::num(numOf(st, "hud_zones")));
    else return fail(err);
    J l = J::arr();
    for (const auto& z : gZones) l.push(J::str(z));
    r.body.set("added_by_this_server", l);
    r.body.set("note", J::str("MoonGlide's status only reports the zone count; the list shows zones added in this MCP session."));
    return r;
  }
  if (action == "clear") {
    Res r = sendAll({"hud clear"});
    if (!r.err) gZones.clear();
    return r;
  }
  if (action == "pop") {
    Res r = sendAll({"hud pop"});
    if (!r.err && !gZones.empty()) gZones.pop_back();
    return r;
  }
  if (action != "add") return fail("action must be list, add, clear or pop");
  std::string shape = "rect";
  argStr(a, "shape", shape);
  if (shape != "rect" && shape != "ellipse") return fail("shape must be rect or ellipse");
  double x, y, w, h, f = 0;
  if (!argNum(a, "x", x) || !argNum(a, "y", y) || !argNum(a, "w", w) || !argNum(a, "h", h))
    return fail("x, y, w, h are required (fractions of the picture, 0..1; x,y is the top-left corner / the ellipse's bounding box)");
  argNum(a, "feather", f);
  if (x < 0 || y < 0 || x > 1 || y > 1 || w <= 0 || h <= 0 || w > 1 || h > 1 || x + w > 1.0001 || y + h > 1.0001)
    return fail("x, y must be in 0..1, w, h in (0..1], and x+w, y+h at most 1");
  if (f < 0 || f > 1) return fail("feather must be 0..1");
  char z[128];
  snprintf(z, sizeof(z), "%s:%.4f,%.4f,%.4f,%.4f", shape.c_str(), x, y, w, h);
  std::string zone = z;
  if (f > 0) { snprintf(z, sizeof(z), ":%.4f", f); zone += z; }
  std::vector<std::string> cmds = {"hud add " + zone};
  bool show;
  if (argBool(a, "show", show)) cmds.push_back(std::string("hud show ") + (show ? "on" : "off"));
  Res r = sendAll(cmds, "Zone added (max 8 zones). Use show:true to display the zone outlines.");
  if (!r.err) gZones.push_back(zone);
  return r;
}

Res hudTool(const Args& a) {
  std::vector<std::string> cmds;
  bool on;
  double d;
  std::string s;
  if (argBool(a, "on", on)) cmds.push_back(std::string("overlay ") + (on ? "on" : "off"));
  if (argStr(a, "position", s)) {
    std::string bad = checkValue("overlay-pos", s);
    if (!bad.empty()) return fail("position: " + bad);
    cmds.push_back("overlay pos " + s);
  }
  if (argStr(a, "mode", s)) {
    std::string bad = checkValue("overlay-mode", s);
    if (!bad.empty()) return fail("mode: " + bad);
    cmds.push_back("overlay mode " + s);
  }
  if (argNum(a, "size", d)) {
    char b[32];
    snprintf(b, sizeof(b), "%.0f", d);
    std::string v = b;
    const std::string bad = checkValue("overlay-size", v);
    if (!bad.empty()) return fail("size: " + bad);
    cmds.push_back("overlay size " + v);
  }
  if (argNum(a, "pc", d)) {
    char b[32];
    snprintf(b, sizeof(b), "%.0f", d);
    std::string v = b;
    const std::string bad = checkValue("overlay-pc", v);
    if (!bad.empty()) return fail("pc: " + bad);
    cmds.push_back("overlay pc " + v);
  }
  if (cmds.empty()) return fail("give at least one of on, position, mode, size, pc");
  return sendAll(cmds);
}

Res waitForStream(const Args& a) {
  double t = 30;
  argNum(a, "timeout_s", t);
  t = std::clamp(t, 1.0, 300.0);
  const double end = nowSec() + t;
  std::string lastErr = "no status yet";
  for (;;) {
    J st;
    double age = 0;
    std::string err;
    if (readStatus(st, age, err)) {
      if (age <= 5 && numOf(st, "real_fps") > 0.5) {
        Res r;
        r.body.set("streaming", J::boolean(true));
        r.body.set("summary", J::str(statusSummary(st, age)));
        return r;
      }
      lastErr = age > 5 ? "status is stale" : "running, but no frames yet (" + (strOf(st, "rp_state").empty() ? std::string("capture") : strOf(st, "rp_state")) + ")";
    } else lastErr = err;
    if (nowSec() >= end) return fail("timed out after " + fnum(t) + " s: " + lastErr);
    usleep(500000);
  }
}

struct Tool {
  const char* name;
  std::string desc;
  J schema;
  Res (*run)(const Args&);
};

J prop(const char* type, const std::string& desc) { J p = J::obj(); p.set("type", J::str(type)); p.set("description", J::str(desc)); return p; }
J enumProp(std::initializer_list<const char*> vals, const std::string& desc) {
  J p = prop("string", desc);
  J e = J::arr();
  for (const char* v : vals) e.push(J::str(v));
  p.set("enum", e);
  return p;
}
J numProp(double lo, double hi, const std::string& desc) {
  J p = prop("number", desc);
  p.set("minimum", J::num(lo));
  p.set("maximum", J::num(hi));
  return p;
}
J schema(std::initializer_list<std::pair<const char*, J>> props, std::initializer_list<const char*> required = {}) {
  J s = J::obj(), p = J::obj(), r = J::arr();
  s.set("type", J::str("object"));
  for (const auto& kv : props) p.set(kv.first, kv.second);
  s.set("properties", p);
  for (const char* q : required) r.push(J::str(q));
  if (!r.a.empty()) s.set("required", r);
  s.set("additionalProperties", J::boolean(false));
  return s;
}

const std::vector<Tool>& tools() {
  static std::vector<Tool> t;
  if (!t.empty()) return t;
  std::string keys;
  for (const Key& e : kKeys) keys += std::string(keys.empty() ? "" : ", ") + e.name;
  J keyEnum = prop("string", "Setting key.");
  { J e = J::arr(); for (const Key& k : kKeys) e.push(J::str(k.name)); keyEnum.set("enum", e); }
  std::string cmds = "nr on|off|toggle; intensity N; structure N; tone N; skin N; split on|off|toggle; ui show|hide|toggle; fullscreen [on|off|toggle]; "
                     "overlay on|off|toggle | pos tl|tr|bl|br | size N | mode full|compact | pc MASK; reset; flow on|off; range auto|limited|full; dedup auto|on|off; "
                     "upscale fsr|bicubic; sharpen N; deband N; deblock N; clean on|off|toggle | strength N; "
                     "fg on|off|toggle | mode off|own|rife|lsfg | mult N | flags N | hold N | real-delay N | debug on|off; rife still|outlier|cut N; "
                     "hud show on|off|toggle | clear | pop | add ZONE | del N | set N ZONE (ZONE = rect:x,y,w,h[:feather] or ellipse:...); "
                     "dongle auto on|off | quality off|N; restart; quit";
  t = {
      {"get_status", "Read MoonGlide's live status (written once a second): fps shown and real, end-to-end latency, source / Remote Play state, frame-generation mode, drops and errors. Returns a short summary plus the full parsed JSON with any credential-like fields removed.",
       schema({}), getStatus},
      {"get_settings", "Current settings: the live values the status reports (NR, intensity, flow, range, dedup, FG mode, HUD zone count) plus the saved settings from the config file (~/.config/moonglide/config.toml, secret-like keys removed). Also lists which keys need a restart (restart_keys) and which apply live (live_keys).",
       schema({}), getSettings},
      {"set_setting", "Set one setting by key (sent as the matching control command, or `set KEY VALUE` for restart-only keys). Live keys apply at once. Restart-only keys (source, rp-*, device, size, fps, format, work, windowed, present, images, fg when own FG was not set up at start, nvvfx*) are only pending until the `command` tool sends `restart`; the result says needs_restart. Keys: " + keys + ".",
       schema({{"key", keyEnum}, {"value", prop("string", "New value as text; booleans true|false, numbers as digits. Ranges/enums are checked (see get_settings key_help).")}}, {"key", "value"}), nullptr},
      {"command", "Send one allowlisted control command to MoonGlide (arbitrary strings are rejected). Allowed shapes: " + cmds + ". NOTE: `quit` stops MoonGlide (the stream ends) and `restart` re-executes it with the current settings (a few seconds of no picture); use them only when asked.",
       schema({{"command", prop("string", "One command line, e.g. \"nr off\" or \"overlay pos br\".")}}, {"command"}), nullptr},
      {"presets", "Named setting presets (stored in the config file). list: names; load: apply a preset live (restart-only keys inside it become pending); save: store the current settings under a name. Deleting is not exposed.",
       schema({{"action", enumProp({"list", "load", "save"}, "What to do.")}, {"name", prop("string", "Preset name (letters, digits, space, . _ -). Required for load and save.")}}, {"action"}), presetsTool},
      {"set_frame_generation", "Frame generation: mode (off = none, own = built-in flow based, rife = RIFE neural, lsfg = lsfg-vk), multiplier 2..4, flags (own-FG bit mask), hold (own FG: hold the real frame when this fraction of the flow fails, 0 = off), real_delay 0..0.5. Mode switches live only if own FG was set up at start; otherwise it is pending until restart.",
       schema({{"mode", enumProp({"off", "own", "rife", "lsfg"}, "Frame generation backend.")}, {"multiplier", numProp(2, 4, "Frames per real frame.")},
               {"flags", prop("integer", "Own-FG flag bits: 2 hard pick, 4 dilate, 8 clamp to source range, 32 static guard, 64 outlier rejection, 128 flow median.")},
               {"hold", numProp(0, 1, "Hold fraction, 0 = off.")}, {"real_delay", numProp(0, 0.5, "Real frame delay.")}}), setFg},
      {"set_denoise", "Denoise (NR) and cleanup: enabled, intensity 0..1, local_structure, local_tone, skin_structure, clean (cleanup pass) with clean_strength, deband 0..8, deblock strength.",
       schema({{"enabled", prop("boolean", "Denoise on/off.")}, {"intensity", numProp(0, 1, "Denoise intensity.")}, {"local_structure", prop("number", "Local structure.")},
               {"local_tone", prop("number", "Local tone.")}, {"skin_structure", prop("number", "Skin structure.")}, {"clean", prop("boolean", "Cleanup pass on/off.")},
               {"clean_strength", prop("number", "Cleanup strength.")}, {"deband", numProp(0, 8, "Deband strength.")}, {"deblock", prop("number", "Deblock strength (>= 0).")}}), setDenoise},
      {"set_upscale", "Upscaler: mode fsr (FSR) or bicubic, and sharpen strength in stops (>= 0).",
       schema({{"mode", enumProp({"fsr", "bicubic"}, "Upscaler.")}, {"sharpen", prop("number", "Sharpen stops, >= 0.")}}), setUpscale},
      {"static_zones", "Static HUD zones that are kept out of denoise/frame generation (e.g. a fixed game HUD). list: count in MoonGlide and zones added by this server; add: a rect or ellipse in fractions of the picture (x, y top-left, w, h size, 0..1); pop: remove the last; clear: remove all (max 8 zones).",
       schema({{"action", enumProp({"list", "add", "pop", "clear"}, "What to do.")}, {"shape", enumProp({"rect", "ellipse"}, "For add; default rect.")},
               {"x", numProp(0, 1, "Left edge as a fraction of width.")}, {"y", numProp(0, 1, "Top edge as a fraction of height.")},
               {"w", numProp(0, 1, "Width fraction.")}, {"h", numProp(0, 1, "Height fraction.")}, {"feather", numProp(0, 1, "Edge softness, optional.")},
               {"show", prop("boolean", "Also show / hide the zone outlines on the picture.")}}, {"action"}), staticZones},
      {"hud", "The performance overlay (fps, latency, stage timings): on/off, position (tl top-left, tr, bl, br), mode (full or compact), size 0..6, pc (PC stat lines, bit mask 0..15: 1 GPU, 2 CPU, 4 RAM, 8 VRAM).",
       schema({{"on", prop("boolean", "Overlay on/off.")}, {"position", enumProp({"tl", "tr", "bl", "br"}, "Corner.")},
               {"mode", enumProp({"full", "compact"}, "Detail level.")}, {"size", numProp(0, 6, "Text size step.")},
               {"pc", numProp(0, 15, "PC stat lines mask: 1 GPU, 2 CPU, 4 RAM, 8 VRAM.")}}), hudTool},
      {"wait_for_stream", "Wait until MoonGlide is running and frames flow (fresh status with real fps > 0), up to timeout_s (default 30, max 300). Useful right after starting or restarting it.",
       schema({{"timeout_s", numProp(1, 300, "Seconds to wait.")}}), waitForStream},
  };
  return t;
}

Res callTool(const std::string& name, const Args& a) {
  if (name == "set_setting") {
    std::string k, v;
    if (!argStr(a, "key", k)) return fail("key is required");
    if (!argStr(a, "value", v)) return fail("value is required");
    return setSetting(k, v);
  }
  if (name == "command") {
    std::string c;
    if (!argStr(a, "command", c)) return fail("command is required");
    c = trim(c);
    if (!commandAllowed(c)) return fail("command not allowed: '" + c.substr(0, 80) + "'. See the tool description for the allowed shapes.");
    Res r = sendAll({c}, c == "quit" ? "MoonGlide is quitting." : c == "restart" ? "MoonGlide is restarting; use wait_for_stream." : nullptr);
    return r;
  }
  for (const Tool& t : tools()) if (name == t.name && t.run) return t.run(a);
  return fail("unknown tool");
}

// ---- JSON-RPC ----------------------------------------------------------------------------
void send(const J& msg) {
  std::string s = toString(msg);
  s += '\n';
  fwrite(s.data(), 1, s.size(), stdout);
  fflush(stdout);
}
void reply(const J& id, J result) {
  J m = J::obj();
  m.set("jsonrpc", J::str("2.0"));
  m.set("id", id);
  m.set("result", std::move(result));
  send(m);
}
void rpcError(const J& id, int code, const std::string& msg) {
  J m = J::obj(), e = J::obj();
  m.set("jsonrpc", J::str("2.0"));
  m.set("id", id);
  e.set("code", J::num(code));
  e.set("message", J::str(msg));
  m.set("error", e);
  send(m);
}

void handle(const J& req) {
  const J* idp = req.get("id");
  const J* mp = req.get("method");
  if (req.t != J::Obj || !mp || mp->t != J::Str) {
    if (idp) rpcError(*idp, -32600, "invalid request");
    else rpcError(J(), -32600, "invalid request");
    return;
  }
  const std::string& m = mp->s;
  if (!idp) return;   // notification (initialized, cancelled, ...)
  const J id = *idp;
  const J* params = req.get("params");
  const J empty = J::obj();
  if (!params || params->t != J::Obj) params = &empty;
  if (m == "initialize") {
    static const char* supported[] = {"2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"};
    std::string ver = "2025-06-18";
    if (const J* pv = params->get("protocolVersion"))
      for (const char* s : supported) if (pv->t == J::Str && pv->s == s) ver = s;
    J r = J::obj(), caps = J::obj(), tl = J::obj(), info = J::obj();
    tl.set("listChanged", J::boolean(false));
    caps.set("tools", tl);
    info.set("name", J::str("moonglide"));
    info.set("version", J::str("1.0"));
    r.set("protocolVersion", J::str(ver));
    r.set("capabilities", caps);
    r.set("serverInfo", info);
    r.set("instructions", J::str("Reads and controls a running MoonGlide (console stream -> denoise -> frame generation -> upscale). Start with get_status. Tools fail with 'MoonGlide is not running' when no instance reads the control fifo."));
    reply(id, r);
  } else if (m == "ping") {
    reply(id, J::obj());
  } else if (m == "tools/list") {
    J l = J::arr();
    for (const Tool& t : tools()) {
      J o = J::obj();
      o.set("name", J::str(t.name));
      o.set("description", J::str(t.desc));
      o.set("inputSchema", t.schema);
      l.push(o);
    }
    J r = J::obj();
    r.set("tools", l);
    reply(id, r);
  } else if (m == "tools/call") {
    const J* np = params->get("name");
    if (!np || np->t != J::Str) { rpcError(id, -32602, "missing tool name"); return; }
    bool known = false;
    for (const Tool& t : tools()) known |= np->s == t.name;
    if (!known) { rpcError(id, -32602, "unknown tool: " + np->s); return; }
    const J* ap = params->get("arguments");
    Res res = callTool(np->s, ap && ap->t == J::Obj ? *ap : empty);
    J r = J::obj(), c = J::arr(), item = J::obj();
    item.set("type", J::str("text"));
    item.set("text", J::str(toString(res.body)));
    c.push(item);
    r.set("content", c);
    if (res.err) r.set("isError", J::boolean(true));
    reply(id, r);
  } else {
    rpcError(id, -32601, "method not found: " + m);
  }
}

}  // namespace

int runMcp(int argc, char** argv) {
  signal(SIGPIPE, SIG_IGN);
  gCtl = defaultCtl();
  gStatus = defaultStatus();
  gConfig = defaultConfig();
  for (int i = 2; i < argc; i++) {
    const std::string a = argv[i];
    const auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "--control") gCtl = next();
    else if (a == "--status") gStatus = next();
    else if (a == "--config") gConfig = next();
    else {
      fprintf(stderr, "usage: moonglide mcp [--control PATH] [--status PATH] [--config PATH]\n"
                      "MCP server over stdio for a running MoonGlide (see docs/MCP.md).\n");
      return a == "-h" || a == "--help" ? 0 : 2;
    }
  }
  std::string line;
  while (std::getline(std::cin, line)) {
    if (trim(line).empty()) continue;
    J req;
    if (!parseJson(line, req)) { rpcError(J(), -32700, "parse error"); continue; }
    handle(req);
  }
  return 0;
}
