#include "config.h"

#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace {

// Boolean settings and the flags that express them.
struct BoolKey {
  const char* key;
  const char* onFlag;    // nullptr: the default (no flag)
  const char* offFlag;
};
const BoolKey kBools[] = {
    {"nr", "--nr", "--no-nr"},
    {"overlay", nullptr, "--no-overlay"},
    {"clean", "--clean", "--no-clean"},
    {"hud-show", "--hud-show", nullptr},
    {"flow", nullptr, "--no-flow"},
};

std::string trim(const std::string& s) {
  const size_t a = s.find_first_not_of(" \t\r");
  if (a == std::string::npos) return "";
  const size_t b = s.find_last_not_of(" \t\r");
  return s.substr(a, b - a + 1);
}

std::string unquote(const std::string& v) {
  if (v.size() >= 2 && v.front() == '"' && v.back() == '"') {
    std::string out;
    for (size_t i = 1; i + 1 < v.size(); ++i) {
      if (v[i] == '\\' && i + 2 < v.size()) ++i;
      out += v[i];
    }
    return out;
  }
  return v;
}

std::string quote(const std::string& v) {
  std::string out = "\"";
  for (char c : v) {
    if (c == '"' || c == '\\') out += '\\';
    out += c;
  }
  return out + "\"";
}

bool isBare(const std::string& v) {
  if (v == "true" || v == "false") return true;
  if (v.empty()) return false;
  char* end = nullptr;
  strtod(v.c_str(), &end);
  return end && *end == 0;
}

void writeSettings(std::ostream& o, const Settings& s) {
  for (const auto& [k, v] : s) {
    if (k == "hud") {
      o << k << " = [";
      std::istringstream in(v);
      std::string z;
      bool first = true;
      while (std::getline(in, z))
        if (!z.empty()) { o << (first ? "" : ", ") << quote(z); first = false; }
      o << "]\n";
    } else {
      o << k << " = " << (isBare(v) ? v : quote(v)) << "\n";
    }
  }
}

}  // namespace

std::string defaultConfigPath() {
  if (const char* p = getenv("MOONGLIDE_CONFIG")) return p;
  const char* xdg = getenv("XDG_CONFIG_HOME");
  const char* home = getenv("HOME");
  const std::string base = xdg && *xdg ? xdg : std::string(home ? home : "") + "/.config";
  return base + "/moonglide/config.toml";
}

bool loadConfig(const std::string& path, ConfigFile& out) {
  out = ConfigFile{};
  out.path = path;
  std::ifstream f(path);
  if (!f) return false;
  Settings* cur = &out.active;
  std::string line;
  int n = 0;
  while (std::getline(f, line)) {
    ++n;
    line = trim(line);
    if (line.empty() || line[0] == '#') continue;
    if (line[0] == '[') {
      // [preset."Name"] or [preset.Name]
      const std::string inner = trim(line.substr(1, line.find(']') - 1));
      if (inner.rfind("preset.", 0) == 0) {
        cur = &out.presets[unquote(trim(inner.substr(7)))];
      } else {
        fprintf(stderr, "config: %s:%d: unknown section [%s]\n", path.c_str(), n, inner.c_str());
        cur = nullptr;
      }
      continue;
    }
    const size_t eq = line.find('=');
    if (eq == std::string::npos || !cur) continue;
    const std::string key = trim(line.substr(0, eq));
    std::string value = trim(line.substr(eq + 1));
    if (key.rfind("kbm", 0) == 0) continue;   // legacy keys of a removed feature: ignored (dropped on the next save)
    if (!value.empty() && value[0] == '[') {
      // Array of strings: joined with '\n'.
      std::string joined;
      size_t i = 1;
      while (i < value.size()) {
        const size_t a = value.find('"', i);
        if (a == std::string::npos) break;
        size_t b = a + 1;
        while (b < value.size() && value[b] != '"') b += value[b] == '\\' ? 2 : 1;
        joined += (joined.empty() ? "" : "\n") + unquote(value.substr(a, b - a + 1));
        i = b + 1;
      }
      (*cur)[key] = joined;
    } else {
      if (!value.empty() && value[0] != '"') value = trim(value.substr(0, value.find('#')));
      (*cur)[key] = unquote(value);
    }
  }
  return true;
}

bool saveConfig(const ConfigFile& cfg) {
  const size_t slash = cfg.path.rfind('/');
  if (slash != std::string::npos) {
    // mkdir -p of the parent.
    std::string dir = cfg.path.substr(0, slash);
    for (size_t i = 1; i <= dir.size(); ++i)
      if (i == dir.size() || dir[i] == '/') mkdir(dir.substr(0, i).c_str(), 0755);
  }
  const std::string tmp = cfg.path + ".tmp";
  {
    std::ofstream o(tmp);
    if (!o) return false;
    o << "# moonglide settings. Written by the app (settings overlay: F1 / Tab); command-line\n"
         "# options override these. Keys are the option names without \"--\".\n";
    writeSettings(o, cfg.active);
    for (const auto& [name, s] : cfg.presets) {
      o << "\n[preset." << quote(name) << "]\n";
      writeSettings(o, s);
    }
    if (!o) return false;
  }
  return rename(tmp.c_str(), cfg.path.c_str()) == 0;
}

std::vector<std::string> settingsToArgs(const Settings& s, const std::vector<std::string>& skip) {
  std::vector<std::string> args;
  for (const auto& [k, v] : s) {
    bool skipped = false;
    for (const std::string& x : skip) skipped |= x == k;
    if (skipped) continue;
    bool handled = false;
    for (const BoolKey& b : kBools) {
      if (k != b.key) continue;
      handled = true;
      const char* flag = v == "true" ? b.onFlag : b.offFlag;
      if (flag) args.push_back(flag);
    }
    if (handled) continue;
    if (k == "hud") {
      std::istringstream in(v);
      std::string z;
      while (std::getline(in, z))
        if (!z.empty()) { args.push_back("--" + k); args.push_back(z); }
      continue;
    }
    if (k == "nvvfx" && v.empty()) continue;   // off
    if (k == "windowed" && v.empty()) continue;   // fullscreen
    args.push_back("--" + k);
    args.push_back(v);
  }
  return args;
}

std::string settingKeyOfFlag(const std::string& flag) {
  if (flag.rfind("--", 0) != 0) return "";
  for (const BoolKey& b : kBools)
    if ((b.onFlag && flag == b.onFlag) || (b.offFlag && flag == b.offFlag)) return b.key;
  if (flag == "--no-dedup") return "dedup";
  if (flag == "--yuv-full-range") return "range";
  if (flag == "--nvvfx-color-fix") return "nvvfx-colour-fix";
  static const char* kNotSettings[] = {"--config", "--no-config", "--control", "--status", "--frames",
                                       "--flow-check", "--clean-debug", "--fg-debug", "--help", "-h"};
  for (const char* x : kNotSettings)
    if (flag == x) return "";
  return flag.substr(2);
}
