// "Install RIFE" from the app: runs scripts/setup-rife.sh (TensorRT from PyPI + the RIFE model, all
// user-local) in the background and keeps its last output line for the UI.
// SPDX-License-Identifier: MIT
#pragma once
#include <glob.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>

namespace moonglide {

class RifeInstaller {
 public:
  enum State { Idle, Running, Done, Failed };

  // TensorRT in the setup venv and at least one exported model.
  static bool filesPresent() {
    const char* home = getenv("HOME");
    const std::string base = std::string(home ? home : "") + "/.local/share/moonglide";
    auto any = [](const std::string& pattern) {
      glob_t g{};
      const bool found = glob(pattern.c_str(), 0, nullptr, &g) == 0 && g.gl_pathc > 0;
      globfree(&g);
      return found;
    };
    const bool trt = getenv("MOONGLIDE_TRT") || any(base + "/rife-venv/lib/python3*/site-packages/tensorrt_libs/libnvinfer.so.10");
    return trt && any(base + "/rife/rife_v4.22_lite_*.onnx");
  }

  // A RIFE start error that means "not installed" (rather than out of memory or a broken engine).
  static bool missingError(const std::string& e) {
    return e.find("not found") != std::string::npos || e.find("no model") != std::string::npos ||
           e.find("libnvinfer") != std::string::npos;
  }

  void start(const std::string& script) {
    std::lock_guard<std::mutex> lock(m_);
    if (state_ == Running) return;
    state_ = Running;
    line_ = "starting";
    const char* home = getenv("HOME");
    log_ = std::string(home ? home : "") + "/.cache/moonglide/setup-rife.log";
    std::thread([this, script] { run(script); }).detach();
  }

  State state() const { std::lock_guard<std::mutex> lock(m_); return state_; }
  std::string line() const { std::lock_guard<std::mutex> lock(m_); return line_; }
  std::string logPath() const { std::lock_guard<std::mutex> lock(m_); return log_; }

 private:
  void run(const std::string& script) {
    const std::string dir = log_.substr(0, log_.rfind('/'));
    const std::string cmd = "mkdir -p '" + dir + "' && '" + script + "' 2>&1 | tee '" + log_ + "'; exit ${PIPESTATUS[0]}";
    FILE* p = popen(("bash -c \"" + escape(cmd) + "\"").c_str(), "r");
    if (!p) { set(Failed, "could not start " + script); return; }
    char buf[512];
    while (fgets(buf, sizeof buf, p)) {
      std::string s = buf;
      while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
      if (s.rfind("== ", 0) == 0) s = s.substr(3);
      if (!s.empty()) set(Running, s);
    }
    const int rc = pclose(p);
    if (rc == 0 && filesPresent()) set(Done, "installed");
    else set(Failed, line().empty() ? "failed" : line());
  }
  void set(State st, const std::string& l) { std::lock_guard<std::mutex> lock(m_); state_ = st; line_ = l; }
  static std::string escape(const std::string& s) {
    std::string o;
    for (char c : s) { if (c == '"' || c == '\\' || c == '$' || c == '`') o += '\\'; o += c; }
    return o;
  }

  mutable std::mutex m_;
  State state_ = Idle;
  std::string line_, log_;
};

}  // namespace moonglide
