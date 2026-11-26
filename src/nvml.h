// PC stats for the UI, the overlay and the status JSON: GPU load / temperature / clock / power / VRAM through NVML
// (loaded at run time, nothing NVIDIA is bundled or linked), CPU and RAM from /proc. A background thread samples once a
// second; readers only copy the cached values. Anything not available stays -1.
// SPDX-License-Identifier: MIT
#pragma once
#include <dlfcn.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <thread>

namespace moonglide {

struct SysStats {
  int gpuUtil = -1, gpuTempC = -1, gpuClockMHz = -1, gpuPowerW = -1;
  long vramUsedMB = -1, vramTotalMB = -1, vramFreeMB = -1, vramAppMB = -1;
  float cpuPct = -1, appCpuPct = -1;
  long ramUsedMB = -1, ramTotalMB = -1, appRssMB = -1;
};

class SysProbe {
 public:
  ~SysProbe() {
    stop_ = true;
    if (th_.joinable()) th_.join();
  }
  SysStats get() {
    if (!started_.exchange(true)) th_ = std::thread([this] { run(); });
    std::lock_guard<std::mutex> lock(mu_);
    return cur_;
  }

 private:
  struct Util { unsigned gpu, memory; };
  struct Mem { unsigned long long total, free, used; };
  struct Proc { unsigned pid; unsigned long long used; unsigned gi, ci; };
  using UtilFn = int (*)(void*, Util*);
  using MemFn = int (*)(void*, Mem*);
  using ProcFn = int (*)(void*, unsigned*, Proc*);
  using U32Fn = int (*)(void*, unsigned*);

  void openNvml() {
    void* h = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!h) return;
    auto init = (int (*)())dlsym(h, "nvmlInit_v2");
    auto count = (int (*)(unsigned*))dlsym(h, "nvmlDeviceGetCount_v2");
    auto byIdx = (int (*)(unsigned, void**))dlsym(h, "nvmlDeviceGetHandleByIndex_v2");
    utilFn_ = (UtilFn)dlsym(h, "nvmlDeviceGetUtilizationRates");
    memFn_ = (MemFn)dlsym(h, "nvmlDeviceGetMemoryInfo");
    tempFn_ = (int (*)(void*, int, unsigned*))dlsym(h, "nvmlDeviceGetTemperature");
    clockFn_ = (int (*)(void*, int, unsigned*))dlsym(h, "nvmlDeviceGetClockInfo");
    powerFn_ = (U32Fn)dlsym(h, "nvmlDeviceGetPowerUsage");
    for (const char* n : {"nvmlDeviceGetComputeRunningProcesses_v3", "nvmlDeviceGetComputeRunningProcesses_v2"})
      if (!compFn_) compFn_ = (ProcFn)dlsym(h, n);
    for (const char* n : {"nvmlDeviceGetGraphicsRunningProcesses_v3", "nvmlDeviceGetGraphicsRunningProcesses_v2"})
      if (!gfxFn_) gfxFn_ = (ProcFn)dlsym(h, n);
    if (!init || !count || !byIdx || !utilFn_ || !memFn_ || init() != 0) return;
    unsigned n = 0;
    if (count(&n) != 0 || n == 0) return;
    void* first = nullptr;
    for (unsigned i = 0; i < n; ++i) {   // the GPU this process uses (it is listed there); else the first
      void* d = nullptr;
      if (byIdx(i, &d) != 0) continue;
      if (!first) first = d;
      for (auto fn : {compFn_, gfxFn_}) {
        if (!fn) continue;
        Proc p[64];
        unsigned c = 64;
        if (fn(d, &c, p) != 0) continue;
        for (unsigned k = 0; k < c && k < 64; ++k)
          if (p[k].pid == (unsigned)getpid()) { dev_ = d; return; }
      }
    }
    dev_ = first;
  }

  static bool readCpu(unsigned long long& busy, unsigned long long& total) {
    FILE* f = fopen("/proc/stat", "r");
    if (!f) return false;
    unsigned long long v[8] = {};
    const int n = fscanf(f, "cpu %llu %llu %llu %llu %llu %llu %llu %llu", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7]);
    fclose(f);
    if (n < 4) return false;
    total = 0;
    for (int i = 0; i < 8; ++i) total += v[i];
    busy = total - v[3] - v[4];
    return true;
  }
  static bool readSelfTicks(unsigned long long& t) {
    FILE* f = fopen("/proc/self/stat", "r");
    if (!f) return false;
    char buf[1024];
    const size_t n = fread(buf, 1, sizeof buf - 1, f);
    fclose(f);
    buf[n] = 0;
    const char* p = strrchr(buf, ')');
    if (!p) return false;
    unsigned long long ut = 0, st = 0;
    // after ") ": state ppid pgrp session tty tpgid flags minflt cminflt majflt cmajflt utime stime
    if (sscanf(p + 2, "%*c %*d %*d %*d %*d %*d %*u %*u %*u %*u %*u %llu %llu", &ut, &st) != 2) return false;
    t = ut + st;
    return true;
  }

  void run() {
    openNvml();
    unsigned long long pb = 0, pt = 0, pa = 0;
    bool havePrev = readCpu(pb, pt) && readSelfTicks(pa);
    const long page = sysconf(_SC_PAGESIZE);
    while (!stop_) {
      for (int i = 0; i < 10 && !stop_; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(100));
      SysStats s;
      unsigned long long b = 0, t = 0, a = 0;
      if (readCpu(b, t) && readSelfTicks(a)) {
        if (havePrev && t > pt) {
          s.cpuPct = 100.f * (float)(b - pb) / (float)(t - pt);
          s.appCpuPct = 100.f * (float)(a - pa) / (float)(t - pt);
        }
        pb = b; pt = t; pa = a; havePrev = true;
      }
      if (FILE* f = fopen("/proc/meminfo", "r")) {
        char line[128];
        long total = -1, avail = -1;
        while (fgets(line, sizeof line, f)) {
          sscanf(line, "MemTotal: %ld kB", &total);
          sscanf(line, "MemAvailable: %ld kB", &avail);
        }
        fclose(f);
        if (total > 0 && avail >= 0) { s.ramTotalMB = total / 1024; s.ramUsedMB = (total - avail) / 1024; }
      }
      if (FILE* f = fopen("/proc/self/statm", "r")) {
        long sz = 0, rss = 0;
        if (fscanf(f, "%ld %ld", &sz, &rss) == 2) s.appRssMB = rss * page / (1024 * 1024);
        fclose(f);
      }
      if (dev_) {
        Util u{};
        if (utilFn_(dev_, &u) == 0) s.gpuUtil = (int)u.gpu;
        Mem m{};
        if (memFn_(dev_, &m) == 0) { s.vramTotalMB = (long)(m.total >> 20); s.vramFreeMB = (long)(m.free >> 20); s.vramUsedMB = (long)(m.used >> 20); }
        unsigned v = 0;
        if (tempFn_ && tempFn_(dev_, 0, &v) == 0) s.gpuTempC = (int)v;
        if (clockFn_ && clockFn_(dev_, 0, &v) == 0) s.gpuClockMHz = (int)v;
        if (powerFn_ && powerFn_(dev_, &v) == 0) s.gpuPowerW = (int)((v + 500) / 1000);
        std::map<unsigned, unsigned long long> byPid;   // listed as compute and graphics: counted once
        for (auto fn : {compFn_, gfxFn_}) {
          if (!fn) continue;
          Proc p[64];
          unsigned n = 64;
          if (fn(dev_, &n, p) != 0) continue;
          for (unsigned i = 0; i < n && i < 64; ++i) byPid[p[i].pid] = std::max<unsigned long long>(byPid[p[i].pid], p[i].used);
        }
        if (auto it = byPid.find((unsigned)getpid()); it != byPid.end()) s.vramAppMB = (long)(it->second >> 20);
      }
      std::lock_guard<std::mutex> lock(mu_);
      cur_ = s;
    }
  }

  std::mutex mu_;
  std::thread th_;
  std::atomic<bool> started_{false}, stop_{false};
  SysStats cur_;
  void* dev_ = nullptr;
  UtilFn utilFn_ = nullptr;
  MemFn memFn_ = nullptr;
  ProcFn compFn_ = nullptr, gfxFn_ = nullptr;
  int (*tempFn_)(void*, int, unsigned*) = nullptr;
  int (*clockFn_)(void*, int, unsigned*) = nullptr;
  U32Fn powerFn_ = nullptr;
};

inline SysProbe& sysProbe() {
  static SysProbe p;
  return p;
}

}  // namespace moonglide
