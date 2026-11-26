// moonglide-rp-fake: a stand-in PS5 for the Remote Play source. Streams an Annex B HEVC / H.264
// elementary file (split into access units here) through the frame tap (rp_tap.h) at a steady frame rate,
// the way moonglide-chiaki-tap does, with optional simulated network loss. For tests on Xvfb.
//
//   moonglide-rp-fake clip.hevc [--shm NAME] [--fps 60] [--loss 0.01] [--seconds N] [--loop] [--h264]
//
// Make a clip that behaves like a 30 fps game on Remote Play (60 Hz stream, every picture twice,
// the repeats tiny P frames):
//   ffmpeg -i in.mp4 -vf fps=30,fps=60 -c:v hevc_nvenc -preset p1 -tune ll -bf 0 -g 100000 -b:v 40M clip.hevc
#include <fcntl.h>
#include <linux/futex.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>

#include <vector>

#include "rp_tap.h"

static volatile sig_atomic_t gStop = 0;
static void onSignal(int) { gStop = 1; }

static uint64_t nowNs() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}


// NAL units of an Annex B stream: (offset of the start code, offset of the NAL header).
static std::vector<std::pair<size_t, size_t>> nals(const uint8_t* d, size_t n) {
  std::vector<std::pair<size_t, size_t>> out;
  for (size_t i = 0; i + 3 <= n; ++i) {
    if (d[i] == 0 && d[i + 1] == 0 && d[i + 2] == 1) {
      const size_t sc = i > 0 && d[i - 1] == 0 ? i - 1 : i;
      out.push_back({sc, i + 3});
      i += 2;
    }
  }
  return out;
}

static int nalType(const uint8_t* p, bool h264) { return h264 ? (p[0] & 31) : ((p[0] >> 1) & 63); }
static bool isVcl(int t, bool h264) { return h264 ? (t >= 1 && t <= 5) : t < 32; }
static bool isPrefix(int t, bool h264) { return h264 ? (t == 6 || t == 7 || t == 8 || t == 9) : (t >= 32 && t <= 35) || t == 39; }
// first_mb_in_slice == 0 / first_slice_segment_in_pic_flag: the first bit after the header.
static bool firstSlice(const uint8_t* p, bool h264) { return h264 ? (p[1] & 0x80) : (p[2] & 0x80); }

static std::vector<std::pair<size_t, size_t>> splitAccessUnits(const std::vector<uint8_t>& f, bool h264) {
  std::vector<std::pair<size_t, size_t>> aus;
  const auto ns = nals(f.data(), f.size());
  size_t start = 0;
  bool haveVcl = false;
  for (const auto& [sc, hdr] : ns) {
    if (hdr + 3 > f.size()) break;
    const int t = nalType(&f[hdr], h264);
    const bool boundary = haveVcl && (isPrefix(t, h264) || (isVcl(t, h264) && firstSlice(&f[hdr], h264)));
    if (boundary) { aus.push_back({start, sc - start}); start = sc; haveVcl = false; }
    if (isVcl(t, h264)) haveVcl = true;
  }
  if (haveVcl) aus.push_back({start, f.size() - start});
  return aus;
}

// Bytes of parameter sets at the start of an access unit.
static size_t paramSetBytes(const uint8_t* d, size_t n, bool h264) {
  for (const auto& [sc, hdr] : nals(d, n)) {
    const int t = nalType(d + hdr, h264);
    const bool ps = h264 ? (t == 7 || t == 8) : (t >= 32 && t <= 34);
    if (!ps && t != (h264 ? 9 : 35)) return sc;
  }
  return 0;
}

// The picture size from the first SPS (enough for the common streams: no cropping parsing).
struct Bits {
  std::vector<uint8_t> b;
  size_t pos = 0;
  uint32_t u(int n) { uint32_t v = 0; while (n--) { v = (v << 1) | ((b[pos / 8] >> (7 - pos % 8)) & 1); ++pos; } return v; }
  uint32_t ue() { int z = 0; while (pos < b.size() * 8 && !u(1)) ++z; return (1u << z) - 1 + u(z); }
};
static bool parseSize(const std::vector<uint8_t>& f, bool h264, uint32_t& w, uint32_t& h) {
  for (const auto& [sc, hdr] : nals(f.data(), std::min<size_t>(f.size(), 4096))) {
    if (nalType(&f[hdr], h264) != (h264 ? 7 : 33)) continue;
    Bits r;
    for (size_t i = hdr + (h264 ? 1 : 2); i < std::min(f.size(), hdr + 200); ++i) {
      if (i >= hdr + 4 && f[i] == 3 && f[i - 1] == 0 && f[i - 2] == 0) continue;   // emulation prevention
      r.b.push_back(f[i]);
    }
    if (h264) return false;   // H.264: the default size
    r.u(4);
    const uint32_t maxSub = r.u(3);
    r.u(1);
    r.u(2); r.u(1); r.u(5); r.u(32); r.u(48); r.u(8);   // profile_tier_level, general part
    bool subP[8], subL[8];
    for (uint32_t i = 0; i < maxSub; ++i) { subP[i] = r.u(1); subL[i] = r.u(1); }
    if (maxSub > 0) for (uint32_t i = maxSub; i < 8; ++i) r.u(2);
    for (uint32_t i = 0; i < maxSub; ++i) { if (subP[i]) r.u(88); if (subL[i]) r.u(8); }
    r.ue();
    const uint32_t chroma = r.ue();
    if (chroma == 3) r.u(1);
    w = r.ue();
    h = r.ue();
    if (r.u(1)) {   // conformance window, in chroma units for 4:2:0
      const uint32_t l = r.ue(), rr = r.ue(), t = r.ue(), b = r.ue();
      w -= 2 * (l + rr);
      h -= 2 * (t + b);
    }
    return w > 0 && h > 0;
  }
  return false;
}

template <typename T> static void store(volatile T* p, T v) { __atomic_store_n(const_cast<T*>(p), v, __ATOMIC_RELEASE); }

static void wake(volatile RpTapHeader* h) {
  __atomic_add_fetch(const_cast<uint32_t*>(&h->futex), 1u, __ATOMIC_RELEASE);
  syscall(SYS_futex, const_cast<uint32_t*>(&h->futex), FUTEX_WAKE, INT_MAX, nullptr, nullptr, 0);
}

int main(int argc, char** argv) {
  std::string input, shm = RP_TAP_DEFAULT_NAME;
  double fps = 60, loss = 0, seconds = 0;
  bool loop = false, h264 = false;
  struct StickEv { double ms; int x; };
  std::vector<StickEv> sticks;   // --stick MS[:X]: publish a right-stick onset (x = +1 / -1) MS after the first frame
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() { if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", a.c_str()); exit(2); } return std::string(argv[++i]); };
    if (a == "--shm") shm = next();
    else if (a == "--fps") fps = atof(next().c_str());
    else if (a == "--loss") loss = atof(next().c_str());
    else if (a == "--seconds") seconds = atof(next().c_str());
    else if (a == "--loop") loop = true;
    else if (a == "--h264") h264 = true;
    else if (a == "--stick") {
      const std::string v = next();
      const size_t c = v.find(':');
      sticks.push_back({atof(v.substr(0, c).c_str()), c == std::string::npos ? 1 : atoi(v.c_str() + c + 1)});
    }
    else if (a[0] != '-') input = a;
    else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
  }
  if (input.empty()) { fprintf(stderr, "usage: moonglide-rp-fake clip.hevc [--shm NAME] [--fps 60] [--loss P] [--seconds N] [--loop]\n"); return 2; }
  signal(SIGINT, onSignal);
  signal(SIGTERM, onSignal);

  FILE* in = fopen(input.c_str(), "rb");
  if (!in) { perror(input.c_str()); return 1; }
  std::vector<uint8_t> file;
  for (uint8_t buf[1 << 16];;) {
    const size_t got = fread(buf, 1, sizeof(buf), in);
    if (!got) break;
    file.insert(file.end(), buf, buf + got);
  }
  fclose(in);
  const std::vector<std::pair<size_t, size_t>> aus = splitAccessUnits(file, h264);
  if (aus.empty()) { fprintf(stderr, "no access units in %s\n", input.c_str()); return 1; }
  uint32_t width = 0, height = 0;
  if (!parseSize(file, h264, width, height)) { width = 1920; height = 1080; }
  const std::string path = "/" + shm;
  int fd = shm_open(path.c_str(), O_RDWR | O_CREAT, 0600);
  if (fd < 0 || ftruncate(fd, RP_TAP_SIZE) != 0) { perror(path.c_str()); return 1; }
  uint8_t* base = static_cast<uint8_t*>(mmap(nullptr, RP_TAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
  if (base == MAP_FAILED) { perror("mmap"); return 1; }
  volatile RpTapHeader* h = reinterpret_cast<volatile RpTapHeader*>(base);

  store(&h->magic, 0u);
  h->version = RP_TAP_VERSION;
  h->slot_count = RP_TAP_SLOT_COUNT;
  h->slot_size = RP_TAP_SLOT_SIZE;
  h->codec = h264 ? RP_TAP_H264 : RP_TAP_H265;
  h->width = width;
  h->height = height;
  h->max_fps = (uint32_t)(fps + 0.5);
  h->qpc_freq = 1000000000ull;
  h->features = 0;
  h->writer_pid = (uint32_t)getpid();
  h->bitrate_kbps = 0;
  store(&h->codec_header_size, 0u);
  store(&h->write_seq, (uint64_t)0);
  h->session = h->session + 1;
  h->state = RP_TAP_STREAMING;
  snprintf(const_cast<char*>(h->status), 128, "fake: %s", input.c_str());
  store(&h->magic, RP_TAP_MAGIC);
  wake(h);
  fprintf(stderr, "rp-fake: %s %ux%u %s, %zu frames @ %.0f into /dev/shm/%s (session %u)\n", input.c_str(), width, height,
          h264 ? "H.264" : "HEVC", aus.size(), fps, shm.c_str(), h->session);

  std::mt19937 rng(7);
  std::uniform_real_distribution<double> uni(0, 1);
  const uint64_t period = (uint64_t)(1e9 / fps);
  uint64_t due = nowNs(), n = 0, lostRun = 0, lostTotal = 0, idrAsked = 0;
  std::sort(sticks.begin(), sticks.end(), [](const StickEv& a, const StickEv& b) { return a.ms < b.ms; });
  size_t stickNext = 0;
  const uint64_t t0 = due;
  const uint64_t end = seconds > 0 ? due + (uint64_t)(seconds * 1e9) : UINT64_MAX;
  bool first = true;
  while (!gStop && nowNs() < end) {
    if (n + lostTotal >= aus.size() * (loop ? UINT32_MAX : 1)) break;
    const auto [off, len] = aus[(n + lostTotal) % aus.size()];
    const uint8_t* data = file.data() + off;
    if (first) {
      // The PS5 sends its parameter sets in the first packet; keep a copy like the tap does.
      const size_t ps = paramSetBytes(data, len, h264);
      if (ps && ps <= RP_TAP_CODEC_HEADER_MAX) {
        memcpy(const_cast<uint8_t*>(h->codec_header), data, ps);
        h->codec_header_session = h->session;
        store(&h->codec_header_size, (uint32_t)ps);
      }
    }
    first = false;
    if (__atomic_exchange_n(const_cast<uint32_t*>(&h->idr_request), 0u, __ATOMIC_ACQ_REL)) idrAsked++;
    // Steady beat, like the network: one frame per period.
    due += period;
    for (uint64_t t = nowNs(); t < due; t = nowNs()) {
      const uint64_t d = due - t;
      timespec ts{(time_t)(d / 1000000000ull), (long)(d % 1000000000ull)};
      nanosleep(&ts, nullptr);
    }
    // Like the tap: the moment the state that deflects the stick is sent.
    while (stickNext < sticks.size() && nowNs() >= t0 + (uint64_t)(sticks[stickNext].ms * 1e6)) {
      volatile RpTapStick* sk = reinterpret_cast<volatile RpTapStick*>(base + RP_TAP_STICK_OFFSET);
      sk->x = (int16_t)(sticks[stickNext].x * 30000);
      sk->y = 0;
      sk->onset_ns = nowNs();
      __atomic_add_fetch(const_cast<uint32_t*>(&sk->seq), 1u, __ATOMIC_RELEASE);
      stickNext++;
    }
    if (loss > 0 && n > 30 && uni(rng) < loss) {   // lost on the network: never reaches the tap
      lostRun++;
      lostTotal++;
      continue;
    }
    const uint64_t seq = __atomic_load_n(const_cast<uint64_t*>(&h->write_seq), __ATOMIC_ACQUIRE);
    uint8_t* slot = base + RP_TAP_HEADER_SIZE + (size_t)(seq % RP_TAP_SLOT_COUNT) * RP_TAP_SLOT_SIZE;
    volatile RpTapSlot* s = reinterpret_cast<volatile RpTapSlot*>(slot);
    store(&s->seq, (uint64_t)~0ull);
    const uint32_t size = std::min<uint32_t>((uint32_t)len, RP_TAP_SLOT_SIZE - RP_TAP_SLOT_HEADER);
    s->arrival_ns = nowNs();
    s->size = size;
    s->frames_lost = (int32_t)lostRun;
    s->recovered = 0;
    s->flags = size < (uint32_t)len ? RP_TAP_SLOT_TRUNCATED : 0u;
    memcpy(slot + RP_TAP_SLOT_HEADER, data, size);
    store(&s->seq, (uint64_t)seq);
    store(&h->write_seq, (uint64_t)(seq + 1));
    h->frames_lost_total = (uint32_t)lostTotal;
    wake(h);
    lostRun = 0;
    ++n;
  }
  store(&h->magic, 0u);
  h->state = RP_TAP_QUIT;
  snprintf(const_cast<char*>(h->status), 128, "fake: done");
  wake(h);
  fprintf(stderr, "rp-fake: %llu frames sent, %llu lost, %llu keyframe requests (not answerable)\n",
          (unsigned long long)n, (unsigned long long)lostTotal, (unsigned long long)idrAsked);
  return 0;
}
