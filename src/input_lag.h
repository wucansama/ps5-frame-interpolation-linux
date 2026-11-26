// Input lag while playing: camera-pan onset. The tap stamps the moment the right stick went from
// neutral to deflected (RpTapStick); this finds the first decoded picture that clearly starts to
// pan sideways and reports the difference. Cheap: a 240x18 luma grid (row-sampled box means) per
// picture and a horizontal shift search against the previous picture, ~0.05 ms on the capture thread.
//
// Safeguards: the pan must stand out from the motion of the 150 ms before the stick moved, a window of
// 20..300 ms, the pan must repeat in the next picture (same direction, within 150 ms), pans in the
// opposite direction to the first sample are rejected, scene cuts / unmatched pictures cancel it.
#pragma once
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <utility>
#include <vector>

namespace moonglide {

class CameraLag {
 public:
  // kMaxShift 64 of 240 columns: a fast camera turn moves the picture by up to ~25% of its width per frame.
  static constexpr int kCols = 240, kRows = 18, kMaxShift = 64, kCells = kCols * kRows;

  // Luma plane (tight, `w` bytes per row) -> kRows sampled rows, each reduced to kCols box means.
  static void profile(const uint8_t* y, uint32_t w, uint32_t h, uint8_t* out) {
    const uint32_t cw = std::max<uint32_t>(1, w / kCols);
    const uint32_t cols = std::min<uint32_t>(kCols, w / cw);
    for (int r = 0; r < kRows; ++r) {
      const uint8_t* row = y + (size_t)((r * 2 + 1) * h / (kRows * 2)) * w;
      uint8_t* o = out + r * kCols;
      for (uint32_t c = 0; c < cols; ++c) {
        const uint8_t* p = row + c * cw;
        uint32_t s = 0;
        for (uint32_t i = 0; i < cw; ++i) s += p[i];
        o[c] = (uint8_t)(s / cw);
      }
      for (uint32_t c = cols; c < (uint32_t)kCols; ++c) o[c] = o[cols - 1];
    }
  }

  struct Result {
    bool reliable = true, distinct = false;
    float shift = 0;   // columns, + = content moved right
    float sad0 = 0, sadBest = 0;
    int best = 0;
  };

  // Shift between two profiles. distinct: not a repeat of the same picture. reliable: not a cut.
  static Result shiftBetween(const uint8_t* prev, const uint8_t* cur) {
    Result r;
    auto sad = [&](int s) {
      uint32_t a = 0;
      for (int y = 0; y < kRows; ++y) {
        const uint8_t* p = prev + y * kCols + kMaxShift;
        const uint8_t* c = cur + y * kCols + kMaxShift + s;
        for (int i = 0; i < kCols - 2 * kMaxShift; ++i) a += (uint32_t)std::abs((int)p[i] - (int)c[i]);
      }
      return (float)a / (float)((kCols - 2 * kMaxShift) * kRows);
    };
    float v[2 * kMaxShift + 1];
    for (int s = -kMaxShift; s <= kMaxShift; ++s) v[s + kMaxShift] = sad(s);
    int best = 0;
    for (int s = -kMaxShift; s <= kMaxShift; ++s)
      if (v[s + kMaxShift] < v[best + kMaxShift]) best = s;
    const float sad0 = v[kMaxShift], sadBest = v[best + kMaxShift];
    r.sad0 = sad0; r.sadBest = sadBest; r.best = best;
    r.distinct = sad0 > 0.35f;
    if (!r.distinct) return r;
    // A turning game camera is not a pure shift (motion blur, parallax, a fixed HUD): a clear
    // improvement over no shift is enough.
    const bool matched = best != 0 && sadBest <= 0.85f * sad0;
    if (sad0 > 8.f && sadBest > 0.95f * sad0) { r.reliable = false; return r; }   // nothing like a pan: a cut
    float sub = 0;
    if (best > -kMaxShift && best < kMaxShift) {
      const float a = v[best + kMaxShift - 1], b = v[best + kMaxShift], c = v[best + kMaxShift + 1];
      const float d = a - 2 * b + c;
      if (d > 1e-6f) sub = std::clamp(0.5f * (a - c) / d, -0.5f, 0.5f);
    }
    if (matched) r.shift = (float)best + sub;
    else if (best == 0 && std::min(v[kMaxShift - 1], v[kMaxShift + 1]) < 2.5f * sad0) r.shift = sub;   // under one column
    else r.shift = 0;
    return r;
  }

  // The stick went from neutral to deflected at `ns` (CLOCK_MONOTONIC). Only a mostly horizontal
  // deflection is used (the pan is measured sideways).
  void stick(uint64_t ns, int x, int y) {
    pend_ = false;
    cand_ = false;
    if (std::abs(x) < 12000 || std::abs(x) < 2 * std::abs(y)) { if (dbg) fprintf(stderr, "inputlag: stick %d,%d not horizontal\n", x, y); return; }
    stickNs_ = ns;
    stickSign_ = x > 0 ? 1 : -1;
    // The picture may already move a little (walking, driving): the pan must stand out from that.
    base_ = 0;
    for (const auto& [m, sh] : motion_)
      if (m + 150000000ull >= ns && m < ns + 20000000ull) base_ = std::max(base_, std::fabs(sh));
    quiet_ = base_ < 1.5f;   // already turning: nothing to measure
    pend_ = true;
    if (dbg) fprintf(stderr, "inputlag: stick onset x=%d base %.2f\n", x, base_);
  }

  // One decoded picture. Returns true with `lagMs` when a sample is complete. Call stick() first
  // if a new onset was published. keyframe: no information about motion.
  bool frame(uint64_t arrivalNs, const uint8_t* lumaY, uint32_t w, uint32_t h, bool keyframe, double& lagMs) {
    uint8_t cur[kCells];
    profile(lumaY, w, h, cur);
    Result r;
    if (havePrev_ && !keyframe) r = shiftBetween(prev_, cur);
    const bool ok = havePrev_ && !keyframe;
    if (!havePrev_ || r.distinct || keyframe) {
      std::copy(cur, cur + kCells, prev_);   // repeats keep the older picture as reference
      havePrev_ = true;
    }
    if (!ok) return false;
    while (!motion_.empty() && motion_.front().first + 2000000000ull < arrivalNs) motion_.pop_front();
    // thr: about 0.35 of a column (3 px at 1080p), above the motion there was before the stick moved
    const float thr = pend_ ? std::max(0.35f, 2.f * base_ + 0.3f) : 0.35f;
    const bool moving = r.reliable && std::fabs(r.shift) >= thr;
    if (r.distinct) motion_.push_back({arrivalNs, r.reliable ? r.shift : 99.f});
    if (!r.reliable) {
      // A change no shift explains: a cut, or a turn too fast to match. While waiting for the pan it
      // is skipped (not a sample, not a cancel); the next matched pictures decide.
      if (dbg && pend_) fprintf(stderr, "inputlag: unmatched picture while pending (sad0 %.1f best %d sad %.1f)\n", r.sad0, r.best, r.sadBest);
      if (!pend_) cand_ = false;
      return false;
    }
    bool done = false;
    if (pend_ && arrivalNs > stickNs_ + 300000000ull) { if (dbg) fprintf(stderr, "inputlag: no pan within 300 ms\n"); pend_ = false; cand_ = false; }
    if (dbg && pend_ && arrivalNs > stickNs_) fprintf(stderr, "inputlag: +%.1f ms shift %.2f distinct %d sad0 %.1f best %d sad %.1f\n", (double)(arrivalNs - stickNs_) / 1e6, r.shift, r.distinct ? 1 : 0, r.sad0, r.best, r.sadBest);
    if (moving) {
      const int sign = r.shift > 0 ? 1 : -1;
      if (pend_ && !cand_) {
        const double dt = (double)(arrivalNs - stickNs_) / 1e6;
        if (arrivalNs <= stickNs_ || dt < 20.0 || !quiet_) { if (dbg) fprintf(stderr, "inputlag: reject dt %.1f quiet %d\n", dt, quiet_ ? 1 : 0); pend_ = false; }   // already panning, or too fast to be ours
        else if (polarity_ != 0 && sign * stickSign_ != polarity_) { if (dbg) fprintf(stderr, "inputlag: reject polarity\n"); pend_ = false; }
        else { cand_ = true; candDt_ = dt; candSign_ = sign; candNs_ = arrivalNs; }
      } else if (pend_ && cand_ && arrivalNs > candNs_) {
        if (sign == candSign_ && arrivalNs - candNs_ <= 150000000ull) {
          polarity_ = candSign_ * stickSign_;
          lagMs = candDt_;
          done = true;
        }
        else if (dbg) fprintf(stderr, "inputlag: reject confirm sign %d/%d gap %.1f\n", sign, candSign_, (double)(arrivalNs - candNs_) / 1e6);
        pend_ = false;
        cand_ = false;
      }
    } else if (cand_ && arrivalNs - candNs_ > 150000000ull) {
      pend_ = false;
      cand_ = false;
    }
    return done;
  }

  bool dbg = false;
  void reset() { havePrev_ = false; pend_ = cand_ = false; motion_.clear(); polarity_ = 0; }

 private:
  uint8_t prev_[kCells] = {};
  bool havePrev_ = false, pend_ = false, cand_ = false, quiet_ = false;
  uint64_t stickNs_ = 0, candNs_ = 0;
  int stickSign_ = 1, candSign_ = 1, polarity_ = 0;
  double candDt_ = 0;
  std::deque<std::pair<uint64_t, float>> motion_;   // distinct pictures: arrival, shift (99 = unmatched)
  float base_ = 0;
};

}  // namespace moonglide
