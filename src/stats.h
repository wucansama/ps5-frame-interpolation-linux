// Rolling numbers for the badge and the once-a-second log line.
#pragma once
#include <cstddef>
#include <deque>
#include <vector>

// Events per second over a sliding window.
class RateMeter {
 public:
  explicit RateMeter(double windowMs = 1000.0) : window_(windowMs) {}
  void tick(double nowMs);
  double rate(double nowMs);

 private:
  double window_;
  std::deque<double> events_;
};

// Last-N samples with mean / percentiles.
class Samples {
 public:
  explicit Samples(size_t capacity = 120) : capacity_(capacity) {}
  void add(double v);
  double mean() const;
  double percentile(double p) const;
  double last() const { return values_.empty() ? 0.0 : values_.back(); }
  size_t size() const { return values_.size(); }
  void clear() { values_.clear(); }

 private:
  size_t capacity_;
  std::deque<double> values_;
};
