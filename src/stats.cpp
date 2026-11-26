#include "stats.h"

#include <algorithm>

void RateMeter::tick(double nowMs) {
  events_.push_back(nowMs);
  rate(nowMs);
}

double RateMeter::rate(double nowMs) {
  while (!events_.empty() && events_.front() < nowMs - window_) events_.pop_front();
  if (events_.size() < 2) return (double)events_.size() * 1000.0 / window_;
  // Intervals between the events in the window: robust to where the window edge falls.
  return (double)(events_.size() - 1) * 1000.0 / std::max(1.0, events_.back() - events_.front());
}

void Samples::add(double v) {
  values_.push_back(v);
  while (values_.size() > capacity_) values_.pop_front();
}

double Samples::mean() const {
  if (values_.empty()) return 0.0;
  double s = 0;
  for (double v : values_) s += v;
  return s / (double)values_.size();
}

double Samples::percentile(double p) const {
  if (values_.empty()) return 0.0;
  std::vector<double> v(values_.begin(), values_.end());
  const size_t k = std::min(v.size() - 1, (size_t)(p * (double)(v.size() - 1) + 0.5));
  std::nth_element(v.begin(), v.begin() + (long)k, v.end());
  return v[k];
}
