#pragma once

#include <cmath>
#include <memory>

namespace iekf_lio
{

class ScalarFilter
{
public:
  virtual ~ScalarFilter() = default;

  virtual void reset() = 0;
  virtual double update(double input, double dt_s) = 0;
  virtual bool isInitialized() const = 0;
};

class FirstOrderLowPassFilter : public ScalarFilter
{
public:
  explicit FirstOrderLowPassFilter(double cutoff_hz = 5.0)
  : cutoff_hz_(cutoff_hz) {}

  void reset() override
  {
    initialized_ = false;
    value_ = 0.0;
  }

  double update(double input, double dt_s) override
  {
    if (!initialized_ || dt_s <= 0.0 || cutoff_hz_ <= 0.0) {
      value_ = input;
      initialized_ = true;
      return value_;
    }

    constexpr double kPi = 3.14159265358979323846;
    const double tau = 1.0 / (2.0 * kPi * cutoff_hz_);
    const double alpha = dt_s / (tau + dt_s);
    value_ += alpha * (input - value_);
    return value_;
  }

  bool isInitialized() const override
  {
    return initialized_;
  }

private:
  double cutoff_hz_ = 5.0;
  double value_ = 0.0;
  bool initialized_ = false;
};

}  // namespace iekf_lio
