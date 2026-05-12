#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "iekf/iekf_state.hpp"
#include "imu/imu_types.hpp"
#include "lidar/cloud_deskewer.hpp"
#include "pdr/pdr_types.hpp"
#include "pdr/scalar_filter.hpp"

namespace iekf_lio
{

struct IekfPredictorNoise
{
  double sigma_acc = 0.10;      // m/s^2/sqrt(Hz)
  double sigma_gyro = 0.01;     // rad/s/sqrt(Hz)
  double sigma_bg_rw = 0.0001;  // rad/s^2/sqrt(Hz)
  double sigma_ba_rw = 0.001;   // m/s^3/sqrt(Hz)
  double sigma_g_rw = 0.0;      // m/s^3/sqrt(Hz)
};

struct IekfPredictorTiming
{
  std::uint64_t state_propagation_ns = 0;
  std::uint64_t covariance_ns = 0;
  std::uint64_t state_record_ns = 0;
};

struct IekfPredictorPdrConfig
{
  double accel_norm_lpf_cutoff_hz = 5.0;
};

class IekfPredictor
{
public:
  explicit IekfPredictor(IekfPredictorNoise noise = {})
  : noise_(noise),
    accel_norm_filter_(std::make_unique<FirstOrderLowPassFilter>()) {}

  void initializeState(IekfState18 & state) const;
  void setPdrConfig(const IekfPredictorPdrConfig & config);
  void setAccelNormFilter(std::unique_ptr<ScalarFilter> filter);
  void resetPdrRuntime();
  void predictWithMidpoint(
    const ImuTrack & imu_track,
    IekfState18 & state,
    std::vector<ImuPredictedState> * predicted_states = nullptr,
    IekfPredictorTiming * timing = nullptr,
    std::vector<PdrPreprocessedSample> * pdr_samples = nullptr);

private:
  IekfPredictorNoise noise_;
  IekfPredictorPdrConfig pdr_config_;
  std::unique_ptr<ScalarFilter> accel_norm_filter_;
};

}  // namespace iekf_lio
