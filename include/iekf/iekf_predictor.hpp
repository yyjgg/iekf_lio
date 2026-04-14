#pragma once

#include <vector>

#include "iekf/iekf_state.hpp"
#include "imu/imu_types.hpp"
#include "lidar/cloud_deskewer.hpp"

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

class IekfPredictor
{
public:
  explicit IekfPredictor(IekfPredictorNoise noise = {}) : noise_(noise) {}

  void initializeState(IekfState18 & state) const;
  void predictWithMidpoint(
    const ImuTrack & imu_track,
    IekfState18 & state,
    std::vector<ImuPredictedState> * predicted_states = nullptr) const;

private:
  IekfPredictorNoise noise_;
};

}  // namespace iekf_lio
