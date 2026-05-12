#pragma once

#include <cstdint>

#include <Eigen/Dense>

namespace iekf_lio
{

struct PdrPreprocessedSample
{
  std::uint64_t sample_id = 0;
  double time_s = 0.0;
  Eigen::Vector3d accel_unbiased_b_mps2 = Eigen::Vector3d::Zero();
  Eigen::Vector3d gyro_unbiased_b_rps = Eigen::Vector3d::Zero();
  Eigen::Matrix3d r_mid_wb = Eigen::Matrix3d::Identity();
  double accel_norm_mps2 = 0.0;
  double accel_norm_minus_g_mps2 = 0.0;
  double accel_norm_minus_g_lpf_mps2 = 0.0;
};

struct PdrPeakEvent
{
  std::uint64_t sample_id = 0;
  double time_s = 0.0;
  double accel_norm_minus_g_lpf_mps2 = 0.0;
};

struct PdrStepCandidate
{
  double start_time_s = 0.0;
  double end_time_s = 0.0;
  double duration_s = 0.0;
  double start_peak_value = 0.0;
  double end_peak_value = 0.0;
};

struct PdrStepEvent
{
  double start_time_s = 0.0;
  double end_time_s = 0.0;
  double duration_s = 0.0;
  double max_value = 0.0;
  double min_value = 0.0;
  double gyro_norm_mean_rps = 0.0;
  double peak_valley_diff = 0.0;
  double step_length_m = 0.0;
  double yaw_avg_rad = 0.0;
  Eigen::Vector3d delta_p_w = Eigen::Vector3d::Zero();
};

}  // namespace iekf_lio
