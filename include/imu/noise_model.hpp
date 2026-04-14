#pragma once

#include <cstddef>
#include <deque>

#include <Eigen/Dense>

#include "imu/imu_types.hpp"

namespace iekf_lio
{

struct ImuInitConfig
{
  std::size_t window_size = 500;
  double gyro_var_threshold = 1e-4;
  double accel_var_threshold = 0.05;
  double gravity_norm = 9.81;
};

struct ImuInitResult
{
  bool initialized = false;
  std::size_t used_samples = 0;
  Eigen::Vector3d gyro_bias = Eigen::Vector3d::Zero();
  Eigen::Vector3d accel_bias = Eigen::Vector3d::Zero();
  Eigen::Vector3d gravity_w = Eigen::Vector3d(0.0, 0.0, -9.81);
  double gyro_var_norm = 0.0;
  double accel_var_norm = 0.0;
};

class ImuStaticInitializer
{
public:
  explicit ImuStaticInitializer(ImuInitConfig config = {});

  void setConfig(const ImuInitConfig & config);
  const ImuInitConfig & config() const;

  ImuInitResult update(const ImuTrack & track);
  bool initialized() const;
  const ImuInitResult & lastResult() const;
  void reset();

private:
  void pushSample(const ImuSample & sample);
  ImuInitResult evaluateWindow() const;

  ImuInitConfig config_;
  std::deque<ImuSample> window_;
  ImuInitResult last_result_;
};

}  // namespace iekf_lio
