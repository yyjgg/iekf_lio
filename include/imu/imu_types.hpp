#pragma once

#include <vector>

#include <Eigen/Dense>

namespace iekf_lio
{

struct ImuSample
{
  double time_s = 0.0;
  Eigen::Vector3d accel_mps2 = Eigen::Vector3d::Zero();
  Eigen::Vector3d gyro_rps = Eigen::Vector3d::Zero();
};

using ImuTrack = std::vector<ImuSample>;

}  // namespace iekf_lio
