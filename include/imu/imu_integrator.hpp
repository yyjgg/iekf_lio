#pragma once

#include <cstddef>

#include <Eigen/Dense>

#include "imu/imu_types.hpp"

namespace iekf_lio
{

struct ImuPreintegrationResult
{
  double delta_t_s = 0.0;
  Eigen::Vector3d delta_p = Eigen::Vector3d::Zero();
  Eigen::Vector3d delta_v = Eigen::Vector3d::Zero();
  Eigen::Matrix3d delta_r = Eigen::Matrix3d::Identity();
  std::size_t used_samples = 0;
};

class ImuIntegrator
{
public:
  ImuPreintegrationResult integrateMidpoint(const ImuTrack & imu_track) const;
};

}  // namespace iekf_lio
