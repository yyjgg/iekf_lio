#pragma once

#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "types.hpp"

namespace iekf_lio
{

struct ImuPredictedState
{
  double time_s = 0.0;
  Eigen::Vector3d p_wi = Eigen::Vector3d::Zero();
  Eigen::Matrix3d r_wi = Eigen::Matrix3d::Identity();
};

struct LidarToImuExtrinsic
{
  Eigen::Matrix3d r_il = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t_il = Eigen::Vector3d::Zero();
};

class CloudDeskewer
{
public:
  bool deskewToImuEnd(
    LidarScan & scan,
    const std::vector<ImuPredictedState> & imu_states,
    const LidarToImuExtrinsic & extrinsic) const;

private:
  bool interpolateState(
    const std::vector<ImuPredictedState> & states,
    double t,
    ImuPredictedState * out) const;
};

}  // namespace iekf_lio
