#pragma once

#include <cstdint>
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

struct CloudDeskewTiming
{
  std::uint64_t end_state_interp_ns = 0;
  std::uint64_t point_interp_ns = 0;
  std::uint64_t point_transform_ns = 0;
  std::uint64_t output_merge_ns = 0;
};

class CloudDeskewer
{
public:
  LidarScanXYZ deskewToImuEnd(
    const LidarScan & scan,
    const std::vector<ImuPredictedState> & imu_states,
    const LidarToImuExtrinsic & extrinsic,
    CloudDeskewTiming * timing = nullptr) const;

private:
  bool interpolateState(
    const std::vector<ImuPredictedState> & states,
    double t,
    ImuPredictedState * out) const;
};

}  // namespace iekf_lio
