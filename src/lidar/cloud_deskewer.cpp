#include "lidar/cloud_deskewer.hpp"

#include <algorithm>
#include <iostream>
#include <limits>

namespace iekf_lio
{

bool CloudDeskewer::interpolateState(
  const std::vector<ImuPredictedState> & states,
  double t,
  ImuPredictedState * out) const
{
  if (out == nullptr || states.empty()) {
    return false;
  }
  if (t < states.front().time_s || t > states.back().time_s) {
    return false;
  }
  if (t == states.front().time_s) {
    *out = states.front();
    return true;
  }
  if (t == states.back().time_s) {
    *out = states.back();
    return true;
  }

  const auto it = std::lower_bound(
    states.begin(),
    states.end(),
    t,
    [](const ImuPredictedState & s, double ts)
    {
      return s.time_s < ts;
    });

  if (it == states.begin()) {
    *out = *it;
    return true;
  }
  const ImuPredictedState & s1 = *it;
  const ImuPredictedState & s0 = *(it - 1);
  const double dt = s1.time_s - s0.time_s;
  if (dt <= 1e-12) {
    *out = s0;
    return true;
  }

  const double alpha = (t - s0.time_s) / dt;
  out->time_s = t;
  out->p_wi = (1.0 - alpha) * s0.p_wi + alpha * s1.p_wi;

  Eigen::Quaterniond q0(s0.r_wi);
  Eigen::Quaterniond q1(s1.r_wi);
  q0.normalize();
  q1.normalize();
  out->r_wi = q0.slerp(alpha, q1).toRotationMatrix();
  return true;
}

bool CloudDeskewer::deskewToImuEnd(
  LidarScan & scan,
  const std::vector<ImuPredictedState> & imu_states,
  const LidarToImuExtrinsic & extrinsic) const
{
  if (scan.points.empty() || imu_states.empty()) {
    return false;
  }

  ImuPredictedState end_state;
  if (!interpolateState(imu_states, scan.scan_end_time_s, &end_state)) {
    return false;
  }
  const Eigen::Matrix3d r_iw_end = end_state.r_wi.transpose();
  bool all_interpolated = true;
  std::size_t skipped_points = 0;

  for (auto & pt : scan.points) {
    const double pt_time = scan.scan_begin_time_s + pt.relative_time_s;
    ImuPredictedState st;
    if (!interpolateState(imu_states, pt_time, &st)) {
      all_interpolated = false;
      ++skipped_points;
      const float qnan = std::numeric_limits<float>::quiet_NaN();
      pt.x = qnan;
      pt.y = qnan;
      pt.z = qnan;
      if (skipped_points <= 5) {
        std::cerr << "[CloudDeskewer][WARN] interpolateState failed for point_time="
                  << pt_time << ", skip this point.\n";
      }
      continue;
    }

    const Eigen::Vector3d p_l(pt.x, pt.y, pt.z);
    const Eigen::Vector3d p_i = extrinsic.r_il * p_l + extrinsic.t_il;
    const Eigen::Vector3d p_w = st.r_wi * p_i + st.p_wi;
    const Eigen::Vector3d p_i_end = r_iw_end * (p_w - end_state.p_wi);
    pt.x = static_cast<float>(p_i_end.x());
    pt.y = static_cast<float>(p_i_end.y());
    pt.z = static_cast<float>(p_i_end.z());
  }
  if (skipped_points > 0) {
    std::cerr << "[CloudDeskewer][WARN] skipped points due to interpolation failure: "
              << skipped_points << "/" << scan.points.size() << "\n";
  }
  return all_interpolated;
}

}  // namespace iekf_lio
