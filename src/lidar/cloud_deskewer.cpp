#include "lidar/cloud_deskewer.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <omp.h>
#include <utility>

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

LidarScanXYZ CloudDeskewer::deskewToImuEnd(
  const LidarScan & scan,
  const std::vector<ImuPredictedState> & imu_states,
  const LidarToImuExtrinsic & extrinsic) const
{
  LidarScanXYZ out;
  out.frame_id = scan.frame_id;
  out.scan_begin_time_s = scan.scan_begin_time_s;
  out.scan_end_time_s = scan.scan_end_time_s;
  out.timebase_ns = scan.timebase_ns;
  out.points->clear();

  if (scan.points.empty() || imu_states.empty()) {
    return out;
  }

  ImuPredictedState end_state;
  if (!interpolateState(imu_states, scan.scan_end_time_s, &end_state)) {
    return out;
  }
  const Eigen::Matrix3d r_iw_end = end_state.r_wi.transpose();
  const double scan_begin_time_s = scan.scan_begin_time_s;
  const double scan_end_time_s = scan.scan_end_time_s;
  const std::size_t total_points = scan.points.size();
  std::uint64_t skipped_points = 0;

  const int num_threads = std::max(1, omp_get_max_threads());
  std::vector<std::vector<pcl::PointXYZ>> thread_points(static_cast<std::size_t>(num_threads));

#pragma omp parallel reduction(+:skipped_points)
  {
    const int tid = omp_get_thread_num();
    auto & local_points = thread_points[static_cast<std::size_t>(tid)];
    local_points.reserve(total_points / static_cast<std::size_t>(num_threads) + 1);

#pragma omp for schedule(static)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(total_points); ++i) {
      const auto & pt = scan.points[static_cast<std::size_t>(i)];
      const double pt_time = scan_begin_time_s + pt.relative_time_s;
      ImuPredictedState st;
      if (!interpolateState(imu_states, pt_time, &st)) {
        ++skipped_points;
        continue;
      }

      const Eigen::Vector3d p_l(pt.x, pt.y, pt.z);
      const Eigen::Vector3d p_i = extrinsic.r_il * p_l + extrinsic.t_il;
      const Eigen::Vector3d p_w = st.r_wi * p_i + st.p_wi;
      const Eigen::Vector3d p_i_end = r_iw_end * (p_w - end_state.p_wi);

      pcl::PointXYZ out_pt;
      out_pt.x = static_cast<float>(p_i_end.x());
      out_pt.y = static_cast<float>(p_i_end.y());
      out_pt.z = static_cast<float>(p_i_end.z());
      local_points.push_back(out_pt);
    }
  }

  std::size_t kept_points = 0;
  for (const auto & local_points : thread_points) {
    kept_points += local_points.size();
  }
  out.points->reserve(kept_points);
  for (auto & local_points : thread_points) {
    out.points->insert(
      out.points->end(),
      std::make_move_iterator(local_points.begin()),
      std::make_move_iterator(local_points.end()));
  }
  out.points->width = static_cast<std::uint32_t>(out.points->size());
  out.points->height = 1;
  out.points->is_dense = false;

  if (skipped_points > 0) {
    std::cerr << "[CloudDeskewer][WARN] skipped points due to interpolation failure: "
              << skipped_points << "/" << total_points
              << ", scan_time_range=[" << scan_begin_time_s << ", " << scan_end_time_s << "]\n";
  }
  return out;
}

}  // namespace iekf_lio
