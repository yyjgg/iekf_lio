#include "lidar/cloud_deskewer.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <omp.h>
#include <utility>

namespace iekf_lio
{

namespace
{
using SteadyClock = std::chrono::steady_clock;

bool isNondecreasingByRelativeTime(const LidarScan & scan)
{
  if (scan.points.size() < 2) {
    return true;
  }
  for (std::size_t i = 1; i < scan.points.size(); ++i) {
    if (scan.points[i].relative_time_s < scan.points[i - 1].relative_time_s) {
      return false;
    }
  }
  return true;
}

std::size_t initialSegmentIndex(
  const std::vector<ImuPredictedState> & states,
  double t)
{
  if (states.size() < 2) {
    return 0;
  }
  if (t <= states.front().time_s) {
    return 0;
  }
  if (t >= states.back().time_s) {
    return states.size() - 2;
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
    return 0;
  }
  return static_cast<std::size_t>((it - states.begin()) - 1);
}

bool interpolateStateWithSegment(
  const std::vector<ImuPredictedState> & states,
  std::size_t * segment_idx,
  double t,
  ImuPredictedState * out)
{
  if (out == nullptr || segment_idx == nullptr || states.size() < 2) {
    return false;
  }
  if (t < states.front().time_s || t > states.back().time_s) {
    return false;
  }
  if (t == states.front().time_s) {
    *out = states.front();
    *segment_idx = 0;
    return true;
  }
  if (t == states.back().time_s) {
    *out = states.back();
    *segment_idx = states.size() - 2;
    return true;
  }

  std::size_t idx = std::min(*segment_idx, states.size() - 2);
  while (idx + 1 < states.size() && states[idx + 1].time_s < t) {
    ++idx;
  }
  while (idx > 0 && states[idx].time_s > t) {
    --idx;
  }

  const ImuPredictedState & s0 = states[idx];
  const ImuPredictedState & s1 = states[idx + 1];
  if (t < s0.time_s || t > s1.time_s) {
    return false;
  }
  const double dt = s1.time_s - s0.time_s;
  if (dt <= 1e-12) {
    *out = s0;
    *segment_idx = idx;
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
  *segment_idx = idx;
  return true;
}
}

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
  const LidarToImuExtrinsic & extrinsic,
  CloudDeskewTiming * timing) const
{
  if (timing != nullptr) {
    *timing = CloudDeskewTiming {};
  }
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
  const auto end_interp_t0 = SteadyClock::now();
  if (!interpolateState(imu_states, scan.scan_end_time_s, &end_state)) {
    return out;
  }
  if (timing != nullptr) {
    timing->end_state_interp_ns = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        SteadyClock::now() - end_interp_t0).count());
  }
  const Eigen::Matrix3d r_iw_end = end_state.r_wi.transpose();
  const double scan_begin_time_s = scan.scan_begin_time_s;
  const double scan_end_time_s = scan.scan_end_time_s;
  const std::size_t total_points = scan.points.size();
  const bool monotonic_points = isNondecreasingByRelativeTime(scan);
  std::uint64_t skipped_points = 0;

  const int num_threads = std::max(1, omp_get_max_threads());
  std::vector<std::vector<pcl::PointXYZ>> thread_points(static_cast<std::size_t>(num_threads));

#pragma omp parallel reduction(+:skipped_points)
  {
    const int tid = omp_get_thread_num();
    auto & local_points = thread_points[static_cast<std::size_t>(tid)];
    local_points.reserve(total_points / static_cast<std::size_t>(num_threads) + 1);
    std::uint64_t point_interp_ns_local = 0;
    std::uint64_t point_transform_ns_local = 0;

    const std::size_t chunk_begin =
      (static_cast<std::size_t>(tid) * total_points) / static_cast<std::size_t>(num_threads);
    const std::size_t chunk_end =
      (static_cast<std::size_t>(tid + 1) * total_points) / static_cast<std::size_t>(num_threads);
    std::size_t segment_idx = 0;
    if (monotonic_points && chunk_begin < chunk_end) {
      const double first_time = scan_begin_time_s + scan.points[chunk_begin].relative_time_s;
      segment_idx = initialSegmentIndex(imu_states, first_time);
    }

    for (std::size_t i = chunk_begin; i < chunk_end; ++i) {
      const auto & pt = scan.points[i];
      const double pt_time = scan_begin_time_s + pt.relative_time_s;
      ImuPredictedState st;
      const auto point_interp_t0 = SteadyClock::now();
      const bool interp_ok = monotonic_points ?
        interpolateStateWithSegment(imu_states, &segment_idx, pt_time, &st) :
        interpolateState(imu_states, pt_time, &st);
      if (!interp_ok) {
        point_interp_ns_local += static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
            SteadyClock::now() - point_interp_t0).count());
        ++skipped_points;
        continue;
      }
      point_interp_ns_local += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
          SteadyClock::now() - point_interp_t0).count());

      const auto point_tf_t0 = SteadyClock::now();
      // Compose the LiDAR->IMU(current)->world->IMU(scan_end) chain into one affine transform:
      // p_i_end = A(t) * p_l + b(t),
      // where A(t) = R_iw_end * R_wi(t) * R_il,
      //       b(t) = R_iw_end * (R_wi(t) * t_il + p_wi(t) - p_wi_end).
      const Eigen::Matrix3d a_iendl = r_iw_end * st.r_wi * extrinsic.r_il;
      const Eigen::Vector3d b_iend = r_iw_end * (
        st.r_wi * extrinsic.t_il + st.p_wi - end_state.p_wi);
      const Eigen::Vector3d p_i_end =
        a_iendl * Eigen::Vector3d(pt.x, pt.y, pt.z) + b_iend;

      pcl::PointXYZ out_pt;
      out_pt.x = static_cast<float>(p_i_end.x());
      out_pt.y = static_cast<float>(p_i_end.y());
      out_pt.z = static_cast<float>(p_i_end.z());
      local_points.push_back(out_pt);
      point_transform_ns_local += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
          SteadyClock::now() - point_tf_t0).count());
    }

#pragma omp critical
    {
      if (timing != nullptr) {
        timing->point_interp_ns += point_interp_ns_local;
        timing->point_transform_ns += point_transform_ns_local;
      }
    }
  }

  const auto merge_t0 = SteadyClock::now();
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
  if (timing != nullptr) {
    timing->output_merge_ns = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        SteadyClock::now() - merge_t0).count());
  }

  if (skipped_points > 0) {
    std::cerr << "[CloudDeskewer][WARN] skipped points due to interpolation failure: "
              << skipped_points << "/" << total_points
              << ", scan_time_range=[" << scan_begin_time_s << ", " << scan_end_time_s << "]\n";
  }
  return out;
}

}  // namespace iekf_lio
