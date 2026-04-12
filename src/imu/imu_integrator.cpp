#include "imu/imu_integrator.hpp"

#include <cmath>

namespace iekf_lio
{
namespace
{

Eigen::Matrix3d skew(const Eigen::Vector3d & w)
{
  Eigen::Matrix3d m = Eigen::Matrix3d::Zero();
  m(0, 1) = -w.z();
  m(0, 2) = w.y();
  m(1, 0) = w.z();
  m(1, 2) = -w.x();
  m(2, 0) = -w.y();
  m(2, 1) = w.x();
  return m;
}

Eigen::Matrix3d expSO3(const Eigen::Vector3d & theta)
{
  const double angle = theta.norm();
  const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d K = skew(theta);
  const Eigen::Matrix3d K2 = K * K;

  if (angle < 1e-10) {
    return I + K;
  }

  const double a = std::sin(angle) / angle;
  const double b = (1.0 - std::cos(angle)) / (angle * angle);
  return I + a * K + b * K2;
}

}  // namespace

ImuPreintegrationResult ImuIntegrator::integrateMidpoint(const ImuTrack & imu_track) const
{
  ImuPreintegrationResult out;
  if (imu_track.size() < 2) {
    out.used_samples = imu_track.size();
    return out;
  }

  for (std::size_t i = 0; i + 1 < imu_track.size(); ++i) {
    const auto & cur = imu_track[i];
    const auto & nxt = imu_track[i + 1];
    const double dt = nxt.time_s - cur.time_s;
    if (dt <= 0.0) {
      continue;
    }

    const Eigen::Vector3d a_mid = 0.5 * (cur.accel_mps2 + nxt.accel_mps2);
    const Eigen::Vector3d w_mid = 0.5 * (cur.gyro_rps + nxt.gyro_rps);

    const Eigen::Matrix3d dR_half = expSO3(0.5 * dt * w_mid);
    const Eigen::Matrix3d R_mid = out.delta_r * dR_half;
    const Eigen::Vector3d a_world = R_mid * a_mid;

    out.delta_p += out.delta_v * dt + 0.5 * a_world * dt * dt;
    out.delta_v += a_world * dt;
    out.delta_r = out.delta_r * expSO3(dt * w_mid);
    out.delta_t_s += dt;
  }

  out.used_samples = imu_track.size();
  return out;
}

}  // namespace iekf_lio
