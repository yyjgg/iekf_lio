#include "iekf/iekf_predictor.hpp"

#include <cmath>

namespace iekf_lio
{
namespace
{

using Mat15 = Eigen::Matrix<double, 15, 15>;
using Mat18x15 = Eigen::Matrix<double, 18, 15>;

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

void IekfPredictor::initializeState(IekfState18 & state) const
{
  state.x = IekfNominalState18 {};
  state.p_cov = IekfMat18::Identity() * 1e-3;
  state.is_initialized = true;
}

void IekfPredictor::predictWithMidpoint(const ImuTrack & imu_track, IekfState18 & state) const
{
  if (!state.is_initialized) {
    initializeState(state);
  }
  if (imu_track.size() < 2) {
    return;
  }

  for (std::size_t i = 0; i + 1 < imu_track.size(); ++i) {
    const ImuSample & cur = imu_track[i];
    const ImuSample & nxt = imu_track[i + 1];
    const double dt = nxt.time_s - cur.time_s;
    if (dt <= 0.0) {
      continue;
    }

    const Eigen::Vector3d w0 = cur.gyro_rps - state.x.b_g;
    const Eigen::Vector3d w1 = nxt.gyro_rps - state.x.b_g;
    const Eigen::Vector3d a0 = cur.accel_mps2 - state.x.b_a;
    const Eigen::Vector3d a1 = nxt.accel_mps2 - state.x.b_a;
    const Eigen::Vector3d w_mid = 0.5 * (w0 + w1);
    const Eigen::Vector3d a_mid = 0.5 * (a0 + a1);

    const Eigen::Matrix3d dR_half = expSO3(0.5 * dt * w_mid);
    const Eigen::Matrix3d R_mid = state.x.r_wb * dR_half;
    const Eigen::Vector3d a_world = R_mid * a_mid + state.x.g_w;

    state.x.p_wb += state.x.v_wb * dt + 0.5 * a_world * dt * dt;
    state.x.v_wb += a_world * dt;
    state.x.r_wb = state.x.r_wb * expSO3(dt * w_mid);

    IekfMat18 F = IekfMat18::Zero();
    F.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
    F.block<3, 3>(3, 6) = -state.x.r_wb * skew(a_mid);
    F.block<3, 3>(3, 12) = -state.x.r_wb;
    F.block<3, 3>(3, 15) = Eigen::Matrix3d::Identity();
    F.block<3, 3>(6, 6) = -skew(w_mid);
    F.block<3, 3>(6, 9) = -Eigen::Matrix3d::Identity();

    Mat18x15 G = Mat18x15::Zero();
    G.block<3, 3>(3, 0) = -state.x.r_wb;                // n_a
    G.block<3, 3>(6, 3) = -Eigen::Matrix3d::Identity(); // n_g
    G.block<3, 3>(9, 6) = Eigen::Matrix3d::Identity();  // n_wg
    G.block<3, 3>(12, 9) = Eigen::Matrix3d::Identity(); // n_wa
    G.block<3, 3>(15, 12) = Eigen::Matrix3d::Identity();// n_wg0

    Mat15 Qc = Mat15::Zero();
    Qc.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * (noise_.sigma_acc * noise_.sigma_acc);
    Qc.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * (noise_.sigma_gyro * noise_.sigma_gyro);
    Qc.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity() * (noise_.sigma_bg_rw * noise_.sigma_bg_rw);
    Qc.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity() * (noise_.sigma_ba_rw * noise_.sigma_ba_rw);
    Qc.block<3, 3>(12, 12) = Eigen::Matrix3d::Identity() * (noise_.sigma_g_rw * noise_.sigma_g_rw);

    const IekfMat18 Phi = IekfMat18::Identity() + F * dt;
    const IekfMat18 Qd = (G * Qc * G.transpose()) * dt;
    state.p_cov = Phi * state.p_cov * Phi.transpose() + Qd;
  }
}

}  // namespace iekf_lio
