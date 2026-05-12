#include "iekf/iekf_predictor.hpp"

#include <chrono>
#include <cmath>

#if defined(IEKF_LIO_USE_SOPHUS) && __has_include(<sophus/so3.hpp>)
#include <sophus/so3.hpp>
#define IEKF_LIO_SOPHUS_ACTIVE 1
#else
#define IEKF_LIO_SOPHUS_ACTIVE 0
#endif

namespace iekf_lio
{
namespace
{

using Mat15 = Eigen::Matrix<double, 15, 15>;
using Mat18x15 = Eigen::Matrix<double, 18, 15>;
using SteadyClock = std::chrono::steady_clock;

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
#if IEKF_LIO_SOPHUS_ACTIVE
  return Sophus::SO3d::exp(theta).matrix();
#else
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
#endif
}

double yawFromRotation(const Eigen::Matrix3d & r_wb)
{
  return std::atan2(r_wb(1, 0), r_wb(0, 0));
}

ImuPredictedState makePredictedState(
  double time_s,
  bool is_propagated)
{
  ImuPredictedState out;
  out.time_s = time_s;
  out.is_propagated = is_propagated;
  return out;
}

}  // namespace

void IekfPredictor::initializeState(IekfState18 & state) const
{
  state.x = IekfNominalState18 {};
  state.p_cov = IekfMat18::Identity() * 1e-3;
  state.is_initialized = true;
}

void IekfPredictor::setPdrConfig(const IekfPredictorPdrConfig & config)
{
  pdr_config_ = config;
  accel_norm_filter_ = std::make_unique<FirstOrderLowPassFilter>(
    pdr_config_.accel_norm_lpf_cutoff_hz);
}

void IekfPredictor::setAccelNormFilter(std::unique_ptr<ScalarFilter> filter)
{
  if (filter == nullptr) {
    accel_norm_filter_ = std::make_unique<FirstOrderLowPassFilter>(
      pdr_config_.accel_norm_lpf_cutoff_hz);
    return;
  }
  accel_norm_filter_ = std::move(filter);
}

void IekfPredictor::resetPdrRuntime()
{
  if (accel_norm_filter_ != nullptr) {
    accel_norm_filter_->reset();
  }
}

void IekfPredictor::predictWithMidpoint(
  const ImuTrack & imu_track,
  IekfState18 & state,
  std::vector<ImuPredictedState> * predicted_states,
  IekfPredictorTiming * timing,
  std::vector<PdrPreprocessedSample> * pdr_samples)
{
  if (timing != nullptr) {
    *timing = IekfPredictorTiming {};
  }
  if (!state.is_initialized) {
    initializeState(state);
  }
  if (imu_track.size() < 2) {
    if (predicted_states) {
      predicted_states->clear();
      if (!imu_track.empty()) {
        predicted_states->push_back(
          makePredictedState(imu_track.front().time_s, false));
      }
    }
    if (pdr_samples) {
      pdr_samples->clear();
    }
    return;
  }

  if (predicted_states) {
    predicted_states->clear();
    predicted_states->reserve(imu_track.size());
    predicted_states->push_back(
      makePredictedState(imu_track.front().time_s, false));
  }
  if (pdr_samples) {
    pdr_samples->clear();
    pdr_samples->reserve(imu_track.size());
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
    const double accel_norm = a_mid.norm();
    const double gravity_norm = state.x.g_w.norm();
    const double accel_norm_minus_g = std::abs(accel_norm - gravity_norm);
    const double accel_norm_minus_g_lpf = (accel_norm_filter_ != nullptr)
      ? accel_norm_filter_->update(accel_norm_minus_g, dt)
      : accel_norm_minus_g;

    const auto state_prop_t0 = SteadyClock::now();
    const Eigen::Matrix3d dR_half = expSO3(0.5 * dt * w_mid);
    const Eigen::Matrix3d R_mid = state.x.r_wb * dR_half;
    const Eigen::Vector3d a_world = R_mid * a_mid + state.x.g_w;
    const Eigen::Vector3d w_world = R_mid * w_mid;

    state.x.p_wb += state.x.v_wb * dt + 0.5 * a_world * dt * dt;
    state.x.v_wb += a_world * dt;
    state.x.r_wb = state.x.r_wb * expSO3(dt * w_mid);
    if (timing != nullptr) {
      timing->state_propagation_ns += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
          SteadyClock::now() - state_prop_t0).count());
    }

    const auto cov_t0 = SteadyClock::now();
    IekfMat18 F = IekfMat18::Zero();
    F.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
    F.block<3, 3>(3, 6) = -state.x.r_wb * skew(a_mid);
    F.block<3, 3>(3, 12) = -state.x.r_wb;
    F.block<3, 3>(3, 15) = Eigen::Matrix3d::Identity();
    F.block<3, 3>(6, 6) = -skew(w_mid);
    F.block<3, 3>(6, 9) = -Eigen::Matrix3d::Identity();

    const IekfMat18 Phi = IekfMat18::Identity() + F * dt;
    IekfMat18 Qd = IekfMat18::Zero();
    // Fast-path Qd construction:
    // Under the current model, the continuous noises are mutually independent and isotropic
    // in each 3D block, and G injects them block-wise into [v, theta, b_g, b_a, g].
    // Therefore G * Qc * G^T collapses to a block-diagonal matrix here, and the velocity
    // block further simplifies because R * (sigma_a^2 I) * R^T = sigma_a^2 I.
    const double sigma_acc2_dt = (noise_.sigma_acc * noise_.sigma_acc) * dt;
    const double sigma_gyro2_dt = (noise_.sigma_gyro * noise_.sigma_gyro) * dt;
    const double sigma_bg_rw2_dt = (noise_.sigma_bg_rw * noise_.sigma_bg_rw) * dt;
    const double sigma_ba_rw2_dt = (noise_.sigma_ba_rw * noise_.sigma_ba_rw) * dt;
    const double sigma_g_rw2_dt = (noise_.sigma_g_rw * noise_.sigma_g_rw) * dt;
    Qd.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * sigma_acc2_dt;
    Qd.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity() * sigma_gyro2_dt;
    Qd.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity() * sigma_bg_rw2_dt;
    Qd.block<3, 3>(12, 12) = Eigen::Matrix3d::Identity() * sigma_ba_rw2_dt;
    Qd.block<3, 3>(15, 15) = Eigen::Matrix3d::Identity() * sigma_g_rw2_dt;

    // Original generic form kept here for reference:
    // Mat18x15 G = Mat18x15::Zero();
    // G.block<3, 3>(3, 0) = -state.x.r_wb;                 // n_a
    // G.block<3, 3>(6, 3) = -Eigen::Matrix3d::Identity();  // n_g
    // G.block<3, 3>(9, 6) = Eigen::Matrix3d::Identity();   // n_bg
    // G.block<3, 3>(12, 9) = Eigen::Matrix3d::Identity();  // n_ba
    // G.block<3, 3>(15, 12) = Eigen::Matrix3d::Identity(); // n_g_rw
    // Mat15 Qc = Mat15::Zero();
    // Qc.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * (noise_.sigma_acc * noise_.sigma_acc);
    // Qc.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * (noise_.sigma_gyro * noise_.sigma_gyro);
    // Qc.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity() * (noise_.sigma_bg_rw * noise_.sigma_bg_rw);
    // Qc.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity() * (noise_.sigma_ba_rw * noise_.sigma_ba_rw);
    // Qc.block<3, 3>(12, 12) = Eigen::Matrix3d::Identity() * (noise_.sigma_g_rw * noise_.sigma_g_rw);
    // const IekfMat18 Qd = (G * Qc * G.transpose()) * dt;
    state.p_cov = Phi * state.p_cov * Phi.transpose() + Qd;
    if (timing != nullptr) {
      timing->covariance_ns += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
          SteadyClock::now() - cov_t0).count());
    }

    if (predicted_states) {
      const auto record_t0 = SteadyClock::now();
      ImuPredictedState pred_state = makePredictedState(nxt.time_s, true);
      pred_state.p_wi = state.x.p_wb;
      pred_state.r_wi = state.x.r_wb;
      pred_state.gyro_mid_unbiased_rps = w_world;
      pred_state.linear_accel_mid_w_mps2 = a_world;
      pred_state.yaw_mid_rad = yawFromRotation(R_mid);
      predicted_states->push_back(pred_state);
      if (timing != nullptr) {
        timing->state_record_ns += static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
            SteadyClock::now() - record_t0).count());
      }
    }

    if (pdr_samples) {
      PdrPreprocessedSample sample;
      sample.time_s = cur.time_s + 0.5 * dt;
      sample.accel_unbiased_b_mps2 = a_mid;
      sample.gyro_unbiased_b_rps = w_mid;
      sample.r_mid_wb = R_mid;
      sample.accel_norm_mps2 = accel_norm;
      sample.accel_norm_minus_g_mps2 = accel_norm_minus_g;
      sample.accel_norm_minus_g_lpf_mps2 = accel_norm_minus_g_lpf;
      pdr_samples->push_back(sample);
    }
  }
}

}  // namespace iekf_lio
