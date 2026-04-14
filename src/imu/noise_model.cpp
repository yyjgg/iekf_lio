#include "imu/noise_model.hpp"

#include <algorithm>

namespace iekf_lio
{

ImuStaticInitializer::ImuStaticInitializer(ImuInitConfig config)
: config_(config)
{
}

void ImuStaticInitializer::setConfig(const ImuInitConfig & config)
{
  config_ = config;
  reset();
}

const ImuInitConfig & ImuStaticInitializer::config() const
{
  return config_;
}

ImuInitResult ImuStaticInitializer::update(const ImuTrack & track)
{
  if (last_result_.initialized) {
    return last_result_;
  }

  for (const auto & sample : track) {
    pushSample(sample);
  }

  last_result_ = evaluateWindow();
  return last_result_;
}

bool ImuStaticInitializer::initialized() const
{
  return last_result_.initialized;
}

const ImuInitResult & ImuStaticInitializer::lastResult() const
{
  return last_result_;
}

void ImuStaticInitializer::reset()
{
  window_.clear();
  last_result_ = ImuInitResult {};
  last_result_.gravity_w = Eigen::Vector3d(0.0, 0.0, -config_.gravity_norm);
}

void ImuStaticInitializer::pushSample(const ImuSample & sample)
{
  window_.push_back(sample);
  const std::size_t max_size = std::max<std::size_t>(1, config_.window_size);
  while (window_.size() > max_size) {
    window_.pop_front();
  }
}

ImuInitResult ImuStaticInitializer::evaluateWindow() const
{
  ImuInitResult out;
  out.used_samples = window_.size();
  out.gravity_w = Eigen::Vector3d(0.0, 0.0, -config_.gravity_norm);
  if (window_.empty()) {
    return out;
  }

  Eigen::Vector3d mean_gyro = Eigen::Vector3d::Zero();
  Eigen::Vector3d mean_acc = Eigen::Vector3d::Zero();
  for (const auto & s : window_) {
    mean_gyro += s.gyro_rps;
    mean_acc += s.accel_mps2;
  }
  mean_gyro /= static_cast<double>(window_.size());
  mean_acc /= static_cast<double>(window_.size());

  Eigen::Vector3d var_gyro = Eigen::Vector3d::Zero();
  Eigen::Vector3d var_acc = Eigen::Vector3d::Zero();
  for (const auto & s : window_) {
    const Eigen::Vector3d d_g = s.gyro_rps - mean_gyro;
    const Eigen::Vector3d d_a = s.accel_mps2 - mean_acc;
    var_gyro += d_g.cwiseProduct(d_g);
    var_acc += d_a.cwiseProduct(d_a);
  }
  var_gyro /= static_cast<double>(window_.size());
  var_acc /= static_cast<double>(window_.size());

  out.gyro_var_norm = var_gyro.mean();
  out.accel_var_norm = var_acc.mean();

  const bool enough_samples = window_.size() >= std::max<std::size_t>(10, config_.window_size);
  const bool static_gyro = out.gyro_var_norm < config_.gyro_var_threshold;
  const bool static_acc = out.accel_var_norm < config_.accel_var_threshold;
  if (!(enough_samples && static_gyro && static_acc)) {
    return out;
  }

  out.gyro_bias = mean_gyro;
  if (mean_acc.norm() > 1e-6) {
    out.gravity_w = -mean_acc.normalized() * config_.gravity_norm;
  }
  // With initial R_wb = I, static model gives: a_m = -g_w + b_a.
  out.accel_bias = mean_acc + out.gravity_w;
  out.initialized = true;
  return out;
}

}  // namespace iekf_lio
