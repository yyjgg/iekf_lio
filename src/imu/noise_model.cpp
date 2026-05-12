#include "imu/noise_model.hpp"

#include <algorithm>
#include <cmath>

namespace
{

Eigen::Matrix3d rotX(double roll_rad)
{
  const double c = std::cos(roll_rad);
  const double s = std::sin(roll_rad);
  Eigen::Matrix3d r = Eigen::Matrix3d::Identity();
  r(1, 1) = c;
  r(1, 2) = -s;
  r(2, 1) = s;
  r(2, 2) = c;
  return r;
}

Eigen::Matrix3d rotY(double pitch_rad)
{
  const double c = std::cos(pitch_rad);
  const double s = std::sin(pitch_rad);
  Eigen::Matrix3d r = Eigen::Matrix3d::Identity();
  r(0, 0) = c;
  r(0, 2) = s;
  r(2, 0) = -s;
  r(2, 2) = c;
  return r;
}

Eigen::Matrix3d estimateGravityAlignedInitialRotation(const Eigen::Vector3d & mean_acc)
{
  if (mean_acc.norm() <= 1e-9) {
    return Eigen::Matrix3d::Identity();
  }

  const double ax = mean_acc.x();
  const double ay = mean_acc.y();
  const double az = mean_acc.z();
  const double roll_rad = std::atan2(ay, az);
  const double pitch_rad = std::atan2(-ax, std::sqrt(ay * ay + az * az));
  return rotY(pitch_rad) * rotX(roll_rad);
}

}  // namespace

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
  last_result_.initial_r_wb = Eigen::Matrix3d::Identity();
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
  out.initial_r_wb = Eigen::Matrix3d::Identity();
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
  out.initial_r_wb = estimateGravityAlignedInitialRotation(mean_acc);
  out.accel_bias = mean_acc + out.initial_r_wb.transpose() * out.gravity_w;
  out.initialized = true;
  return out;
}

}  // namespace iekf_lio
