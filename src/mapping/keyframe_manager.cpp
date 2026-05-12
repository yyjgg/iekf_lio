#include "mapping/keyframe_manager.hpp"

#include <algorithm>
#include <cmath>

namespace iekf_lio
{

void KeyframeManager::setConfig(const KeyframeManagerConfig & config)
{
  config_ = config;
  config_.translation_thresh_m = std::max(0.0, config_.translation_thresh_m);
  config_.rotation_thresh_rad = std::max(0.0, config_.rotation_thresh_rad);
}

void KeyframeManager::reset()
{
  has_last_keyframe_ = false;
  keyframe_count_ = 0;
  last_time_s_ = 0.0;
  last_p_wb_.setZero();
  last_r_wb_.setIdentity();
}

KeyframeDecision KeyframeManager::update(
  double time_s,
  const Eigen::Vector3d & p_wb,
  const Eigen::Matrix3d & r_wb)
{
  KeyframeDecision decision;

  if (!config_.enable) {
    return decision;
  }

  if (!has_last_keyframe_) {
    has_last_keyframe_ = true;
    last_time_s_ = time_s;
    last_p_wb_ = p_wb;
    last_r_wb_ = r_wb;
    keyframe_count_ = 1;
    decision.is_keyframe = true;
    decision.is_first_keyframe = true;
    decision.total_keyframes = keyframe_count_;
    return decision;
  }

  decision.translation_m = (p_wb - last_p_wb_).norm();
  decision.rotation_rad = relativeRotationAngle(last_r_wb_, r_wb);
  if (
    decision.translation_m >= config_.translation_thresh_m ||
    decision.rotation_rad >= config_.rotation_thresh_rad)
  {
    has_last_keyframe_ = true;
    last_time_s_ = time_s;
    last_p_wb_ = p_wb;
    last_r_wb_ = r_wb;
    ++keyframe_count_;
    decision.is_keyframe = true;
  }

  decision.total_keyframes = keyframe_count_;
  return decision;
}

double KeyframeManager::relativeRotationAngle(
  const Eigen::Matrix3d & r_ref,
  const Eigen::Matrix3d & r_cur)
{
  const Eigen::Matrix3d r_rel = r_ref.transpose() * r_cur;
  const double cos_theta = std::clamp(0.5 * (r_rel.trace() - 1.0), -1.0, 1.0);
  return std::acos(cos_theta);
}

}  // namespace iekf_lio
