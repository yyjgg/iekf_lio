#pragma once

#include <cstddef>

#include <Eigen/Dense>

namespace iekf_lio
{

struct KeyframeManagerConfig
{
  bool enable = true;
  double translation_thresh_m = 0.5;
  double rotation_thresh_rad = 10.0 * 3.14159265358979323846 / 180.0;
  double time_thresh_s = 0.0;
};

struct KeyframeDecision
{
  bool is_keyframe = false;
  bool is_first_keyframe = false;
  double translation_m = 0.0;
  double rotation_rad = 0.0;
  double delta_time_s = 0.0;
  std::size_t total_keyframes = 0;
};

class KeyframeManager
{
public:
  void setConfig(const KeyframeManagerConfig & config);
  void reset();

  KeyframeDecision update(
    double time_s,
    const Eigen::Vector3d & p_wb,
    const Eigen::Matrix3d & r_wb);

  bool hasKeyframe() const { return has_last_keyframe_; }
  std::size_t keyframeCount() const { return keyframe_count_; }
  double lastKeyframeTime() const { return last_time_s_; }
  const Eigen::Vector3d & lastKeyframePosition() const { return last_p_wb_; }
  const Eigen::Matrix3d & lastKeyframeRotation() const { return last_r_wb_; }

private:
  static double relativeRotationAngle(
    const Eigen::Matrix3d & r_ref,
    const Eigen::Matrix3d & r_cur);

  KeyframeManagerConfig config_;
  bool has_last_keyframe_ = false;
  std::size_t keyframe_count_ = 0;
  double last_time_s_ = 0.0;
  Eigen::Vector3d last_p_wb_ = Eigen::Vector3d::Zero();
  Eigen::Matrix3d last_r_wb_ = Eigen::Matrix3d::Identity();
};

}  // namespace iekf_lio
