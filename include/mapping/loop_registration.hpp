#pragma once

#include <cstddef>

#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace iekf_lio
{

struct BackendLoopConstraint
{
  std::size_t from_id = 0;
  std::size_t to_id = 0;
  Eigen::Vector3d t_ij = Eigen::Vector3d::Zero();
  Eigen::Matrix3d r_ij = Eigen::Matrix3d::Identity();
  Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Identity();
  double fitness = 0.0;
  bool valid = false;
};

struct LoopRegistrationConfig
{
  bool enable = false;
  double voxel_leaf_size = 0.5;
  double max_corr_distance = 2.0;
  int max_iterations = 40;
  double fitness_threshold = 0.5;
  double translation_sigma = 0.5;
  double rotation_sigma_rad = 0.2;
};

struct LoopRegistrationKeyframeView
{
  std::size_t id = 0;
  Eigen::Vector3d p_wb = Eigen::Vector3d::Zero();
  Eigen::Matrix3d r_wb = Eigen::Matrix3d::Identity();
  pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud_i;
};

class LoopRegistration
{
public:
  void setConfig(const LoopRegistrationConfig & config);

  bool alignKeyframes(
    const LoopRegistrationKeyframeView & query,
    const LoopRegistrationKeyframeView & target,
    double yaw_init_rad,
    BackendLoopConstraint * constraint) const;

private:
  pcl::PointCloud<pcl::PointXYZ>::Ptr transformCloud(
    const pcl::PointCloud<pcl::PointXYZ>::ConstPtr & cloud_i,
    const Eigen::Vector3d & p_wb,
    const Eigen::Matrix3d & r_wb) const;

  Eigen::Matrix4f makeTransformMatrix(
    const Eigen::Vector3d & p_wb,
    const Eigen::Matrix3d & r_wb) const;

  LoopRegistrationConfig config_;
};

}  // namespace iekf_lio
