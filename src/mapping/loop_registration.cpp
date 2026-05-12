#include "mapping/loop_registration.hpp"

#include <cmath>

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>

namespace iekf_lio
{
namespace
{

constexpr double kPi = 3.14159265358979323846;

Eigen::Matrix3d yawRotation(double yaw_rad)
{
  const double c = std::cos(yaw_rad);
  const double s = std::sin(yaw_rad);
  Eigen::Matrix3d r = Eigen::Matrix3d::Identity();
  r(0, 0) = c;
  r(0, 1) = -s;
  r(1, 0) = s;
  r(1, 1) = c;
  return r;
}

}  // namespace

void LoopRegistration::setConfig(const LoopRegistrationConfig & config)
{
  config_ = config;
  config_.voxel_leaf_size = std::max(1e-3, config_.voxel_leaf_size);
  config_.max_corr_distance = std::max(1e-3, config_.max_corr_distance);
  config_.max_iterations = std::max(1, config_.max_iterations);
  config_.fitness_threshold = std::max(0.0, config_.fitness_threshold);
  config_.translation_sigma = std::max(1e-3, config_.translation_sigma);
  config_.rotation_sigma_rad = std::max(1e-3, config_.rotation_sigma_rad);
}

bool LoopRegistration::alignKeyframes(
  const LoopRegistrationKeyframeView & query,
  const LoopRegistrationKeyframeView & target,
  double yaw_init_rad,
  BackendLoopConstraint * constraint) const
{
  if (constraint == nullptr) {
    return false;
  }
  *constraint = BackendLoopConstraint {};
  if (!config_.enable || query.cloud_i == nullptr || target.cloud_i == nullptr ||
    query.cloud_i->empty() || target.cloud_i->empty())
  {
    return false;
  }

  const Eigen::Matrix3d r_wq_init = yawRotation(yaw_init_rad) * query.r_wb;
  auto source_world = transformCloud(query.cloud_i, query.p_wb, r_wq_init);
  auto target_world = transformCloud(target.cloud_i, target.p_wb, target.r_wb);
  if (source_world->empty() || target_world->empty()) {
    return false;
  }

  pcl::VoxelGrid<pcl::PointXYZ> vg;
  vg.setLeafSize(
    static_cast<float>(config_.voxel_leaf_size),
    static_cast<float>(config_.voxel_leaf_size),
    static_cast<float>(config_.voxel_leaf_size));

  auto source_ds = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  auto target_ds = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  vg.setInputCloud(source_world);
  vg.filter(*source_ds);
  vg.setInputCloud(target_world);
  vg.filter(*target_ds);
  if (source_ds->empty() || target_ds->empty()) {
    return false;
  }

  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
  icp.setInputSource(source_ds);
  icp.setInputTarget(target_ds);
  icp.setMaxCorrespondenceDistance(config_.max_corr_distance);
  icp.setMaximumIterations(config_.max_iterations);
  icp.setTransformationEpsilon(1e-8);
  icp.setEuclideanFitnessEpsilon(1e-6);
  pcl::PointCloud<pcl::PointXYZ> aligned;
  icp.align(aligned);
  if (!icp.hasConverged()) {
    return false;
  }

  const double fitness = icp.getFitnessScore();
  if (!std::isfinite(fitness) || fitness > config_.fitness_threshold) {
    return false;
  }

  const Eigen::Matrix4d t_corr = icp.getFinalTransformation().cast<double>();
  const Eigen::Matrix4d t_wq_init = makeTransformMatrix(query.p_wb, r_wq_init).cast<double>();
  const Eigen::Matrix4d t_wq_corr = t_corr * t_wq_init;
  const Eigen::Matrix4d t_wm = makeTransformMatrix(target.p_wb, target.r_wb).cast<double>();
  const Eigen::Matrix4d t_mq = t_wm.inverse() * t_wq_corr;

  constraint->from_id = target.id;
  constraint->to_id = query.id;
  constraint->t_ij = t_mq.block<3, 1>(0, 3);
  constraint->r_ij = t_mq.block<3, 3>(0, 0);
  constraint->fitness = fitness;
  constraint->valid = true;
  constraint->information.setZero();
  const double inv_t_var = 1.0 / (config_.translation_sigma * config_.translation_sigma);
  const double inv_r_var = 1.0 / (config_.rotation_sigma_rad * config_.rotation_sigma_rad);
  constraint->information.block<3, 3>(0, 0).diagonal().setConstant(inv_t_var);
  constraint->information.block<3, 3>(3, 3).diagonal().setConstant(inv_r_var);
  return true;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr LoopRegistration::transformCloud(
  const pcl::PointCloud<pcl::PointXYZ>::ConstPtr & cloud_i,
  const Eigen::Vector3d & p_wb,
  const Eigen::Matrix3d & r_wb) const
{
  auto cloud_w = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  if (cloud_i == nullptr || cloud_i->empty()) {
    return cloud_w;
  }
  pcl::transformPointCloud(*cloud_i, *cloud_w, makeTransformMatrix(p_wb, r_wb));
  return cloud_w;
}

Eigen::Matrix4f LoopRegistration::makeTransformMatrix(
  const Eigen::Vector3d & p_wb,
  const Eigen::Matrix3d & r_wb) const
{
  Eigen::Matrix4f t = Eigen::Matrix4f::Identity();
  t.block<3, 3>(0, 0) = r_wb.cast<float>();
  t.block<3, 1>(0, 3) = p_wb.cast<float>();
  return t;
}

}  // namespace iekf_lio
