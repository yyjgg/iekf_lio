#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "ikd_Tree.h"

namespace iekf_lio
{

class TreePointMap
{
public:
  TreePointMap();

  void clear();
  void setDownsampleSize(double box_length_m);
  void setDeleteCriterion(double delete_criterion);
  void setBalanceCriterion(double balance_criterion);
  void setInputCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud_w);
  void addWorldPoints(const std::vector<Eigen::Vector3d> & points_world, bool downsample_on);
  void deleteOutsideCube(
    const Eigen::Vector3d & center_w,
    double cube_length_xy_m,
    double cube_height_m);
  std::size_t size() const;
  std::size_t radiusSearchPoints(
    const Eigen::Vector3d & query_w,
    double radius_m,
    std::size_t max_results,
    std::vector<Eigen::Vector3d> * neighbors) const;

private:
  using IkdTreeType = KD_TREE<pcl::PointXYZ>;

  static pcl::PointXYZ toPclPoint(const Eigen::Vector3d & p_w);

  void resetTree();

  std::unique_ptr<IkdTreeType> tree_;
  float delete_criterion_ = 0.5f;
  float balance_criterion_ = 0.7f;
  float downsample_size_m_ = 0.2f;
};

}  // namespace iekf_lio
