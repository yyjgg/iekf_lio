#pragma once

#include <cstddef>
#include <vector>

#include <Eigen/Dense>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace iekf_lio
{

class TreePointMap
{
public:
  TreePointMap();

  void clear();
  void setInputCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud_w);
  std::size_t size() const;
  pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud() const;
  std::size_t radiusSearchPoints(
    const Eigen::Vector3d & query_w,
    double radius_m,
    std::size_t max_results,
    std::vector<Eigen::Vector3d> * neighbors) const;

private:
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_w_;
  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree_;
};

}  // namespace iekf_lio
