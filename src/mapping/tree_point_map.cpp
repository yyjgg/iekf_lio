#include "mapping/tree_point_map.hpp"

#include <algorithm>
#include <utility>

namespace iekf_lio
{

TreePointMap::TreePointMap()
: cloud_w_(std::make_shared<pcl::PointCloud<pcl::PointXYZ>>())
{
}

void TreePointMap::clear()
{
  cloud_w_->clear();
  kdtree_.setInputCloud(cloud_w_);
}

void TreePointMap::setInputCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud_w)
{
  if (cloud_w == nullptr) {
    clear();
    return;
  }
  cloud_w_ = cloud_w;
  kdtree_.setInputCloud(cloud_w_);
}

std::size_t TreePointMap::size() const
{
  return cloud_w_ == nullptr ? 0U : cloud_w_->size();
}

pcl::PointCloud<pcl::PointXYZ>::ConstPtr TreePointMap::cloud() const
{
  return cloud_w_;
}

std::size_t TreePointMap::radiusSearchPoints(
  const Eigen::Vector3d & query_w,
  double radius_m,
  std::size_t max_results,
  std::vector<Eigen::Vector3d> * neighbors) const
{
  if (
    neighbors == nullptr ||
    cloud_w_ == nullptr ||
    cloud_w_->empty() ||
    radius_m <= 0.0 ||
    max_results == 0U)
  {
    return 0U;
  }

  pcl::PointXYZ query;
  query.x = static_cast<float>(query_w.x());
  query.y = static_cast<float>(query_w.y());
  query.z = static_cast<float>(query_w.z());

  std::vector<int> indices;
  std::vector<float> sq_dists;
  const int found = kdtree_.radiusSearch(query, radius_m, indices, sq_dists);
  if (found <= 0) {
    neighbors->clear();
    return 0U;
  }

  std::vector<std::pair<float, int>> ordered;
  ordered.reserve(static_cast<std::size_t>(found));
  for (int i = 0; i < found; ++i) {
    ordered.emplace_back(sq_dists[static_cast<std::size_t>(i)], indices[static_cast<std::size_t>(i)]);
  }
  std::sort(
    ordered.begin(),
    ordered.end(),
    [](const std::pair<float, int> & a, const std::pair<float, int> & b) {
      return a.first < b.first;
    });

  const std::size_t take = std::min(max_results, ordered.size());
  neighbors->clear();
  neighbors->reserve(take);
  for (std::size_t i = 0; i < take; ++i) {
    const pcl::PointXYZ & pt = cloud_w_->at(static_cast<std::size_t>(ordered[i].second));
    neighbors->emplace_back(pt.x, pt.y, pt.z);
  }
  return take;
}

}  // namespace iekf_lio
