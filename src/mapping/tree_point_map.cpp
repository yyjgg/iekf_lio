#include "mapping/tree_point_map.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace iekf_lio
{

TreePointMap::TreePointMap()
: tree_(std::make_unique<IkdTreeType>())
{
  tree_->Set_delete_criterion_param(delete_criterion_);
  tree_->Set_balance_criterion_param(balance_criterion_);
  tree_->set_downsample_param(downsample_size_m_);
}

void TreePointMap::resetTree()
{
  tree_ = std::make_unique<IkdTreeType>();
  tree_->Set_delete_criterion_param(delete_criterion_);
  tree_->Set_balance_criterion_param(balance_criterion_);
  tree_->set_downsample_param(downsample_size_m_);
}

void TreePointMap::clear()
{
  resetTree();
}

void TreePointMap::setDownsampleSize(double box_length_m)
{
  downsample_size_m_ = static_cast<float>(std::max(1e-3, box_length_m));
  tree_->set_downsample_param(downsample_size_m_);
}

void TreePointMap::setDeleteCriterion(double delete_criterion)
{
  delete_criterion_ = static_cast<float>(std::clamp(delete_criterion, 1e-3, 0.999));
  tree_->Set_delete_criterion_param(delete_criterion_);
}

void TreePointMap::setBalanceCriterion(double balance_criterion)
{
  balance_criterion_ = static_cast<float>(std::clamp(balance_criterion, 0.5, 0.999));
  tree_->Set_balance_criterion_param(balance_criterion_);
}

pcl::PointXYZ TreePointMap::toPclPoint(const Eigen::Vector3d & p_w)
{
  pcl::PointXYZ pt;
  pt.x = static_cast<float>(p_w.x());
  pt.y = static_cast<float>(p_w.y());
  pt.z = static_cast<float>(p_w.z());
  return pt;
}

void TreePointMap::setInputCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud_w)
{
  resetTree();
  if (cloud_w == nullptr || cloud_w->empty()) {
    return;
  }
  typename IkdTreeType::PointVector points;
  points.reserve(cloud_w->size());
  for (const auto & pt : cloud_w->points) {
    if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
      continue;
    }
    points.push_back(pt);
  }
  if (!points.empty()) {
    tree_->Build(points);
  }
}

void TreePointMap::addWorldPoints(
  const std::vector<Eigen::Vector3d> & points_world,
  bool downsample_on)
{
  typename IkdTreeType::PointVector points;
  points.reserve(points_world.size());
  for (const auto & p_w : points_world) {
    if (!std::isfinite(p_w.x()) || !std::isfinite(p_w.y()) || !std::isfinite(p_w.z())) {
      continue;
    }
    points.push_back(toPclPoint(p_w));
  }
  if (points.empty()) {
    return;
  }
  if (tree_->size() <= 0 || tree_->validnum() <= 0) {
    tree_->Build(points);
    return;
  }
  tree_->Add_Points(points, downsample_on);
}

void TreePointMap::deleteOutsideCube(
  const Eigen::Vector3d & center_w,
  double cube_length_xy_m,
  double cube_height_m)
{
  if (tree_->size() <= 0) {
    return;
  }

  const BoxPointType range = tree_->tree_range();
  const float min_x = static_cast<float>(center_w.x() - 0.5 * cube_length_xy_m);
  const float max_x = static_cast<float>(center_w.x() + 0.5 * cube_length_xy_m);
  const float min_y = static_cast<float>(center_w.y() - 0.5 * cube_length_xy_m);
  const float max_y = static_cast<float>(center_w.y() + 0.5 * cube_length_xy_m);
  const float min_z = static_cast<float>(center_w.z() - 0.5 * cube_height_m);
  const float max_z = static_cast<float>(center_w.z() + 0.5 * cube_height_m);

  std::vector<BoxPointType> boxes;
  boxes.reserve(6);
  auto push_box = [&boxes](float x0, float x1, float y0, float y1, float z0, float z1) {
    if (x0 >= x1 || y0 >= y1 || z0 >= z1) {
      return;
    }
    BoxPointType box {};
    box.vertex_min[0] = x0;
    box.vertex_max[0] = x1;
    box.vertex_min[1] = y0;
    box.vertex_max[1] = y1;
    box.vertex_min[2] = z0;
    box.vertex_max[2] = z1;
    boxes.push_back(box);
  };

  push_box(range.vertex_min[0], min_x, range.vertex_min[1], range.vertex_max[1], range.vertex_min[2], range.vertex_max[2]);
  push_box(max_x, range.vertex_max[0], range.vertex_min[1], range.vertex_max[1], range.vertex_min[2], range.vertex_max[2]);
  push_box(min_x, max_x, range.vertex_min[1], min_y, range.vertex_min[2], range.vertex_max[2]);
  push_box(min_x, max_x, max_y, range.vertex_max[1], range.vertex_min[2], range.vertex_max[2]);
  push_box(min_x, max_x, min_y, max_y, range.vertex_min[2], min_z);
  push_box(min_x, max_x, min_y, max_y, max_z, range.vertex_max[2]);

  if (!boxes.empty()) {
    tree_->Delete_Point_Boxes(boxes);
  }
}

std::size_t TreePointMap::size() const
{
  return static_cast<std::size_t>(std::max(0, tree_->validnum()));
}

std::size_t TreePointMap::radiusSearchPoints(
  const Eigen::Vector3d & query_w,
  double radius_m,
  std::size_t max_results,
  std::vector<Eigen::Vector3d> * neighbors) const
{
  if (neighbors == nullptr || radius_m <= 0.0 || max_results == 0U || tree_->validnum() <= 0) {
    return 0U;
  }

  typename IkdTreeType::PointVector found_points;
  std::vector<float> point_distances;
  tree_->Nearest_Search(
    toPclPoint(query_w),
    static_cast<int>(max_results),
    found_points,
    point_distances,
    radius_m);
  if (found_points.empty()) {
    neighbors->clear();
    return 0U;
  }

  const std::size_t take = std::min(max_results, found_points.size());
  neighbors->clear();
  neighbors->reserve(take);
  for (std::size_t i = 0; i < take; ++i) {
    const auto & pt = found_points[i];
    neighbors->emplace_back(pt.x, pt.y, pt.z);
  }
  return take;
}

}  // namespace iekf_lio
