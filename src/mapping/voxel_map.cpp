#include "mapping/voxel_map.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace iekf_lio
{

VoxelMap::VoxelMap(double voxel_size_m, std::size_t max_voxels)
: voxel_size_m_(std::max(1e-3, voxel_size_m)),
  max_voxels_(std::max<std::size_t>(1, max_voxels))
{
}

void VoxelMap::setVoxelSize(double voxel_size_m)
{
  voxel_size_m_ = std::max(1e-3, voxel_size_m);
}

void VoxelMap::setMaxVoxels(std::size_t max_voxels)
{
  max_voxels_ = std::max<std::size_t>(1, max_voxels);
  if (voxels_.size() > max_voxels_) {
    voxels_.clear();
  }
}

void VoxelMap::setLocalWindow(double radius_xy_m, double half_height_m)
{
  window_radius_xy_m_ = std::max(1.0, radius_xy_m);
  window_half_height_m_ = std::max(0.5, half_height_m);
}

void VoxelMap::setLocalWindowEnabled(bool enabled)
{
  local_window_enabled_ = enabled;
}

void VoxelMap::clear()
{
  voxels_.clear();
}

std::size_t VoxelMap::insertPoints(const std::vector<Eigen::Vector3d> & points_w)
{
  std::size_t inserted = 0;
  for (const auto & p_w : points_w) {
    if (!std::isfinite(p_w.x()) || !std::isfinite(p_w.y()) || !std::isfinite(p_w.z())) {
      continue;
    }

    const VoxelKey key = pointToKey(p_w);
    pcl::PointXYZ p;
    p.x = static_cast<float>(p_w.x());
    p.y = static_cast<float>(p_w.y());
    p.z = static_cast<float>(p_w.z());
    const auto [it, ok] = voxels_.emplace(key, p);
    (void)it;
    if (ok) {
      ++inserted;
      if (voxels_.size() >= max_voxels_) {
        break;
      }
    }
  }
  return inserted;
}

std::size_t VoxelMap::pruneOutsideLocalWindow(const Eigen::Vector3d & center_w)
{
  if (!local_window_enabled_ || voxels_.empty()) {
    return 0;
  }

  const double r2_xy = window_radius_xy_m_ * window_radius_xy_m_;
  std::size_t erased = 0;
  for (auto it = voxels_.begin(); it != voxels_.end();) {
    const double dx = static_cast<double>(it->second.x) - center_w.x();
    const double dy = static_cast<double>(it->second.y) - center_w.y();
    const double dz = static_cast<double>(it->second.z) - center_w.z();
    const bool out_xy = (dx * dx + dy * dy) > r2_xy;
    const bool out_z = std::abs(dz) > window_half_height_m_;
    if (out_xy || out_z) {
      it = voxels_.erase(it);
      ++erased;
    } else {
      ++it;
    }
  }
  return erased;
}

std::size_t VoxelMap::size() const
{
  return voxels_.size();
}

pcl::PointCloud<pcl::PointXYZ>::Ptr VoxelMap::toPointCloud() const
{
  auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  cloud->reserve(voxels_.size());
  for (const auto & kv : voxels_) {
    cloud->push_back(kv.second);
  }
  cloud->width = static_cast<std::uint32_t>(cloud->size());
  cloud->height = 1;
  cloud->is_dense = false;
  return cloud;
}

std::size_t VoxelMap::VoxelKeyHash::operator()(const VoxelKey & k) const
{
  // Spatial hash with large coprime constants.
  const std::uint64_t hx = static_cast<std::uint64_t>(static_cast<std::uint32_t>(k.x)) * 73856093ULL;
  const std::uint64_t hy = static_cast<std::uint64_t>(static_cast<std::uint32_t>(k.y)) * 19349663ULL;
  const std::uint64_t hz = static_cast<std::uint64_t>(static_cast<std::uint32_t>(k.z)) * 83492791ULL;
  return static_cast<std::size_t>(hx ^ hy ^ hz);
}

VoxelMap::VoxelKey VoxelMap::pointToKey(const Eigen::Vector3d & p_w) const
{
  const double inv = 1.0 / voxel_size_m_;
  VoxelKey key;
  key.x = static_cast<std::int32_t>(std::floor(p_w.x() * inv));
  key.y = static_cast<std::int32_t>(std::floor(p_w.y() * inv));
  key.z = static_cast<std::int32_t>(std::floor(p_w.z() * inv));
  return key;
}

}  // namespace iekf_lio
