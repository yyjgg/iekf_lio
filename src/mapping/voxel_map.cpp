#include "mapping/voxel_map.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace iekf_lio
{

VoxelMap::VoxelMap(double voxel_size_m, std::size_t max_blocks)
: voxel_size_m_(std::max(1e-3, voxel_size_m)),
  max_blocks_(std::max<std::size_t>(1, max_blocks))
{
}

void VoxelMap::setVoxelSize(double voxel_size_m)
{
  voxel_size_m_ = std::max(1e-3, voxel_size_m);
}

void VoxelMap::setBlockSize(double block_size_m)
{
  block_size_m_ = std::max(1e-3, block_size_m);
}

void VoxelMap::setMaxBlocks(std::size_t max_blocks)
{
  max_blocks_ = std::max<std::size_t>(1, max_blocks);
}

void VoxelMap::setHistoryWindow(double radius_xy_m, double half_height_m)
{
  history_radius_xy_m_ = std::max(1.0, radius_xy_m);
  history_half_height_m_ = std::max(0.5, half_height_m);
}

void VoxelMap::setHistoryWindowEnabled(bool enabled)
{
  history_window_enabled_ = enabled;
}

void VoxelMap::setActiveWindow(double radius_xy_m, double half_height_m)
{
  active_radius_xy_m_ = std::max(1.0, radius_xy_m);
  active_half_height_m_ = std::max(0.5, half_height_m);
}

void VoxelMap::setActiveWindowEnabled(bool enabled)
{
  active_window_enabled_ = enabled;
}

void VoxelMap::setActiveAngleFilterEnabled(bool enabled)
{
  active_angle_filter_enabled_ = enabled;
}

void VoxelMap::setActiveHalfFovDeg(double half_fov_deg)
{
  constexpr double kPi = 3.14159265358979323846;
  active_half_fov_rad_ = std::max(1.0, half_fov_deg) * kPi / 180.0;
}

void VoxelMap::clear()
{
  blocks_.clear();
  total_voxels_ = 0;
  cached_active_blocks_ = 0;
}

std::size_t VoxelMap::insertPoints(const std::vector<Eigen::Vector3d> & points_w)
{
  std::size_t inserted = 0;
  for (const auto & p_w : points_w) {
    if (!std::isfinite(p_w.x()) || !std::isfinite(p_w.y()) || !std::isfinite(p_w.z())) {
      continue;
    }

    const BlockKey block_key = pointToBlockKey(p_w);
    auto block_it = blocks_.find(block_key);
    if (block_it == blocks_.end()) {
      block_it = blocks_.emplace(block_key, BlockData {}).first;
    }

    auto & voxels = block_it->second.voxels;
    const VoxelKey voxel_key = pointToKey(p_w);
    auto voxel_it = voxels.find(voxel_key);
    if (voxel_it == voxels.end()) {
      VoxelData voxel;
      voxel.sum = p_w;
      voxel.count = 1;
      voxels.emplace(voxel_key, voxel);
      ++total_voxels_;
      ++inserted;
      continue;
    }

    voxel_it->second.sum += p_w;
    ++voxel_it->second.count;
  }
  return inserted;
}

std::size_t VoxelMap::pruneHistoryBlocks(const Eigen::Vector3d & center_w)
{
  if (blocks_.empty()) {
    return 0;
  }

  std::size_t erased = 0;
  if (history_window_enabled_) {
    const double r2_xy = history_radius_xy_m_ * history_radius_xy_m_;
    for (auto block_it = blocks_.begin(); block_it != blocks_.end();) {
      const Eigen::Vector3d center = blockCenter(block_it->first);
      const double dx = center.x() - center_w.x();
      const double dy = center.y() - center_w.y();
      const double dz = center.z() - center_w.z();
      const bool out_xy = (dx * dx + dy * dy) > r2_xy;
      const bool out_z = std::abs(dz) > history_half_height_m_;
      if (out_xy || out_z) {
        erased += block_it->second.voxels.size();
        total_voxels_ -= block_it->second.voxels.size();
        block_it = blocks_.erase(block_it);
      } else {
        ++block_it;
      }
    }
  }

  if (blocks_.size() > max_blocks_) {
    struct BlockDistance
    {
      BlockKey key;
      double dist2_xy = 0.0;
      double abs_dz = 0.0;
    };

    std::vector<BlockDistance> distances;
    distances.reserve(blocks_.size());
    for (const auto & block_kv : blocks_) {
      const Eigen::Vector3d center = blockCenter(block_kv.first);
      const double dx = center.x() - center_w.x();
      const double dy = center.y() - center_w.y();
      const double dz = center.z() - center_w.z();
      distances.push_back(BlockDistance {
          block_kv.first,
          dx * dx + dy * dy,
          std::abs(dz)});
    }

    std::sort(
      distances.begin(),
      distances.end(),
      [](const BlockDistance & a, const BlockDistance & b) {
        if (a.dist2_xy != b.dist2_xy) {
          return a.dist2_xy > b.dist2_xy;
        }
        return a.abs_dz > b.abs_dz;
      });

    std::size_t remove_count = blocks_.size() - max_blocks_;
    for (std::size_t i = 0; i < remove_count; ++i) {
      const auto block_it = blocks_.find(distances[i].key);
      if (block_it == blocks_.end()) {
        continue;
      }
      erased += block_it->second.voxels.size();
      total_voxels_ -= block_it->second.voxels.size();
      blocks_.erase(block_it);
    }
  }

  return erased;
}

std::size_t VoxelMap::size() const
{
  return total_voxels_;
}

std::size_t VoxelMap::blockCount() const
{
  return blocks_.size();
}

std::size_t VoxelMap::activeBlockCount(
  const Eigen::Vector3d & center_w,
  const Eigen::Vector3d & lidar_origin_w,
  const Eigen::Vector3d & lidar_forward_w) const
{
  if (!active_window_enabled_) {
    cached_active_blocks_ = blocks_.size();
    return blocks_.size();
  }

  std::size_t count = 0;
  const BlockKey center_block = pointToBlockKey(center_w);
  const int range_xy = std::max(0, static_cast<int>(std::ceil(active_radius_xy_m_ / block_size_m_)));
  const int range_z = std::max(0, static_cast<int>(std::ceil(active_half_height_m_ / block_size_m_)));

  for (int dx = -range_xy; dx <= range_xy; ++dx) {
    for (int dy = -range_xy; dy <= range_xy; ++dy) {
      for (int dz = -range_z; dz <= range_z; ++dz) {
        const BlockKey key {
          center_block.x + dx,
          center_block.y + dy,
          center_block.z + dz};
        if (blocks_.find(key) != blocks_.end() &&
          isBlockActive(key, center_w, lidar_origin_w, lidar_forward_w))
        {
          ++count;
        }
      }
    }
  }

  cached_active_blocks_ = count;
  return count;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr VoxelMap::exportFullPointCloud() const
{
  auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  cloud->reserve(total_voxels_);
  for (const auto & block_kv : blocks_) {
    appendBlockToCloud(block_kv.second, *cloud);
  }
  cloud->width = static_cast<std::uint32_t>(cloud->size());
  cloud->height = 1;
  cloud->is_dense = false;
  return cloud;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr VoxelMap::exportActivePointCloud(
  const Eigen::Vector3d & center_w,
  const Eigen::Vector3d & lidar_origin_w,
  const Eigen::Vector3d & lidar_forward_w) const
{
  if (!active_window_enabled_) {
    cached_active_blocks_ = blocks_.size();
    return exportFullPointCloud();
  }

  auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  cloud->reserve(total_voxels_);
  std::size_t active_blocks = 0;

  const BlockKey center_block = pointToBlockKey(center_w);
  const int range_xy = std::max(0, static_cast<int>(std::ceil(active_radius_xy_m_ / block_size_m_)));
  const int range_z = std::max(0, static_cast<int>(std::ceil(active_half_height_m_ / block_size_m_)));

  for (int dx = -range_xy; dx <= range_xy; ++dx) {
    for (int dy = -range_xy; dy <= range_xy; ++dy) {
      for (int dz = -range_z; dz <= range_z; ++dz) {
        const BlockKey key {
          center_block.x + dx,
          center_block.y + dy,
          center_block.z + dz};
        const auto it = blocks_.find(key);
        if (it == blocks_.end()) {
          continue;
        }
        if (!isBlockActive(key, center_w, lidar_origin_w, lidar_forward_w)) {
          continue;
        }
        ++active_blocks;
        appendBlockToCloud(it->second, *cloud);
      }
    }
  }

  cached_active_blocks_ = active_blocks;
  cloud->width = static_cast<std::uint32_t>(cloud->size());
  cloud->height = 1;
  cloud->is_dense = false;
  return cloud;
}

std::size_t VoxelMap::cachedActiveBlockCount() const
{
  return cached_active_blocks_;
}

std::size_t VoxelMap::VoxelKeyHash::operator()(const VoxelKey & k) const
{
  // Spatial hash with large coprime constants.
  const std::uint64_t hx = static_cast<std::uint64_t>(static_cast<std::uint32_t>(k.x)) * 73856093ULL;
  const std::uint64_t hy = static_cast<std::uint64_t>(static_cast<std::uint32_t>(k.y)) * 19349663ULL;
  const std::uint64_t hz = static_cast<std::uint64_t>(static_cast<std::uint32_t>(k.z)) * 83492791ULL;
  return static_cast<std::size_t>(hx ^ hy ^ hz);
}

std::size_t VoxelMap::BlockKeyHash::operator()(const BlockKey & k) const
{
  const std::uint64_t hx = static_cast<std::uint64_t>(static_cast<std::uint32_t>(k.x)) * 1640531513ULL;
  const std::uint64_t hy = static_cast<std::uint64_t>(static_cast<std::uint32_t>(k.y)) * 2654435761ULL;
  const std::uint64_t hz = static_cast<std::uint64_t>(static_cast<std::uint32_t>(k.z)) * 805459861ULL;
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

VoxelMap::BlockKey VoxelMap::pointToBlockKey(const Eigen::Vector3d & p_w) const
{
  const double inv = 1.0 / block_size_m_;
  BlockKey key;
  key.x = static_cast<std::int32_t>(std::floor(p_w.x() * inv));
  key.y = static_cast<std::int32_t>(std::floor(p_w.y() * inv));
  key.z = static_cast<std::int32_t>(std::floor(p_w.z() * inv));
  return key;
}

Eigen::Vector3d VoxelMap::blockCenter(const BlockKey & key) const
{
  return Eigen::Vector3d(
    (static_cast<double>(key.x) + 0.5) * block_size_m_,
    (static_cast<double>(key.y) + 0.5) * block_size_m_,
    (static_cast<double>(key.z) + 0.5) * block_size_m_);
}

Eigen::Vector3d VoxelMap::voxelCentroid(const VoxelData & voxel) const
{
  if (voxel.count == 0) {
    return Eigen::Vector3d::Zero();
  }
  return voxel.sum / static_cast<double>(voxel.count);
}

bool VoxelMap::isBlockActive(
  const BlockKey & key,
  const Eigen::Vector3d & center_w,
  const Eigen::Vector3d & lidar_origin_w,
  const Eigen::Vector3d & lidar_forward_w) const
{
  const Eigen::Vector3d center = blockCenter(key);
  const double dx = center.x() - center_w.x();
  const double dy = center.y() - center_w.y();
  const double dz = center.z() - center_w.z();
  const bool inside_radius = (dx * dx + dy * dy) <= (active_radius_xy_m_ * active_radius_xy_m_);
  const bool inside_height = std::abs(dz) <= active_half_height_m_;
  if (!inside_radius || !inside_height) {
    return false;
  }

  if (!active_angle_filter_enabled_) {
    return true;
  }

  Eigen::Vector2d forward_xy(lidar_forward_w.x(), lidar_forward_w.y());
  const double forward_norm = forward_xy.norm();
  if (forward_norm <= 1e-9) {
    return true;
  }
  forward_xy /= forward_norm;

  Eigen::Vector2d dir_xy(center.x() - lidar_origin_w.x(), center.y() - lidar_origin_w.y());
  const double dir_norm = dir_xy.norm();
  if (dir_norm <= 1e-9) {
    return true;
  }
  dir_xy /= dir_norm;

  const double cos_angle = std::clamp(forward_xy.dot(dir_xy), -1.0, 1.0);
  return cos_angle >= std::cos(active_half_fov_rad_);
}

void VoxelMap::appendBlockToCloud(
  const BlockData & block,
  pcl::PointCloud<pcl::PointXYZ> & cloud) const
{
  for (const auto & voxel_kv : block.voxels) {
    const Eigen::Vector3d centroid = voxelCentroid(voxel_kv.second);
    pcl::PointXYZ p;
    p.x = static_cast<float>(centroid.x());
    p.y = static_cast<float>(centroid.y());
    p.z = static_cast<float>(centroid.z());
    cloud.push_back(p);
  }
}

}  // namespace iekf_lio
