#include "mapping/voxel_map.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#if defined(_OPENMP)
#include <omp.h>
#endif

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

void VoxelMap::clear()
{
  blocks_.clear();
  voxel_index_.clear();
  total_voxels_ = 0;
}

std::size_t VoxelMap::insertPoints(const std::vector<Eigen::Vector3d> & points_w)
{
  BlockMap frame_blocks;
  frame_blocks.reserve(std::max<std::size_t>(1, points_w.size() / 16));
  for (const auto & p_w : points_w) {
    if (!std::isfinite(p_w.x()) || !std::isfinite(p_w.y()) || !std::isfinite(p_w.z())) {
      continue;
    }
    accumulatePointIntoBlocks(p_w, frame_blocks);
  }
  return mergeBlocks(frame_blocks);
}

VoxelMap::InsertResult VoxelMap::insertTransformedScan(
  const pcl::PointCloud<pcl::PointXYZ> & scan_i,
  const Eigen::Matrix3d & r_wi,
  const Eigen::Vector3d & p_wi)
{
  InsertResult result;
  if (scan_i.empty()) {
    return result;
  }

#if defined(_OPENMP)
  const int thread_count = std::max(1, omp_get_max_threads());
#else
  const int thread_count = 1;
#endif
  std::vector<BlockMap> thread_blocks(static_cast<std::size_t>(thread_count));
  const std::size_t reserve_blocks =
    std::max<std::size_t>(1, scan_i.size() / (16 * static_cast<std::size_t>(thread_count)));
  for (auto & local_blocks : thread_blocks) {
    local_blocks.reserve(reserve_blocks);
  }

  std::size_t valid_points = 0;
#pragma omp parallel reduction(+:valid_points)
  {
#if defined(_OPENMP)
    const int tid = omp_get_thread_num();
#else
    const int tid = 0;
#endif
    auto & local_blocks = thread_blocks[static_cast<std::size_t>(tid)];

#pragma omp for schedule(static)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(scan_i.size()); ++i) {
      const auto & pt = scan_i[static_cast<std::size_t>(i)];
      if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
        continue;
      }
      const Eigen::Vector3d p_i(pt.x, pt.y, pt.z);
      const Eigen::Vector3d p_w = r_wi * p_i + p_wi;
      accumulatePointIntoBlocks(p_w, local_blocks);
      ++valid_points;
    }
  }

  BlockMap frame_blocks;
  frame_blocks.reserve(std::max<std::size_t>(1, scan_i.size() / 16));
  for (const auto & local_blocks : thread_blocks) {
    mergeIntoBlocks(frame_blocks, local_blocks);
  }
  result.valid_points = valid_points;
  result.inserted_voxels = mergeBlocks(frame_blocks);
  return result;
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
        for (const auto & voxel_kv : block_it->second.voxels) {
          voxel_index_.erase(voxel_kv.first);
        }
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
      for (const auto & voxel_kv : block_it->second.voxels) {
        voxel_index_.erase(voxel_kv.first);
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

std::size_t VoxelMap::radiusSearchVoxels(
  const Eigen::Vector3d & query_w,
  double radius_m,
  std::size_t max_results,
  std::vector<NearbyVoxel> * neighbors) const
{
  if (neighbors == nullptr) {
    return 0;
  }
  neighbors->clear();
  if (!std::isfinite(query_w.x()) || !std::isfinite(query_w.y()) || !std::isfinite(query_w.z())) {
    return 0;
  }

  const double radius = std::max(1e-3, radius_m);
  const double radius2 = radius * radius;
  const VoxelKey center_key = pointToKey(query_w);
  const int voxel_range = std::max(0, static_cast<int>(std::ceil(radius / voxel_size_m_)));
  const auto max_heap_cmp = [](const NearbyVoxel & a, const NearbyVoxel & b) {
      return a.dist2 < b.dist2;
    };

  for (int dx = -voxel_range; dx <= voxel_range; ++dx) {
    for (int dy = -voxel_range; dy <= voxel_range; ++dy) {
      for (int dz = -voxel_range; dz <= voxel_range; ++dz) {
        const VoxelKey voxel_key {
          center_key.x + dx,
          center_key.y + dy,
          center_key.z + dz};
        const auto voxel_ref_it = voxel_index_.find(voxel_key);
        if (voxel_ref_it == voxel_index_.end() || voxel_ref_it->second.voxel == nullptr) {
          continue;
        }

        const VoxelData & voxel = *voxel_ref_it->second.voxel;
        const Eigen::Vector3d & centroid = voxel.centroid;
        const double dist2 = (centroid - query_w).squaredNorm();
        if (dist2 > radius2) {
          continue;
        }

        NearbyVoxel sample;
        sample.centroid_w = centroid;
        sample.sum_w = voxel.sum;
        sample.sum_outer_w = voxel.sum_outer;
        sample.count = voxel.count;
        sample.dist2 = dist2;
        if (max_results == 0) {
          neighbors->push_back(sample);
          continue;
        }

        if (neighbors->size() < max_results) {
          neighbors->push_back(sample);
          std::push_heap(neighbors->begin(), neighbors->end(), max_heap_cmp);
          continue;
        }

        if (dist2 >= neighbors->front().dist2) {
          continue;
        }

        std::pop_heap(neighbors->begin(), neighbors->end(), max_heap_cmp);
        neighbors->back() = sample;
        std::push_heap(neighbors->begin(), neighbors->end(), max_heap_cmp);
      }
    }
  }

  if (max_results == 0) {
    std::sort(
      neighbors->begin(),
      neighbors->end(),
      [](const NearbyVoxel & a, const NearbyVoxel & b) {
        return a.dist2 < b.dist2;
      });
  } else {
    std::sort_heap(neighbors->begin(), neighbors->end(), max_heap_cmp);
  }
  return neighbors->size();
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

VoxelMap::BlockKey VoxelMap::voxelKeyToBlockKey(const VoxelKey & key) const
{
  const Eigen::Vector3d center_w(
    (static_cast<double>(key.x) + 0.5) * voxel_size_m_,
    (static_cast<double>(key.y) + 0.5) * voxel_size_m_,
    (static_cast<double>(key.z) + 0.5) * voxel_size_m_);
  return pointToBlockKey(center_w);
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
  return voxel.centroid;
}

Eigen::Matrix3d VoxelMap::voxelCovariance(const VoxelData & voxel) const
{
  if (voxel.count < 2) {
    return Eigen::Matrix3d::Zero();
  }
  const Eigen::Vector3d mean = voxelCentroid(voxel);
  Eigen::Matrix3d cov =
    (voxel.sum_outer - static_cast<double>(voxel.count) * mean * mean.transpose()) /
    static_cast<double>(voxel.count - 1);
  cov = 0.5 * (cov + cov.transpose());
  return cov;
}

void VoxelMap::accumulatePointIntoBlocks(const Eigen::Vector3d & p_w, BlockMap & blocks) const
{
  const BlockKey block_key = pointToBlockKey(p_w);
  auto block_it = blocks.find(block_key);
  if (block_it == blocks.end()) {
    block_it = blocks.emplace(block_key, BlockData {}).first;
  }

  auto & voxels = block_it->second.voxels;
  const VoxelKey voxel_key = pointToKey(p_w);
  auto voxel_it = voxels.find(voxel_key);
  if (voxel_it == voxels.end()) {
    VoxelData voxel;
    voxel.centroid = p_w;
    voxel.sum = p_w;
    voxel.sum_outer = p_w * p_w.transpose();
    voxel.count = 1;
    voxels.emplace(voxel_key, voxel);
    return;
  }

  voxel_it->second.sum += p_w;
  voxel_it->second.sum_outer += p_w * p_w.transpose();
  ++voxel_it->second.count;
  voxel_it->second.centroid = voxel_it->second.sum / static_cast<double>(voxel_it->second.count);
}

std::size_t VoxelMap::mergeBlocks(const BlockMap & frame_blocks)
{
  std::size_t inserted = 0;
  for (const auto & src_block_kv : frame_blocks) {
    auto block_it = blocks_.find(src_block_kv.first);
    if (block_it == blocks_.end()) {
      block_it = blocks_.emplace(src_block_kv.first, BlockData {}).first;
    }

    auto & target_voxels = block_it->second.voxels;
    for (const auto & src_voxel_kv : src_block_kv.second.voxels) {
      auto voxel_it = target_voxels.find(src_voxel_kv.first);
      if (voxel_it == target_voxels.end()) {
        auto [new_it, ok] = target_voxels.emplace(src_voxel_kv.first, src_voxel_kv.second);
        if (ok) {
          voxel_index_[src_voxel_kv.first] = VoxelRef {&new_it->second};
          ++inserted;
        }
        continue;
      }

      voxel_it->second.sum += src_voxel_kv.second.sum;
      voxel_it->second.sum_outer += src_voxel_kv.second.sum_outer;
      voxel_it->second.count += src_voxel_kv.second.count;
      voxel_it->second.centroid =
        voxel_it->second.sum / static_cast<double>(voxel_it->second.count);
    }
  }
  total_voxels_ += inserted;
  return inserted;
}

std::size_t VoxelMap::mergeIntoBlocks(BlockMap & target, const BlockMap & source) const
{
  std::size_t inserted = 0;
  for (const auto & src_block_kv : source) {
    auto block_it = target.find(src_block_kv.first);
    if (block_it == target.end()) {
      block_it = target.emplace(src_block_kv.first, BlockData {}).first;
    }

    auto & target_voxels = block_it->second.voxels;
    for (const auto & src_voxel_kv : src_block_kv.second.voxels) {
      auto voxel_it = target_voxels.find(src_voxel_kv.first);
      if (voxel_it == target_voxels.end()) {
        target_voxels.emplace(src_voxel_kv.first, src_voxel_kv.second);
        ++inserted;
        continue;
      }

      voxel_it->second.sum += src_voxel_kv.second.sum;
      voxel_it->second.sum_outer += src_voxel_kv.second.sum_outer;
      voxel_it->second.count += src_voxel_kv.second.count;
      voxel_it->second.centroid =
        voxel_it->second.sum / static_cast<double>(voxel_it->second.count);
    }
  }
  return inserted;
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
