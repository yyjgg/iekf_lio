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
  max_voxels_(max_blocks)
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
  max_voxels_ = max_blocks;
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
  voxels_.clear();
  total_voxels_ = 0;
}

std::size_t VoxelMap::insertPoints(const std::vector<Eigen::Vector3d> & points_w)
{
  std::unordered_map<VoxelKey, VoxelData, VoxelKeyHash> frame_voxels;
  frame_voxels.reserve(std::max<std::size_t>(1, points_w.size() / 4));
  for (const auto & p_w : points_w) {
    if (!std::isfinite(p_w.x()) || !std::isfinite(p_w.y()) || !std::isfinite(p_w.z())) {
      continue;
    }
    accumulatePointIntoVoxels(p_w, frame_voxels);
  }
  return mergeVoxels(frame_voxels);
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
  const auto frame_voxels = aggregateTransformedScanVoxels(scan_i, r_wi, p_wi, &result.valid_points);
  result.inserted_voxels = mergeVoxels(frame_voxels);
  return result;
}

std::vector<VoxelMap::VoxelContribution> VoxelMap::buildTransformedScanContributions(
  const pcl::PointCloud<pcl::PointXYZ> & scan_i,
  const Eigen::Matrix3d & r_wi,
  const Eigen::Vector3d & p_wi) const
{
  const auto frame_voxels = aggregateTransformedScanVoxels(scan_i, r_wi, p_wi, nullptr);
  std::vector<VoxelContribution> contributions;
  contributions.reserve(frame_voxels.size());
  for (const auto & voxel_kv : frame_voxels) {
    VoxelContribution contribution;
    contribution.key = voxel_kv.first;
    contribution.sum_w = voxel_kv.second.sum;
    contribution.sum_outer_w = voxel_kv.second.sum_outer;
    contribution.count = voxel_kv.second.count;
    contributions.push_back(contribution);
  }
  return contributions;
}

std::size_t VoxelMap::addVoxelContributions(const std::vector<VoxelContribution> & contributions)
{
  std::size_t inserted = 0;
  for (const auto & contribution : contributions) {
    if (contribution.count == 0) {
      continue;
    }

    auto voxel_it = voxels_.find(contribution.key);
    if (voxel_it == voxels_.end()) {
      VoxelData voxel;
      voxel.sum = contribution.sum_w;
      voxel.sum_outer = contribution.sum_outer_w;
      voxel.count = contribution.count;
      voxel.centroid = voxel.sum / static_cast<double>(voxel.count);
      voxels_.emplace(contribution.key, voxel);
      ++inserted;
      continue;
    }

    voxel_it->second.sum += contribution.sum_w;
    voxel_it->second.sum_outer += contribution.sum_outer_w;
    voxel_it->second.count += contribution.count;
    voxel_it->second.centroid =
      voxel_it->second.sum / static_cast<double>(voxel_it->second.count);
  }

  total_voxels_ = voxels_.size();
  return inserted;
}

std::size_t VoxelMap::removeVoxelContributions(const std::vector<VoxelContribution> & contributions)
{
  std::size_t erased = 0;
  for (const auto & contribution : contributions) {
    if (contribution.count == 0) {
      continue;
    }

    auto voxel_it = voxels_.find(contribution.key);
    if (voxel_it == voxels_.end()) {
      continue;
    }

    if (voxel_it->second.count <= contribution.count) {
      voxels_.erase(voxel_it);
      ++erased;
      continue;
    }

    voxel_it->second.sum -= contribution.sum_w;
    voxel_it->second.sum_outer -= contribution.sum_outer_w;
    voxel_it->second.count -= contribution.count;
    voxel_it->second.centroid =
      voxel_it->second.sum / static_cast<double>(voxel_it->second.count);
  }

  total_voxels_ = voxels_.size();
  return erased;
}

std::size_t VoxelMap::pruneHistoryBlocks(const Eigen::Vector3d & center_w)
{
  std::size_t erased = 0;
  if (history_window_enabled_) {
    const double r2_xy = history_radius_xy_m_ * history_radius_xy_m_;
    for (auto it = voxels_.begin(); it != voxels_.end();) {
      const Eigen::Vector3d center = voxelCentroid(it->second);
      const double dx = center.x() - center_w.x();
      const double dy = center.y() - center_w.y();
      const double dz = center.z() - center_w.z();
      const bool out_xy = (dx * dx + dy * dy) > r2_xy;
      const bool out_z = std::abs(dz) > history_half_height_m_;
      if (out_xy || out_z) {
        it = voxels_.erase(it);
        ++erased;
      } else {
        ++it;
      }
    }
  }

  if (max_voxels_ > 0 && voxels_.size() > max_voxels_) {
    struct VoxelDistance
    {
      VoxelKey key;
      double dist2_xy = 0.0;
      double abs_dz = 0.0;
    };

    std::vector<VoxelDistance> distances;
    distances.reserve(voxels_.size());
    for (const auto & voxel_kv : voxels_) {
      const Eigen::Vector3d center = voxelCentroid(voxel_kv.second);
      const double dx = center.x() - center_w.x();
      const double dy = center.y() - center_w.y();
      const double dz = center.z() - center_w.z();
      distances.push_back(VoxelDistance {voxel_kv.first, dx * dx + dy * dy, std::abs(dz)});
    }

    std::sort(
      distances.begin(),
      distances.end(),
      [](const VoxelDistance & a, const VoxelDistance & b) {
        if (a.dist2_xy != b.dist2_xy) {
          return a.dist2_xy > b.dist2_xy;
        }
        return a.abs_dz > b.abs_dz;
      });

    const std::size_t remove_count = voxels_.size() - max_voxels_;
    for (std::size_t i = 0; i < remove_count; ++i) {
      erased += static_cast<std::size_t>(voxels_.erase(distances[i].key));
    }
  }

  total_voxels_ = voxels_.size();
  return erased;
}

std::size_t VoxelMap::size() const
{
  return total_voxels_;
}

std::size_t VoxelMap::blockCount() const
{
  return voxels_.size();
}

pcl::PointCloud<pcl::PointXYZ>::Ptr VoxelMap::exportFullPointCloud() const
{
  auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  cloud->reserve(total_voxels_);
  for (const auto & voxel_kv : voxels_) {
    const Eigen::Vector3d centroid = voxelCentroid(voxel_kv.second);
    pcl::PointXYZ p;
    p.x = static_cast<float>(centroid.x());
    p.y = static_cast<float>(centroid.y());
    p.z = static_cast<float>(centroid.z());
    cloud->push_back(p);
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
        const auto voxel_it = voxels_.find(voxel_key);
        if (voxel_it == voxels_.end()) {
          continue;
        }

        const VoxelData & voxel = voxel_it->second;
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

void VoxelMap::accumulatePointIntoVoxels(
  const Eigen::Vector3d & p_w,
  std::unordered_map<VoxelKey, VoxelData, VoxelKeyHash> & voxels) const
{
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

std::unordered_map<VoxelMap::VoxelKey, VoxelMap::VoxelData, VoxelMap::VoxelKeyHash>
VoxelMap::aggregateTransformedScanVoxels(
  const pcl::PointCloud<pcl::PointXYZ> & scan_i,
  const Eigen::Matrix3d & r_wi,
  const Eigen::Vector3d & p_wi,
  std::size_t * valid_points) const
{
#if defined(_OPENMP)
  const int thread_count = std::max(1, omp_get_max_threads());
#else
  const int thread_count = 1;
#endif
  std::vector<std::unordered_map<VoxelKey, VoxelData, VoxelKeyHash>> thread_voxels(
    static_cast<std::size_t>(thread_count));
  const std::size_t reserve_voxels =
    std::max<std::size_t>(1, scan_i.size() / (4 * static_cast<std::size_t>(thread_count)));
  for (auto & local_voxels : thread_voxels) {
    local_voxels.reserve(reserve_voxels);
  }

  std::size_t local_valid_points = 0;
#pragma omp parallel reduction(+:local_valid_points)
  {
#if defined(_OPENMP)
    const int tid = omp_get_thread_num();
#else
    const int tid = 0;
#endif
    auto & local_voxels = thread_voxels[static_cast<std::size_t>(tid)];

#pragma omp for schedule(static)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(scan_i.size()); ++i) {
      const auto & pt = scan_i[static_cast<std::size_t>(i)];
      if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
        continue;
      }
      const Eigen::Vector3d p_i(pt.x, pt.y, pt.z);
      const Eigen::Vector3d p_w = r_wi * p_i + p_wi;
      accumulatePointIntoVoxels(p_w, local_voxels);
      ++local_valid_points;
    }
  }

  std::unordered_map<VoxelKey, VoxelData, VoxelKeyHash> frame_voxels;
  frame_voxels.reserve(std::max<std::size_t>(1, scan_i.size() / 4));
  for (const auto & local_voxels : thread_voxels) {
    for (const auto & voxel_kv : local_voxels) {
      auto it = frame_voxels.find(voxel_kv.first);
      if (it == frame_voxels.end()) {
        frame_voxels.emplace(voxel_kv.first, voxel_kv.second);
        continue;
      }
      it->second.sum += voxel_kv.second.sum;
      it->second.sum_outer += voxel_kv.second.sum_outer;
      it->second.count += voxel_kv.second.count;
      it->second.centroid = it->second.sum / static_cast<double>(it->second.count);
    }
  }

  if (valid_points != nullptr) {
    *valid_points = local_valid_points;
  }
  return frame_voxels;
}

std::size_t VoxelMap::mergeVoxels(
  const std::unordered_map<VoxelKey, VoxelData, VoxelKeyHash> & frame_voxels)
{
  std::size_t inserted = 0;
  for (const auto & src_voxel_kv : frame_voxels) {
    auto voxel_it = voxels_.find(src_voxel_kv.first);
    if (voxel_it == voxels_.end()) {
      voxels_.emplace(src_voxel_kv.first, src_voxel_kv.second);
      ++inserted;
      continue;
    }

    voxel_it->second.sum += src_voxel_kv.second.sum;
    voxel_it->second.sum_outer += src_voxel_kv.second.sum_outer;
    voxel_it->second.count += src_voxel_kv.second.count;
    voxel_it->second.centroid =
      voxel_it->second.sum / static_cast<double>(voxel_it->second.count);
  }
  total_voxels_ = voxels_.size();
  return inserted;
}

}  // namespace iekf_lio
