#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace iekf_lio
{

class VoxelMap
{
public:
  explicit VoxelMap(double voxel_size_m = 0.5, std::size_t max_blocks = 200);

  void setVoxelSize(double voxel_size_m);
  void setBlockSize(double block_size_m);
  void setMaxBlocks(std::size_t max_blocks);
  void setHistoryWindow(double radius_xy_m, double half_height_m);
  void setHistoryWindowEnabled(bool enabled);
  void setActiveWindow(double radius_xy_m, double half_height_m);
  void setActiveWindowEnabled(bool enabled);
  void setActiveAngleFilterEnabled(bool enabled);
  void setActiveHalfFovDeg(double half_fov_deg);
  void clear();

  // Insert world-frame points into voxel map, one centroid representative per voxel.
  std::size_t insertPoints(const std::vector<Eigen::Vector3d> & points_w);
  // Keep stored history blocks within configured history radius and block budget.
  std::size_t pruneHistoryBlocks(const Eigen::Vector3d & center_w);
  pcl::PointCloud<pcl::PointXYZ>::Ptr exportFullPointCloud() const;
  // Export the currently active local-map cloud for scan matching and
  // update the cached active-block count in the same traversal.
  pcl::PointCloud<pcl::PointXYZ>::Ptr exportActivePointCloud(
    const Eigen::Vector3d & center_w,
    const Eigen::Vector3d & lidar_origin_w,
    const Eigen::Vector3d & lidar_forward_w) const;
  std::size_t size() const;
  std::size_t blockCount() const;
  std::size_t activeBlockCount(
    const Eigen::Vector3d & center_w,
    const Eigen::Vector3d & lidar_origin_w,
    const Eigen::Vector3d & lidar_forward_w) const;
  std::size_t cachedActiveBlockCount() const;

private:
  struct VoxelKey
  {
    std::int32_t x = 0;
    std::int32_t y = 0;
    std::int32_t z = 0;

    bool operator==(const VoxelKey & other) const
    {
      return x == other.x && y == other.y && z == other.z;
    }
  };

  struct VoxelKeyHash
  {
    std::size_t operator()(const VoxelKey & k) const;
  };

  struct BlockKey
  {
    std::int32_t x = 0;
    std::int32_t y = 0;
    std::int32_t z = 0;

    bool operator==(const BlockKey & other) const
    {
      return x == other.x && y == other.y && z == other.z;
    }
  };

  struct BlockKeyHash
  {
    std::size_t operator()(const BlockKey & k) const;
  };

  struct VoxelData
  {
    Eigen::Vector3d sum = Eigen::Vector3d::Zero();
    std::size_t count = 0;
  };

  struct BlockData
  {
    std::unordered_map<VoxelKey, VoxelData, VoxelKeyHash> voxels;
  };

  VoxelKey pointToKey(const Eigen::Vector3d & p_w) const;
  BlockKey pointToBlockKey(const Eigen::Vector3d & p_w) const;
  Eigen::Vector3d blockCenter(const BlockKey & key) const;
  Eigen::Vector3d voxelCentroid(const VoxelData & voxel) const;
  bool isBlockActive(
    const BlockKey & key,
    const Eigen::Vector3d & center_w,
    const Eigen::Vector3d & lidar_origin_w,
    const Eigen::Vector3d & lidar_forward_w) const;
  void appendBlockToCloud(
    const BlockData & block,
    pcl::PointCloud<pcl::PointXYZ> & cloud) const;

  double voxel_size_m_ = 0.5;
  double block_size_m_ = 20.0;
  std::size_t max_blocks_ = 200;
  bool history_window_enabled_ = true;
  double history_radius_xy_m_ = 120.0;
  double history_half_height_m_ = 30.0;
  bool active_window_enabled_ = true;
  double active_radius_xy_m_ = 60.0;
  double active_half_height_m_ = 15.0;
  bool active_angle_filter_enabled_ = false;
  double active_half_fov_rad_ = 0.5235987755982988;  // 30 deg
  std::size_t total_voxels_ = 0;
  mutable std::size_t cached_active_blocks_ = 0;
  std::unordered_map<BlockKey, BlockData, BlockKeyHash> blocks_;
};

}  // namespace iekf_lio
