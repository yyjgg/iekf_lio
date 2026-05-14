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

  struct InsertResult
  {
    std::size_t valid_points = 0;
    std::size_t inserted_voxels = 0;
  };

  struct VoxelContribution
  {
    VoxelKey key;
    Eigen::Vector3d sum_w = Eigen::Vector3d::Zero();
    Eigen::Matrix3d sum_outer_w = Eigen::Matrix3d::Zero();
    std::size_t count = 0;
  };

  struct NearbyVoxel
  {
    Eigen::Vector3d centroid_w = Eigen::Vector3d::Zero();
    Eigen::Vector3d sum_w = Eigen::Vector3d::Zero();
    Eigen::Matrix3d sum_outer_w = Eigen::Matrix3d::Zero();
    std::size_t count = 0;
    double dist2 = 0.0;
  };

  explicit VoxelMap(double voxel_size_m = 0.5, std::size_t max_blocks = 200);

  void setVoxelSize(double voxel_size_m);
  void setBlockSize(double block_size_m);
  void setMaxBlocks(std::size_t max_blocks);
  void setHistoryWindow(double radius_xy_m, double half_height_m);
  void setHistoryWindowEnabled(bool enabled);
  void clear();

  // Insert world-frame points into voxel map, one centroid representative per voxel.
  std::size_t insertPoints(const std::vector<Eigen::Vector3d> & points_w);
  // Transform scan points to world and aggregate them into frame-local voxels before
  // merging into the global voxel map. This avoids repeatedly hashing the same
  // global voxel for many points from one scan.
  InsertResult insertTransformedScan(
    const pcl::PointCloud<pcl::PointXYZ> & scan_i,
    const Eigen::Matrix3d & r_wi,
    const Eigen::Vector3d & p_wi);
  std::vector<VoxelContribution> buildTransformedScanContributions(
    const pcl::PointCloud<pcl::PointXYZ> & scan_i,
    const Eigen::Matrix3d & r_wi,
    const Eigen::Vector3d & p_wi) const;
  std::size_t addVoxelContributions(const std::vector<VoxelContribution> & contributions);
  std::size_t removeVoxelContributions(const std::vector<VoxelContribution> & contributions);
  // Keep stored voxels within configured history window and budget.
  std::size_t pruneHistoryBlocks(const Eigen::Vector3d & center_w);
  pcl::PointCloud<pcl::PointXYZ>::Ptr exportFullPointCloud() const;
  std::size_t size() const;
  std::size_t blockCount() const;
  std::size_t radiusSearchVoxels(
    const Eigen::Vector3d & query_w,
    double radius_m,
    std::size_t max_results,
    std::vector<NearbyVoxel> * neighbors) const;

private:
  struct VoxelKeyHash
  {
    std::size_t operator()(const VoxelKey & k) const;
  };

  struct VoxelData
  {
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    Eigen::Vector3d sum = Eigen::Vector3d::Zero();
    Eigen::Matrix3d sum_outer = Eigen::Matrix3d::Zero();
    std::size_t count = 0;
  };

  VoxelKey pointToKey(const Eigen::Vector3d & p_w) const;
  Eigen::Vector3d voxelCentroid(const VoxelData & voxel) const;
  Eigen::Matrix3d voxelCovariance(const VoxelData & voxel) const;
  void accumulatePointIntoVoxels(
    const Eigen::Vector3d & p_w,
    std::unordered_map<VoxelKey, VoxelData, VoxelKeyHash> & voxels) const;
  std::unordered_map<VoxelKey, VoxelData, VoxelKeyHash> aggregateTransformedScanVoxels(
    const pcl::PointCloud<pcl::PointXYZ> & scan_i,
    const Eigen::Matrix3d & r_wi,
    const Eigen::Vector3d & p_wi,
    std::size_t * valid_points) const;
  std::size_t mergeVoxels(
    const std::unordered_map<VoxelKey, VoxelData, VoxelKeyHash> & frame_voxels);

  double voxel_size_m_ = 0.5;
  double block_size_m_ = 20.0;
  std::size_t max_voxels_ = 0;
  bool history_window_enabled_ = true;
  double history_radius_xy_m_ = 120.0;
  double history_half_height_m_ = 30.0;
  std::size_t total_voxels_ = 0;
  std::unordered_map<VoxelKey, VoxelData, VoxelKeyHash> voxels_;
};

}  // namespace iekf_lio
