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
  explicit VoxelMap(double voxel_size_m = 0.5, std::size_t max_voxels = 200000);

  void setVoxelSize(double voxel_size_m);
  void setMaxVoxels(std::size_t max_voxels);
  void setLocalWindow(double radius_xy_m, double half_height_m);
  void setLocalWindowEnabled(bool enabled);
  void clear();

  // Insert world-frame points into voxel map, one representative point per voxel.
  std::size_t insertPoints(const std::vector<Eigen::Vector3d> & points_w);
  // Keep only voxels within local window centered at center_w.
  std::size_t pruneOutsideLocalWindow(const Eigen::Vector3d & center_w);
  pcl::PointCloud<pcl::PointXYZ>::Ptr toPointCloud() const;
  std::size_t size() const;

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

  VoxelKey pointToKey(const Eigen::Vector3d & p_w) const;

  double voxel_size_m_ = 0.5;
  std::size_t max_voxels_ = 200000;
  bool local_window_enabled_ = true;
  double window_radius_xy_m_ = 60.0;
  double window_half_height_m_ = 15.0;
  std::unordered_map<VoxelKey, pcl::PointXYZ, VoxelKeyHash> voxels_;
};

}  // namespace iekf_lio
