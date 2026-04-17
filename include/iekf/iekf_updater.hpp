#pragma once

#include <cstddef>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "iekf/iekf_state.hpp"
#include "types.hpp"

namespace iekf_lio
{

struct IekfUpdaterConfig
{
  int max_iterations = 2;
  double max_correspondence_distance = 1.0;  // meter, neighbor search gate
  int plane_k_neighbors = 10;
  double plane_max_eigen_ratio = 0.15;       // planarity check: lambda0/lambda1
  double sigma_point_to_plane = 0.2;         // meter
  double max_abs_point_to_plane_residual = 0.5;  // meter
  int max_update_points = 1200;
};

struct IekfUpdateResult
{
  bool updated = false;
  std::size_t correspondences = 0;
  double rmse = 0.0;
  int iterations = 0;
};

class IekfUpdater
{
public:
  explicit IekfUpdater(IekfUpdaterConfig config = {}) : config_(config) {}

  void setConfig(const IekfUpdaterConfig & config) { config_ = config; }

  bool updatePoseWithPointToMap(
    const LidarScanXYZ & scan_i_end,
    const pcl::PointCloud<pcl::PointXYZ>::ConstPtr & map_cloud_w,
    IekfState18 & state,
    IekfUpdateResult * result = nullptr) const;

private:
  IekfUpdaterConfig config_;
};

}  // namespace iekf_lio
