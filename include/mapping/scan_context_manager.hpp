#pragma once

#include <cstddef>
#include <mutex>
#include <vector>

#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace iekf_lio
{

struct ScanContextManagerConfig
{
  bool enable = false;
  int num_rings = 20;
  int num_sectors = 60;
  double max_radius_m = 80.0;
  std::size_t max_entries = 2000;
  std::size_t min_keyframes_before_loop = 30;
  std::size_t exclude_recent_keyframes = 30;
  std::size_t num_candidates = 10;
  double distance_threshold = 0.15;
};

struct ScanContextEntry
{
  std::size_t keyframe_id = 0;
  double time_s = 0.0;
  Eigen::MatrixXd descriptor;
  Eigen::VectorXd ring_key;
};

struct LoopCandidate
{
  std::size_t query_id = 0;
  std::size_t match_id = 0;
  double score = 0.0;
  double yaw_init_rad = 0.0;
  std::size_t sector_shift = 0;
};

class ScanContextManager
{
public:
  void setConfig(const ScanContextManagerConfig & config);

  ScanContextEntry makeDescriptor(
    std::size_t keyframe_id,
    double time_s,
    const pcl::PointCloud<pcl::PointXYZ>::ConstPtr & cloud) const;

  bool detectLoop(
    const ScanContextEntry & query,
    LoopCandidate * candidate) const;

  void addEntry(const ScanContextEntry & entry);
  void reset();
  std::size_t entryCount() const;

private:
  double ringKeyDistance(
    const Eigen::VectorXd & lhs,
    const Eigen::VectorXd & rhs) const;

  double descriptorDistance(
    const Eigen::MatrixXd & query,
    const Eigen::MatrixXd & reference,
    std::size_t * best_shift) const;

  Eigen::MatrixXd circularShiftColumns(
    const Eigen::MatrixXd & mat,
    std::size_t shift) const;

  mutable std::mutex mutex_;
  ScanContextManagerConfig config_;
  std::vector<ScanContextEntry> entries_;
};

}  // namespace iekf_lio
