#pragma once

#include <cstddef>
#include <cstdint>
#include <array>
#include <vector>

#include "iekf/iekf_state.hpp"
#include "mapping/tree_point_map.hpp"
#include "types.hpp"

namespace iekf_lio
{

struct IekfUpdaterConfig
{
  int max_iterations = 2;
  double max_correspondence_distance = 1.0;  // meter, local-map radius search gate
  int plane_k_neighbors = 10;
  double plane_max_eigen_ratio = 0.15;       // planarity check: lambda0/lambda1
  double sigma_point_to_plane = 0.2;         // meter
  double max_abs_point_to_plane_residual = 0.5;  // meter
  int max_update_points = 1200;
  int min_correspondences = 0;
  double convergence_delta_pos_m = 1e-3;
  double convergence_delta_rot_rad = 1e-4;
  bool degeneracy_projection_enable = true;
  bool degeneracy_projection_log_only = false;
  double degeneracy_trigger_ratio = 0.02;
  double degeneracy_min_eigenvalue_threshold = 1e-3;
  int degeneracy_poor_match_min_correspondences = 80;
  double degeneracy_poor_match_rmse_threshold = 0.25;
  double degeneracy_relative_scale = 0.05;
  double degeneracy_abs_floor = 1e-6;
  double degeneracy_min_weight = 0.0;
};

struct IekfUpdateResult
{
  struct IterationLog
  {
    int iekf_iter = 0;
    std::size_t valid_correspondences = 0;
    double residual_rmse = 0.0;
    double lambda_min = 0.0;
    double lambda_max = 0.0;
    double lambda_ratio = 0.0;
    double condition_number = 0.0;
    bool bad_ratio = false;
    bool weak_abs = false;
    bool bad_quality = false;
    bool is_degenerate_raw = false;
    bool degeneracy_projection_triggered = false;
    bool too_few_correspondences = false;
    double threshold_ref = 0.0;
    std::array<double, 6> weights {{1.0, 1.0, 1.0, 1.0, 1.0, 1.0}};
    double update_dx_rot_norm = 0.0;
    double update_dx_pos_norm = 0.0;
  };

  bool updated = false;
  std::size_t correspondences = 0;
  double rmse = 0.0;
  int iterations = 0;
  double lambda_min = 0.0;
  double lambda_max = 0.0;
  double lambda_ratio = 0.0;
  double condition_number = 0.0;
  bool bad_ratio = false;
  bool weak_abs = false;
  bool bad_quality = false;
  bool is_degenerate_raw = false;
  double threshold_ref = 0.0;
  std::array<double, 6> weights {{1.0, 1.0, 1.0, 1.0, 1.0, 1.0}};
  double update_dx_rot_norm = 0.0;
  double update_dx_pos_norm = 0.0;
  std::uint64_t radius_search_ns = 0;
  std::uint64_t plane_fit_ns = 0;
  std::uint64_t accumulate_ns = 0;
  std::uint64_t solve_ns = 0;
  std::vector<IterationLog> iter_logs;
};

class IekfUpdater
{
public:
  explicit IekfUpdater(IekfUpdaterConfig config = {}) : config_(config) {}

  void setConfig(const IekfUpdaterConfig & config) { config_ = config; }

  bool updatePoseWithPointToMap(
    const LidarScanXYZ & scan_i_end,
    const TreePointMap & map_w,
    IekfState18 & state,
    IekfUpdateResult * result = nullptr) const;

private:
  IekfUpdaterConfig config_;
};

}  // namespace iekf_lio
