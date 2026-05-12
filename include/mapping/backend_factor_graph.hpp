#pragma once

#include <cstddef>
#include <mutex>
#include <vector>

#include <Eigen/Dense>

namespace iekf_lio
{

struct BackendPoseNode
{
  std::size_t keyframe_id = 0;
  double time_s = 0.0;
  Eigen::Vector3d p_wb = Eigen::Vector3d::Zero();
  Eigen::Matrix3d r_wb = Eigen::Matrix3d::Identity();
};

struct BackendPriorFactor
{
  std::size_t keyframe_id = 0;
  Eigen::Vector3d p_wb = Eigen::Vector3d::Zero();
  Eigen::Matrix3d r_wb = Eigen::Matrix3d::Identity();
  Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Identity();
};

struct BackendBetweenFactor
{
  std::size_t from_id = 0;
  std::size_t to_id = 0;
  Eigen::Vector3d t_ij = Eigen::Vector3d::Zero();
  Eigen::Matrix3d r_ij = Eigen::Matrix3d::Identity();
  Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Identity();
};

struct BackendLoopFactor
{
  std::size_t from_id = 0;
  std::size_t to_id = 0;
  Eigen::Vector3d t_ij = Eigen::Vector3d::Zero();
  Eigen::Matrix3d r_ij = Eigen::Matrix3d::Identity();
  Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Identity();
  double fitness = 0.0;
};

struct BackendOptimizationResult
{
  bool success = false;
  std::size_t node_count = 0;
  std::size_t prior_factor_count = 0;
  std::size_t between_factor_count = 0;
  std::size_t optimized_node_count = 0;
  std::size_t iteration_count = 0;
  double final_cost = 0.0;
};

class BackendFactorGraph
{
public:
  void reset();

  void addPoseNode(const BackendPoseNode & node);
  void addPriorFactor(const BackendPriorFactor & factor);
  void addBetweenFactor(const BackendBetweenFactor & factor);
  void addLoopFactor(const BackendLoopFactor & factor);

  std::size_t nodeCount() const;
  std::size_t priorFactorCount() const;
  std::size_t betweenFactorCount() const;
  std::size_t loopFactorCount() const;
  std::size_t totalFactorCount() const;
  bool isGtsamActive() const;

  bool getLatestNode(BackendPoseNode * out) const;
  bool getLatestOptimizedNode(BackendPoseNode * out) const;

  // Return thread-safe snapshots of the current factor-graph contents so
  // later backend modules can inspect or visualize graph topology without
  // touching the internal containers directly.
  std::vector<BackendPoseNode> getPoseNodesSnapshot() const;
  std::vector<BackendPoseNode> getOptimizedPoseNodesSnapshot() const;
  std::vector<BackendPriorFactor> getPriorFactorsSnapshot() const;
  std::vector<BackendBetweenFactor> getBetweenFactorsSnapshot() const;
  std::vector<BackendLoopFactor> getLoopFactorsSnapshot() const;

  // Backend optimization entry skeleton. The current implementation keeps the
  // graph-building pipeline intact and simply mirrors raw nodes into the
  // optimized-node buffer as a placeholder for a future nonlinear optimizer.
  BackendOptimizationResult optimizeOnce();
  BackendOptimizationResult latestOptimizationResult() const;

private:
  mutable std::mutex mutex_;
  std::vector<BackendPoseNode> nodes_;
  std::vector<BackendPoseNode> optimized_nodes_;
  std::vector<BackendPriorFactor> prior_factors_;
  std::vector<BackendBetweenFactor> between_factors_;
  std::vector<BackendLoopFactor> loop_factors_;
  BackendOptimizationResult latest_optimization_result_;
};

}  // namespace iekf_lio
