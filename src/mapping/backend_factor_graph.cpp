#include "mapping/backend_factor_graph.hpp"

#include <limits>

#if defined(IEKF_LIO_USE_GTSAM) && __has_include(<gtsam/nonlinear/LevenbergMarquardtOptimizer.h>)
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#define IEKF_LIO_GTSAM_ACTIVE 1
#else
#define IEKF_LIO_GTSAM_ACTIVE 0
#endif

namespace iekf_lio
{
namespace
{

gtsam::Pose3 toGtsamPose(const BackendPoseNode & node)
{
#if IEKF_LIO_GTSAM_ACTIVE
  return gtsam::Pose3(
    gtsam::Rot3(node.r_wb),
    gtsam::Point3(node.p_wb.x(), node.p_wb.y(), node.p_wb.z()));
#else
  (void)node;
  return gtsam::Pose3();
#endif
}

BackendPoseNode fromGtsamPose(
  const BackendPoseNode & reference,
  const gtsam::Pose3 & pose)
{
  BackendPoseNode out = reference;
#if IEKF_LIO_GTSAM_ACTIVE
  out.p_wb = pose.translation();
  out.r_wb = pose.rotation().matrix();
#else
  (void)pose;
#endif
  return out;
}

// The project stores 6x6 information matrices in [t, r] order, while GTSAM's
// Pose3 tangent vector uses [r, t]. Reorder explicitly to keep noise semantics
// consistent when building PriorFactor<Pose3> and BetweenFactor<Pose3>.
Eigen::Matrix<double, 6, 6> reorderInformationToGtsam(
  const Eigen::Matrix<double, 6, 6> & info_tr)
{
  Eigen::Matrix<double, 6, 6> perm = Eigen::Matrix<double, 6, 6>::Zero();
  perm.block<3, 3>(0, 3).setIdentity();
  perm.block<3, 3>(3, 0).setIdentity();
  return perm * info_tr * perm.transpose();
}

}  // namespace

void BackendFactorGraph::reset()
{
  std::lock_guard<std::mutex> lk(mutex_);
  nodes_.clear();
  optimized_nodes_.clear();
  prior_factors_.clear();
  between_factors_.clear();
  loop_factors_.clear();
  latest_optimization_result_ = BackendOptimizationResult {};
}

void BackendFactorGraph::addPoseNode(const BackendPoseNode & node)
{
  std::lock_guard<std::mutex> lk(mutex_);
  nodes_.push_back(node);
}

void BackendFactorGraph::addPriorFactor(const BackendPriorFactor & factor)
{
  std::lock_guard<std::mutex> lk(mutex_);
  prior_factors_.push_back(factor);
}

void BackendFactorGraph::addBetweenFactor(const BackendBetweenFactor & factor)
{
  std::lock_guard<std::mutex> lk(mutex_);
  between_factors_.push_back(factor);
}

void BackendFactorGraph::addLoopFactor(const BackendLoopFactor & factor)
{
  std::lock_guard<std::mutex> lk(mutex_);
  loop_factors_.push_back(factor);
}

std::size_t BackendFactorGraph::nodeCount() const
{
  std::lock_guard<std::mutex> lk(mutex_);
  return nodes_.size();
}

std::size_t BackendFactorGraph::priorFactorCount() const
{
  std::lock_guard<std::mutex> lk(mutex_);
  return prior_factors_.size();
}

std::size_t BackendFactorGraph::betweenFactorCount() const
{
  std::lock_guard<std::mutex> lk(mutex_);
  return between_factors_.size();
}

std::size_t BackendFactorGraph::totalFactorCount() const
{
  std::lock_guard<std::mutex> lk(mutex_);
  return prior_factors_.size() + between_factors_.size() + loop_factors_.size();
}

bool BackendFactorGraph::isGtsamActive() const
{
#if IEKF_LIO_GTSAM_ACTIVE
  return true;
#else
  return false;
#endif
}

std::size_t BackendFactorGraph::loopFactorCount() const
{
  std::lock_guard<std::mutex> lk(mutex_);
  return loop_factors_.size();
}

bool BackendFactorGraph::getLatestNode(BackendPoseNode * out) const
{
  if (out == nullptr) {
    return false;
  }
  std::lock_guard<std::mutex> lk(mutex_);
  if (nodes_.empty()) {
    return false;
  }
  *out = nodes_.back();
  return true;
}

bool BackendFactorGraph::getLatestOptimizedNode(BackendPoseNode * out) const
{
  if (out == nullptr) {
    return false;
  }
  std::lock_guard<std::mutex> lk(mutex_);
  if (optimized_nodes_.empty()) {
    return false;
  }
  *out = optimized_nodes_.back();
  return true;
}

std::vector<BackendPoseNode> BackendFactorGraph::getPoseNodesSnapshot() const
{
  std::lock_guard<std::mutex> lk(mutex_);
  return nodes_;
}

std::vector<BackendPoseNode> BackendFactorGraph::getOptimizedPoseNodesSnapshot() const
{
  std::lock_guard<std::mutex> lk(mutex_);
  return optimized_nodes_;
}

std::vector<BackendPriorFactor> BackendFactorGraph::getPriorFactorsSnapshot() const
{
  std::lock_guard<std::mutex> lk(mutex_);
  return prior_factors_;
}

std::vector<BackendBetweenFactor> BackendFactorGraph::getBetweenFactorsSnapshot() const
{
  std::lock_guard<std::mutex> lk(mutex_);
  return between_factors_;
}

std::vector<BackendLoopFactor> BackendFactorGraph::getLoopFactorsSnapshot() const
{
  std::lock_guard<std::mutex> lk(mutex_);
  return loop_factors_;
}

BackendOptimizationResult BackendFactorGraph::optimizeOnce()
{
  std::lock_guard<std::mutex> lk(mutex_);
  latest_optimization_result_ = BackendOptimizationResult {};
  latest_optimization_result_.node_count = nodes_.size();
  latest_optimization_result_.prior_factor_count = prior_factors_.size();
  latest_optimization_result_.between_factor_count = between_factors_.size() + loop_factors_.size();

  if (nodes_.empty()) {
    return latest_optimization_result_;
  }

#if IEKF_LIO_GTSAM_ACTIVE
  try {
    gtsam::NonlinearFactorGraph graph;
    gtsam::Values initial;

    const std::vector<BackendPoseNode> & seed_nodes =
      (optimized_nodes_.size() == nodes_.size()) ? optimized_nodes_ : nodes_;
    for (const auto & node : seed_nodes) {
      initial.insert(gtsam::Symbol('x', node.keyframe_id), toGtsamPose(node));
    }

    for (const auto & factor : prior_factors_) {
      const BackendPoseNode node_ref {
        factor.keyframe_id, 0.0, factor.p_wb, factor.r_wb};
      graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
        gtsam::Symbol('x', factor.keyframe_id),
        toGtsamPose(node_ref),
        gtsam::noiseModel::Gaussian::Information(
          reorderInformationToGtsam(factor.information)));
    }

    for (const auto & factor : between_factors_) {
      BackendPoseNode relative_pose;
      relative_pose.keyframe_id = factor.to_id;
      relative_pose.p_wb = factor.t_ij;
      relative_pose.r_wb = factor.r_ij;
      graph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
        gtsam::Symbol('x', factor.from_id),
        gtsam::Symbol('x', factor.to_id),
        toGtsamPose(relative_pose),
        gtsam::noiseModel::Gaussian::Information(
          reorderInformationToGtsam(factor.information)));
    }

    for (const auto & factor : loop_factors_) {
      BackendPoseNode relative_pose;
      relative_pose.keyframe_id = factor.to_id;
      relative_pose.p_wb = factor.t_ij;
      relative_pose.r_wb = factor.r_ij;
      graph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
        gtsam::Symbol('x', factor.from_id),
        gtsam::Symbol('x', factor.to_id),
        toGtsamPose(relative_pose),
        gtsam::noiseModel::Gaussian::Information(
          reorderInformationToGtsam(factor.information)));
    }

    gtsam::LevenbergMarquardtParams params;
    params.maxIterations = 20;
    params.relativeErrorTol = 1e-5;
    params.absoluteErrorTol = 1e-5;
    params.lambdaInitial = 1e-3;
    gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial, params);
    const gtsam::Values result_values = optimizer.optimize();

    optimized_nodes_.clear();
    optimized_nodes_.reserve(nodes_.size());
    for (const auto & node : nodes_) {
      const auto key = gtsam::Symbol('x', node.keyframe_id);
      const gtsam::Pose3 pose = result_values.at<gtsam::Pose3>(key);
      optimized_nodes_.push_back(fromGtsamPose(node, pose));
    }

    latest_optimization_result_.success = true;
    latest_optimization_result_.optimized_node_count = optimized_nodes_.size();
    latest_optimization_result_.iteration_count = params.maxIterations;
    latest_optimization_result_.final_cost = graph.error(result_values);
  } catch (...) {
    optimized_nodes_ = nodes_;
    latest_optimization_result_.success = false;
    latest_optimization_result_.optimized_node_count = optimized_nodes_.size();
    latest_optimization_result_.iteration_count = 0;
    latest_optimization_result_.final_cost = std::numeric_limits<double>::infinity();
  }
#else
  optimized_nodes_ = nodes_;
  latest_optimization_result_.success = false;
  latest_optimization_result_.optimized_node_count = optimized_nodes_.size();
  latest_optimization_result_.iteration_count = 0;
  latest_optimization_result_.final_cost = std::numeric_limits<double>::infinity();
#endif

  return latest_optimization_result_;
}

BackendOptimizationResult BackendFactorGraph::latestOptimizationResult() const
{
  std::lock_guard<std::mutex> lk(mutex_);
  return latest_optimization_result_;
}

}  // namespace iekf_lio
