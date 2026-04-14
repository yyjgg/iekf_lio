#include "iekf/iekf_updater.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <Eigen/Dense>
#include <pcl/kdtree/kdtree_flann.h>

#if defined(IEKF_LIO_USE_SOPHUS) && __has_include(<sophus/so3.hpp>)
#include <sophus/so3.hpp>
#define IEKF_LIO_SOPHUS_ACTIVE 1
#else
#define IEKF_LIO_SOPHUS_ACTIVE 0
#endif

namespace iekf_lio
{
namespace
{

Eigen::Matrix3d skew(const Eigen::Vector3d & w)
{
  Eigen::Matrix3d m = Eigen::Matrix3d::Zero();
  m(0, 1) = -w.z();
  m(0, 2) = w.y();
  m(1, 0) = w.z();
  m(1, 2) = -w.x();
  m(2, 0) = -w.y();
  m(2, 1) = w.x();
  return m;
}

Eigen::Matrix3d expSO3(const Eigen::Vector3d & theta)
{
#if IEKF_LIO_SOPHUS_ACTIVE
  return Sophus::SO3d::exp(theta).matrix();
#else
  const double angle = theta.norm();
  const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d K = skew(theta);
  const Eigen::Matrix3d K2 = K * K;
  if (angle < 1e-10) {
    return I + K;
  }
  const double a = std::sin(angle) / angle;
  const double b = (1.0 - std::cos(angle)) / (angle * angle);
  return I + a * K + b * K2;
#endif
}

bool fitPlaneFromNeighbors(
  const std::vector<Eigen::Vector3d> & pts,
  double max_eigen_ratio,
  Eigen::Vector3d * normal,
  Eigen::Vector3d * centroid)
{
  if (normal == nullptr || centroid == nullptr || pts.size() < 3) {
    return false;
  }

  Eigen::Vector3d c = Eigen::Vector3d::Zero();
  for (const auto & p : pts) {
    c += p;
  }
  c /= static_cast<double>(pts.size());

  Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
  for (const auto & p : pts) {
    const Eigen::Vector3d d = p - c;
    cov += d * d.transpose();
  }

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
  if (solver.info() != Eigen::Success) {
    return false;
  }
  const Eigen::Vector3d evals = solver.eigenvalues();
  if (evals(1) <= 1e-12) {
    return false;
  }
  if ((evals(0) / evals(1)) > max_eigen_ratio) {
    return false;
  }
  *normal = solver.eigenvectors().col(0).normalized();
  *centroid = c;
  return true;
}

}  // namespace

bool IekfUpdater::updatePoseWithPointToMap(
  const LidarScan & scan_i_end,
  const pcl::PointCloud<pcl::PointXYZ>::ConstPtr & map_cloud_w,
  IekfState18 & state,
  IekfUpdateResult * result) const
{
  if (result != nullptr) {
    *result = IekfUpdateResult {};
  }
  if (!state.is_initialized || map_cloud_w == nullptr || scan_i_end.points.empty()) {
    return false;
  }

  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
  kdtree.setInputCloud(map_cloud_w);

  const int max_iters = std::max(1, config_.max_iterations);// IEKF迭代的最大次数
  const double max_dist2 = config_.max_correspondence_distance ;//点的最近临近邻搜索的距离阈值，单位为米
  const int k_neighbors = std::max(3, config_.plane_k_neighbors);//平面拟合时的邻居数量
  const double sigma2 = config_.sigma_point_to_plane * config_.sigma_point_to_plane;
  const double max_abs_residual = std::max(1e-3, config_.max_abs_point_to_plane_residual);
  const std::size_t max_points = static_cast<std::size_t>(std::max(10, config_.max_update_points));
  const std::size_t total_points = scan_i_end.points.size();
  const std::size_t stride = std::max<std::size_t>(1, (total_points + max_points - 1) / max_points);
  bool updated_any = false;

  for (int iter = 0; iter < max_iters; ++iter) {
    std::vector<Eigen::Matrix<double, 1, 18>> H_rows;
    std::vector<double> r_vals;
    H_rows.reserve(std::min<std::size_t>(scan_i_end.points.size(), max_points));
    r_vals.reserve(std::min<std::size_t>(scan_i_end.points.size(), max_points));

    std::vector<int> nn_idx(static_cast<std::size_t>(k_neighbors));
    std::vector<float> nn_dist2(static_cast<std::size_t>(k_neighbors));
    double err2_sum = 0.0;

    for (std::size_t idx = 0; idx < scan_i_end.points.size(); idx += stride) {
      const auto & pt = scan_i_end.points[idx];
      if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
        continue;
      }
      const Eigen::Vector3d p_i(pt.x, pt.y, pt.z);
      const Eigen::Vector3d p_w = state.x.r_wb * p_i + state.x.p_wb;

      pcl::PointXYZ q;
      q.x = static_cast<float>(p_w.x());
      q.y = static_cast<float>(p_w.y());
      q.z = static_cast<float>(p_w.z());
      const int found = kdtree.nearestKSearch(q, k_neighbors, nn_idx, nn_dist2);
      if (found < k_neighbors || nn_dist2[static_cast<std::size_t>(k_neighbors - 1)] > max_dist2) {
        continue;
      }

      std::vector<Eigen::Vector3d> neighbors;
      neighbors.reserve(static_cast<std::size_t>(k_neighbors));
      for (int k = 0; k < k_neighbors; ++k) {
        const pcl::PointXYZ & mp = map_cloud_w->at(static_cast<std::size_t>(nn_idx[static_cast<std::size_t>(k)]));
        neighbors.emplace_back(mp.x, mp.y, mp.z);
      }

      Eigen::Vector3d n_w = Eigen::Vector3d::Zero();
      Eigen::Vector3d c_w = Eigen::Vector3d::Zero();
      if (!fitPlaneFromNeighbors(neighbors, config_.plane_max_eigen_ratio, &n_w, &c_w)) {
        continue;
      }

      const double r_i = n_w.dot(p_w - c_w);
      if (std::abs(r_i) > max_abs_residual) {
        continue;
      }
      Eigen::Matrix<double, 1, 18> H_i = Eigen::Matrix<double, 1, 18>::Zero();
      H_i.block<1, 3>(0, 0) = n_w.transpose();
      H_i.block<1, 3>(0, 6) = -n_w.transpose() * state.x.r_wb * skew(p_i);

      H_rows.push_back(H_i);
      r_vals.push_back(r_i);
      err2_sum += r_i * r_i;
      // if (H_rows.size() >= max_points) {
      //   break;
      // }
    }

    const std::size_t corr = H_rows.size();
    if (corr == 0) {
      if (result != nullptr) {
        result->correspondences = corr;
        result->rmse = 0.0;
        result->iterations = iter + 1;
      }
      break;
    }
    const double rmse = std::sqrt(err2_sum / static_cast<double>(corr));
    if (result != nullptr) {
      result->correspondences = corr;
      result->rmse = rmse;
      result->iterations = iter + 1;
    }

    Eigen::MatrixXd H(corr, 18);
    Eigen::VectorXd r(corr);
    for (std::size_t i = 0; i < corr; ++i) {
      H.row(static_cast<Eigen::Index>(i)) = H_rows[i];
      r(static_cast<Eigen::Index>(i)) = r_vals[i];
    }

    const Eigen::MatrixXd S = H * state.p_cov * H.transpose()
      + sigma2 * Eigen::MatrixXd::Identity(static_cast<Eigen::Index>(corr), static_cast<Eigen::Index>(corr));
    const Eigen::LDLT<Eigen::MatrixXd> ldlt(S);
    if (ldlt.info() != Eigen::Success) {
      break;
    }
    // Innovation is z - h(x). Here z = 0 for point-to-plane, h(x) = r, so innovation = -r.
    const Eigen::VectorXd y = ldlt.solve(-r);// y = S^-1 * (-r) 最初优化方向错误
    const Eigen::VectorXd delta = state.p_cov * H.transpose() * y;

    state.x.p_wb += delta.segment<3>(0);
    state.x.v_wb += delta.segment<3>(3);
    state.x.r_wb = state.x.r_wb * expSO3(delta.segment<3>(6));//不是
    state.x.b_g += delta.segment<3>(9);
    state.x.b_a += delta.segment<3>(12);
    state.x.g_w += delta.segment<3>(15);
    updated_any = true;

    if (iter == max_iters - 1) {
      const Eigen::MatrixXd HP = H * state.p_cov;
      const Eigen::MatrixXd X = ldlt.solve(HP);
      const Eigen::MatrixXd K = X.transpose();  // 18 x N
      const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(18, 18);
      const Eigen::MatrixXd KH = K * H;
      const Eigen::MatrixXd KR = K * (sigma2 * Eigen::MatrixXd::Identity(
        static_cast<Eigen::Index>(corr), static_cast<Eigen::Index>(corr)));
      state.p_cov = (I - KH) * state.p_cov * (I - KH).transpose() + KR * K.transpose();
    }

    if (result != nullptr) {
      result->updated = true;
    }
  }

  if (result != nullptr) {
    result->updated = updated_any;
  }
  return updated_any;
}

}  // namespace iekf_lio
