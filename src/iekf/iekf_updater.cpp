#include "iekf/iekf_updater.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <vector>

#include <Eigen/Dense>

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

using SteadyClock = std::chrono::steady_clock;

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

bool fitPlaneFromVoxelNeighbors(
  const std::vector<VoxelMap::NearbyVoxel> & voxels,
  double max_eigen_ratio,
  Eigen::Vector3d * normal,
  Eigen::Vector3d * centroid)
{
  if (normal == nullptr || centroid == nullptr || voxels.size() < 3) {
    return false;
  }

  Eigen::Vector3d total_sum = Eigen::Vector3d::Zero();
  Eigen::Matrix3d total_sum_outer = Eigen::Matrix3d::Zero();
  std::size_t total_count = 0;
  for (const auto & voxel : voxels) {
    total_sum += voxel.sum_w;
    total_sum_outer += voxel.sum_outer_w;
    total_count += voxel.count;
  }
  if (total_count < 3) {
    return false;
  }

  const Eigen::Vector3d c = total_sum / static_cast<double>(total_count);
  Eigen::Matrix3d cov =
    (total_sum_outer - static_cast<double>(total_count) * c * c.transpose());
  if (total_count > 1) {
    cov /= static_cast<double>(total_count - 1);
  }
  cov = 0.5 * (cov + cov.transpose());

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
  const LidarScanXYZ & scan_i_end,
  const VoxelMap & voxel_map_w,
  IekfState18 & state,
  IekfUpdateResult * result) const
{
  if (result != nullptr) {
    *result = IekfUpdateResult {};
  }
  if (!state.is_initialized || scan_i_end.points == nullptr || scan_i_end.points->empty()) {
    return false;
  }

  const int max_iters = std::max(1, config_.max_iterations);// IEKF迭代的最大次数
  const double search_radius = std::max(1e-3, config_.max_correspondence_distance);
  const int k_neighbors = std::max(3, config_.plane_k_neighbors);//平面拟合时的邻居数量
  const double sigma2 = config_.sigma_point_to_plane * config_.sigma_point_to_plane;
  const double inv_sigma2 = 1.0 / sigma2;
  const double max_abs_residual = std::max(1e-3, config_.max_abs_point_to_plane_residual);
  const std::size_t max_points = static_cast<std::size_t>(std::max(10, config_.max_update_points));
  const std::size_t total_points = scan_i_end.points->size();
  const std::size_t stride = std::max<std::size_t>(1, (total_points + max_points - 1) / max_points);
  constexpr int kGeomDim = 6;
  bool updated_any = false;
  const Eigen::Matrix<double, 18, 18> I18 = Eigen::Matrix<double, 18, 18>::Identity();

  const Eigen::LDLT<Eigen::Matrix<double, 18, 18>> prior_ldlt(state.p_cov);
  if (prior_ldlt.info() != Eigen::Success) {
    return false;
  }
  const Eigen::Matrix<double, 18, 18> P_inv = prior_ldlt.solve(I18);

  for (int iter = 0; iter < max_iters; ++iter) {
    const std::size_t sampled_points = (total_points + stride - 1) / stride;
    Eigen::Matrix<double, 18, 18> info_HtRinvH = Eigen::Matrix<double, 18, 18>::Zero();
    Eigen::Matrix<double, 18, 1> info_HtRinvNu = Eigen::Matrix<double, 18, 1>::Zero();
    std::size_t corr = 0;
    double err2_sum = 0.0;
    std::uint64_t radius_search_ns_iter = 0;
    std::uint64_t plane_fit_ns_iter = 0;
    std::uint64_t accumulate_ns_iter = 0;

    // Information-form accumulation (parallel):
    // A = P^{-1} + sum(H_i^T R^{-1} H_i), b = sum(H_i^T R^{-1} * (-r_i)).
#pragma omp parallel
    {
      Eigen::Matrix<double, 18, 18> info_local = Eigen::Matrix<double, 18, 18>::Zero();
      Eigen::Matrix<double, 18, 1> b_local = Eigen::Matrix<double, 18, 1>::Zero();
      std::size_t corr_local = 0;
      double err2_local = 0.0;
      std::uint64_t radius_search_ns_local = 0;
      std::uint64_t plane_fit_ns_local = 0;
      std::uint64_t accumulate_ns_local = 0;
      std::vector<VoxelMap::NearbyVoxel> neighbors;
      neighbors.reserve(static_cast<std::size_t>(k_neighbors));

#pragma omp for schedule(static)
      for (std::int64_t s = 0; s < static_cast<std::int64_t>(sampled_points); ++s) {
        const std::size_t sampled_idx = static_cast<std::size_t>(s);
        const std::size_t idx = sampled_idx * stride;
        if (idx >= scan_i_end.points->size()) {
          continue;
        }

        const auto & pt = scan_i_end.points->at(idx);
        if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
          continue;
        }
        const Eigen::Vector3d p_i(pt.x, pt.y, pt.z);
        const Eigen::Vector3d p_w = state.x.r_wb * p_i + state.x.p_wb;

        const auto radius_search_t0 = SteadyClock::now();
        const std::size_t found = voxel_map_w.radiusSearchVoxels(
          p_w, search_radius, static_cast<std::size_t>(k_neighbors), &neighbors);
        radius_search_ns_local += static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
            SteadyClock::now() - radius_search_t0).count());
        if (found < static_cast<std::size_t>(k_neighbors)) {
          continue;
        }

        Eigen::Vector3d n_w = Eigen::Vector3d::Zero();
        Eigen::Vector3d c_w = Eigen::Vector3d::Zero();
        const auto plane_fit_t0 = SteadyClock::now();
        if (!fitPlaneFromVoxelNeighbors(neighbors, config_.plane_max_eigen_ratio, &n_w, &c_w)) {
          plane_fit_ns_local += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
              SteadyClock::now() - plane_fit_t0).count());
          continue;
        }
        plane_fit_ns_local += static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
            SteadyClock::now() - plane_fit_t0).count());

        const double r_i = n_w.dot(p_w - c_w);
        if (std::abs(r_i) > max_abs_residual) {
          continue;
        }

        Eigen::Matrix<double, 1, 18> H_i = Eigen::Matrix<double, 1, 18>::Zero();
        H_i.block<1, 3>(0, 0) = n_w.transpose();
        H_i.block<1, 3>(0, 6) = -n_w.transpose() * state.x.r_wb * skew(p_i);

        const auto accumulate_t0 = SteadyClock::now();
        info_local.noalias() += inv_sigma2 * (H_i.transpose() * H_i);
        b_local.noalias() += inv_sigma2 * (H_i.transpose() * (-r_i));
        accumulate_ns_local += static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
            SteadyClock::now() - accumulate_t0).count());
        ++corr_local;
        err2_local += r_i * r_i;
      }

#pragma omp critical
      {
        info_HtRinvH += info_local;
        info_HtRinvNu += b_local;
        corr += corr_local;
        err2_sum += err2_local;
        radius_search_ns_iter += radius_search_ns_local;
        plane_fit_ns_iter += plane_fit_ns_local;
        accumulate_ns_iter += accumulate_ns_local;
      }
    }

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

    if (config_.degeneracy_projection_enable) {
      Eigen::Matrix<double, kGeomDim, kGeomDim> lambda_g =
        Eigen::Matrix<double, kGeomDim, kGeomDim>::Zero();
      lambda_g.block<3, 3>(0, 0) = info_HtRinvH.block<3, 3>(0, 0);
      lambda_g.block<3, 3>(0, 3) = info_HtRinvH.block<3, 3>(0, 6);
      lambda_g.block<3, 3>(3, 0) = info_HtRinvH.block<3, 3>(6, 0);
      lambda_g.block<3, 3>(3, 3) = info_HtRinvH.block<3, 3>(6, 6);
      lambda_g = 0.5 * (lambda_g + lambda_g.transpose());

      Eigen::Matrix<double, kGeomDim, 1> eta_g = Eigen::Matrix<double, kGeomDim, 1>::Zero();
      eta_g.segment<3>(0) = info_HtRinvNu.segment<3>(0);
      eta_g.segment<3>(3) = info_HtRinvNu.segment<3>(6);

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, kGeomDim, kGeomDim>> solver(lambda_g);
      if (solver.info() == Eigen::Success) {
        const Eigen::Matrix<double, kGeomDim, 1> eigvals = solver.eigenvalues();
        const Eigen::Matrix<double, kGeomDim, kGeomDim> eigvecs = solver.eigenvectors();
        const double lambda_max =
          std::max(config_.degeneracy_abs_floor, eigvals.maxCoeff());
        const double lambda_min = std::max(0.0, eigvals.minCoeff());
        const double trigger_ratio = lambda_min / lambda_max;
        if (trigger_ratio < config_.degeneracy_trigger_ratio) {
          const double lambda_ref = std::max(
            config_.degeneracy_abs_floor,
            config_.degeneracy_relative_scale * lambda_max);

          Eigen::Matrix<double, kGeomDim, 1> weights =
            Eigen::Matrix<double, kGeomDim, 1>::Ones();
          for (int k = 0; k < kGeomDim; ++k) {
            const double lambda_k = std::max(0.0, eigvals(k));
            const double w = lambda_k / (lambda_k + lambda_ref);
            weights(k) = std::clamp(w, config_.degeneracy_min_weight, 1.0);
          }

          const Eigen::Matrix<double, kGeomDim, kGeomDim> W = weights.asDiagonal();
          // Only when the geometric information matrix is clearly degenerate do we
          // suppress weakly observed eigendirections before solving the 18D system.
          const Eigen::Matrix<double, kGeomDim, kGeomDim> D = eigvecs * W * eigvecs.transpose();
          const Eigen::Matrix<double, kGeomDim, kGeomDim> lambda_g_deg =
            D * lambda_g * D.transpose();
          const Eigen::Matrix<double, kGeomDim, 1> eta_g_deg = D * eta_g;

          info_HtRinvH.block<3, 3>(0, 0) = lambda_g_deg.block<3, 3>(0, 0);
          info_HtRinvH.block<3, 3>(0, 6) = lambda_g_deg.block<3, 3>(0, 3);
          info_HtRinvH.block<3, 3>(6, 0) = lambda_g_deg.block<3, 3>(3, 0);
          info_HtRinvH.block<3, 3>(6, 6) = lambda_g_deg.block<3, 3>(3, 3);
          info_HtRinvNu.segment<3>(0) = eta_g_deg.segment<3>(0);
          info_HtRinvNu.segment<3>(6) = eta_g_deg.segment<3>(3);
        }
      }
    }

    const auto solve_t0 = SteadyClock::now();
    const Eigen::Matrix<double, 18, 18> A = P_inv + info_HtRinvH;
    const Eigen::LDLT<Eigen::Matrix<double, 18, 18>> ldlt_A(A);
    if (ldlt_A.info() != Eigen::Success) {
      break;
    }
    const Eigen::Matrix<double, 18, 1> delta = ldlt_A.solve(info_HtRinvNu);
    const double delta_pos_norm = delta.segment<3>(0).norm();
    const double delta_rot_norm = delta.segment<3>(6).norm();
    const bool converged =
      delta_pos_norm <= config_.convergence_delta_pos_m &&
      delta_rot_norm <= config_.convergence_delta_rot_rad;

    state.x.p_wb += delta.segment<3>(0);
    state.x.v_wb += delta.segment<3>(3);
    state.x.r_wb = state.x.r_wb * expSO3(delta.segment<3>(6));//不是
    state.x.b_g += delta.segment<3>(9);
    state.x.b_a += delta.segment<3>(12);
    state.x.g_w += delta.segment<3>(15);
    updated_any = true;

    if (iter == max_iters - 1 || converged) {
      state.p_cov = ldlt_A.solve(I18);
    }

    if (result != nullptr) {
      result->updated = true;
    }

    if (converged) {
      if (result != nullptr) {
        result->radius_search_ns += radius_search_ns_iter;
        result->plane_fit_ns += plane_fit_ns_iter;
        result->accumulate_ns += accumulate_ns_iter;
        result->solve_ns += static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
            SteadyClock::now() - solve_t0).count());
      }
      break;
    }

    if (result != nullptr) {
      result->radius_search_ns += radius_search_ns_iter;
      result->plane_fit_ns += plane_fit_ns_iter;
      result->accumulate_ns += accumulate_ns_iter;
      result->solve_ns += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
          SteadyClock::now() - solve_t0).count());
    }
  }

  if (result != nullptr) {
    result->updated = updated_any;
  }
  return updated_any;
}

}  // namespace iekf_lio
