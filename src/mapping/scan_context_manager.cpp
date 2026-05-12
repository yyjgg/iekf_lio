#include "mapping/scan_context_manager.hpp"

#include <algorithm>
#include <cmath>
#include <flann/flann.hpp>
#include <limits>
#include <numeric>
#include <vector>

namespace iekf_lio
{
namespace
{

constexpr double kPi = 3.14159265358979323846;
constexpr double kTwoPi = 2.0 * kPi;

double wrapAnglePositive(double angle_rad)
{
  double out = std::fmod(angle_rad, kTwoPi);
  if (out < 0.0) {
    out += kTwoPi;
  }
  return out;
}

}  // namespace

void ScanContextManager::setConfig(const ScanContextManagerConfig & config)
{
  std::lock_guard<std::mutex> lk(mutex_);
  config_ = config;
  config_.num_rings = std::max(1, config_.num_rings);
  config_.num_sectors = std::max(1, config_.num_sectors);
  config_.max_radius_m = std::max(1.0, config_.max_radius_m);
  config_.max_entries = std::max<std::size_t>(1, config_.max_entries);
  config_.num_candidates = std::max<std::size_t>(1, config_.num_candidates);
}

ScanContextEntry ScanContextManager::makeDescriptor(
  std::size_t keyframe_id,
  double time_s,
  const pcl::PointCloud<pcl::PointXYZ>::ConstPtr & cloud) const
{
  ScanContextManagerConfig config;
  {
    std::lock_guard<std::mutex> lk(mutex_);
    config = config_;
  }

  ScanContextEntry entry;
  entry.keyframe_id = keyframe_id;
  entry.time_s = time_s;
  entry.descriptor = Eigen::MatrixXd::Zero(config.num_rings, config.num_sectors);
  entry.ring_key = Eigen::VectorXd::Zero(config.num_rings);

  if (cloud == nullptr || cloud->empty()) {
    return entry;
  }

  for (const auto & pt : cloud->points) {
    if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
      continue;
    }

    const double radius = std::hypot(static_cast<double>(pt.x), static_cast<double>(pt.y));
    if (radius < 1e-6 || radius > config.max_radius_m) {
      continue;
    }

    const double theta = wrapAnglePositive(std::atan2(static_cast<double>(pt.y), static_cast<double>(pt.x)));
    const int ring_idx = std::min(
      config.num_rings - 1,
      static_cast<int>(std::floor(radius / config.max_radius_m * static_cast<double>(config.num_rings))));
    const int sector_idx = std::min(
      config.num_sectors - 1,
      static_cast<int>(std::floor(theta / kTwoPi * static_cast<double>(config.num_sectors))));

    entry.descriptor(ring_idx, sector_idx) =
      std::max(entry.descriptor(ring_idx, sector_idx), static_cast<double>(pt.z));
  }

  for (int ring_idx = 0; ring_idx < config.num_rings; ++ring_idx) {
    entry.ring_key(ring_idx) = entry.descriptor.row(ring_idx).mean();
  }
  return entry;
}

bool ScanContextManager::detectLoop(
  const ScanContextEntry & query,
  LoopCandidate * candidate) const
{
  if (candidate == nullptr) {
    return false;
  }
  *candidate = LoopCandidate {};

  std::vector<ScanContextEntry> entries_snapshot;
  ScanContextManagerConfig config;
  {
    std::lock_guard<std::mutex> lk(mutex_);
    config = config_;
    entries_snapshot = entries_;
  }

  if (!config.enable) {
    return false;
  }
  if (entries_snapshot.size() < config.min_keyframes_before_loop) {
    return false;
  }
  if (query.descriptor.size() == 0 || query.ring_key.size() == 0) {
    return false;
  }

  const std::size_t eligible_count = entries_snapshot.size() >
    config.exclude_recent_keyframes
    ? (entries_snapshot.size() - config.exclude_recent_keyframes)
    : 0;
  if (eligible_count == 0) {
    return false;
  }

  struct RankedCandidate
  {
    std::size_t idx = 0;
    double ring_dist = std::numeric_limits<double>::infinity();
  };

  const std::size_t ring_dim = static_cast<std::size_t>(query.ring_key.size());
  if (ring_dim == 0) {
    return false;
  }

  std::vector<std::size_t> eligible_indices;
  eligible_indices.reserve(eligible_count);
  for (std::size_t i = 0; i < eligible_count; ++i) {
    if (static_cast<std::size_t>(entries_snapshot[i].ring_key.size()) != ring_dim) {
      continue;
    }
    eligible_indices.push_back(i);
  }
  if (eligible_indices.empty()) {
    return false;
  }

  std::vector<double> dataset_storage(eligible_indices.size() * ring_dim, 0.0);
  for (std::size_t row = 0; row < eligible_indices.size(); ++row) {
    const Eigen::VectorXd & ring_key = entries_snapshot[eligible_indices[row]].ring_key;
    for (std::size_t col = 0; col < ring_dim; ++col) {
      dataset_storage[row * ring_dim + col] = ring_key(static_cast<Eigen::Index>(col));
    }
  }
  flann::Matrix<double> dataset(
    dataset_storage.data(),
    eligible_indices.size(),
    ring_dim);
  flann::Index<flann::L2<double>> index(dataset, flann::KDTreeIndexParams(4));
  index.buildIndex();

  std::vector<double> query_storage(ring_dim, 0.0);
  for (std::size_t col = 0; col < ring_dim; ++col) {
    query_storage[col] = query.ring_key(static_cast<Eigen::Index>(col));
  }
  flann::Matrix<double> query_mat(query_storage.data(), 1, ring_dim);

  const std::size_t take_count = std::min(config.num_candidates, eligible_indices.size());
  std::vector<int> candidate_ids(take_count, -1);
  std::vector<double> candidate_dists(take_count, std::numeric_limits<double>::infinity());
  flann::Matrix<int> candidate_ids_mat(candidate_ids.data(), 1, take_count);
  flann::Matrix<double> candidate_dists_mat(candidate_dists.data(), 1, take_count);
  index.knnSearch(
    query_mat,
    candidate_ids_mat,
    candidate_dists_mat,
    static_cast<int>(take_count),
    flann::SearchParams(32));

  std::vector<RankedCandidate> ranked;
  ranked.reserve(take_count);
  for (std::size_t i = 0; i < take_count; ++i) {
    if (candidate_ids[i] < 0) {
      continue;
    }
    RankedCandidate item;
    item.idx = eligible_indices[static_cast<std::size_t>(candidate_ids[i])];
    item.ring_dist = std::sqrt(std::max(0.0, candidate_dists[i])) / static_cast<double>(ring_dim);
    ranked.push_back(item);
  }
  if (ranked.empty()) {
    return false;
  }

  double best_score = std::numeric_limits<double>::infinity();
  std::size_t best_idx = 0;
  std::size_t best_shift = 0;
  for (const auto & ranked_item : ranked) {
    std::size_t shift = 0;
    const double score = descriptorDistance(
      query.descriptor,
      entries_snapshot[ranked_item.idx].descriptor,
      &shift);
    if (score < best_score) {
      best_score = score;
      best_idx = ranked_item.idx;
      best_shift = shift;
    }
  }

  if (!std::isfinite(best_score) || best_score > config.distance_threshold) {
    return false;
  }

  candidate->query_id = query.keyframe_id;
  candidate->match_id = entries_snapshot[best_idx].keyframe_id;
  candidate->score = best_score;
  candidate->sector_shift = best_shift;
  candidate->yaw_init_rad =
    -static_cast<double>(best_shift) * kTwoPi / static_cast<double>(config.num_sectors);
  return true;
}

void ScanContextManager::addEntry(const ScanContextEntry & entry)
{
  if (entry.descriptor.size() == 0 || entry.ring_key.size() == 0) {
    return;
  }
  std::lock_guard<std::mutex> lk(mutex_);
  if (config_.max_entries > 0 && entries_.size() >= config_.max_entries) {
    const std::size_t overflow = entries_.size() - config_.max_entries + 1;
    entries_.erase(
      entries_.begin(),
      entries_.begin() + static_cast<std::ptrdiff_t>(overflow));
  }
  entries_.push_back(entry);
}

void ScanContextManager::reset()
{
  std::lock_guard<std::mutex> lk(mutex_);
  entries_.clear();
}

std::size_t ScanContextManager::entryCount() const
{
  std::lock_guard<std::mutex> lk(mutex_);
  return entries_.size();
}

double ScanContextManager::ringKeyDistance(
  const Eigen::VectorXd & lhs,
  const Eigen::VectorXd & rhs) const
{
  if (lhs.size() == 0 || rhs.size() == 0 || lhs.size() != rhs.size()) {
    return std::numeric_limits<double>::infinity();
  }
  return (lhs - rhs).norm() / static_cast<double>(lhs.size());
}

double ScanContextManager::descriptorDistance(
  const Eigen::MatrixXd & query,
  const Eigen::MatrixXd & reference,
  std::size_t * best_shift) const
{
  if (
    query.rows() == 0 || query.cols() == 0 ||
    query.rows() != reference.rows() || query.cols() != reference.cols())
  {
    if (best_shift != nullptr) {
      *best_shift = 0;
    }
    return std::numeric_limits<double>::infinity();
  }

  double best_score = std::numeric_limits<double>::infinity();
  std::size_t best_score_shift = 0;
  for (int shift = 0; shift < query.cols(); ++shift) {
    const Eigen::MatrixXd shifted = circularShiftColumns(query, static_cast<std::size_t>(shift));

    double cosine_sum = 0.0;
    std::size_t valid_cols = 0;
    for (int col = 0; col < shifted.cols(); ++col) {
      const Eigen::VectorXd lhs = shifted.col(col);
      const Eigen::VectorXd rhs = reference.col(col);
      const double lhs_norm = lhs.norm();
      const double rhs_norm = rhs.norm();
      if (lhs_norm < 1e-9 || rhs_norm < 1e-9) {
        continue;
      }
      cosine_sum += lhs.dot(rhs) / (lhs_norm * rhs_norm);
      ++valid_cols;
    }
    if (valid_cols == 0) {
      continue;
    }
    const double score = 1.0 - cosine_sum / static_cast<double>(valid_cols);
    if (score < best_score) {
      best_score = score;
      best_score_shift = static_cast<std::size_t>(shift);
    }
  }

  if (best_shift != nullptr) {
    *best_shift = best_score_shift;
  }
  return best_score;
}

Eigen::MatrixXd ScanContextManager::circularShiftColumns(
  const Eigen::MatrixXd & mat,
  std::size_t shift) const
{
  if (mat.cols() == 0) {
    return mat;
  }
  const std::size_t mod_shift = shift % static_cast<std::size_t>(mat.cols());
  if (mod_shift == 0) {
    return mat;
  }

  Eigen::MatrixXd out(mat.rows(), mat.cols());
  for (int col = 0; col < mat.cols(); ++col) {
    const int src_col =
      static_cast<int>((static_cast<std::size_t>(col) + static_cast<std::size_t>(mat.cols()) -
      mod_shift) % static_cast<std::size_t>(mat.cols()));
    out.col(col) = mat.col(src_col);
  }
  return out;
}

}  // namespace iekf_lio
