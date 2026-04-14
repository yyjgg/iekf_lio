#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_types.h>
#include "builtin_interfaces/msg/time.hpp"
#include "iekf/iekf_predictor.hpp"
#include "iekf/iekf_updater.hpp"
#include "imu/imu_integrator.hpp"
#include "imu/noise_model.hpp"
#include "imu/imu_types.hpp"
#include "lidar/cloud_deskewer.hpp"
#include "mapping/voxel_map.hpp"
#include "livox_ros_driver2/msg/custom_msg.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "types.hpp"

using LidarMsg = livox_ros_driver2::msg::CustomMsg;

using namespace std::chrono_literals;

class IekfSlamNode : public rclcpp::Node
{
public:
  IekfSlamNode()
  : Node("iekf_slam_node")
  {
    imu_topic_ = this->declare_parameter<std::string>("imu_topic", "/imu_raw");
    lidar_topic_ = this->declare_parameter<std::string>("lidar_topic", "/points_raw");
    imu_buffer_size_ = this->declare_parameter<int>("imu_buffer_size", 2000);
    lidar_buffer_size_ = this->declare_parameter<int>("lidar_buffer_size", 100);
    reflectivity_filter_enable_ = this->declare_parameter<bool>("reflectivity_filter.enable", true);
    reflectivity_min_ = this->declare_parameter<int>("reflectivity_filter.min", 1);
    reflectivity_max_ = this->declare_parameter<int>("reflectivity_filter.max", 255);
    imu_init_window_size_ = this->declare_parameter<int>("imu_init.window_size", 600);
    imu_init_gyro_var_threshold_ = this->declare_parameter<double>("imu_init.gyro_var_threshold", 1e-4);
    imu_init_accel_var_threshold_ = this->declare_parameter<double>("imu_init.accel_var_threshold", 0.05);
    imu_init_gravity_norm_ = this->declare_parameter<double>("imu_init.gravity_norm", 9.81);
    sigma_acc_ = this->declare_parameter<double>("predictor_noise.sigma_acc", 0.10);
    sigma_gyro_ = this->declare_parameter<double>("predictor_noise.sigma_gyro", 0.01);
    sigma_bg_rw_ = this->declare_parameter<double>("predictor_noise.sigma_bg_rw", 1e-4);
    sigma_ba_rw_ = this->declare_parameter<double>("predictor_noise.sigma_ba_rw", 1e-3);
    sigma_g_rw_ = this->declare_parameter<double>("predictor_noise.sigma_g_rw", 0.0);
    updater_enable_ = this->declare_parameter<bool>("updater.enable", true);
    updater_max_iterations_ = this->declare_parameter<int>("updater.max_iterations", 2);
    updater_max_corr_dist_ = this->declare_parameter<double>("updater.max_correspondence_distance", 1.0);
    updater_plane_k_ = this->declare_parameter<int>("updater.plane_k_neighbors", 10);
    updater_plane_eigen_ratio_ = this->declare_parameter<double>("updater.plane_max_eigen_ratio", 0.1);
    updater_sigma_plane_ = this->declare_parameter<double>("updater.sigma_point_to_plane", 0.2);
    updater_max_abs_residual_ = this->declare_parameter<double>("updater.max_abs_point_to_plane_residual", 0.5);
    updater_max_update_points_ = this->declare_parameter<int>("updater.max_update_points", 1200);
    deskew_enable_ = this->declare_parameter<bool>("deskew.enable", true);
    downsample_enable_ = this->declare_parameter<bool>("downsample.enable", true);
    downsample_leaf_size_ = this->declare_parameter<double>("downsample.leaf_size", 0.2);
    map_voxel_size_ = this->declare_parameter<double>("local_map.voxel_size", 0.5);
    map_max_voxels_ = this->declare_parameter<int>("local_map.max_voxels", 200000);
    map_local_window_enable_ = this->declare_parameter<bool>("local_map.window.enable", true);
    map_local_window_radius_xy_ = this->declare_parameter<double>("local_map.window.radius_xy", 60.0);
    map_local_window_half_height_ = this->declare_parameter<double>("local_map.window.half_height", 15.0);
    const std::vector<double> ext_r = this->declare_parameter<std::vector<double>>(
      "extrinsic.lidar_to_imu.R",
      {1.0, 0.0, 0.0,
       0.0, 1.0, 0.0,
       0.0, 0.0, 1.0});
    const std::vector<double> ext_p = this->declare_parameter<std::vector<double>>(
      "extrinsic.lidar_to_imu.P",
      {0.0, 0.0, 0.0});

    if (ext_r.size() == 9) {
      extrinsic_r_il_ <<
        ext_r[0], ext_r[1], ext_r[2],
        ext_r[3], ext_r[4], ext_r[5],
        ext_r[6], ext_r[7], ext_r[8];
    } else {
      RCLCPP_WARN(
        this->get_logger(),
        "extrinsic.lidar_to_imu.R size=%zu, expected 9. Use identity instead.",
        ext_r.size());
      extrinsic_r_il_.setIdentity();
    }

    if (ext_p.size() == 3) {
      extrinsic_t_il_ << ext_p[0], ext_p[1], ext_p[2];
    } else {
      RCLCPP_WARN(
        this->get_logger(),
        "extrinsic.lidar_to_imu.P size=%zu, expected 3. Use zero instead.",
        ext_p.size());
      extrinsic_t_il_.setZero();
    }

    iekf_lio::ImuInitConfig init_cfg;
    init_cfg.window_size = static_cast<std::size_t>(std::max(1, imu_init_window_size_));
    init_cfg.gyro_var_threshold = imu_init_gyro_var_threshold_;
    init_cfg.accel_var_threshold = imu_init_accel_var_threshold_;
    init_cfg.gravity_norm = imu_init_gravity_norm_;
    imu_initializer_.setConfig(init_cfg);

    iekf_lio::IekfPredictorNoise pred_noise;
    pred_noise.sigma_acc = sigma_acc_;
    pred_noise.sigma_gyro = sigma_gyro_;
    pred_noise.sigma_bg_rw = sigma_bg_rw_;
    pred_noise.sigma_ba_rw = sigma_ba_rw_;
    pred_noise.sigma_g_rw = sigma_g_rw_;
    iekf_predictor_ = iekf_lio::IekfPredictor(pred_noise);
    iekf_lio::IekfUpdaterConfig updater_cfg;
    updater_cfg.max_iterations = std::max(1, updater_max_iterations_);
    updater_cfg.max_correspondence_distance = std::max(0.05, updater_max_corr_dist_);
    updater_cfg.plane_k_neighbors = std::max(3, updater_plane_k_);
    updater_cfg.plane_max_eigen_ratio = std::max(1e-4, updater_plane_eigen_ratio_);
    updater_cfg.sigma_point_to_plane = std::max(1e-3, updater_sigma_plane_);
    updater_cfg.max_abs_point_to_plane_residual = std::max(1e-3, updater_max_abs_residual_);
    updater_cfg.max_update_points = std::max(100, updater_max_update_points_);
    iekf_updater_.setConfig(updater_cfg);
    voxel_map_.setVoxelSize(map_voxel_size_);
    voxel_map_.setMaxVoxels(static_cast<std::size_t>(std::max(1, map_max_voxels_)));
    voxel_map_.setLocalWindowEnabled(map_local_window_enable_);
    voxel_map_.setLocalWindow(map_local_window_radius_xy_, map_local_window_half_height_);

    const auto sensor_qos = rclcpp::SensorDataQoS();

    imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
      imu_topic_,
      sensor_qos,
      std::bind(&IekfSlamNode::imuCallback, this, std::placeholders::_1));

    lidar_sub_ = this->create_subscription<LidarMsg>(
      lidar_topic_,
      sensor_qos,
      std::bind(&IekfSlamNode::lidarCallback, this, std::placeholders::_1));

    // Algorithms should not run in subscription callbacks:
    // heavy compute here blocks executor threads, increases callback latency,
    // and makes IMU/LiDAR scheduling nondeterministic under load.
    processing_thread_ = std::thread(&IekfSlamNode::processingLoop, this);

    status_timer_ = this->create_wall_timer(
      2s, std::bind(&IekfSlamNode::reportInputStatus, this));

    RCLCPP_INFO(
      this->get_logger(),
      "iekf_slam_node started. Subscribing to IMU: %s, LiDAR: %s",
      imu_topic_.c_str(),
      lidar_topic_.c_str());
  }

  ~IekfSlamNode() override
  {
    {
      std::lock_guard<std::mutex> lk(processing_mutex_);
      stop_requested_ = true;
    }
    data_cv_.notify_all();

    if (processing_thread_.joinable()) {
      processing_thread_.join();
    }
  }

private:
  void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
  {
    {
      std::lock_guard<std::mutex> lock(imu_mutex_);
      if (!isMonotonicStamp(msg->header.stamp, latest_imu_stamp_)) {
        RCLCPP_WARN(
          this->get_logger(),
          "Dropped out-of-order IMU msg: current=%.3f last=%.3f",
          stampToSec(msg->header.stamp),
          stampToSec(latest_imu_stamp_));
        return;
      }
      imu_buffer_.push_back(msg);
      trimBuffer(imu_buffer_, static_cast<std::size_t>(std::max(1, imu_buffer_size_)));
      latest_imu_stamp_ = msg->header.stamp;
    }
    data_cv_.notify_one();
  }

  void lidarCallback(const LidarMsg::SharedPtr msg)
  {
    {
      std::lock_guard<std::mutex> lock(lidar_mutex_);
      if (!isMonotonicStamp(msg->header.stamp, latest_lidar_stamp_)) {
        RCLCPP_WARN(
          this->get_logger(),
          "Dropped out-of-order LiDAR msg: current=%.3f last=%.3f",
          stampToSec(msg->header.stamp),
          stampToSec(latest_lidar_stamp_));
        return;
      }
      lidar_buffer_.push_back(msg);
      trimBuffer(lidar_buffer_, static_cast<std::size_t>(std::max(1, lidar_buffer_size_)));
      latest_lidar_stamp_ = msg->header.stamp;
    }
    data_cv_.notify_one();
  }

  void processingLoop()
  {
    while (true) {
      {
        std::unique_lock<std::mutex> lk(processing_mutex_);
        data_cv_.wait_for(
          lk, 20ms,
          [this]()
          {
            return stop_requested_ || hasPendingLidar();
          });

        if (stop_requested_) {
          return;
        }
      }

      (void)tryProcessOneScan();
    }
  }

  bool tryProcessOneScan()
  {
    LidarMsg::SharedPtr lidar_msg;
    {
      std::lock_guard<std::mutex> lock(lidar_mutex_);
      if (lidar_buffer_.empty()) {
        return false;
      }
      // Copy the oldest scan and release lock quickly.
      lidar_msg = lidar_buffer_.front();
    }

    const auto scan_begin_time = lidar_msg->header.stamp;
    const auto scan_end_time = computeScanEndTime(*lidar_msg);

    std::size_t raw_point_count = getPointCount(*lidar_msg);
    LidarMsg::SharedPtr filtered_lidar_msg = filterByReflectivity(lidar_msg);
    std::size_t filtered_point_count = getPointCount(*filtered_lidar_msg);
    if (filtered_point_count == 0) {
      {
        std::lock_guard<std::mutex> lock(lidar_mutex_);
        if (!lidar_buffer_.empty()) {
          lidar_buffer_.pop_front();
        }
      }
      RCLCPP_WARN_THROTTLE(
        this->get_logger(),
        *this->get_clock(),
        1000,
        "Drop one scan after reflectivity filter: raw_points=%zu",
        raw_point_count);
      return true;
    }

    std::vector<sensor_msgs::msg::Imu::SharedPtr> imu_slice;
    bool drop_stale_scan = false;
    builtin_interfaces::msg::Time imu_begin;
    builtin_interfaces::msg::Time imu_end;
    {
      std::lock_guard<std::mutex> lock(imu_mutex_);
      // TODO: Check imu_buffer_ coverage for [scan_begin_time, scan_end_time].
      if (imu_buffer_.empty()) {
        RCLCPP_DEBUG(this->get_logger(), "Waiting IMU: imu_buffer_ is empty.");
        return false;
      }

      imu_begin = imu_buffer_.front()->header.stamp;
      imu_end = imu_buffer_.back()->header.stamp;

      // Old IMU samples are already dropped, this scan can no longer be covered.
      if (isTimeLT(scan_begin_time, imu_begin)) {
        drop_stale_scan = true;
      }

      const bool covered = isTimeLE(imu_begin, scan_begin_time) && isTimeLE(scan_end_time, imu_end);
      if (!covered && !drop_stale_scan) {
        RCLCPP_INFO_THROTTLE(
          this->get_logger(),
          *this->get_clock(),
          1000,
          "Waiting more IMU for scan %.3f: imu_range=[%.3f, %.3f]",
          stampToSec(scan_begin_time),
          stampToSec(imu_begin),
          stampToSec(imu_end));
        return false;
      }

      if (!drop_stale_scan) {
        // Extract IMU messages for [scan_begin, scan_end], and keep one support sample
        // before/after interval for robust interpolation at scan boundaries.
        sensor_msgs::msg::Imu::SharedPtr before_begin;
        sensor_msgs::msg::Imu::SharedPtr after_end;
        for (const auto & imu_msg : imu_buffer_) {
          const auto & t = imu_msg->header.stamp;
          if (isTimeLT(t, scan_begin_time)) {
            before_begin = imu_msg;
            continue;
          }
          if (isTimeLE(scan_begin_time, t) && isTimeLE(t, scan_end_time)) {
            imu_slice.push_back(imu_msg);
            continue;
          }
          if (isTimeLT(scan_end_time, t)) {
            after_end = imu_msg;
            break;
          }
        }
        if (before_begin) {
          imu_slice.insert(imu_slice.begin(), before_begin);
        }
        if (after_end) {
          imu_slice.push_back(after_end);
        }
      }
    }

    if (drop_stale_scan) {
      {
        std::lock_guard<std::mutex> lock(lidar_mutex_);
        if (!lidar_buffer_.empty()) {
          lidar_buffer_.pop_front();
        }
      }
      RCLCPP_WARN_THROTTLE(
        this->get_logger(),
        *this->get_clock(),
        1000,
        "Drop stale scan: scan_begin=%.3f imu_range=[%.3f, %.3f]",
        stampToSec(scan_begin_time),
        stampToSec(imu_begin),
        stampToSec(imu_end));
      return true;
    }

    {
      std::lock_guard<std::mutex> lock(lidar_mutex_);
      if (!lidar_buffer_.empty()) {
        lidar_buffer_.pop_front();
      }
    }

    iekf_lio::LidarScan lidar_scan_internal =
      convertLidarScanToInternal(*filtered_lidar_msg, scan_begin_time, scan_end_time);
    const iekf_lio::ImuTrack imu_track_internal = convertImuSliceToInternal(imu_slice);
    const iekf_lio::ImuInitResult imu_init_result = imu_initializer_.update(imu_track_internal);
    if (!iekf_state_.is_initialized) {
      if (!imu_init_result.initialized) {
        RCLCPP_INFO_THROTTLE(
          this->get_logger(),
          *this->get_clock(),
          1000,
          "IMU init waiting: samples=%zu gyro_var=%.6e accel_var=%.6e",
          imu_init_result.used_samples,
          imu_init_result.gyro_var_norm,
          imu_init_result.accel_var_norm);
        return true;
      }

      iekf_predictor_.initializeState(iekf_state_);
      iekf_state_.x.b_g = imu_init_result.gyro_bias;
      iekf_state_.x.b_a = imu_init_result.accel_bias;
      iekf_state_.x.g_w = imu_init_result.gravity_w;
      RCLCPP_INFO(
        this->get_logger(),
        "IMU init done: b_g=(%.6f,%.6f,%.6f) b_a=(%.6f,%.6f,%.6f) g=(%.6f,%.6f,%.6f)",
        iekf_state_.x.b_g.x(), iekf_state_.x.b_g.y(), iekf_state_.x.b_g.z(),
        iekf_state_.x.b_a.x(), iekf_state_.x.b_a.y(), iekf_state_.x.b_a.z(),
        iekf_state_.x.g_w.x(), iekf_state_.x.g_w.y(), iekf_state_.x.g_w.z());
    }

    std::vector<iekf_lio::ImuPredictedState> imu_pred_states;
    iekf_predictor_.predictWithMidpoint(imu_track_internal, iekf_state_, &imu_pred_states);
    if (deskew_enable_) {
      const iekf_lio::LidarToImuExtrinsic extrinsic {extrinsic_r_il_, extrinsic_t_il_};
      (void)deskewer_.deskewToImuEnd(lidar_scan_internal, imu_pred_states, extrinsic);
    }

    const iekf_lio::LidarScan scan_for_update =
      (downsample_enable_ && downsample_leaf_size_ > 1e-3)
      ? downsampleScan(lidar_scan_internal, downsample_leaf_size_)
      : lidar_scan_internal;

    std::vector<Eigen::Vector3d> points_world;
    std::size_t map_inserted = 0;
    std::size_t map_pruned = 0;
    std::size_t update_corr = 0;
    double update_rmse = 0.0;
    bool used_update = false;
    std::string update_skip_reason = "none";

    if (!local_map_initialized_) {
      // First valid scan: explicitly skip IEKF update, use current predicted state to bootstrap map.
      points_world = transformDeskewedScanToWorld(scan_for_update, iekf_state_.x);
      map_inserted = voxel_map_.insertPoints(points_world);
      local_map_initialized_ = true;
      RCLCPP_INFO(
        this->get_logger(),
        "Local map initialized from first valid scan (skip IEKF update): world_points=%zu inserted=%zu map_voxels=%zu",
        points_world.size(),
        map_inserted,
        voxel_map_.size());
    } else {
      // Update branch: prune local map, run IEKF updater, then insert points with posterior state.
      map_pruned = voxel_map_.pruneOutsideLocalWindow(iekf_state_.x.p_wb);
      if (updater_enable_) {
        const pcl::PointCloud<pcl::PointXYZ>::ConstPtr map_cloud = voxel_map_.toPointCloud();
        iekf_lio::IekfUpdateResult upd;
        used_update = iekf_updater_.updatePoseWithPointToMap(
          scan_for_update,
          map_cloud,
          iekf_state_,
          &upd);
        update_corr = upd.correspondences;
        update_rmse = upd.rmse;
        if (!used_update) {
          update_skip_reason = "update_rejected";
        }
      } else {
        update_skip_reason = "update_disabled";
      }
      points_world = transformDeskewedScanToWorld(scan_for_update, iekf_state_.x);
      map_inserted = voxel_map_.insertPoints(points_world);
      RCLCPP_DEBUG(
        this->get_logger(),
        "Local map integrated one scan: used_update=%s skip=%s corr=%zu rmse=%.4f pruned=%zu world_points=%zu inserted=%zu map_voxels=%zu",
        used_update ? "true" : "false",
        update_skip_reason.c_str(),
        update_corr,
        update_rmse,
        map_pruned,
        points_world.size(),
        map_inserted,
        voxel_map_.size());
    }
    const std::array<double, 3> pred_rpy_rad = rotationMatrixToRpy(iekf_state_.x.r_wb);//debug用的

    RCLCPP_INFO(
      this->get_logger(),
      "Scheduled one scan: [%.3f, %.3f], imu_in_window=%zu, points=%zu->%zu, deskew_enable=%s, downsampled=%zu, used_update=%s, update_skip=%s, update_corr=%zu, update_rmse=%.4f, map_pruned=%zu, world_points=%zu, map_voxels=%zu, pred_p=(%.3f,%.3f,%.3f), pred_v=(%.3f,%.3f,%.3f), pred_rpy_rad=(%.3f,%.3f,%.3f)",
      stampToSec(scan_begin_time),
      stampToSec(scan_end_time),
      imu_track_internal.size(),
      raw_point_count,
      filtered_point_count,
      deskew_enable_ ? "true" : "false",
      scan_for_update.points.size(),
      used_update ? "true" : "false",
      update_skip_reason.c_str(),
      update_corr,
      update_rmse,
      map_pruned,
      points_world.size(),
      voxel_map_.size(),
      iekf_state_.x.p_wb.x(),
      iekf_state_.x.p_wb.y(),
      iekf_state_.x.p_wb.z(),
      iekf_state_.x.v_wb.x(),
      iekf_state_.x.v_wb.y(),
      iekf_state_.x.v_wb.z(),
      pred_rpy_rad[0],
      pred_rpy_rad[1],
      pred_rpy_rad[2]);
    return true;
  }

  std::array<double, 3> rotationMatrixToRpy(const iekf_lio::IekfMat3 & R) const
  {
    const double roll = std::atan2(R(2, 1), R(2, 2));
    const double pitch = std::asin(std::clamp(-R(2, 0), -1.0, 1.0));
    const double yaw = std::atan2(R(1, 0), R(0, 0));
    return {roll, pitch, yaw};
  }

  std::vector<Eigen::Vector3d> transformDeskewedScanToWorld(
    const iekf_lio::LidarScan & scan,
    const iekf_lio::IekfNominalState18 & state) const
  {
    std::vector<Eigen::Vector3d> points_world;
    points_world.reserve(scan.points.size());
    for (const auto & pt : scan.points) {
      if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
        continue;
      }
      const Eigen::Vector3d p_i_end(pt.x, pt.y, pt.z);
      points_world.push_back(state.r_wb * p_i_end + state.p_wb);
    }
    return points_world;
  }

  iekf_lio::LidarScan downsampleScan(
    const iekf_lio::LidarScan & scan,
    double leaf_size) const
  {
    if (scan.points.empty() || leaf_size <= 1e-6) {
      return scan;
    }

    auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    cloud->reserve(scan.points.size());
    for (const auto & pt : scan.points) {
      if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
        continue;
      }
      pcl::PointXYZ p;
      p.x = pt.x;
      p.y = pt.y;
      p.z = pt.z;
      cloud->push_back(p);
    }
    if (cloud->empty()) {
      return scan;
    }
    cloud->width = static_cast<std::uint32_t>(cloud->size());
    cloud->height = 1;
    cloud->is_dense = false;

    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(static_cast<float>(leaf_size),
      static_cast<float>(leaf_size),
      static_cast<float>(leaf_size));
    pcl::PointCloud<pcl::PointXYZ> filtered;
    vg.filter(filtered);

    iekf_lio::LidarScan out;
    out.frame_id = scan.frame_id;
    out.scan_begin_time_s = scan.scan_begin_time_s;
    out.scan_end_time_s = scan.scan_end_time_s;
    out.timebase_ns = scan.timebase_ns;
    out.points.reserve(filtered.size());
    for (const auto & p : filtered.points) {
      iekf_lio::PointXYZIRTL out_pt;
      out_pt.x = p.x;
      out_pt.y = p.y;
      out_pt.z = p.z;
      out.points.push_back(out_pt);
    }
    return out;
  }

  iekf_lio::ImuTrack convertImuSliceToInternal(
    const std::vector<sensor_msgs::msg::Imu::SharedPtr> & imu_slice) const
  {
    iekf_lio::ImuTrack out;
    out.reserve(imu_slice.size());

    for (const auto & imu_msg : imu_slice) {
      iekf_lio::ImuSample sample;
      sample.time_s = stampToSec(imu_msg->header.stamp);
      sample.accel_mps2 << imu_msg->linear_acceleration.x,
        imu_msg->linear_acceleration.y,
        imu_msg->linear_acceleration.z;
      sample.gyro_rps << imu_msg->angular_velocity.x,
        imu_msg->angular_velocity.y,
        imu_msg->angular_velocity.z;
      out.push_back(sample);
    }
    return out;
  }

  iekf_lio::LidarScan convertLidarScanToInternal(
    const LidarMsg & lidar_msg,
    const builtin_interfaces::msg::Time & scan_begin_time,
    const builtin_interfaces::msg::Time & scan_end_time) const
  {
    iekf_lio::LidarScan out;
    out.frame_id = lidar_msg.header.frame_id;
    out.scan_begin_time_s = stampToSec(scan_begin_time);
    out.scan_end_time_s = stampToSec(scan_end_time);
    out.timebase_ns = lidar_msg.timebase;
    out.points.reserve(lidar_msg.points.size());
    for (const auto & pt : lidar_msg.points) {
      iekf_lio::PointXYZIRTL p;
      p.x = pt.x;
      p.y = pt.y;
      p.z = pt.z;
      p.reflectivity = pt.reflectivity;
      p.tag = pt.tag;
      p.line = pt.line;
      p.relative_time_s = static_cast<double>(pt.offset_time) * 1e-9;
      out.points.push_back(p);
    }
    return out;
  }

  builtin_interfaces::msg::Time computeScanEndTime(const LidarMsg & lidar_msg) const
  {
    std::uint32_t max_offset_ns = 0;
    for (const auto & point : lidar_msg.points) {
      if (point.offset_time > max_offset_ns) {
        max_offset_ns = point.offset_time;
      }
    }
    return addNanoseconds(lidar_msg.header.stamp, static_cast<std::uint64_t>(max_offset_ns));
  }

  LidarMsg::SharedPtr filterByReflectivity(const LidarMsg::SharedPtr & lidar_msg) const
  {
    if (!reflectivity_filter_enable_) {
      return lidar_msg;
    }

    auto out = std::make_shared<LidarMsg>(*lidar_msg);
    out->points.clear();
    out->points.reserve(lidar_msg->points.size());

    const std::uint8_t min_refl = static_cast<std::uint8_t>(std::max(0, reflectivity_min_));
    const std::uint8_t max_refl = static_cast<std::uint8_t>(std::min(255, reflectivity_max_));
    const std::uint8_t lower = std::min(min_refl, max_refl);
    const std::uint8_t upper = std::max(min_refl, max_refl);

    for (const auto & pt : lidar_msg->points) {
      if (pt.reflectivity >= lower && pt.reflectivity <= upper) {
        out->points.push_back(pt);
      }
    }
    out->point_num = static_cast<std::uint32_t>(out->points.size());
    return out;
  }

  std::size_t getPointCount(const LidarMsg & lidar_msg) const
  {
    return lidar_msg.points.size();
  }

  builtin_interfaces::msg::Time addNanoseconds(
    const builtin_interfaces::msg::Time & stamp,
    std::uint64_t add_ns) const
  {
    constexpr std::uint64_t kNsPerSec = 1000000000ULL;

    const std::uint64_t base_ns = static_cast<std::uint64_t>(stamp.nanosec);
    const std::uint64_t sum_ns = base_ns + add_ns;

    builtin_interfaces::msg::Time out = stamp;
    out.sec = stamp.sec + static_cast<std::int32_t>(sum_ns / kNsPerSec);
    out.nanosec = static_cast<std::uint32_t>(sum_ns % kNsPerSec);
    return out;
  }

  void reportInputStatus()
  {
    std::size_t imu_size = 0;
    std::size_t lidar_size = 0;

    {
      std::lock_guard<std::mutex> lock(imu_mutex_);
      imu_size = imu_buffer_.size();
    }

    {
      std::lock_guard<std::mutex> lock(lidar_mutex_);
      lidar_size = lidar_buffer_.size();
    }

    RCLCPP_INFO(
      this->get_logger(),
      "input buffers: imu=%zu lidar=%zu latest_imu=%.3f latest_lidar=%.3f",
      imu_size,
      lidar_size,
      stampToSec(latest_imu_stamp_),
      stampToSec(latest_lidar_stamp_));
  }

  template<typename T>
  void trimBuffer(std::deque<T> & buffer, std::size_t max_size)
  {
    while (buffer.size() > max_size) {
      buffer.pop_front();
    }
  }

  double stampToSec(const builtin_interfaces::msg::Time & stamp) const
  {
    return rclcpp::Time(stamp).seconds();
  }

  bool hasPendingLidar()
  {
    std::lock_guard<std::mutex> lock(lidar_mutex_);
    return !lidar_buffer_.empty();
  }

  bool isMonotonicStamp(
    const builtin_interfaces::msg::Time & current,
    const builtin_interfaces::msg::Time & previous) const
  {
    if (previous.sec == 0 && previous.nanosec == 0) {
      return true;
    }
    return isTimeLE(previous, current);
  }

  bool isTimeLE(
    const builtin_interfaces::msg::Time & lhs,
    const builtin_interfaces::msg::Time & rhs) const
  {
    if (lhs.sec < rhs.sec) {
      return true;
    }
    if (lhs.sec > rhs.sec) {
      return false;
    }
    return lhs.nanosec <= rhs.nanosec;
  }

  bool isTimeLT(
    const builtin_interfaces::msg::Time & lhs,
    const builtin_interfaces::msg::Time & rhs) const
  {
    return isTimeLE(lhs, rhs) && !isTimeLE(rhs, lhs);
  }

  std::string imu_topic_;
  std::string lidar_topic_;
  int imu_buffer_size_;
  int lidar_buffer_size_;
  bool reflectivity_filter_enable_;
  int reflectivity_min_;
  int reflectivity_max_;
  int imu_init_window_size_;
  double imu_init_gyro_var_threshold_;
  double imu_init_accel_var_threshold_;
  double imu_init_gravity_norm_;
  double sigma_acc_;
  double sigma_gyro_;
  double sigma_bg_rw_;
  double sigma_ba_rw_;
  double sigma_g_rw_;
  bool updater_enable_;
  int updater_max_iterations_;
  double updater_max_corr_dist_;
  int updater_plane_k_;
  double updater_plane_eigen_ratio_;
  double updater_sigma_plane_;
  double updater_max_abs_residual_;
  int updater_max_update_points_;
  bool deskew_enable_;
  bool downsample_enable_;
  double downsample_leaf_size_;
  double map_voxel_size_;
  int map_max_voxels_;
  bool map_local_window_enable_;
  double map_local_window_radius_xy_;
  double map_local_window_half_height_;
  Eigen::Vector3d extrinsic_t_il_ = Eigen::Vector3d::Zero();
  Eigen::Matrix3d extrinsic_r_il_ = Eigen::Matrix3d::Identity();

  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<LidarMsg>::SharedPtr lidar_sub_;
  rclcpp::TimerBase::SharedPtr status_timer_;

  std::deque<sensor_msgs::msg::Imu::SharedPtr> imu_buffer_;
  std::deque<LidarMsg::SharedPtr> lidar_buffer_;
  std::mutex imu_mutex_;
  std::mutex lidar_mutex_;
  std::condition_variable data_cv_;
  std::mutex processing_mutex_;
  std::thread processing_thread_;
  bool stop_requested_ = false;
  iekf_lio::ImuIntegrator imu_integrator_;
  iekf_lio::ImuStaticInitializer imu_initializer_;
  iekf_lio::IekfPredictor iekf_predictor_;
  iekf_lio::IekfUpdater iekf_updater_;
  iekf_lio::IekfState18 iekf_state_;
  iekf_lio::CloudDeskewer deskewer_;
  iekf_lio::VoxelMap voxel_map_;
  bool local_map_initialized_ = false;

  builtin_interfaces::msg::Time latest_imu_stamp_;
  builtin_interfaces::msg::Time latest_lidar_stamp_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<IekfSlamNode>());
  rclcpp::shutdown();
  return 0;
}
