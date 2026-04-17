#include <algorithm>
#include <array>
#include <atomic>
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
#include <pcl_conversions/pcl_conversions.h>
#include "builtin_interfaces/msg/time.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "iekf/iekf_predictor.hpp"
#include "iekf/iekf_updater.hpp"
#include "imu/imu_integrator.hpp"
#include "imu/noise_model.hpp"
#include "imu/imu_types.hpp"
#include "lidar/cloud_deskewer.hpp"
#include "mapping/voxel_map.hpp"
#include "livox_ros_driver2/msg/custom_msg.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "tf2_ros/transform_broadcaster.h"
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
    distance_filter_enable_ = this->declare_parameter<bool>("distance_filter.enable", true);
    distance_min_m_ = this->declare_parameter<double>("distance_filter.min_m", 1.0);
    distance_max_m_ = this->declare_parameter<double>("distance_filter.max_m", 100.0);
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
    const double downsample_leaf_size_legacy =
      this->declare_parameter<double>("downsample.leaf_size", 0.2);
    downsample_update_leaf_size_ = this->declare_parameter<double>(
      "downsample.update_leaf_size", downsample_leaf_size_legacy);
    downsample_map_leaf_size_ = this->declare_parameter<double>(
      "downsample.map_leaf_size", std::max(1e-3, 0.5 * downsample_leaf_size_legacy));
    map_voxel_size_ = this->declare_parameter<double>("local_map.voxel_size", 0.5);
    map_block_size_ = this->declare_parameter<double>("local_map.block_size", 20.0);
    map_history_max_blocks_ = this->declare_parameter<int>("local_map.history.max_blocks", 200);
    map_history_enable_ = this->declare_parameter<bool>("local_map.history.enable", true);
    map_history_radius_xy_ = this->declare_parameter<double>("local_map.history.radius_xy", 120.0);
    map_history_half_height_ = this->declare_parameter<double>("local_map.history.half_height", 30.0);
    map_active_window_enable_ = this->declare_parameter<bool>("local_map.active.enable", true);
    map_active_radius_xy_ = this->declare_parameter<double>("local_map.active.radius_xy", 60.0);
    map_active_half_height_ = this->declare_parameter<double>("local_map.active.half_height", 15.0);
    map_active_angle_enable_ = this->declare_parameter<bool>("local_map.active.angle.enable", false);
    map_active_half_fov_deg_ = this->declare_parameter<double>("local_map.active.angle.half_fov_deg", 30.0);
    odom_topic_ = this->declare_parameter<std::string>("publish.odom_topic", "/iekf/odom");
    path_topic_ = this->declare_parameter<std::string>("publish.path_topic", "/iekf/path");
    points_world_topic_ = this->declare_parameter<std::string>(
      "publish.points_world_topic", "/iekf/points_world");
    map_frame_id_ = this->declare_parameter<std::string>("publish.map_frame_id", "map");
    base_frame_id_ = this->declare_parameter<std::string>("publish.base_frame_id", "base_link");
    path_max_length_ = this->declare_parameter<int>("publish.path.max_length", 2000);
    publish_tf_ = this->declare_parameter<bool>("publish.tf.enable", true);
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
    voxel_map_.setBlockSize(map_block_size_);
    voxel_map_.setMaxBlocks(static_cast<std::size_t>(std::max(1, map_history_max_blocks_)));
    voxel_map_.setHistoryWindowEnabled(map_history_enable_);
    voxel_map_.setHistoryWindow(map_history_radius_xy_, map_history_half_height_);
    voxel_map_.setActiveWindowEnabled(map_active_window_enable_);
    voxel_map_.setActiveWindow(map_active_radius_xy_, map_active_half_height_);
    voxel_map_.setActiveAngleFilterEnabled(map_active_angle_enable_);
    voxel_map_.setActiveHalfFovDeg(map_active_half_fov_deg_);

    const auto sensor_qos = rclcpp::SensorDataQoS();

    imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
      imu_topic_,
      sensor_qos,
      std::bind(&IekfSlamNode::imuCallback, this, std::placeholders::_1));

    lidar_sub_ = this->create_subscription<LidarMsg>(
      lidar_topic_,
      sensor_qos,
      std::bind(&IekfSlamNode::lidarCallback, this, std::placeholders::_1));

    odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>(odom_topic_, 10);
    path_pub_ = this->create_publisher<nav_msgs::msg::Path>(path_topic_, 10);
    points_world_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(points_world_topic_, 10);
    if (publish_tf_) {
      tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    }
    path_msg_.header.frame_id = map_frame_id_;

    // Algorithms should not run in subscription callbacks:
    // heavy compute here blocks executor threads, increases callback latency,
    // and makes IMU/LiDAR scheduling nondeterministic under load.
    processing_thread_ = std::thread(&IekfSlamNode::processingLoop, this);

    status_timer_ = this->create_wall_timer(
      2s, std::bind(&IekfSlamNode::reportInputStatus, this));

    RCLCPP_INFO(
      this->get_logger(),
      "iekf_slam_node started. Subscribing to IMU: %s, LiDAR: %s. Publish odom: %s, path: %s, tf(map->%s): %s",
      imu_topic_.c_str(),
      lidar_topic_.c_str(),
      odom_topic_.c_str(),
      path_topic_.c_str(),
      base_frame_id_.c_str(),
      publish_tf_ ? "true" : "false");
    RCLCPP_INFO(
      this->get_logger(),
      "Publish world cloud: %s",
      points_world_topic_.c_str());
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
  struct LidarPreprocessResult
  {
    iekf_lio::LidarScan scan;
    std::size_t raw_point_count = 0;
    std::size_t filtered_point_count = 0;
  };

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
      ++lidar_received_count_;
      const std::size_t max_lidar_size = static_cast<std::size_t>(std::max(1, lidar_buffer_size_));
      if (lidar_buffer_.size() >= max_lidar_size) {
        lidar_buffer_.pop_front();
        ++lidar_drop_overflow_count_;
      }
      lidar_buffer_.push_back(msg);
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
    using SteadyClock = std::chrono::steady_clock;
    const auto total_t0 = SteadyClock::now();

    LidarMsg::SharedPtr lidar_msg;
    {
      std::lock_guard<std::mutex> lock(lidar_mutex_);
      if (lidar_buffer_.empty()) {
        return false;
      }
      // Copy the oldest scan and release lock quickly.
      lidar_msg = lidar_buffer_.front();
    }

    const auto prep_t0 = SteadyClock::now();
    const auto scan_begin_time = lidar_msg->header.stamp;
    const auto scan_end_time = computeScanEndTime(*lidar_msg);

    const auto prep_lidar_result = preprocessLidarScanToInternal(
      *lidar_msg, scan_begin_time, scan_end_time);
    const std::size_t raw_point_count = prep_lidar_result.raw_point_count;
    const std::size_t filtered_point_count = prep_lidar_result.filtered_point_count;
    if (filtered_point_count == 0) {
      {
        std::lock_guard<std::mutex> lock(lidar_mutex_);
        if (!lidar_buffer_.empty()) {
          lidar_buffer_.pop_front();
        }
      }
      ++lidar_drop_reflectivity_count_;
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
      ++lidar_drop_stale_count_;
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

    std::uint64_t prep_ns = 0;
    std::uint64_t predict_ns = 0;
    std::uint64_t deskew_ns = 0;
    std::uint64_t downsample_ns = 0;
    std::uint64_t update_ns = 0;
    std::uint64_t map_ns = 0;

    iekf_lio::LidarScan lidar_scan_internal = prep_lidar_result.scan;
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
    prep_ns = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(SteadyClock::now() - prep_t0).count());

    std::vector<iekf_lio::ImuPredictedState> imu_pred_states;
    const auto predict_t0 = SteadyClock::now();
    iekf_predictor_.predictWithMidpoint(imu_track_internal, iekf_state_, &imu_pred_states);
    predict_ns = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(SteadyClock::now() - predict_t0).count());

    iekf_lio::LidarScanXYZ lidar_scan_xyz;
    if (deskew_enable_) {
      const auto deskew_t0 = SteadyClock::now();
      const iekf_lio::LidarToImuExtrinsic extrinsic {extrinsic_r_il_, extrinsic_t_il_};
      lidar_scan_xyz = deskewer_.deskewToImuEnd(
        lidar_scan_internal, imu_pred_states, extrinsic);
      deskew_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(SteadyClock::now() - deskew_t0).count());
    } else {
      lidar_scan_xyz = convertLidarScanToXYZ(lidar_scan_internal);
    }

    const auto downsample_t0 = SteadyClock::now();
    const iekf_lio::LidarScanXYZ scan_for_update =
      (downsample_enable_ && downsample_update_leaf_size_ > 1e-3)
      ? downsampleScan(lidar_scan_xyz, downsample_update_leaf_size_)
      : lidar_scan_xyz;
    const iekf_lio::LidarScanXYZ scan_for_map =
      (downsample_enable_ && downsample_map_leaf_size_ > 1e-3)
      ? downsampleScan(lidar_scan_xyz, downsample_map_leaf_size_)
      : lidar_scan_xyz;
    downsample_ns = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(SteadyClock::now() - downsample_t0).count());

    std::vector<Eigen::Vector3d> points_world;
    std::size_t map_inserted = 0;
    std::size_t map_pruned = 0;
    std::size_t history_blocks = 0;
    std::size_t active_blocks = 0;
    std::size_t update_corr = 0;
    double update_rmse = 0.0;
    bool used_update = false;
    std::string update_skip_reason = "none";
    const Eigen::Vector3d lidar_origin_w = currentLidarOriginWorld();
    const Eigen::Vector3d lidar_forward_w = currentLidarForwardWorld();

    if (!local_map_initialized_) {
      // First valid scan: explicitly skip IEKF update, use current predicted state to bootstrap map.
      const auto map_t0 = SteadyClock::now();
      points_world = transformDeskewedScanToWorld(scan_for_map, iekf_state_.x);
      map_inserted = voxel_map_.insertPoints(points_world);
      map_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(SteadyClock::now() - map_t0).count());
      local_map_initialized_ = true;
      const auto history_prune_t0 = SteadyClock::now();
      map_pruned = voxel_map_.pruneHistoryBlocks(iekf_state_.x.p_wb);
      map_ns += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(SteadyClock::now() - history_prune_t0).count());
      history_blocks = voxel_map_.blockCount();
      active_blocks = voxel_map_.activeBlockCount(
        iekf_state_.x.p_wb, lidar_origin_w, lidar_forward_w);
      RCLCPP_INFO(
        this->get_logger(),
        "Local map initialized from first valid scan (skip IEKF update): world_points=%zu inserted=%zu pruned=%zu map_voxels=%zu history_blocks=%zu active_blocks=%zu",
        points_world.size(),
        map_inserted,
        map_pruned,
        voxel_map_.size(),
        history_blocks,
        active_blocks);
    } else {
      // Update branch: run IEKF updater on active blocks,
      // then insert points with posterior state.
      if (updater_enable_) {
        const auto update_t0 = SteadyClock::now();
        const pcl::PointCloud<pcl::PointXYZ>::ConstPtr map_cloud =
          voxel_map_.exportActivePointCloud(iekf_state_.x.p_wb, lidar_origin_w, lidar_forward_w);
        active_blocks = voxel_map_.cachedActiveBlockCount();
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
        update_ns = static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(SteadyClock::now() - update_t0).count());
      } else {
        update_skip_reason = "update_disabled";
      }

      const auto map_insert_t0 = SteadyClock::now();
      points_world = transformDeskewedScanToWorld(scan_for_map, iekf_state_.x);
      map_inserted = voxel_map_.insertPoints(points_world);
      map_ns += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(SteadyClock::now() - map_insert_t0).count());
      const auto history_prune_t0 = SteadyClock::now();
      map_pruned = voxel_map_.pruneHistoryBlocks(iekf_state_.x.p_wb);
      map_ns += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(SteadyClock::now() - history_prune_t0).count());
      history_blocks = voxel_map_.blockCount();
      if (!updater_enable_) {
        active_blocks = voxel_map_.activeBlockCount(
          iekf_state_.x.p_wb, lidar_origin_w, lidar_forward_w);
      }
      RCLCPP_DEBUG(
        this->get_logger(),
        "Local map integrated one scan: used_update=%s skip=%s corr=%zu rmse=%.4f pruned=%zu world_points=%zu inserted=%zu map_voxels=%zu history_blocks=%zu active_blocks=%zu",
        used_update ? "true" : "false",
        update_skip_reason.c_str(),
        update_corr,
        update_rmse,
        map_pruned,
        points_world.size(),
        map_inserted,
        voxel_map_.size(),
        history_blocks,
        active_blocks);
    }

    last_history_blocks_.store(history_blocks);
    last_active_blocks_.store(active_blocks);

    // Keep this verbose per-scan log disabled for now to reduce runtime overhead.
    // RCLCPP_INFO(
    //   this->get_logger(),
    //   "Scheduled one scan: [%.3f, %.3f], imu_in_window=%zu, points=%zu->%zu, deskew_enable=%s, downsampled=%zu, used_update=%s, update_skip=%s, update_corr=%zu, update_rmse=%.4f, map_pruned=%zu, world_points=%zu, map_voxels=%zu, pred_p=(%.3f,%.3f,%.3f), pred_v=(%.3f,%.3f,%.3f), pred_rpy_rad=(%.3f,%.3f,%.3f)",
    //   stampToSec(scan_begin_time),
    //   stampToSec(scan_end_time),
    //   imu_track_internal.size(),
    //   raw_point_count,
    //   filtered_point_count,
    //   deskew_enable_ ? "true" : "false",
    //   scan_for_update.points.size(),
    //   used_update ? "true" : "false",
    //   update_skip_reason.c_str(),
    //   update_corr,
    //   update_rmse,
    //   map_pruned,
    //   points_world.size(),
    //   voxel_map_.size(),
    //   iekf_state_.x.p_wb.x(),
    //   iekf_state_.x.p_wb.y(),
    //   iekf_state_.x.p_wb.z(),
    //   iekf_state_.x.v_wb.x(),
    //   iekf_state_.x.v_wb.y(),
    //   iekf_state_.x.v_wb.z());
    const auto publish_t0 = SteadyClock::now();
    publishScanState(scan_end_time);
    publishPointsWorld(points_world, scan_end_time);
    const std::uint64_t publish_ns = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(SteadyClock::now() - publish_t0).count());
    const std::uint64_t total_ns = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(SteadyClock::now() - total_t0).count());

    ++lidar_processed_count_;
    timing_prep_ns_total_.fetch_add(prep_ns);
    timing_predict_ns_total_.fetch_add(predict_ns);
    timing_deskew_ns_total_.fetch_add(deskew_ns);
    timing_downsample_ns_total_.fetch_add(downsample_ns);
    timing_update_ns_total_.fetch_add(update_ns);
    timing_map_ns_total_.fetch_add(map_ns);
    timing_publish_ns_total_.fetch_add(publish_ns);
    timing_total_ns_total_.fetch_add(total_ns);
    return true;
  }

  void publishScanState(const builtin_interfaces::msg::Time & stamp)
  {
    Eigen::Quaterniond q_wb(iekf_state_.x.r_wb);
    if (q_wb.norm() < 1e-12) {
      q_wb = Eigen::Quaterniond::Identity();
    } else {
      q_wb.normalize();
    }

    nav_msgs::msg::Odometry odom_msg;
    odom_msg.header.stamp = stamp;
    odom_msg.header.frame_id = map_frame_id_;
    odom_msg.child_frame_id = base_frame_id_;
    odom_msg.pose.pose.position.x = iekf_state_.x.p_wb.x();
    odom_msg.pose.pose.position.y = iekf_state_.x.p_wb.y();
    odom_msg.pose.pose.position.z = iekf_state_.x.p_wb.z();
    odom_msg.pose.pose.orientation.w = q_wb.w();
    odom_msg.pose.pose.orientation.x = q_wb.x();
    odom_msg.pose.pose.orientation.y = q_wb.y();
    odom_msg.pose.pose.orientation.z = q_wb.z();
    odom_msg.twist.twist.linear.x = iekf_state_.x.v_wb.x();
    odom_msg.twist.twist.linear.y = iekf_state_.x.v_wb.y();
    odom_msg.twist.twist.linear.z = iekf_state_.x.v_wb.z();
    odom_pub_->publish(odom_msg);

    geometry_msgs::msg::PoseStamped pose_stamped;
    pose_stamped.header = odom_msg.header;
    pose_stamped.pose = odom_msg.pose.pose;
    path_msg_.header.stamp = stamp;
    path_msg_.poses.push_back(pose_stamped);
    const std::size_t max_len = static_cast<std::size_t>(std::max(1, path_max_length_));
    if (path_msg_.poses.size() > max_len) {
      const auto erase_count = path_msg_.poses.size() - max_len;
      path_msg_.poses.erase(
        path_msg_.poses.begin(),
        path_msg_.poses.begin() + static_cast<std::ptrdiff_t>(erase_count));
    }
    path_pub_->publish(path_msg_);

    if (publish_tf_ && tf_broadcaster_) {
      geometry_msgs::msg::TransformStamped tf_msg;
      tf_msg.header.stamp = stamp;
      tf_msg.header.frame_id = map_frame_id_;
      tf_msg.child_frame_id = base_frame_id_;
      tf_msg.transform.translation.x = iekf_state_.x.p_wb.x();
      tf_msg.transform.translation.y = iekf_state_.x.p_wb.y();
      tf_msg.transform.translation.z = iekf_state_.x.p_wb.z();
      tf_msg.transform.rotation.w = q_wb.w();
      tf_msg.transform.rotation.x = q_wb.x();
      tf_msg.transform.rotation.y = q_wb.y();
      tf_msg.transform.rotation.z = q_wb.z();
      tf_broadcaster_->sendTransform(tf_msg);
    }
  }

  void publishPointsWorld(
    const std::vector<Eigen::Vector3d> & points_world,
    const builtin_interfaces::msg::Time & stamp)
  {
    pcl::PointCloud<pcl::PointXYZ> cloud;
    cloud.reserve(points_world.size());
    for (const auto & p_w : points_world) {
      if (!std::isfinite(p_w.x()) || !std::isfinite(p_w.y()) || !std::isfinite(p_w.z())) {
        continue;
      }
      pcl::PointXYZ p;
      p.x = static_cast<float>(p_w.x());
      p.y = static_cast<float>(p_w.y());
      p.z = static_cast<float>(p_w.z());
      cloud.push_back(p);
    }
    cloud.width = static_cast<std::uint32_t>(cloud.size());
    cloud.height = 1;
    cloud.is_dense = false;

    sensor_msgs::msg::PointCloud2 cloud_msg;
    pcl::toROSMsg(cloud, cloud_msg);
    cloud_msg.header.stamp = stamp;
    cloud_msg.header.frame_id = map_frame_id_;
    points_world_pub_->publish(cloud_msg);
  }

  std::array<double, 3> rotationMatrixToRpy(const iekf_lio::IekfMat3 & R) const
  {
    const double roll = std::atan2(R(2, 1), R(2, 2));
    const double pitch = std::asin(std::clamp(-R(2, 0), -1.0, 1.0));
    const double yaw = std::atan2(R(1, 0), R(0, 0));
    return {roll, pitch, yaw};
  }

  Eigen::Vector3d currentLidarOriginWorld() const
  {
    return iekf_state_.x.p_wb + iekf_state_.x.r_wb * extrinsic_t_il_;
  }

  Eigen::Vector3d currentLidarForwardWorld() const
  {
    const Eigen::Matrix3d r_wl = iekf_state_.x.r_wb * extrinsic_r_il_;
    return r_wl.col(0);
  }

  std::vector<Eigen::Vector3d> transformDeskewedScanToWorld(
    const iekf_lio::LidarScanXYZ & scan,
    const iekf_lio::IekfNominalState18 & state) const
  {
    if (scan.points == nullptr) {
      return {};
    }

    const std::size_t total_points = scan.points->size();
    if (total_points == 0) {
      return {};
    }

    std::vector<Eigen::Vector3d> transformed(total_points, Eigen::Vector3d::Zero());
    std::vector<std::uint8_t> keep(total_points, 0);

#pragma omp parallel for
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(total_points); ++i) {
      const std::size_t idx = static_cast<std::size_t>(i);
      const auto & pt = scan.points->at(idx);
      if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
        continue;
      }

      const Eigen::Vector3d p_i_end(pt.x, pt.y, pt.z);
      transformed[idx] = state.r_wb * p_i_end + state.p_wb;
      keep[idx] = 1;
    }

    std::size_t kept_count = 0;
    for (const auto flag : keep) {
      kept_count += static_cast<std::size_t>(flag);
    }

    std::vector<Eigen::Vector3d> points_world;
    points_world.reserve(kept_count);
    for (std::size_t i = 0; i < total_points; ++i) {
      if (keep[i] != 0) {
        points_world.push_back(transformed[i]);
      }
    }
    return points_world;
  }

  iekf_lio::LidarScanXYZ downsampleScan(
    const iekf_lio::LidarScanXYZ & scan,
    double leaf_size) const
  {
    if (scan.points == nullptr || scan.points->empty() || leaf_size <= 1e-6) {
      return scan;
    }

    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(scan.points);
    vg.setLeafSize(static_cast<float>(leaf_size),
      static_cast<float>(leaf_size),
      static_cast<float>(leaf_size));
    auto filtered = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    vg.filter(*filtered);

    iekf_lio::LidarScanXYZ out;
    out.frame_id = scan.frame_id;
    out.scan_begin_time_s = scan.scan_begin_time_s;
    out.scan_end_time_s = scan.scan_end_time_s;
    out.timebase_ns = scan.timebase_ns;
    out.points = filtered;
    return out;
  }

  iekf_lio::LidarScanXYZ convertLidarScanToXYZ(const iekf_lio::LidarScan & scan) const
  {
    iekf_lio::LidarScanXYZ out;
    out.frame_id = scan.frame_id;
    out.scan_begin_time_s = scan.scan_begin_time_s;
    out.scan_end_time_s = scan.scan_end_time_s;
    out.timebase_ns = scan.timebase_ns;
    out.points->reserve(scan.points.size());
    for (const auto & pt : scan.points) {
      if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
        continue;
      }
      pcl::PointXYZ out_pt;
      out_pt.x = pt.x;
      out_pt.y = pt.y;
      out_pt.z = pt.z;
      out.points->push_back(out_pt);
    }
    out.points->width = static_cast<std::uint32_t>(out.points->size());
    out.points->height = 1;
    out.points->is_dense = false;
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

  LidarPreprocessResult preprocessLidarScanToInternal(
    const LidarMsg & lidar_msg,
    const builtin_interfaces::msg::Time & scan_begin_time,
    const builtin_interfaces::msg::Time & scan_end_time) const
  {
    LidarPreprocessResult result;
    result.raw_point_count = lidar_msg.points.size();
    result.scan.frame_id = lidar_msg.header.frame_id;
    result.scan.scan_begin_time_s = stampToSec(scan_begin_time);
    result.scan.scan_end_time_s = stampToSec(scan_end_time);
    result.scan.timebase_ns = lidar_msg.timebase;

    if (!reflectivity_filter_enable_ && !distance_filter_enable_) {
      result.filtered_point_count = lidar_msg.points.size();
      result.scan.points.reserve(lidar_msg.points.size());
      for (const auto & pt : lidar_msg.points) {
        iekf_lio::PointXYZIRTL p;
        p.x = pt.x;
        p.y = pt.y;
        p.z = pt.z;
        p.reflectivity = pt.reflectivity;
        p.tag = pt.tag;
        p.line = pt.line;
        p.relative_time_s = static_cast<double>(pt.offset_time) * 1e-9;
        result.scan.points.push_back(p);
      }
      return result;
    }

    const std::uint8_t min_refl = static_cast<std::uint8_t>(std::max(0, reflectivity_min_));
    const std::uint8_t max_refl = static_cast<std::uint8_t>(std::min(255, reflectivity_max_));
    const std::uint8_t lower = std::min(min_refl, max_refl);
    const std::uint8_t upper = std::max(min_refl, max_refl);
    const double d_min = std::max(0.0, distance_min_m_);
    const double d_max = std::max(d_min, distance_max_m_);
    const double d_min2 = d_min * d_min;
    const double d_max2 = d_max * d_max;
    const std::size_t total_points = lidar_msg.points.size();

    // Parallel stage 1: mark kept points.
    std::vector<std::uint8_t> keep(total_points, 0);
#pragma omp parallel for
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(total_points); ++i) {
      const std::size_t idx = static_cast<std::size_t>(i);
      const auto & pt = lidar_msg.points[idx];
      const bool pass_reflectivity = !reflectivity_filter_enable_ ||
        (pt.reflectivity >= lower && pt.reflectivity <= upper);
      const double r2 = static_cast<double>(pt.x) * static_cast<double>(pt.x) +
        static_cast<double>(pt.y) * static_cast<double>(pt.y) +
        static_cast<double>(pt.z) * static_cast<double>(pt.z);
      const bool pass_distance = !distance_filter_enable_ || (r2 >= d_min2 && r2 <= d_max2);
      if (pass_reflectivity && pass_distance) {
        keep[idx] = 1;
      }
    }

    // Prefix sum by original index gives deterministic write positions.
    std::vector<std::uint32_t> write_pos(total_points, 0);
    std::size_t kept_count = 0;
    for (std::size_t i = 0; i < total_points; ++i) {
      write_pos[i] = static_cast<std::uint32_t>(kept_count);
      kept_count += static_cast<std::size_t>(keep[i]);
    }

    result.filtered_point_count = kept_count;
    result.scan.points.resize(kept_count);

    // Parallel stage 2: convert and write kept points to internal scan in stable order.
#pragma omp parallel for
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(total_points); ++i) {
      const std::size_t idx = static_cast<std::size_t>(i);
      if (keep[idx] == 0) {
        continue;
      }
      const auto & pt = lidar_msg.points[idx];
      iekf_lio::PointXYZIRTL p;
      p.x = pt.x;
      p.y = pt.y;
      p.z = pt.z;
      p.reflectivity = pt.reflectivity;
      p.tag = pt.tag;
      p.line = pt.line;
      p.relative_time_s = static_cast<double>(pt.offset_time) * 1e-9;
      result.scan.points[write_pos[idx]] = p;
    }

    return result;
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

    const std::uint64_t lidar_rx = lidar_received_count_.load();
    const std::uint64_t lidar_processed = lidar_processed_count_.load();
    const std::uint64_t lidar_drop_overflow = lidar_drop_overflow_count_.load();
    const std::uint64_t lidar_drop_stale = lidar_drop_stale_count_.load();
    const std::uint64_t lidar_drop_refl = lidar_drop_reflectivity_count_.load();
    const std::uint64_t timing_prep_ns = timing_prep_ns_total_.load();
    const std::uint64_t timing_predict_ns = timing_predict_ns_total_.load();
    const std::uint64_t timing_deskew_ns = timing_deskew_ns_total_.load();
    const std::uint64_t timing_downsample_ns = timing_downsample_ns_total_.load();
    const std::uint64_t timing_update_ns = timing_update_ns_total_.load();
    const std::uint64_t timing_map_ns = timing_map_ns_total_.load();
    const std::uint64_t timing_publish_ns = timing_publish_ns_total_.load();
    const std::uint64_t timing_total_ns = timing_total_ns_total_.load();
    const std::uint64_t delta_rx = lidar_rx - last_report_lidar_rx_;
    const std::uint64_t delta_processed = lidar_processed - last_report_lidar_processed_;
    const std::uint64_t delta_drop_overflow =
      lidar_drop_overflow - last_report_lidar_drop_overflow_;
    const std::uint64_t delta_prep_ns = timing_prep_ns - last_report_timing_prep_ns_;
    const std::uint64_t delta_predict_ns = timing_predict_ns - last_report_timing_predict_ns_;
    const std::uint64_t delta_deskew_ns = timing_deskew_ns - last_report_timing_deskew_ns_;
    const std::uint64_t delta_downsample_ns =
      timing_downsample_ns - last_report_timing_downsample_ns_;
    const std::uint64_t delta_update_ns = timing_update_ns - last_report_timing_update_ns_;
    const std::uint64_t delta_map_ns = timing_map_ns - last_report_timing_map_ns_;
    const std::uint64_t delta_publish_ns = timing_publish_ns - last_report_timing_publish_ns_;
    const std::uint64_t delta_total_ns = timing_total_ns - last_report_timing_total_ns_;
    const std::size_t history_blocks = last_history_blocks_.load();
    const std::size_t active_blocks = last_active_blocks_.load();
    const double overflow_drop_rate_window = (delta_rx > 0)
      ? (100.0 * static_cast<double>(delta_drop_overflow) / static_cast<double>(delta_rx))
      : 0.0;
    const double overflow_drop_rate_total = (lidar_rx > 0)
      ? (100.0 * static_cast<double>(lidar_drop_overflow) / static_cast<double>(lidar_rx))
      : 0.0;

    const auto avg_ms = [delta_processed](std::uint64_t delta_ns) -> double {
        if (delta_processed == 0) {
          return 0.0;
        }
        return static_cast<double>(delta_ns) / static_cast<double>(delta_processed) * 1e-6;
      };
    const double avg_prep_ms = avg_ms(delta_prep_ns);
    const double avg_predict_ms = avg_ms(delta_predict_ns);
    const double avg_deskew_ms = avg_ms(delta_deskew_ns);
    const double avg_downsample_ms = avg_ms(delta_downsample_ns);
    const double avg_update_ms = avg_ms(delta_update_ns);
    const double avg_map_ms = avg_ms(delta_map_ns);
    const double avg_publish_ms = avg_ms(delta_publish_ns);
    const double avg_total_ms = avg_ms(delta_total_ns);

    last_report_lidar_rx_ = lidar_rx;
    last_report_lidar_processed_ = lidar_processed;
    last_report_lidar_drop_overflow_ = lidar_drop_overflow;
    last_report_timing_prep_ns_ = timing_prep_ns;
    last_report_timing_predict_ns_ = timing_predict_ns;
    last_report_timing_deskew_ns_ = timing_deskew_ns;
    last_report_timing_downsample_ns_ = timing_downsample_ns;
    last_report_timing_update_ns_ = timing_update_ns;
    last_report_timing_map_ns_ = timing_map_ns;
    last_report_timing_publish_ns_ = timing_publish_ns;
    last_report_timing_total_ns_ = timing_total_ns;

    RCLCPP_INFO(
      this->get_logger(),
      "input buffers: imu=%zu lidar=%zu latest_imu=%.3f latest_lidar=%.3f | map: history_blocks=%zu active_blocks=%zu | lidar_rx=%llu proc=%llu drop_overflow=%llu(win=%.2f%% total=%.2f%%) drop_stale=%llu drop_reflectivity=%llu | stage_ms_per_scan(win): prep=%.2f predict=%.2f deskew=%.2f downsample=%.2f update=%.2f map=%.2f publish=%.2f total=%.2f",
      imu_size,
      lidar_size,
      stampToSec(latest_imu_stamp_),
      stampToSec(latest_lidar_stamp_),
      history_blocks,
      active_blocks,
      static_cast<unsigned long long>(lidar_rx),
      static_cast<unsigned long long>(lidar_processed),
      static_cast<unsigned long long>(lidar_drop_overflow),
      overflow_drop_rate_window,
      overflow_drop_rate_total,
      static_cast<unsigned long long>(lidar_drop_stale),
      static_cast<unsigned long long>(lidar_drop_refl),
      avg_prep_ms,
      avg_predict_ms,
      avg_deskew_ms,
      avg_downsample_ms,
      avg_update_ms,
      avg_map_ms,
      avg_publish_ms,
      avg_total_ms);
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
  bool distance_filter_enable_;
  double distance_min_m_;
  double distance_max_m_;
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
  double downsample_update_leaf_size_;
  double downsample_map_leaf_size_;
  double map_voxel_size_;
  double map_block_size_;
  int map_history_max_blocks_;
  bool map_history_enable_;
  double map_history_radius_xy_;
  double map_history_half_height_;
  bool map_active_window_enable_;
  double map_active_radius_xy_;
  double map_active_half_height_;
  bool map_active_angle_enable_;
  double map_active_half_fov_deg_;
  std::string odom_topic_;
  std::string path_topic_;
  std::string points_world_topic_;
  std::string map_frame_id_;
  std::string base_frame_id_;
  int path_max_length_;
  bool publish_tf_;
  Eigen::Vector3d extrinsic_t_il_ = Eigen::Vector3d::Zero();
  Eigen::Matrix3d extrinsic_r_il_ = Eigen::Matrix3d::Identity();

  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<LidarMsg>::SharedPtr lidar_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr points_world_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::TimerBase::SharedPtr status_timer_;
  nav_msgs::msg::Path path_msg_;

  std::deque<sensor_msgs::msg::Imu::SharedPtr> imu_buffer_;
  std::deque<LidarMsg::SharedPtr> lidar_buffer_;
  std::mutex imu_mutex_;
  std::mutex lidar_mutex_;
  std::condition_variable data_cv_;
  std::mutex processing_mutex_;
  std::thread processing_thread_;
  bool stop_requested_ = false;
  std::atomic<std::uint64_t> lidar_received_count_ {0};
  std::atomic<std::uint64_t> lidar_processed_count_ {0};
  std::atomic<std::uint64_t> lidar_drop_overflow_count_ {0};
  std::atomic<std::uint64_t> lidar_drop_stale_count_ {0};
  std::atomic<std::uint64_t> lidar_drop_reflectivity_count_ {0};
  std::atomic<std::uint64_t> timing_prep_ns_total_ {0};
  std::atomic<std::uint64_t> timing_predict_ns_total_ {0};
  std::atomic<std::uint64_t> timing_deskew_ns_total_ {0};
  std::atomic<std::uint64_t> timing_downsample_ns_total_ {0};
  std::atomic<std::uint64_t> timing_update_ns_total_ {0};
  std::atomic<std::uint64_t> timing_map_ns_total_ {0};
  std::atomic<std::uint64_t> timing_publish_ns_total_ {0};
  std::atomic<std::uint64_t> timing_total_ns_total_ {0};
  std::atomic<std::size_t> last_history_blocks_ {0};
  std::atomic<std::size_t> last_active_blocks_ {0};
  std::uint64_t last_report_lidar_rx_ = 0;
  std::uint64_t last_report_lidar_processed_ = 0;
  std::uint64_t last_report_lidar_drop_overflow_ = 0;
  std::uint64_t last_report_timing_prep_ns_ = 0;
  std::uint64_t last_report_timing_predict_ns_ = 0;
  std::uint64_t last_report_timing_deskew_ns_ = 0;
  std::uint64_t last_report_timing_downsample_ns_ = 0;
  std::uint64_t last_report_timing_update_ns_ = 0;
  std::uint64_t last_report_timing_map_ns_ = 0;
  std::uint64_t last_report_timing_publish_ns_ = 0;
  std::uint64_t last_report_timing_total_ns_ = 0;
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
