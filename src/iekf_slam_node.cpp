#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <ctime>
#include <cstdint>
#include <cstddef>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
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
#include "iekf_lio/msg/imu_prediction_state.hpp"
#include "imu/imu_integrator.hpp"
#include "imu/noise_model.hpp"
#include "imu/imu_types.hpp"
#include "lidar/cloud_deskewer.hpp"
#include "mapping/backend_factor_graph.hpp"
#include "mapping/keyframe_manager.hpp"
#include "mapping/loop_registration.hpp"
#include "mapping/scan_context_manager.hpp"
#include "mapping/voxel_map.hpp"
#include "pdr/pdr_types.hpp"
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
    pdr_buffer_size_ = this->declare_parameter<int>("pdr.buffer_size", 12000);
    pdr_sample_rate_hz_ = this->declare_parameter<double>("pdr.sample_rate_hz", 100.0);
    pdr_accel_norm_lpf_cutoff_hz_ = this->declare_parameter<double>(
      "pdr.accel_norm_lpf_cutoff_hz", 5.0);
    pdr_output_dir_ = this->declare_parameter<std::string>(
      "pdr.output_dir", "/home/yyj/IEKF_LIO_node/src/iekf_lio/pdroutput");
    time_output_dir_ = this->declare_parameter<std::string>(
      "time.output_dir", "/home/yyj/IEKF_LIO_node/src/iekf_lio/timeoutput");
    pdr_peak_enable_ = this->declare_parameter<bool>("pdr.peak.enable", true);
    pdr_peak_buffer_size_ = this->declare_parameter<int>("pdr.peak.buffer_size", 2000);
    pdr_peak_min_height_ = this->declare_parameter<double>(
      "pdr.peak.min_height", 0.2);
    pdr_peak_min_interval_s_ = this->declare_parameter<double>(
      "pdr.peak.min_interval_s", 0.25);
    pdr_peak_nms_interval_s_ = this->declare_parameter<double>(
      "pdr.peak.nms_interval_s", 0.15);
    pdr_ampd_enable_ = this->declare_parameter<bool>("pdr.ampd.enable", true);
    pdr_ampd_smin_ = this->declare_parameter<int>("pdr.ampd.smin", 5);
    pdr_ampd_smax_ = this->declare_parameter<int>("pdr.ampd.smax", 30);
    pdr_ampd_pass_ratio_ = this->declare_parameter<double>(
      "pdr.ampd.pass_ratio", 0.5);
    pdr_step_buffer_size_ = this->declare_parameter<int>("pdr.step.buffer_size", 2000);
    pdr_step_min_interval_s_ = this->declare_parameter<double>(
      "pdr.step.min_interval_s", 0.3);
    pdr_step_max_interval_s_ = this->declare_parameter<double>(
      "pdr.step.max_interval_s", 1.0);
    pdr_step_min_peak_valley_diff_ = this->declare_parameter<double>(
      "pdr.step.min_peak_valley_diff", 0.15);
    // ZUPT is temporarily disabled in code while we stabilize the front-end path.
    // Keep the config block for future tuning, but do not bind it into runtime logic.
    updater_enable_ = this->declare_parameter<bool>("updater.enable", true);
    updater_max_iterations_ = this->declare_parameter<int>("updater.max_iterations", 2);
    updater_max_corr_dist_ = this->declare_parameter<double>("updater.max_correspondence_distance", 1.0);
    updater_plane_k_ = this->declare_parameter<int>("updater.plane_k_neighbors", 10);
    updater_plane_eigen_ratio_ = this->declare_parameter<double>("updater.plane_max_eigen_ratio", 0.1);
    updater_sigma_plane_ = this->declare_parameter<double>("updater.sigma_point_to_plane", 0.2);
    updater_max_abs_residual_ = this->declare_parameter<double>("updater.max_abs_point_to_plane_residual", 0.5);
    updater_max_update_points_ = this->declare_parameter<int>("updater.max_update_points", 1200);
    updater_min_correspondences_ = this->declare_parameter<int>("updater.min_correspondences", 0);
    updater_convergence_delta_pos_m_ = this->declare_parameter<double>(
      "updater.convergence_delta_pos_m", 1e-3);
    updater_convergence_delta_rot_rad_ = this->declare_parameter<double>(
      "updater.convergence_delta_rot_rad", 1e-4);
    updater_degeneracy_enable_ = this->declare_parameter<bool>(
      "updater.degeneracy_projection.enable", true);
    updater_degeneracy_log_only_ = this->declare_parameter<bool>(
      "updater.degeneracy_projection.log_only", false);
    updater_degeneracy_trigger_ratio_ = this->declare_parameter<double>(
      "updater.degeneracy_projection.trigger_ratio", 0.02);
    updater_degeneracy_min_eigenvalue_threshold_ = this->declare_parameter<double>(
      "updater.degeneracy_projection.min_eigenvalue_threshold", 1e-3);
    updater_degeneracy_poor_match_min_correspondences_ = this->declare_parameter<int>(
      "updater.degeneracy_projection.poor_match_min_correspondences", 80);
    updater_degeneracy_poor_match_rmse_threshold_ = this->declare_parameter<double>(
      "updater.degeneracy_projection.poor_match_rmse_threshold", 0.25);
    updater_degeneracy_relative_scale_ = this->declare_parameter<double>(
      "updater.degeneracy_projection.relative_scale", 0.05);
    updater_degeneracy_abs_floor_ = this->declare_parameter<double>(
      "updater.degeneracy_projection.absolute_floor", 1e-6);
    updater_degeneracy_min_weight_ = this->declare_parameter<double>(
      "updater.degeneracy_projection.min_weight", 0.0);
    deskew_enable_ = this->declare_parameter<bool>("deskew.enable", true);
    deskew_max_input_points_ = this->declare_parameter<int>("deskew.max_input_points", 0);
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
    keyframe_enable_ = this->declare_parameter<bool>("keyframe.enable", true);
    keyframe_translation_thresh_m_ = this->declare_parameter<double>(
      "keyframe.translation_thresh_m", 0.5);
    keyframe_rotation_thresh_deg_ = this->declare_parameter<double>(
      "keyframe.rotation_thresh_deg", 10.0);
    keyframe_time_thresh_s_ = this->declare_parameter<double>(
      "keyframe.time_thresh_s", 0.0);
    keyframe_backend_max_cached_ = this->declare_parameter<int>(
      "keyframe.backend.max_cached", 200);
    keyframe_backend_max_archive_cached_ = this->declare_parameter<int>(
      "keyframe.backend.max_archive_cached", 2000);
    local_map_active_recent_count_ = this->declare_parameter<int>(
      "local_map.active.recent_count", 30);
    local_map_active_radius_m_ = this->declare_parameter<double>(
      "local_map.active.radius_m", 30.0);
    local_map_active_max_keyframes_ = this->declare_parameter<int>(
      "local_map.active.max_keyframes", 30);
    loop_enable_ = this->declare_parameter<bool>("loop.enable", true);
    loop_min_keyframes_before_loop_ = this->declare_parameter<int>(
      "loop.min_keyframes_before_loop", 30);
    loop_exclude_recent_keyframes_ = this->declare_parameter<int>(
      "loop.exclude_recent_keyframes", 30);
    loop_sc_num_rings_ = this->declare_parameter<int>("loop.scan_context.num_rings", 20);
    loop_sc_num_sectors_ = this->declare_parameter<int>("loop.scan_context.num_sectors", 60);
    loop_sc_max_radius_m_ = this->declare_parameter<double>("loop.scan_context.max_radius", 80.0);
    loop_sc_max_entries_ = this->declare_parameter<int>("loop.scan_context.max_entries", 2000);
    loop_sc_num_candidates_ = this->declare_parameter<int>("loop.scan_context.num_candidates", 10);
    loop_sc_distance_threshold_ = this->declare_parameter<double>(
      "loop.scan_context.distance_threshold", 0.15);
    loop_reg_enable_ = this->declare_parameter<bool>("loop.registration.enable", true);
    loop_reg_voxel_leaf_size_ = this->declare_parameter<double>(
      "loop.registration.voxel_leaf_size", 0.5);
    loop_reg_max_corr_distance_ = this->declare_parameter<double>(
      "loop.registration.max_corr_distance", 2.0);
    loop_reg_max_iterations_ = this->declare_parameter<int>(
      "loop.registration.max_iterations", 40);
    loop_reg_fitness_threshold_ = this->declare_parameter<double>(
      "loop.registration.fitness_threshold", 0.5);
    loop_factor_translation_sigma_ = this->declare_parameter<double>(
      "loop.factor.translation_sigma", 0.5);
    loop_factor_rotation_sigma_rad_ = this->declare_parameter<double>(
      "loop.factor.rotation_sigma_rad", 0.2);
    backend_optimize_enable_ = this->declare_parameter<bool>(
      "backend.optimize.enable", false);
    odom_topic_ = this->declare_parameter<std::string>("publish.odom_topic", "/iekf/odom");
    path_topic_ = this->declare_parameter<std::string>("publish.path_topic", "/iekf/path");
    backend_path_topic_ = this->declare_parameter<std::string>(
      "publish.backend_path_topic", "/iekf/backend_path");
    points_world_topic_ = this->declare_parameter<std::string>(
      "publish.points_world_topic", "/iekf/points_world");
    imu_prediction_topic_ = this->declare_parameter<std::string>(
      "publish.imu_prediction_topic", "/iekf/imu_prediction_state");
    imu_prediction_qos_depth_ = this->declare_parameter<int>(
      "publish.imu_prediction_qos_depth", 256);
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
    iekf_lio::IekfPredictorPdrConfig predictor_pdr_cfg;
    predictor_pdr_cfg.accel_norm_lpf_cutoff_hz = std::max(0.0, pdr_accel_norm_lpf_cutoff_hz_);
    iekf_predictor_.setPdrConfig(predictor_pdr_cfg);
    iekf_lio::IekfUpdaterConfig updater_cfg;
    updater_cfg.max_iterations = std::max(1, updater_max_iterations_);
    updater_cfg.max_correspondence_distance = std::max(0.05, updater_max_corr_dist_);
    updater_cfg.plane_k_neighbors = std::max(3, updater_plane_k_);
    updater_cfg.plane_max_eigen_ratio = std::max(1e-4, updater_plane_eigen_ratio_);
    updater_cfg.sigma_point_to_plane = std::max(1e-3, updater_sigma_plane_);
    updater_cfg.max_abs_point_to_plane_residual = std::max(1e-3, updater_max_abs_residual_);
    updater_cfg.max_update_points = std::max(100, updater_max_update_points_);
    updater_cfg.min_correspondences = std::max(0, updater_min_correspondences_);
    updater_cfg.convergence_delta_pos_m = std::max(0.0, updater_convergence_delta_pos_m_);
    updater_cfg.convergence_delta_rot_rad = std::max(0.0, updater_convergence_delta_rot_rad_);
    updater_cfg.degeneracy_projection_enable = updater_degeneracy_enable_;
    updater_cfg.degeneracy_projection_log_only = updater_degeneracy_log_only_;
    updater_cfg.degeneracy_trigger_ratio = std::max(0.0, updater_degeneracy_trigger_ratio_);
    updater_cfg.degeneracy_min_eigenvalue_threshold = std::max(
      0.0, updater_degeneracy_min_eigenvalue_threshold_);
    updater_cfg.degeneracy_poor_match_min_correspondences = std::max(
      1, updater_degeneracy_poor_match_min_correspondences_);
    updater_cfg.degeneracy_poor_match_rmse_threshold = std::max(
      0.0, updater_degeneracy_poor_match_rmse_threshold_);
    updater_cfg.degeneracy_relative_scale = std::max(1e-6, updater_degeneracy_relative_scale_);
    updater_cfg.degeneracy_abs_floor = std::max(1e-12, updater_degeneracy_abs_floor_);
    updater_cfg.degeneracy_min_weight = std::clamp(updater_degeneracy_min_weight_, 0.0, 1.0);
    iekf_updater_.setConfig(updater_cfg);
    voxel_map_.setVoxelSize(map_voxel_size_);
    voxel_map_.setBlockSize(map_block_size_);
    voxel_map_.setMaxBlocks(static_cast<std::size_t>(std::max(1, map_history_max_blocks_)));
    voxel_map_.setHistoryWindowEnabled(map_history_enable_);
    voxel_map_.setHistoryWindow(map_history_radius_xy_, map_history_half_height_);
    iekf_lio::KeyframeManagerConfig keyframe_cfg;
    keyframe_cfg.enable = keyframe_enable_;
    keyframe_cfg.translation_thresh_m = std::max(0.0, keyframe_translation_thresh_m_);
    keyframe_cfg.rotation_thresh_rad = std::max(0.0, keyframe_rotation_thresh_deg_) * M_PI / 180.0;
    keyframe_cfg.time_thresh_s = std::max(0.0, keyframe_time_thresh_s_);
    keyframe_manager_.setConfig(keyframe_cfg);
    iekf_lio::ScanContextManagerConfig scan_context_cfg;
    scan_context_cfg.enable = loop_enable_;
    scan_context_cfg.min_keyframes_before_loop =
      static_cast<std::size_t>(std::max(1, loop_min_keyframes_before_loop_));
    scan_context_cfg.exclude_recent_keyframes =
      static_cast<std::size_t>(std::max(0, loop_exclude_recent_keyframes_));
    scan_context_cfg.num_rings = std::max(1, loop_sc_num_rings_);
    scan_context_cfg.num_sectors = std::max(1, loop_sc_num_sectors_);
    scan_context_cfg.max_radius_m = std::max(1.0, loop_sc_max_radius_m_);
    scan_context_cfg.max_entries = static_cast<std::size_t>(std::max(1, loop_sc_max_entries_));
    scan_context_cfg.num_candidates = static_cast<std::size_t>(std::max(1, loop_sc_num_candidates_));
    scan_context_cfg.distance_threshold = std::max(0.0, loop_sc_distance_threshold_);
    scan_context_manager_.setConfig(scan_context_cfg);
    iekf_lio::LoopRegistrationConfig loop_reg_cfg;
    loop_reg_cfg.enable = loop_reg_enable_;
    loop_reg_cfg.voxel_leaf_size = std::max(1e-3, loop_reg_voxel_leaf_size_);
    loop_reg_cfg.max_corr_distance = std::max(1e-3, loop_reg_max_corr_distance_);
    loop_reg_cfg.max_iterations = std::max(1, loop_reg_max_iterations_);
    loop_reg_cfg.fitness_threshold = std::max(0.0, loop_reg_fitness_threshold_);
    loop_reg_cfg.translation_sigma = std::max(1e-3, loop_factor_translation_sigma_);
    loop_reg_cfg.rotation_sigma_rad = std::max(1e-3, loop_factor_rotation_sigma_rad_);
    loop_registration_.setConfig(loop_reg_cfg);
    initializePdrOutputs();
    initializeTimingOutputs();

    const auto sensor_qos = rclcpp::SensorDataQoS();
    const auto imu_prediction_qos = rclcpp::QoS(
      rclcpp::KeepLast(static_cast<std::size_t>(std::max(1, imu_prediction_qos_depth_))))
      .best_effort()
      .durability_volatile();

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
    backend_path_pub_ = this->create_publisher<nav_msgs::msg::Path>(backend_path_topic_, 10);
    points_world_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(points_world_topic_, 10);
    imu_prediction_pub_ =
      this->create_publisher<iekf_lio::msg::ImuPredictionState>(
      imu_prediction_topic_, imu_prediction_qos);
    if (publish_tf_) {
      tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    }
    path_msg_.header.frame_id = map_frame_id_;
    backend_path_msg_.header.frame_id = map_frame_id_;

    const std::filesystem::path output_dir = std::filesystem::current_path() / "output";
    std::error_code fs_ec;
    std::filesystem::create_directories(output_dir, fs_ec);
    const auto now = std::chrono::system_clock::now();
    const std::time_t now_time_t = std::chrono::system_clock::to_time_t(now);
    std::tm now_tm {};
#if defined(_WIN32)
    localtime_s(&now_tm, &now_time_t);
#else
    localtime_r(&now_time_t, &now_tm);
#endif
    std::ostringstream tum_name_ss;
    tum_name_ss << "trajectory_tum_"
                << std::put_time(&now_tm, "%Y%m%d_%H%M%S")
                << ".txt";
    tum_output_path_ = (output_dir / tum_name_ss.str()).string();
    tum_traj_ofs_.open(tum_output_path_, std::ios::out | std::ios::trunc);
    if (!tum_traj_ofs_.is_open()) {
      RCLCPP_WARN(
        this->get_logger(),
        "Failed to open TUM trajectory output file: %s",
        tum_output_path_.c_str());
    } else {
      tum_traj_ofs_ << std::fixed << std::setprecision(9);
      RCLCPP_INFO(
        this->get_logger(),
        "TUM trajectory output: %s",
        tum_output_path_.c_str());
    }

    // Algorithms should not run in subscription callbacks:
    // heavy compute here blocks executor threads, increases callback latency,
    // and makes IMU/LiDAR scheduling nondeterministic under load.
    processing_thread_ = std::thread(&IekfSlamNode::processingLoop, this);
    backend_thread_ = std::thread(&IekfSlamNode::backendLoop, this);

    // Temporarily disable the wall timer status report to reduce runtime noise.
    // status_timer_ = this->create_wall_timer(
    //   2s, std::bind(&IekfSlamNode::reportInputStatus, this));

    RCLCPP_INFO(
      this->get_logger(),
      "Backend optimizer branch: %s",
      backend_factor_graph_.isGtsamActive() ? "GTSAM" : "fallback");
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
    RCLCPP_INFO(
      this->get_logger(),
      "Publish IMU prediction state: %s",
      imu_prediction_topic_.c_str());
  }

  ~IekfSlamNode() override
  {
    stop_requested_.store(true);
    data_cv_.notify_all();
    keyframe_cv_.notify_all();

    if (processing_thread_.joinable()) {
      processing_thread_.join();
    }
    if (backend_thread_.joinable()) {
      backend_thread_.join();
    }
    if (tum_traj_ofs_.is_open()) {
      tum_traj_ofs_.close();
    }
  }

 private:
  struct LidarPreprocessResult
  {
    iekf_lio::LidarScan scan;
    std::size_t raw_point_count = 0;
    std::size_t filtered_point_count = 0;
  };

  struct KeyframeData
  {
    std::size_t id = 0;
    double time_s = 0.0;
    Eigen::Vector3d p_wb = Eigen::Vector3d::Zero();
    Eigen::Matrix3d r_wb = Eigen::Matrix3d::Identity();
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_i =
      std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  };

  KeyframeData makeKeyframeData(
    std::size_t id,
    double time_s,
    const Eigen::Vector3d & p_wb,
    const Eigen::Matrix3d & r_wb,
    const iekf_lio::LidarScanXYZ & scan_for_map) const
  {
    KeyframeData keyframe;
    keyframe.id = id;
    keyframe.time_s = time_s;
    keyframe.p_wb = p_wb;
    keyframe.r_wb = r_wb;
    if (scan_for_map.points != nullptr) {
      keyframe.cloud_i = scan_for_map.points;
    }
    return keyframe;
  }

  void appendActiveKeyframe(const KeyframeData & keyframe)
  {
    frontend_keyframe_history_.push_back(keyframe);
  }

  void cacheKeyframeVoxelContributions(const KeyframeData & keyframe)
  {
    if (keyframe_voxel_cache_.find(keyframe.id) != keyframe_voxel_cache_.end()) {
      return;
    }

    if (keyframe.cloud_i == nullptr || keyframe.cloud_i->empty()) {
      keyframe_voxel_cache_.emplace(
        keyframe.id, std::vector<iekf_lio::VoxelMap::VoxelContribution> {});
      return;
    }

    keyframe_voxel_cache_[keyframe.id] = voxel_map_.buildTransformedScanContributions(
      *keyframe.cloud_i, keyframe.r_wb, keyframe.p_wb);
  }

  std::deque<KeyframeData> selectActiveKeyframes(
    const Eigen::Vector3d & center_w) const
  {
    std::deque<KeyframeData> selected;
    if (frontend_keyframe_history_.empty()) {
      return selected;
    }

    const std::size_t max_active =
      static_cast<std::size_t>(std::max(1, local_map_active_max_keyframes_));
    const std::size_t recent_count =
      static_cast<std::size_t>(std::max(0, local_map_active_recent_count_));
    const double radius_m = std::max(0.0, local_map_active_radius_m_);
    const double radius2 = radius_m * radius_m;

    std::unordered_set<std::size_t> selected_ids;
    selected_ids.reserve(max_active * 2);

    const std::size_t take_recent = std::min(recent_count, frontend_keyframe_history_.size());
    const auto recent_begin =
      frontend_keyframe_history_.end() - static_cast<std::ptrdiff_t>(take_recent);
    for (auto it = recent_begin; it != frontend_keyframe_history_.end(); ++it) {
      if (selected.size() >= max_active) {
        return selected;
      }
      selected.push_back(*it);
      selected_ids.insert(it->id);
    }

    struct SpatialCandidate
    {
      double dist2 = 0.0;
      const KeyframeData * keyframe = nullptr;
    };

    std::vector<SpatialCandidate> spatial_candidates;
    spatial_candidates.reserve(frontend_keyframe_history_.size());
    for (const auto & keyframe : frontend_keyframe_history_) {
      if (selected_ids.count(keyframe.id) != 0U) {
        continue;
      }
      const double dist2 = (keyframe.p_wb - center_w).squaredNorm();
      if (dist2 <= radius2) {
        spatial_candidates.push_back(SpatialCandidate {dist2, &keyframe});
      }
    }

    std::sort(
      spatial_candidates.begin(),
      spatial_candidates.end(),
      [](const SpatialCandidate & a, const SpatialCandidate & b) {
        if (a.dist2 != b.dist2) {
          return a.dist2 < b.dist2;
        }
        return a.keyframe->time_s < b.keyframe->time_s;
      });

    for (const auto & candidate : spatial_candidates) {
      if (selected.size() >= max_active) {
        break;
      }
      selected.push_back(*candidate.keyframe);
      selected_ids.insert(candidate.keyframe->id);
    }

    return selected;
  }

  bool refreshActiveKeyframes(const Eigen::Vector3d & center_w)
  {
    const std::deque<KeyframeData> selected = selectActiveKeyframes(center_w);
    if (selected.size() != local_map_active_keyframes_.size()) {
      local_map_active_keyframes_ = selected;
      return true;
    }

    for (std::size_t i = 0; i < selected.size(); ++i) {
      if (selected[i].id != local_map_active_keyframes_[i].id) {
        local_map_active_keyframes_ = selected;
        return true;
      }
    }
    return false;
  }

  void updateLocalMapByActiveKeyframeDiff(const Eigen::Vector3d & center_w)
  {
    const std::deque<KeyframeData> selected = selectActiveKeyframes(center_w);
    if (selected.empty()) {
      voxel_map_.clear();
      local_map_active_keyframes_.clear();
      local_map_active_keyframe_ids_.clear();
      local_map_initialized_ = false;
      return;
    }

    std::unordered_set<std::size_t> new_active_ids;
    new_active_ids.reserve(selected.size() * 2);
    for (const auto & keyframe : selected) {
      new_active_ids.insert(keyframe.id);
    }

    for (const auto & active_id : local_map_active_keyframe_ids_) {
      if (new_active_ids.count(active_id) != 0U) {
        continue;
      }
      const auto cache_it = keyframe_voxel_cache_.find(active_id);
      if (cache_it != keyframe_voxel_cache_.end()) {
        voxel_map_.removeVoxelContributions(cache_it->second);
      }
    }

    for (const auto & keyframe : selected) {
      if (local_map_active_keyframe_ids_.count(keyframe.id) != 0U) {
        continue;
      }
      cacheKeyframeVoxelContributions(keyframe);
      const auto cache_it = keyframe_voxel_cache_.find(keyframe.id);
      if (cache_it != keyframe_voxel_cache_.end()) {
        voxel_map_.addVoxelContributions(cache_it->second);
      }
    }

    local_map_active_keyframes_ = selected;
    local_map_active_keyframe_ids_ = std::move(new_active_ids);
    local_map_initialized_ = voxel_map_.size() > 0;
  }

#if 0
  // ZUPT implementation is temporarily commented out while we stabilize the
  // front-end estimation chain. Keep this block for future re-enable/debug.
  struct ZuptDetectionResult
  {
    bool is_static = false;
    std::size_t sample_count = 0;
    double accel_mean_abs_error_mps2 = 0.0;
    double accel_var_mps2 = 0.0;
    double gyro_mean_norm_rps = 0.0;
    double gyro_var_rps2 = 0.0;
  };

  Eigen::Matrix3d skewMatrix(const Eigen::Vector3d & w) const;
  Eigen::Matrix3d expSO3(const Eigen::Vector3d & theta) const;
  iekf_lio::ImuTrack collectRecentImuTrack(
    const builtin_interfaces::msg::Time & end_time,
    double window_s);
  ZuptDetectionResult detectStationaryFromImuWindow(
    const iekf_lio::ImuTrack & imu_track) const;
  bool applyZuptVelocityUpdate(double velocity_sigma_mps);
#endif

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
            return stop_requested_.load() || hasPendingLidar();
          });

        if (stop_requested_.load()) {
          return;
        }
      }

      (void)tryProcessOneScan();
    }
  }

  void backendLoop()
  {
    while (true) {
      KeyframeData keyframe;
      {
        std::unique_lock<std::mutex> lk(keyframe_mutex_);
        keyframe_cv_.wait(
          lk,
          [this]()
          {
            return stop_requested_.load() || !keyframe_queue_.empty();
          });

        if (stop_requested_.load() && keyframe_queue_.empty()) {
          return;
        }

        keyframe = std::move(keyframe_queue_.front());
        keyframe_queue_.pop_front();
      }

      processBackendKeyframe(std::move(keyframe));
    }
  }

  void enqueueKeyframe(KeyframeData keyframe)
  {
    {
      std::lock_guard<std::mutex> lk(keyframe_mutex_);
      keyframe_queue_.push_back(std::move(keyframe));
    }
    keyframe_cv_.notify_one();
  }

  void processBackendKeyframe(KeyframeData keyframe)
  {
    std::size_t cached_size = 0;
    std::size_t max_cached = 0;
    std::size_t archive_cached_size = 0;
    std::size_t max_archive_cached = 0;
    {
      std::lock_guard<std::mutex> lk(backend_keyframes_mutex_);
      backend_keyframes_.push_back(std::move(keyframe));
      backend_keyframe_archive_[backend_keyframes_.back().id] = backend_keyframes_.back();
      backend_keyframe_archive_order_.push_back(backend_keyframes_.back().id);
      max_cached = static_cast<std::size_t>(std::max(1, keyframe_backend_max_cached_));
      max_archive_cached = static_cast<std::size_t>(std::max(1, keyframe_backend_max_archive_cached_));
      while (backend_keyframes_.size() > max_cached) {
        backend_keyframes_.pop_front();
      }
      while (backend_keyframe_archive_order_.size() > max_archive_cached) {
        const std::size_t oldest_id = backend_keyframe_archive_order_.front();
        backend_keyframe_archive_order_.pop_front();
        backend_keyframe_archive_.erase(oldest_id);
      }
      cached_size = backend_keyframes_.size();
      archive_cached_size = backend_keyframe_archive_.size();
    }
    const auto recent = getRecentBackendKeyframes(1);
    if (recent.empty()) {
      return;
    }
    const auto & stored = recent.back();
    const iekf_lio::BackendPoseNode current_node {
      stored.id,
      stored.time_s,
      stored.p_wb,
      stored.r_wb};
    iekf_lio::BackendPoseNode previous_node;
    const bool has_previous_node = backend_factor_graph_.getLatestNode(&previous_node);
    backend_factor_graph_.addPoseNode(current_node);
    if (!has_previous_node) {
      iekf_lio::BackendPriorFactor prior_factor;
      prior_factor.keyframe_id = current_node.keyframe_id;
      prior_factor.p_wb = current_node.p_wb;
      prior_factor.r_wb = current_node.r_wb;
      backend_factor_graph_.addPriorFactor(prior_factor);
    } else {
      iekf_lio::BackendBetweenFactor between_factor;
      between_factor.from_id = previous_node.keyframe_id;
      between_factor.to_id = current_node.keyframe_id;
      between_factor.r_ij = previous_node.r_wb.transpose() * current_node.r_wb;
      between_factor.t_ij = previous_node.r_wb.transpose() * (current_node.p_wb - previous_node.p_wb);
      backend_factor_graph_.addBetweenFactor(between_factor);
    }

    const iekf_lio::ScanContextEntry query_entry = scan_context_manager_.makeDescriptor(
      stored.id,
      stored.time_s,
      stored.cloud_i);
    iekf_lio::LoopCandidate loop_candidate;
    const bool has_loop_candidate = scan_context_manager_.detectLoop(query_entry, &loop_candidate);
    bool has_loop_constraint = false;
    iekf_lio::BackendLoopConstraint loop_constraint;
    if (has_loop_candidate) {
      KeyframeData matched_keyframe;
      if (getBackendKeyframeById(loop_candidate.match_id, &matched_keyframe)) {
        const iekf_lio::LoopRegistrationKeyframeView query_view {
          stored.id, stored.p_wb, stored.r_wb, stored.cloud_i};
        const iekf_lio::LoopRegistrationKeyframeView target_view {
          matched_keyframe.id,
          matched_keyframe.p_wb,
          matched_keyframe.r_wb,
          matched_keyframe.cloud_i};
        has_loop_constraint = loop_registration_.alignKeyframes(
          query_view,
          target_view,
          loop_candidate.yaw_init_rad,
          &loop_constraint);
        if (has_loop_constraint && loop_constraint.valid) {
          iekf_lio::BackendLoopFactor loop_factor;
          loop_factor.from_id = loop_constraint.from_id;
          loop_factor.to_id = loop_constraint.to_id;
          loop_factor.t_ij = loop_constraint.t_ij;
          loop_factor.r_ij = loop_constraint.r_ij;
          loop_factor.information = loop_constraint.information;
          loop_factor.fitness = loop_constraint.fitness;
          backend_factor_graph_.addLoopFactor(loop_factor);
        }
      }
    }
    scan_context_manager_.addEntry(query_entry);

    iekf_lio::BackendOptimizationResult opt_result;
    iekf_lio::BackendPoseNode latest_optimized_node;
    bool has_optimized_node = false;
    if (backend_optimize_enable_) {
      opt_result = backend_factor_graph_.optimizeOnce();
      has_optimized_node = backend_factor_graph_.getLatestOptimizedNode(&latest_optimized_node);
      publishBackendOptimizedPath();
    }

    if (has_loop_candidate) {
      RCLCPP_INFO(
        this->get_logger(),
        "Loop candidate detected: query=%zu match=%zu score=%.4f yaw_init_deg=%.2f sc_entries=%zu loop_factor=%s fitness=%.4f",
        loop_candidate.query_id,
        loop_candidate.match_id,
        loop_candidate.score,
        loop_candidate.yaw_init_rad * 180.0 / M_PI,
        scan_context_manager_.entryCount(),
        has_loop_constraint ? "true" : "false",
        has_loop_constraint ? loop_constraint.fitness : -1.0);
    }

    RCLCPP_DEBUG(
      this->get_logger(),
      "Backend keyframe stored: id=%zu total=%zu max_cached=%zu archive=%zu max_archive_cached=%zu graph(nodes=%zu priors=%zu between=%zu loops=%zu) opt_enabled=%s opt(success=%s iters=%zu cost=%.6f optimized_nodes=%zu latest_id=%zu) loop_candidate=%s loop_factor=%s sc_entries=%zu time=%.3f points=%zu",
      stored.id,
      cached_size,
      max_cached,
      archive_cached_size,
      max_archive_cached,
      backend_factor_graph_.nodeCount(),
      backend_factor_graph_.priorFactorCount(),
      backend_factor_graph_.betweenFactorCount(),
      backend_factor_graph_.loopFactorCount(),
      backend_optimize_enable_ ? "true" : "false",
      opt_result.success ? "true" : "false",
      opt_result.iteration_count,
      opt_result.final_cost,
      opt_result.optimized_node_count,
      has_optimized_node ? latest_optimized_node.keyframe_id : 0U,
      has_loop_candidate ? "true" : "false",
      has_loop_constraint ? "true" : "false",
      scan_context_manager_.entryCount(),
      stored.time_s,
      stored.cloud_i == nullptr ? 0U : stored.cloud_i->size());
  }

  std::size_t backendKeyframeCount() const
  {
    std::lock_guard<std::mutex> lk(backend_keyframes_mutex_);
    return backend_keyframes_.size();
  }

  std::vector<KeyframeData> getRecentBackendKeyframes(std::size_t max_count) const
  {
    std::vector<KeyframeData> out;
    if (max_count == 0) {
      return out;
    }

    std::lock_guard<std::mutex> lk(backend_keyframes_mutex_);
    const std::size_t take_count = std::min(max_count, backend_keyframes_.size());
    out.reserve(take_count);
    const auto begin_it = backend_keyframes_.end() - static_cast<std::ptrdiff_t>(take_count);
    for (auto it = begin_it; it != backend_keyframes_.end(); ++it) {
      out.push_back(*it);
    }
    return out;
  }

  bool getBackendKeyframeById(std::size_t id, KeyframeData * out) const
  {
    if (out == nullptr) {
      return false;
    }
    std::lock_guard<std::mutex> lk(backend_keyframes_mutex_);
    const auto it = backend_keyframe_archive_.find(id);
    if (it == backend_keyframe_archive_.end()) {
      return false;
    }
    *out = it->second;
    return true;
  }

  void publishBackendOptimizedPath()
  {
    if (backend_path_pub_ == nullptr) {
      return;
    }

    const auto optimized_nodes = backend_factor_graph_.getOptimizedPoseNodesSnapshot();
    nav_msgs::msg::Path backend_path_msg;
    backend_path_msg.header.frame_id = map_frame_id_;

    for (const auto & node : optimized_nodes) {
      geometry_msgs::msg::PoseStamped pose_stamped;
      pose_stamped.header.frame_id = map_frame_id_;
      pose_stamped.header.stamp = secToStamp(node.time_s);
      pose_stamped.pose.position.x = node.p_wb.x();
      pose_stamped.pose.position.y = node.p_wb.y();
      pose_stamped.pose.position.z = node.p_wb.z();

      Eigen::Quaterniond q_wb(node.r_wb);
      if (q_wb.norm() < 1e-12) {
        q_wb = Eigen::Quaterniond::Identity();
      } else {
        q_wb.normalize();
      }
      pose_stamped.pose.orientation.w = q_wb.w();
      pose_stamped.pose.orientation.x = q_wb.x();
      pose_stamped.pose.orientation.y = q_wb.y();
      pose_stamped.pose.orientation.z = q_wb.z();
      backend_path_msg.poses.push_back(pose_stamped);
    }

    if (!backend_path_msg.poses.empty()) {
      backend_path_msg.header.stamp = backend_path_msg.poses.back().header.stamp;
    }

    {
      std::lock_guard<std::mutex> lk(backend_path_mutex_);
      backend_path_msg_ = backend_path_msg;
    }
    backend_path_pub_->publish(backend_path_msg);
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
    std::uint64_t map_transform_ns = 0;
    std::uint64_t map_insert_ns = 0;
    std::uint64_t map_prune_ns = 0;
    std::uint64_t predict_state_prop_ns = 0;
    std::uint64_t predict_cov_ns = 0;
    std::uint64_t predict_record_ns = 0;
    std::uint64_t deskew_end_interp_ns = 0;
    std::uint64_t deskew_point_interp_ns = 0;
    std::uint64_t deskew_point_tf_ns = 0;
    std::uint64_t deskew_merge_ns = 0;

    iekf_lio::LidarScan lidar_scan_internal =
      capLidarScanForDeskew(prep_lidar_result.scan, static_cast<std::size_t>(std::max(0, deskew_max_input_points_)));
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
      iekf_predictor_.resetPdrRuntime();
      pdr_buffer_.clear();
      pdr_coarse_peak_buffer_.clear();
      pdr_peak_debug_buffer_.clear();
      last_pdr_coarse_peak_time_s_ = -1.0;
      has_pending_nms_peak_ = false;
      has_last_committed_peak_ = false;
      pdr_step_buffer_.clear();
      pdr_step_event_buffer_.clear();
      last_pdr_downsample_time_s_ = -1.0;
      next_pdr_sample_id_ = 0;
      keyframe_manager_.reset();
      frontend_keyframe_history_.clear();
      local_map_active_keyframes_.clear();
      local_map_active_keyframe_ids_.clear();
      keyframe_voxel_cache_.clear();
      voxel_map_.clear();
      local_map_initialized_ = false;
      iekf_state_.x.r_wb = imu_init_result.initial_r_wb;
      iekf_state_.x.b_g = imu_init_result.gyro_bias;
      iekf_state_.x.b_a = imu_init_result.accel_bias;
      iekf_state_.x.g_w = imu_init_result.gravity_w;
      const auto init_rpy = rotationMatrixToRpy(iekf_state_.x.r_wb);
      RCLCPP_INFO(
        this->get_logger(),
        "IMU init done: rpy=(%.6f,%.6f,%.6f) b_g=(%.6f,%.6f,%.6f) b_a=(%.6f,%.6f,%.6f) g=(%.6f,%.6f,%.6f)",
        init_rpy[0], init_rpy[1], init_rpy[2],
        iekf_state_.x.b_g.x(), iekf_state_.x.b_g.y(), iekf_state_.x.b_g.z(),
        iekf_state_.x.b_a.x(), iekf_state_.x.b_a.y(), iekf_state_.x.b_a.z(),
        iekf_state_.x.g_w.x(), iekf_state_.x.g_w.y(), iekf_state_.x.g_w.z());
    }
    prep_ns = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(SteadyClock::now() - prep_t0).count());

    std::vector<iekf_lio::ImuPredictedState> imu_pred_states;
    std::vector<iekf_lio::PdrPreprocessedSample> pdr_pred_samples;
    iekf_lio::IekfPredictorTiming predict_timing;
    const auto predict_t0 = SteadyClock::now();
    iekf_predictor_.predictWithMidpoint(
      imu_track_internal, iekf_state_, &imu_pred_states, &predict_timing, &pdr_pred_samples);
    predict_ns = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(SteadyClock::now() - predict_t0).count());
    predict_state_prop_ns = predict_timing.state_propagation_ns;
    predict_cov_ns = predict_timing.covariance_ns;
    predict_record_ns = predict_timing.state_record_ns;
    appendPdrSamples(pdr_pred_samples);
    publishImuPredictedStates(imu_pred_states);

    iekf_lio::LidarScanXYZ lidar_scan_xyz;
    if (deskew_enable_) {
      const auto deskew_t0 = SteadyClock::now();
      const iekf_lio::LidarToImuExtrinsic extrinsic {extrinsic_r_il_, extrinsic_t_il_};
      iekf_lio::CloudDeskewTiming deskew_timing;
      lidar_scan_xyz = deskewer_.deskewToImuEnd(
        lidar_scan_internal, imu_pred_states, extrinsic, &deskew_timing);
      deskew_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(SteadyClock::now() - deskew_t0).count());
      deskew_end_interp_ns = deskew_timing.end_state_interp_ns;
      deskew_point_interp_ns = deskew_timing.point_interp_ns;
      deskew_point_tf_ns = deskew_timing.point_transform_ns;
      deskew_merge_ns = deskew_timing.output_merge_ns;
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

    std::size_t map_voxels = 0;
    std::uint64_t update_search_ns = 0;
    std::uint64_t update_plane_ns = 0;
    std::uint64_t update_accumulate_ns = 0;
    std::uint64_t update_solve_ns = 0;
    bool used_update = false;
    std::string update_skip_reason = "none";
    iekf_lio::IekfUpdateResult upd;
    if (!local_map_initialized_) {
      update_skip_reason = "local_map_not_initialized";
    } else {
      // Update branch: run IEKF updater on active-keyframe local map only.
      if (updater_enable_) {
        const auto update_t0 = SteadyClock::now();
        used_update = iekf_updater_.updatePoseWithPointToMap(
          scan_for_update,
          voxel_map_,
          iekf_state_,
          &upd);
        update_search_ns = upd.radius_search_ns;
        update_plane_ns = upd.plane_fit_ns;
        update_accumulate_ns = upd.accumulate_ns;
        update_solve_ns = upd.solve_ns;
        if (!used_update) {
          if (
            updater_min_correspondences_ > 0 &&
            upd.correspondences > 0 &&
            upd.correspondences < static_cast<std::size_t>(updater_min_correspondences_))
          {
            update_skip_reason = "too_few_correspondences";
          } else {
            update_skip_reason = "update_rejected";
          }
        }
        update_ns = static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(SteadyClock::now() - update_t0).count());
      } else {
        update_skip_reason = "update_disabled";
      }
    }

    const iekf_lio::KeyframeDecision keyframe_decision = keyframe_manager_.update(
      stampToSec(scan_end_time), iekf_state_.x.p_wb, iekf_state_.x.r_wb);
    if (keyframe_decision.is_keyframe) {
      const KeyframeData keyframe = makeKeyframeData(
        keyframe_decision.total_keyframes,
        stampToSec(scan_end_time),
        iekf_state_.x.p_wb,
        iekf_state_.x.r_wb,
        scan_for_map);
      cacheKeyframeVoxelContributions(keyframe);
      appendActiveKeyframe(keyframe);
      enqueueKeyframe(keyframe);
    }

    if (keyframe_decision.is_keyframe) {
      const auto map_insert_t0 = SteadyClock::now();
      updateLocalMapByActiveKeyframeDiff(iekf_state_.x.p_wb);
      map_insert_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(SteadyClock::now() - map_insert_t0).count());
    } else {
      map_insert_ns = 0;
    }
    map_ns = map_insert_ns;

    map_voxels = voxel_map_.size();

    last_history_blocks_.store(map_voxels);

    if (updater_degeneracy_log_only_ && degeneracy_iter_log_ofs_.is_open()) {
      for (const auto & iter_log : upd.iter_logs) {
        degeneracy_iter_log_ofs_
          << std::fixed << std::setprecision(9)
          << stampToSec(scan_end_time) << ","
          << iter_log.iekf_iter << ","
          << iter_log.valid_correspondences << ","
          << std::setprecision(6)
          << iter_log.residual_rmse << ","
          << iter_log.lambda_min << ","
          << iter_log.lambda_max << ","
          << iter_log.lambda_ratio << ","
          << iter_log.condition_number << ","
          << (iter_log.bad_ratio ? 1 : 0) << ","
          << (iter_log.weak_abs ? 1 : 0) << ","
          << (iter_log.bad_quality ? 1 : 0) << ","
          << (iter_log.is_degenerate_raw ? 1 : 0) << ","
          << (iter_log.degeneracy_projection_triggered ? 1 : 0) << ","
          << (iter_log.too_few_correspondences ? 1 : 0) << ","
          << iter_log.threshold_ref << ","
          << iter_log.weights[0] << ","
          << iter_log.weights[1] << ","
          << iter_log.weights[2] << ","
          << iter_log.weights[3] << ","
          << iter_log.weights[4] << ","
          << iter_log.weights[5] << ","
          << iter_log.update_dx_rot_norm << ","
          << iter_log.update_dx_pos_norm << "\n";
      }
    }

    RCLCPP_INFO(
      this->get_logger(),
      "Scheduled one scan: [%.3f, %.3f], imu_in_window=%zu, points=%zu->%zu, deskew_enable=%s, downsample(update=%zu map=%zu), used_update=%s, update_skip=%s, local_map_voxels=%zu selected_map_voxels=%zu active_keyframes=%zu keyframe=%s dt=%.3f trans=%.3f rot_deg=%.2f map_ms(rebuild=%.2f total=%.2f), pred_p=(%.3f,%.3f,%.3f), pred_v=(%.3f,%.3f,%.3f), pred_rpy_rad=(%.3f,%.3f,%.3f)",
      stampToSec(scan_begin_time),
      stampToSec(scan_end_time),
      imu_track_internal.size(),
      raw_point_count,
      filtered_point_count,
      deskew_enable_ ? "true" : "false",
      scan_for_update.points == nullptr ? 0U : scan_for_update.points->size(),
      scan_for_map.points == nullptr ? 0U : scan_for_map.points->size(),
      used_update ? "true" : "false",
      update_skip_reason.c_str(),
      voxel_map_.size(),
      map_voxels,
      local_map_active_keyframes_.size(),
      keyframe_decision.is_keyframe ? "true" : "false",
      keyframe_decision.delta_time_s,
      keyframe_decision.translation_m,
      keyframe_decision.rotation_rad * 180.0 / M_PI,
      static_cast<double>(map_insert_ns) * 1e-6,
      static_cast<double>(map_ns) * 1e-6,
      iekf_state_.x.p_wb.x(),
      iekf_state_.x.p_wb.y(),
      iekf_state_.x.p_wb.z(),
      iekf_state_.x.v_wb.x(),
      iekf_state_.x.v_wb.y(),
      iekf_state_.x.v_wb.z(),
      rotationMatrixToRpy(iekf_state_.x.r_wb)[0],
      rotationMatrixToRpy(iekf_state_.x.r_wb)[1],
      rotationMatrixToRpy(iekf_state_.x.r_wb)[2]);
    const auto publish_t0 = SteadyClock::now();
    publishScanState(scan_end_time);
    if (shouldPublishPointsWorld()) {
      const std::vector<Eigen::Vector3d> points_world =
        transformDeskewedScanToWorld(scan_for_map, iekf_state_.x);
      publishPointsWorld(points_world, scan_end_time);
    }
    writeTumPose(scan_end_time);
    const std::uint64_t publish_ns = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(SteadyClock::now() - publish_t0).count());
    const std::uint64_t total_ns = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(SteadyClock::now() - total_t0).count());

    if (time_scan_ofs_.is_open()) {
      time_scan_ofs_
        << std::fixed << std::setprecision(9)
        << stampToSec(scan_end_time) << " "
        << std::setprecision(6)
        << static_cast<double>(prep_ns) * 1e-6 << " "
        << static_cast<double>(predict_ns) * 1e-6 << " "
        << static_cast<double>(predict_state_prop_ns) * 1e-6 << " "
        << static_cast<double>(predict_cov_ns) * 1e-6 << " "
        << static_cast<double>(predict_record_ns) * 1e-6 << " "
        << static_cast<double>(deskew_ns) * 1e-6 << " "
        << static_cast<double>(deskew_end_interp_ns) * 1e-6 << " "
        << static_cast<double>(deskew_point_interp_ns) * 1e-6 << " "
        << static_cast<double>(deskew_point_tf_ns) * 1e-6 << " "
        << static_cast<double>(deskew_merge_ns) * 1e-6 << " "
        << static_cast<double>(downsample_ns) * 1e-6 << " "
        << static_cast<double>(update_ns) * 1e-6 << " "
        << static_cast<double>(update_search_ns) * 1e-6 << " "
        << static_cast<double>(update_plane_ns) * 1e-6 << " "
        << static_cast<double>(update_accumulate_ns) * 1e-6 << " "
        << static_cast<double>(update_solve_ns) * 1e-6 << " "
        << static_cast<double>(map_insert_ns) * 1e-6 << " "
        << static_cast<double>(map_prune_ns) * 1e-6 << " "
        << static_cast<double>(publish_ns) * 1e-6 << " "
        << static_cast<double>(total_ns) * 1e-6 << " "
        << map_voxels << " "
        << (keyframe_decision.is_keyframe ? 1 : 0) << " "
        << keyframe_decision.total_keyframes << "\n";
    }

    ++lidar_processed_count_;
    timing_prep_ns_total_.fetch_add(prep_ns);
    timing_predict_ns_total_.fetch_add(predict_ns);
    timing_deskew_ns_total_.fetch_add(deskew_ns);
    timing_downsample_ns_total_.fetch_add(downsample_ns);
    timing_update_ns_total_.fetch_add(update_ns);
    timing_map_ns_total_.fetch_add(map_ns);
    timing_map_transform_ns_total_.fetch_add(map_transform_ns);
    timing_map_insert_ns_total_.fetch_add(map_insert_ns);
    timing_map_prune_ns_total_.fetch_add(map_prune_ns);
    timing_publish_ns_total_.fetch_add(publish_ns);
    timing_total_ns_total_.fetch_add(total_ns);
    return true;
  }

  void publishScanState(const builtin_interfaces::msg::Time & stamp)
  {
    Eigen::Quaterniond q_wb_frontend(iekf_state_.x.r_wb);
    if (q_wb_frontend.norm() < 1e-12) {
      q_wb_frontend = Eigen::Quaterniond::Identity();
    } else {
      q_wb_frontend.normalize();
    }

    nav_msgs::msg::Odometry odom_msg;
    odom_msg.header.stamp = stamp;
    odom_msg.header.frame_id = map_frame_id_;
    odom_msg.child_frame_id = base_frame_id_;
    odom_msg.pose.pose.position.x = iekf_state_.x.p_wb.x();
    odom_msg.pose.pose.position.y = iekf_state_.x.p_wb.y();
    odom_msg.pose.pose.position.z = iekf_state_.x.p_wb.z();
    odom_msg.pose.pose.orientation.w = q_wb_frontend.w();
    odom_msg.pose.pose.orientation.x = q_wb_frontend.x();
    odom_msg.pose.pose.orientation.y = q_wb_frontend.y();
    odom_msg.pose.pose.orientation.z = q_wb_frontend.z();
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
      iekf_lio::BackendPoseNode tf_node;
      const bool has_backend_tf = backend_factor_graph_.getLatestOptimizedNode(&tf_node);
      const Eigen::Vector3d tf_p_wb = has_backend_tf ? tf_node.p_wb : iekf_state_.x.p_wb;
      const Eigen::Matrix3d tf_r_wb = has_backend_tf ? tf_node.r_wb : iekf_state_.x.r_wb;
      Eigen::Quaterniond q_wb_tf(tf_r_wb);
      if (q_wb_tf.norm() < 1e-12) {
        q_wb_tf = Eigen::Quaterniond::Identity();
      } else {
        q_wb_tf.normalize();
      }

      geometry_msgs::msg::TransformStamped tf_msg;
      tf_msg.header.stamp = stamp;
      tf_msg.header.frame_id = map_frame_id_;
      tf_msg.child_frame_id = base_frame_id_;
      tf_msg.transform.translation.x = tf_p_wb.x();
      tf_msg.transform.translation.y = tf_p_wb.y();
      tf_msg.transform.translation.z = tf_p_wb.z();
      tf_msg.transform.rotation.w = q_wb_tf.w();
      tf_msg.transform.rotation.x = q_wb_tf.x();
      tf_msg.transform.rotation.y = q_wb_tf.y();
      tf_msg.transform.rotation.z = q_wb_tf.z();
      tf_broadcaster_->sendTransform(tf_msg);
    }
  }

  void publishImuPredictedStates(const std::vector<iekf_lio::ImuPredictedState> & imu_pred_states)
  {
    if (imu_prediction_pub_ == nullptr) {
      return;
    }

    const auto sub_count =
      imu_prediction_pub_->get_subscription_count() +
      imu_prediction_pub_->get_intra_process_subscription_count();
    if (sub_count == 0) {
      return;
    }

    std::size_t batch_publish_count = 0;
    for (const auto & pred_state : imu_pred_states) {
      if (!pred_state.is_propagated) {
        continue;
      }

      iekf_lio::msg::ImuPredictionState msg;
      msg.stamp = secToStamp(pred_state.time_s);
      for (int i = 0; i < 3; ++i) {
        msg.gyro_mid_unbiased[static_cast<std::size_t>(i)] = pred_state.gyro_mid_unbiased_rps(i);
        msg.linear_accel_mid_w[static_cast<std::size_t>(i)] = pred_state.linear_accel_mid_w_mps2(i);
      }
      msg.yaw_mid_rad = pred_state.yaw_mid_rad;
      imu_prediction_pub_->publish(msg);
      ++batch_publish_count;
      ++imu_prediction_pub_count_;
    }

    // if (batch_publish_count > 0) {
    //   RCLCPP_INFO_THROTTLE(
    //     this->get_logger(),
    //     *this->get_clock(),
    //     1000,
    //     "IMU prediction topic stats: published_total=%llu last_batch=%zu qos_depth=%d subscribers=%zu",
    //     static_cast<unsigned long long>(imu_prediction_pub_count_),
    //     batch_publish_count,
    //     imu_prediction_qos_depth_,
    //     sub_count);
    // }
  }

  void appendPdrSamples(const std::vector<iekf_lio::PdrPreprocessedSample> & pdr_samples)
  {
    if (pdr_samples.empty()) {
      return;
    }

    const double sample_dt_s = (pdr_sample_rate_hz_ > 1e-6) ? (1.0 / pdr_sample_rate_hz_) : 0.0;
    for (const auto & sample : pdr_samples) {
      if (sample_dt_s > 0.0 && last_pdr_downsample_time_s_ >= 0.0 &&
        (sample.time_s - last_pdr_downsample_time_s_) < sample_dt_s)
      {
        continue;
      }
      auto accepted_sample = sample;
      accepted_sample.sample_id = next_pdr_sample_id_++;
      pdr_buffer_.push_back(accepted_sample);
      last_pdr_downsample_time_s_ = accepted_sample.time_s;
      if (pdr_signal_ofs_.is_open()) {
        pdr_signal_ofs_
          << accepted_sample.sample_id << " "
          << std::fixed << std::setprecision(9)
          << accepted_sample.time_s << " "
          << std::setprecision(6)
          << accepted_sample.accel_norm_minus_g_lpf_mps2 << "\n";
      }
      maybeDetectCoarsePeakFromTail();
      validatePendingCoarsePeaks();
      flushPendingNmsPeakIfReady(accepted_sample.time_s);
    }
    trimBuffer(
      pdr_buffer_,
      static_cast<std::size_t>(std::max(1, pdr_buffer_size_)));
  }

  bool isThreePointPeakAt(std::size_t idx) const
  {
    if (idx == 0 || idx + 1 >= pdr_buffer_.size()) {
      return false;
    }
    const auto & s0 = pdr_buffer_[idx - 1];
    const auto & s1 = pdr_buffer_[idx];
    const auto & s2 = pdr_buffer_[idx + 1];
    return
      (s1.accel_norm_minus_g_lpf_mps2 > s0.accel_norm_minus_g_lpf_mps2) &&
      (s1.accel_norm_minus_g_lpf_mps2 >= s2.accel_norm_minus_g_lpf_mps2);
  }

  bool passesPeakHeightAt(std::size_t idx) const
  {
    return pdr_buffer_[idx].accel_norm_minus_g_lpf_mps2 >= pdr_peak_min_height_;
  }

  bool passesCoarsePeakInterval(double peak_time_s) const
  {
    return !(last_pdr_coarse_peak_time_s_ >= 0.0 &&
      (peak_time_s - last_pdr_coarse_peak_time_s_) < pdr_peak_min_interval_s_);
  }

  void maybeDetectCoarsePeakFromTail()
  {
    if (!pdr_peak_enable_ || pdr_buffer_.size() < 3) {
      return;
    }

    const std::size_t center_idx = pdr_buffer_.size() - 2;
    if (!isThreePointPeakAt(center_idx) || !passesPeakHeightAt(center_idx)) {
      return;
    }
    const auto & center = pdr_buffer_[center_idx];
    if (!passesCoarsePeakInterval(center.time_s)) {
      return;
    }

    iekf_lio::PdrPeakEvent coarse_peak;
    coarse_peak.sample_id = center.sample_id;
    coarse_peak.time_s = center.time_s;
    coarse_peak.accel_norm_minus_g_lpf_mps2 = center.accel_norm_minus_g_lpf_mps2;
    pdr_coarse_peak_buffer_.push_back(coarse_peak);
    last_pdr_coarse_peak_time_s_ = coarse_peak.time_s;
    trimBuffer(
      pdr_coarse_peak_buffer_,
      static_cast<std::size_t>(std::max(1, pdr_peak_buffer_size_)));
    if (pdr_coarse_peak_ofs_.is_open()) {
      pdr_coarse_peak_ofs_
        << coarse_peak.sample_id << " "
        << std::fixed << std::setprecision(9)
        << coarse_peak.time_s << " "
        << std::setprecision(6)
        << coarse_peak.accel_norm_minus_g_lpf_mps2 << "\n";
    }
  }

  bool findPdrBufferIndexBySampleId(std::uint64_t sample_id, std::size_t * index_out) const
  {
    if (index_out == nullptr) {
      return false;
    }
    for (std::size_t i = 0; i < pdr_buffer_.size(); ++i) {
      if (pdr_buffer_[i].sample_id == sample_id) {
        *index_out = i;
        return true;
      }
    }
    return false;
  }

  bool passesAmpdValidationAt(std::size_t center_idx) const
  {
    if (!pdr_ampd_enable_) {
      return true;
    }
    if (center_idx >= pdr_buffer_.size()) {
      return false;
    }

    const int smin = std::max(1, pdr_ampd_smin_);
    const int smax = std::max(smin, pdr_ampd_smax_);
    int valid_scales = 0;
    int passed_scales = 0;
    const double center_value = pdr_buffer_[center_idx].accel_norm_minus_g_lpf_mps2;

    for (int s = smin; s <= smax; ++s) {
      if (center_idx < static_cast<std::size_t>(s) ||
        (center_idx + static_cast<std::size_t>(s)) >= pdr_buffer_.size())
      {
        continue;
      }
      ++valid_scales;
      const double left_value =
        pdr_buffer_[center_idx - static_cast<std::size_t>(s)].accel_norm_minus_g_lpf_mps2;
      const double right_value =
        pdr_buffer_[center_idx + static_cast<std::size_t>(s)].accel_norm_minus_g_lpf_mps2;
      if (center_value > left_value && center_value >= right_value) {
        ++passed_scales;
      }
    }

    if (valid_scales == 0) {
      return false;
    }
    const double pass_ratio =
      static_cast<double>(passed_scales) / static_cast<double>(valid_scales);
    return pass_ratio >= pdr_ampd_pass_ratio_;
  }

  void maybeBuildPdrStepEvent(const iekf_lio::PdrStepCandidate & candidate)
  {
    double max_value = -std::numeric_limits<double>::infinity();
    double min_value = std::numeric_limits<double>::infinity();
    double gyro_norm_sum = 0.0;
    std::size_t gyro_norm_count = 0;

    for (const auto & sample : pdr_buffer_) {
      if (sample.time_s < candidate.start_time_s || sample.time_s > candidate.end_time_s) {
        continue;
      }
      max_value = std::max(max_value, sample.accel_norm_minus_g_lpf_mps2);
      min_value = std::min(min_value, sample.accel_norm_minus_g_lpf_mps2);
      gyro_norm_sum += sample.gyro_unbiased_b_rps.norm();
      ++gyro_norm_count;
    }

    if (!std::isfinite(max_value) || !std::isfinite(min_value) || gyro_norm_count == 0)
    {
      return;
    }

    const double peak_valley_diff = max_value - min_value;
    const double gyro_norm_mean_rps =
      gyro_norm_sum / static_cast<double>(gyro_norm_count);
    if (peak_valley_diff < pdr_step_min_peak_valley_diff_) {
      return;
    }
    const double peak_valley_diff_nonlinear = std::pow(peak_valley_diff, 0.25);

    iekf_lio::PdrStepEvent event;
    event.start_time_s = candidate.start_time_s;
    event.end_time_s = candidate.end_time_s;
    event.duration_s = candidate.duration_s;
    event.max_value = max_value;
    event.min_value = min_value;
    event.gyro_norm_mean_rps = gyro_norm_mean_rps;
    event.peak_valley_diff = peak_valley_diff;
    event.peak_valley_diff_nonlinear = peak_valley_diff_nonlinear;
    pdr_step_event_buffer_.push_back(event);
    trimBuffer(
      pdr_step_event_buffer_,
      static_cast<std::size_t>(std::max(1, pdr_step_buffer_size_)));
    if (pdr_step_event_ofs_.is_open()) {
      pdr_step_event_ofs_
        << std::fixed << std::setprecision(9)
        << event.start_time_s << " "
        << event.end_time_s << " "
        << event.duration_s << " "
        << std::setprecision(6)
        << event.max_value << " "
        << event.min_value << " "
        << event.gyro_norm_mean_rps << " "
        << event.peak_valley_diff << " "
        << event.peak_valley_diff_nonlinear << "\n";
    }

    RCLCPP_INFO(
      this->get_logger(),
      "PDR step event: t0=%.3f t1=%.3f diff=%.6f nonlinear=%.4f gyro_mean=%.4f total_events=%zu",
      event.start_time_s,
      event.end_time_s,
      event.peak_valley_diff,
      event.peak_valley_diff_nonlinear,
      event.gyro_norm_mean_rps,
      pdr_step_event_buffer_.size());
  }

  void maybeBuildPdrStepCandidateFromCommittedPeaks(
    const iekf_lio::PdrPeakEvent & previous_peak,
    const iekf_lio::PdrPeakEvent & current_peak)
  {
    const double dt_s = current_peak.time_s - previous_peak.time_s;
    if (dt_s < pdr_step_min_interval_s_ || dt_s > pdr_step_max_interval_s_) {
      return;
    }

    iekf_lio::PdrStepCandidate candidate;
    candidate.start_time_s = previous_peak.time_s;
    candidate.end_time_s = current_peak.time_s;
    candidate.duration_s = dt_s;
    candidate.start_peak_value = previous_peak.accel_norm_minus_g_lpf_mps2;
    candidate.end_peak_value = current_peak.accel_norm_minus_g_lpf_mps2;
    pdr_step_buffer_.push_back(candidate);
    trimBuffer(
      pdr_step_buffer_,
      static_cast<std::size_t>(std::max(1, pdr_step_buffer_size_)));
    if (pdr_step_candidate_ofs_.is_open()) {
      pdr_step_candidate_ofs_
        << std::fixed << std::setprecision(9)
        << candidate.start_time_s << " "
        << candidate.end_time_s << " "
        << candidate.duration_s << " "
        << std::setprecision(6)
        << candidate.start_peak_value << " "
        << candidate.end_peak_value << "\n";
    }
    RCLCPP_INFO(
      this->get_logger(),
      "PDR step candidate: t0=%.3f t1=%.3f dt=%.3f total_steps=%zu",
      candidate.start_time_s,
      candidate.end_time_s,
      candidate.duration_s,
      pdr_step_buffer_.size());
    maybeBuildPdrStepEvent(candidate);
  }

  void commitFinalPeak(const iekf_lio::PdrPeakEvent & peak)
  {
    const bool has_previous_peak = has_last_committed_peak_;
    const iekf_lio::PdrPeakEvent previous_peak = last_committed_peak_;
    last_committed_peak_ = peak;
    has_last_committed_peak_ = true;
    pdr_peak_debug_buffer_.push_back(peak);
    trimBuffer(
      pdr_peak_debug_buffer_,
      static_cast<std::size_t>(std::max(1, pdr_peak_buffer_size_)));
    if (pdr_peak_debug_ofs_.is_open()) {
      pdr_peak_debug_ofs_
        << peak.sample_id << " "
        << std::fixed << std::setprecision(9)
        << peak.time_s << " "
        << std::setprecision(6)
        << peak.accel_norm_minus_g_lpf_mps2 << "\n";
    }

    RCLCPP_INFO(
      this->get_logger(),
      "PDR final peak: id=%llu t=%.3f accel_norm_minus_g_lpf=%.6f total_peaks=%zu",
      static_cast<unsigned long long>(peak.sample_id),
      peak.time_s,
      peak.accel_norm_minus_g_lpf_mps2,
      pdr_peak_debug_buffer_.size());

    if (has_previous_peak) {
      maybeBuildPdrStepCandidateFromCommittedPeaks(previous_peak, peak);
    }
  }

  void handleAmpdValidatedPeak(const iekf_lio::PdrPeakEvent & peak)
  {
    if (!has_pending_nms_peak_) {
      pending_nms_peak_ = peak;
      has_pending_nms_peak_ = true;
      return;
    }

    const double dt_s = peak.time_s - pending_nms_peak_.time_s;
    if (dt_s < pdr_peak_nms_interval_s_) {
      if (peak.accel_norm_minus_g_lpf_mps2 > pending_nms_peak_.accel_norm_minus_g_lpf_mps2) {
        pending_nms_peak_ = peak;
      }
      return;
    }

    commitFinalPeak(pending_nms_peak_);
    pending_nms_peak_ = peak;
    has_pending_nms_peak_ = true;
  }

  void flushPendingNmsPeakIfReady(double current_time_s)
  {
    if (!has_pending_nms_peak_) {
      return;
    }
    if ((current_time_s - pending_nms_peak_.time_s) < pdr_peak_nms_interval_s_) {
      return;
    }
    commitFinalPeak(pending_nms_peak_);
    has_pending_nms_peak_ = false;
  }

  void validatePendingCoarsePeaks()
  {
    while (!pdr_coarse_peak_buffer_.empty()) {
      const auto coarse_peak = pdr_coarse_peak_buffer_.front();
      std::size_t center_idx = 0;
      if (!findPdrBufferIndexBySampleId(coarse_peak.sample_id, &center_idx)) {
        pdr_coarse_peak_buffer_.pop_front();
        continue;
      }

      const std::size_t required_right =
        pdr_ampd_enable_ ? static_cast<std::size_t>(std::max(0, pdr_ampd_smax_)) : 0U;
      if ((center_idx + required_right) >= pdr_buffer_.size()) {
        break;
      }

      if (passesAmpdValidationAt(center_idx)) {
        handleAmpdValidatedPeak(coarse_peak);
      }
      pdr_coarse_peak_buffer_.pop_front();
    }
  }

  void initializePdrOutputs()
  {
    try {
      const std::filesystem::path output_dir(pdr_output_dir_);
      std::filesystem::create_directories(output_dir);
      const std::time_t now = std::time(nullptr);
      std::tm local_tm {};
#if defined(_WIN32)
      localtime_s(&local_tm, &now);
#else
      localtime_r(&now, &local_tm);
#endif
      std::ostringstream run_tag_ss;
      run_tag_ss << std::put_time(&local_tm, "%Y%m%d_%H%M%S");
      pdr_run_tag_ = run_tag_ss.str();

      const std::filesystem::path signal_path =
        output_dir / ("pdr_accel_norm_lpf_" + pdr_run_tag_ + ".txt");
      const std::filesystem::path coarse_peak_path =
        output_dir / ("pdr_coarse_peaks_" + pdr_run_tag_ + ".txt");
      const std::filesystem::path peak_debug_path =
        output_dir / ("pdr_peaks_" + pdr_run_tag_ + ".txt");
      const std::filesystem::path step_path =
        output_dir / ("pdr_step_candidates_" + pdr_run_tag_ + ".txt");
      const std::filesystem::path step_event_path =
        output_dir / ("pdr_step_events_" + pdr_run_tag_ + ".txt");
      pdr_signal_ofs_.open(signal_path, std::ios::out | std::ios::trunc);
      pdr_coarse_peak_ofs_.open(coarse_peak_path, std::ios::out | std::ios::trunc);
      pdr_peak_debug_ofs_.open(peak_debug_path, std::ios::out | std::ios::trunc);
      pdr_step_candidate_ofs_.open(step_path, std::ios::out | std::ios::trunc);
      pdr_step_event_ofs_.open(step_event_path, std::ios::out | std::ios::trunc);
      if (!pdr_signal_ofs_.is_open() || !pdr_coarse_peak_ofs_.is_open() ||
        !pdr_peak_debug_ofs_.is_open() || !pdr_step_candidate_ofs_.is_open() ||
        !pdr_step_event_ofs_.is_open())
      {
        RCLCPP_WARN(
          this->get_logger(),
          "PDR output files failed to open under %s",
          output_dir.string().c_str());
        return;
      }
      pdr_signal_ofs_ << "# sample_id time_s accel_norm_minus_g_lpf_mps2\n";
      pdr_coarse_peak_ofs_ << "# sample_id time_s coarse_peak_accel_norm_minus_g_lpf_mps2\n";
      pdr_peak_debug_ofs_ << "# sample_id time_s peak_accel_norm_minus_g_lpf_mps2\n";
      pdr_step_candidate_ofs_ <<
        "# start_time_s end_time_s duration_s start_peak_value end_peak_value\n";
      pdr_step_event_ofs_ <<
        "# start_time_s end_time_s duration_s max_value min_value gyro_norm_mean_rps peak_valley_diff peak_valley_diff_nonlinear\n";
      RCLCPP_INFO(
        this->get_logger(),
        "PDR outputs(run=%s): signal=%s coarse_peaks=%s final_peaks=%s steps=%s step_events=%s",
        pdr_run_tag_.c_str(),
        signal_path.string().c_str(),
        coarse_peak_path.string().c_str(),
        peak_debug_path.string().c_str(),
        step_path.string().c_str(),
        step_event_path.string().c_str());
    } catch (const std::exception & e) {
      RCLCPP_WARN(
        this->get_logger(),
        "PDR output initialization failed: %s",
        e.what());
    }
  }

  void initializeTimingOutputs()
  {
    try {
      const std::filesystem::path output_dir(time_output_dir_);
      std::filesystem::create_directories(output_dir);
      const std::time_t now = std::time(nullptr);
      std::tm local_tm {};
#if defined(_WIN32)
      localtime_s(&local_tm, &now);
#else
      localtime_r(&now, &local_tm);
#endif
      std::ostringstream run_tag_ss;
      run_tag_ss << std::put_time(&local_tm, "%Y%m%d_%H%M%S");
      time_run_tag_ = run_tag_ss.str();
      const std::filesystem::path scan_path =
        output_dir / ("scan_timing_" + time_run_tag_ + ".txt");
      const std::filesystem::path degeneracy_iter_csv_path =
        output_dir / ("degeneracy_iter_log_" + time_run_tag_ + ".csv");
      time_scan_ofs_.open(scan_path, std::ios::out | std::ios::trunc);
      if (!time_scan_ofs_.is_open()) {
        RCLCPP_WARN(
          this->get_logger(),
          "Timing output file failed to open under %s",
          output_dir.string().c_str());
        return;
      }
      if (updater_degeneracy_log_only_) {
        degeneracy_iter_log_ofs_.open(
          degeneracy_iter_csv_path, std::ios::out | std::ios::trunc);
        if (!degeneracy_iter_log_ofs_.is_open()) {
          RCLCPP_WARN(
            this->get_logger(),
            "Degeneracy iteration log file failed to open under %s",
            output_dir.string().c_str());
          return;
        }
      }
      time_scan_ofs_ <<
        "# scan_end_time_s prep_ms predict_ms predict_prop_ms predict_cov_ms predict_record_ms "
        "deskew_ms deskew_end_interp_ms deskew_point_interp_ms deskew_point_tf_ms deskew_merge_ms "
        "downsample_ms update_ms update_search_ms update_plane_ms update_accumulate_ms update_solve_ms "
        "map_insert_ms map_prune_ms publish_ms total_ms local_map_voxels is_keyframe total_keyframes\n";
      if (updater_degeneracy_log_only_) {
        degeneracy_iter_log_ofs_ <<
          "scan_timestamp,iekf_iter,valid_correspondences,residual_rmse,"
          "lambda_min,lambda_max,lambda_ratio,condition_number,"
          "bad_ratio,weak_abs,bad_quality,is_degenerate_raw,"
          "degeneracy_projection_triggered,too_few_correspondences,"
          "threshold_ref,weight_0,weight_1,weight_2,weight_3,weight_4,weight_5,"
          "update_dx_rot_norm,update_dx_pos_norm\n";
      }
      RCLCPP_INFO(
        this->get_logger(),
        "Timing outputs(run=%s): scan=%s%s%s",
        time_run_tag_.c_str(),
        scan_path.string().c_str(),
        updater_degeneracy_log_only_ ? " degeneracy_iter=" : "",
        updater_degeneracy_log_only_ ? degeneracy_iter_csv_path.string().c_str() : "");
    } catch (const std::exception & e) {
      RCLCPP_WARN(
        this->get_logger(),
        "Timing output initialization failed: %s",
        e.what());
    }
  }

  bool shouldPublishPointsWorld() const
  {
    if (points_world_pub_ == nullptr) {
      return false;
    }
    return points_world_pub_->get_subscription_count() > 0 ||
           points_world_pub_->get_intra_process_subscription_count() > 0;
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

  void writeTumPose(const builtin_interfaces::msg::Time & stamp)
  {
    if (!tum_traj_ofs_.is_open()) {
      return;
    }

    Eigen::Quaterniond q_wb(iekf_state_.x.r_wb);
    if (q_wb.norm() < 1e-12) {
      q_wb = Eigen::Quaterniond::Identity();
    } else {
      q_wb.normalize();
    }

    tum_traj_ofs_
      << stampToSec(stamp) << ' '
      << iekf_state_.x.p_wb.x() << ' '
      << iekf_state_.x.p_wb.y() << ' '
      << iekf_state_.x.p_wb.z() << ' '
      << q_wb.x() << ' '
      << q_wb.y() << ' '
      << q_wb.z() << ' '
      << q_wb.w() << '\n';
  }

  std::array<double, 3> rotationMatrixToRpy(const iekf_lio::IekfMat3 & R) const
  {
    const double roll = std::atan2(R(2, 1), R(2, 2));
    const double pitch = std::asin(std::clamp(-R(2, 0), -1.0, 1.0));
    const double yaw = std::atan2(R(1, 0), R(0, 0));
    return {roll, pitch, yaw};
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

  iekf_lio::LidarScan capLidarScanForDeskew(
    const iekf_lio::LidarScan & scan,
    std::size_t max_points) const
  {
    if (max_points == 0 || scan.points.size() <= max_points) {
      return scan;
    }

    iekf_lio::LidarScan out;
    out.frame_id = scan.frame_id;
    out.scan_begin_time_s = scan.scan_begin_time_s;
    out.scan_end_time_s = scan.scan_end_time_s;
    out.timebase_ns = scan.timebase_ns;
    out.points.reserve(max_points);

    const std::size_t total_points = scan.points.size();
    const std::size_t stride = std::max<std::size_t>(1, (total_points + max_points - 1) / max_points);
    for (std::size_t i = 0; i < total_points && out.points.size() < max_points; i += stride) {
      out.points.push_back(scan.points[i]);
    }
    if (out.points.size() > max_points) {
      out.points.resize(max_points);
    }
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
    const std::uint64_t timing_map_transform_ns = timing_map_transform_ns_total_.load();
    const std::uint64_t timing_map_insert_ns = timing_map_insert_ns_total_.load();
    const std::uint64_t timing_map_prune_ns = timing_map_prune_ns_total_.load();
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
    const std::uint64_t delta_map_transform_ns =
      timing_map_transform_ns - last_report_timing_map_transform_ns_;
    const std::uint64_t delta_map_insert_ns =
      timing_map_insert_ns - last_report_timing_map_insert_ns_;
    const std::uint64_t delta_map_prune_ns =
      timing_map_prune_ns - last_report_timing_map_prune_ns_;
    const std::uint64_t delta_publish_ns = timing_publish_ns - last_report_timing_publish_ns_;
    const std::uint64_t delta_total_ns = timing_total_ns - last_report_timing_total_ns_;
    const std::size_t local_map_voxels = last_history_blocks_.load();
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
    const double avg_map_transform_ms = avg_ms(delta_map_transform_ns);
    const double avg_map_insert_ms = avg_ms(delta_map_insert_ns);
    const double avg_map_prune_ms = avg_ms(delta_map_prune_ns);
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
    last_report_timing_map_transform_ns_ = timing_map_transform_ns;
    last_report_timing_map_insert_ns_ = timing_map_insert_ns;
    last_report_timing_map_prune_ns_ = timing_map_prune_ns;
    last_report_timing_publish_ns_ = timing_publish_ns;
    last_report_timing_total_ns_ = timing_total_ns;

    RCLCPP_INFO(
      this->get_logger(),
      "input buffers: imu=%zu lidar=%zu latest_imu=%.3f latest_lidar=%.3f | map: local_map_voxels=%zu | lidar_rx=%llu proc=%llu drop_overflow=%llu(win=%.2f%% total=%.2f%%) drop_stale=%llu drop_reflectivity=%llu | stage_ms_per_scan(win): prep=%.2f predict=%.2f deskew=%.2f downsample=%.2f update=%.2f map_tf=%.2f map_insert=%.2f map_prune=%.2f map=%.2f publish=%.2f total=%.2f",
      imu_size,
      lidar_size,
      stampToSec(latest_imu_stamp_),
      stampToSec(latest_lidar_stamp_),
      local_map_voxels,
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
      avg_map_transform_ms,
      avg_map_insert_ms,
      avg_map_prune_ms,
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

  builtin_interfaces::msg::Time secToStamp(double time_s) const
  {
    if (!std::isfinite(time_s) || time_s <= 0.0) {
      return builtin_interfaces::msg::Time {};
    }

    const double sec_floor = std::floor(time_s);
    builtin_interfaces::msg::Time stamp;
    stamp.sec = static_cast<std::int32_t>(sec_floor);
    stamp.nanosec = static_cast<std::uint32_t>(
      std::llround((time_s - sec_floor) * 1e9));
    if (stamp.nanosec >= 1000000000U) {
      ++stamp.sec;
      stamp.nanosec -= 1000000000U;
    }
    return stamp;
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
  int pdr_buffer_size_;
  double pdr_sample_rate_hz_;
  double pdr_accel_norm_lpf_cutoff_hz_;
  std::string pdr_output_dir_;
  std::string time_output_dir_;
  std::string pdr_run_tag_;
  std::string time_run_tag_;
  bool pdr_peak_enable_;
  int pdr_peak_buffer_size_;
  double pdr_peak_min_height_;
  double pdr_peak_min_interval_s_;
  double pdr_peak_nms_interval_s_;
  bool pdr_ampd_enable_;
  int pdr_ampd_smin_;
  int pdr_ampd_smax_;
  double pdr_ampd_pass_ratio_;
  double last_pdr_coarse_peak_time_s_ = -1.0;
  bool has_pending_nms_peak_ = false;
  iekf_lio::PdrPeakEvent pending_nms_peak_;
  bool has_last_committed_peak_ = false;
  iekf_lio::PdrPeakEvent last_committed_peak_;
  int pdr_step_buffer_size_;
  double pdr_step_min_interval_s_;
  double pdr_step_max_interval_s_;
  double pdr_step_min_peak_valley_diff_;
  double last_pdr_downsample_time_s_ = -1.0;
  std::uint64_t next_pdr_sample_id_ = 0;
  bool updater_enable_;
  int updater_max_iterations_;
  double updater_max_corr_dist_;
  int updater_plane_k_;
  double updater_plane_eigen_ratio_;
  double updater_sigma_plane_;
  double updater_max_abs_residual_;
  int updater_max_update_points_;
  int updater_min_correspondences_;
  double updater_convergence_delta_pos_m_;
  double updater_convergence_delta_rot_rad_;
  bool updater_degeneracy_enable_;
  bool updater_degeneracy_log_only_;
  double updater_degeneracy_trigger_ratio_;
  double updater_degeneracy_min_eigenvalue_threshold_;
  int updater_degeneracy_poor_match_min_correspondences_;
  double updater_degeneracy_poor_match_rmse_threshold_;
  double updater_degeneracy_relative_scale_;
  double updater_degeneracy_abs_floor_;
  double updater_degeneracy_min_weight_;
  bool deskew_enable_;
  int deskew_max_input_points_;
  bool downsample_enable_;
  double downsample_update_leaf_size_;
  double downsample_map_leaf_size_;
  double map_voxel_size_;
  double map_block_size_;
  int map_history_max_blocks_;
  bool map_history_enable_;
  double map_history_radius_xy_;
  double map_history_half_height_;
  bool keyframe_enable_;
  double keyframe_translation_thresh_m_;
  double keyframe_rotation_thresh_deg_;
  double keyframe_time_thresh_s_;
  int keyframe_backend_max_cached_;
  int keyframe_backend_max_archive_cached_;
  int local_map_active_recent_count_;
  double local_map_active_radius_m_;
  int local_map_active_max_keyframes_;
  bool loop_enable_;
  bool backend_optimize_enable_;
  int loop_min_keyframes_before_loop_;
  int loop_exclude_recent_keyframes_;
  int loop_sc_num_rings_;
  int loop_sc_num_sectors_;
  double loop_sc_max_radius_m_;
  int loop_sc_max_entries_;
  int loop_sc_num_candidates_;
  double loop_sc_distance_threshold_;
  bool loop_reg_enable_;
  double loop_reg_voxel_leaf_size_;
  double loop_reg_max_corr_distance_;
  int loop_reg_max_iterations_;
  double loop_reg_fitness_threshold_;
  double loop_factor_translation_sigma_;
  double loop_factor_rotation_sigma_rad_;
  std::string odom_topic_;
  std::string path_topic_;
  std::string backend_path_topic_;
  std::string points_world_topic_;
  std::string imu_prediction_topic_;
  int imu_prediction_qos_depth_;
  std::uint64_t imu_prediction_pub_count_ = 0;
  std::string tum_output_path_;
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
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr backend_path_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr points_world_pub_;
  rclcpp::Publisher<iekf_lio::msg::ImuPredictionState>::SharedPtr imu_prediction_pub_;
  std::ofstream tum_traj_ofs_;
  std::ofstream pdr_signal_ofs_;
  std::ofstream pdr_coarse_peak_ofs_;
  std::ofstream pdr_peak_debug_ofs_;
  std::ofstream pdr_step_candidate_ofs_;
  std::ofstream pdr_step_event_ofs_;
  std::ofstream time_scan_ofs_;
  std::ofstream degeneracy_iter_log_ofs_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::TimerBase::SharedPtr status_timer_;
  nav_msgs::msg::Path path_msg_;
  nav_msgs::msg::Path backend_path_msg_;

  std::deque<sensor_msgs::msg::Imu::SharedPtr> imu_buffer_;
  std::deque<LidarMsg::SharedPtr> lidar_buffer_;
  std::deque<iekf_lio::PdrPreprocessedSample> pdr_buffer_;
  std::deque<iekf_lio::PdrPeakEvent> pdr_coarse_peak_buffer_;
  std::deque<iekf_lio::PdrPeakEvent> pdr_peak_debug_buffer_;
  std::deque<iekf_lio::PdrStepCandidate> pdr_step_buffer_;
  std::deque<iekf_lio::PdrStepEvent> pdr_step_event_buffer_;
  std::mutex imu_mutex_;
  std::mutex lidar_mutex_;
  std::condition_variable data_cv_;
  std::condition_variable keyframe_cv_;
  std::mutex processing_mutex_;
  std::mutex keyframe_mutex_;
  mutable std::mutex backend_keyframes_mutex_;
  mutable std::mutex backend_path_mutex_;
  std::thread processing_thread_;
  std::thread backend_thread_;
  std::atomic<bool> stop_requested_ {false};
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
  std::atomic<std::uint64_t> timing_map_transform_ns_total_ {0};
  std::atomic<std::uint64_t> timing_map_insert_ns_total_ {0};
  std::atomic<std::uint64_t> timing_map_prune_ns_total_ {0};
  std::atomic<std::uint64_t> timing_publish_ns_total_ {0};
  std::atomic<std::uint64_t> timing_total_ns_total_ {0};
  std::atomic<std::size_t> last_history_blocks_ {0};
  std::uint64_t last_report_lidar_rx_ = 0;
  std::uint64_t last_report_lidar_processed_ = 0;
  std::uint64_t last_report_lidar_drop_overflow_ = 0;
  std::uint64_t last_report_timing_prep_ns_ = 0;
  std::uint64_t last_report_timing_predict_ns_ = 0;
  std::uint64_t last_report_timing_deskew_ns_ = 0;
  std::uint64_t last_report_timing_downsample_ns_ = 0;
  std::uint64_t last_report_timing_update_ns_ = 0;
  std::uint64_t last_report_timing_map_ns_ = 0;
  std::uint64_t last_report_timing_map_transform_ns_ = 0;
  std::uint64_t last_report_timing_map_insert_ns_ = 0;
  std::uint64_t last_report_timing_map_prune_ns_ = 0;
  std::uint64_t last_report_timing_publish_ns_ = 0;
  std::uint64_t last_report_timing_total_ns_ = 0;
  iekf_lio::ImuIntegrator imu_integrator_;
  iekf_lio::ImuStaticInitializer imu_initializer_;
  iekf_lio::IekfPredictor iekf_predictor_;
  iekf_lio::IekfUpdater iekf_updater_;
  iekf_lio::IekfState18 iekf_state_;
  iekf_lio::CloudDeskewer deskewer_;
  iekf_lio::VoxelMap voxel_map_;
  iekf_lio::BackendFactorGraph backend_factor_graph_;
  iekf_lio::KeyframeManager keyframe_manager_;
  iekf_lio::LoopRegistration loop_registration_;
  iekf_lio::ScanContextManager scan_context_manager_;
  std::deque<KeyframeData> keyframe_queue_;
  std::deque<KeyframeData> frontend_keyframe_history_;
  std::deque<KeyframeData> local_map_active_keyframes_;
  std::unordered_set<std::size_t> local_map_active_keyframe_ids_;
  std::unordered_map<std::size_t, std::vector<iekf_lio::VoxelMap::VoxelContribution>>
  keyframe_voxel_cache_;
  std::deque<KeyframeData> backend_keyframes_;
  std::deque<std::size_t> backend_keyframe_archive_order_;
  std::unordered_map<std::size_t, KeyframeData> backend_keyframe_archive_;
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
