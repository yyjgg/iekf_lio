#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "builtin_interfaces/msg/time.hpp"
#include "iekf/iekf_predictor.hpp"
#include "imu/imu_integrator.hpp"
#include "imu/imu_types.hpp"
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
        // TODO: Extract IMU messages within [scan_begin_time, scan_end_time].
        for (const auto & imu_msg : imu_buffer_) {
          const auto & t = imu_msg->header.stamp;
          if (isTimeLE(scan_begin_time, t) && isTimeLE(t, scan_end_time)) {
            imu_slice.push_back(imu_msg);
          }
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

    const iekf_lio::LidarScan lidar_scan_internal =
      convertLidarScanToInternal(*filtered_lidar_msg, scan_begin_time, scan_end_time);
    const iekf_lio::ImuTrack imu_track_internal = convertImuSliceToInternal(imu_slice);
    const iekf_lio::ImuPreintegrationResult imu_preint =
      imu_integrator_.integrateMidpoint(imu_track_internal);
    iekf_predictor_.predictWithMidpoint(imu_track_internal, iekf_state_);

    RCLCPP_INFO(
      this->get_logger(),
      "Scheduled one scan: [%.3f, %.3f], imu_in_window=%zu, points=%zu->%zu, internal_points=%zu, preint_dt=%.4f dv=(%.3f,%.3f,%.3f), pred_p=(%.3f,%.3f,%.3f)",
      stampToSec(scan_begin_time),
      stampToSec(scan_end_time),
      imu_track_internal.size(),
      raw_point_count,
      filtered_point_count,
      lidar_scan_internal.points.size(),
      imu_preint.delta_t_s,
      imu_preint.delta_v.x(),
      imu_preint.delta_v.y(),
      imu_preint.delta_v.z(),
      iekf_state_.x.p_wb.x(),
      iekf_state_.x.p_wb.y(),
      iekf_state_.x.p_wb.z());
    return true;
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
  iekf_lio::IekfPredictor iekf_predictor_;
  iekf_lio::IekfState18 iekf_state_;

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
