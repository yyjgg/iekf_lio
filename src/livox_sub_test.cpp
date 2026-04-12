#include <cinttypes>
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "livox_ros_driver2/msg/custom_msg.hpp"

class LivoxSubTest : public rclcpp::Node
{
public:
  LivoxSubTest() : Node("livox_sub_test")
  {
    topic_name_ = this->declare_parameter<std::string>("topic", "/livox/lidar");

    sub_ = this->create_subscription<livox_ros_driver2::msg::CustomMsg>(
      topic_name_,
      rclcpp::SensorDataQoS(),
      std::bind(&LivoxSubTest::callback, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "Subscribing to %s", topic_name_.c_str());
  }

private:
  void callback(const livox_ros_driver2::msg::CustomMsg::SharedPtr msg)
  {
    const auto & stamp = msg->header.stamp;
    const std::size_t n = msg->points.size();

    RCLCPP_INFO(
      this->get_logger(),
      "recv livox msg | stamp=%.9f | timebase=%" PRIu64 " | point_num=%u | points.size=%zu",
      rclcpp::Time(stamp).seconds(),
      msg->timebase,
      msg->point_num,
      n);

    if (n == 0) {
      RCLCPP_WARN(this->get_logger(), "points is empty");
      return;
    }

    const auto & first_pt = msg->points.front();
    const auto & last_pt  = msg->points.back();

    RCLCPP_INFO(
      this->get_logger(),
      "first_pt: offset=%u xyz=(%.3f, %.3f, %.3f) refl=%u line=%u",
      first_pt.offset_time,
      first_pt.x, first_pt.y, first_pt.z,
      first_pt.reflectivity,
      first_pt.line);

    RCLCPP_INFO(
      this->get_logger(),
      "last_pt : offset=%u xyz=(%.3f, %.3f, %.3f) refl=%u line=%u",
      last_pt.offset_time,
      last_pt.x, last_pt.y, last_pt.z,
      last_pt.reflectivity,
      last_pt.line);
  }

  std::string topic_name_;
  rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr sub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LivoxSubTest>());
  rclcpp::shutdown();
  return 0;
}