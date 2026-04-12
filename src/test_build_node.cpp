#include <chrono>
#include <memory>

#include "rclcpp/rclcpp.hpp"

using namespace std::chrono_literals;

class TestBuildNode : public rclcpp::Node
{
public:
  TestBuildNode()
  : Node("test_build_node"), count_(0)
  {
    RCLCPP_INFO(this->get_logger(), "test_build_node started successfully.");

    timer_ = this->create_wall_timer(
      1s, [this]()
      {
        ++count_;
        RCLCPP_INFO(this->get_logger(), "heartbeat %d", count_);
      });
  }

private:
  rclcpp::TimerBase::SharedPtr timer_;
  int count_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TestBuildNode>());
  rclcpp::shutdown();
  return 0;
}
