cmake_minimum_required(VERSION 3.8)
project(iekf_lio)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(livox_ros_driver2 REQUIRED)

add_executable(iekf_slam_node
  src/iekf_slam_node.cpp
  src/imu/imu_integrator.cpp
  src/iekf/iekf_predictor.cpp
)
target_include_directories(iekf_slam_node PUBLIC include)
ament_target_dependencies(iekf_slam_node
  rclcpp
  sensor_msgs
  livox_ros_driver2
)
target_link_libraries(iekf_slam_node Eigen3::Eigen)

add_executable(test_build_node src/test_build_node.cpp)
ament_target_dependencies(test_build_node rclcpp)

install(TARGETS
  iekf_slam_node
  test_build_node
  DESTINATION lib/${PROJECT_NAME}
)

add_executable(livox_sub_test src/livox_sub_test.cpp)
ament_target_dependencies(livox_sub_test rclcpp livox_ros_driver2)

install(TARGETS
  livox_sub_test
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
