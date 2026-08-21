cmake_minimum_required(VERSION 3.8)
project(iekf_lio)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rclcpp REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED COMPONENTS common kdtree filters)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(livox_ros_driver2 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Sophus QUIET)
find_package(GTSAM QUIET CONFIG PATHS /usr/local/lib/cmake/GTSAM)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ImuPredictionState.msg"
  DEPENDENCIES builtin_interfaces std_msgs
)

add_executable(iekf_slam_node
  src/iekf_slam_node.cpp
  src/imu/imu_integrator.cpp
  src/imu/noise_model.cpp
  src/lidar/cloud_deskewer.cpp
  src/mapping/backend_factor_graph.cpp
  src/mapping/keyframe_manager.cpp
  third_party/ikd_tree/ikd_Tree.cpp
  src/mapping/loop_registration.cpp
  src/mapping/scan_context_manager.cpp
  src/mapping/tree_point_map.cpp
  src/iekf/iekf_predictor.cpp
  src/iekf/iekf_updater.cpp
)
target_include_directories(iekf_slam_node PUBLIC include)
target_include_directories(iekf_slam_node PUBLIC third_party/ikd_tree)
target_include_directories(iekf_slam_node SYSTEM PUBLIC ${PCL_INCLUDE_DIRS})
ament_target_dependencies(iekf_slam_node
  rclcpp
  sensor_msgs
  std_msgs
  nav_msgs
  tf2
  tf2_ros
  geometry_msgs
  pcl_conversions
  livox_ros_driver2
)
target_link_libraries(iekf_slam_node Eigen3::Eigen ${PCL_LIBRARIES})
target_link_libraries(iekf_slam_node OpenMP::OpenMP_CXX)
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(iekf_slam_node ${cpp_typesupport_target})
if(Sophus_FOUND)
  target_compile_definitions(iekf_slam_node PRIVATE IEKF_LIO_USE_SOPHUS=1)
  if(TARGET Sophus::Sophus)
    target_link_libraries(iekf_slam_node Sophus::Sophus)
  endif()
endif()
if(GTSAM_FOUND)
  target_compile_definitions(iekf_slam_node PRIVATE IEKF_LIO_USE_GTSAM=1)
  if(TARGET GTSAM::gtsam)
    target_link_libraries(iekf_slam_node GTSAM::gtsam)
  elseif(TARGET gtsam)
    target_link_libraries(iekf_slam_node gtsam)
  endif()
endif()

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

install(DIRECTORY
  launch
  config
  rviz
  msg
  DESTINATION share/${PROJECT_NAME}
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

ament_export_dependencies(rosidl_default_runtime)

ament_package()
