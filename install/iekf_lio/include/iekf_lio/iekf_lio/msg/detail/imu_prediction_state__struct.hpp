// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from iekf_lio:msg/ImuPredictionState.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "iekf_lio/msg/imu_prediction_state.hpp"


#ifndef IEKF_LIO__MSG__DETAIL__IMU_PREDICTION_STATE__STRUCT_HPP_
#define IEKF_LIO__MSG__DETAIL__IMU_PREDICTION_STATE__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'stamp'
#include "builtin_interfaces/msg/detail/time__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__iekf_lio__msg__ImuPredictionState __attribute__((deprecated))
#else
# define DEPRECATED__iekf_lio__msg__ImuPredictionState __declspec(deprecated)
#endif

namespace iekf_lio
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct ImuPredictionState_
{
  using Type = ImuPredictionState_<ContainerAllocator>;

  explicit ImuPredictionState_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : stamp(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 3>::iterator, double>(this->gyro_mid_unbiased.begin(), this->gyro_mid_unbiased.end(), 0.0);
      std::fill<typename std::array<double, 3>::iterator, double>(this->linear_accel_mid_w.begin(), this->linear_accel_mid_w.end(), 0.0);
      this->yaw_mid_rad = 0.0;
    }
  }

  explicit ImuPredictionState_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : stamp(_alloc, _init),
    gyro_mid_unbiased(_alloc),
    linear_accel_mid_w(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 3>::iterator, double>(this->gyro_mid_unbiased.begin(), this->gyro_mid_unbiased.end(), 0.0);
      std::fill<typename std::array<double, 3>::iterator, double>(this->linear_accel_mid_w.begin(), this->linear_accel_mid_w.end(), 0.0);
      this->yaw_mid_rad = 0.0;
    }
  }

  // field types and members
  using _stamp_type =
    builtin_interfaces::msg::Time_<ContainerAllocator>;
  _stamp_type stamp;
  using _gyro_mid_unbiased_type =
    std::array<double, 3>;
  _gyro_mid_unbiased_type gyro_mid_unbiased;
  using _linear_accel_mid_w_type =
    std::array<double, 3>;
  _linear_accel_mid_w_type linear_accel_mid_w;
  using _yaw_mid_rad_type =
    double;
  _yaw_mid_rad_type yaw_mid_rad;

  // setters for named parameter idiom
  Type & set__stamp(
    const builtin_interfaces::msg::Time_<ContainerAllocator> & _arg)
  {
    this->stamp = _arg;
    return *this;
  }
  Type & set__gyro_mid_unbiased(
    const std::array<double, 3> & _arg)
  {
    this->gyro_mid_unbiased = _arg;
    return *this;
  }
  Type & set__linear_accel_mid_w(
    const std::array<double, 3> & _arg)
  {
    this->linear_accel_mid_w = _arg;
    return *this;
  }
  Type & set__yaw_mid_rad(
    const double & _arg)
  {
    this->yaw_mid_rad = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    iekf_lio::msg::ImuPredictionState_<ContainerAllocator> *;
  using ConstRawPtr =
    const iekf_lio::msg::ImuPredictionState_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<iekf_lio::msg::ImuPredictionState_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<iekf_lio::msg::ImuPredictionState_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      iekf_lio::msg::ImuPredictionState_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<iekf_lio::msg::ImuPredictionState_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      iekf_lio::msg::ImuPredictionState_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<iekf_lio::msg::ImuPredictionState_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<iekf_lio::msg::ImuPredictionState_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<iekf_lio::msg::ImuPredictionState_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__iekf_lio__msg__ImuPredictionState
    std::shared_ptr<iekf_lio::msg::ImuPredictionState_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__iekf_lio__msg__ImuPredictionState
    std::shared_ptr<iekf_lio::msg::ImuPredictionState_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const ImuPredictionState_ & other) const
  {
    if (this->stamp != other.stamp) {
      return false;
    }
    if (this->gyro_mid_unbiased != other.gyro_mid_unbiased) {
      return false;
    }
    if (this->linear_accel_mid_w != other.linear_accel_mid_w) {
      return false;
    }
    if (this->yaw_mid_rad != other.yaw_mid_rad) {
      return false;
    }
    return true;
  }
  bool operator!=(const ImuPredictionState_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct ImuPredictionState_

// alias to use template instance with default allocator
using ImuPredictionState =
  iekf_lio::msg::ImuPredictionState_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace iekf_lio

#endif  // IEKF_LIO__MSG__DETAIL__IMU_PREDICTION_STATE__STRUCT_HPP_
