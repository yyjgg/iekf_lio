// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from iekf_lio:msg/ImuPredictionState.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "iekf_lio/msg/imu_prediction_state.hpp"


#ifndef IEKF_LIO__MSG__DETAIL__IMU_PREDICTION_STATE__BUILDER_HPP_
#define IEKF_LIO__MSG__DETAIL__IMU_PREDICTION_STATE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "iekf_lio/msg/detail/imu_prediction_state__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace iekf_lio
{

namespace msg
{

namespace builder
{

class Init_ImuPredictionState_yaw_mid_rad
{
public:
  explicit Init_ImuPredictionState_yaw_mid_rad(::iekf_lio::msg::ImuPredictionState & msg)
  : msg_(msg)
  {}
  ::iekf_lio::msg::ImuPredictionState yaw_mid_rad(::iekf_lio::msg::ImuPredictionState::_yaw_mid_rad_type arg)
  {
    msg_.yaw_mid_rad = std::move(arg);
    return std::move(msg_);
  }

private:
  ::iekf_lio::msg::ImuPredictionState msg_;
};

class Init_ImuPredictionState_linear_accel_mid_w
{
public:
  explicit Init_ImuPredictionState_linear_accel_mid_w(::iekf_lio::msg::ImuPredictionState & msg)
  : msg_(msg)
  {}
  Init_ImuPredictionState_yaw_mid_rad linear_accel_mid_w(::iekf_lio::msg::ImuPredictionState::_linear_accel_mid_w_type arg)
  {
    msg_.linear_accel_mid_w = std::move(arg);
    return Init_ImuPredictionState_yaw_mid_rad(msg_);
  }

private:
  ::iekf_lio::msg::ImuPredictionState msg_;
};

class Init_ImuPredictionState_gyro_mid_unbiased
{
public:
  explicit Init_ImuPredictionState_gyro_mid_unbiased(::iekf_lio::msg::ImuPredictionState & msg)
  : msg_(msg)
  {}
  Init_ImuPredictionState_linear_accel_mid_w gyro_mid_unbiased(::iekf_lio::msg::ImuPredictionState::_gyro_mid_unbiased_type arg)
  {
    msg_.gyro_mid_unbiased = std::move(arg);
    return Init_ImuPredictionState_linear_accel_mid_w(msg_);
  }

private:
  ::iekf_lio::msg::ImuPredictionState msg_;
};

class Init_ImuPredictionState_stamp
{
public:
  Init_ImuPredictionState_stamp()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_ImuPredictionState_gyro_mid_unbiased stamp(::iekf_lio::msg::ImuPredictionState::_stamp_type arg)
  {
    msg_.stamp = std::move(arg);
    return Init_ImuPredictionState_gyro_mid_unbiased(msg_);
  }

private:
  ::iekf_lio::msg::ImuPredictionState msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::iekf_lio::msg::ImuPredictionState>()
{
  return iekf_lio::msg::builder::Init_ImuPredictionState_stamp();
}

}  // namespace iekf_lio

#endif  // IEKF_LIO__MSG__DETAIL__IMU_PREDICTION_STATE__BUILDER_HPP_
