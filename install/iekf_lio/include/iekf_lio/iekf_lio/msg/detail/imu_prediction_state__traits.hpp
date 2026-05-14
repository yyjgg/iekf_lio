// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from iekf_lio:msg/ImuPredictionState.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "iekf_lio/msg/imu_prediction_state.hpp"


#ifndef IEKF_LIO__MSG__DETAIL__IMU_PREDICTION_STATE__TRAITS_HPP_
#define IEKF_LIO__MSG__DETAIL__IMU_PREDICTION_STATE__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "iekf_lio/msg/detail/imu_prediction_state__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'stamp'
#include "builtin_interfaces/msg/detail/time__traits.hpp"

namespace iekf_lio
{

namespace msg
{

inline void to_flow_style_yaml(
  const ImuPredictionState & msg,
  std::ostream & out)
{
  out << "{";
  // member: stamp
  {
    out << "stamp: ";
    to_flow_style_yaml(msg.stamp, out);
    out << ", ";
  }

  // member: gyro_mid_unbiased
  {
    if (msg.gyro_mid_unbiased.size() == 0) {
      out << "gyro_mid_unbiased: []";
    } else {
      out << "gyro_mid_unbiased: [";
      size_t pending_items = msg.gyro_mid_unbiased.size();
      for (auto item : msg.gyro_mid_unbiased) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: linear_accel_mid_w
  {
    if (msg.linear_accel_mid_w.size() == 0) {
      out << "linear_accel_mid_w: []";
    } else {
      out << "linear_accel_mid_w: [";
      size_t pending_items = msg.linear_accel_mid_w.size();
      for (auto item : msg.linear_accel_mid_w) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: yaw_mid_rad
  {
    out << "yaw_mid_rad: ";
    rosidl_generator_traits::value_to_yaml(msg.yaw_mid_rad, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const ImuPredictionState & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: stamp
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "stamp:\n";
    to_block_style_yaml(msg.stamp, out, indentation + 2);
  }

  // member: gyro_mid_unbiased
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.gyro_mid_unbiased.size() == 0) {
      out << "gyro_mid_unbiased: []\n";
    } else {
      out << "gyro_mid_unbiased:\n";
      for (auto item : msg.gyro_mid_unbiased) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: linear_accel_mid_w
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.linear_accel_mid_w.size() == 0) {
      out << "linear_accel_mid_w: []\n";
    } else {
      out << "linear_accel_mid_w:\n";
      for (auto item : msg.linear_accel_mid_w) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: yaw_mid_rad
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "yaw_mid_rad: ";
    rosidl_generator_traits::value_to_yaml(msg.yaw_mid_rad, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const ImuPredictionState & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace iekf_lio

namespace rosidl_generator_traits
{

[[deprecated("use iekf_lio::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const iekf_lio::msg::ImuPredictionState & msg,
  std::ostream & out, size_t indentation = 0)
{
  iekf_lio::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use iekf_lio::msg::to_yaml() instead")]]
inline std::string to_yaml(const iekf_lio::msg::ImuPredictionState & msg)
{
  return iekf_lio::msg::to_yaml(msg);
}

template<>
inline const char * data_type<iekf_lio::msg::ImuPredictionState>()
{
  return "iekf_lio::msg::ImuPredictionState";
}

template<>
inline const char * name<iekf_lio::msg::ImuPredictionState>()
{
  return "iekf_lio/msg/ImuPredictionState";
}

template<>
struct has_fixed_size<iekf_lio::msg::ImuPredictionState>
  : std::integral_constant<bool, has_fixed_size<builtin_interfaces::msg::Time>::value> {};

template<>
struct has_bounded_size<iekf_lio::msg::ImuPredictionState>
  : std::integral_constant<bool, has_bounded_size<builtin_interfaces::msg::Time>::value> {};

template<>
struct is_message<iekf_lio::msg::ImuPredictionState>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // IEKF_LIO__MSG__DETAIL__IMU_PREDICTION_STATE__TRAITS_HPP_
