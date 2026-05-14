// generated from rosidl_typesupport_fastrtps_cpp/resource/idl__rosidl_typesupport_fastrtps_cpp.hpp.em
// with input from iekf_lio:msg/ImuPredictionState.idl
// generated code does not contain a copyright notice

#ifndef IEKF_LIO__MSG__DETAIL__IMU_PREDICTION_STATE__ROSIDL_TYPESUPPORT_FASTRTPS_CPP_HPP_
#define IEKF_LIO__MSG__DETAIL__IMU_PREDICTION_STATE__ROSIDL_TYPESUPPORT_FASTRTPS_CPP_HPP_

#include <cstddef>
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_interface/macros.h"
#include "iekf_lio/msg/rosidl_typesupport_fastrtps_cpp__visibility_control.h"
#include "iekf_lio/msg/detail/imu_prediction_state__struct.hpp"

#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-parameter"
# ifdef __clang__
#  pragma clang diagnostic ignored "-Wdeprecated-register"
#  pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
# endif
#endif
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif

#include "fastcdr/Cdr.h"

namespace iekf_lio
{

namespace msg
{

namespace typesupport_fastrtps_cpp
{

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_iekf_lio
cdr_serialize(
  const iekf_lio::msg::ImuPredictionState & ros_message,
  eprosima::fastcdr::Cdr & cdr);

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_iekf_lio
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  iekf_lio::msg::ImuPredictionState & ros_message);

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_iekf_lio
get_serialized_size(
  const iekf_lio::msg::ImuPredictionState & ros_message,
  size_t current_alignment);

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_iekf_lio
max_serialized_size_ImuPredictionState(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_iekf_lio
cdr_serialize_key(
  const iekf_lio::msg::ImuPredictionState & ros_message,
  eprosima::fastcdr::Cdr &);

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_iekf_lio
get_serialized_size_key(
  const iekf_lio::msg::ImuPredictionState & ros_message,
  size_t current_alignment);

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_iekf_lio
max_serialized_size_key_ImuPredictionState(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

}  // namespace typesupport_fastrtps_cpp

}  // namespace msg

}  // namespace iekf_lio

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_iekf_lio
const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, iekf_lio, msg, ImuPredictionState)();

#ifdef __cplusplus
}
#endif

#endif  // IEKF_LIO__MSG__DETAIL__IMU_PREDICTION_STATE__ROSIDL_TYPESUPPORT_FASTRTPS_CPP_HPP_
