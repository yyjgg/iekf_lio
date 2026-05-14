// generated from rosidl_typesupport_fastrtps_c/resource/idl__rosidl_typesupport_fastrtps_c.h.em
// with input from iekf_lio:msg/ImuPredictionState.idl
// generated code does not contain a copyright notice
#ifndef IEKF_LIO__MSG__DETAIL__IMU_PREDICTION_STATE__ROSIDL_TYPESUPPORT_FASTRTPS_C_H_
#define IEKF_LIO__MSG__DETAIL__IMU_PREDICTION_STATE__ROSIDL_TYPESUPPORT_FASTRTPS_C_H_


#include <stddef.h>
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_interface/macros.h"
#include "iekf_lio/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "iekf_lio/msg/detail/imu_prediction_state__struct.h"
#include "fastcdr/Cdr.h"

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_iekf_lio
bool cdr_serialize_iekf_lio__msg__ImuPredictionState(
  const iekf_lio__msg__ImuPredictionState * ros_message,
  eprosima::fastcdr::Cdr & cdr);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_iekf_lio
bool cdr_deserialize_iekf_lio__msg__ImuPredictionState(
  eprosima::fastcdr::Cdr &,
  iekf_lio__msg__ImuPredictionState * ros_message);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_iekf_lio
size_t get_serialized_size_iekf_lio__msg__ImuPredictionState(
  const void * untyped_ros_message,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_iekf_lio
size_t max_serialized_size_iekf_lio__msg__ImuPredictionState(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_iekf_lio
bool cdr_serialize_key_iekf_lio__msg__ImuPredictionState(
  const iekf_lio__msg__ImuPredictionState * ros_message,
  eprosima::fastcdr::Cdr & cdr);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_iekf_lio
size_t get_serialized_size_key_iekf_lio__msg__ImuPredictionState(
  const void * untyped_ros_message,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_iekf_lio
size_t max_serialized_size_key_iekf_lio__msg__ImuPredictionState(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_iekf_lio
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, iekf_lio, msg, ImuPredictionState)();

#ifdef __cplusplus
}
#endif

#endif  // IEKF_LIO__MSG__DETAIL__IMU_PREDICTION_STATE__ROSIDL_TYPESUPPORT_FASTRTPS_C_H_
