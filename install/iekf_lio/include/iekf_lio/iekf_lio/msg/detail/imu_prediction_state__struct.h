// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from iekf_lio:msg/ImuPredictionState.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "iekf_lio/msg/imu_prediction_state.h"


#ifndef IEKF_LIO__MSG__DETAIL__IMU_PREDICTION_STATE__STRUCT_H_
#define IEKF_LIO__MSG__DETAIL__IMU_PREDICTION_STATE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Constants defined in the message

// Include directives for member types
// Member 'stamp'
#include "builtin_interfaces/msg/detail/time__struct.h"

/// Struct defined in msg/ImuPredictionState in the package iekf_lio.
typedef struct iekf_lio__msg__ImuPredictionState
{
  builtin_interfaces__msg__Time stamp;
  double gyro_mid_unbiased[3];
  double linear_accel_mid_w[3];
  double yaw_mid_rad;
} iekf_lio__msg__ImuPredictionState;

// Struct for a sequence of iekf_lio__msg__ImuPredictionState.
typedef struct iekf_lio__msg__ImuPredictionState__Sequence
{
  iekf_lio__msg__ImuPredictionState * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} iekf_lio__msg__ImuPredictionState__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // IEKF_LIO__MSG__DETAIL__IMU_PREDICTION_STATE__STRUCT_H_
