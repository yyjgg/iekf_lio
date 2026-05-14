// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from iekf_lio:msg/ImuPredictionState.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "iekf_lio/msg/detail/imu_prediction_state__functions.h"
#include "iekf_lio/msg/detail/imu_prediction_state__struct.hpp"
#include "rosidl_typesupport_introspection_cpp/field_types.hpp"
#include "rosidl_typesupport_introspection_cpp/identifier.hpp"
#include "rosidl_typesupport_introspection_cpp/message_introspection.hpp"
#include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_introspection_cpp/visibility_control.h"

namespace iekf_lio
{

namespace msg
{

namespace rosidl_typesupport_introspection_cpp
{

void ImuPredictionState_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) iekf_lio::msg::ImuPredictionState(_init);
}

void ImuPredictionState_fini_function(void * message_memory)
{
  auto typed_message = static_cast<iekf_lio::msg::ImuPredictionState *>(message_memory);
  typed_message->~ImuPredictionState();
}

size_t size_function__ImuPredictionState__gyro_mid_unbiased(const void * untyped_member)
{
  (void)untyped_member;
  return 3;
}

const void * get_const_function__ImuPredictionState__gyro_mid_unbiased(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 3> *>(untyped_member);
  return &member[index];
}

void * get_function__ImuPredictionState__gyro_mid_unbiased(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 3> *>(untyped_member);
  return &member[index];
}

void fetch_function__ImuPredictionState__gyro_mid_unbiased(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__ImuPredictionState__gyro_mid_unbiased(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__ImuPredictionState__gyro_mid_unbiased(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__ImuPredictionState__gyro_mid_unbiased(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

size_t size_function__ImuPredictionState__linear_accel_mid_w(const void * untyped_member)
{
  (void)untyped_member;
  return 3;
}

const void * get_const_function__ImuPredictionState__linear_accel_mid_w(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 3> *>(untyped_member);
  return &member[index];
}

void * get_function__ImuPredictionState__linear_accel_mid_w(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 3> *>(untyped_member);
  return &member[index];
}

void fetch_function__ImuPredictionState__linear_accel_mid_w(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__ImuPredictionState__linear_accel_mid_w(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__ImuPredictionState__linear_accel_mid_w(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__ImuPredictionState__linear_accel_mid_w(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember ImuPredictionState_message_member_array[4] = {
  {
    "stamp",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<builtin_interfaces::msg::Time>(),  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(iekf_lio::msg::ImuPredictionState, stamp),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "gyro_mid_unbiased",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is key
    true,  // is array
    3,  // array size
    false,  // is upper bound
    offsetof(iekf_lio::msg::ImuPredictionState, gyro_mid_unbiased),  // bytes offset in struct
    nullptr,  // default value
    size_function__ImuPredictionState__gyro_mid_unbiased,  // size() function pointer
    get_const_function__ImuPredictionState__gyro_mid_unbiased,  // get_const(index) function pointer
    get_function__ImuPredictionState__gyro_mid_unbiased,  // get(index) function pointer
    fetch_function__ImuPredictionState__gyro_mid_unbiased,  // fetch(index, &value) function pointer
    assign_function__ImuPredictionState__gyro_mid_unbiased,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "linear_accel_mid_w",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is key
    true,  // is array
    3,  // array size
    false,  // is upper bound
    offsetof(iekf_lio::msg::ImuPredictionState, linear_accel_mid_w),  // bytes offset in struct
    nullptr,  // default value
    size_function__ImuPredictionState__linear_accel_mid_w,  // size() function pointer
    get_const_function__ImuPredictionState__linear_accel_mid_w,  // get_const(index) function pointer
    get_function__ImuPredictionState__linear_accel_mid_w,  // get(index) function pointer
    fetch_function__ImuPredictionState__linear_accel_mid_w,  // fetch(index, &value) function pointer
    assign_function__ImuPredictionState__linear_accel_mid_w,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "yaw_mid_rad",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(iekf_lio::msg::ImuPredictionState, yaw_mid_rad),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers ImuPredictionState_message_members = {
  "iekf_lio::msg",  // message namespace
  "ImuPredictionState",  // message name
  4,  // number of fields
  sizeof(iekf_lio::msg::ImuPredictionState),
  false,  // has_any_key_member_
  ImuPredictionState_message_member_array,  // message members
  ImuPredictionState_init_function,  // function to initialize message memory (memory has to be allocated)
  ImuPredictionState_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t ImuPredictionState_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &ImuPredictionState_message_members,
  get_message_typesupport_handle_function,
  &iekf_lio__msg__ImuPredictionState__get_type_hash,
  &iekf_lio__msg__ImuPredictionState__get_type_description,
  &iekf_lio__msg__ImuPredictionState__get_type_description_sources,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace msg

}  // namespace iekf_lio


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<iekf_lio::msg::ImuPredictionState>()
{
  return &::iekf_lio::msg::rosidl_typesupport_introspection_cpp::ImuPredictionState_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, iekf_lio, msg, ImuPredictionState)() {
  return &::iekf_lio::msg::rosidl_typesupport_introspection_cpp::ImuPredictionState_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
