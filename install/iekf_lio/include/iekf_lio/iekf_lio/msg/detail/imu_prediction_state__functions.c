// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from iekf_lio:msg/ImuPredictionState.idl
// generated code does not contain a copyright notice
#include "iekf_lio/msg/detail/imu_prediction_state__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `stamp`
#include "builtin_interfaces/msg/detail/time__functions.h"

bool
iekf_lio__msg__ImuPredictionState__init(iekf_lio__msg__ImuPredictionState * msg)
{
  if (!msg) {
    return false;
  }
  // stamp
  if (!builtin_interfaces__msg__Time__init(&msg->stamp)) {
    iekf_lio__msg__ImuPredictionState__fini(msg);
    return false;
  }
  // gyro_mid_unbiased
  // linear_accel_mid_w
  // yaw_mid_rad
  return true;
}

void
iekf_lio__msg__ImuPredictionState__fini(iekf_lio__msg__ImuPredictionState * msg)
{
  if (!msg) {
    return;
  }
  // stamp
  builtin_interfaces__msg__Time__fini(&msg->stamp);
  // gyro_mid_unbiased
  // linear_accel_mid_w
  // yaw_mid_rad
}

bool
iekf_lio__msg__ImuPredictionState__are_equal(const iekf_lio__msg__ImuPredictionState * lhs, const iekf_lio__msg__ImuPredictionState * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // stamp
  if (!builtin_interfaces__msg__Time__are_equal(
      &(lhs->stamp), &(rhs->stamp)))
  {
    return false;
  }
  // gyro_mid_unbiased
  for (size_t i = 0; i < 3; ++i) {
    if (lhs->gyro_mid_unbiased[i] != rhs->gyro_mid_unbiased[i]) {
      return false;
    }
  }
  // linear_accel_mid_w
  for (size_t i = 0; i < 3; ++i) {
    if (lhs->linear_accel_mid_w[i] != rhs->linear_accel_mid_w[i]) {
      return false;
    }
  }
  // yaw_mid_rad
  if (lhs->yaw_mid_rad != rhs->yaw_mid_rad) {
    return false;
  }
  return true;
}

bool
iekf_lio__msg__ImuPredictionState__copy(
  const iekf_lio__msg__ImuPredictionState * input,
  iekf_lio__msg__ImuPredictionState * output)
{
  if (!input || !output) {
    return false;
  }
  // stamp
  if (!builtin_interfaces__msg__Time__copy(
      &(input->stamp), &(output->stamp)))
  {
    return false;
  }
  // gyro_mid_unbiased
  for (size_t i = 0; i < 3; ++i) {
    output->gyro_mid_unbiased[i] = input->gyro_mid_unbiased[i];
  }
  // linear_accel_mid_w
  for (size_t i = 0; i < 3; ++i) {
    output->linear_accel_mid_w[i] = input->linear_accel_mid_w[i];
  }
  // yaw_mid_rad
  output->yaw_mid_rad = input->yaw_mid_rad;
  return true;
}

iekf_lio__msg__ImuPredictionState *
iekf_lio__msg__ImuPredictionState__create(void)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  iekf_lio__msg__ImuPredictionState * msg = (iekf_lio__msg__ImuPredictionState *)allocator.allocate(sizeof(iekf_lio__msg__ImuPredictionState), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(iekf_lio__msg__ImuPredictionState));
  bool success = iekf_lio__msg__ImuPredictionState__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
iekf_lio__msg__ImuPredictionState__destroy(iekf_lio__msg__ImuPredictionState * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    iekf_lio__msg__ImuPredictionState__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
iekf_lio__msg__ImuPredictionState__Sequence__init(iekf_lio__msg__ImuPredictionState__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  iekf_lio__msg__ImuPredictionState * data = NULL;

  if (size) {
    data = (iekf_lio__msg__ImuPredictionState *)allocator.zero_allocate(size, sizeof(iekf_lio__msg__ImuPredictionState), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = iekf_lio__msg__ImuPredictionState__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        iekf_lio__msg__ImuPredictionState__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
iekf_lio__msg__ImuPredictionState__Sequence__fini(iekf_lio__msg__ImuPredictionState__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      iekf_lio__msg__ImuPredictionState__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

iekf_lio__msg__ImuPredictionState__Sequence *
iekf_lio__msg__ImuPredictionState__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  iekf_lio__msg__ImuPredictionState__Sequence * array = (iekf_lio__msg__ImuPredictionState__Sequence *)allocator.allocate(sizeof(iekf_lio__msg__ImuPredictionState__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = iekf_lio__msg__ImuPredictionState__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
iekf_lio__msg__ImuPredictionState__Sequence__destroy(iekf_lio__msg__ImuPredictionState__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    iekf_lio__msg__ImuPredictionState__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
iekf_lio__msg__ImuPredictionState__Sequence__are_equal(const iekf_lio__msg__ImuPredictionState__Sequence * lhs, const iekf_lio__msg__ImuPredictionState__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!iekf_lio__msg__ImuPredictionState__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
iekf_lio__msg__ImuPredictionState__Sequence__copy(
  const iekf_lio__msg__ImuPredictionState__Sequence * input,
  iekf_lio__msg__ImuPredictionState__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(iekf_lio__msg__ImuPredictionState);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    iekf_lio__msg__ImuPredictionState * data =
      (iekf_lio__msg__ImuPredictionState *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!iekf_lio__msg__ImuPredictionState__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          iekf_lio__msg__ImuPredictionState__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!iekf_lio__msg__ImuPredictionState__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
