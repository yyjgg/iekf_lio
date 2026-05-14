// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from iekf_lio:msg/ImuPredictionState.idl
// generated code does not contain a copyright notice

#include "iekf_lio/msg/detail/imu_prediction_state__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_iekf_lio
const rosidl_type_hash_t *
iekf_lio__msg__ImuPredictionState__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0xe4, 0x99, 0xbd, 0x16, 0xcc, 0xb1, 0x07, 0x71,
      0xec, 0x99, 0x48, 0x6c, 0xe3, 0xad, 0xe2, 0xd3,
      0xa4, 0x69, 0xc1, 0xe4, 0x82, 0x90, 0xbc, 0xca,
      0x1d, 0xcf, 0x66, 0xde, 0xbd, 0x2d, 0xdb, 0x5d,
    }};
  return &hash;
}

#include <assert.h>
#include <string.h>

// Include directives for referenced types
#include "builtin_interfaces/msg/detail/time__functions.h"

// Hashes for external referenced types
#ifndef NDEBUG
static const rosidl_type_hash_t builtin_interfaces__msg__Time__EXPECTED_HASH = {1, {
    0xb1, 0x06, 0x23, 0x5e, 0x25, 0xa4, 0xc5, 0xed,
    0x35, 0x09, 0x8a, 0xa0, 0xa6, 0x1a, 0x3e, 0xe9,
    0xc9, 0xb1, 0x8d, 0x19, 0x7f, 0x39, 0x8b, 0x0e,
    0x42, 0x06, 0xce, 0xa9, 0xac, 0xf9, 0xc1, 0x97,
  }};
#endif

static char iekf_lio__msg__ImuPredictionState__TYPE_NAME[] = "iekf_lio/msg/ImuPredictionState";
static char builtin_interfaces__msg__Time__TYPE_NAME[] = "builtin_interfaces/msg/Time";

// Define type names, field names, and default values
static char iekf_lio__msg__ImuPredictionState__FIELD_NAME__stamp[] = "stamp";
static char iekf_lio__msg__ImuPredictionState__FIELD_NAME__gyro_mid_unbiased[] = "gyro_mid_unbiased";
static char iekf_lio__msg__ImuPredictionState__FIELD_NAME__linear_accel_mid_w[] = "linear_accel_mid_w";
static char iekf_lio__msg__ImuPredictionState__FIELD_NAME__yaw_mid_rad[] = "yaw_mid_rad";

static rosidl_runtime_c__type_description__Field iekf_lio__msg__ImuPredictionState__FIELDS[] = {
  {
    {iekf_lio__msg__ImuPredictionState__FIELD_NAME__stamp, 5, 5},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {builtin_interfaces__msg__Time__TYPE_NAME, 27, 27},
    },
    {NULL, 0, 0},
  },
  {
    {iekf_lio__msg__ImuPredictionState__FIELD_NAME__gyro_mid_unbiased, 17, 17},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_DOUBLE_ARRAY,
      3,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {iekf_lio__msg__ImuPredictionState__FIELD_NAME__linear_accel_mid_w, 18, 18},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_DOUBLE_ARRAY,
      3,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {iekf_lio__msg__ImuPredictionState__FIELD_NAME__yaw_mid_rad, 11, 11},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_DOUBLE,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
};

static rosidl_runtime_c__type_description__IndividualTypeDescription iekf_lio__msg__ImuPredictionState__REFERENCED_TYPE_DESCRIPTIONS[] = {
  {
    {builtin_interfaces__msg__Time__TYPE_NAME, 27, 27},
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
iekf_lio__msg__ImuPredictionState__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {iekf_lio__msg__ImuPredictionState__TYPE_NAME, 31, 31},
      {iekf_lio__msg__ImuPredictionState__FIELDS, 4, 4},
    },
    {iekf_lio__msg__ImuPredictionState__REFERENCED_TYPE_DESCRIPTIONS, 1, 1},
  };
  if (!constructed) {
    assert(0 == memcmp(&builtin_interfaces__msg__Time__EXPECTED_HASH, builtin_interfaces__msg__Time__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[0].fields = builtin_interfaces__msg__Time__get_type_description(NULL)->type_description.fields;
    constructed = true;
  }
  return &description;
}

static char toplevel_type_raw_source[] =
  "builtin_interfaces/Time stamp\n"
  "float64[3] gyro_mid_unbiased\n"
  "float64[3] linear_accel_mid_w\n"
  "float64 yaw_mid_rad";

static char msg_encoding[] = "msg";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
iekf_lio__msg__ImuPredictionState__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {iekf_lio__msg__ImuPredictionState__TYPE_NAME, 31, 31},
    {msg_encoding, 3, 3},
    {toplevel_type_raw_source, 109, 109},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
iekf_lio__msg__ImuPredictionState__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[2];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 2, 2};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *iekf_lio__msg__ImuPredictionState__get_individual_type_description_source(NULL),
    sources[1] = *builtin_interfaces__msg__Time__get_individual_type_description_source(NULL);
    constructed = true;
  }
  return &source_sequence;
}
