#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace iekf_lio
{

struct PointXYZIRTL
{
  float x = 0.0F;
  float y = 0.0F;
  float z = 0.0F;
  std::uint8_t reflectivity = 0U;
  std::uint8_t tag = 0U;
  std::uint8_t line = 0U;
  double relative_time_s = 0.0;
};

struct LidarScan
{
  std::string frame_id;
  double scan_begin_time_s = 0.0;
  double scan_end_time_s = 0.0;
  std::uint64_t timebase_ns = 0ULL;
  std::vector<PointXYZIRTL> points;
};

}  // namespace iekf_lio
