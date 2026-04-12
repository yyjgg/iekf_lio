#pragma once

#include <Eigen/Dense>

namespace iekf_lio
{

using IekfVec3 = Eigen::Vector3d;
using IekfMat3 = Eigen::Matrix3d;
using IekfMat18 = Eigen::Matrix<double, 18, 18>;

struct IekfNominalState18
{
  IekfVec3 p_wb = IekfVec3::Zero();
  IekfVec3 v_wb = IekfVec3::Zero();
  IekfMat3 r_wb = IekfMat3::Identity();
  IekfVec3 b_g = IekfVec3::Zero();
  IekfVec3 b_a = IekfVec3::Zero();
  IekfVec3 g_w = IekfVec3(0.0, 0.0, -9.81);
};

struct IekfState18
{
  IekfNominalState18 x;
  IekfMat18 p_cov = IekfMat18::Zero();
  bool is_initialized = false;
};

}  // namespace iekf_lio
