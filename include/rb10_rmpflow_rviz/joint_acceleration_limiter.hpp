#pragma once

#include <algorithm>
#include <cmath>

#include <Eigen/Core>

namespace rb10_rmpflow_rviz
{

using JointAccelerationVector = Eigen::Matrix<double, 6, 1>;

inline JointAccelerationVector limit_joint_acceleration(
  const JointAccelerationVector & qdd,
  double maximum_acceleration,
  bool preserve_direction)
{
  if (!qdd.allFinite() || std::isnan(maximum_acceleration)) {
    return JointAccelerationVector::Zero();
  }

  const double limit = std::max(maximum_acceleration, 0.0);
  if (std::isinf(limit)) {
    return qdd;
  }

  if (preserve_direction) {
    const double max_abs = qdd.cwiseAbs().maxCoeff();
    if (max_abs <= limit || max_abs <= 0.0) {
      return qdd;
    }
    return (limit / max_abs) * qdd;
  }

  JointAccelerationVector limited = qdd;
  for (Eigen::Index index = 0; index < limited.size(); ++index) {
    limited[index] = std::clamp(limited[index], -limit, limit);
  }
  return limited;
}

}  // namespace rb10_rmpflow_rviz
