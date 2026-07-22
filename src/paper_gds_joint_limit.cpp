#include "rb10_rmpflow_rviz/paper_gds_joint_limit.hpp"

#include <cmath>
#include <initializer_list>

namespace rb10_rmpflow_rviz
{

namespace
{

bool all_finite(std::initializer_list<double> values) noexcept
{
  for (const double value : values) {
    if (!std::isfinite(value)) {
      return false;
    }
  }
  return true;
}

PaperGdsJointLimitResult invalid_result(
  PaperGdsJointLimitStatus status,
  double q,
  double qdot) noexcept
{
  PaperGdsJointLimitResult result;
  result.status = status;
  result.q = q;
  result.qdot = qdot;
  return result;
}

}  // namespace

bool valid_paper_gds_joint_limit_config(
  const PaperGdsJointLimitConfig & config) noexcept
{
  if (!all_finite({
      config.lower,
      config.upper,
      config.center_fraction,
      config.task_metric,
      config.potential_gain,
      config.damping,
      config.boundary_epsilon}))
  {
    return false;
  }

  const double width = config.upper - config.lower;
  const double center_lower_clearance = config.center_fraction * width;
  const double center_upper_clearance = (1.0 - config.center_fraction) * width;
  return
    std::isfinite(width) &&
    std::isfinite(center_lower_clearance) &&
    std::isfinite(center_upper_clearance) &&
    width > 0.0 &&
    config.center_fraction > 0.0 &&
    config.center_fraction < 1.0 &&
    config.task_metric > 0.0 &&
    config.potential_gain > 0.0 &&
    config.damping > 0.0 &&
    config.boundary_epsilon >= 0.0 &&
    config.boundary_epsilon < 0.5 * width &&
    center_lower_clearance > config.boundary_epsilon &&
    center_upper_clearance > config.boundary_epsilon;
}

PaperGdsJointLimitResult evaluate_paper_gds_joint_limit(
  const PaperGdsJointLimitConfig & config,
  double q,
  double qdot) noexcept
{
  if (!valid_paper_gds_joint_limit_config(config)) {
    return invalid_result(PaperGdsJointLimitStatus::kInvalidConfiguration, q, qdot);
  }
  if (!all_finite({q, qdot})) {
    return invalid_result(PaperGdsJointLimitStatus::kNonFiniteState, q, qdot);
  }
  if (!(q > config.lower && q < config.upper)) {
    return invalid_result(PaperGdsJointLimitStatus::kOutsideOpenInterval, q, qdot);
  }

  const double lower_clearance = q - config.lower;
  const double upper_clearance = config.upper - q;
  if (!(
      lower_clearance > config.boundary_epsilon &&
      upper_clearance > config.boundary_epsilon))
  {
    auto result = invalid_result(
      PaperGdsJointLimitStatus::kInsideBoundaryEpsilon,
      q,
      qdot);
    result.lower_clearance = lower_clearance;
    result.upper_clearance = upper_clearance;
    return result;
  }

  PaperGdsJointLimitResult result;
  result.q = q;
  result.qdot = qdot;
  result.lower_clearance = lower_clearance;
  result.upper_clearance = upper_clearance;

  const double width = config.upper - config.lower;
  result.center_q = config.lower + config.center_fraction * width;
  result.center_z =
    std::log(config.center_fraction) - std::log1p(-config.center_fraction);

  // Difference-of-logs avoids forming a ratio that may overflow close to a
  // boundary.  No clearance is clamped or otherwise changed.
  result.z = std::log(lower_clearance) - std::log(upper_clearance);
  result.jacobian = 1.0 / lower_clearance + 1.0 / upper_clearance;
  result.jacobian_derivative =
    -1.0 / (lower_clearance * lower_clearance) +
    1.0 / (upper_clearance * upper_clearance);
  result.zdot = result.jacobian * qdot;
  result.curvature = result.jacobian_derivative * qdot * qdot;

  result.task_metric = config.task_metric;
  result.task_inertia = config.task_metric;
  result.damping = config.damping;

  const double potential_error = result.z - result.center_z;
  result.potential =
    0.5 * config.potential_gain * potential_error * potential_error;
  result.potential_gradient = config.potential_gain * potential_error;
  result.task_natural_force =
    -result.potential_gradient - config.damping * result.zdot;
  result.task_acceleration = result.task_natural_force / result.task_inertia;

  result.kinetic_energy =
    0.5 * result.task_metric * result.zdot * result.zdot;
  result.total_energy = result.kinetic_energy + result.potential;
  result.energy_rate = -config.damping * result.zdot * result.zdot;

  const double jacobian_squared = result.jacobian * result.jacobian;
  result.root_metric = result.task_metric * jacobian_squared;
  result.root_inertia = result.task_inertia * jacobian_squared;
  result.root_potential_gradient =
    result.jacobian * result.potential_gradient;
  result.root_damping = config.damping * jacobian_squared;
  result.root_curvature_force =
    result.jacobian * result.task_inertia * result.curvature;
  result.root_natural_force =
    result.jacobian *
    (result.task_natural_force - result.task_inertia * result.curvature);
  result.root_acceleration = result.root_natural_force / result.root_inertia;

  if (!all_finite({
      result.lower_clearance,
      result.upper_clearance,
      result.center_q,
      result.center_z,
      result.z,
      result.zdot,
      result.jacobian,
      result.jacobian_derivative,
      result.curvature,
      result.task_metric,
      result.task_inertia,
      result.potential,
      result.potential_gradient,
      result.damping,
      result.task_natural_force,
      result.task_acceleration,
      result.kinetic_energy,
      result.total_energy,
      result.energy_rate,
      result.root_metric,
      result.root_inertia,
      result.root_potential_gradient,
      result.root_damping,
      result.root_curvature_force,
      result.root_natural_force,
      result.root_acceleration}) ||
    !(result.root_metric > 0.0) ||
    !(result.root_inertia > 0.0))
  {
    return invalid_result(PaperGdsJointLimitStatus::kNumericalFailure, q, qdot);
  }

  result.valid = true;
  result.status = PaperGdsJointLimitStatus::kValid;
  return result;
}

}  // namespace rb10_rmpflow_rviz
