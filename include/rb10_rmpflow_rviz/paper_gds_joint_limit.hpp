#pragma once

#include <limits>

namespace rb10_rmpflow_rviz
{

// Parameters for one revolute-joint limit task.  The mathematical task domain
// is the open interval (lower, upper).  boundary_epsilon only rejects states
// that are too close to that domain boundary for reliable floating-point
// evaluation; it is never used to clamp or alter q.
struct PaperGdsJointLimitConfig
{
  double lower{-1.0};
  double upper{1.0};
  double center_fraction{0.5};
  double task_metric{1.0};
  double potential_gain{1.0};
  double damping{1.0};
  double boundary_epsilon{1e-9};
};

enum class PaperGdsJointLimitStatus
{
  kValid = 0,
  kInvalidConfiguration,
  kNonFiniteState,
  kOutsideOpenInterval,
  kInsideBoundaryEpsilon,
  kNumericalFailure,
};

// Evaluation of the strict structured GDS in the unconstrained scalar task
//
//   z(q) = log((q - lower) / (upper - q)).
//
// The task-space GDS has constant G_z = M_z = task_metric,
// Phi(z) = 0.5 * potential_gain * (z - center_z)^2, and
// B_z = damping.  Consequently Xi_Gz = xi_Gz = 0 and its natural force is
//
//   f_z = -dPhi/dz - B_z * zdot.
//
// Pullback through z(q) gives the root scalar contribution
//
//   M_q = J^2 M_z,
//   f_q = J (f_z - M_z c),
//
// where c = J'(q) qdot^2.  A six-joint caller can add M_q to the matching
// root diagonal entry and f_q to the matching root-force entry.
struct PaperGdsJointLimitResult
{
  bool valid{false};
  PaperGdsJointLimitStatus status{PaperGdsJointLimitStatus::kInvalidConfiguration};

  double q{std::numeric_limits<double>::quiet_NaN()};
  double qdot{std::numeric_limits<double>::quiet_NaN()};
  double lower_clearance{std::numeric_limits<double>::quiet_NaN()};
  double upper_clearance{std::numeric_limits<double>::quiet_NaN()};

  double center_q{std::numeric_limits<double>::quiet_NaN()};
  double center_z{std::numeric_limits<double>::quiet_NaN()};
  double z{std::numeric_limits<double>::quiet_NaN()};
  double zdot{std::numeric_limits<double>::quiet_NaN()};
  double jacobian{std::numeric_limits<double>::quiet_NaN()};
  double jacobian_derivative{std::numeric_limits<double>::quiet_NaN()};
  double curvature{std::numeric_limits<double>::quiet_NaN()};

  double task_metric{std::numeric_limits<double>::quiet_NaN()};
  double task_inertia{std::numeric_limits<double>::quiet_NaN()};
  double potential{std::numeric_limits<double>::quiet_NaN()};
  double potential_gradient{std::numeric_limits<double>::quiet_NaN()};
  double damping{std::numeric_limits<double>::quiet_NaN()};
  double task_natural_force{std::numeric_limits<double>::quiet_NaN()};
  double task_acceleration{std::numeric_limits<double>::quiet_NaN()};
  double kinetic_energy{std::numeric_limits<double>::quiet_NaN()};
  double total_energy{std::numeric_limits<double>::quiet_NaN()};
  double energy_rate{std::numeric_limits<double>::quiet_NaN()};

  double root_metric{std::numeric_limits<double>::quiet_NaN()};
  double root_inertia{std::numeric_limits<double>::quiet_NaN()};
  double root_potential_gradient{std::numeric_limits<double>::quiet_NaN()};
  double root_damping{std::numeric_limits<double>::quiet_NaN()};
  double root_curvature_force{std::numeric_limits<double>::quiet_NaN()};
  double root_natural_force{std::numeric_limits<double>::quiet_NaN()};
  double root_acceleration{std::numeric_limits<double>::quiet_NaN()};
};

bool valid_paper_gds_joint_limit_config(
  const PaperGdsJointLimitConfig & config) noexcept;

PaperGdsJointLimitResult evaluate_paper_gds_joint_limit(
  const PaperGdsJointLimitConfig & config,
  double q,
  double qdot) noexcept;

}  // namespace rb10_rmpflow_rviz
