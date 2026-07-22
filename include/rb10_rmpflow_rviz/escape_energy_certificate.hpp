#pragma once

#include <cstdint>
#include <limits>

#include <Eigen/Core>

namespace rb10_rmpflow_rviz
{

// Power injected into a base GDS Lyapunov balance by the rank-one Escape
// interconnection.  This is not merely qdot^T f_escape: because Escape adds
// both metric and force, its exact contribution is
//
//   P_escape = qdot^T (f_escape - M_escape qdd)
//            = m * v_t * (a_y - ydd).
struct DirectEscapePower
{
  bool valid{false};
  double metric{0.0};
  double tangent_velocity{0.0};
  double desired_acceleration{0.0};
  double realized_acceleration{0.0};
  double acceleration_residual{0.0};
  double power{0.0};
};

DirectEscapePower compute_direct_escape_power(
  double metric,
  double tangent_velocity,
  double desired_acceleration,
  double realized_acceleration);

// For an exact, unregularized SPD base solve, h = j M0^{-1} j^T and
// mu = m / (1 + m h).  The same Escape power can then be evaluated without
// the final combined acceleration:
//
//   P_escape = mu * v_t * (a_y - ydd_0).
struct RankOneEscapePower
{
  bool valid{false};
  double compliance{0.0};
  double metric{0.0};
  double effective_metric{0.0};
  double tangent_velocity{0.0};
  double desired_acceleration{0.0};
  double base_realized_acceleration{0.0};
  double acceleration_residual{0.0};
  double power{0.0};
};

RankOneEscapePower compute_rank_one_escape_power(
  double compliance,
  double metric,
  double tangent_velocity,
  double desired_acceleration,
  double base_realized_acceleration);

// Matrix overload used by the six-dimensional controller and by lower
// dimensional synthetic tests.  base_qdd must be the solution associated
// with base_metric (including diagonal_regularization, if nonzero).
RankOneEscapePower compute_rank_one_escape_power(
  const Eigen::MatrixXd & base_metric,
  const Eigen::RowVectorXd & scalar_jacobian,
  const Eigen::VectorXd & base_qdd,
  double metric,
  double tangent_velocity,
  double desired_acceleration,
  double scalar_curvature,
  double diagonal_regularization = 0.0);

// Exact metric cap for
//
//   P(m) = m / (1 + m h) * signed_drive,
//
// where signed_drive = v_t * (a_y - ydd_0).  The cap changes metric and
// natural force together, as required by the canonical rank-one Escape leaf.
struct RankOneMetricCap
{
  bool valid{false};
  bool limited{false};
  double compliance{0.0};
  double signed_drive{0.0};
  double positive_power_budget{0.0};
  double requested_metric{0.0};
  double metric_cap{0.0};
  double allowed_metric{0.0};
  double metric_scale{0.0};
  double requested_power{0.0};
  double allowed_power{0.0};
};

RankOneMetricCap compute_rank_one_metric_cap(
  double compliance,
  double signed_drive,
  double positive_power_budget,
  double requested_metric);

// Exact instantaneous balance for the base Lyapunov candidate:
//
//   Vdot_0 = -D_0 + P_escape + P_solve + P_clamp + P_time
//          = -nonincrease_margin.
struct BaseEnergyBalance
{
  bool valid{false};
  double damping_dissipation{0.0};
  double escape_power{0.0};
  double solve_power{0.0};
  double clamp_power{0.0};
  double explicit_time_power{0.0};
  double total_injection_power{0.0};
  double nonincrease_margin{0.0};
  double base_lyapunov_rate{0.0};
};

BaseEnergyBalance compute_base_energy_balance(
  double damping_dissipation,
  double escape_power,
  double solve_power,
  double clamp_power,
  double explicit_time_power);

// A positive-excess energy tank permits the passive instantaneous budget
// sigma*D and spends stored tank energy only on power above that budget.
// Unused damping reserve, at most (1-sigma)*D, recharges the tank.
//
// With p_x = max(P_allowed - sigma*D, 0) and
//   Edot = recharge - p_x,  0 <= recharge <= (1-sigma)*D,
// the composite storage W = V_0 + E satisfies
//
//   Wdot = -D + P_allowed + Edot <= 0.
//
// Capacity clipping only reduces recharge and therefore preserves the
// inequality.  power_scale scales requested positive power; a caller that
// scales the rank-one metric should use compute_rank_one_metric_cap and then
// re-evaluate the direct power after the combined solve.
struct PositiveExcessEnergyTankUpdate
{
  bool valid{false};
  double dt{0.0};
  double sigma{0.0};
  double capacity{0.0};
  double previous_energy{0.0};
  double new_energy{0.0};
  double requested_power{0.0};
  double allowed_power{0.0};
  double passive_positive_power_budget{0.0};
  double tank_discharge_power_budget{0.0};
  double allowable_positive_power_budget{0.0};
  double positive_excess_power{0.0};
  double recharge_power{0.0};
  double power_scale{0.0};
  double composite_storage_rate{0.0};
};

PositiveExcessEnergyTankUpdate update_positive_excess_energy_tank(
  double dt,
  double sigma,
  double tank_capacity,
  double current_tank_energy,
  double damping_dissipation,
  double requested_escape_power);

struct EpisodeEnergyIntegrals
{
  double elapsed_time{0.0};
  double damping_energy{0.0};
  double escape_energy{0.0};
  double positive_escape_energy{0.0};
  double negative_escape_energy{0.0};
  double positive_excess_escape_energy{0.0};
  double solve_energy{0.0};
  double clamp_energy{0.0};
  double explicit_time_energy{0.0};
  double total_injection_energy{0.0};
  double nonincrease_margin_integral{0.0};
  double base_lyapunov_change{0.0};
  std::uint64_t accepted_samples{0};
  std::uint64_t rejected_samples{0};
};

EpisodeEnergyIntegrals integrate_episode_energy(
  const EpisodeEnergyIntegrals & previous,
  double dt,
  double sigma,
  const BaseEnergyBalance & balance);

}  // namespace rb10_rmpflow_rviz
