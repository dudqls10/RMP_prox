#include "rb10_rmpflow_rviz/escape_energy_certificate.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include <Eigen/Cholesky>

namespace rb10_rmpflow_rviz
{

namespace
{

constexpr double kNumericalTolerance = 1e-12;

bool all_finite(std::initializer_list<double> values)
{
  return std::all_of(
    values.begin(),
    values.end(),
    [](double value) {return std::isfinite(value);});
}

double nonnegative_or_zero(double value)
{
  return std::isfinite(value) ? std::max(value, 0.0) : 0.0;
}

double finite_power_sum(double lhs, double rhs)
{
  if (std::isinf(lhs) || std::isinf(rhs)) {
    return std::numeric_limits<double>::infinity();
  }
  if (lhs > std::numeric_limits<double>::max() - rhs) {
    return std::numeric_limits<double>::infinity();
  }
  return lhs + rhs;
}

double rank_one_power(double compliance, double metric, double signed_drive)
{
  const double denominator = 1.0 + metric * compliance;
  if (
    denominator <= kNumericalTolerance ||
    !std::isfinite(denominator))
  {
    return 0.0;
  }
  return (metric / denominator) * signed_drive;
}

}  // namespace

DirectEscapePower compute_direct_escape_power(
  double metric,
  double tangent_velocity,
  double desired_acceleration,
  double realized_acceleration)
{
  DirectEscapePower result;
  result.metric = nonnegative_or_zero(metric);
  result.tangent_velocity = std::isfinite(tangent_velocity) ? tangent_velocity : 0.0;
  result.desired_acceleration =
    std::isfinite(desired_acceleration) ? desired_acceleration : 0.0;
  result.realized_acceleration =
    std::isfinite(realized_acceleration) ? realized_acceleration : 0.0;
  result.valid =
    all_finite(
    {metric, tangent_velocity, desired_acceleration, realized_acceleration}) &&
    metric >= 0.0;
  if (!result.valid) {
    return result;
  }

  result.acceleration_residual = desired_acceleration - realized_acceleration;
  result.power =
    metric * tangent_velocity * result.acceleration_residual;
  if (!std::isfinite(result.power)) {
    result.valid = false;
    result.acceleration_residual = 0.0;
    result.power = 0.0;
  }
  return result;
}

RankOneEscapePower compute_rank_one_escape_power(
  double compliance,
  double metric,
  double tangent_velocity,
  double desired_acceleration,
  double base_realized_acceleration)
{
  RankOneEscapePower result;
  result.compliance = nonnegative_or_zero(compliance);
  result.metric = nonnegative_or_zero(metric);
  result.tangent_velocity = std::isfinite(tangent_velocity) ? tangent_velocity : 0.0;
  result.desired_acceleration =
    std::isfinite(desired_acceleration) ? desired_acceleration : 0.0;
  result.base_realized_acceleration =
    std::isfinite(base_realized_acceleration) ? base_realized_acceleration : 0.0;
  result.valid =
    all_finite(
    {
      compliance,
      metric,
      tangent_velocity,
      desired_acceleration,
      base_realized_acceleration
    }) &&
    compliance >= 0.0 &&
    metric >= 0.0;
  if (!result.valid) {
    return result;
  }

  const double denominator = 1.0 + metric * compliance;
  if (
    denominator <= kNumericalTolerance ||
    !std::isfinite(denominator))
  {
    result.valid = false;
    return result;
  }

  result.effective_metric = metric / denominator;
  result.acceleration_residual =
    desired_acceleration - base_realized_acceleration;
  result.power =
    result.effective_metric *
    tangent_velocity *
    result.acceleration_residual;
  if (
    !std::isfinite(result.effective_metric) ||
    !std::isfinite(result.power))
  {
    result.valid = false;
    result.effective_metric = 0.0;
    result.acceleration_residual = 0.0;
    result.power = 0.0;
  }
  return result;
}

RankOneEscapePower compute_rank_one_escape_power(
  const Eigen::MatrixXd & base_metric,
  const Eigen::RowVectorXd & scalar_jacobian,
  const Eigen::VectorXd & base_qdd,
  double metric,
  double tangent_velocity,
  double desired_acceleration,
  double scalar_curvature,
  double diagonal_regularization)
{
  RankOneEscapePower invalid;
  if (
    base_metric.rows() <= 0 ||
    base_metric.rows() != base_metric.cols() ||
    scalar_jacobian.size() != base_metric.cols() ||
    base_qdd.size() != base_metric.cols() ||
    !base_metric.allFinite() ||
    !scalar_jacobian.allFinite() ||
    !base_qdd.allFinite() ||
    !all_finite(
      {
        metric,
        tangent_velocity,
        desired_acceleration,
        scalar_curvature,
        diagonal_regularization
      }) ||
    metric < 0.0 ||
    diagonal_regularization < 0.0)
  {
    return invalid;
  }

  const double matrix_scale = std::max(1.0, base_metric.cwiseAbs().maxCoeff());
  const double symmetry_error =
    (base_metric - base_metric.transpose()).cwiseAbs().maxCoeff();
  if (symmetry_error > 1e-9 * matrix_scale) {
    return invalid;
  }

  Eigen::MatrixXd regularized_metric =
    0.5 * (base_metric + base_metric.transpose());
  regularized_metric.diagonal().array() += diagonal_regularization;
  Eigen::LDLT<Eigen::MatrixXd> factorization(regularized_metric);
  if (
    factorization.info() != Eigen::Success ||
    factorization.vectorD().size() == 0 ||
    factorization.vectorD().minCoeff() <=
    kNumericalTolerance * matrix_scale)
  {
    return invalid;
  }

  const Eigen::VectorXd response =
    factorization.solve(scalar_jacobian.transpose());
  if (
    factorization.info() != Eigen::Success ||
    !response.allFinite())
  {
    return invalid;
  }
  double compliance = scalar_jacobian.dot(response);
  if (!std::isfinite(compliance) || compliance < -1e-9 * matrix_scale) {
    return invalid;
  }
  compliance = std::max(compliance, 0.0);

  const double base_realized_acceleration =
    scalar_jacobian.dot(base_qdd) + scalar_curvature;
  return compute_rank_one_escape_power(
    compliance,
    metric,
    tangent_velocity,
    desired_acceleration,
    base_realized_acceleration);
}

RankOneMetricCap compute_rank_one_metric_cap(
  double compliance,
  double signed_drive,
  double positive_power_budget,
  double requested_metric)
{
  RankOneMetricCap result;
  result.compliance = nonnegative_or_zero(compliance);
  result.signed_drive = std::isfinite(signed_drive) ? signed_drive : 0.0;
  result.positive_power_budget =
    positive_power_budget == std::numeric_limits<double>::infinity() ?
    positive_power_budget :
    nonnegative_or_zero(positive_power_budget);
  result.requested_metric = nonnegative_or_zero(requested_metric);
  result.metric_cap = 0.0;
  result.valid =
    std::isfinite(compliance) &&
    compliance >= 0.0 &&
    std::isfinite(signed_drive) &&
    (std::isfinite(positive_power_budget) ||
    positive_power_budget == std::numeric_limits<double>::infinity()) &&
    positive_power_budget >= 0.0 &&
    std::isfinite(requested_metric) &&
    requested_metric >= 0.0;
  if (!result.valid) {
    return result;
  }

  result.requested_power =
    rank_one_power(compliance, requested_metric, signed_drive);
  const bool unlimited_budget =
    positive_power_budget == std::numeric_limits<double>::infinity();
  if (signed_drive <= 0.0 || unlimited_budget) {
    result.metric_cap = std::numeric_limits<double>::infinity();
    result.allowed_metric = requested_metric;
    result.metric_scale = 1.0;
    result.allowed_power = result.requested_power;
    return result;
  }

  if (positive_power_budget <= 0.0) {
    result.metric_cap = 0.0;
    result.allowed_metric = 0.0;
    result.metric_scale = requested_metric > 0.0 ? 0.0 : 1.0;
    result.allowed_power = 0.0;
    result.limited = requested_metric > 0.0;
    return result;
  }

  // If signed_drive / compliance is already below the budget, the
  // asymptotic m -> infinity power is admissible and no finite cap exists.
  if (
    compliance > 0.0 &&
    signed_drive <= positive_power_budget * compliance)
  {
    result.metric_cap = std::numeric_limits<double>::infinity();
    result.allowed_metric = requested_metric;
    result.metric_scale = 1.0;
    result.allowed_power = result.requested_power;
    return result;
  }

  const double denominator =
    signed_drive - positive_power_budget * compliance;
  if (denominator <= kNumericalTolerance || !std::isfinite(denominator)) {
    result.metric_cap = std::numeric_limits<double>::infinity();
    result.allowed_metric = requested_metric;
    result.metric_scale = 1.0;
    result.allowed_power = result.requested_power;
    return result;
  }

  result.metric_cap = positive_power_budget / denominator;
  result.allowed_metric = std::min(requested_metric, result.metric_cap);
  result.metric_scale =
    requested_metric > 0.0 ?
    std::clamp(result.allowed_metric / requested_metric, 0.0, 1.0) :
    1.0;
  result.allowed_power =
    rank_one_power(compliance, result.allowed_metric, signed_drive);
  result.limited =
    result.allowed_metric + kNumericalTolerance < requested_metric;
  return result;
}

BaseEnergyBalance compute_base_energy_balance(
  double damping_dissipation,
  double escape_power,
  double solve_power,
  double clamp_power,
  double explicit_time_power)
{
  BaseEnergyBalance result;
  result.damping_dissipation =
    nonnegative_or_zero(damping_dissipation);
  result.escape_power = std::isfinite(escape_power) ? escape_power : 0.0;
  result.solve_power = std::isfinite(solve_power) ? solve_power : 0.0;
  result.clamp_power = std::isfinite(clamp_power) ? clamp_power : 0.0;
  result.explicit_time_power =
    std::isfinite(explicit_time_power) ? explicit_time_power : 0.0;
  result.valid =
    all_finite(
    {
      damping_dissipation,
      escape_power,
      solve_power,
      clamp_power,
      explicit_time_power
    }) &&
    damping_dissipation >= 0.0;
  if (!result.valid) {
    return result;
  }

  result.total_injection_power =
    escape_power + solve_power + clamp_power + explicit_time_power;
  result.nonincrease_margin =
    damping_dissipation - result.total_injection_power;
  result.base_lyapunov_rate = -result.nonincrease_margin;
  if (
    !std::isfinite(result.total_injection_power) ||
    !std::isfinite(result.nonincrease_margin) ||
    !std::isfinite(result.base_lyapunov_rate))
  {
    result.valid = false;
    result.total_injection_power = 0.0;
    result.nonincrease_margin = 0.0;
    result.base_lyapunov_rate = 0.0;
  }
  return result;
}

PositiveExcessEnergyTankUpdate update_positive_excess_energy_tank(
  double dt,
  double sigma,
  double tank_capacity,
  double current_tank_energy,
  double damping_dissipation,
  double requested_escape_power)
{
  PositiveExcessEnergyTankUpdate result;
  const bool scalar_inputs_finite =
    all_finite(
    {
      dt,
      sigma,
      tank_capacity,
      current_tank_energy,
      damping_dissipation,
      requested_escape_power
    });
  result.valid =
    scalar_inputs_finite &&
    dt >= 0.0 &&
    sigma >= 0.0 &&
    sigma <= 1.0 &&
    tank_capacity >= 0.0 &&
    current_tank_energy >= 0.0 &&
    current_tank_energy <= tank_capacity &&
    damping_dissipation >= 0.0;

  result.dt = std::isfinite(dt) ? std::max(dt, 0.0) : 0.0;
  result.sigma =
    std::isfinite(sigma) ? std::clamp(sigma, 0.0, 1.0) : 0.0;
  result.capacity = nonnegative_or_zero(tank_capacity);
  result.previous_energy =
    std::isfinite(current_tank_energy) ?
    std::clamp(current_tank_energy, 0.0, result.capacity) :
    0.0;
  result.new_energy = result.previous_energy;
  result.requested_power =
    std::isfinite(requested_escape_power) ? requested_escape_power : 0.0;
  result.passive_positive_power_budget =
    result.sigma * nonnegative_or_zero(damping_dissipation);

  if (!result.valid) {
    result.power_scale = requested_escape_power <= 0.0 &&
      std::isfinite(requested_escape_power) ? 1.0 : 0.0;
    result.allowed_power =
      result.power_scale * result.requested_power;
    result.allowable_positive_power_budget =
      result.passive_positive_power_budget;
    result.composite_storage_rate =
      -nonnegative_or_zero(damping_dissipation) + result.allowed_power;
    return result;
  }

  if (result.dt > kNumericalTolerance && result.previous_energy > 0.0) {
    const double inverse_dt = 1.0 / result.dt;
    result.tank_discharge_power_budget =
      result.previous_energy > std::numeric_limits<double>::max() / inverse_dt ?
      std::numeric_limits<double>::infinity() :
      result.previous_energy * inverse_dt;
  }
  result.allowable_positive_power_budget = finite_power_sum(
    result.passive_positive_power_budget,
    result.tank_discharge_power_budget);

  if (result.requested_power <= 0.0) {
    result.power_scale = 1.0;
    result.allowed_power = result.requested_power;
  } else {
    result.power_scale = std::clamp(
      result.allowable_positive_power_budget / result.requested_power,
      0.0,
      1.0);
    result.allowed_power = result.power_scale * result.requested_power;
  }

  result.positive_excess_power = std::max(
    result.allowed_power - result.passive_positive_power_budget,
    0.0);
  double energy_after_discharge = result.previous_energy;
  if (result.dt > 0.0) {
    energy_after_discharge -= result.dt * result.positive_excess_power;
  }
  energy_after_discharge = std::clamp(
    energy_after_discharge,
    0.0,
    result.capacity);

  const double requested_recharge_power =
    (1.0 - result.sigma) * damping_dissipation;
  if (result.dt > kNumericalTolerance) {
    const double capacity_power =
      (result.capacity - energy_after_discharge) / result.dt;
    result.recharge_power = std::clamp(
      requested_recharge_power,
      0.0,
      std::max(capacity_power, 0.0));
    result.new_energy =
      energy_after_discharge + result.dt * result.recharge_power;
  }
  result.new_energy = std::clamp(
    result.new_energy,
    0.0,
    result.capacity);

  const double tank_energy_rate =
    result.dt > kNumericalTolerance ?
    (result.new_energy - result.previous_energy) / result.dt :
    0.0;
  result.composite_storage_rate =
    -damping_dissipation + result.allowed_power + tank_energy_rate;
  if (
    !std::isfinite(result.power_scale) ||
    !std::isfinite(result.allowed_power) ||
    !std::isfinite(result.new_energy) ||
    !std::isfinite(result.composite_storage_rate))
  {
    result.valid = false;
    result.power_scale = 0.0;
    result.allowed_power = 0.0;
    result.new_energy = result.previous_energy;
    result.composite_storage_rate = -damping_dissipation;
  }
  return result;
}

EpisodeEnergyIntegrals integrate_episode_energy(
  const EpisodeEnergyIntegrals & previous,
  double dt,
  double sigma,
  const BaseEnergyBalance & balance)
{
  EpisodeEnergyIntegrals result = previous;
  if (
    !balance.valid ||
    !std::isfinite(dt) ||
    dt <= 0.0 ||
    !std::isfinite(sigma) ||
    sigma < 0.0 ||
    sigma > 1.0)
  {
    ++result.rejected_samples;
    return result;
  }

  result.elapsed_time += dt;
  result.damping_energy += dt * balance.damping_dissipation;
  result.escape_energy += dt * balance.escape_power;
  result.positive_escape_energy +=
    dt * std::max(balance.escape_power, 0.0);
  result.negative_escape_energy +=
    dt * std::max(-balance.escape_power, 0.0);
  result.positive_excess_escape_energy +=
    dt * std::max(
    balance.escape_power - sigma * balance.damping_dissipation,
    0.0);
  result.solve_energy += dt * balance.solve_power;
  result.clamp_energy += dt * balance.clamp_power;
  result.explicit_time_energy +=
    dt * balance.explicit_time_power;
  result.total_injection_energy +=
    dt * balance.total_injection_power;
  result.nonincrease_margin_integral +=
    dt * balance.nonincrease_margin;
  result.base_lyapunov_change +=
    dt * balance.base_lyapunov_rate;
  ++result.accepted_samples;
  return result;
}

}  // namespace rb10_rmpflow_rviz
