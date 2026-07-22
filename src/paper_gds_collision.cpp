#include "rb10_rmpflow_rviz/paper_gds_collision.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace rb10_rmpflow_rviz
{
namespace paper_gds_collision
{
namespace
{

constexpr double kValidityTolerance = 1.0e-12;
constexpr double kExpUnderflowLimit = 700.0;

bool all_finite(const Params & params) noexcept
{
  const std::array<double, 13> values{
    params.metric_scalar,
    params.metric_modulation_radius,
    params.metric_exploder_std_dev,
    params.metric_exploder_eps,
    params.clearance_smoothing,
    params.metric_velocity_floor,
    params.metric_velocity_scale,
    params.repulsion_gain,
    params.repulsion_std_dev,
    params.damping_gain,
    params.damping_std_dev,
    params.damping_robustness_eps,
    params.damping_velocity_scale};

  return std::all_of(
    values.begin(), values.end(),
    [](double value) {return std::isfinite(value);});
}

double stable_sigmoid(double value)
{
  if (value >= 0.0) {
    return 1.0 / (1.0 + std::exp(-value));
  }
  const double exponential = std::exp(value);
  return exponential / (1.0 + exponential);
}

struct GateSample
{
  double value;
  double derivative;
};

GateSample quintic_distance_gate(double signed_clearance, double radius)
{
  if (signed_clearance <= 0.0) {
    return {1.0, 0.0};
  }
  if (signed_clearance >= radius) {
    return {0.0, 0.0};
  }

  const double z = (radius - signed_clearance) / radius;
  const double z2 = z * z;
  const double z3 = z2 * z;
  const double smoothstep = z3 * (10.0 + z * (-15.0 + 6.0 * z));
  const double smoothstep_derivative =
    30.0 * z2 * (1.0 - z) * (1.0 - z);

  // Squaring preserves the legacy quadratic gate's midpoint while making the
  // connection at both ends C2.
  return {
    smoothstep * smoothstep,
    -2.0 * smoothstep * smoothstep_derivative / radius};
}

struct PositivePartSample
{
  double value;
  double derivative;
};

PositivePartSample smooth_positive_part(double value, double smoothing)
{
  const double radius = std::hypot(value, smoothing);

  // The alternative expression for negative values avoids cancellation in
  // 0.5 * (value + hypot(value, smoothing)).
  const double positive_value = value >= 0.0 ?
    0.5 * (value + radius) :
    0.5 * smoothing * smoothing / (radius - value);

  return {
    positive_value,
    0.5 * (1.0 + value / radius)};
}

struct VelocityProfileSample
{
  double value;
  double derivative;
  double rate_times_derivative;
};

VelocityProfileSample velocity_profile(
  double clearance_rate,
  double floor,
  double scale)
{
  if (clearance_rate >= 0.0) {
    return {floor, 0.0, 0.0};
  }

  const double ratio = scale / (-clearance_rate);
  if (
    ratio >= std::sqrt(kExpUnderflowLimit) ||
    !std::isfinite(ratio))
  {
    // exp(-scale^2 / rate^2) and every derivative are flat at zero.
    return {floor, 0.0, 0.0};
  }

  const double ratio_squared = ratio * ratio;
  const double approach_profile = std::exp(-ratio_squared);
  const double rate_times_derivative =
    2.0 * (1.0 - floor) * approach_profile * ratio_squared;
  const double derivative = rate_times_derivative / clearance_rate;

  return {
    floor + (1.0 - floor) * approach_profile,
    derivative,
    rate_times_derivative};
}

bool all_finite(const Result & result)
{
  const std::array<double, 23> values{
    result.signed_clearance,
    result.clearance_rate,
    result.smooth_positive_clearance,
    result.smooth_positive_clearance_derivative,
    result.distance_gate,
    result.distance_gate_derivative,
    result.w,
    result.w_derivative,
    result.u,
    result.u_derivative,
    result.clearance_rate_times_u_derivative,
    result.G,
    result.Xi,
    result.xi,
    result.M,
    result.repulsion_acceleration,
    result.potential_force,
    result.damping_gate,
    result.B,
    result.damping_force,
    result.natural_force,
    result.w + result.M,
    result.G + result.Xi};

  return std::all_of(
    values.begin(), values.end(),
    [](double value) {return std::isfinite(value);});
}

}  // namespace

bool parameters_are_valid(const Params & params) noexcept
{
  return
    all_finite(params) &&
    params.metric_scalar >= 0.0 &&
    params.metric_modulation_radius > 0.0 &&
    params.metric_exploder_std_dev > 0.0 &&
    params.metric_exploder_eps > 0.0 &&
    params.clearance_smoothing > 0.0 &&
    params.metric_velocity_floor >= 0.0 &&
    params.metric_velocity_floor <= 1.0 &&
    params.metric_velocity_scale > 0.0 &&
    params.repulsion_gain >= 0.0 &&
    params.repulsion_std_dev > 0.0 &&
    params.damping_gain >= 0.0 &&
    params.damping_std_dev > 0.0 &&
    params.damping_robustness_eps > 0.0 &&
    params.damping_velocity_scale > 0.0;
}

Result evaluate(
  double signed_clearance,
  double clearance_rate,
  const Params & params)
{
  if (!parameters_are_valid(params)) {
    throw std::invalid_argument("Invalid paper GDS collision parameters");
  }
  if (!std::isfinite(signed_clearance) || !std::isfinite(clearance_rate)) {
    throw std::invalid_argument("Collision state must be finite");
  }

  Result result;
  result.signed_clearance = signed_clearance;
  result.clearance_rate = clearance_rate;

  const auto positive =
    smooth_positive_part(signed_clearance, params.clearance_smoothing);
  result.smooth_positive_clearance = positive.value;
  result.smooth_positive_clearance_derivative = positive.derivative;

  const auto distance_gate =
    quintic_distance_gate(signed_clearance, params.metric_modulation_radius);
  result.distance_gate = distance_gate.value;
  result.distance_gate_derivative = distance_gate.derivative;

  const double metric_denominator =
    params.metric_exploder_eps +
    positive.value / params.metric_exploder_std_dev;
  const double metric_denominator_derivative =
    positive.derivative / params.metric_exploder_std_dev;

  result.w =
    params.metric_scalar * distance_gate.value / metric_denominator;
  result.w_derivative =
    params.metric_scalar *
    (distance_gate.derivative * metric_denominator -
    distance_gate.value * metric_denominator_derivative) /
    (metric_denominator * metric_denominator);

  const auto velocity = velocity_profile(
    clearance_rate,
    params.metric_velocity_floor,
    params.metric_velocity_scale);
  result.u = velocity.value;
  result.u_derivative = velocity.derivative;
  result.clearance_rate_times_u_derivative =
    velocity.rate_times_derivative;

  result.G = result.w * result.u;
  result.Xi =
    0.5 * result.w * result.clearance_rate_times_u_derivative;
  result.xi =
    0.5 * result.u * result.w_derivative *
    clearance_rate * clearance_rate;
  result.M = result.G + result.Xi;

  result.repulsion_acceleration =
    params.repulsion_gain *
    std::exp(
    -positive.value / params.repulsion_std_dev);

  // Define Phi(s) = integral_s^infinity potential_force(r) dr.  Thus this
  // term is exactly -dPhi/ds and depends on position only.
  result.potential_force =
    result.w * params.metric_velocity_floor *
    result.repulsion_acceleration;

  result.damping_gate =
    stable_sigmoid(
    -clearance_rate / params.damping_velocity_scale);
  const double damping_denominator =
    params.damping_robustness_eps +
    positive.value / params.damping_std_dev;
  result.B =
    result.M * params.damping_gain *
    result.damping_gate / damping_denominator;
  result.damping_force = -result.B * clearance_rate;

  result.natural_force =
    result.potential_force - result.xi + result.damping_force;

  result.finite = all_finite(result);
  result.metric_psd =
    result.G >= -kValidityTolerance &&
    result.Xi >= -kValidityTolerance &&
    result.M >= -kValidityTolerance;
  result.damping_psd = result.B >= -kValidityTolerance;
  result.theorem_condition =
    result.clearance_rate_times_u_derivative >= -kValidityTolerance;
  result.distance_metric_nonincreasing =
    result.w_derivative <= kValidityTolerance;
  result.valid =
    result.finite &&
    result.metric_psd &&
    result.damping_psd &&
    result.theorem_condition &&
    result.distance_metric_nonincreasing;
  return result;
}

}  // namespace paper_gds_collision
}  // namespace rb10_rmpflow_rviz
