#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

#include <gtest/gtest.h>

#include "rb10_rmpflow_rviz/paper_gds_collision.hpp"

namespace
{

namespace collision = rb10_rmpflow_rviz::paper_gds_collision;

collision::Params make_params()
{
  collision::Params params;
  params.metric_scalar = 100000.0;
  params.metric_modulation_radius = 0.3;
  params.metric_exploder_std_dev = 0.02;
  params.metric_exploder_eps = 0.8;
  params.clearance_smoothing = 1.0e-4;
  params.metric_velocity_floor = 0.5;
  params.metric_velocity_scale = 0.03;
  params.repulsion_gain = 500.0;
  params.repulsion_std_dev = 0.05;
  params.damping_gain = 50.0;
  params.damping_std_dev = 0.01;
  params.damping_robustness_eps = 0.01;
  params.damping_velocity_scale = 0.03;
  return params;
}

TEST(PaperGdsCollision, AnalyticDistanceWeightDerivativeMatchesFiniteDifference)
{
  const auto params = make_params();
  constexpr double clearance = 0.083;
  constexpr double clearance_rate = -0.041;
  constexpr double step = 1.0e-6;

  const auto center =
    collision::evaluate(clearance, clearance_rate, params);
  const auto plus =
    collision::evaluate(clearance + step, clearance_rate, params);
  const auto minus =
    collision::evaluate(clearance - step, clearance_rate, params);

  const double finite_difference =
    (plus.w - minus.w) / (2.0 * step);
  const double tolerance =
    2.0e-6 * std::max(1.0, std::abs(center.w_derivative));

  EXPECT_NEAR(center.w_derivative, finite_difference, tolerance);

  const auto velocity_plus =
    collision::evaluate(clearance, clearance_rate + step, params);
  const auto velocity_minus =
    collision::evaluate(clearance, clearance_rate - step, params);
  const double g_velocity_derivative =
    (velocity_plus.G - velocity_minus.G) / (2.0 * step);
  EXPECT_NEAR(
    center.Xi,
    0.5 * clearance_rate * g_velocity_derivative,
    2.0e-5 * std::max(1.0, std::abs(center.Xi)));

  const double g_clearance_derivative =
    (plus.G - minus.G) / (2.0 * step);
  EXPECT_NEAR(
    center.xi,
    0.5 * g_clearance_derivative *
    clearance_rate * clearance_rate,
    2.0e-6 * std::max(1.0, std::abs(center.xi)));
}

TEST(PaperGdsCollision, StructuredGdsConditionsAndEnergyIdentityHold)
{
  const auto params = make_params();
  const std::array<double, 8> clearances{
    -0.2, -0.01, 0.0, 0.001, 0.05, 0.15, 0.299, 0.31};
  const std::array<double, 9> rates{
    -0.5, -0.1, -0.03, -1.0e-6, 0.0,
    1.0e-6, 0.03, 0.1, 0.5};

  for (const double clearance : clearances) {
    for (const double rate : rates) {
      const auto result =
        collision::evaluate(clearance, rate, params);
      EXPECT_TRUE(result.finite);
      EXPECT_TRUE(result.metric_psd);
      EXPECT_TRUE(result.damping_psd);
      EXPECT_TRUE(result.theorem_condition);
      EXPECT_TRUE(result.distance_metric_nonincreasing);
      EXPECT_TRUE(result.valid);
      EXPECT_GE(result.M, -1.0e-12);
      EXPECT_GE(result.B, -1.0e-12);
      EXPECT_GE(
        result.clearance_rate_times_u_derivative,
        -1.0e-12);
    }
  }

  const auto result = collision::evaluate(0.04, -0.06, params);
  ASSERT_GT(result.M, 0.0);
  const double acceleration = result.natural_force / result.M;

  // V = 0.5 G s_dot^2 + Phi and Phi' = -potential_force.
  // The structured-GDS equation must give V_dot = -B s_dot^2.
  const double energy_rate =
    result.clearance_rate *
    (result.M * acceleration + result.xi -
    result.potential_force);
  const double expected_energy_rate =
    -result.B * result.clearance_rate * result.clearance_rate;
  EXPECT_NEAR(
    energy_rate,
    expected_energy_rate,
    1.0e-10 * std::max(1.0, std::abs(expected_energy_rate)));
  EXPECT_LE(energy_rate, 1.0e-12);
}

TEST(PaperGdsCollision, CutoffAndPenetrationAreFinite)
{
  const auto params = make_params();
  const std::array<double, 6> penetration_clearances{
    -1.0e6, -1.0, -0.1, -1.0e-3, -1.0e-9, 0.0};

  for (const double clearance : penetration_clearances) {
    const auto result =
      collision::evaluate(clearance, -0.05, params);
    EXPECT_TRUE(result.finite);
    EXPECT_TRUE(result.valid);
    EXPECT_GT(result.w, 0.0);
    EXPECT_LE(
      result.w,
      params.metric_scalar / params.metric_exploder_eps *
      (1.0 + 1.0e-12));
    EXPECT_GT(result.potential_force, 0.0);
  }

  for (const double rate : {-0.2, 0.0, 0.2}) {
    const auto cutoff = collision::evaluate(
      params.metric_modulation_radius, rate, params);
    const auto outside = collision::evaluate(
      params.metric_modulation_radius + 0.1, rate, params);

    EXPECT_TRUE(cutoff.valid);
    EXPECT_DOUBLE_EQ(cutoff.distance_gate, 0.0);
    EXPECT_DOUBLE_EQ(cutoff.w, 0.0);
    EXPECT_DOUBLE_EQ(cutoff.M, 0.0);
    EXPECT_DOUBLE_EQ(cutoff.potential_force, 0.0);
    EXPECT_DOUBLE_EQ(cutoff.B, 0.0);
    EXPECT_DOUBLE_EQ(cutoff.natural_force, 0.0);

    EXPECT_TRUE(outside.valid);
    EXPECT_DOUBLE_EQ(outside.w, 0.0);
    EXPECT_DOUBLE_EQ(outside.natural_force, 0.0);
  }
}

TEST(PaperGdsCollision, RestAccelerationPreservesLegacyMapping)
{
  auto params = make_params();
  params.clearance_smoothing = 1.0e-9;

  const auto result = collision::evaluate(0.0, 0.0, params);
  ASSERT_TRUE(result.valid);
  ASSERT_GT(result.M, 0.0);

  const double expected_metric =
    params.metric_scalar /
    (result.smooth_positive_clearance /
    params.metric_exploder_std_dev +
    params.metric_exploder_eps) *
    params.metric_velocity_floor;
  EXPECT_NEAR(
    result.M,
    expected_metric,
    1.0e-12 * std::max(1.0, expected_metric));

  const double expected_repulsion_acceleration =
    params.repulsion_gain *
    std::exp(
    -result.smooth_positive_clearance /
    params.repulsion_std_dev);
  EXPECT_DOUBLE_EQ(result.xi, 0.0);
  EXPECT_DOUBLE_EQ(result.damping_force, 0.0);
  EXPECT_NEAR(
    result.natural_force / result.M,
    expected_repulsion_acceleration,
    1.0e-12 * std::max(1.0, expected_repulsion_acceleration));

  // With floor=0.5, this reproduces the legacy sigmoid gate at rest.
  EXPECT_DOUBLE_EQ(params.metric_velocity_floor, 0.5);
}

TEST(PaperGdsCollision, InvalidParametersAreRejected)
{
  auto params = make_params();
  params.metric_velocity_scale = 0.0;
  EXPECT_FALSE(collision::parameters_are_valid(params));
  EXPECT_THROW(
    collision::evaluate(0.1, 0.0, params),
    std::invalid_argument);
}

}  // namespace
