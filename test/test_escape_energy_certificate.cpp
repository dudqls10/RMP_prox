#include <cmath>
#include <limits>

#include <gtest/gtest.h>

#include <Eigen/Core>

#include "rb10_rmpflow_rviz/escape_energy_certificate.hpp"

namespace
{

using rb10_rmpflow_rviz::BaseEnergyBalance;
using rb10_rmpflow_rviz::EpisodeEnergyIntegrals;
using rb10_rmpflow_rviz::compute_base_energy_balance;
using rb10_rmpflow_rviz::compute_direct_escape_power;
using rb10_rmpflow_rviz::compute_rank_one_escape_power;
using rb10_rmpflow_rviz::compute_rank_one_metric_cap;
using rb10_rmpflow_rviz::integrate_episode_energy;
using rb10_rmpflow_rviz::update_positive_excess_energy_tank;

TEST(EscapeEnergyCertificate, DirectAndRankOnePowerAgreeForScalarSolve)
{
  Eigen::MatrixXd base_metric(1, 1);
  base_metric(0, 0) = 2.0;
  Eigen::RowVectorXd jacobian(1);
  jacobian[0] = 1.5;
  Eigen::VectorXd base_qdd(1);
  base_qdd[0] = 0.5;

  constexpr double metric = 3.0;
  constexpr double qd = 0.4;
  constexpr double tangent_velocity = 1.5 * qd;
  constexpr double desired_acceleration = 1.1;
  constexpr double scalar_curvature = 0.2;
  constexpr double base_force = 1.0;

  const double combined_metric =
    base_metric(0, 0) + metric * jacobian[0] * jacobian[0];
  const double combined_force =
    base_force +
    metric * jacobian[0] *
    (desired_acceleration - scalar_curvature);
  const double combined_qdd = combined_force / combined_metric;
  const double realized_acceleration =
    jacobian[0] * combined_qdd + scalar_curvature;

  const auto direct = compute_direct_escape_power(
    metric,
    tangent_velocity,
    desired_acceleration,
    realized_acceleration);
  const auto rank_one = compute_rank_one_escape_power(
    base_metric,
    jacobian,
    base_qdd,
    metric,
    tangent_velocity,
    desired_acceleration,
    scalar_curvature);

  ASSERT_TRUE(direct.valid);
  ASSERT_TRUE(rank_one.valid);
  EXPECT_NEAR(rank_one.compliance, 1.125, 1e-12);
  EXPECT_NEAR(
    rank_one.effective_metric,
    metric / (1.0 + metric * rank_one.compliance),
    1e-12);
  EXPECT_NEAR(direct.power, rank_one.power, 1e-12);
}

TEST(EscapeEnergyCertificate, MetricCapHandlesAllAnalyticCases)
{
  const auto dissipative = compute_rank_one_metric_cap(
    0.5, -2.0, 0.0, 10.0);
  ASSERT_TRUE(dissipative.valid);
  EXPECT_FALSE(dissipative.limited);
  EXPECT_DOUBLE_EQ(dissipative.allowed_metric, 10.0);
  EXPECT_DOUBLE_EQ(dissipative.metric_scale, 1.0);
  EXPECT_LT(dissipative.allowed_power, 0.0);

  const auto asymptotically_safe = compute_rank_one_metric_cap(
    2.0, 1.0, 0.6, 100.0);
  ASSERT_TRUE(asymptotically_safe.valid);
  EXPECT_FALSE(asymptotically_safe.limited);
  EXPECT_TRUE(std::isinf(asymptotically_safe.metric_cap));
  EXPECT_LE(asymptotically_safe.allowed_power, 0.6);

  const auto capped = compute_rank_one_metric_cap(
    0.5, 2.0, 0.5, 10.0);
  ASSERT_TRUE(capped.valid);
  EXPECT_TRUE(capped.limited);
  EXPECT_NEAR(capped.metric_cap, 0.5 / 1.75, 1e-12);
  EXPECT_NEAR(capped.allowed_metric, capped.metric_cap, 1e-12);
  EXPECT_NEAR(capped.allowed_power, 0.5, 1e-12);
  EXPECT_GE(capped.metric_scale, 0.0);
  EXPECT_LE(capped.metric_scale, 1.0);

  const auto zero_budget = compute_rank_one_metric_cap(
    0.5, 2.0, 0.0, 10.0);
  ASSERT_TRUE(zero_budget.valid);
  EXPECT_TRUE(zero_budget.limited);
  EXPECT_DOUBLE_EQ(zero_budget.allowed_metric, 0.0);
  EXPECT_DOUBLE_EQ(zero_budget.metric_scale, 0.0);
  EXPECT_DOUBLE_EQ(zero_budget.allowed_power, 0.0);

  const auto unlimited = compute_rank_one_metric_cap(
    0.5,
    2.0,
    std::numeric_limits<double>::infinity(),
    10.0);
  ASSERT_TRUE(unlimited.valid);
  EXPECT_FALSE(unlimited.limited);
  EXPECT_TRUE(std::isinf(unlimited.positive_power_budget));
  EXPECT_DOUBLE_EQ(unlimited.allowed_metric, 10.0);
  EXPECT_DOUBLE_EQ(unlimited.metric_scale, 1.0);
}

TEST(EscapeEnergyCertificate, PositiveExcessTankNeverBecomesNegative)
{
  const auto update = update_positive_excess_energy_tank(
    0.1,
    0.5,
    3.0,
    2.0,
    10.0,
    30.0);

  ASSERT_TRUE(update.valid);
  EXPECT_NEAR(update.passive_positive_power_budget, 5.0, 1e-12);
  EXPECT_NEAR(update.tank_discharge_power_budget, 20.0, 1e-12);
  EXPECT_NEAR(update.allowable_positive_power_budget, 25.0, 1e-12);
  EXPECT_NEAR(update.power_scale, 25.0 / 30.0, 1e-12);
  EXPECT_NEAR(update.positive_excess_power, 20.0, 1e-12);
  EXPECT_GE(update.new_energy, 0.0);
  EXPECT_LE(update.new_energy, update.capacity);
  EXPECT_NEAR(update.new_energy, 0.5, 1e-12);
  EXPECT_LE(update.composite_storage_rate, 1e-12);

  const auto empty = update_positive_excess_energy_tank(
    0.1,
    0.25,
    1.0,
    0.0,
    4.0,
    1000.0);
  ASSERT_TRUE(empty.valid);
  EXPECT_GE(empty.new_energy, 0.0);
  EXPECT_LE(empty.new_energy, empty.capacity);
  EXPECT_GE(empty.power_scale, 0.0);
  EXPECT_LE(empty.power_scale, 1.0);
  EXPECT_LE(empty.composite_storage_rate, 1e-12);

  const auto zero_dt = update_positive_excess_energy_tank(
    0.0,
    0.5,
    2.0,
    1.0,
    10.0,
    10.0);
  ASSERT_TRUE(zero_dt.valid);
  EXPECT_DOUBLE_EQ(zero_dt.new_energy, 1.0);
  EXPECT_GE(zero_dt.power_scale, 0.0);
  EXPECT_LE(zero_dt.power_scale, 1.0);
}

TEST(EscapeEnergyCertificate, BaseBalanceMarginHasCorrectSign)
{
  const BaseEnergyBalance decreasing =
    compute_base_energy_balance(10.0, 3.0, 1.0, 2.0, 1.0);
  ASSERT_TRUE(decreasing.valid);
  EXPECT_DOUBLE_EQ(decreasing.total_injection_power, 7.0);
  EXPECT_DOUBLE_EQ(decreasing.nonincrease_margin, 3.0);
  EXPECT_DOUBLE_EQ(decreasing.base_lyapunov_rate, -3.0);

  const BaseEnergyBalance increasing =
    compute_base_energy_balance(5.0, 4.0, 1.0, 1.0, 1.0);
  ASSERT_TRUE(increasing.valid);
  EXPECT_DOUBLE_EQ(increasing.total_injection_power, 7.0);
  EXPECT_DOUBLE_EQ(increasing.nonincrease_margin, -2.0);
  EXPECT_DOUBLE_EQ(increasing.base_lyapunov_rate, 2.0);
}

TEST(EscapeEnergyCertificate, EpisodeIntegralsSeparateSignedPower)
{
  EpisodeEnergyIntegrals episode;
  const auto balance =
    compute_base_energy_balance(4.0, 3.0, 0.5, -0.25, 0.25);
  ASSERT_TRUE(balance.valid);
  episode = integrate_episode_energy(episode, 0.2, 0.5, balance);

  EXPECT_EQ(episode.accepted_samples, 1U);
  EXPECT_EQ(episode.rejected_samples, 0U);
  EXPECT_NEAR(episode.elapsed_time, 0.2, 1e-12);
  EXPECT_NEAR(episode.damping_energy, 0.8, 1e-12);
  EXPECT_NEAR(episode.escape_energy, 0.6, 1e-12);
  EXPECT_NEAR(episode.positive_escape_energy, 0.6, 1e-12);
  EXPECT_DOUBLE_EQ(episode.negative_escape_energy, 0.0);
  EXPECT_NEAR(episode.positive_excess_escape_energy, 0.2, 1e-12);
  EXPECT_NEAR(
    episode.base_lyapunov_change,
    balance.base_lyapunov_rate * 0.2,
    1e-12);

  const auto rejected = integrate_episode_energy(
    episode, -0.1, 0.5, balance);
  EXPECT_EQ(rejected.accepted_samples, 1U);
  EXPECT_EQ(rejected.rejected_samples, 1U);
}

}  // namespace
