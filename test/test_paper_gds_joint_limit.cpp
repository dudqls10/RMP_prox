#include <cmath>
#include <limits>

#include <gtest/gtest.h>

#include "rb10_rmpflow_rviz/paper_gds_joint_limit.hpp"

namespace
{

using rb10_rmpflow_rviz::PaperGdsJointLimitConfig;
using rb10_rmpflow_rviz::PaperGdsJointLimitStatus;
using rb10_rmpflow_rviz::evaluate_paper_gds_joint_limit;
using rb10_rmpflow_rviz::valid_paper_gds_joint_limit_config;

PaperGdsJointLimitConfig test_config()
{
  PaperGdsJointLimitConfig config;
  config.lower = -1.2;
  config.upper = 2.3;
  config.center_fraction = 0.37;
  config.task_metric = 2.4;
  config.potential_gain = 3.2;
  config.damping = 1.7;
  config.boundary_epsilon = 1e-12;
  return config;
}

double q_at_fraction(const PaperGdsJointLimitConfig & config, double fraction)
{
  return config.lower + fraction * (config.upper - config.lower);
}

TEST(PaperGdsJointLimit, TaskMapJacobianAndCurvatureMatchFiniteDifferences)
{
  const auto config = test_config();
  const double q = 0.41;
  const double qdot = -0.73;
  const double spatial_step = 1e-6;
  const double time_step = 1e-6;

  const auto center = evaluate_paper_gds_joint_limit(config, q, qdot);
  const auto spatial_plus =
    evaluate_paper_gds_joint_limit(config, q + spatial_step, qdot);
  const auto spatial_minus =
    evaluate_paper_gds_joint_limit(config, q - spatial_step, qdot);
  ASSERT_TRUE(center.valid);
  ASSERT_TRUE(spatial_plus.valid);
  ASSERT_TRUE(spatial_minus.valid);

  const double finite_difference_jacobian =
    (spatial_plus.z - spatial_minus.z) / (2.0 * spatial_step);
  const double finite_difference_jacobian_derivative =
    (spatial_plus.jacobian - spatial_minus.jacobian) /
    (2.0 * spatial_step);
  EXPECT_NEAR(finite_difference_jacobian, center.jacobian, 1e-9);
  EXPECT_NEAR(
    finite_difference_jacobian_derivative,
    center.jacobian_derivative,
    1e-8);

  const auto time_plus =
    evaluate_paper_gds_joint_limit(config, q + qdot * time_step, qdot);
  const auto time_minus =
    evaluate_paper_gds_joint_limit(config, q - qdot * time_step, qdot);
  ASSERT_TRUE(time_plus.valid);
  ASSERT_TRUE(time_minus.valid);
  const double finite_difference_curvature =
    (time_plus.zdot - time_minus.zdot) / (2.0 * time_step);
  EXPECT_NEAR(finite_difference_curvature, center.curvature, 1e-8);
}

TEST(PaperGdsJointLimit, NaturalDynamicsHaveExactDissipation)
{
  const auto config = test_config();
  const auto result = evaluate_paper_gds_joint_limit(config, 0.18, -0.62);
  ASSERT_TRUE(result.valid);

  const double task_energy_derivative =
    result.task_metric * result.zdot * result.task_acceleration +
    result.potential_gradient * result.zdot;
  EXPECT_NEAR(
    task_energy_derivative,
    -config.damping * result.zdot * result.zdot,
    1e-12);
  EXPECT_NEAR(task_energy_derivative, result.energy_rate, 1e-12);
  EXPECT_DOUBLE_EQ(result.task_metric, config.task_metric);
  EXPECT_DOUBLE_EQ(result.task_inertia, config.task_metric);

  // A solve using only this root contribution must reconstruct the task-space
  // acceleration after the nonlinear task-map curvature is added back.
  const double reconstructed_task_acceleration =
    result.jacobian * result.root_acceleration + result.curvature;
  EXPECT_NEAR(
    reconstructed_task_acceleration,
    result.task_acceleration,
    1e-12);
  EXPECT_NEAR(
    result.root_natural_force,
    -result.root_potential_gradient -
    result.root_damping * result.qdot -
    result.root_curvature_force,
    1e-12);

  // The same dissipation identity in q coordinates includes the pullback
  // curvature force of G_q = task_metric * J(q)^2.
  const double root_energy_derivative =
    result.root_inertia * result.qdot * result.root_acceleration +
    result.root_curvature_force * result.qdot +
    result.root_potential_gradient * result.qdot;
  EXPECT_NEAR(
    root_energy_derivative,
    -result.root_damping * result.qdot * result.qdot,
    1e-12);
  EXPECT_NEAR(root_energy_derivative, result.energy_rate, 1e-12);
}

TEST(PaperGdsJointLimit, ConfigurableCenterHasZeroPotentialForce)
{
  const auto config = test_config();
  const double center_q = q_at_fraction(config, config.center_fraction);
  const auto result = evaluate_paper_gds_joint_limit(config, center_q, 0.0);
  ASSERT_TRUE(result.valid);

  EXPECT_NEAR(result.center_q, center_q, 1e-15);
  EXPECT_NEAR(result.z, result.center_z, 1e-15);
  EXPECT_NEAR(result.potential, 0.0, 1e-28);
  EXPECT_NEAR(result.potential_gradient, 0.0, 1e-14);
  EXPECT_NEAR(result.task_natural_force, 0.0, 1e-14);
  EXPECT_NEAR(result.root_natural_force, 0.0, 1e-13);
}

TEST(PaperGdsJointLimit, PotentialGrowsTowardBothOpenBoundaries)
{
  auto config = test_config();
  config.boundary_epsilon = 0.0;

  const auto lower_far =
    evaluate_paper_gds_joint_limit(config, q_at_fraction(config, 1e-2), 0.0);
  const auto lower_near =
    evaluate_paper_gds_joint_limit(config, q_at_fraction(config, 1e-6), 0.0);
  const auto lower_nearest =
    evaluate_paper_gds_joint_limit(config, q_at_fraction(config, 1e-10), 0.0);
  const auto upper_far =
    evaluate_paper_gds_joint_limit(config, q_at_fraction(config, 1.0 - 1e-2), 0.0);
  const auto upper_near =
    evaluate_paper_gds_joint_limit(config, q_at_fraction(config, 1.0 - 1e-6), 0.0);
  const auto upper_nearest =
    evaluate_paper_gds_joint_limit(config, q_at_fraction(config, 1.0 - 1e-10), 0.0);

  ASSERT_TRUE(lower_far.valid);
  ASSERT_TRUE(lower_near.valid);
  ASSERT_TRUE(lower_nearest.valid);
  ASSERT_TRUE(upper_far.valid);
  ASSERT_TRUE(upper_near.valid);
  ASSERT_TRUE(upper_nearest.valid);

  EXPECT_LT(lower_far.potential, lower_near.potential);
  EXPECT_LT(lower_near.potential, lower_nearest.potential);
  EXPECT_LT(upper_far.potential, upper_near.potential);
  EXPECT_LT(upper_near.potential, upper_nearest.potential);
  EXPECT_GT(lower_nearest.potential, 100.0);
  EXPECT_GT(upper_nearest.potential, 100.0);
}

TEST(PaperGdsJointLimit, RejectsInvalidDomainWithoutClamping)
{
  auto config = test_config();
  config.lower = -1.0;
  config.upper = 1.0;
  config.boundary_epsilon = 1e-4;

  const auto below =
    evaluate_paper_gds_joint_limit(config, config.lower - 1e-3, 0.0);
  const auto at_lower =
    evaluate_paper_gds_joint_limit(config, config.lower, 0.0);
  const auto inside_epsilon =
    evaluate_paper_gds_joint_limit(
      config,
      config.lower + 0.5 * config.boundary_epsilon,
      0.0);
  const auto inside_upper_epsilon =
    evaluate_paper_gds_joint_limit(
      config,
      config.upper - 0.5 * config.boundary_epsilon,
      0.0);
  const auto valid_near_boundary =
    evaluate_paper_gds_joint_limit(
      config,
      config.lower + 2.0 * config.boundary_epsilon,
      0.0);
  const auto at_upper =
    evaluate_paper_gds_joint_limit(config, config.upper, 0.0);
  const auto above =
    evaluate_paper_gds_joint_limit(config, config.upper + 1e-3, 0.0);

  EXPECT_FALSE(below.valid);
  EXPECT_EQ(below.status, PaperGdsJointLimitStatus::kOutsideOpenInterval);
  EXPECT_FALSE(at_lower.valid);
  EXPECT_EQ(at_lower.status, PaperGdsJointLimitStatus::kOutsideOpenInterval);
  EXPECT_FALSE(inside_epsilon.valid);
  EXPECT_EQ(
    inside_epsilon.status,
    PaperGdsJointLimitStatus::kInsideBoundaryEpsilon);
  EXPECT_NEAR(
    inside_epsilon.lower_clearance,
    0.5 * config.boundary_epsilon,
    1e-15);
  EXPECT_FALSE(inside_upper_epsilon.valid);
  EXPECT_EQ(
    inside_upper_epsilon.status,
    PaperGdsJointLimitStatus::kInsideBoundaryEpsilon);
  ASSERT_TRUE(valid_near_boundary.valid);
  EXPECT_DOUBLE_EQ(
    valid_near_boundary.q,
    config.lower + 2.0 * config.boundary_epsilon);
  EXPECT_NEAR(
    valid_near_boundary.z,
    std::log(valid_near_boundary.lower_clearance) -
    std::log(valid_near_boundary.upper_clearance),
    1e-15);
  EXPECT_FALSE(at_upper.valid);
  EXPECT_EQ(at_upper.status, PaperGdsJointLimitStatus::kOutsideOpenInterval);
  EXPECT_FALSE(above.valid);
  EXPECT_EQ(above.status, PaperGdsJointLimitStatus::kOutsideOpenInterval);

  const auto nonfinite = evaluate_paper_gds_joint_limit(
    config,
    0.0,
    std::numeric_limits<double>::quiet_NaN());
  EXPECT_FALSE(nonfinite.valid);
  EXPECT_EQ(nonfinite.status, PaperGdsJointLimitStatus::kNonFiniteState);
}

TEST(PaperGdsJointLimit, RejectsConfigurationsThatAreNotStrictGdsParameters)
{
  auto config = test_config();
  EXPECT_TRUE(valid_paper_gds_joint_limit_config(config));

  config.upper = config.lower;
  EXPECT_FALSE(valid_paper_gds_joint_limit_config(config));
  EXPECT_EQ(
    evaluate_paper_gds_joint_limit(config, 0.0, 0.0).status,
    PaperGdsJointLimitStatus::kInvalidConfiguration);

  config = test_config();
  config.center_fraction = 1.0;
  EXPECT_FALSE(valid_paper_gds_joint_limit_config(config));

  config = test_config();
  config.center_fraction = 1e-15;
  EXPECT_FALSE(valid_paper_gds_joint_limit_config(config));

  config = test_config();
  config.task_metric = 0.0;
  EXPECT_FALSE(valid_paper_gds_joint_limit_config(config));

  config = test_config();
  config.potential_gain = 0.0;
  EXPECT_FALSE(valid_paper_gds_joint_limit_config(config));

  config = test_config();
  config.damping = 0.0;
  EXPECT_FALSE(valid_paper_gds_joint_limit_config(config));

  config = test_config();
  config.boundary_epsilon = 0.5 * (config.upper - config.lower);
  EXPECT_FALSE(valid_paper_gds_joint_limit_config(config));
}

}  // namespace
