#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include "rb10_rmpflow_rviz/joint_acceleration_limiter.hpp"
#include "rb10_rmpflow_rviz/pinocchio_direct_solver.hpp"
#include "rb10_rmpflow_rviz/pinocchio_model.hpp"

namespace
{

using rb10_rmpflow_rviz::EigenRmpConfig;
using rb10_rmpflow_rviz::KinematicsContext;
using rb10_rmpflow_rviz::JointAccelerationVector;
using rb10_rmpflow_rviz::ObstacleSphere;
using rb10_rmpflow_rviz::PinocchioDirectRmpSolver;
using rb10_rmpflow_rviz::PinocchioModel;
using rb10_rmpflow_rviz::RB10Model;
using rb10_rmpflow_rviz::RmpSolveResult;

using JointVector = RB10Model::JointVector;
using Matrix6 = Eigen::Matrix<double, 6, 6>;

constexpr std::size_t kDebugActive = 0;
constexpr std::size_t kDebugControlPoint = 1;
constexpr std::size_t kDebugClearance = 2;
constexpr std::size_t kDebugAlphaDistance = 4;
constexpr std::size_t kDebugAlphaBlocking = 5;
constexpr std::size_t kDebugActivation = 6;
constexpr std::size_t kDebugScalarAcceleration = 9;
constexpr std::size_t kDebugEffectiveMetric = 10;
constexpr std::size_t kDebugOutwardNormal = 17;
constexpr std::size_t kDebugTangent = 20;
constexpr std::size_t kDebugLeafMode = 26;
constexpr std::size_t kDebugObstacleEpisode = 31;
constexpr std::size_t kDebugHandoffPhase = 33;
constexpr std::size_t kDebugCandidateCount = 37;
constexpr std::size_t kDebugLambda = 45;
constexpr std::size_t kDebugHighLevelState = 47;
constexpr std::size_t kDebugRawActivation = 50;
constexpr std::size_t kDebugFilteredClearanceRate = 61;
constexpr std::size_t kDebugClosingSpeed = 62;
constexpr std::size_t kDebugRiskDistanceAcceleration = 63;
constexpr std::size_t kDebugRiskApproachAcceleration = 64;
constexpr std::size_t kDebugRiskAcceleration = 65;
constexpr std::size_t kDebugRiskDampingAcceleration = 66;
constexpr std::size_t kDebugRiskAccelerationInput = 67;
constexpr std::size_t kRequiredActiveDebugSize = kDebugHighLevelState + 1;
constexpr std::size_t kRequiredRiskDebugSize = kDebugRiskAccelerationInput + 1;

[[noreturn]] void fail(const std::string & message)
{
  throw std::runtime_error(message);
}

void require(bool condition, const std::string & message)
{
  if (!condition) {
    fail(message);
  }
}

bool finite_result(const RmpSolveResult & result)
{
  return result.qdd.allFinite() &&
         result.metric.allFinite() &&
         result.force.allFinite();
}

bool finite_vector(const std::vector<double> & values)
{
  return std::all_of(
    values.begin(),
    values.end(),
    [](double value) {return std::isfinite(value);});
}

bool approximately_integer(double value, int expected, double tolerance = 1e-9)
{
  return std::isfinite(value) &&
         std::abs(value - static_cast<double>(expected)) <= tolerance;
}

EigenRmpConfig make_test_config(bool escape_enabled)
{
  EigenRmpConfig config;
  config.solve_method = "rmp2";
  config.rmp_type = "canonical";
  config.solve_offset = 1e-4;

  std::size_t tangent_node_count = 0;
  for (auto & node : config.graph_nodes) {
    if (node.name == "cspace_target") {
      node.enabled = false;
    }
    if (
      node.name == "tangent_escape" ||
      node.leaf_rmp_type == "tangent_escape" ||
      node.handcrafted_leaf_rmp_type == "tangent_escape")
    {
      node.enabled = true;
      ++tangent_node_count;
    }
  }
  require(
    tangent_node_count == 1,
    "test configuration must enable exactly one tangent_escape graph leaf");

  config.target.accel_p_gain = 30.0;
  config.target.accel_d_gain = 10.0;
  config.target.accel_norm_eps = 0.05;
  config.target.max_metric_scalar = 1.0;
  config.target.min_metric_scalar = 0.3;
  config.target.min_metric_alpha = 0.05;
  config.target.proximity_metric_boost_scalar = 1.0;

  config.damping.accel_d_gain = 2.0;
  config.damping.metric_scalar = 0.001;
  config.damping.inertia = 0.01;

  config.collision.margin = 0.0;
  config.collision.repulsion_gain = 20.0;
  config.collision.repulsion_std_dev = 0.05;
  config.collision.damping_gain = 5.0;
  config.collision.damping_std_dev = 0.05;
  config.collision.metric_scalar = 0.2;
  config.collision.metric_modulation_radius = 0.5;
  config.collision.metric_exploder_std_dev = 0.05;
  config.collision.metric_exploder_eps = 0.01;

  config.joint_velocity_cap.max_velocity = 1e6;
  config.joint_velocity_cap.velocity_damping_region = 1.0;
  config.joint_velocity_cap.metric_weight = 0.0;
  config.max_joint_accel = 1e6;
  config.joint_limit_buffers.fill(0.0);

  auto & escape = config.tangent_escape;
  escape.enabled = escape_enabled;
  escape.metric_scalar = 20.0;
  escape.clearance_margin = 0.0;
  escape.safe_distance = 0.10;
  escape.influence_distance = 0.40;
  escape.goal_block_beta_on = 0.10;
  escape.goal_block_beta_full = 0.60;
  escape.nominal_prediction_dt = 0.05;
  escape.min_activation = 0.0;
  escape.min_tangent_norm = 1e-8;
  escape.normal_tolerance = 0.25;
  escape.control_dt = 0.01;

  escape.prevent_weight = 0.5;
  escape.activation_time_constant = 0.01;
  escape.activation_rise_rate = 100.0;
  escape.activation_fall_rate = 100.0;
  escape.progress_filter_time_constant = 0.01;
  escape.progress_low_threshold = 0.001;
  escape.progress_ok_threshold = 0.01;
  escape.still_speed_threshold = 0.001;
  escape.moving_speed_threshold = 0.10;
  escape.intent_on_speed = 1e-10;
  escape.intent_full_speed = 1e-8;
  escape.prevent_speed = 0.02;
  escape.recovery_speed = 0.05;
  escape.desired_velocity_time_constant = 0.01;
  escape.velocity_gain = 10.0;
  escape.max_accel = 20.0;
  escape.release_stop_speed = 1e-5;
  escape.release_hold_speed = 0.10;
  escape.drive_ramp_duration = 0.02;
  escape.handoff_duration = 0.02;
  escape.minimum_drive_duration = 0.0;
  escape.release_blockage_threshold = 0.0;

  escape.candidate_lookahead = 0.10;
  escape.refinement_iterations = 2;
  escape.pair_switch_margin = 1.0;
  escape.direction_switch_margin = 1.0;
  escape.goal_score_scale = 0.10;
  escape.sector_risk_scale = 1e6;
  escape.sector_risk_hard_limit = 1e6;
  escape.accel_jump_scale = 1e6;
  escape.candidate_min_displacement = 1e-7;
  escape.command_test_distance = 10.0;
  escape.minimum_move_ratio = 0.0;
  escape.goal_change_reset_distance = 10.0;

  return config;
}

EigenRmpConfig make_risk_damped_test_config(bool escape_enabled)
{
  EigenRmpConfig config = make_test_config(escape_enabled);
  auto & escape = config.tangent_escape;
  escape.acceleration_model = "risk_damped";
  escape.metric_scalar = 20.0;
  escape.max_accel = 1.0;
  escape.max_speed = 0.20;
  escape.risk_distance_gain = 0.40;
  escape.risk_distance_scale = 0.08;
  escape.risk_approach_gain = 0.50;
  escape.risk_approach_distance_scale = 0.08;
  escape.risk_approach_epsilon = 0.25;
  escape.risk_velocity_gate_scale = 0.03;
  escape.risk_clearance_rate_filter_time_constant = 0.01;
  escape.risk_tangent_damping_gain = 4.0;
  return config;
}

std::vector<Eigen::Vector3d> goal_directions()
{
  std::vector<Eigen::Vector3d> directions{
    Eigen::Vector3d::UnitX(),
    -Eigen::Vector3d::UnitX(),
    Eigen::Vector3d::UnitY(),
    -Eigen::Vector3d::UnitY(),
    Eigen::Vector3d::UnitZ(),
    -Eigen::Vector3d::UnitZ(),
  };
  const std::array<Eigen::Vector3d, 8> diagonal_directions{{
    {1.0, 1.0, 1.0},
    {1.0, 1.0, -1.0},
    {1.0, -1.0, 1.0},
    {1.0, -1.0, -1.0},
    {-1.0, 1.0, 1.0},
    {-1.0, 1.0, -1.0},
    {-1.0, -1.0, 1.0},
    {-1.0, -1.0, -1.0},
  }};
  for (const auto & direction : diagonal_directions) {
    directions.push_back(direction.normalized());
  }
  return directions;
}

struct NominalScenario
{
  Eigen::Vector3d goal{Eigen::Vector3d::Zero()};
  std::size_t control_point_index{0};
  Eigen::Vector3d control_point_motion{Eigen::Vector3d::Zero()};
  double speed{0.0};
  double tangent_mobility{0.0};
  double selection_score{0.0};
};

NominalScenario find_strongest_nominal_scenario(
  PinocchioDirectRmpSolver & disabled_solver,
  const KinematicsContext & context,
  const JointVector & q,
  const JointVector & qd,
  double nominal_preview_dt)
{
  NominalScenario best;
  for (const auto & direction : goal_directions()) {
    const Eigen::Vector3d goal = context.tcp_position + 0.25 * direction;
    const std::unordered_map<std::string, Eigen::Vector3d> targets{{"goal", goal}};
    const RmpSolveResult result = disabled_solver.solve(q, qd, targets, {});
    require(finite_result(result), "non-Escape nominal probe solve is not finite");

    for (std::size_t index = 0; index < context.control_point_jacobians.size(); ++index) {
      const Eigen::Vector3d motion =
        context.control_point_jacobians[index] *
        (qd + nominal_preview_dt * result.qdd);
      const double speed = motion.norm();
      if (speed <= 1e-10) {
        continue;
      }
      const Eigen::Vector3d approach_direction = motion / speed;
      const Eigen::Matrix3d tangent_projector =
        Eigen::Matrix3d::Identity() -
        approach_direction * approach_direction.transpose();
      const double tangent_mobility =
        (tangent_projector * context.control_point_jacobians[index]).norm();
      const double selection_score = speed * tangent_mobility;
      if (tangent_mobility > 1e-6 && selection_score > best.selection_score) {
        best.goal = goal;
        best.control_point_index = index;
        best.control_point_motion = motion;
        best.speed = speed;
        best.tangent_mobility = tangent_mobility;
        best.selection_score = selection_score;
      }
    }
  }

  require(best.speed > 1e-8, "could not create a usable nominal control-point motion");
  require(
    best.tangent_mobility > 1e-6,
    "could not create a nominal scenario with realizable tangent motion");
  require(
    best.control_point_index < context.control_points.size(),
    "selected nominal control-point index is invalid");
  return best;
}

struct ScalarContributionCheck
{
  int numerical_rank{0};
  double norm{0.0};
  double minimum_eigenvalue{0.0};
  double force_off_axis_norm{0.0};
};

ScalarContributionCheck check_single_scalar_contribution(
  const Matrix6 & delta_metric_raw,
  const JointVector & delta_force)
{
  const Matrix6 delta_metric =
    0.5 * (delta_metric_raw + delta_metric_raw.transpose());
  Eigen::SelfAdjointEigenSolver<Matrix6> eigen_solver(delta_metric);
  require(
    eigen_solver.info() == Eigen::Success,
    "Escape delta metric eigendecomposition failed");

  const auto eigenvalues = eigen_solver.eigenvalues();
  const double norm = delta_metric.norm();
  const double tolerance = 1e-8 * std::max(1.0, norm);
  const double minimum_eigenvalue = eigenvalues.minCoeff();
  require(
    minimum_eigenvalue >= -tolerance,
    "Escape delta metric is not positive semidefinite");

  int rank = 0;
  for (Eigen::Index index = 0; index < eigenvalues.size(); ++index) {
    if (eigenvalues[index] > tolerance) {
      ++rank;
    }
  }
  require(rank <= 1, "Escape delta metric has numerical rank greater than one");

  double force_off_axis_norm = 0.0;
  if (rank == 1) {
    Eigen::Index dominant_index = 0;
    eigenvalues.maxCoeff(&dominant_index);
    const JointVector axis = eigen_solver.eigenvectors().col(dominant_index);
    const JointVector residual = delta_force - axis * axis.dot(delta_force);
    force_off_axis_norm = residual.norm();
    const double force_tolerance = 1e-6 * std::max(1.0, delta_force.norm());
    require(
      force_off_axis_norm <= force_tolerance,
      "Escape delta force is not in the single scalar leaf metric direction");
  } else {
    require(
      delta_force.norm() <= 1e-7,
      "zero-rank Escape delta metric has a nonzero delta force");
  }

  return ScalarContributionCheck{rank, norm, minimum_eigenvalue, force_off_axis_norm};
}

Eigen::Vector3d debug_vector3(
  const std::vector<double> & debug,
  std::size_t offset,
  const std::string & name)
{
  require(debug.size() >= offset + 3, "debug vector is missing " + name);
  const Eigen::Vector3d value(debug[offset], debug[offset + 1], debug[offset + 2]);
  require(value.allFinite(), name + " is not finite");
  return value;
}

void run_test(const std::string & urdf_path)
{
  const JointVector q = JointVector::Zero();
  const JointVector qd = JointVector::Zero();

  PinocchioModel model(urdf_path);
  const KinematicsContext context = model.forward_context(q, qd);
  require(
    context.control_points.size() == RB10Model::sensor_control_points.size(),
    "Pinocchio model did not produce all sensor control points");

  EigenRmpConfig disabled_config = make_test_config(false);
  PinocchioDirectRmpSolver disabled_solver(disabled_config, urdf_path);
  const NominalScenario nominal = find_strongest_nominal_scenario(
    disabled_solver,
    context,
    q,
    qd,
    disabled_config.tangent_escape.nominal_prediction_dt);

  const Eigen::Vector3d approach_direction = nominal.control_point_motion.normalized();
  const auto & selected_cp = context.control_points[nominal.control_point_index];
  const double obstacle_radius = 0.04;
  const double clearance = 0.02;
  const double center_distance = selected_cp.radius + obstacle_radius + clearance;
  ObstacleSphere obstacle{
    selected_cp.position + center_distance * approach_direction,
    obstacle_radius,
    static_cast<int>(nominal.control_point_index)};
  obstacle.source_id = 101;
  const std::vector<ObstacleSphere> obstacles{obstacle};
  const std::unordered_map<std::string, Eigen::Vector3d> targets{{"goal", nominal.goal}};

  const RmpSolveResult disabled_result =
    disabled_solver.solve(q, qd, targets, obstacles);
  require(finite_result(disabled_result), "Escape-disabled reference solve is not finite");

  EigenRmpConfig enabled_config = disabled_config;
  enabled_config.tangent_escape.enabled = true;
  PinocchioDirectRmpSolver enabled_solver(enabled_config, urdf_path);

  bool saw_active = false;
  bool saw_nonzero_contribution = false;
  bool saw_candidate_records = false;
  bool checked_lambda_zero = false;
  int active_sample_count = 0;
  double maximum_lambda = 0.0;
  double maximum_effective_metric = 0.0;
  double maximum_contribution_norm = 0.0;
  double maximum_scalar_jacobian_norm = 0.0;
  std::size_t last_active_control_point = 0;
  Eigen::Vector3d last_active_tangent = Eigen::Vector3d::Zero();
  bool last_active_direction_valid = false;

  constexpr int kCycles = 60;
  for (int cycle = 0; cycle < kCycles; ++cycle) {
    const RmpSolveResult enabled_result =
      enabled_solver.solve(q, qd, targets, obstacles);
    require(finite_result(enabled_result), "canonical Escape solve is not finite");
    const auto & dual = enabled_result.tangent_escape_dual_solve_data;
    require(
      dual.size() == 43 && finite_vector(dual),
      "same-state Escape dual-solve schema is missing or non-finite");

    const Matrix6 delta_metric = enabled_result.metric - disabled_result.metric;
    const JointVector delta_force = enabled_result.force - disabled_result.force;
    const ScalarContributionCheck contribution =
      check_single_scalar_contribution(delta_metric, delta_force);

    const auto & debug = enabled_result.tangent_escape_rmp_data;
    const bool active =
      debug.size() >= kRequiredActiveDebugSize &&
      std::isfinite(debug[kDebugActive]) &&
      debug[kDebugActive] > 0.5;
    if (!active) {
      require(
        contribution.norm <= 1e-7,
        "inactive canonical Escape sample has a nonzero delta metric");
      require(
        dual[0] <= 0.5,
        "inactive canonical Escape sample reports an active dual solve");
      continue;
    }

    saw_active = true;
    ++active_sample_count;
    require(
      approximately_integer(debug[kDebugLeafMode], 6),
      "active canonical Escape sample did not report wire schema id 6");
    require(
      std::isfinite(debug[kDebugCandidateCount]) &&
      debug[kDebugCandidateCount] >= 0.0,
      "active canonical Escape candidate record count is invalid");
    saw_candidate_records =
      saw_candidate_records || debug[kDebugCandidateCount] >= 1.0;
    require(
      std::isfinite(debug[kDebugAlphaDistance]) &&
      debug[kDebugAlphaDistance] > 0.0,
      "active canonical Escape sample has no distance activation");
    require(
      std::isfinite(debug[kDebugAlphaBlocking]) &&
      debug[kDebugAlphaBlocking] > 0.0,
      "active canonical Escape sample has no blocking activation");
    require(
      std::isfinite(debug[kDebugActivation]) &&
      debug[kDebugActivation] >= 0.0 &&
      debug[kDebugActivation] <= 1.0 + 1e-9,
      "canonical Escape activation is outside [0, 1]");
    require(
      std::isfinite(debug[kDebugHighLevelState]) &&
      debug[kDebugHighLevelState] >= 0.0 &&
      debug[kDebugHighLevelState] <= 4.0,
      "canonical Escape high-level state id is invalid");

    const Eigen::Vector3d outward_normal =
      debug_vector3(debug, kDebugOutwardNormal, "outward normal");
    const Eigen::Vector3d tangent =
      debug_vector3(debug, kDebugTangent, "world tangent");
    const auto debug_control_point =
      static_cast<std::size_t>(std::llround(debug[kDebugControlPoint]));
    require(
      debug_control_point < context.control_point_jacobians.size(),
      "reported canonical Escape control point is invalid");
    maximum_scalar_jacobian_norm = std::max(
      maximum_scalar_jacobian_norm,
      (tangent.transpose() * context.control_point_jacobians[debug_control_point]).norm());
    last_active_control_point = debug_control_point;
    last_active_tangent = tangent;
    last_active_direction_valid = true;
    require(
      std::abs(outward_normal.norm() - 1.0) <= 5e-3,
      "reported outward normal is not unit length");
    require(
      std::abs(tangent.norm() - 1.0) <= 5e-3,
      "reported tangent is not unit length");
    require(
      std::abs(outward_normal.dot(tangent)) <= 2e-2,
      "reported tangent is not orthogonal to the obstacle normal");

    const double lambda = debug[kDebugLambda];
    const double effective_metric = debug[kDebugEffectiveMetric];
    maximum_lambda = std::max(maximum_lambda, lambda);
    maximum_effective_metric = std::max(maximum_effective_metric, effective_metric);
    maximum_contribution_norm = std::max(maximum_contribution_norm, contribution.norm);
    require(
      std::isfinite(lambda) && lambda >= -1e-9 && lambda <= 1.0 + 1e-9,
      "canonical Escape lambda is outside [0, 1]");
    require(
      std::isfinite(effective_metric) && effective_metric >= -1e-9,
      "canonical Escape effective scalar metric is invalid");

    if (lambda <= 1e-10) {
      checked_lambda_zero = true;
      require(
        contribution.norm <= 1e-7,
        "lambda=0 canonical Escape sample has a nonzero delta metric");
      require(
        delta_force.norm() <= 1e-7,
        "lambda=0 canonical Escape sample has a nonzero delta force");
    }

    if (effective_metric > 1e-8 && contribution.norm > 1e-8) {
      require(
        contribution.numerical_rank == 1,
        "active nonzero Escape contribution is not a single rank-one scalar leaf");
      require(
        dual[0] > 0.5,
        "nonzero Escape contribution is absent from the dual solve");
      double dual_delta_norm_squared = 0.0;
      for (int joint = 0; joint < 6; ++joint) {
        require(
          std::abs(
            dual[13 + joint] -
            (dual[1 + joint] - dual[7 + joint])) <= 1e-10,
          "dual-solve joint acceleration delta is inconsistent");
        dual_delta_norm_squared += dual[13 + joint] * dual[13 + joint];
      }
      require(
        dual_delta_norm_squared > 1e-16,
        "active Escape dual solve produced no joint acceleration difference");
      saw_nonzero_contribution = true;
    }
  }

  require(saw_active, "canonical Escape never became active");
  require(active_sample_count >= 2, "canonical Escape did not remain active across cycles");
  require(saw_candidate_records, "canonical Escape never reported a direction candidate");
  if (!saw_nonzero_contribution) {
    std::ostringstream message;
    message
      << "canonical Escape never produced a nonzero contribution after ramp-in"
      << " (max_lambda=" << maximum_lambda
      << ", max_effective_metric=" << maximum_effective_metric
      << ", max_delta_metric_norm=" << maximum_contribution_norm
      << ", max_scalar_jacobian_norm=" << maximum_scalar_jacobian_norm << ")";
    fail(message.str());
  }

  ObstacleSphere generic_obstacle = obstacle;
  generic_obstacle.proximity_control_point_index = -1;
  generic_obstacle.source_id = 102;
  PinocchioDirectRmpSolver generic_obstacle_solver(enabled_config, urdf_path);
  bool generic_obstacle_became_active = false;
  for (int cycle = 0; cycle < kCycles; ++cycle) {
    const auto generic_result =
      generic_obstacle_solver.solve(q, qd, targets, {generic_obstacle});
    require(
      finite_result(generic_result),
      "generic-obstacle canonical Escape solve is not finite");
    const auto & debug = generic_result.tangent_escape_rmp_data;
    if (
      debug.size() >= kRequiredActiveDebugSize &&
      debug[kDebugActive] > 0.5 &&
      debug[kDebugEffectiveMetric] > 1e-8)
    {
      generic_obstacle_became_active = true;
      require(
        static_cast<std::int64_t>(std::llround(debug[kDebugObstacleEpisode])) ==
        generic_obstacle.source_id,
        "generic obstacle did not preserve its episode/source id");
      break;
    }
  }
  require(
    generic_obstacle_became_active,
    "canonical Escape ignored generic geometry without a proximity CP index");

  require(last_active_direction_valid, "canonical Escape did not retain an active tangent");
  const auto release_scalar_jacobian =
    last_active_tangent.transpose() *
    context.control_point_jacobians[last_active_control_point];
  require(
    release_scalar_jacobian.squaredNorm() > 1e-10,
    "active tangent cannot create a residual-speed release test");
  const JointVector residual_qd =
    release_scalar_jacobian.transpose() *
    (0.05 / release_scalar_jacobian.squaredNorm());
  bool retained_braking_metric = false;
  for (int cycle = 0; cycle < 12; ++cycle) {
    const RmpSolveResult release_result =
      enabled_solver.solve(q, residual_qd, targets, {});
    require(finite_result(release_result), "residual-speed release solve is not finite");
    const auto & debug = release_result.tangent_escape_rmp_data;
    require(
      debug.size() >= kRequiredActiveDebugSize && debug[kDebugActive] > 0.5,
      "canonical Escape removed the branch while residual tangent speed remained");
    retained_braking_metric =
      retained_braking_metric ||
      (
        approximately_integer(debug[kDebugHighLevelState], 3) &&
        debug[kDebugLambda] > 0.99 &&
        debug[kDebugEffectiveMetric] > 1e-8);
  }
  require(
    retained_braking_metric,
    "canonical Escape did not retain RELEASE braking authority");

  const RmpSolveResult disabled_clear_result =
    disabled_solver.solve(q, qd, targets, {});
  bool saw_release = false;
  bool returned_off = false;
  for (int cycle = 0; cycle < 40; ++cycle) {
    const RmpSolveResult release_result =
      enabled_solver.solve(q, qd, targets, {});
    require(finite_result(release_result), "canonical Escape release solve is not finite");
    const auto & debug = release_result.tangent_escape_rmp_data;
    const bool active =
      debug.size() >= kRequiredActiveDebugSize &&
      std::isfinite(debug[kDebugActive]) &&
      debug[kDebugActive] > 0.5;
    if (active) {
      saw_release =
        saw_release ||
        approximately_integer(debug[kDebugHighLevelState], 3);
      continue;
    }
    returned_off = true;
    require(
      (release_result.metric - disabled_clear_result.metric).norm() <= 1e-7,
      "released canonical Escape left a nonzero metric contribution");
    require(
      (release_result.force - disabled_clear_result.force).norm() <= 1e-7,
      "released canonical Escape left a nonzero force contribution");
    break;
  }
  require(saw_release, "canonical Escape did not enter RELEASE");
  require(returned_off, "canonical Escape did not return to OFF after braking");

  EigenRmpConfig handoff_config = enabled_config;
  handoff_config.tangent_escape.pair_switch_margin = 0.05;
  PinocchioDirectRmpSolver handoff_solver(handoff_config, urdf_path);
  const double first_clearance = 0.25;
  ObstacleSphere first_episode{
    selected_cp.position +
    (selected_cp.radius + obstacle_radius + first_clearance) * approach_direction,
    obstacle_radius,
    static_cast<int>(nominal.control_point_index)};
  first_episode.source_id = 201;
  ObstacleSphere second_episode = obstacle;
  second_episode.source_id = 202;

  Eigen::Vector3d handoff_tangent = Eigen::Vector3d::Zero();
  std::size_t handoff_control_point = 0;
  bool handoff_direction_valid = false;
  for (int cycle = 0; cycle < 12; ++cycle) {
    const auto result = handoff_solver.solve(q, qd, targets, {first_episode});
    require(finite_result(result), "canonical Escape initial episode solve is not finite");
    const auto & debug = result.tangent_escape_rmp_data;
    if (
      debug.size() >= kRequiredActiveDebugSize &&
      debug[kDebugActive] > 0.5)
    {
      handoff_tangent =
        debug_vector3(debug, kDebugTangent, "handoff tangent");
      handoff_control_point =
        static_cast<std::size_t>(std::llround(debug[kDebugControlPoint]));
      handoff_direction_valid = true;
    }
  }
  require(
    handoff_direction_valid &&
    handoff_control_point < context.control_point_jacobians.size(),
    "canonical Escape did not expose an active direction for handoff testing");
  const auto handoff_scalar_jacobian =
    handoff_tangent.transpose() *
    context.control_point_jacobians[handoff_control_point];
  require(
    handoff_scalar_jacobian.squaredNorm() > 1e-10,
    "active handoff tangent cannot create persistent scalar velocity");
  // Keep the old tangent velocity above release_stop_speed.  Without a
  // bounded RESELECT brake phase, this makes the handoff wait forever.
  const JointVector persistent_handoff_qd =
    handoff_scalar_jacobian.transpose() *
    (0.05 / handoff_scalar_jacobian.squaredNorm());

  bool saw_reselect = false;
  bool saw_reselect_brake = false;
  bool saw_reselect_lambda_down = false;
  bool switched_episode = false;
  bool switched_at_zero_effect = false;
  for (int cycle = 0; cycle < 80; ++cycle) {
    const auto result =
      handoff_solver.solve(
      q, persistent_handoff_qd, targets, {first_episode, second_episode});
    require(finite_result(result), "canonical Escape handoff solve is not finite");
    const auto & debug = result.tangent_escape_rmp_data;
    if (
      debug.size() < kRequiredActiveDebugSize ||
      debug[kDebugActive] <= 0.5)
    {
      continue;
    }
    saw_reselect =
      saw_reselect ||
      approximately_integer(debug[kDebugHighLevelState], 4);
    saw_reselect_brake =
      saw_reselect_brake ||
      approximately_integer(debug[kDebugHandoffPhase], 7);
    saw_reselect_lambda_down =
      saw_reselect_lambda_down ||
      approximately_integer(debug[kDebugHandoffPhase], 8);
    const auto episode_id =
      static_cast<std::int64_t>(std::llround(debug[kDebugObstacleEpisode]));
    if (episode_id == second_episode.source_id) {
      switched_episode = true;
      switched_at_zero_effect =
        debug[kDebugLambda] <= 1e-10 &&
        approximately_integer(debug[kDebugHandoffPhase], 1);
      const auto no_escape_result =
        disabled_solver.solve(
        q, persistent_handoff_qd, targets, {first_episode, second_episode});
      require(
        finite_result(no_escape_result),
        "Escape-disabled handoff reference solve is not finite");
      require(
        (result.metric - no_escape_result.metric).norm() <= 1e-7,
        "canonical Escape changed obstacle episode with a nonzero metric contribution");
      require(
        (result.force - no_escape_result.force).norm() <= 1e-7,
        "canonical Escape changed obstacle episode with a nonzero force contribution");
      break;
    }
  }
  require(saw_reselect, "canonical Escape did not enter RESELECT for a stronger pair");
  require(
    saw_reselect_brake,
    "canonical Escape did not enter RESELECT braking with persistent tangent speed");
  require(
    saw_reselect_lambda_down,
    "canonical Escape did not bound RESELECT braking and ramp lambda down");
  require(switched_episode, "canonical Escape did not switch obstacle episode");
  require(
    switched_at_zero_effect,
    "canonical Escape changed obstacle episode before lambda reached zero");

  EigenRmpConfig proof_config = make_test_config(false);
  proof_config.solve_offset = 0.0;
  proof_config.collision.policy = "paper_gds";
  proof_config.joint_limit.policy = "paper_gds";
  proof_config.target.min_metric_alpha = 1.0;
  proof_config.target.proximity_metric_boost_scalar = 1.0;
  proof_config.axis_target.proximity_metric_boost_scalar = 1.0;
  proof_config.damping.metric_scalar = 0.0;
  for (auto & node : proof_config.graph_nodes) {
    if (node.name == "joint_velocity_cap") {
      node.enabled = false;
    }
  }
  PinocchioDirectRmpSolver proof_solver(proof_config, urdf_path);
  (void)proof_solver.solve(q, qd, targets, obstacles);
  const auto proof_result = proof_solver.solve(q, qd, targets, obstacles);
  const auto & certificate = proof_result.stability_certificate_data;
  require(
    certificate.size() == 35 && finite_vector(certificate),
    "stability certificate schema is missing detailed structural flags");
  require(
    certificate[1] > 0.5 &&
    certificate[2] > 0.5 &&
    certificate[3] > 0.5 &&
    certificate[5] > 0.5,
    "structured-GDS proof profile did not produce a conditional certificate");
  require(
    certificate[31] > 0.5 &&
    certificate[32] > 0.5 &&
    certificate[33] > 0.5 &&
    certificate[34] > 0.5,
    "stability certificate structural sub-checks did not all pass");

  EigenRmpConfig regularized_config = proof_config;
  regularized_config.solve_offset = 1e-13;
  PinocchioDirectRmpSolver regularized_solver(regularized_config, urdf_path);
  const auto regularized_result =
    regularized_solver.solve(q, qd, targets, obstacles);
  require(
    regularized_result.stability_certificate_data.size() == 35 &&
    regularized_result.stability_certificate_data[1] < 0.5 &&
    regularized_result.stability_certificate_data[31] < 0.5,
    "nonzero root regularization was incorrectly accepted as exact GDS");

  EigenRmpConfig diagnostic_config = make_test_config(false);
  diagnostic_config.enable_leaf_ablation_diagnostics = true;
  diagnostic_config.max_joint_accel = 1e-3;
  PinocchioDirectRmpSolver diagnostic_solver(diagnostic_config, urdf_path);
  const auto diagnostic_result =
    diagnostic_solver.solve(q, qd, targets, obstacles);
  const auto & ablation = diagnostic_result.leaf_ablation_data;
  require(
    ablation.size() >= 25 && approximately_integer(ablation[0], 1),
    "leaf-ablation schema header is missing");
  const auto ablation_record_count =
    static_cast<std::size_t>(std::llround(ablation[1]));
  require(
    ablation.size() == 25 + 17 * ablation_record_count &&
    finite_vector(ablation),
    "leaf-ablation schema length or values are invalid");
  for (int joint = 0; joint < 6; ++joint) {
    require(
      std::abs(ablation[7 + joint] - diagnostic_result.qdd[joint]) <= 1e-10,
      "leaf-ablation raw qdd does not match the commanded solve input");
    require(
      std::abs(ablation[13 + joint]) <= diagnostic_config.max_joint_accel + 1e-12,
      "leaf-ablation command qdd exceeds the configured clamp");
  }
  EigenRmpConfig reference_config = diagnostic_config;
  reference_config.enable_leaf_ablation_diagnostics = false;
  PinocchioDirectRmpSolver reference_solver(reference_config, urdf_path);
  const auto reference_result = reference_solver.solve(q, qd, targets, obstacles);
  require(
    (diagnostic_result.qdd - reference_result.qdd).norm() <= 1e-10 &&
    (diagnostic_result.metric - reference_result.metric).norm() <= 1e-10 &&
    (diagnostic_result.force - reference_result.force).norm() <= 1e-10,
    "leaf-ablation diagnostics changed the primary RMP solve");

  std::cout
    << "canonical tangent Escape integration test passed"
    << " (cp=" << nominal.control_point_index
    << ", nominal_speed=" << nominal.speed
    << ", active_samples=" << active_sample_count
    << ", lambda_zero_checked=" << (checked_lambda_zero ? "yes" : "not observed")
    << ")\n";
}

void run_risk_damped_test(const std::string & urdf_path)
{
  const JointVector q = JointVector::Zero();
  const JointVector qd = JointVector::Zero();

  PinocchioModel model(urdf_path);
  const KinematicsContext context = model.forward_context(q, qd);

  EigenRmpConfig disabled_config = make_risk_damped_test_config(false);
  PinocchioDirectRmpSolver disabled_solver(disabled_config, urdf_path);
  const NominalScenario nominal = find_strongest_nominal_scenario(
    disabled_solver,
    context,
    q,
    qd,
    disabled_config.tangent_escape.nominal_prediction_dt);

  const Eigen::Vector3d approach_direction = nominal.control_point_motion.normalized();
  const auto & selected_cp = context.control_points[nominal.control_point_index];
  constexpr double obstacle_radius = 0.04;
  constexpr double initial_clearance = 0.04;
  const auto obstacle_at_clearance = [&](double clearance) {
      ObstacleSphere obstacle{
        selected_cp.position +
        (selected_cp.radius + obstacle_radius + clearance) * approach_direction,
        obstacle_radius,
        static_cast<int>(nominal.control_point_index)};
      obstacle.source_id = 701;
      return obstacle;
    };

  const ObstacleSphere initial_obstacle = obstacle_at_clearance(initial_clearance);
  const std::unordered_map<std::string, Eigen::Vector3d> targets{{"goal", nominal.goal}};
  const RmpSolveResult disabled_initial =
    disabled_solver.solve(q, qd, targets, {initial_obstacle});
  require(finite_result(disabled_initial), "risk-damped reference solve is not finite");

  EigenRmpConfig risk_config = disabled_config;
  risk_config.tangent_escape.enabled = true;
  PinocchioDirectRmpSolver risk_solver(risk_config, urdf_path);
  const RmpSolveResult initial_result =
    risk_solver.solve(q, qd, targets, {initial_obstacle});
  require(finite_result(initial_result), "risk-damped first solve is not finite");

  const auto & initial_debug = initial_result.tangent_escape_rmp_data;
  const bool finite_risk_prefix =
    initial_debug.size() >= kRequiredRiskDebugSize &&
    std::all_of(
    initial_debug.begin(),
    initial_debug.begin() + static_cast<std::ptrdiff_t>(kRequiredRiskDebugSize),
    [](double value) {return std::isfinite(value);});
  if (!finite_risk_prefix) {
    std::ostringstream message;
    message
      << "risk-damped schema-7 debug record is missing or non-finite"
      << " (size=" << initial_debug.size();
    if (!initial_debug.empty()) {
      message << ", active=" << initial_debug.front();
    }
    message << ")";
    fail(message.str());
  }
  require(
    initial_debug[kDebugActive] > 0.5 &&
    approximately_integer(initial_debug[kDebugLeafMode], 7),
    "risk-damped first solve did not report active schema id 7");
  require(
    approximately_integer(initial_debug[kDebugHighLevelState], 1),
    "risk-damped first solve did not enter direct DRIVE/PREVENT state");

  const double alpha_distance = initial_debug[kDebugAlphaDistance];
  const double alpha_blocking = initial_debug[kDebugAlphaBlocking];
  const double blockage = initial_debug[kDebugActivation];
  const double expected_blockage = alpha_distance * alpha_blocking;
  require(alpha_distance > 0.0, "risk-damped distance activation is zero");
  require(alpha_blocking > 0.0, "risk-damped blocking activation is zero");
  require(
    std::abs(blockage - expected_blockage) <= 1e-12,
    "risk-damped activation is not the direct B=alpha_d*alpha_b product");
  require(
    std::abs(initial_debug[kDebugRawActivation] - expected_blockage) <= 1e-12,
    "risk-damped raw activation is not the direct blockage product");

  const double expected_metric =
    risk_config.tangent_escape.metric_scalar *
    initial_debug[kDebugLambda] * expected_blockage;
  require(
    std::abs(initial_debug[kDebugEffectiveMetric] - expected_metric) <=
    1e-10 * std::max(1.0, expected_metric),
    "risk-damped metric is not m_t*lambda*B");
  require(
    initial_debug[kDebugEffectiveMetric] > 0.0,
    "risk-damped first solve has no scalar metric");
  require(
    initial_debug[kDebugScalarAcceleration] > 0.0,
    "risk-damped distance drive did not accelerate from zero tangent speed");
  require(
    initial_debug[kDebugRiskDistanceAcceleration] > 0.0 &&
    std::abs(initial_debug[kDebugFilteredClearanceRate]) <= 1e-12 &&
    std::abs(initial_debug[kDebugClosingSpeed]) <= 1e-12 &&
    std::abs(initial_debug[kDebugRiskApproachAcceleration]) <= 1e-12,
    "risk-damped first clearance sample was not initialized with zero rate");
  require(
    std::abs(initial_debug[kDebugRiskDampingAcceleration]) <= 1e-12 &&
    initial_debug[kDebugRiskAcceleration] > 0.0 &&
    initial_debug[kDebugRiskAccelerationInput] > 0.0,
    "risk-damped zero-speed acceleration components are inconsistent");

  // Escape clearance must use its own numeric margin, not collision.margin.
  EigenRmpConfig independent_margin_config = risk_config;
  independent_margin_config.collision.margin = 0.02;
  independent_margin_config.tangent_escape.clearance_margin = 0.01;
  PinocchioDirectRmpSolver independent_margin_solver(
    independent_margin_config,
    urdf_path);
  const RmpSolveResult independent_margin_result =
    independent_margin_solver.solve(q, qd, targets, {initial_obstacle});
  require(
    independent_margin_result.tangent_escape_rmp_data.size() >=
    kRequiredRiskDebugSize,
    "independent-margin risk-damped solve did not activate");
  require(
    std::abs(
      independent_margin_result.tangent_escape_rmp_data[kDebugClearance] -
      (initial_clearance - 0.01)) <= 1e-12,
    "Escape clearance still depends on collision.margin");

  const Matrix6 initial_delta_metric =
    initial_result.metric - disabled_initial.metric;
  const JointVector initial_delta_force =
    initial_result.force - disabled_initial.force;
  const ScalarContributionCheck initial_contribution =
    check_single_scalar_contribution(initial_delta_metric, initial_delta_force);
  require(
    initial_contribution.numerical_rank == 1 && initial_contribution.norm > 1e-8,
    "risk-damped active contribution is not a nonzero PSD rank-one leaf");

  const ObstacleSphere closer_obstacle = obstacle_at_clearance(0.01);
  const RmpSolveResult approach_result =
    risk_solver.solve(q, qd, targets, {closer_obstacle});
  require(finite_result(approach_result), "risk-damped approach solve is not finite");
  const auto & approach_debug = approach_result.tangent_escape_rmp_data;
  require(
    approach_debug.size() >= kRequiredRiskDebugSize &&
    approach_debug[kDebugActive] > 0.5 &&
    approximately_integer(approach_debug[kDebugLeafMode], 7),
    "risk-damped approach sample lost schema-7 activation");
  require(
    approach_debug[kDebugFilteredClearanceRate] < -1e-6 &&
    approach_debug[kDebugClosingSpeed] > 1e-6 &&
    approach_debug[kDebugRiskApproachAcceleration] > 0.0,
    "risk-damped decreasing clearance did not create an approach-risk term");
  require(
    !approximately_integer(approach_debug[kDebugHighLevelState], 3),
    "risk-damped mode incorrectly entered legacy RELEASE");

  const Eigen::Vector3d tangent =
    debug_vector3(approach_debug, kDebugTangent, "risk-damped tangent");
  const auto active_control_point =
    static_cast<std::size_t>(std::llround(approach_debug[kDebugControlPoint]));
  require(
    active_control_point < context.control_point_jacobians.size(),
    "risk-damped active control-point index is invalid");
  const auto scalar_jacobian =
    tangent.transpose() * context.control_point_jacobians[active_control_point];
  require(
    scalar_jacobian.squaredNorm() > 1e-10,
    "risk-damped tangent cannot create the residual-speed removal test");
  const JointVector residual_qd =
    scalar_jacobian.transpose() * (0.05 / scalar_jacobian.squaredNorm());

  const RmpSolveResult disabled_clear =
    disabled_solver.solve(q, residual_qd, targets, {});
  const RmpSolveResult removed_result =
    risk_solver.solve(q, residual_qd, targets, {});
  require(
    finite_result(disabled_clear) && finite_result(removed_result),
    "risk-damped no-obstacle residual-speed solve is not finite");
  require(
    removed_result.tangent_escape_rmp_data.size() == 1 &&
    removed_result.tangent_escape_rmp_data.front() <= 0.5,
    "risk-damped branch remained active after same-cycle obstacle removal");
  require(
    (removed_result.metric - disabled_clear.metric).norm() <= 1e-7,
    "risk-damped obstacle removal left a residual metric contribution");
  require(
    (removed_result.force - disabled_clear.force).norm() <= 1e-7,
    "risk-damped obstacle removal left a residual braking force");
  require(
    removed_result.tangent_escape_dual_solve_data.size() == 43 &&
    removed_result.tangent_escape_dual_solve_data[0] <= 0.5,
    "risk-damped obstacle removal still reports an active dual solve");

  const RmpSolveResult reappearance_result =
    risk_solver.solve(q, qd, targets, {closer_obstacle});
  require(
    finite_result(reappearance_result),
    "risk-damped obstacle reappearance solve is not finite");
  const auto & reappearance_debug = reappearance_result.tangent_escape_rmp_data;
  require(
    reappearance_debug.size() >= kRequiredRiskDebugSize &&
    reappearance_debug[kDebugActive] > 0.5 &&
    approximately_integer(reappearance_debug[kDebugLeafMode], 7),
    "risk-damped obstacle did not reactivate with schema id 7");
  require(
    std::abs(reappearance_debug[kDebugFilteredClearanceRate]) <= 1e-12 &&
    std::abs(reappearance_debug[kDebugClosingSpeed]) <= 1e-12 &&
    std::abs(reappearance_debug[kDebugRiskApproachAcceleration]) <= 1e-12,
    "risk-damped clearance derivative was not reset after disappearance");
  require(
    !approximately_integer(reappearance_debug[kDebugHighLevelState], 3),
    "risk-damped reappearance incorrectly used legacy RELEASE");

  std::cout
    << "risk-damped tangent Escape integration test passed"
    << " (cp=" << nominal.control_point_index
    << ", B=" << blockage
    << ", initial_accel=" << initial_debug[kDebugScalarAcceleration]
    << ", approach_rate=" << approach_debug[kDebugFilteredClearanceRate]
    << ")\n";
}

}  // namespace

int main(int argc, char ** argv)
{
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " /path/to/robot.urdf\n";
    return EXIT_FAILURE;
  }

  try {
    JointAccelerationVector raw;
    raw << 20.0, 5.0, -2.0, 0.0, 1.0, -4.0;
    const auto scaled = rb10_rmpflow_rviz::limit_joint_acceleration(raw, 10.0, true);
    require(
      std::abs(scaled.cwiseAbs().maxCoeff() - 10.0) <= 1e-12 &&
      (scaled - 0.5 * raw).norm() <= 1e-12 &&
      std::abs(raw.normalized().dot(scaled.normalized()) - 1.0) <= 1e-12,
      "direction-preserving joint acceleration limit is incorrect");
    const auto clipped = rb10_rmpflow_rviz::limit_joint_acceleration(raw, 10.0, false);
    require(
      std::abs(clipped[0] - 10.0) <= 1e-12 &&
      std::abs(clipped[1] - 5.0) <= 1e-12,
      "legacy per-joint acceleration limit is incorrect");

    run_test(argv[1]);
    run_risk_damped_test(argv[1]);
  } catch (const std::exception & error) {
    std::cerr << "test_tangent_escape_canonical: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
