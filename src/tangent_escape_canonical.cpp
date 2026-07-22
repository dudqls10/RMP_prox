#include "rb10_rmpflow_rviz/pinocchio_direct_solver.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace rb10_rmpflow_rviz
{

namespace
{

constexpr double kPi = 3.14159265358979323846;
constexpr std::size_t kCoarseCandidateCount = 16;
constexpr std::size_t kCanonicalDebugSize = 61;
constexpr std::size_t kRiskDampedDebugSize = 68;
constexpr double kCandidateDuplicateDot = 0.985;
constexpr double kTangentOrthogonalityTolerance = 1e-5;
constexpr double kMinimumScalarJacobianNorm = 1e-8;
constexpr double kMinimumPredictedTangentDisplacement = 1e-10;

using JointVector = PinocchioDirectRmpSolver::JointVector;
using Matrix6 = PinocchioDirectRmpSolver::Matrix6;
using RowVector6 = PinocchioDirectRmpSolver::RowVector6;

double clamp01(double value)
{
  return std::clamp(value, 0.0, 1.0);
}

double quintic01(double value)
{
  const double t = clamp01(value);
  return t * t * t * (10.0 + t * (-15.0 + 6.0 * t));
}

double smooth_gate_up(double value, double lower, double upper)
{
  if (upper <= lower + 1e-12) {
    return value >= upper ? 1.0 : 0.0;
  }
  return quintic01((value - lower) / (upper - lower));
}

double smooth_gate_down(double value, double lower, double upper)
{
  return 1.0 - smooth_gate_up(value, lower, upper);
}

double ramp_between(double start, double target, double elapsed, double duration)
{
  if (duration <= 1e-9) {
    return target;
  }
  return start + (target - start) * quintic01(elapsed / duration);
}

double bounded_velocity_acceleration(double input, double maximum_acceleration)
{
  if (maximum_acceleration <= 0.0) {
    return input;
  }
  return maximum_acceleration * std::tanh(input / maximum_acceleration);
}

std::optional<Eigen::Vector3d> tangent_projection(
  const Eigen::Vector3d & direction,
  const Eigen::Vector3d & normal,
  double minimum_norm)
{
  if (!direction.allFinite() || !normal.allFinite()) {
    return std::nullopt;
  }
  const Eigen::Vector3d projected = direction - direction.dot(normal) * normal;
  const double norm = projected.norm();
  if (norm <= minimum_norm) {
    return std::nullopt;
  }
  return projected / norm;
}

JointVector resolve_trial_root(
  const Matrix6 & metric,
  const JointVector & force,
  double solve_offset)
{
  const double max_abs = std::max(metric.cwiseAbs().maxCoeff() * 0.01, 1.0);
  Matrix6 scaled_metric = metric / max_abs;
  const JointVector scaled_force = force / max_abs;
  scaled_metric += std::max(solve_offset, 0.0) * Matrix6::Identity();

  JointVector qdd = scaled_metric.ldlt().solve(scaled_force).eval();
  if (!qdd.allFinite()) {
    qdd = scaled_metric.completeOrthogonalDecomposition().solve(scaled_force).eval();
  }
  if (!qdd.allFinite()) {
    qdd.setZero();
  }
  return qdd;
}

struct SensorPairEvaluation
{
  std::size_t control_point_index{0};
  std::int64_t obstacle_key{-1};
  std::size_t obstacle_index{0};
  Eigen::Vector3d control_point{Eigen::Vector3d::Zero()};
  Eigen::Vector3d obstacle_center{Eigen::Vector3d::Zero()};
  Eigen::Vector3d outward_normal{Eigen::Vector3d::UnitX()};
  Eigen::Vector3d obstacle_direction{Eigen::Vector3d::UnitX()};
  Eigen::Vector3d point_velocity{Eigen::Vector3d::Zero()};
  Eigen::Vector3d nominal_velocity{Eigen::Vector3d::Zero()};
  double clearance{0.0};
  double beta{-1.0};
  double alpha_distance{0.0};
  double alpha_blocking{0.0};
  double alpha_stuck{0.0};
  double blockage{0.0};
  double raw_activation{0.0};
  double clearance_rate{0.0};
  double closing_speed{0.0};
  double risk_distance_acceleration{0.0};
  double risk_approach_acceleration{0.0};
  double risk_acceleration{0.0};
};

struct CanonicalCandidateEvaluation
{
  bool feasible{false};
  bool saturation_risk{false};
  int seed_index{-1};
  double theta{0.0};
  Eigen::Vector3d direction{Eigen::Vector3d::UnitX()};
  JointVector qdd{JointVector::Zero()};
  double score{-std::numeric_limits<double>::infinity()};
  double goal_score{0.0};
  double sector_risk{0.0};
  double continuity_score{0.5};
  double blocked_penalty{0.0};
  double acceleration_jump_penalty{0.0};
  double metric_scalar{0.0};
  double scalar_velocity{0.0};
  double scalar_acceleration{0.0};
  double tangent_displacement{0.0};
};

bool same_pair(
  const SensorPairEvaluation & pair,
  std::size_t control_point_index,
  std::int64_t obstacle_key)
{
  return
    pair.control_point_index == control_point_index &&
    pair.obstacle_key == obstacle_key;
}

double candidate_score_or_negative_infinity(const CanonicalCandidateEvaluation & candidate)
{
  return candidate.feasible ?
         candidate.score :
         -std::numeric_limits<double>::infinity();
}

void append_candidate_record(
  std::vector<double> & debug_values,
  const CanonicalCandidateEvaluation & candidate,
  double alpha_stuck)
{
  debug_values.push_back(static_cast<double>(candidate.seed_index));
  debug_values.push_back(candidate.feasible ? 1.0 : 0.0);
  debug_values.push_back(
    candidate.feasible ? candidate.score : -std::numeric_limits<double>::infinity());
  debug_values.push_back(candidate.goal_score);
  debug_values.push_back(candidate.continuity_score);
  debug_values.push_back(candidate.sector_risk);
  debug_values.push_back(candidate.acceleration_jump_penalty);
  debug_values.push_back(candidate.blocked_penalty);
  debug_values.push_back(alpha_stuck);
  debug_values.push_back(candidate.tangent_displacement);
  debug_values.push_back(candidate.direction.x());
  debug_values.push_back(candidate.direction.y());
  debug_values.push_back(candidate.direction.z());
  debug_values.push_back(candidate.metric_scalar);
  debug_values.push_back(candidate.scalar_velocity);
  debug_values.push_back(candidate.scalar_acceleration);
  debug_values.push_back(candidate.feasible ? 1.0 : 0.0);
}

}  // namespace

void PinocchioDirectRmpSolver::accumulate_tangent_escape_canonical(
  const JointVector & q,
  const KinematicsContext & context,
  const NodeGeometry & geometry,
  const JointVector & qd,
  const Eigen::Vector3d & goal,
  const std::vector<ObstacleSphere> & obstacles,
  const JointVector & nominal_qdd,
  const Matrix6 & base_metric,
  const JointVector & base_force,
  Matrix6 & metric,
  JointVector & force,
  std::vector<double> * debug_data) const
{
  accumulate_tangent_escape_impl(
    tangent_escape_canonical_state_,
    false,
    q,
    context,
    geometry,
    qd,
    goal,
    obstacles,
    nominal_qdd,
    base_metric,
    base_force,
    metric,
    force,
    debug_data);
}

void PinocchioDirectRmpSolver::accumulate_tangent_escape_risk_damped(
  const JointVector & q,
  const KinematicsContext & context,
  const NodeGeometry & geometry,
  const JointVector & qd,
  const Eigen::Vector3d & goal,
  const std::vector<ObstacleSphere> & obstacles,
  const JointVector & nominal_qdd,
  const Matrix6 & base_metric,
  const JointVector & base_force,
  Matrix6 & metric,
  JointVector & force,
  std::vector<double> * debug_data) const
{
  accumulate_tangent_escape_impl(
    tangent_escape_risk_damped_state_,
    true,
    q,
    context,
    geometry,
    qd,
    goal,
    obstacles,
    nominal_qdd,
    base_metric,
    base_force,
    metric,
    force,
    debug_data);
}

void PinocchioDirectRmpSolver::accumulate_tangent_escape_impl(
  TangentEscapeCanonicalState & state,
  bool risk_damped_mode,
  const JointVector & q,
  const KinematicsContext & context,
  const NodeGeometry & geometry,
  const JointVector & qd,
  const Eigen::Vector3d & goal,
  const std::vector<ObstacleSphere> & obstacles,
  const JointVector & nominal_qdd,
  const Matrix6 & base_metric,
  const JointVector & base_force,
  Matrix6 & metric,
  JointVector & force,
  std::vector<double> * debug_data) const
{
  const auto & params = config_.tangent_escape;

  const auto publish_inactive = [&debug_data]() {
      if (debug_data != nullptr) {
        *debug_data = std::vector<double>{0.0};
      }
    };
  const auto reset_all_state = [&state]() {
      state = TangentEscapeCanonicalState{};
    };
  const auto clear_failure_memory = [&state]() {
      for (auto & memory : state.failure_memory) {
        memory = TangentEscapeFailureMemory{};
      }
      state.failure_memory_cursor = 0;
    };

  const std::size_t point_count = std::min(
    {
      static_cast<std::size_t>(std::max<Eigen::Index>(geometry.x.size() / 3, 0)),
      context.control_points.size(),
      context.control_point_jacobians.size(),
      context.control_point_velocities.size(),
      context.control_point_curvatures.size(),
      RB10Model::sensor_control_points.size()
    });
  if (
    !params.enabled ||
    params.metric_scalar <= 0.0 ||
    geometry.x.size() % 3 != 0 ||
    geometry.jacobian.cols() != 6 ||
    point_count == 0)
  {
    reset_all_state();
    publish_inactive();
    return;
  }

  const double dt = std::max(params.control_dt, 1e-4);
  const double preview_time = std::max(params.candidate_lookahead, 1e-4);
  const double nominal_prediction_time = std::max(params.nominal_prediction_dt, 0.0);
  const double minimum_tangent_norm = std::max(params.min_tangent_norm, 1e-9);
  const double maximum_leaf_acceleration = std::max(params.max_accel, 0.0);
  const double velocity_gain = std::max(params.velocity_gain, 0.0);
  const double maximum_tangent_speed = std::max(params.max_speed, 0.0);
  const double prevent_speed = std::min(
    std::max(params.prevent_speed, 0.0),
    maximum_tangent_speed);
  const double recovery_speed = std::min(
    std::max(params.recovery_speed, prevent_speed),
    maximum_tangent_speed);
  const double velocity_time_constant =
    std::max(params.desired_velocity_time_constant, 1e-6);
  // risk_damped follows the geometric definition B>0 directly.  The legacy
  // velocity mode retains its historical entry threshold.
  const double minimum_pair_activation = risk_damped_mode ?
    1e-12 :
    std::max(params.min_activation, 0.0);
  const double risk_distance_gain = std::max(params.risk_distance_gain, 0.0);
  const double risk_distance_scale = std::max(params.risk_distance_scale, 1e-9);
  const double risk_approach_gain = std::max(params.risk_approach_gain, 0.0);
  const double risk_approach_distance_scale =
    std::max(params.risk_approach_distance_scale, 1e-9);
  const double risk_approach_epsilon =
    std::max(params.risk_approach_epsilon, 1e-9);
  const double risk_velocity_gate_scale =
    std::max(params.risk_velocity_gate_scale, 1e-9);
  const double risk_clearance_rate_filter_time_constant =
    std::max(params.risk_clearance_rate_filter_time_constant, 0.0);
  const double risk_tangent_damping_gain =
    std::max(params.risk_tangent_damping_gain, 0.0);

  const double goal_error = (context.tcp_position - goal).norm();
  const bool goal_changed =
    state.previous_goal_valid &&
    (goal - state.previous_goal).norm() >
    std::max(params.goal_change_reset_distance, 0.0);
  if (goal_changed) {
    state.previous_goal_error_valid = false;
    state.filtered_goal_progress = 0.0;
  }
  state.previous_goal = goal;
  state.previous_goal_valid = true;

  if (state.previous_goal_error_valid) {
    const double raw_progress =
      -(goal_error - state.previous_goal_error) / dt;
    const double progress_filter_alpha =
      dt / (std::max(params.progress_filter_time_constant, 0.0) + dt);
    state.filtered_goal_progress +=
      clamp01(progress_filter_alpha) *
      (raw_progress - state.filtered_goal_progress);
  } else {
    state.filtered_goal_progress = 0.0;
    state.previous_goal_error_valid = true;
  }
  state.previous_goal_error = goal_error;

  const double low_progress_gate = smooth_gate_down(
    state.filtered_goal_progress,
    std::min(params.progress_low_threshold, params.progress_ok_threshold),
    std::max(params.progress_low_threshold, params.progress_ok_threshold));

  std::vector<SensorPairEvaluation> nearest_pairs;
  nearest_pairs.reserve(obstacles.size() * point_count);
  for (std::size_t obstacle_index = 0; obstacle_index < obstacles.size(); ++obstacle_index) {
    const auto & obstacle = obstacles[obstacle_index];
    if (
      obstacle.radius <= 0.0 ||
      !obstacle.center.allFinite())
    {
      continue;
    }

    std::size_t first_control_point = 0;
    std::size_t last_control_point = point_count;
    if (obstacle.proximity_control_point_index >= 0) {
      first_control_point =
        static_cast<std::size_t>(obstacle.proximity_control_point_index);
      if (first_control_point >= point_count) {
        continue;
      }
      last_control_point = first_control_point + 1;
    } else if (obstacle.source_id < 0) {
      // The controller uses an unlabelled far-away sphere as its empty-obstacle
      // sentinel. Real generic geometry received from MarkerArray has a stable
      // source_id and is paired with every sensor control point.
      continue;
    }

    for (
      std::size_t control_point_index = first_control_point;
      control_point_index < last_control_point;
      ++control_point_index)
    {
      const Eigen::Vector3d control_point =
        geometry.x.segment<3>(static_cast<Eigen::Index>(3 * control_point_index));
      const Eigen::Vector3d delta = control_point - obstacle.center;
      const double center_distance = delta.norm();
      if (center_distance <= 1e-9) {
        continue;
      }

      const double point_radius =
        RB10Model::sensor_control_points[control_point_index].radius;
      SensorPairEvaluation pair;
      pair.control_point_index = control_point_index;
      pair.obstacle_key =
        obstacle.source_id >= 0 ?
        obstacle.source_id :
        static_cast<std::int64_t>(obstacle.proximity_control_point_index);
      pair.obstacle_index = obstacle_index;
      pair.control_point = control_point;
      pair.obstacle_center = obstacle.center;
      pair.outward_normal = delta / center_distance;
      pair.obstacle_direction = -pair.outward_normal;
      pair.clearance =
        center_distance - (point_radius + obstacle.radius) - params.clearance_margin;

      const auto nearest = std::find_if(
        nearest_pairs.begin(),
        nearest_pairs.end(),
        [&pair](const auto & candidate) {
          return
            candidate.control_point_index == pair.control_point_index &&
            candidate.obstacle_key == pair.obstacle_key;
        });
      if (nearest == nearest_pairs.end()) {
        nearest_pairs.push_back(pair);
      } else if (pair.clearance < nearest->clearance) {
        *nearest = pair;
      }
    }
  }

  std::vector<SensorPairEvaluation> pairs;
  pairs.reserve(nearest_pairs.size());
  if (risk_damped_mode) {
    for (auto & rate_state : state.clearance_rate_states) {
      rate_state.seen_this_cycle = false;
    }
  }
  for (auto pair : nearest_pairs) {
    const std::size_t index = pair.control_point_index;
    const Eigen::Matrix<double, 3, 6> & point_jacobian =
      context.control_point_jacobians[index];
    pair.point_velocity = context.control_point_velocities[index];
    pair.nominal_velocity =
      point_jacobian * (qd + nominal_prediction_time * nominal_qdd);

    const double distance_span =
      std::max(params.influence_distance - params.safe_distance, 1e-9);
    pair.alpha_distance = quintic01(
      (params.influence_distance - pair.clearance) / distance_span);

    const double nominal_speed = pair.nominal_velocity.norm();
    const double minimum_valid_intent_speed =
      std::max(params.intent_on_speed, 1e-9);
    if (
      nominal_speed >= minimum_valid_intent_speed &&
      pair.nominal_velocity.allFinite())
    {
      pair.beta =
        (pair.nominal_velocity / nominal_speed).dot(pair.obstacle_direction);
      pair.alpha_blocking = smooth_gate_up(
        pair.beta,
        std::min(params.goal_block_beta_on, params.goal_block_beta_full),
        std::max(params.goal_block_beta_on, params.goal_block_beta_full));
    } else {
      pair.beta = -1.0;
      pair.alpha_blocking = 0.0;
    }

    const double low_motion_gate = smooth_gate_down(
      pair.point_velocity.norm(),
      std::min(params.still_speed_threshold, params.moving_speed_threshold),
      std::max(params.still_speed_threshold, params.moving_speed_threshold));
    const double intent_gate = smooth_gate_up(
      nominal_speed,
      std::min(params.intent_on_speed, params.intent_full_speed),
      std::max(params.intent_on_speed, params.intent_full_speed));
    pair.alpha_stuck = clamp01(low_progress_gate * low_motion_gate * intent_gate);
    pair.blockage = clamp01(pair.alpha_distance * pair.alpha_blocking);
    if (risk_damped_mode) {
      auto rate_state = std::find_if(
        state.clearance_rate_states.begin(),
        state.clearance_rate_states.end(),
        [&pair](const auto & candidate) {
          return
            candidate.control_point_index == pair.control_point_index &&
            candidate.obstacle_key == pair.obstacle_key;
        });
      if (rate_state == state.clearance_rate_states.end()) {
        TangentEscapeClearanceRateState initial;
        initial.control_point_index = pair.control_point_index;
        initial.obstacle_key = pair.obstacle_key;
        initial.previous_clearance = pair.clearance;
        initial.filtered_rate = 0.0;
        initial.previous_clearance_valid = true;
        initial.seen_this_cycle = true;
        state.clearance_rate_states.push_back(initial);
        pair.clearance_rate = 0.0;
      } else {
        double raw_clearance_rate = 0.0;
        if (rate_state->previous_clearance_valid) {
          raw_clearance_rate =
            (pair.clearance - rate_state->previous_clearance) / dt;
        }
        const double rate_filter_alpha =
          dt /
          (risk_clearance_rate_filter_time_constant + dt);
        rate_state->filtered_rate +=
          clamp01(rate_filter_alpha) *
          (raw_clearance_rate - rate_state->filtered_rate);
        rate_state->previous_clearance = pair.clearance;
        rate_state->previous_clearance_valid = true;
        rate_state->seen_this_cycle = true;
        pair.clearance_rate = rate_state->filtered_rate;
      }

      const double positive_clearance = std::max(pair.clearance, 0.0);
      pair.closing_speed = std::max(0.0, -pair.clearance_rate);
      const double velocity_gate =
        1.0 /
        (
          1.0 +
          std::exp(std::clamp(
            pair.clearance_rate / risk_velocity_gate_scale,
            -60.0,
            60.0)));
      pair.risk_distance_acceleration =
        risk_distance_gain *
        std::exp(-positive_clearance / risk_distance_scale);
      pair.risk_approach_acceleration =
        velocity_gate * risk_approach_gain * pair.closing_speed /
        (
          positive_clearance / risk_approach_distance_scale +
          risk_approach_epsilon);
      pair.risk_acceleration = std::max(
        pair.risk_distance_acceleration + pair.risk_approach_acceleration,
        0.0);
      if (maximum_tangent_speed > 0.0 && risk_tangent_damping_gain > 0.0) {
        pair.risk_acceleration = std::min(
          pair.risk_acceleration,
          risk_tangent_damping_gain * maximum_tangent_speed);
      }
      pair.raw_activation = pair.blockage;
    } else {
      pair.raw_activation = clamp01(
        pair.blockage *
        (
          clamp01(params.prevent_weight) +
          (1.0 - clamp01(params.prevent_weight)) * pair.alpha_stuck));
    }
    pairs.push_back(pair);
  }
  if (risk_damped_mode) {
    state.clearance_rate_states.erase(
      std::remove_if(
        state.clearance_rate_states.begin(),
        state.clearance_rate_states.end(),
        [](const auto & rate_state) {return !rate_state.seen_this_cycle;}),
      state.clearance_rate_states.end());
  }

  const auto find_pair = [&pairs](
      std::size_t control_point_index,
      std::int64_t obstacle_key) -> const SensorPairEvaluation *
    {
      const auto found = std::find_if(
        pairs.begin(),
        pairs.end(),
        [control_point_index, obstacle_key](const auto & pair) {
          return same_pair(pair, control_point_index, obstacle_key);
        });
      return found == pairs.end() ? nullptr : &(*found);
    };

  const SensorPairEvaluation * best_pair = nullptr;
  for (const auto & pair : pairs) {
    if (best_pair == nullptr || pair.blockage > best_pair->blockage) {
      best_pair = &pair;
    }
  }
  const SensorPairEvaluation * current_pair = state.active_pair_valid ?
    find_pair(state.control_point_index, state.obstacle_key) :
    nullptr;

  const bool release_phase =
    state.phase == TangentEscapeCanonicalPhase::ReleaseDriveDown ||
    state.phase == TangentEscapeCanonicalPhase::ReleaseBrake ||
    state.phase == TangentEscapeCanonicalPhase::ReleaseLambdaDown;
  const double activation_target = release_phase ?
    0.0 :
    (
      current_pair != nullptr ?
      current_pair->raw_activation :
      ((!state.active_pair_valid && best_pair != nullptr) ? best_pair->raw_activation : 0.0));
  if (risk_damped_mode) {
    state.z = clamp01(activation_target);
  } else {
    const double activation_time_constant =
      std::max(params.activation_time_constant, 1e-6);
    const double unrestricted_z_rate =
      (activation_target - state.z) / activation_time_constant;
    const double limited_z_rate = std::clamp(
      unrestricted_z_rate,
      -std::max(params.activation_fall_rate, 0.0),
      std::max(params.activation_rise_rate, 0.0));
    state.z = clamp01(state.z + dt * limited_z_rate);
  }

  const JointVector base_qdd =
    resolve_trial_root(base_metric, base_force, config_.solve_offset);
  const Eigen::Vector3d baseline_tcp_position =
    context.tcp_position +
    preview_time * context.tcp_velocity +
    0.5 * preview_time * preview_time *
    (context.tcp_jacobian * base_qdd + context.tcp_curvature);
  const double baseline_goal_error = (baseline_tcp_position - goal).norm();

  const auto sector_risk = [
      &pairs,
      &context,
      preview_time,
      &params](const JointVector & trial_qdd)
    {
      double raw_risk = 0.0;
      for (const auto & sector : pairs) {
        const std::size_t index = sector.control_point_index;
        const Eigen::Vector3d displacement =
          preview_time * context.control_point_velocities[index] +
          0.5 * preview_time * preview_time *
          (
            context.control_point_jacobians[index] * trial_qdd +
            context.control_point_curvatures[index]);
        const double displacement_norm = displacement.norm();
        if (displacement_norm <= 1e-12) {
          continue;
        }
        const double approach =
          std::max(0.0, sector.obstacle_direction.dot(displacement / displacement_norm));
        raw_risk += sector.alpha_distance * approach;
      }
      const double scale = std::max(params.sector_risk_scale, 1e-6);
      return clamp01(1.0 - std::exp(-raw_risk / scale));
    };

  const auto failure_memory_penalty = [
      &state,
      &params](const Eigen::Vector3d & direction)
    {
      const double sigma = std::max(params.blocked_memory_sigma, 1e-3);
      double penalty = 0.0;
      for (const auto & memory : state.failure_memory) {
        if (!memory.valid || memory.strength <= 0.0) {
          continue;
        }
        const double alignment =
          std::clamp(direction.dot(memory.tangent), -1.0, 1.0);
        penalty = std::max(
          penalty,
          memory.strength *
          std::exp(-(1.0 - alignment) / (2.0 * sigma * sigma)));
      }
      return clamp01(penalty);
    };

  const auto evaluate_candidate = [
      this,
      &q,
      &qd,
      &context,
      &goal,
      &state,
      &params,
      &base_metric,
      &base_force,
      &base_qdd,
      &failure_memory_penalty,
      &sector_risk,
      preview_time,
      baseline_goal_error,
      maximum_leaf_acceleration,
      velocity_gain,
      prevent_speed,
      recovery_speed,
      risk_damped_mode,
      risk_tangent_damping_gain](
      const SensorPairEvaluation & pair,
      const Eigen::Vector3d & requested_direction,
      int seed_index,
      double theta,
      bool apply_hard_gates)
    {
      CanonicalCandidateEvaluation candidate;
      candidate.seed_index = seed_index;
      candidate.theta = theta;
      candidate.direction = requested_direction;
      if (!requested_direction.allFinite() || requested_direction.norm() <= 1e-12) {
        return candidate;
      }
      candidate.direction.normalize();

      if (
        apply_hard_gates &&
        std::abs(candidate.direction.dot(pair.obstacle_direction)) >
        kTangentOrthogonalityTolerance)
      {
        return candidate;
      }

      const std::size_t index = pair.control_point_index;
      const Eigen::Matrix<double, 3, 6> & point_jacobian =
        context.control_point_jacobians[index];
      const RowVector6 scalar_jacobian =
        candidate.direction.transpose() * point_jacobian;
      if (
        apply_hard_gates &&
        scalar_jacobian.norm() <= kMinimumScalarJacobianNorm)
      {
        return candidate;
      }
      const double scalar_curvature =
        candidate.direction.dot(context.control_point_curvatures[index]);
      candidate.scalar_velocity =
        candidate.direction.dot(context.control_point_velocities[index]);

      const double trial_activation = risk_damped_mode ?
        pair.blockage :
        std::max(state.z, pair.raw_activation);
      candidate.metric_scalar =
        std::max(params.metric_scalar, 0.0) * trial_activation;
      if (risk_damped_mode) {
        candidate.scalar_acceleration = bounded_velocity_acceleration(
          pair.risk_acceleration -
          risk_tangent_damping_gain * candidate.scalar_velocity,
          maximum_leaf_acceleration);
      } else {
        const double trial_velocity_reference =
          prevent_speed +
          (recovery_speed - prevent_speed) * pair.alpha_stuck;
        candidate.scalar_acceleration = bounded_velocity_acceleration(
          velocity_gain * (trial_velocity_reference - candidate.scalar_velocity),
          maximum_leaf_acceleration);
      }

      Matrix6 trial_metric = base_metric;
      JointVector trial_force = base_force;
      trial_metric +=
        scalar_jacobian.transpose() * candidate.metric_scalar * scalar_jacobian;
      trial_force +=
        scalar_jacobian.transpose() *
        (candidate.metric_scalar * (candidate.scalar_acceleration - scalar_curvature));
      candidate.qdd =
        resolve_trial_root(trial_metric, trial_force, config_.solve_offset);
      if (!candidate.qdd.allFinite()) {
        return candidate;
      }

      const double joint_velocity_limit =
        std::max(config_.joint_velocity_cap.max_velocity, 0.0);
      const double joint_acceleration_limit =
        std::max(config_.max_joint_accel, 0.0);
      for (int joint = 0; joint < candidate.qdd.size(); ++joint) {
        const auto joint_index = static_cast<std::size_t>(joint);
        const double predicted_position =
          q[joint] + preview_time * qd[joint] +
          0.5 * preview_time * preview_time * candidate.qdd[joint];
        const double predicted_velocity =
          qd[joint] + preview_time * candidate.qdd[joint];
        const double buffered_lower_limit =
          config_.joint_lower_limits[joint_index] +
          std::max(config_.joint_limit_buffers[joint_index], 0.0);
        const double buffered_upper_limit =
          config_.joint_upper_limits[joint_index] -
          std::max(config_.joint_limit_buffers[joint_index], 0.0);
        const bool position_invalid =
          predicted_position < buffered_lower_limit ||
          predicted_position > buffered_upper_limit;
        const bool velocity_invalid =
          joint_velocity_limit > 0.0 &&
          std::abs(predicted_velocity) > joint_velocity_limit;
        const bool acceleration_invalid =
          joint_acceleration_limit > 0.0 &&
          std::abs(candidate.qdd[joint]) > joint_acceleration_limit;
        if (apply_hard_gates && (position_invalid || velocity_invalid || acceleration_invalid)) {
          return candidate;
        }

        const double joint_range = std::max(
          config_.joint_upper_limits[joint_index] -
          config_.joint_lower_limits[joint_index],
          1e-6);
        const bool position_near_limit =
          predicted_position - config_.joint_lower_limits[joint_index] < 0.02 * joint_range ||
          config_.joint_upper_limits[joint_index] - predicted_position < 0.02 * joint_range;
        const bool velocity_near_limit =
          joint_velocity_limit > 0.0 &&
          std::abs(predicted_velocity) > 0.95 * joint_velocity_limit;
        const bool acceleration_near_limit =
          joint_acceleration_limit > 0.0 &&
          std::abs(candidate.qdd[joint]) > 0.95 * joint_acceleration_limit;
        candidate.saturation_risk =
          candidate.saturation_risk ||
          position_near_limit ||
          velocity_near_limit ||
          acceleration_near_limit;
      }

      const Eigen::Vector3d active_displacement =
        preview_time * context.control_point_velocities[index] +
        0.5 * preview_time * preview_time *
        (
          point_jacobian * candidate.qdd +
          context.control_point_curvatures[index]);
      candidate.tangent_displacement =
        candidate.direction.dot(active_displacement);
      if (
        apply_hard_gates &&
        candidate.tangent_displacement <
        std::max(
          params.candidate_min_displacement,
          kMinimumPredictedTangentDisplacement))
      {
        return candidate;
      }

      candidate.sector_risk = sector_risk(candidate.qdd);
      if (
        apply_hard_gates &&
        candidate.sector_risk > clamp01(params.sector_risk_hard_limit))
      {
        return candidate;
      }

      const Eigen::Vector3d predicted_tcp_position =
        context.tcp_position +
        preview_time * context.tcp_velocity +
        0.5 * preview_time * preview_time *
        (context.tcp_jacobian * candidate.qdd + context.tcp_curvature);
      const double predicted_goal_error = (predicted_tcp_position - goal).norm();
      candidate.goal_score = std::clamp(
        (baseline_goal_error - predicted_goal_error) /
        std::max(params.goal_score_scale, 1e-6),
        -1.0,
        1.0);
      candidate.continuity_score = state.tangent_valid ?
        0.5 * (
        1.0 +
        std::clamp(candidate.direction.dot(state.tangent), -1.0, 1.0)) :
        0.5;
      candidate.blocked_penalty = failure_memory_penalty(candidate.direction);
      candidate.acceleration_jump_penalty = state.previous_qdd_valid ?
        clamp01(
        (candidate.qdd - state.previous_qdd).norm() /
        std::max(params.accel_jump_scale, 1e-6)) :
        clamp01(
        (candidate.qdd - base_qdd).norm() /
        std::max(params.accel_jump_scale, 1e-6));
      candidate.score =
        params.goal_weight * candidate.goal_score -
        std::max(params.sector_risk_weight, 0.0) * candidate.sector_risk +
        params.continuity_weight * candidate.continuity_score -
        std::max(params.blocked_memory_penalty_weight, 0.0) *
        candidate.blocked_penalty -
        std::max(params.accel_jump_weight, 0.0) *
        candidate.acceleration_jump_penalty;
      candidate.feasible = std::isfinite(candidate.score);
      return candidate;
    };

  struct SelectionResult
  {
    std::optional<CanonicalCandidateEvaluation> selected;
    std::vector<CanonicalCandidateEvaluation> records;
  };

  const auto select_direction = [
      &context,
      &state,
      &params,
      &evaluate_candidate,
      minimum_tangent_norm](const SensorPairEvaluation & pair)
    {
      SelectionResult result;
      const std::size_t point_index = pair.control_point_index;
      const auto & sensor = RB10Model::sensor_control_points[point_index];
      const Eigen::Vector3d tangent_bias_world =
        context.link_rotations[sensor.parent_link] * sensor.local_tangent_bias;
      const auto nominal_tangent = tangent_projection(
        pair.nominal_velocity,
        pair.obstacle_direction,
        minimum_tangent_norm);
      const auto previous_tangent = state.tangent_valid ?
        tangent_projection(
        state.tangent,
        pair.obstacle_direction,
        minimum_tangent_norm) :
        std::nullopt;
      const auto bias_tangent = tangent_projection(
        tangent_bias_world,
        pair.obstacle_direction,
        minimum_tangent_norm);

      std::optional<Eigen::Vector3d> basis_u = nominal_tangent;
      if (!basis_u.has_value()) {
        basis_u = previous_tangent;
      }
      if (!basis_u.has_value()) {
        basis_u = bias_tangent;
      }
      if (!basis_u.has_value()) {
        const Eigen::Vector3d reference =
          std::abs(pair.obstacle_direction.z()) < 0.9 ?
          Eigen::Vector3d::UnitZ() :
          Eigen::Vector3d::UnitY();
        basis_u = tangent_projection(
          reference,
          pair.obstacle_direction,
          minimum_tangent_norm);
      }
      if (!basis_u.has_value()) {
        basis_u = tangent_projection(
          Eigen::Vector3d::UnitX(),
          pair.obstacle_direction,
          minimum_tangent_norm);
      }
      if (!basis_u.has_value()) {
        return result;
      }

      Eigen::Vector3d basis_v = pair.obstacle_direction.cross(basis_u.value());
      const double basis_v_norm = basis_v.norm();
      if (basis_v_norm <= minimum_tangent_norm) {
        return result;
      }
      basis_v /= basis_v_norm;

      const auto direction_at = [&basis_u, &basis_v](double theta) {
          return
            std::cos(theta) * basis_u.value() +
            std::sin(theta) * basis_v;
        };
      const auto consider = [&result](const CanonicalCandidateEvaluation & candidate) {
          result.records.push_back(candidate);
          if (
            candidate.feasible &&
            (
              !result.selected.has_value() ||
              candidate.score > result.selected->score))
          {
            result.selected = candidate;
          }
        };

      std::optional<CanonicalCandidateEvaluation> best_coarse;
      for (std::size_t seed = 0; seed < kCoarseCandidateCount; ++seed) {
        const double theta =
          2.0 * kPi * static_cast<double>(seed) /
          static_cast<double>(kCoarseCandidateCount);
        const auto candidate = evaluate_candidate(
          pair,
          direction_at(theta),
          static_cast<int>(seed),
          theta,
          true);
        consider(candidate);
        if (
          candidate.feasible &&
          (
            !best_coarse.has_value() ||
            candidate.score > best_coarse->score))
        {
          best_coarse = candidate;
        }
      }

      if (best_coarse.has_value()) {
        const double half_width =
          2.0 * kPi / static_cast<double>(kCoarseCandidateCount);
        double lower = best_coarse->theta - half_width;
        double upper = best_coarse->theta + half_width;
        constexpr double golden_ratio_conjugate = 0.6180339887498948482;
        double left_theta = upper - golden_ratio_conjugate * (upper - lower);
        double right_theta = lower + golden_ratio_conjugate * (upper - lower);
        auto left = evaluate_candidate(pair, direction_at(left_theta), 100, left_theta, true);
        auto right = evaluate_candidate(pair, direction_at(right_theta), 101, right_theta, true);
        const int refinement_iterations = std::clamp(params.refinement_iterations, 0, 12);
        for (int iteration = 0; iteration < refinement_iterations; ++iteration) {
          if (
            candidate_score_or_negative_infinity(left) <
            candidate_score_or_negative_infinity(right))
          {
            lower = left_theta;
            left_theta = right_theta;
            left = right;
            right_theta = lower + golden_ratio_conjugate * (upper - lower);
            right = evaluate_candidate(
              pair,
              direction_at(right_theta),
              102 + 2 * iteration,
              right_theta,
              true);
          } else {
            upper = right_theta;
            right_theta = left_theta;
            right = left;
            left_theta = upper - golden_ratio_conjugate * (upper - lower);
            left = evaluate_candidate(
              pair,
              direction_at(left_theta),
              103 + 2 * iteration,
              left_theta,
              true);
          }
        }
        const auto refined =
          candidate_score_or_negative_infinity(left) >
          candidate_score_or_negative_infinity(right) ?
          left :
          right;
        consider(refined);
      }

      const auto append_extra = [
          &result,
          &consider,
          &evaluate_candidate,
          &pair](const std::optional<Eigen::Vector3d> & direction, int seed_index)
        {
          if (!direction.has_value()) {
            return;
          }
          for (const auto & existing : result.records) {
            if (existing.direction.dot(direction.value()) >= kCandidateDuplicateDot) {
              return;
            }
          }
          consider(evaluate_candidate(pair, direction.value(), seed_index, 0.0, true));
        };
      append_extra(previous_tangent, 200);
      append_extra(nominal_tangent, 201);
      append_extra(bias_tangent, 202);
      append_extra(
        tangent_projection(
          -tangent_bias_world,
          pair.obstacle_direction,
          minimum_tangent_norm),
        203);

      return result;
    };

  std::vector<CanonicalCandidateEvaluation> diagnostic_candidates;
  std::optional<CanonicalCandidateEvaluation> selected_diagnostic_candidate;

  const auto remember_failed_direction = [&state]() {
      if (!state.pending_failure_memory || !state.tangent_valid) {
        state.pending_failure_memory = false;
        return;
      }

      auto matching = std::find_if(
        state.failure_memory.begin(),
        state.failure_memory.end(),
        [&state](const auto & memory) {
          return
            memory.valid &&
            memory.tangent.dot(state.tangent) >= kCandidateDuplicateDot;
        });
      if (matching != state.failure_memory.end()) {
        matching->tangent = state.tangent;
        matching->strength = 1.0;
      } else {
        auto & memory =
          state.failure_memory[state.failure_memory_cursor % state.failure_memory.size()];
        memory.valid = true;
        memory.tangent = state.tangent;
        memory.strength = 1.0;
        state.failure_memory_cursor =
          (state.failure_memory_cursor + 1) % state.failure_memory.size();
      }
      state.pending_failure_memory = false;
    };

  const auto start_phase = [&state](TangentEscapeCanonicalPhase phase) {
      state.phase = phase;
      state.phase_elapsed_s = 0.0;
      state.phase_start_lambda = state.lambda;
      state.phase_start_drive_ramp = state.drive_ramp;
      state.phase_start_release_brake = state.release_brake;
    };

  const auto clear_active_episode = [
      &state,
      &clear_failure_memory]()
    {
      state.active_pair_valid = false;
      state.pending_pair_valid = false;
      state.force_direction_change = false;
      state.pending_failure_memory = false;
      state.current_score = -std::numeric_limits<double>::infinity();
      state.lambda = 0.0;
      state.drive_ramp = 0.0;
      state.release_brake = 0.0;
      state.command_distance = 0.0;
      state.actual_distance = 0.0;
      state.active_age_s = 0.0;
      state.z = 0.0;
      state.phase = TangentEscapeCanonicalPhase::Off;
      state.phase_elapsed_s = 0.0;
      clear_failure_memory();
    };

  const auto activate_pair = [
      &state,
      &select_direction,
      &evaluate_candidate,
      &diagnostic_candidates,
      &selected_diagnostic_candidate,
      &clear_failure_memory,
      &params](
      const SensorPairEvaluation & pair,
      bool same_episode)
    {
      if (!same_episode) {
        clear_failure_memory();
      }
      SelectionResult selection = select_direction(pair);
      diagnostic_candidates = selection.records;
      if (!selection.selected.has_value()) {
        return false;
      }
      if (
        same_episode &&
        !state.force_direction_change &&
        state.tangent_valid)
      {
        const auto current_direction = evaluate_candidate(
          pair,
          state.tangent,
          -2,
          0.0,
          true);
        if (
          current_direction.feasible &&
          selection.selected->score <=
          current_direction.score + std::max(params.direction_switch_margin, 0.0))
        {
          selection.records.push_back(current_direction);
          selection.selected = current_direction;
        }
      }
      selected_diagnostic_candidate = selection.selected;

      state.active_pair_valid = true;
      state.control_point_index = pair.control_point_index;
      state.obstacle_key = pair.obstacle_key;
      state.pending_pair_valid = false;
      state.tangent_valid = true;
      state.tangent = selection.selected->direction;
      state.obstacle_direction_at_selection = pair.obstacle_direction;
      state.current_score = selection.selected->score;
      state.force_direction_change = false;
      state.handoff_reason = 0;
      state.lambda = 0.0;
      state.drive_ramp = 0.0;
      state.release_brake = 0.0;
      state.desired_velocity = 0.0;
      state.phase = TangentEscapeCanonicalPhase::Engage;
      state.phase_elapsed_s = 0.0;
      state.phase_start_lambda = 0.0;
      state.phase_start_drive_ramp = 0.0;
      state.phase_start_release_brake = 0.0;
      state.active_age_s = 0.0;
      state.command_distance = 0.0;
      state.actual_distance = 0.0;
      state.episode_start_sector_risk = selection.selected->sector_risk;
      state.last_sector_risk = selection.selected->sector_risk;
      ++state.handoff_generation;
      return true;
    };

  if (
    state.phase == TangentEscapeCanonicalPhase::Off &&
    !state.active_pair_valid &&
    best_pair != nullptr &&
    best_pair->blockage >= minimum_pair_activation &&
    (
      state.handoff_reason != 6 ||
      state.phase_elapsed_s >= std::max(params.minimum_drive_duration, 0.0)))
  {
    if (!activate_pair(*best_pair, false)) {
      clear_active_episode();
    } else if (risk_damped_mode) {
      // There is no old Escape direction to hand off from on first entry.
      // B is the complete activation, so normal driving starts with lambda=1.
      state.lambda = 1.0;
      state.drive_ramp = 1.0;
      state.phase = TangentEscapeCanonicalPhase::Drive;
      state.phase_elapsed_s = 0.0;
    }
    current_pair = state.active_pair_valid ?
      find_pair(state.control_point_index, state.obstacle_key) :
      nullptr;
  }

  double current_scalar_velocity = 0.0;
  if (
    state.active_pair_valid &&
    state.tangent_valid &&
    state.control_point_index < point_count)
  {
    current_scalar_velocity =
      state.tangent.dot(context.control_point_velocities[state.control_point_index]);
  }

  std::optional<CanonicalCandidateEvaluation> active_trial;
  if (state.active_pair_valid && state.tangent_valid && current_pair != nullptr) {
    active_trial = evaluate_candidate(
      *current_pair,
      state.tangent,
      -1,
      0.0,
      false);
  }

  bool zero_effect_pair_replacement = false;
  const bool in_engage_or_drive =
    state.phase == TangentEscapeCanonicalPhase::Engage ||
    state.phase == TangentEscapeCanonicalPhase::Drive;
  if (
    risk_damped_mode &&
    state.active_pair_valid &&
    (
      current_pair == nullptr ||
      current_pair->blockage <= 0.0))
  {
    // The risk-damped leaf has no RELEASE authority.  Invalid geometry or
    // B=0 removes its metric and force in this same solve.  If another valid
    // pair already exists, start a fresh zero-effect engage episode for it.
    clear_active_episode();
    current_pair = nullptr;
    active_trial.reset();
    if (
      best_pair != nullptr &&
      best_pair->blockage >= minimum_pair_activation)
    {
      // This is a pair replacement, not a first-ever OFF -> active entry.
      // Keep the newly selected axis at zero effect for this solve, then let
      // Engage ramp lambda from 0 to 1 on subsequent solves.
      zero_effect_pair_replacement = activate_pair(*best_pair, false);
      current_pair = state.active_pair_valid ?
        find_pair(state.control_point_index, state.obstacle_key) :
        nullptr;
      if (current_pair != nullptr && state.tangent_valid) {
        active_trial = evaluate_candidate(
          *current_pair,
          state.tangent,
          -1,
          0.0,
          false);
      }
    }
  }
  if (state.active_pair_valid && in_engage_or_drive) {
    const bool tangent_invalid =
      current_pair != nullptr &&
      std::abs(state.tangent.dot(current_pair->obstacle_direction)) >
      std::clamp(
        params.normal_tolerance,
        kTangentOrthogonalityTolerance,
        1.0);
    const bool critical_sector =
      active_trial.has_value() &&
      active_trial->sector_risk > clamp01(params.sector_risk_hard_limit);
    const bool switch_pair =
      current_pair != nullptr &&
      best_pair != nullptr &&
      !same_pair(
        *best_pair,
        state.control_point_index,
        state.obstacle_key) &&
      best_pair->blockage >
      current_pair->blockage + std::max(params.pair_switch_margin, 0.0);

    if (goal_changed || tangent_invalid || critical_sector || switch_pair) {
      const SensorPairEvaluation * requested_pair = switch_pair ? best_pair : current_pair;
      if (requested_pair != nullptr) {
        state.pending_pair_valid = true;
        state.pending_control_point_index = requested_pair->control_point_index;
        state.pending_obstacle_key = requested_pair->obstacle_key;
        state.force_direction_change = tangent_invalid || critical_sector;
        state.handoff_reason = switch_pair ? 2 : (goal_changed ? 3 : 4);
        start_phase(
          risk_damped_mode ?
          TangentEscapeCanonicalPhase::ReselectLambdaDown :
          TangentEscapeCanonicalPhase::ReselectDriveDown);
      }
    } else if (
      !risk_damped_mode &&
      (
        current_pair == nullptr ||
        current_pair->blockage <
        std::max(params.release_blockage_threshold, 0.0)))
    {
      state.handoff_reason = 1;
      start_phase(TangentEscapeCanonicalPhase::ReleaseDriveDown);
    } else if (
      !risk_damped_mode &&
      state.active_age_s >= std::max(params.minimum_drive_duration, 0.0) &&
      state.filtered_goal_progress >=
      std::max(params.progress_ok_threshold, params.progress_low_threshold) &&
      current_pair->alpha_stuck <= 0.1)
    {
      // A recovered goal-progress signal closes this episode. Reason 6
      // enforces a short OFF cooldown before preventive re-entry.
      state.handoff_reason = 6;
      start_phase(TangentEscapeCanonicalPhase::ReleaseDriveDown);
    }
  }

  if (!zero_effect_pair_replacement) {
    state.phase_elapsed_s += dt;
  }
  const double drive_ramp_duration = std::max(params.drive_ramp_duration, 0.0);
  const double handoff_duration = std::max(params.handoff_duration, 0.0);
  switch (state.phase) {
    case TangentEscapeCanonicalPhase::Off:
      state.lambda = 0.0;
      state.drive_ramp = 0.0;
      state.release_brake = 0.0;
      break;
    case TangentEscapeCanonicalPhase::Engage:
      state.lambda = ramp_between(
        state.phase_start_lambda,
        1.0,
        state.phase_elapsed_s,
        handoff_duration);
      state.drive_ramp = risk_damped_mode ?
        1.0 :
        ramp_between(
          state.phase_start_drive_ramp,
          1.0,
          state.phase_elapsed_s,
          drive_ramp_duration);
      state.release_brake = 0.0;
      if (
        state.phase_elapsed_s >=
        (risk_damped_mode ?
        handoff_duration :
        std::max(handoff_duration, drive_ramp_duration)))
      {
        state.lambda = 1.0;
        state.drive_ramp = 1.0;
        state.release_brake = 0.0;
        start_phase(TangentEscapeCanonicalPhase::Drive);
      }
      break;
    case TangentEscapeCanonicalPhase::Drive:
      state.lambda = 1.0;
      state.drive_ramp = 1.0;
      state.release_brake = 0.0;
      break;
    case TangentEscapeCanonicalPhase::ReleaseDriveDown:
    case TangentEscapeCanonicalPhase::ReselectDriveDown:
      state.drive_ramp = ramp_between(
        state.phase_start_drive_ramp,
        0.0,
        state.phase_elapsed_s,
        drive_ramp_duration);
      state.release_brake = ramp_between(
        state.phase_start_release_brake,
        1.0,
        state.phase_elapsed_s,
        drive_ramp_duration);
      if (state.phase_elapsed_s >= drive_ramp_duration) {
        state.drive_ramp = 0.0;
        state.release_brake = 1.0;
        start_phase(
          state.phase == TangentEscapeCanonicalPhase::ReleaseDriveDown ?
          TangentEscapeCanonicalPhase::ReleaseBrake :
          TangentEscapeCanonicalPhase::ReselectBrake);
      }
      break;
    case TangentEscapeCanonicalPhase::ReleaseBrake:
    case TangentEscapeCanonicalPhase::ReselectBrake:
      state.drive_ramp = 0.0;
      state.release_brake = 1.0;
      {
        const bool tangent_speed_stopped =
        std::abs(current_scalar_velocity) <=
        std::max(params.release_stop_speed, 0.0) &&
        std::abs(state.desired_velocity) <=
        std::max(params.release_stop_speed, 0.0);
        // A base leaf can keep driving the old scalar axis while a reselect is
        // braking.  Waiting only for zero tangent speed can therefore leave a
        // stale world-frame tangent attached indefinitely as the obstacle
        // normal changes.  Bound only RESELECT braking; RELEASE keeps its
        // residual-velocity braking contract.  The subsequent quintic
        // lambda-down and the actual axis swap still occur at zero effect.
        const double reselect_brake_timeout =
          std::max(handoff_duration, 3.0 * velocity_time_constant);
        const bool reselect_brake_timed_out =
          state.phase == TangentEscapeCanonicalPhase::ReselectBrake &&
          state.phase_elapsed_s >= reselect_brake_timeout;
        if (!tangent_speed_stopped && !reselect_brake_timed_out) {
          break;
        }
        start_phase(
          state.phase == TangentEscapeCanonicalPhase::ReleaseBrake ?
          TangentEscapeCanonicalPhase::ReleaseLambdaDown :
          TangentEscapeCanonicalPhase::ReselectLambdaDown);
      }
      break;
    case TangentEscapeCanonicalPhase::ReleaseLambdaDown:
    case TangentEscapeCanonicalPhase::ReselectLambdaDown:
      state.drive_ramp = 0.0;
      state.release_brake = risk_damped_mode ? 0.0 : 1.0;
      state.lambda = ramp_between(
        state.phase_start_lambda,
        0.0,
        state.phase_elapsed_s,
        handoff_duration);
      if (state.phase_elapsed_s >= handoff_duration) {
        const bool reselect =
          state.phase == TangentEscapeCanonicalPhase::ReselectLambdaDown;
        state.lambda = 0.0;
        if (!reselect) {
          clear_active_episode();
        } else {
          const std::size_t old_control_point = state.control_point_index;
          const std::int64_t old_obstacle_key = state.obstacle_key;
          const std::size_t target_control_point = state.pending_pair_valid ?
            state.pending_control_point_index :
            state.control_point_index;
          const std::int64_t target_obstacle_key = state.pending_pair_valid ?
            state.pending_obstacle_key :
            state.obstacle_key;
          const bool continuing_episode =
            old_control_point == target_control_point &&
            old_obstacle_key == target_obstacle_key;
          if (continuing_episode) {
            remember_failed_direction();
          } else {
            state.pending_failure_memory = false;
            clear_failure_memory();
          }

          const SensorPairEvaluation * target_pair =
            find_pair(target_control_point, target_obstacle_key);
          if (
            target_pair == nullptr ||
            target_pair->blockage < minimum_pair_activation ||
            !activate_pair(*target_pair, continuing_episode))
          {
            clear_active_episode();
          }
        }
      }
      break;
  }

  current_pair = state.active_pair_valid ?
    find_pair(state.control_point_index, state.obstacle_key) :
    nullptr;
  if (risk_damped_mode) {
    state.z = current_pair != nullptr ? clamp01(current_pair->blockage) : 0.0;
  }
  if (
    state.active_pair_valid &&
    state.tangent_valid &&
    state.control_point_index < point_count)
  {
    current_scalar_velocity =
      state.tangent.dot(context.control_point_velocities[state.control_point_index]);
  } else {
    current_scalar_velocity = 0.0;
  }

  const double alpha_stuck = current_pair != nullptr ?
    current_pair->alpha_stuck :
    0.0;
  const double legacy_speed_reference =
    prevent_speed +
    (recovery_speed - prevent_speed) * alpha_stuck;
  double velocity_reference = state.drive_ramp * legacy_speed_reference;
  double desired_velocity_derivative = 0.0;
  if (risk_damped_mode) {
    state.desired_velocity = 0.0;
    velocity_reference =
      current_pair != nullptr && risk_tangent_damping_gain > 0.0 ?
      current_pair->risk_acceleration / risk_tangent_damping_gain :
      0.0;
  } else {
    const double old_desired_velocity = state.desired_velocity;
    const double desired_velocity_alpha =
      clamp01(dt / velocity_time_constant);
    state.desired_velocity +=
      desired_velocity_alpha * (velocity_reference - state.desired_velocity);
    desired_velocity_derivative =
      (state.desired_velocity - old_desired_velocity) / dt;
  }

  double scalar_acceleration = 0.0;
  double release_velocity_gate = 0.0;
  double effective_metric_scalar = 0.0;
  double risk_distance_acceleration = 0.0;
  double risk_approach_acceleration = 0.0;
  double risk_acceleration = 0.0;
  double risk_damping_acceleration = 0.0;
  double risk_acceleration_input = 0.0;
  double filtered_clearance_rate = 0.0;
  double closing_speed = 0.0;
  Eigen::Vector3d control_point = Eigen::Vector3d::Zero();
  Eigen::Vector3d obstacle_center = Eigen::Vector3d::Zero();
  Eigen::Vector3d outward_normal = -state.obstacle_direction_at_selection;
  double scalar_coordinate = 0.0;
  if (
    state.active_pair_valid &&
    state.tangent_valid &&
    state.control_point_index < point_count)
  {
    const std::size_t point_index = state.control_point_index;
    control_point = context.control_points[point_index].position;
    if (current_pair != nullptr) {
      obstacle_center = current_pair->obstacle_center;
      outward_normal = current_pair->outward_normal;
    }
    scalar_coordinate = state.tangent.dot(control_point);
    double effective_activation = 0.0;
    if (risk_damped_mode) {
      if (current_pair != nullptr) {
        filtered_clearance_rate = current_pair->clearance_rate;
        closing_speed = current_pair->closing_speed;
        risk_distance_acceleration = current_pair->risk_distance_acceleration;
        risk_approach_acceleration = current_pair->risk_approach_acceleration;
        risk_acceleration = current_pair->risk_acceleration;
      }
      risk_damping_acceleration =
        risk_tangent_damping_gain * current_scalar_velocity;
      risk_acceleration_input =
        risk_acceleration - risk_damping_acceleration;
      scalar_acceleration = bounded_velocity_acceleration(
        risk_acceleration_input,
        maximum_leaf_acceleration);
      effective_activation =
        state.lambda *
        (current_pair != nullptr ? current_pair->blockage : 0.0);
    } else {
      const double acceleration_input =
        desired_velocity_derivative +
        velocity_gain * (state.desired_velocity - current_scalar_velocity);
      scalar_acceleration = bounded_velocity_acceleration(
        acceleration_input,
        maximum_leaf_acceleration);
      release_velocity_gate = quintic01(
        (
          std::abs(current_scalar_velocity) -
          std::max(params.release_stop_speed, 0.0)) /
        std::max(
          params.release_hold_speed - params.release_stop_speed,
          1e-9));
      effective_activation =
        state.lambda *
        (
          state.z +
          (1.0 - state.z) * state.release_brake * release_velocity_gate);
    }
    effective_metric_scalar =
      std::max(params.metric_scalar, 0.0) * std::max(effective_activation, 0.0);

    const RowVector6 scalar_jacobian =
      state.tangent.transpose() * context.control_point_jacobians[point_index];
    const double scalar_curvature =
      state.tangent.dot(context.control_point_curvatures[point_index]);
    metric +=
      scalar_jacobian.transpose() * effective_metric_scalar * scalar_jacobian;
    force +=
      scalar_jacobian.transpose() *
      (effective_metric_scalar * (scalar_acceleration - scalar_curvature));
  }

  if (state.active_pair_valid) {
    state.active_age_s += dt;
  }

  active_trial.reset();
  if (state.active_pair_valid && state.tangent_valid && current_pair != nullptr) {
    active_trial = evaluate_candidate(
      *current_pair,
      state.tangent,
      -1,
      0.0,
      false);
    if (active_trial->feasible) {
      state.last_sector_risk = active_trial->sector_risk;
    }
  }

  const double monitoring_gate = state.lambda * state.z;
  const double monitoring_reference_speed = risk_damped_mode ?
    velocity_reference :
    std::max(state.desired_velocity, 0.0);
  state.command_distance +=
    dt * monitoring_gate * std::max(monitoring_reference_speed, 0.0);
  state.actual_distance +=
    dt * monitoring_gate * std::max(current_scalar_velocity, 0.0);
  const double move_ratio =
    state.actual_distance / std::max(state.command_distance, 1e-9);

  if (
    state.phase == TangentEscapeCanonicalPhase::Drive &&
    current_pair != nullptr &&
    active_trial.has_value() &&
    state.active_age_s >= std::max(params.minimum_drive_duration, 0.0) &&
    state.command_distance >= std::max(params.command_test_distance, 0.0) &&
    move_ratio < std::max(params.minimum_move_ratio, 0.0))
  {
    const bool sector_risk_increased =
      active_trial->sector_risk - state.episode_start_sector_risk >
      std::max(params.sector_risk_change_threshold, 0.0);
    if (sector_risk_increased || active_trial->saturation_risk) {
      state.pending_pair_valid = true;
      state.pending_control_point_index = state.control_point_index;
      state.pending_obstacle_key = state.obstacle_key;
      state.pending_failure_memory = true;
      state.force_direction_change = true;
      state.handoff_reason = 5;
      start_phase(
        risk_damped_mode ?
        TangentEscapeCanonicalPhase::ReselectLambdaDown :
        TangentEscapeCanonicalPhase::ReselectDriveDown);
    }
  }

  if (current_pair != nullptr) {
    state.last_alpha_stuck = current_pair->alpha_stuck;
    state.last_raw_activation = current_pair->raw_activation;
    state.last_blockage = current_pair->blockage;
    state.last_clearance = current_pair->clearance;
    state.last_beta = current_pair->beta;
  } else {
    state.last_alpha_stuck = 0.0;
    state.last_raw_activation = 0.0;
    state.last_blockage = 0.0;
  }

  state.previous_qdd =
    resolve_trial_root(metric, force, config_.solve_offset);
  state.previous_qdd_valid = state.previous_qdd.allFinite();

  if (!state.active_pair_valid || !state.tangent_valid) {
    if (std::abs(state.desired_velocity) < 1e-8) {
      state.desired_velocity = 0.0;
    }
    publish_inactive();
    return;
  }

  CanonicalCandidateEvaluation diagnostic_candidate;
  if (selected_diagnostic_candidate.has_value()) {
    diagnostic_candidate = selected_diagnostic_candidate.value();
  } else if (active_trial.has_value()) {
    diagnostic_candidate = active_trial.value();
  } else {
    diagnostic_candidate.direction = state.tangent;
    diagnostic_candidate.score = state.current_score;
  }

  double maximum_memory = 0.0;
  for (const auto & memory : state.failure_memory) {
    if (memory.valid) {
      maximum_memory = std::max(maximum_memory, memory.strength);
    }
  }

  int high_level_state = 0;
  switch (state.phase) {
    case TangentEscapeCanonicalPhase::Off:
      high_level_state = 0;
      break;
    case TangentEscapeCanonicalPhase::Engage:
    case TangentEscapeCanonicalPhase::Drive:
      high_level_state = risk_damped_mode ? 1 : (alpha_stuck >= 0.5 ? 2 : 1);
      break;
    case TangentEscapeCanonicalPhase::ReleaseDriveDown:
    case TangentEscapeCanonicalPhase::ReleaseBrake:
    case TangentEscapeCanonicalPhase::ReleaseLambdaDown:
      high_level_state = 3;
      break;
    case TangentEscapeCanonicalPhase::ReselectDriveDown:
    case TangentEscapeCanonicalPhase::ReselectBrake:
    case TangentEscapeCanonicalPhase::ReselectLambdaDown:
      high_level_state = 4;
      break;
  }

  std::vector<double> values(
    risk_damped_mode ? kRiskDampedDebugSize : kCanonicalDebugSize,
    0.0);
  values[0] = 1.0;
  values[1] = static_cast<double>(state.control_point_index);
  values[2] = current_pair != nullptr ? current_pair->clearance : state.last_clearance;
  values[3] = current_pair != nullptr ? current_pair->beta : state.last_beta;
  values[4] = current_pair != nullptr ? current_pair->alpha_distance : 0.0;
  values[5] = current_pair != nullptr ? current_pair->alpha_blocking : 0.0;
  values[6] = state.z;
  values[7] = state.current_score;
  values[8] = current_scalar_velocity;
  values[9] = scalar_acceleration;
  values[10] = effective_metric_scalar;
  values[11] = control_point.x();
  values[12] = control_point.y();
  values[13] = control_point.z();
  values[14] = obstacle_center.x();
  values[15] = obstacle_center.y();
  values[16] = obstacle_center.z();
  values[17] = outward_normal.x();
  values[18] = outward_normal.y();
  values[19] = outward_normal.z();
  values[20] = state.tangent.x();
  values[21] = state.tangent.y();
  values[22] = state.tangent.z();
  values[23] = scalar_acceleration * state.tangent.x();
  values[24] = scalar_acceleration * state.tangent.y();
  values[25] = scalar_acceleration * state.tangent.z();
  values[26] = risk_damped_mode ? 7.0 : 6.0;
  values[27] = scalar_coordinate;
  values[28] = velocity_reference;
  values[29] = current_scalar_velocity;
  values[30] = risk_damped_mode ?
    velocity_reference - current_scalar_velocity :
    state.desired_velocity - current_scalar_velocity;
  values[31] = static_cast<double>(state.obstacle_key);
  values[32] = static_cast<double>(state.handoff_reason);
  values[33] = static_cast<double>(static_cast<int>(state.phase));
  values[34] = state.tangent.x();
  values[35] = state.tangent.y();
  values[36] = state.tangent.z();
  values[37] = static_cast<double>(diagnostic_candidates.size());
  values[38] = static_cast<double>(diagnostic_candidate.seed_index);
  values[39] = 1.0;
  values[40] = diagnostic_candidate.score;
  values[41] = diagnostic_candidate.goal_score;
  values[42] = diagnostic_candidate.continuity_score;
  values[43] = diagnostic_candidate.sector_risk;
  values[44] = alpha_stuck;
  values[45] = state.lambda;
  values[46] = release_velocity_gate;
  values[47] = static_cast<double>(high_level_state);
  values[48] = state.drive_ramp;
  values[49] = state.release_brake;
  values[50] = current_pair != nullptr ? current_pair->raw_activation : 0.0;
  values[51] = state.filtered_goal_progress;
  values[52] = alpha_stuck;
  values[53] = alpha_stuck >= 0.5 ? 1.0 : 0.0;
  values[54] = state.z;
  values[55] = state.lambda;
  values[56] = diagnostic_candidate.blocked_penalty;
  values[57] = maximum_memory;
  values[58] = state.command_distance;
  values[59] = state.actual_distance;
  values[60] = move_ratio;

  if (risk_damped_mode) {
    values[61] = filtered_clearance_rate;
    values[62] = closing_speed;
    values[63] = risk_distance_acceleration;
    values[64] = risk_approach_acceleration;
    values[65] = risk_acceleration;
    values[66] = risk_damping_acceleration;
    values[67] = risk_acceleration_input;
  }

  for (const auto & candidate : diagnostic_candidates) {
    append_candidate_record(values, candidate, alpha_stuck);
  }
  if (debug_data != nullptr) {
    *debug_data = std::move(values);
  }
}

}  // namespace rb10_rmpflow_rviz
