#include "rb10_rmpflow_rviz/pinocchio_direct_solver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>

#include "rb10_rmpflow_rviz/casadi_task_graph.hpp"

namespace rb10_rmpflow_rviz
{

namespace
{

struct SolverImpl
{
  std::unordered_map<std::string, int> node_dims;
  std::unordered_map<std::string, CompiledCasadiTaskMap> compiled_task_maps;
  std::vector<std::size_t> topo_indices;
};

const SolverImpl & get_impl(const std::shared_ptr<const void> & state)
{
  return *std::static_pointer_cast<const SolverImpl>(state);
}

std::vector<std::size_t> build_topological_order(const EigenRmpConfig & config)
{
  std::unordered_map<std::string, std::size_t> enabled_nodes;
  for (std::size_t index = 0; index < config.graph_nodes.size(); ++index) {
    const auto & node = config.graph_nodes[index];
    if (!node.enabled) {
      continue;
    }
    const auto inserted = enabled_nodes.emplace(node.name, index);
    if (!inserted.second) {
      throw std::runtime_error("Duplicate graph node name: " + node.name);
    }
  }

  std::unordered_map<std::string, int> indegree;
  std::unordered_map<std::string, std::vector<std::string>> outgoing;
  for (const auto & entry : enabled_nodes) {
    indegree.emplace(entry.first, 0);
  }

  for (const auto & entry : enabled_nodes) {
    const auto & node = config.graph_nodes[entry.second];
    for (const auto & parent : node.parents) {
      if (parent == "root") {
        continue;
      }
      if (!enabled_nodes.count(parent)) {
        throw std::runtime_error(
                "Graph node " + node.name + " references missing parent " + parent);
      }
      ++indegree[node.name];
      outgoing[parent].push_back(node.name);
    }
  }

  std::vector<std::string> ready;
  ready.reserve(enabled_nodes.size());
  for (const auto & entry : indegree) {
    if (entry.second == 0) {
      ready.push_back(entry.first);
    }
  }
  std::sort(ready.begin(), ready.end());

  std::vector<std::size_t> order;
  order.reserve(enabled_nodes.size());
  while (!ready.empty()) {
    const auto name = ready.front();
    ready.erase(ready.begin());
    order.push_back(enabled_nodes.at(name));
    auto out_it = outgoing.find(name);
    if (out_it == outgoing.end()) {
      continue;
    }
    for (const auto & child : out_it->second) {
      auto & child_indegree = indegree.at(child);
      --child_indegree;
      if (child_indegree == 0) {
        ready.push_back(child);
      }
    }
    std::sort(ready.begin(), ready.end());
  }

  if (order.size() != enabled_nodes.size()) {
    throw std::runtime_error("Cycle detected in RMP graph configuration");
  }
  return order;
}

Eigen::VectorXd to_eigen_vector(const RB10Model::JointVector & value)
{
  return Eigen::VectorXd(value);
}

casadi::DM to_dm(const Eigen::VectorXd & value)
{
  auto out = casadi::DM::zeros(value.size(), 1);
  auto & nonzeros = out.nonzeros();
  nonzeros.resize(static_cast<std::size_t>(value.size()));
  for (Eigen::Index index = 0; index < value.size(); ++index) {
    nonzeros[static_cast<std::size_t>(index)] = value[index];
  }
  return out;
}

Eigen::VectorXd dm_to_vector(const casadi::DM & value)
{
  const auto dense = casadi::DM::densify(value);
  const auto elements = dense.get_elements();
  Eigen::VectorXd out(static_cast<Eigen::Index>(elements.size()));
  for (std::size_t index = 0; index < elements.size(); ++index) {
    out[static_cast<Eigen::Index>(index)] = elements[index];
  }
  return out;
}

Eigen::MatrixXd dm_to_matrix(const casadi::DM & value)
{
  const auto dense = casadi::DM::densify(value);
  const auto elements = dense.get_elements();
  Eigen::MatrixXd out(dense.size1(), dense.size2());
  for (casadi_int col = 0; col < dense.size2(); ++col) {
    for (casadi_int row = 0; row < dense.size1(); ++row) {
      const std::size_t flat = static_cast<std::size_t>(col * dense.size1() + row);
      out(row, col) = elements[flat];
    }
  }
  return out;
}

int link_index_from_name(const std::string & link_name)
{
  for (std::size_t index = 0; index < RB10Model::link_names.size(); ++index) {
    if (link_name == RB10Model::link_names[index]) {
      return static_cast<int>(index);
    }
  }
  throw std::runtime_error("Unsupported RB10 link name: " + link_name);
}

bool body_obstacle_interacts_with_sensor_control_point(
  int,
  const BodyObstacle & obstacle)
{
  (void)obstacle;
  return false;
}

double smoothstep01(double value)
{
  const double t = std::clamp(value, 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

double tangent_escape_activation(
  double clearance,
  const TangentEscapeRmpParams & params)
{
  if (clearance >= params.influence_distance) {
    return 0.0;
  }
  if (clearance <= params.safe_distance) {
    return 1.0;
  }

  const double span = std::max(params.influence_distance - params.safe_distance, 1e-9);
  const double normalized = (clearance - params.safe_distance) / span;
  return 1.0 - smoothstep01(normalized);
}

double tangent_escape_blocking_activation(
  double beta,
  const TangentEscapeRmpParams & params)
{
  const double beta_on = std::clamp(
    std::min(params.goal_block_beta_on, params.goal_block_beta_full), -1.0, 1.0);
  const double beta_full = std::clamp(
    std::max(params.goal_block_beta_on, params.goal_block_beta_full), -1.0, 1.0);
  const double span = std::max(beta_full - beta_on, 1e-9);
  return smoothstep01((beta - beta_on) / span);
}

bool is_nominal_motion_leaf_type(const std::string & leaf_type)
{
  return leaf_type == "cspace_target" ||
         leaf_type == "joint_limit" ||
         leaf_type == "joint_velocity_cap" ||
         leaf_type == "target" ||
         leaf_type == "axis_target" ||
         leaf_type == "wrist_axis_target" ||
         leaf_type == "damping";
}

std::optional<Eigen::Vector3d> project_to_tangent_direction(
  const Eigen::Vector3d & direction,
  const Eigen::Vector3d & normal,
  double min_tangent_norm)
{
  if (!direction.allFinite()) {
    return std::nullopt;
  }

  Eigen::Vector3d tangent = direction - direction.dot(normal) * normal;
  const double tangent_norm = tangent.norm();
  if (tangent_norm <= min_tangent_norm) {
    return std::nullopt;
  }
  return tangent / tangent_norm;
}

Eigen::Vector3d reproject_mode_tangent(
  const Eigen::Vector3d & stored_tangent,
  const Eigen::Vector3d & fallback_tangent,
  const Eigen::Vector3d & normal,
  double min_tangent_norm)
{
  auto projected = project_to_tangent_direction(stored_tangent, normal, min_tangent_norm);
  if (!projected.has_value()) {
    projected = project_to_tangent_direction(fallback_tangent, normal, min_tangent_norm);
  }
  if (!projected.has_value()) {
    return fallback_tangent;
  }

  Eigen::Vector3d tangent = projected.value();
  if (fallback_tangent.allFinite() && tangent.dot(fallback_tangent) < 0.0) {
    tangent = -tangent;
  }
  return tangent;
}

double stable_log_cosh(double value)
{
  const double magnitude = std::abs(value);
  return magnitude + std::log1p(std::exp(-2.0 * magnitude)) - std::log(2.0);
}

double bounded_spring_acceleration(double error, double gain, double acceleration_limit)
{
  if (gain <= 0.0) {
    return 0.0;
  }
  if (acceleration_limit <= 0.0) {
    return gain * error;
  }
  return acceleration_limit * std::tanh(gain * error / acceleration_limit);
}

double bounded_spring_potential(
  double error,
  double gain,
  double acceleration_limit,
  double metric_scalar)
{
  if (gain <= 0.0 || metric_scalar <= 0.0) {
    return 0.0;
  }
  if (acceleration_limit <= 0.0) {
    return 0.5 * metric_scalar * gain * error * error;
  }
  const double scaled_error = gain * error / acceleration_limit;
  return metric_scalar * acceleration_limit * acceleration_limit / gain *
         stable_log_cosh(scaled_error);
}

struct TangentEscapeSoftmaxCandidate
{
  std::size_t slot{0};
  Eigen::Vector3d direction{Eigen::Vector3d::UnitX()};
  double goal_score{0.0};
  double continuity_score{0.0};
  double duplicate_risk{0.0};
  double adjacent_risk{0.0};
  double hold_bonus{0.0};
  double blocked_penalty{0.0};
  double stuck_bonus{0.0};
  double base_score{0.0};
  double score{0.0};
  double weight{0.0};
  double metric_scalar{0.0};
  double metric_boost{1.0};
  double accel_boost{1.0};
  double scalar_s{0.0};
  double scalar_target{0.0};
  double scalar_velocity{0.0};
  double scalar_error{0.0};
  double desired_tangent_accel{0.0};
  double clearance_rate{0.0};
  double collision_accel{0.0};
  double scaled_collision_accel{0.0};
  double potential_energy{0.0};
  double kinetic_energy{0.0};
  double lyapunov_energy{0.0};
  double damping_vdot{0.0};
  std::uint64_t mode_generation{0};
  bool weights_latched{false};
  bool bounded_potential{false};
  double mode_normal_dot_tangent{0.0};
  Eigen::Vector3d origin{Eigen::Vector3d::Zero()};
  Eigen::Vector3d applied_tangent{Eigen::Vector3d::UnitX()};
};

void add_unique_tangent_candidate(
  std::vector<TangentEscapeSoftmaxCandidate> & candidates,
  std::size_t slot,
  const std::optional<Eigen::Vector3d> & direction,
  double duplicate_dot_threshold = 0.985)
{
  if (!direction.has_value() || !direction->allFinite()) {
    return;
  }
  for (const auto & candidate : candidates) {
    // Opposite tangents are distinct escape branches; only drop near-identical directions.
    if (candidate.direction.dot(direction.value()) >= duplicate_dot_threshold) {
      return;
    }
  }

  TangentEscapeSoftmaxCandidate candidate;
  candidate.slot = slot;
  candidate.direction = direction.value();
  candidates.push_back(candidate);
}

std::optional<Eigen::Vector3d> horizontal_unit_direction(
  const Eigen::Vector3d & direction,
  double min_norm)
{
  if (!direction.allFinite()) {
    return std::nullopt;
  }

  const Eigen::Vector3d horizontal =
    direction - direction.dot(Eigen::Vector3d::UnitZ()) * Eigen::Vector3d::UnitZ();
  const double norm = horizontal.norm();
  if (norm <= min_norm) {
    return std::nullopt;
  }
  return horizontal / norm;
}

double tangent_escape_predictive_duplicate_risk(
  const TangentEscapeRmpParams & params,
  const PinocchioDirectRmpSolver::NodeGeometry & geometry,
  std::size_t active_point_index,
  const Eigen::Vector3d & direction,
  const Eigen::Vector3d & normal,
  double active_detection)
{
  if (
    active_point_index >= RB10Model::sensor_control_points.size() ||
    geometry.x.size() < static_cast<Eigen::Index>(3 * (active_point_index + 1)) ||
    active_detection <= 0.0)
  {
    return 0.0;
  }

  const auto successors = RB10Model::predictive_duplicate_successors(active_point_index);
  const Eigen::Vector3d active_position =
    geometry.x.segment<3>(static_cast<Eigen::Index>(3 * active_point_index));
  const double min_alignment = std::clamp(params.duplicate_risk_min_alignment, 0.0, 1.0);
  double risk = 0.0;
  for (std::size_t offset = 0; offset < successors.count; ++offset) {
    const std::size_t successor_index = successors.indices[offset];
    if (
      successor_index >= RB10Model::sensor_control_points.size() ||
      geometry.x.size() < static_cast<Eigen::Index>(3 * (successor_index + 1)))
    {
      continue;
    }

    const Eigen::Vector3d successor_position =
      geometry.x.segment<3>(static_cast<Eigen::Index>(3 * successor_index));
    const auto tangent_axis = project_to_tangent_direction(
      successor_position - active_position,
      normal,
      params.min_tangent_norm);
    if (!tangent_axis.has_value()) {
      continue;
    }
    const auto horizontal_axis = horizontal_unit_direction(
      tangent_axis.value(),
      params.min_tangent_norm);
    if (!horizontal_axis.has_value()) {
      continue;
    }

    const double toward_successor = std::max(0.0, direction.dot(horizontal_axis.value()));
    if (toward_successor <= min_alignment) {
      continue;
    }
    risk = std::max(risk, active_detection * toward_successor * toward_successor);
  }
  return risk;
}

double tangent_escape_instantaneous_adjacent_risk(
  const TangentEscapeRmpParams & params,
  const PinocchioDirectRmpSolver::NodeGeometry & geometry,
  std::size_t point_count,
  std::size_t active_point_index,
  const Eigen::Vector3d & direction,
  const std::vector<ObstacleSphere> & obstacles)
{
  double risk = 0.0;
  for (const auto & obstacle : obstacles) {
    if (
      obstacle.radius <= 0.0 ||
      !obstacle.center.allFinite() ||
      obstacle.proximity_control_point_index < 0 ||
      obstacle.proximity_control_point_index == static_cast<int>(active_point_index))
    {
      continue;
    }

    const auto other_index = static_cast<std::size_t>(obstacle.proximity_control_point_index);
    if (other_index >= point_count) {
      continue;
    }

    const Eigen::Vector3d other_cp =
      geometry.x.segment<3>(static_cast<Eigen::Index>(3 * other_index));
    const double other_radius = RB10Model::sensor_control_points[other_index].radius;
    const Eigen::Vector3d delta = other_cp - obstacle.center;
    const double center_distance = delta.norm();
    if (center_distance <= 1e-9) {
      continue;
    }

    const double clearance =
      center_distance - (other_radius + obstacle.radius);
    const double distance_risk = tangent_escape_activation(clearance, params);
    const Eigen::Vector3d obstacle_direction = -delta / center_distance;
    risk += distance_risk * std::max(0.0, direction.dot(obstacle_direction));
  }
  return risk;
}

PinocchioDirectRmpSolver::JointVector resolve_root_direct(
  const PinocchioDirectRmpSolver::Matrix6 & metric,
  const PinocchioDirectRmpSolver::JointVector & force,
  double solve_offset)
{
  const double max_abs = std::max(metric.cwiseAbs().maxCoeff() * 0.01, 1.0);
  PinocchioDirectRmpSolver::Matrix6 scaled_metric = metric / max_abs;
  PinocchioDirectRmpSolver::JointVector scaled_force = force / max_abs;
  scaled_metric += solve_offset * PinocchioDirectRmpSolver::Matrix6::Identity();

  PinocchioDirectRmpSolver::JointVector qdd =
    scaled_metric.ldlt().solve(scaled_force).eval();
  if (!qdd.allFinite()) {
    qdd = scaled_metric.completeOrthogonalDecomposition().solve(scaled_force).eval();
  }
  if (!qdd.allFinite()) {
    qdd.setZero();
  }
  return qdd;
}

PinocchioDirectRmpSolver::JointVector resolve_root_rmp2(
  const PinocchioDirectRmpSolver::Matrix6 & metric,
  const PinocchioDirectRmpSolver::JointVector & force,
  double solve_offset)
{
  const double max_abs = std::max(metric.cwiseAbs().maxCoeff() * 0.01, 1.0);
  PinocchioDirectRmpSolver::Matrix6 scaled_metric = metric / max_abs;
  PinocchioDirectRmpSolver::JointVector scaled_force = force / max_abs;
  scaled_metric += solve_offset * PinocchioDirectRmpSolver::Matrix6::Identity();

  PinocchioDirectRmpSolver::JointVector qdd =
    scaled_metric.ldlt().solve(scaled_force).eval();
  if (!qdd.allFinite()) {
    qdd = scaled_metric.completeOrthogonalDecomposition().solve(scaled_force).eval();
  }
  if (!qdd.allFinite()) {
    qdd.setZero();
  }
  return qdd;
}

}  // namespace

PinocchioDirectRmpSolver::PinocchioDirectRmpSolver(
  EigenRmpConfig config,
  std::string urdf_path)
: config_(std::move(config)),
  model_(std::make_shared<PinocchioModel>(urdf_path))
{
  if (config_.collision.policy != "repulsive") {
    throw std::runtime_error("collision_policy must be repulsive");
  }
  auto impl = std::make_shared<SolverImpl>();
  impl->topo_indices = build_topological_order(config_);
  impl->node_dims.emplace("root", 6);
  for (const auto index : impl->topo_indices) {
    const auto & node = config_.graph_nodes[index];
    impl->node_dims.emplace(node.name, infer_node_dim(node, impl->node_dims));
  }

  std::vector<RmpNodeConfig> casadi_nodes;
  for (const auto index : impl->topo_indices) {
    const auto & node = config_.graph_nodes[index];
    if (uses_casadi_task_map(node.task_map_type)) {
      casadi_nodes.push_back(node);
    }
  }
  impl->compiled_task_maps = CasadiTaskGraph(casadi_nodes).compile(impl->node_dims);
  compiled_state_ = impl;
}

RmpSolveResult PinocchioDirectRmpSolver::solve(
  const JointVector & q,
  const JointVector & qd,
  const std::unordered_map<std::string, Eigen::Vector3d> & vector_targets,
  const std::vector<ObstacleSphere> & obstacles,
  const std::unordered_map<std::string, ExternalRmpFeature> & external_rmps) const
{
  const auto context = model_->forward_context(q, qd);
  const auto & impl = get_impl(compiled_state_);
  const bool use_rmp2 = uses_rmp2_solve();
  const bool use_natural_rmp = uses_natural_rmp();
  if (!use_rmp2 && use_natural_rmp) {
    throw std::runtime_error("rmp_type=natural requires solve_method=rmp2");
  }
  Matrix6 metric = Matrix6::Zero();
  JointVector force = JointVector::Zero();

  std::unordered_map<std::string, NodeGeometry> cache;
  cache.emplace(
    "root",
    NodeGeometry{
      to_eigen_vector(q),
      Matrix6::Identity(),
      to_eigen_vector(qd),
      Eigen::VectorXd::Zero(6)
    });

  for (const auto index : impl.topo_indices) {
    const auto & node = config_.graph_nodes[index];

    const auto geometry = evaluate_node(
      node,
      q,
      context,
      obstacles,
      cache);
    cache[node.name] = geometry;
  }

  JointVector nominal_qdd = JointVector::Zero();
  const JointVector * nominal_qdd_ptr = nullptr;
  if (config_.tangent_escape.enabled) {
    nominal_qdd = compute_nominal_joint_acceleration(
      qd,
      cache,
      context,
      vector_targets,
      external_rmps);
    nominal_qdd_ptr = &nominal_qdd;
  }
  std::vector<double> tangent_escape_debug_data{0.0};

  for (const auto index : impl.topo_indices) {
    const auto & node = config_.graph_nodes[index];
    const auto & geometry = cache.at(node.name);
    const auto accumulate_leaf = [&](const std::string & leaf_type) {
        accumulate_leaf_type(
          leaf_type,
          node,
          geometry,
          qd,
	          cache,
          context,
	          vector_targets,
          obstacles,
	          external_rmps,
	          nominal_qdd_ptr,
	          metric,
          force,
          &tangent_escape_debug_data);
      };
    accumulate_leaf(node.leaf_rmp_type);
    accumulate_leaf(node.handcrafted_leaf_rmp_type);
  }

  JointVector qdd = use_rmp2 ?
    resolve_root_rmp2(metric, force, config_.solve_offset) :
    resolve_root_direct(metric, force, config_.solve_offset);

  return RmpSolveResult{qdd, metric, force, tangent_escape_debug_data};
}

std::vector<RmpSolveResult> PinocchioDirectRmpSolver::solve_batch(
  const std::vector<RmpBatchInput> & batch_inputs) const
{
  std::vector<RmpSolveResult> results;
  results.reserve(batch_inputs.size());
  for (const auto & input : batch_inputs) {
    results.push_back(solve(
        input.q,
        input.qd,
        input.vector_targets,
        input.obstacles,
        input.external_rmps));
  }
  return results;
}

double PinocchioDirectRmpSolver::sigmoid(double value)
{
  return 1.0 / (1.0 + std::exp(-value));
}

double PinocchioDirectRmpSolver::collision_scalar_acceleration(
  double clearance,
  double clearance_rate,
  const CollisionRmpParams & params)
{
  const double x = std::max(clearance, 0.0);
  const double repel = params.repulsion_gain * std::exp(-(x / params.repulsion_std_dev));
  const double sigma = sigmoid(clearance_rate / params.damping_velocity_gate_length_scale);
  const double damping =
    -(1.0 - sigma) * params.damping_gain * clearance_rate /
    (x / params.damping_std_dev + params.damping_robustness_eps);
  return repel + damping;
}

Eigen::VectorXd PinocchioDirectRmpSolver::make_default_q(const EigenRmpConfig & config)
{
  Eigen::VectorXd out(6);
  for (int index = 0; index < 6; ++index) {
    out[index] = config.default_q[static_cast<std::size_t>(index)];
  }
  return out;
}

Eigen::VectorXd PinocchioDirectRmpSolver::velocity_of(
  const NodeGeometry & geometry,
  const JointVector &)
{
  return geometry.velocity;
}

int PinocchioDirectRmpSolver::control_point_count()
{
  return static_cast<int>(RB10Model::sensor_control_points.size());
}

bool PinocchioDirectRmpSolver::uses_casadi_task_map(const std::string & task_map_type)
{
  return task_map_type == "affine" ||
         task_map_type == "elem_multiply" ||
         task_map_type == "elem_divide" ||
         task_map_type == "sin" ||
         task_map_type == "cos" ||
         task_map_type == "tanh" ||
         task_map_type == "square" ||
         task_map_type == "abs" ||
         task_map_type == "sum" ||
         task_map_type == "weighted_sum" ||
         task_map_type == "difference" ||
         task_map_type == "concat" ||
         task_map_type == "slice" ||
         task_map_type == "norm" ||
         task_map_type == "normalize";
}

bool PinocchioDirectRmpSolver::uses_rmp2_solve() const
{
  if (config_.solve_method == "direct") {
    return false;
  }
  if (config_.solve_method == "rmp2") {
    return true;
  }
  throw std::runtime_error("Unsupported solve_method: " + config_.solve_method);
}

bool PinocchioDirectRmpSolver::uses_natural_rmp() const
{
  if (config_.rmp_type == "canonical") {
    return false;
  }
  if (config_.rmp_type == "natural") {
    return true;
  }
  throw std::runtime_error("Unsupported rmp_type: " + config_.rmp_type);
}

Eigen::Vector3d PinocchioDirectRmpSolver::axis_unit_vector(const std::string & axis_name)
{
  if (axis_name == "x") {
    return Eigen::Vector3d::UnitX();
  }
  if (axis_name == "y") {
    return Eigen::Vector3d::UnitY();
  }
  if (axis_name == "z") {
    return Eigen::Vector3d::UnitZ();
  }
  throw std::runtime_error("Unsupported axis name: " + axis_name);
}

Eigen::VectorXd PinocchioDirectRmpSolver::flatten_control_points(
  const KinematicsContext & context)
{
  Eigen::VectorXd out(3 * static_cast<Eigen::Index>(context.control_points.size()));
  for (std::size_t index = 0; index < context.control_points.size(); ++index) {
    out.segment<3>(static_cast<Eigen::Index>(3 * index)) = context.control_points[index].position;
  }
  return out;
}

Eigen::MatrixXd PinocchioDirectRmpSolver::stack_control_point_jacobians(
  const KinematicsContext & context)
{
  Eigen::MatrixXd out(3 * static_cast<Eigen::Index>(context.control_point_jacobians.size()), 6);
  for (std::size_t index = 0; index < context.control_point_jacobians.size(); ++index) {
    out.block<3, 6>(static_cast<Eigen::Index>(3 * index), 0) =
      context.control_point_jacobians[index];
  }
  return out;
}

int PinocchioDirectRmpSolver::infer_node_dim(
  const RmpNodeConfig & node,
  const std::unordered_map<std::string, int> & node_dims)
{
  if (node.task_map_type == "tcp_position" ||
    node.task_map_type == "link_position" ||
    node.task_map_type == "link_orientation_axis")
  {
    return 3;
  }
  if (node.task_map_type == "joint_limit") {
    return 12;
  }
  if (node.task_map_type == "control_points") {
    return 3 * control_point_count();
  }
  if (node.task_map_type == "collision_distance") {
    return control_point_count();
  }
  if (node.task_map_type == "norm") {
    return 1;
  }
  if (node.task_map_type == "affine") {
    if (!node.bias.empty()) {
      return static_cast<int>(node.bias.size());
    }
    if (!node.matrix.empty()) {
      int input_dim = 0;
      for (const auto & parent : node.parents) {
        input_dim += node_dims.at(parent);
      }
      if (input_dim <= 0 || static_cast<int>(node.matrix.size()) % input_dim != 0) {
        throw std::runtime_error("Invalid affine matrix size for node " + node.name);
      }
      return static_cast<int>(node.matrix.size()) / input_dim;
    }
    return node_dims.at(node.parents.front());
  }
  if (node.task_map_type == "concat") {
    int dim = 0;
    for (const auto & parent : node.parents) {
      dim += node_dims.at(parent);
    }
    return dim;
  }
  if (
    node.task_map_type == "elem_multiply" ||
    node.task_map_type == "elem_divide" ||
    node.task_map_type == "sin" ||
    node.task_map_type == "cos" ||
    node.task_map_type == "tanh" ||
    node.task_map_type == "square" ||
    node.task_map_type == "abs")
  {
    return node_dims.at(node.parents.front());
  }
  if (node.task_map_type == "slice") {
    return node.slice_length > 0 ? node.slice_length :
           node_dims.at(node.parents.front()) - node.slice_start;
  }
  return node_dims.at(node.parents.front());
}

bool PinocchioDirectRmpSolver::node_enabled(const std::string & name) const
{
  for (const auto & node : config_.graph_nodes) {
    if (node.name == name) {
      return node.enabled;
    }
  }
  return false;
}

PinocchioDirectRmpSolver::NodeGeometry PinocchioDirectRmpSolver::evaluate_node(
  const RmpNodeConfig & node,
  const JointVector & q,
  const KinematicsContext & context,
  const std::vector<ObstacleSphere> & obstacles,
  std::unordered_map<std::string, NodeGeometry> & cache) const
{
  std::vector<NodeGeometry> parents;
  parents.reserve(node.parents.size());
  for (const auto & parent_name : node.parents) {
    const auto it = cache.find(parent_name);
    if (it == cache.end()) {
      throw std::runtime_error("Missing parent node state for " + node.name + ": " + parent_name);
    }
    parents.push_back(it->second);
  }

  if (uses_casadi_task_map(node.task_map_type)) {
    return evaluate_casadi_node(node, parents);
  }
  return evaluate_native_node(node, q, context, obstacles, parents);
}

PinocchioDirectRmpSolver::NodeGeometry PinocchioDirectRmpSolver::evaluate_native_node(
  const RmpNodeConfig & node,
  const JointVector &,
  const KinematicsContext & context,
  const std::vector<ObstacleSphere> & obstacles,
  const std::vector<NodeGeometry> & parents) const
{
  if (node.task_map_type == "cspace_target" || node.task_map_type == "identity") {
    return parents.front();
  }

  if (node.task_map_type == "joint_limit") {
    NodeGeometry out;
    out.x.resize(12);
    out.jacobian = Eigen::MatrixXd::Zero(12, 6);
    out.velocity.resize(12);
    out.curvature = Eigen::VectorXd::Zero(12);
    for (int joint = 0; joint < 6; ++joint) {
      const double lower =
        config_.joint_lower_limits[static_cast<std::size_t>(joint)] +
        config_.joint_limit_buffers[static_cast<std::size_t>(joint)];
      const double upper =
        config_.joint_upper_limits[static_cast<std::size_t>(joint)] -
        config_.joint_limit_buffers[static_cast<std::size_t>(joint)];
      out.x[2 * joint] = upper - parents.front().x[joint];
      out.x[2 * joint + 1] = parents.front().x[joint] - lower;
      out.velocity[2 * joint] = -parents.front().velocity[joint];
      out.velocity[2 * joint + 1] = parents.front().velocity[joint];
      out.jacobian(2 * joint, joint) = -1.0;
      out.jacobian(2 * joint + 1, joint) = 1.0;
    }
    return out;
  }

  if (node.task_map_type == "tcp_position") {
    return NodeGeometry{
      context.tcp_position,
      context.tcp_jacobian,
      context.tcp_velocity,
      context.tcp_curvature
    };
  }

  if (node.task_map_type == "link_position") {
    const auto link_index = link_index_from_name(node.link_name);
    return NodeGeometry{
      context.link_positions[static_cast<std::size_t>(link_index)],
      context.link_jacobians[static_cast<std::size_t>(link_index)],
      context.link_velocities[static_cast<std::size_t>(link_index)],
      context.link_curvatures[static_cast<std::size_t>(link_index)]
    };
  }

  if (node.task_map_type == "link_orientation_axis") {
    const auto link_index = link_index_from_name(node.link_name);
    const Eigen::Vector3d axis_world =
      context.link_rotations[static_cast<std::size_t>(link_index)] * axis_unit_vector(node.axis);
    const Eigen::Vector3d angular_velocity =
      context.link_angular_velocities[static_cast<std::size_t>(link_index)];
    const Eigen::Vector3d angular_curvature =
      context.link_angular_curvatures[static_cast<std::size_t>(link_index)];
    const Eigen::Matrix<double, 3, 6> & angular_jacobian =
      context.link_angular_jacobians[static_cast<std::size_t>(link_index)];
    Eigen::Matrix<double, 3, 6> axis_jacobian;
    for (int column = 0; column < 6; ++column) {
      axis_jacobian.col(column) = angular_jacobian.col(column).cross(axis_world);
    }
    const Eigen::Vector3d axis_velocity = angular_velocity.cross(axis_world);
    const Eigen::Vector3d axis_curvature =
      angular_curvature.cross(axis_world) + angular_velocity.cross(axis_velocity);
    return NodeGeometry{axis_world, axis_jacobian, axis_velocity, axis_curvature};
  }

  if (node.task_map_type == "control_points") {
    return NodeGeometry{
      flatten_control_points(context),
      stack_control_point_jacobians(context),
      [&context]() {
        Eigen::VectorXd out(3 * static_cast<Eigen::Index>(context.control_point_velocities.size()));
        for (std::size_t index = 0; index < context.control_point_velocities.size(); ++index) {
          out.segment<3>(static_cast<Eigen::Index>(3 * index)) =
            context.control_point_velocities[index];
        }
        return out;
      }(),
      [&context]() {
        Eigen::VectorXd out(3 * static_cast<Eigen::Index>(context.control_point_curvatures.size()));
        for (std::size_t index = 0; index < context.control_point_curvatures.size(); ++index) {
          out.segment<3>(static_cast<Eigen::Index>(3 * index)) =
            context.control_point_curvatures[index];
        }
        return out;
      }()
    };
  }

  if (node.task_map_type == "collision_distance") {
    const auto & parent = parents.front();
    const int num_control_points = static_cast<int>(parent.x.size() / 3);
    if (num_control_points != control_point_count()) {
      throw std::runtime_error("collision_distance expects full RB10 control point vector");
    }

    NodeGeometry out;
    const int total_obstacles =
      static_cast<int>(obstacles.size() + config_.body_obstacles.size());
    if (total_obstacles == 0) {
      out.x = Eigen::VectorXd::Zero(0);
      out.jacobian = Eigen::MatrixXd::Zero(0, 6);
      out.velocity = Eigen::VectorXd::Zero(0);
      out.curvature = Eigen::VectorXd::Zero(0);
      return out;
    }

    std::vector<double> xs;
    std::vector<double> velocities;
    std::vector<double> curvatures;
    std::vector<Eigen::RowVectorXd> jacobians;
    const auto & collision_params = config_.collision;
    xs.reserve(static_cast<std::size_t>(
      num_control_points * (static_cast<int>(obstacles.size()) + (config_.body_obstacles.empty() ? 0 : 1))));
    velocities.reserve(xs.capacity());
    curvatures.reserve(xs.capacity());
    jacobians.reserve(xs.capacity());

    const auto append_term =
      [&xs, &velocities, &curvatures, &jacobians](
        double x,
        const Eigen::RowVectorXd & jacobian,
        double velocity,
        double curvature)
      {
        xs.push_back(x);
        jacobians.push_back(jacobian);
        velocities.push_back(velocity);
        curvatures.push_back(curvature);
      };

    for (int point_index = 0; point_index < num_control_points; ++point_index) {
      const double point_radius =
        RB10Model::sensor_control_points[static_cast<std::size_t>(point_index)].radius;
      const auto position = parent.x.segment<3>(3 * point_index);
      const auto point_jacobian = parent.jacobian.block(3 * point_index, 0, 3, 6);
      const Eigen::Vector3d point_velocity = parent.velocity.segment<3>(3 * point_index);
      const Eigen::Vector3d point_curvature = parent.curvature.segment<3>(3 * point_index);
      for (const auto & obstacle : obstacles) {
        const Eigen::Vector3d delta = position - obstacle.center;
        const double center_distance = std::max(delta.norm(), 1e-9);
        const double signed_distance =
          center_distance - (point_radius + obstacle.radius) - collision_params.margin;
        const double x = std::max(signed_distance, 0.0);
        const Eigen::RowVectorXd jacobian =
          (delta / center_distance).transpose() * point_jacobian;
        const double velocity = (delta / center_distance).dot(point_velocity);
        double curvature = 0.0;
        if (signed_distance > 0.0) {
          const Eigen::Matrix3d projector =
            (Eigen::Matrix3d::Identity() -
            (delta / center_distance) * (delta / center_distance).transpose()) /
            center_distance;
          curvature =
            (delta / center_distance).dot(point_curvature) +
            point_velocity.transpose() * projector * point_velocity;
        }
        append_term(
          x,
          jacobian,
          signed_distance > 0.0 ? velocity : 0.0,
          curvature);
      }

      bool has_best_body_term = false;
      double best_signed_distance = 0.0;
      double best_x = 0.0;
      Eigen::RowVectorXd best_jacobian = Eigen::RowVectorXd::Zero(6);
      double best_velocity = 0.0;
      double best_curvature = 0.0;

      for (const auto & obstacle : config_.body_obstacles) {
        if (!body_obstacle_interacts_with_sensor_control_point(point_index, obstacle)) {
          continue;
        }
        if (obstacle.type == "ball") {
          Eigen::Vector3d obstacle_center = obstacle.center;
          if (!obstacle.link_name.empty()) {
            const auto link_index =
              link_index_from_name(obstacle.link_name);
            obstacle_center =
              context.link_positions[static_cast<std::size_t>(link_index)] +
              context.link_rotations[static_cast<std::size_t>(link_index)] * obstacle.center;
          }
          const Eigen::Vector3d delta = position - obstacle_center;
          const double center_distance = std::max(delta.norm(), 1e-9);
          const double signed_distance =
            center_distance - (point_radius + obstacle.radius) - collision_params.margin;
          const double x = std::max(signed_distance, 0.0);
          const Eigen::RowVectorXd jacobian =
            (delta / center_distance).transpose() * point_jacobian;
          const double velocity = (delta / center_distance).dot(point_velocity);
          double curvature = 0.0;
          if (signed_distance > 0.0) {
            const Eigen::Matrix3d projector =
              (Eigen::Matrix3d::Identity() -
              (delta / center_distance) * (delta / center_distance).transpose()) /
              center_distance;
            curvature =
              (delta / center_distance).dot(point_curvature) +
              point_velocity.transpose() * projector * point_velocity;
          }
          if (!has_best_body_term || signed_distance < best_signed_distance) {
            has_best_body_term = true;
            best_signed_distance = signed_distance;
            best_x = x;
            best_jacobian = jacobian;
            best_velocity = signed_distance > 0.0 ? velocity : 0.0;
            best_curvature = curvature;
          }
        } else if (obstacle.type == "box") {
          Eigen::Vector3d local_position = position;
          Eigen::Matrix3d obstacle_rotation = Eigen::Matrix3d::Identity();
          Eigen::Vector3d obstacle_origin = Eigen::Vector3d::Zero();
          if (!obstacle.link_name.empty()) {
            const auto link_index =
              link_index_from_name(obstacle.link_name);
            obstacle_rotation =
              context.link_rotations[static_cast<std::size_t>(link_index)];
            obstacle_origin =
              context.link_positions[static_cast<std::size_t>(link_index)];
            local_position =
              obstacle_rotation.transpose() * (position - obstacle_origin);
          }
          const Eigen::Vector3d clamped =
            local_position.cwiseMax(obstacle.mins).cwiseMin(obstacle.maxs);
          const Eigen::Vector3d delta = local_position - clamped;
          const double outside_distance = delta.norm();
          const double signed_distance =
            outside_distance - point_radius - collision_params.margin;
          const double x = std::max(signed_distance, 0.0);
          Eigen::Vector3d grad_local = Eigen::Vector3d::Zero();
          if (outside_distance > 1e-9) {
            grad_local = delta / outside_distance;
          } else {
            const Eigen::Vector3d dist_to_min = local_position - obstacle.mins;
            const Eigen::Vector3d dist_to_max = obstacle.maxs - local_position;
            Eigen::Index axis = 0;
            double min_face = dist_to_min[0];
            for (Eigen::Index idx = 0; idx < 3; ++idx) {
              if (dist_to_min[idx] < min_face) {
                min_face = dist_to_min[idx];
                axis = idx;
              }
              if (dist_to_max[idx] < min_face) {
                min_face = dist_to_max[idx];
                axis = idx;
              }
            }
            grad_local[axis] = (dist_to_min[axis] < dist_to_max[axis]) ? -1.0 : 1.0;
          }
          const Eigen::Vector3d grad = obstacle_rotation * grad_local;
          const Eigen::RowVectorXd jacobian = grad.transpose() * point_jacobian;
          const double velocity = grad.dot(point_velocity);
          const double curvature = grad.dot(point_curvature);
          if (!has_best_body_term || signed_distance < best_signed_distance) {
            has_best_body_term = true;
            best_signed_distance = signed_distance;
            best_x = x;
            best_jacobian = jacobian;
            best_velocity = signed_distance > 0.0 ? velocity : 0.0;
            best_curvature = signed_distance > 0.0 ? curvature : 0.0;
          }
        } else {
          throw std::runtime_error("Unsupported body obstacle type: " + obstacle.type);
        }
      }

      if (has_best_body_term) {
        append_term(best_x, best_jacobian, best_velocity, best_curvature);
      }
    }

    out.x.resize(static_cast<Eigen::Index>(xs.size()));
    out.jacobian.resize(static_cast<Eigen::Index>(xs.size()), 6);
    out.velocity.resize(static_cast<Eigen::Index>(xs.size()));
    out.curvature.resize(static_cast<Eigen::Index>(xs.size()));
    for (std::size_t index = 0; index < xs.size(); ++index) {
      out.x[static_cast<Eigen::Index>(index)] = xs[index];
      out.jacobian.row(static_cast<Eigen::Index>(index)) = jacobians[index];
      out.velocity[static_cast<Eigen::Index>(index)] = velocities[index];
      out.curvature[static_cast<Eigen::Index>(index)] = curvatures[index];
    }
    return out;
  }

  throw std::runtime_error("Unsupported native task map for pinocchio_direct: " + node.task_map_type);
}

PinocchioDirectRmpSolver::NodeGeometry PinocchioDirectRmpSolver::evaluate_casadi_node(
  const RmpNodeConfig & node,
  const std::vector<NodeGeometry> & parents) const
{
  const auto & impl = get_impl(compiled_state_);
  const auto task_it = impl.compiled_task_maps.find(node.name);
  if (task_it == impl.compiled_task_maps.end()) {
    throw std::runtime_error("Missing compiled CasADi task map for node: " + node.name);
  }

  std::vector<casadi::DM> args;
  args.reserve(2 * parents.size());
  for (const auto & parent : parents) {
    args.push_back(to_dm(parent.x));
    args.push_back(to_dm(parent.velocity));
  }
  const auto outputs = task_it->second.function(args);

  NodeGeometry out;
  out.x = dm_to_vector(outputs.front());
  out.velocity = dm_to_vector(outputs[1]);
  out.curvature = dm_to_vector(outputs[2]);
  if (out.velocity.size() != out.x.size()) {
    throw std::runtime_error(
            "CasADi task map velocity dimension mismatch for node " + node.name);
  }
  if (out.curvature.size() != out.x.size()) {
    throw std::runtime_error(
            "CasADi task map curvature dimension mismatch for node " + node.name);
  }
  out.jacobian = Eigen::MatrixXd::Zero(out.x.size(), 6);
  for (std::size_t index = 0; index < parents.size(); ++index) {
    const auto local_jacobian = dm_to_matrix(outputs[index + 3]);
    const auto chained_jacobian = local_jacobian * parents[index].jacobian;
    const auto chained_curvature = local_jacobian * parents[index].curvature;
    if (chained_curvature.size() != out.curvature.size()) {
      throw std::runtime_error(
              "CasADi chained curvature dimension mismatch for node " + node.name);
    }
    out.jacobian += chained_jacobian;
    out.curvature += chained_curvature;
  }
  return out;
}

void PinocchioDirectRmpSolver::accumulate_scalar_leaf(
  bool use_natural_rmp,
  const RowVector6 & jacobian,
  double metric_scalar,
  double acceleration,
  double curvature,
  Matrix6 & metric,
  JointVector & force)
{
  metric += jacobian.transpose() * metric_scalar * jacobian;
  if (use_natural_rmp) {
    const double natural_force = metric_scalar * acceleration;
    force += jacobian.transpose() * (natural_force - metric_scalar * curvature);
  } else {
    force += jacobian.transpose() * (metric_scalar * (acceleration - curvature));
  }
}

void PinocchioDirectRmpSolver::accumulate_vector_leaf(
  bool use_natural_rmp,
  const Eigen::MatrixXd & jacobian,
  const Eigen::MatrixXd & leaf_metric,
  const Eigen::VectorXd & acceleration,
  const Eigen::VectorXd & curvature,
  Matrix6 & metric,
  JointVector & force)
{
  metric += jacobian.transpose() * leaf_metric * jacobian;
  if (use_natural_rmp) {
    const Eigen::VectorXd natural_force = leaf_metric * acceleration;
    force += jacobian.transpose() * (natural_force - leaf_metric * curvature);
  } else {
    force += jacobian.transpose() * leaf_metric * (acceleration - curvature);
  }
}

void PinocchioDirectRmpSolver::accumulate_external(
  const RmpNodeConfig & node,
  const NodeGeometry & geometry,
  const std::unordered_map<std::string, ExternalRmpFeature> & external_rmps,
  Matrix6 & metric,
  JointVector & force) const
{
  const auto feature_it = external_rmps.find(node.target_key);
  if (feature_it == external_rmps.end()) {
    return;
  }
  const auto & feature = feature_it->second;
  if (feature.acceleration.size() != geometry.x.size()) {
    throw std::runtime_error("External RMP acceleration dimension mismatch for node " + node.name);
  }
  if (
    feature.metric_sqrt.rows() != geometry.x.size() ||
    feature.metric_sqrt.cols() != geometry.x.size())
  {
    throw std::runtime_error("External RMP metric_sqrt dimension mismatch for node " + node.name);
  }
  Eigen::MatrixXd metric_sqrt = feature.metric_sqrt;
  if (node.identity_multiplier != 0.0) {
    metric_sqrt += node.identity_multiplier *
      Eigen::MatrixXd::Identity(metric_sqrt.rows(), metric_sqrt.cols());
  }
  const Eigen::MatrixXd leaf_metric = metric_sqrt * metric_sqrt.transpose();
  accumulate_vector_leaf(
    uses_rmp2_solve() && uses_natural_rmp(),
    geometry.jacobian,
    leaf_metric,
    feature.acceleration,
    geometry.curvature,
    metric,
    force);
}

void PinocchioDirectRmpSolver::accumulate_leaf_type(
  const std::string & leaf_type,
  const RmpNodeConfig & node,
  const NodeGeometry & geometry,
  const JointVector & qd,
  const std::unordered_map<std::string, NodeGeometry> & cache,
  const KinematicsContext & context,
  const std::unordered_map<std::string, Eigen::Vector3d> & vector_targets,
  const std::vector<ObstacleSphere> & obstacles,
  const std::unordered_map<std::string, ExternalRmpFeature> & external_rmps,
  const JointVector * nominal_qdd,
  Matrix6 & metric,
  JointVector & force,
  std::vector<double> * tangent_escape_debug_data) const
{
  if (leaf_type == "none" || leaf_type.empty()) {
    return;
  }

  if (leaf_type == "cspace_target") {
    accumulate_cspace_target(geometry, qd, metric, force);
    return;
  }
  if (leaf_type == "joint_limit") {
    accumulate_joint_limits(geometry, qd, metric, force);
    return;
  }
  if (leaf_type == "joint_velocity_cap") {
    accumulate_joint_velocity_cap(geometry, qd, metric, force);
    return;
  }
  if (leaf_type == "target") {
    const auto target_it = vector_targets.find(node.target_key);
    if (target_it != vector_targets.end()) {
      accumulate_target(geometry, qd, target_it->second, metric, force);
    }
    return;
  }
  if (leaf_type == "axis_target") {
    const auto target_it = vector_targets.find(node.target_key);
    if (target_it != vector_targets.end()) {
      Eigen::Vector3d current_position_value = Eigen::Vector3d::Zero();
      const Eigen::Vector3d * current_position = nullptr;
      const Eigen::Vector3d * position_goal = nullptr;
      if (node_enabled("target")) {
        const auto position_goal_it = vector_targets.find("goal");
        if (position_goal_it != vector_targets.end()) {
          position_goal = &position_goal_it->second;
        }
        const auto tcp_position_it = cache.find("tcp_position");
        if (
          tcp_position_it != cache.end() &&
          tcp_position_it->second.x.size() == 3)
        {
          current_position_value = tcp_position_it->second.x.head<3>();
          current_position = &current_position_value;
        }
      }
      accumulate_axis_target(
        geometry,
        qd,
        target_it->second,
        current_position,
        position_goal,
        metric,
        force);
    }
    return;
  }
  if (leaf_type == "wrist_axis_target") {
    const auto target_it = vector_targets.find(node.target_key);
    if (target_it != vector_targets.end()) {
      Eigen::Vector3d current_position_value = Eigen::Vector3d::Zero();
      const Eigen::Vector3d * current_position = nullptr;
      const Eigen::Vector3d * position_goal = nullptr;
      if (node_enabled("target")) {
        const auto position_goal_it = vector_targets.find("goal");
        if (position_goal_it != vector_targets.end()) {
          position_goal = &position_goal_it->second;
        }
        const auto tcp_position_it = cache.find("tcp_position");
        if (
          tcp_position_it != cache.end() &&
          tcp_position_it->second.x.size() == 3)
        {
          current_position_value = tcp_position_it->second.x.head<3>();
          current_position = &current_position_value;
        }
      }
      accumulate_wrist_axis_target(
        geometry,
        qd,
        target_it->second,
        current_position,
        position_goal,
        metric,
        force);
    }
    return;
  }
  if (leaf_type == "collision") {
    accumulate_collision(geometry, qd, metric, force);
    return;
  }
  if (leaf_type == "tangent_escape") {
    const auto target_it = vector_targets.find(node.target_key);
    if (target_it != vector_targets.end() && nominal_qdd != nullptr) {
      accumulate_tangent_escape(
        context,
        geometry,
        qd,
        target_it->second,
        obstacles,
        *nominal_qdd,
        metric,
        force,
        tangent_escape_debug_data);
    }
    return;
  }
  if (leaf_type == "damping") {
    accumulate_joint_damping(geometry, qd, metric, force);
    return;
  }
  if (leaf_type == "external") {
    accumulate_external(node, geometry, external_rmps, metric, force);
    return;
  }
  throw std::runtime_error("Unsupported pinocchio_direct leaf RMP: " + leaf_type);
}

PinocchioDirectRmpSolver::JointVector
PinocchioDirectRmpSolver::compute_nominal_joint_acceleration(
  const JointVector & qd,
  const std::unordered_map<std::string, NodeGeometry> & cache,
  const KinematicsContext & context,
  const std::unordered_map<std::string, Eigen::Vector3d> & vector_targets,
  const std::unordered_map<std::string, ExternalRmpFeature> & external_rmps) const
{
  Matrix6 nominal_metric = Matrix6::Zero();
  JointVector nominal_force = JointVector::Zero();
  const std::vector<ObstacleSphere> no_obstacles;
  const auto & impl = get_impl(compiled_state_);

  for (const auto index : impl.topo_indices) {
    const auto & node = config_.graph_nodes[index];
    const auto & geometry = cache.at(node.name);
    const auto accumulate_nominal_leaf = [&](const std::string & leaf_type) {
        if (!is_nominal_motion_leaf_type(leaf_type)) {
          return;
        }
        accumulate_leaf_type(
          leaf_type,
          node,
          geometry,
          qd,
          cache,
          context,
          vector_targets,
          no_obstacles,
          external_rmps,
          nullptr,
          nominal_metric,
          nominal_force,
          nullptr);
      };
    accumulate_nominal_leaf(node.leaf_rmp_type);
    accumulate_nominal_leaf(node.handcrafted_leaf_rmp_type);
  }

  return uses_rmp2_solve() ?
         resolve_root_rmp2(nominal_metric, nominal_force, config_.solve_offset) :
         resolve_root_direct(nominal_metric, nominal_force, config_.solve_offset);
}

void PinocchioDirectRmpSolver::accumulate_cspace_target(
  const NodeGeometry & geometry,
  const JointVector & qd,
  Matrix6 & metric,
  JointVector & force) const
{
  const bool use_natural_rmp = uses_rmp2_solve() && uses_natural_rmp();
  const auto velocity = velocity_of(geometry, qd);
  Eigen::VectorXd delta = geometry.x - make_default_q(config_);

  const double norm = delta.norm();
  Eigen::VectorXd position_term = -config_.cspace_target.position_gain * delta;
  if (norm > config_.cspace_target.robust_position_term_thresh && norm > 1e-9) {
    position_term =
      -config_.cspace_target.robust_position_term_thresh *
      config_.cspace_target.position_gain *
      (delta / norm);
  }

  const Eigen::VectorXd acceleration = position_term - config_.cspace_target.damping_gain * velocity;
  const Eigen::MatrixXd leaf_metric =
    (config_.cspace_target.metric_scalar + config_.cspace_target.inertia) *
    Eigen::MatrixXd::Identity(geometry.x.size(), geometry.x.size());
  accumulate_vector_leaf(
    use_natural_rmp,
    geometry.jacobian,
    leaf_metric,
    acceleration,
    geometry.curvature,
    metric,
    force);
}

void PinocchioDirectRmpSolver::accumulate_joint_limits(
  const NodeGeometry & geometry,
  const JointVector & qd,
  Matrix6 & metric,
  JointVector & force) const
{
  const bool use_natural_rmp = uses_rmp2_solve() && uses_natural_rmp();
  const auto velocity = velocity_of(geometry, qd);
  for (Eigen::Index row = 0; row < geometry.x.size(); ++row) {
    const double x = std::max(geometry.x[row], 0.0);
    const double xd = velocity[row];
    const double metric_before_gate =
      config_.joint_limit.metric_scalar /
      (x / config_.joint_limit.metric_length_scale + config_.joint_limit.metric_exploder_eps);
    const double metric_scalar =
      (1.0 - sigmoid(xd / config_.joint_limit.metric_velocity_gate_length_scale)) *
      metric_before_gate;
    const double scaled_x =
      x / config_.joint_limit.accel_potential_exploder_length_scale;
    const double acceleration =
      config_.joint_limit.accel_potential_gain /
      (scaled_x * scaled_x + config_.joint_limit.accel_potential_exploder_eps) -
      config_.joint_limit.accel_damper_gain * xd;
    accumulate_scalar_leaf(
      use_natural_rmp,
      geometry.jacobian.row(row),
      metric_scalar,
      acceleration,
      geometry.curvature[row],
      metric,
      force);
  }
}

void PinocchioDirectRmpSolver::accumulate_joint_velocity_cap(
  const NodeGeometry & geometry,
  const JointVector & qd,
  Matrix6 & metric,
  JointVector & force) const
{
  const bool use_natural_rmp = uses_rmp2_solve() && uses_natural_rmp();
  const auto velocity = velocity_of(geometry, qd);
  for (Eigen::Index joint = 0; joint < velocity.size(); ++joint) {
    const double delta_velocity =
      std::abs(velocity[joint]) - config_.joint_velocity_cap.max_velocity +
      config_.joint_velocity_cap.velocity_damping_region;
    if (std::abs(velocity[joint]) < (
        config_.joint_velocity_cap.max_velocity -
        config_.joint_velocity_cap.velocity_damping_region))
    {
      continue;
    }

    const double xdd =
      -std::abs(config_.joint_velocity_cap.damping_gain * delta_velocity) *
      ((velocity[joint] >= 0.0) ? 1.0 : -1.0);
    const double clipped_relative_velocity = std::min(
      delta_velocity,
      config_.joint_velocity_cap.velocity_damping_region - config_.joint_velocity_cap.eps);
    const double velocity_ratio =
      clipped_relative_velocity / config_.joint_velocity_cap.velocity_damping_region;
    const double metric_scalar =
      config_.joint_velocity_cap.metric_weight /
      (1.0 - velocity_ratio * velocity_ratio);

    accumulate_scalar_leaf(
      use_natural_rmp,
      geometry.jacobian.row(joint),
      metric_scalar,
      xdd,
      geometry.curvature[joint],
      metric,
      force);
  }
}

void PinocchioDirectRmpSolver::accumulate_target(
  const NodeGeometry & geometry,
  const JointVector & qd,
  const Eigen::Vector3d & goal,
  Matrix6 & metric,
  JointVector & force) const
{
  const bool use_natural_rmp = uses_rmp2_solve() && uses_natural_rmp();
  if (geometry.x.size() != 3) {
    throw std::runtime_error("target leaf expects 3D task map output");
  }

  const auto velocity = velocity_of(geometry, qd);
  const Eigen::Vector3d x = geometry.x;
  const Eigen::Vector3d xd = velocity;
  const Eigen::Vector3d delta = goal - x;
  const double delta_norm = delta.norm();
  const double soft_delta_norm =
    std::max(delta_norm, config_.target.accel_norm_eps / 10.0);
  const Eigen::Vector3d delta_hat = delta / soft_delta_norm;

  const Eigen::Vector3d acceleration =
    config_.target.accel_p_gain * delta / (delta_norm + config_.target.accel_norm_eps) -
    config_.target.accel_d_gain * xd;

  const Eigen::Matrix3d eye = Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d shape = delta_hat * delta_hat.transpose();
  const double scaled_dist = delta_norm / config_.target.metric_alpha_length_scale;
  const double alpha =
    (1.0 - config_.target.min_metric_alpha) * std::exp(-0.5 * scaled_dist * scaled_dist) +
    config_.target.min_metric_alpha;
  Eigen::Matrix3d leaf_metric =
    alpha * config_.target.max_metric_scalar * eye +
    (1.0 - alpha) * config_.target.min_metric_scalar * shape;

  const double boost_scaled_dist =
    delta_norm / config_.target.proximity_metric_boost_length_scale;
  const double boost_alpha = std::exp(-0.5 * boost_scaled_dist * boost_scaled_dist);
  const double metric_boost_scalar =
    boost_alpha * config_.target.proximity_metric_boost_scalar + (1.0 - boost_alpha);
  leaf_metric *= metric_boost_scalar;

  accumulate_vector_leaf(
    use_natural_rmp,
    geometry.jacobian,
    leaf_metric,
    acceleration,
    geometry.curvature,
    metric,
    force);
}

void PinocchioDirectRmpSolver::accumulate_axis_target(
  const NodeGeometry & geometry,
  const JointVector & qd,
  const Eigen::Vector3d & goal,
  const Eigen::Vector3d * current_position,
  const Eigen::Vector3d * position_goal,
  Matrix6 & metric,
  JointVector & force) const
{
  const bool use_natural_rmp = uses_rmp2_solve() && uses_natural_rmp();
  if (geometry.x.size() != 3) {
    throw std::runtime_error("axis_target leaf expects 3D task map output");
  }

  const auto velocity = velocity_of(geometry, qd);
  const Eigen::Vector3d x = geometry.x.normalized();
  const Eigen::Vector3d xd = velocity;
  const Eigen::Vector3d goal_unit = goal.normalized();
  const Eigen::Vector3d delta = goal_unit - x;
  const Eigen::Vector3d acceleration =
    config_.axis_target.accel_p_gain * delta -
    config_.axis_target.accel_d_gain * xd;

  Eigen::Matrix3d leaf_metric =
    config_.axis_target.metric_scalar * Eigen::Matrix3d::Identity();
  if (current_position != nullptr && position_goal != nullptr) {
    const double delta_norm = (*position_goal - *current_position).norm();
    const double boost_length_scale =
      std::max(config_.axis_target.proximity_metric_boost_length_scale, 1e-9);
    const double boost_scaled_dist =
      delta_norm / boost_length_scale;
    const double boost_alpha = std::exp(-0.5 * boost_scaled_dist * boost_scaled_dist);
    const double metric_boost_scalar =
      boost_alpha * config_.axis_target.proximity_metric_boost_scalar + (1.0 - boost_alpha);
    leaf_metric *= metric_boost_scalar;
  }

  accumulate_vector_leaf(
    use_natural_rmp,
    geometry.jacobian,
    leaf_metric,
    acceleration,
    geometry.curvature,
    metric,
    force);
}

void PinocchioDirectRmpSolver::accumulate_wrist_axis_target(
  const NodeGeometry & geometry,
  const JointVector & qd,
  const Eigen::Vector3d & goal,
  const Eigen::Vector3d * current_position,
  const Eigen::Vector3d * position_goal,
  Matrix6 & metric,
  JointVector & force) const
{
  NodeGeometry wrist_geometry = geometry;
  if (wrist_geometry.jacobian.cols() != 6) {
    throw std::runtime_error("wrist_axis_target leaf expects a 6-column Jacobian");
  }
  // wrist_geometry.jacobian.leftCols(3).setZero();
  wrist_geometry.velocity = wrist_geometry.jacobian * qd;
  wrist_geometry.curvature = Eigen::VectorXd::Zero(wrist_geometry.x.size());

  const bool use_natural_rmp = uses_rmp2_solve() && uses_natural_rmp();
  if (wrist_geometry.x.size() != 3) {
    throw std::runtime_error("wrist_axis_target leaf expects 3D task map output");
  }

  const Eigen::Vector3d x = wrist_geometry.x.normalized();
  const Eigen::Vector3d xd = wrist_geometry.velocity;
  const Eigen::Vector3d goal_unit = goal.normalized();
  const Eigen::Vector3d delta = goal_unit - x;
  const Eigen::Vector3d acceleration =
    config_.wrist_axis_target.accel_p_gain * delta -
    config_.wrist_axis_target.accel_d_gain * xd;

  Eigen::Matrix3d leaf_metric =
    config_.wrist_axis_target.metric_scalar * Eigen::Matrix3d::Identity();
  if (current_position != nullptr && position_goal != nullptr) {
    const double delta_norm = (*position_goal - *current_position).norm();
    const double boost_length_scale =
      std::max(config_.wrist_axis_target.proximity_metric_boost_length_scale, 1e-9);
    const double boost_scaled_dist =
      delta_norm / boost_length_scale;
    const double boost_alpha = std::exp(-0.5 * boost_scaled_dist * boost_scaled_dist);
    const double metric_boost_scalar =
      boost_alpha * config_.wrist_axis_target.proximity_metric_boost_scalar +
      (1.0 - boost_alpha);
    leaf_metric *= metric_boost_scalar;
  }

  accumulate_vector_leaf(
    use_natural_rmp,
    wrist_geometry.jacobian,
    leaf_metric,
    acceleration,
    wrist_geometry.curvature,
    metric,
    force);
}

void PinocchioDirectRmpSolver::accumulate_collision(
  const NodeGeometry & geometry,
  const JointVector & qd,
  Matrix6 & metric,
  JointVector & force) const
{
  const bool use_natural_rmp = uses_rmp2_solve() && uses_natural_rmp();
  const auto velocity = velocity_of(geometry, qd);
  for (Eigen::Index row = 0; row < geometry.x.size(); ++row) {
    const double x = std::max(geometry.x[row], 0.0);
    const double xd = velocity[row];

    double metric_scalar =
      config_.collision.metric_scalar /
      (x / config_.collision.metric_exploder_std_dev + config_.collision.metric_exploder_eps);
    const double radius = config_.collision.metric_modulation_radius;
    double gate = x * x / (radius * radius) - 2.0 * x / radius + 1.0;
    if (x > radius) {
      gate = 0.0;
    }
    metric_scalar *= gate;

    const double sigma = sigmoid(xd / config_.collision.damping_velocity_gate_length_scale);
    const double collision_acceleration =
      collision_scalar_acceleration(x, xd, config_.collision);
    if (x > radius) {
      metric_scalar = 0.0;
    } else {
      metric_scalar *= (1.0 - sigma);
    }

    accumulate_scalar_leaf(
      use_natural_rmp,
      geometry.jacobian.row(row),
      metric_scalar,
      collision_acceleration,
      geometry.curvature[row],
      metric,
      force);
  }
}

void PinocchioDirectRmpSolver::accumulate_tangent_escape(
  const KinematicsContext & context,
  const NodeGeometry & geometry,
  const JointVector & qd,
  const Eigen::Vector3d & goal,
  const std::vector<ObstacleSphere> & obstacles,
  const JointVector & nominal_qdd,
  Matrix6 & metric,
  JointVector & force,
  std::vector<double> * debug_data) const
{
  (void)goal;
  const auto & params = config_.tangent_escape;
  const auto reset_gds_modes = [this]() {
      for (auto & mode : tangent_escape_gds_modes_) {
        mode.active = false;
        mode.activation = 0.0;
      }
    };
  const auto reset_softmax_gds_modes = [this]() {
      for (auto & sensor_modes : tangent_escape_softmax_gds_modes_) {
        for (auto & mode : sensor_modes) {
          mode.active = false;
          mode.activation = 0.0;
        }
      }
    };

  double best_debug_score = -std::numeric_limits<double>::infinity();
  std::vector<double> best_debug_data{0.0};
  const double tangent_metric_scalar = std::max(params.metric_scalar, 0.0);
  if (
    !params.enabled ||
    tangent_metric_scalar <= 0.0 ||
    geometry.x.size() % 3 != 0 ||
    geometry.jacobian.cols() != 6)
  {
    reset_gds_modes();
    reset_softmax_gds_modes();
    tangent_escape_supervisor_.active = false;
    tangent_escape_supervisor_.mode = 0;
    if (debug_data != nullptr) {
      *debug_data = best_debug_data;
    }
    return;
  }

  const std::string leaf_mode = params.leaf_mode.empty() ? "gds" : params.leaf_mode;
  const bool use_gds_branch = leaf_mode == "gds" || leaf_mode == "gds_branch";
  const bool use_stable_hybrid_gds =
    leaf_mode == "stable_hybrid_gds" || leaf_mode == "latched_gds" ||
    leaf_mode == "stage4_stable";
  const bool use_collision_scaled_accel =
    leaf_mode == "collision_scaled" || leaf_mode == "collision_scaled_accel";
  const bool use_softmax_gds_branch =
    use_stable_hybrid_gds || leaf_mode == "softmax_gds" || leaf_mode == "score_gds" ||
    leaf_mode == "stage3" || use_collision_scaled_accel;
  const bool use_direct_accel =
    leaf_mode == "direct" || leaf_mode == "direct_accel" || leaf_mode == "stage1";
  if (!use_gds_branch && !use_softmax_gds_branch && !use_direct_accel) {
    throw std::runtime_error(
            "Unsupported tangent_escape_rmp_leaf_mode: " + leaf_mode +
            " (expected gds, softmax_gds, stable_hybrid_gds, collision_scaled, or "
            "direct_accel)");
  }
  const bool use_stage4_supervisor = params.supervisor_enabled && use_softmax_gds_branch;
  const double supervisor_dt = std::max(params.supervisor_dt, 1e-4);
  const double recovery_duration = std::max(params.recovery_duration, 0.0);
  const double blocked_decay_time = std::max(params.blocked_memory_decay_time, 1e-3);
  const double blocked_decay = std::exp(-supervisor_dt / blocked_decay_time);
  if (use_stage4_supervisor) {
    for (auto & sensor_memory : tangent_escape_blocked_memory_) {
      for (auto & memory : sensor_memory) {
        memory *= blocked_decay;
        if (memory < 1e-4) {
          memory = 0.0;
        }
      }
    }
  }

  const auto max_blocked_memory = [this]() {
      double maximum = 0.0;
      for (const auto & sensor_memory : tangent_escape_blocked_memory_) {
        for (const double memory : sensor_memory) {
          maximum = std::max(maximum, memory);
        }
      }
      return maximum;
    };
  const auto advance_supervisor_recovery = [
      this,
      use_stage4_supervisor,
      supervisor_dt,
      recovery_duration,
      use_stable_hybrid_gds,
      &best_debug_data,
      &max_blocked_memory]() {
      if (!use_stage4_supervisor || !tangent_escape_supervisor_.active) {
        tangent_escape_supervisor_.active = false;
        tangent_escape_supervisor_.mode = 0;
        return;
      }

      tangent_escape_supervisor_.mode = 3;
      tangent_escape_supervisor_.recovery_timer_s += supervisor_dt;
      if (
        recovery_duration <= 0.0 ||
        tangent_escape_supervisor_.recovery_timer_s >= recovery_duration)
      {
        tangent_escape_supervisor_.active = false;
        tangent_escape_supervisor_.mode = 0;
        tangent_escape_supervisor_.branch_age_s = 0.0;
        tangent_escape_supervisor_.hold_age_s = 0.0;
        tangent_escape_supervisor_.stuck_timer_s = 0.0;
        tangent_escape_supervisor_.recovery_timer_s = 0.0;
        return;
      }

      // Preserve the inactive recovery mode in the debug stream. Candidate data
      // still starts at index 61, so this remains backward-compatible.
      best_debug_data.assign(61, 0.0);
      best_debug_data[1] = static_cast<double>(
        tangent_escape_supervisor_.control_point_index);
      best_debug_data[20] = tangent_escape_supervisor_.tangent.x();
      best_debug_data[21] = tangent_escape_supervisor_.tangent.y();
      best_debug_data[22] = tangent_escape_supervisor_.tangent.z();
      best_debug_data[26] = use_stable_hybrid_gds ? 4.0 : 3.0;
      best_debug_data[38] = static_cast<double>(tangent_escape_supervisor_.slot);
      best_debug_data[47] = 3.0;
      best_debug_data[49] = tangent_escape_supervisor_.hold_age_s;
      best_debug_data[52] = tangent_escape_supervisor_.stuck_timer_s;
      best_debug_data[54] = 1.0;
      best_debug_data[55] = 1.0;
      best_debug_data[57] = max_blocked_memory();
      best_debug_data[58] = tangent_escape_supervisor_.branch_age_s;
    };

  if (obstacles.empty()) {
    reset_gds_modes();
    reset_softmax_gds_modes();
    advance_supervisor_recovery();
    if (debug_data != nullptr) {
      *debug_data = best_debug_data;
    }
    return;
  }

  if (use_direct_accel) {
    reset_gds_modes();
    reset_softmax_gds_modes();
  } else if (use_gds_branch) {
    reset_softmax_gds_modes();
  } else if (use_softmax_gds_branch) {
    reset_gds_modes();
  }

  const bool use_natural_rmp = uses_rmp2_solve() && uses_natural_rmp();
  const auto velocity = velocity_of(geometry, qd);
  const std::size_t point_count = static_cast<std::size_t>(geometry.x.size() / 3);
  const double prediction_dt = std::max(params.nominal_prediction_dt, 0.0);
  const double min_nominal_speed = std::max(params.nominal_min_speed, 1e-9);
  const double tangent_bias_weight = std::max(params.tangent_bias_weight, 0.0);
  const double max_accel = std::max(params.max_accel, 0.0);
  const double gds_position_gain = std::max(params.position_gain, 0.0);
  const double gds_damping_gain = std::max(params.damping_gain, 0.0);
  const double escape_length = std::max(params.escape_length, 0.0);
  const double collision_accel_scale = std::max(params.collision_accel_scale, 0.0);
  const double softmax_beta = std::max(params.softmax_beta, 0.0);
  const double goal_score_weight = params.goal_weight;
  const double continuity_score_weight = params.continuity_weight;
  const double duplicate_risk_weight = std::max(params.duplicate_risk_weight, 0.0);
  const double adjacent_risk_weight = std::max(params.adjacent_block_weight, 0.0);
  const double branch_hold_duration = std::max(params.branch_hold_duration, 0.0);
  const double branch_hold_weight = std::max(params.branch_hold_weight, 0.0);
  const double branch_hold_max_adjacent_risk =
    std::max(params.branch_hold_max_adjacent_risk, 0.0);
  const double stable_mode_normal_tolerance =
    std::clamp(params.stable_mode_normal_tolerance, 1e-6, 1.0);
  const double stuck_activation_threshold =
    std::clamp(params.stuck_activation_threshold, 0.0, 1.0);
  const double stuck_velocity_threshold = std::max(params.stuck_velocity_threshold, 0.0);
  const double stuck_progress_threshold = std::max(params.stuck_progress_threshold, 0.0);
  const double stuck_time_threshold = std::max(params.stuck_time_threshold, 0.0);
  const double stuck_metric_boost = std::max(params.stuck_metric_boost, 1.0);
  const double stuck_accel_boost = std::max(params.stuck_accel_boost, 1.0);
  const double blocked_update_duration =
    std::max(params.blocked_memory_update_duration, 0.0);
  const double blocked_progress_threshold =
    std::max(params.blocked_memory_progress_threshold, 0.0);
  const double blocked_clearance_improvement =
    std::max(params.blocked_memory_clearance_improvement, 0.0);
  const double blocked_penalty_weight = std::max(params.blocked_memory_penalty_weight, 0.0);
  std::array<bool, RB10Model::sensor_control_points.size()> gds_mode_used{};
  std::array<
    std::array<bool, tangent_escape_softmax_candidate_count_>,
    RB10Model::sensor_control_points.size()> softmax_mode_used{};
  const TangentEscapeSupervisorState supervisor_before_solve = tangent_escape_supervisor_;
  TangentEscapeSupervisorState pending_supervisor_state = supervisor_before_solve;
  bool pending_supervisor_update = false;
  bool pending_blocked_memory_set = false;
  bool pending_blocked_memory_clear = false;
  std::size_t pending_blocked_point_index = 0;
  std::size_t pending_blocked_slot = 0;

  for (std::size_t point_index = 0; point_index < point_count; ++point_index) {
    if (point_index >= RB10Model::sensor_control_points.size()) {
      break;
    }

    const Eigen::Vector3d control_point =
      geometry.x.segment<3>(static_cast<Eigen::Index>(3 * point_index));
    const Eigen::Matrix<double, 3, 6> point_jacobian =
      geometry.jacobian.block<3, 6>(static_cast<Eigen::Index>(3 * point_index), 0);
    const Eigen::Vector3d point_velocity =
      velocity.segment<3>(static_cast<Eigen::Index>(3 * point_index));
    const Eigen::Vector3d point_curvature =
      geometry.curvature.segment<3>(static_cast<Eigen::Index>(3 * point_index));
    const double point_radius = RB10Model::sensor_control_points[point_index].radius;

    for (const auto & obstacle : obstacles) {
      if (
        obstacle.radius <= 0.0 ||
        !obstacle.center.allFinite() ||
        obstacle.proximity_control_point_index != static_cast<int>(point_index))
      {
        continue;
      }

      const Eigen::Vector3d delta = control_point - obstacle.center;
      const double center_distance = delta.norm();
      if (center_distance <= 1e-9) {
        continue;
      }

      const double clearance =
        center_distance - (point_radius + obstacle.radius) - config_.collision.margin;
      const double proximity_activation = tangent_escape_activation(clearance, params);
      if (proximity_activation <= 0.0) {
        continue;
      }

      const Eigen::Vector3d collision_normal = delta / center_distance;
      const Eigen::Vector3d obstacle_direction = -collision_normal;
      const double clearance_rate = collision_normal.dot(point_velocity);
      const double collision_accel =
        collision_scalar_acceleration(clearance, clearance_rate, config_.collision);
      const double scaled_collision_accel = collision_accel_scale * collision_accel;
      if (!std::isfinite(collision_accel) || !std::isfinite(scaled_collision_accel)) {
        continue;
      }
      const Eigen::Vector3d nominal_velocity =
        point_jacobian * (qd + prediction_dt * nominal_qdd);
      const double nominal_speed = nominal_velocity.norm();
      if (nominal_speed < min_nominal_speed || !nominal_velocity.allFinite()) {
        continue;
      }

      const Eigen::Vector3d nominal_direction = nominal_velocity / nominal_speed;
      const double beta = nominal_direction.dot(obstacle_direction);
      const double blocking_activation = tangent_escape_blocking_activation(beta, params);
      const double activation = proximity_activation * blocking_activation;
      if (activation < params.min_activation) {
        continue;
      }
      const double debug_score = activation * (1.0 + beta);

      const auto & sensor = RB10Model::sensor_control_points[point_index];
      if (sensor.parent_link >= context.link_rotations.size()) {
        continue;
      }
      const Eigen::Vector3d tangent_bias_world =
        context.link_rotations[sensor.parent_link] * sensor.local_tangent_bias;
      const auto nominal_tangent = project_to_tangent_direction(
        nominal_direction,
        obstacle_direction,
        params.min_tangent_norm);
      const auto bias_tangent = project_to_tangent_direction(
        tangent_bias_world,
        obstacle_direction,
        params.min_tangent_norm);
      if (!nominal_tangent.has_value() && !bias_tangent.has_value()) {
        continue;
      }

      Eigen::Vector3d tangent_raw = Eigen::Vector3d::Zero();
      if (nominal_tangent.has_value()) {
        tangent_raw += nominal_tangent.value();
      }
      if (bias_tangent.has_value()) {
        tangent_raw += tangent_bias_weight * bias_tangent.value();
      }
      const double tangent_norm = tangent_raw.norm();
      if (tangent_norm <= params.min_tangent_norm || !tangent_raw.allFinite()) {
        continue;
      }
      const Eigen::Vector3d tangent = tangent_raw / tangent_norm;
      const double effective_metric_scalar = activation * tangent_metric_scalar;
      if (effective_metric_scalar <= 0.0) {
        continue;
      }
      double reported_effective_metric_scalar = effective_metric_scalar;

      double mode_id = 1.0;
      double scalar_s = 0.0;
      double scalar_target = 0.0;
      double scalar_velocity = point_velocity.dot(tangent);
      double scalar_error = 0.0;
      Eigen::Vector3d mode_origin = control_point;
      Eigen::Vector3d applied_tangent = tangent;
      double desired_tangent_accel = 0.0;
      Eigen::Vector3d desired_acceleration = Eigen::Vector3d::Zero();
      double stage3_candidate_count = 0.0;
      double stage3_selected_candidate_index = -1.0;
      double stage3_selected_weight = 0.0;
      double stage3_selected_score = 0.0;
      double stage3_selected_goal_score = 0.0;
      double stage3_selected_continuity_score = 0.0;
      double stage3_selected_adjacent_risk = 0.0;
      double stage3_branch_weight_sum = 0.0;
      double stage3_weight_entropy = 0.0;
      double stage4_supervisor_mode_id = 0.0;
      double stage4_hold_active = 0.0;
      double stage4_hold_age_s = 0.0;
      double stage4_selected_hold_bonus = 0.0;
      double stage4_stuck_score = 0.0;
      double stage4_stuck_timer_s = 0.0;
      double stage4_stuck_active = 0.0;
      double stage4_metric_boost = 1.0;
      double stage4_accel_boost = 1.0;
      double stage4_selected_blocked_penalty = 0.0;
      double stage4_max_blocked_memory = 0.0;
      double stage4_branch_age_s = 0.0;
      double stage4_branch_progress_m = 0.0;
      double stage4_clearance_improvement_m = 0.0;
      std::vector<double> stage3_candidate_debug_values;

      if (use_softmax_gds_branch) {
        const bool supervisor_owns_point =
          use_stage4_supervisor && supervisor_before_solve.active &&
          supervisor_before_solve.mode != 3 &&
          supervisor_before_solve.control_point_index == point_index;
        const int stable_supervisor_mode = supervisor_owns_point ?
          supervisor_before_solve.mode : 0;
        const bool stable_hold_phase =
          supervisor_owns_point && branch_hold_duration > 0.0 &&
          supervisor_before_solve.branch_age_s <= branch_hold_duration;
        bool stable_modes_latched = false;
        if (use_stable_hybrid_gds) {
          for (const auto & mode : tangent_escape_softmax_gds_modes_[point_index]) {
            stable_modes_latched = stable_modes_latched || mode.active;
          }
          if (stable_modes_latched) {
            bool relatch_required = false;
            for (const auto & mode : tangent_escape_softmax_gds_modes_[point_index]) {
              if (!mode.active) {
                continue;
              }
              relatch_required = relatch_required ||
                std::abs(mode.tangent.dot(obstacle_direction)) > stable_mode_normal_tolerance ||
                mode.supervisor_mode != stable_supervisor_mode ||
                mode.hold_phase != stable_hold_phase;
            }
            if (relatch_required) {
              for (auto & mode : tangent_escape_softmax_gds_modes_[point_index]) {
                mode.active = false;
                mode.activation = 0.0;
              }
              stable_modes_latched = false;
            }
          }
        }

        std::vector<TangentEscapeSoftmaxCandidate> candidates;
        candidates.reserve(tangent_escape_softmax_candidate_count_);
        if (stable_modes_latched) {
          for (std::size_t slot = 0; slot < tangent_escape_softmax_candidate_count_; ++slot) {
            const auto & mode = tangent_escape_softmax_gds_modes_[point_index][slot];
            if (!mode.active) {
              continue;
            }
            TangentEscapeSoftmaxCandidate candidate;
            candidate.slot = slot;
            candidate.direction = mode.tangent;
            candidates.push_back(candidate);
          }
        } else {
          add_unique_tangent_candidate(candidates, 0, nominal_tangent);
          add_unique_tangent_candidate(candidates, 1, bias_tangent);
          add_unique_tangent_candidate(
            candidates,
            2,
            project_to_tangent_direction(
              -tangent_bias_world,
              obstacle_direction,
              params.min_tangent_norm));
          add_unique_tangent_candidate(
            candidates,
            3,
            project_to_tangent_direction(
              Eigen::Vector3d::UnitZ(),
              obstacle_direction,
              params.min_tangent_norm));
          add_unique_tangent_candidate(
            candidates,
            4,
            project_to_tangent_direction(
              -Eigen::Vector3d::UnitZ(),
              obstacle_direction,
              params.min_tangent_norm));
        }

        if (candidates.empty()) {
          continue;
        }

        const std::uint64_t stable_generation =
          use_stable_hybrid_gds && !stable_modes_latched ?
          ++tangent_escape_mode_generation_ : 0;

        const Eigen::Vector3d tangent_velocity =
          point_velocity - point_velocity.dot(obstacle_direction) * obstacle_direction;
        const double tangent_velocity_norm = tangent_velocity.norm();
        const bool has_tangent_velocity =
          tangent_velocity_norm > params.min_tangent_norm && tangent_velocity.allFinite();
        Eigen::Vector3d tangent_velocity_direction = Eigen::Vector3d::Zero();
        if (has_tangent_velocity) {
          tangent_velocity_direction = tangent_velocity / tangent_velocity_norm;
        }

        double max_score = -std::numeric_limits<double>::infinity();
        for (auto & candidate : candidates) {
          auto & mode_state =
            tangent_escape_softmax_gds_modes_[point_index][candidate.slot];
          if (!mode_state.active || !mode_state.tangent.allFinite()) {
            mode_state.active = true;
            mode_state.origin = control_point;
            mode_state.tangent = candidate.direction;
            mode_state.obstacle_direction = obstacle_direction;
            mode_state.activation = activation;
            mode_state.generation = stable_generation;
            mode_state.supervisor_mode = stable_supervisor_mode;
            mode_state.hold_phase = stable_hold_phase;
          }
          if (!use_stable_hybrid_gds) {
            mode_state.activation = activation;
          }
          softmax_mode_used[point_index][candidate.slot] = true;

          candidate.origin = mode_state.origin;
          if (use_stable_hybrid_gds && stable_modes_latched) {
            candidate.applied_tangent = mode_state.tangent;
            candidate.goal_score = mode_state.goal_score;
            candidate.continuity_score = mode_state.continuity_score;
            candidate.duplicate_risk = mode_state.duplicate_risk;
            candidate.adjacent_risk = mode_state.adjacent_risk;
            candidate.hold_bonus = mode_state.hold_bonus;
            candidate.blocked_penalty = mode_state.blocked_penalty;
            candidate.base_score = mode_state.base_score;
            candidate.score = mode_state.score;
            candidate.weight = mode_state.branch_weight;
            candidate.metric_boost = mode_state.metric_boost;
            candidate.accel_boost = mode_state.accel_boost;
            candidate.mode_generation = mode_state.generation;
            candidate.weights_latched = true;
          } else {
            candidate.applied_tangent = use_stable_hybrid_gds ?
              mode_state.tangent :
              reproject_mode_tangent(
                mode_state.tangent,
                candidate.direction,
                obstacle_direction,
                params.min_tangent_norm);
            mode_state.tangent = candidate.applied_tangent;
            candidate.goal_score = candidate.applied_tangent.dot(nominal_direction);
            candidate.continuity_score = has_tangent_velocity ?
              candidate.applied_tangent.dot(tangent_velocity_direction) :
              0.0;
            // Vertical branches are intentionally exempt from predictive duplicate risk.
            candidate.duplicate_risk = candidate.slot == 3 || candidate.slot == 4 ?
              0.0 :
              tangent_escape_predictive_duplicate_risk(
                params,
                geometry,
                point_index,
                candidate.applied_tangent,
                obstacle_direction,
                proximity_activation);
            candidate.adjacent_risk = tangent_escape_instantaneous_adjacent_risk(
              params,
              geometry,
              point_count,
              point_index,
              candidate.applied_tangent,
              obstacles);
            candidate.base_score =
              goal_score_weight * candidate.goal_score +
              continuity_score_weight * candidate.continuity_score -
              duplicate_risk_weight * candidate.duplicate_risk -
              adjacent_risk_weight * candidate.adjacent_risk;
            if (use_stage4_supervisor) {
              const bool same_held_branch =
                supervisor_before_solve.active &&
                supervisor_before_solve.mode != 3 &&
                supervisor_before_solve.control_point_index == point_index &&
                supervisor_before_solve.slot == candidate.slot;
              if (
                same_held_branch &&
                branch_hold_duration > 0.0 &&
                supervisor_before_solve.branch_age_s <= branch_hold_duration &&
                candidate.adjacent_risk <= branch_hold_max_adjacent_risk)
              {
                const double hold_phase =
                  1.0 - std::clamp(
                    supervisor_before_solve.branch_age_s / branch_hold_duration,
                    0.0,
                    1.0);
                candidate.hold_bonus = branch_hold_weight * hold_phase;
              }
              const double memory =
                tangent_escape_blocked_memory_[point_index][candidate.slot];
              candidate.blocked_penalty = blocked_penalty_weight * memory;
              stage4_max_blocked_memory = std::max(stage4_max_blocked_memory, memory);
            }
            candidate.score =
              candidate.base_score +
              candidate.hold_bonus +
              candidate.stuck_bonus -
              candidate.blocked_penalty;
          }
        }

        if (
          use_stage4_supervisor &&
          supervisor_before_solve.active &&
          supervisor_before_solve.mode != 3 &&
          supervisor_before_solve.control_point_index == point_index &&
          branch_hold_duration > 0.0 &&
          supervisor_before_solve.branch_age_s <= branch_hold_duration &&
          !(use_stable_hybrid_gds && stable_modes_latched))
        {
          auto held_candidate = std::find_if(
            candidates.begin(),
            candidates.end(),
            [&supervisor_before_solve, branch_hold_max_adjacent_risk](const auto & candidate) {
              return
                candidate.slot == supervisor_before_solve.slot &&
                candidate.adjacent_risk <= branch_hold_max_adjacent_risk;
            });
          if (held_candidate != candidates.end()) {
            double best_other_score = -std::numeric_limits<double>::infinity();
            for (const auto & candidate : candidates) {
              if (candidate.slot != held_candidate->slot) {
                best_other_score = std::max(best_other_score, candidate.score);
              }
            }
            if (std::isfinite(best_other_score) && held_candidate->score <= best_other_score) {
              const double hold_phase =
                1.0 - std::clamp(
                supervisor_before_solve.branch_age_s / branch_hold_duration,
                0.0,
                1.0);
              const double hold_margin = std::max(branch_hold_weight * hold_phase, 1e-6);
              const double lock_bonus = best_other_score + hold_margin - held_candidate->score;
              held_candidate->hold_bonus += lock_bonus;
              held_candidate->score += lock_bonus;
            }
          }
        }

        for (const auto & candidate : candidates) {
          max_score = std::max(max_score, candidate.score);
        }

        double exp_sum = 0.0;
        if (use_stable_hybrid_gds && stable_modes_latched) {
          for (const auto & candidate : candidates) {
            exp_sum += candidate.weight;
          }
        } else {
          for (auto & candidate : candidates) {
            const double exponent = softmax_beta * (candidate.score - max_score);
            candidate.weight = std::exp(std::clamp(exponent, -60.0, 60.0));
            exp_sum += candidate.weight;
          }
        }
        if (exp_sum <= 1e-12 || !std::isfinite(exp_sum)) {
          continue;
        }

        TangentEscapeSoftmaxCandidate * selected_candidate = nullptr;
        Eigen::Vector3d weighted_desired_acceleration = Eigen::Vector3d::Zero();
        for (auto & candidate : candidates) {
          if (!(use_stable_hybrid_gds && stable_modes_latched)) {
            candidate.weight /= exp_sum;
          }
          const bool stuck_boost_applies =
            use_stage4_supervisor &&
            supervisor_before_solve.active &&
            supervisor_before_solve.mode == 2 &&
            supervisor_before_solve.control_point_index == point_index &&
            supervisor_before_solve.slot == candidate.slot;
          if (!(use_stable_hybrid_gds && stable_modes_latched)) {
            candidate.metric_boost = stuck_boost_applies ? stuck_metric_boost : 1.0;
            candidate.accel_boost = stuck_boost_applies ? stuck_accel_boost : 1.0;
          }
          auto & mode_state =
            tangent_escape_softmax_gds_modes_[point_index][candidate.slot];
          const double candidate_activation = use_stable_hybrid_gds ?
            mode_state.activation : activation;
          candidate.metric_scalar =
            candidate_activation * tangent_metric_scalar *
            candidate.weight * candidate.metric_boost;
          candidate.scalar_s = candidate.applied_tangent.dot(control_point - candidate.origin);
          candidate.scalar_target = escape_length;
          candidate.scalar_velocity = candidate.applied_tangent.dot(point_velocity);
          candidate.scalar_error = candidate.scalar_target - candidate.scalar_s;
          candidate.clearance_rate = clearance_rate;
          candidate.collision_accel = collision_accel;
          candidate.scaled_collision_accel = scaled_collision_accel;
          if (use_collision_scaled_accel) {
            candidate.scalar_target = 0.0;
            candidate.scalar_error = 0.0;
            const double effective_max_accel = max_accel * candidate.accel_boost;
            const double collision_drive =
              candidate.accel_boost * candidate.scaled_collision_accel;
            candidate.desired_tangent_accel =
              collision_drive - gds_damping_gain * candidate.scalar_velocity;
            if (effective_max_accel > 0.0) {
              candidate.desired_tangent_accel = std::clamp(
                candidate.desired_tangent_accel,
                -effective_max_accel,
                effective_max_accel);
            }
          } else if (use_stable_hybrid_gds) {
            const double effective_position_gain =
              gds_position_gain * candidate.accel_boost;
            const double effective_damping_gain =
              gds_damping_gain * candidate.accel_boost;
            const double effective_max_accel = max_accel * candidate.accel_boost;
            const double spring_acceleration = bounded_spring_acceleration(
              candidate.scalar_error,
              effective_position_gain,
              effective_max_accel);
            candidate.desired_tangent_accel =
              spring_acceleration - effective_damping_gain * candidate.scalar_velocity;
            candidate.potential_energy = bounded_spring_potential(
              candidate.scalar_error,
              effective_position_gain,
              effective_max_accel,
              candidate.metric_scalar);
            candidate.kinetic_energy =
              0.5 * candidate.metric_scalar * candidate.scalar_velocity *
              candidate.scalar_velocity;
            candidate.lyapunov_energy =
              candidate.potential_energy + candidate.kinetic_energy;
            candidate.damping_vdot =
              -candidate.metric_scalar * effective_damping_gain *
              candidate.scalar_velocity * candidate.scalar_velocity;
            candidate.mode_generation = mode_state.generation;
            candidate.weights_latched = true;
            candidate.bounded_potential = true;
            candidate.mode_normal_dot_tangent =
              mode_state.obstacle_direction.dot(candidate.applied_tangent);
          } else if (max_accel > 0.0) {
            candidate.desired_tangent_accel =
              gds_position_gain * candidate.scalar_error -
              gds_damping_gain * candidate.scalar_velocity;
            candidate.desired_tangent_accel = std::clamp(
              candidate.desired_tangent_accel * candidate.accel_boost,
              -max_accel * candidate.accel_boost,
              max_accel * candidate.accel_boost);
          } else {
            candidate.desired_tangent_accel =
              candidate.accel_boost *
              (gds_position_gain * candidate.scalar_error -
              gds_damping_gain * candidate.scalar_velocity);
          }
          if (!use_stable_hybrid_gds) {
            candidate.mode_normal_dot_tangent =
              obstacle_direction.dot(candidate.applied_tangent);
          }

          if (use_stable_hybrid_gds && !stable_modes_latched) {
            mode_state.branch_weight = candidate.weight;
            mode_state.goal_score = candidate.goal_score;
            mode_state.continuity_score = candidate.continuity_score;
            mode_state.duplicate_risk = candidate.duplicate_risk;
            mode_state.adjacent_risk = candidate.adjacent_risk;
            mode_state.hold_bonus = candidate.hold_bonus;
            mode_state.blocked_penalty = candidate.blocked_penalty;
            mode_state.base_score = candidate.base_score;
            mode_state.score = candidate.score;
            mode_state.metric_boost = candidate.metric_boost;
            mode_state.accel_boost = candidate.accel_boost;
            mode_state.generation = stable_generation;
            mode_state.supervisor_mode = stable_supervisor_mode;
            mode_state.hold_phase = stable_hold_phase;
            candidate.mode_generation = stable_generation;
          }

          const Eigen::Vector3d candidate_acceleration =
            candidate.desired_tangent_accel * candidate.applied_tangent;
          weighted_desired_acceleration += candidate.weight * candidate_acceleration;
          stage3_branch_weight_sum += candidate.weight;
          if (candidate.weight > 1e-12) {
            stage3_weight_entropy -= candidate.weight * std::log(candidate.weight);
          }

          const RowVector6 scalar_jacobian =
            candidate.applied_tangent.transpose() * point_jacobian;
          const double scalar_curvature = candidate.applied_tangent.dot(point_curvature);
          accumulate_scalar_leaf(
            use_natural_rmp,
            scalar_jacobian,
            candidate.metric_scalar,
            candidate.desired_tangent_accel,
            scalar_curvature,
            metric,
            force);

          if (
            selected_candidate == nullptr ||
            candidate.weight > selected_candidate->weight)
          {
            selected_candidate = &candidate;
          }
        }

        if (selected_candidate == nullptr) {
          continue;
        }

        mode_id = use_collision_scaled_accel ? 5.0 : (use_stable_hybrid_gds ? 4.0 : 3.0);
        stage3_candidate_count = static_cast<double>(candidates.size());
        stage3_selected_candidate_index = static_cast<double>(selected_candidate->slot);
        stage3_selected_weight = selected_candidate->weight;
        stage3_selected_score = selected_candidate->score;
        stage3_selected_goal_score = selected_candidate->goal_score;
        stage3_selected_continuity_score = selected_candidate->continuity_score;
        stage3_selected_adjacent_risk = selected_candidate->adjacent_risk;
        mode_origin = selected_candidate->origin;
        applied_tangent = selected_candidate->applied_tangent;
        scalar_s = selected_candidate->scalar_s;
        scalar_target = selected_candidate->scalar_target;
        scalar_velocity = selected_candidate->scalar_velocity;
        scalar_error = selected_candidate->scalar_error;
        desired_tangent_accel = selected_candidate->desired_tangent_accel;
        desired_acceleration = weighted_desired_acceleration;
        if (use_stable_hybrid_gds) {
          const auto & selected_mode =
            tangent_escape_softmax_gds_modes_[point_index][selected_candidate->slot];
          reported_effective_metric_scalar =
            selected_mode.activation * tangent_metric_scalar;
        }
        if (use_stage4_supervisor) {
          stage4_max_blocked_memory = max_blocked_memory();
        }

        if (use_stage4_supervisor && debug_score > best_debug_score) {
          const bool same_branch =
            supervisor_before_solve.active &&
            supervisor_before_solve.mode != 3 &&
            supervisor_before_solve.control_point_index == point_index &&
            supervisor_before_solve.slot == selected_candidate->slot;
          TangentEscapeSupervisorState next_state = supervisor_before_solve;
          if (!same_branch) {
            next_state.active = true;
            next_state.control_point_index = point_index;
            next_state.slot = selected_candidate->slot;
            next_state.tangent = selected_candidate->applied_tangent;
            next_state.branch_age_s = 0.0;
            next_state.hold_age_s = 0.0;
            next_state.stuck_timer_s = 0.0;
            next_state.recovery_timer_s = 0.0;
            next_state.start_scalar_s = selected_candidate->scalar_s;
            next_state.best_scalar_s = selected_candidate->scalar_s;
            next_state.start_clearance = clearance;
            next_state.best_clearance = clearance;
          } else {
            next_state.branch_age_s += supervisor_dt;
            next_state.hold_age_s += supervisor_dt;
            next_state.tangent = selected_candidate->applied_tangent;
            next_state.best_scalar_s = std::max(
              next_state.best_scalar_s,
              selected_candidate->scalar_s);
            next_state.best_clearance = std::max(
              next_state.best_clearance,
              clearance);
          }

          const double branch_progress =
            next_state.best_scalar_s - next_state.start_scalar_s;
          const double clearance_improvement =
            next_state.best_clearance - next_state.start_clearance;
          const bool low_progress = branch_progress < stuck_progress_threshold;
          const bool low_velocity =
            std::abs(selected_candidate->scalar_velocity) < stuck_velocity_threshold;
          const bool stuck_candidate =
            activation >= stuck_activation_threshold &&
            next_state.branch_age_s >= stuck_time_threshold &&
            low_progress &&
            low_velocity;
          if (stuck_candidate) {
            next_state.stuck_timer_s += supervisor_dt;
          } else {
            next_state.stuck_timer_s =
              std::max(0.0, next_state.stuck_timer_s - supervisor_dt);
          }
          const bool stuck_active =
            next_state.stuck_timer_s >= stuck_time_threshold;
          next_state.mode = stuck_active ? 2 : 1;

          pending_supervisor_state = next_state;
          pending_supervisor_update = true;
          pending_blocked_point_index = point_index;
          pending_blocked_slot = selected_candidate->slot;
          pending_blocked_memory_set =
            next_state.branch_age_s >= blocked_update_duration &&
            branch_progress < blocked_progress_threshold &&
            clearance_improvement < blocked_clearance_improvement;
          pending_blocked_memory_clear =
            !pending_blocked_memory_set &&
            (branch_progress >= blocked_progress_threshold ||
            clearance_improvement >= blocked_clearance_improvement);

          stage4_supervisor_mode_id = static_cast<double>(next_state.mode);
          stage4_hold_active =
            same_branch &&
            branch_hold_duration > 0.0 &&
            next_state.branch_age_s <= branch_hold_duration ?
            1.0 :
            0.0;
          stage4_hold_age_s = next_state.hold_age_s;
          stage4_selected_hold_bonus = selected_candidate->hold_bonus;
          stage4_stuck_score = stuck_candidate ? 1.0 : 0.0;
          stage4_stuck_timer_s = next_state.stuck_timer_s;
          stage4_stuck_active = stuck_active ? 1.0 : 0.0;
          stage4_metric_boost = selected_candidate->metric_boost;
          stage4_accel_boost = selected_candidate->accel_boost;
          stage4_selected_blocked_penalty = selected_candidate->blocked_penalty;
          stage4_branch_age_s = next_state.branch_age_s;
          stage4_branch_progress_m = branch_progress;
          stage4_clearance_improvement_m = clearance_improvement;
        }

        stage3_candidate_debug_values.reserve(candidates.size() * 29);
        for (const auto & candidate : candidates) {
          stage3_candidate_debug_values.push_back(static_cast<double>(candidate.slot));
          stage3_candidate_debug_values.push_back(candidate.weight);
          stage3_candidate_debug_values.push_back(candidate.score);
          stage3_candidate_debug_values.push_back(candidate.goal_score);
          stage3_candidate_debug_values.push_back(candidate.continuity_score);
          stage3_candidate_debug_values.push_back(candidate.adjacent_risk);
          stage3_candidate_debug_values.push_back(candidate.hold_bonus);
          stage3_candidate_debug_values.push_back(candidate.blocked_penalty);
          stage3_candidate_debug_values.push_back(candidate.stuck_bonus);
          stage3_candidate_debug_values.push_back(candidate.base_score);
          stage3_candidate_debug_values.push_back(candidate.applied_tangent.x());
          stage3_candidate_debug_values.push_back(candidate.applied_tangent.y());
          stage3_candidate_debug_values.push_back(candidate.applied_tangent.z());
          stage3_candidate_debug_values.push_back(candidate.metric_scalar);
          stage3_candidate_debug_values.push_back(candidate.metric_boost);
          stage3_candidate_debug_values.push_back(candidate.accel_boost);
          stage3_candidate_debug_values.push_back(1.0);
          stage3_candidate_debug_values.push_back(candidate.duplicate_risk);
          stage3_candidate_debug_values.push_back(candidate.scalar_s);
          stage3_candidate_debug_values.push_back(candidate.scalar_velocity);
          stage3_candidate_debug_values.push_back(candidate.scalar_error);
          stage3_candidate_debug_values.push_back(candidate.potential_energy);
          stage3_candidate_debug_values.push_back(candidate.kinetic_energy);
          stage3_candidate_debug_values.push_back(candidate.lyapunov_energy);
          stage3_candidate_debug_values.push_back(candidate.damping_vdot);
          stage3_candidate_debug_values.push_back(candidate.weights_latched ? 1.0 : 0.0);
          stage3_candidate_debug_values.push_back(
            static_cast<double>(candidate.mode_generation));
          stage3_candidate_debug_values.push_back(candidate.bounded_potential ? 1.0 : 0.0);
          stage3_candidate_debug_values.push_back(candidate.mode_normal_dot_tangent);
          stage3_candidate_debug_values.push_back(candidate.clearance_rate);
          stage3_candidate_debug_values.push_back(candidate.collision_accel);
          stage3_candidate_debug_values.push_back(candidate.scaled_collision_accel);
        }
      } else if (use_gds_branch) {
        auto & mode_state = tangent_escape_gds_modes_[point_index];
        if (!mode_state.active || !mode_state.tangent.allFinite()) {
          mode_state.active = true;
          mode_state.origin = control_point;
          mode_state.tangent = tangent;
        }
        mode_state.activation = activation;
        gds_mode_used[point_index] = 1;

        mode_id = 2.0;
        mode_origin = mode_state.origin;
        applied_tangent = mode_state.tangent;
        scalar_s = applied_tangent.dot(control_point - mode_origin);
        scalar_target = escape_length;
        scalar_velocity = applied_tangent.dot(point_velocity);
        scalar_error = scalar_target - scalar_s;
        desired_tangent_accel =
          gds_position_gain * scalar_error - gds_damping_gain * scalar_velocity;
        if (max_accel > 0.0) {
          desired_tangent_accel = std::clamp(
            desired_tangent_accel,
            -max_accel,
            max_accel);
        }
        desired_acceleration = desired_tangent_accel * applied_tangent;

        const RowVector6 scalar_jacobian = applied_tangent.transpose() * point_jacobian;
        const double scalar_curvature = applied_tangent.dot(point_curvature);
        accumulate_scalar_leaf(
          use_natural_rmp,
          scalar_jacobian,
          reported_effective_metric_scalar,
          desired_tangent_accel,
          scalar_curvature,
          metric,
          force);
      } else {
        desired_tangent_accel =
          std::max(params.tangent_gain, 0.0) -
          std::max(params.damping_gain, 0.0) * scalar_velocity;
        if (max_accel > 0.0) {
          desired_tangent_accel = std::clamp(
            desired_tangent_accel,
            -max_accel,
            max_accel);
        }
        desired_acceleration = desired_tangent_accel * applied_tangent;
        const Eigen::Matrix3d leaf_metric =
          effective_metric_scalar * (applied_tangent * applied_tangent.transpose());
        if (leaf_metric.cwiseAbs().maxCoeff() <= 0.0) {
          continue;
        }

        accumulate_vector_leaf(
          use_natural_rmp,
          point_jacobian,
          leaf_metric,
          desired_acceleration,
          point_curvature,
          metric,
          force);
      }

      if (debug_score > best_debug_score) {
        best_debug_score = debug_score;
        best_debug_data = {
          1.0,
          static_cast<double>(point_index),
          clearance,
          beta,
          proximity_activation,
          blocking_activation,
          activation,
          debug_score,
          scalar_velocity,
          desired_tangent_accel,
          reported_effective_metric_scalar,
          control_point.x(),
          control_point.y(),
          control_point.z(),
          obstacle.center.x(),
          obstacle.center.y(),
          obstacle.center.z(),
          collision_normal.x(),
          collision_normal.y(),
          collision_normal.z(),
          applied_tangent.x(),
          applied_tangent.y(),
          applied_tangent.z(),
          desired_acceleration.x(),
          desired_acceleration.y(),
          desired_acceleration.z(),
          mode_id,
          scalar_s,
          scalar_target,
          scalar_velocity,
          scalar_error,
          mode_origin.x(),
          mode_origin.y(),
          mode_origin.z(),
          applied_tangent.x(),
          applied_tangent.y(),
          applied_tangent.z(),
          stage3_candidate_count,
          stage3_selected_candidate_index,
          stage3_selected_weight,
          stage3_selected_score,
          stage3_selected_goal_score,
          stage3_selected_continuity_score,
          stage3_selected_adjacent_risk,
          softmax_beta,
          stage3_branch_weight_sum,
          stage3_weight_entropy,
          stage4_supervisor_mode_id,
          stage4_hold_active,
          stage4_hold_age_s,
          stage4_selected_hold_bonus,
          stage4_stuck_score,
          stage4_stuck_timer_s,
          stage4_stuck_active,
          stage4_metric_boost,
          stage4_accel_boost,
          stage4_selected_blocked_penalty,
          stage4_max_blocked_memory,
          stage4_branch_age_s,
          stage4_branch_progress_m,
          stage4_clearance_improvement_m,
        };
        best_debug_data.insert(
          best_debug_data.end(),
          stage3_candidate_debug_values.begin(),
          stage3_candidate_debug_values.end());
      }
    }
  }

  if (use_gds_branch) {
    for (std::size_t index = 0; index < tangent_escape_gds_modes_.size(); ++index) {
      if (!gds_mode_used[index]) {
        tangent_escape_gds_modes_[index].active = false;
        tangent_escape_gds_modes_[index].activation = 0.0;
      }
    }
  }
  if (use_softmax_gds_branch) {
    for (std::size_t point_index = 0; point_index < tangent_escape_softmax_gds_modes_.size();
      ++point_index)
    {
      for (std::size_t slot = 0; slot < tangent_escape_softmax_candidate_count_; ++slot) {
        if (!softmax_mode_used[point_index][slot]) {
          tangent_escape_softmax_gds_modes_[point_index][slot].active = false;
          tangent_escape_softmax_gds_modes_[point_index][slot].activation = 0.0;
        }
      }
    }
  }
  if (use_stage4_supervisor && !std::isfinite(best_debug_score)) {
    advance_supervisor_recovery();
  } else if (use_stage4_supervisor) {
    if (pending_supervisor_update) {
      tangent_escape_supervisor_ = pending_supervisor_state;
      auto & memory =
        tangent_escape_blocked_memory_[pending_blocked_point_index][pending_blocked_slot];
      bool blocked_memory_changed = false;
      if (pending_blocked_memory_set) {
        blocked_memory_changed = memory < 0.5;
        memory = std::max(memory, 1.0);
      } else if (pending_blocked_memory_clear) {
        blocked_memory_changed = memory > 0.5;
        memory *= 0.5;
      }
      if (use_stable_hybrid_gds && blocked_memory_changed) {
        for (auto & mode : tangent_escape_softmax_gds_modes_[pending_blocked_point_index]) {
          mode.active = false;
          mode.activation = 0.0;
        }
      }
      if (best_debug_data.size() > 57) {
        best_debug_data[57] = max_blocked_memory();
      }
    }
    tangent_escape_supervisor_.recovery_timer_s = 0.0;
  } else {
    tangent_escape_supervisor_.active = false;
    tangent_escape_supervisor_.mode = 0;
  }

  if (debug_data != nullptr) {
    *debug_data = best_debug_data;
  }
}

void PinocchioDirectRmpSolver::accumulate_joint_damping(
  const NodeGeometry & geometry,
  const JointVector & qd,
  Matrix6 & metric,
  JointVector & force) const
{
  const bool use_natural_rmp = uses_rmp2_solve() && uses_natural_rmp();
  const auto velocity = velocity_of(geometry, qd);
  const double velocity_norm = velocity.norm();
  const double nonlinear_gain = config_.damping.accel_d_gain * velocity_norm;
  const Eigen::VectorXd acceleration = -nonlinear_gain * velocity;
  const double metric_scalar =
    config_.damping.metric_scalar * velocity_norm + config_.damping.inertia;
  const Eigen::MatrixXd leaf_metric =
    metric_scalar * Eigen::MatrixXd::Identity(velocity.size(), velocity.size());
  accumulate_vector_leaf(
    use_natural_rmp,
    geometry.jacobian,
    leaf_metric,
    acceleration,
    geometry.curvature,
    metric,
    force);
}

}  // namespace rb10_rmpflow_rviz
