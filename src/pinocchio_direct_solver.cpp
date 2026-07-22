#include "rb10_rmpflow_rviz/pinocchio_direct_solver.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "rb10_rmpflow_rviz/casadi_task_graph.hpp"
#include "rb10_rmpflow_rviz/escape_energy_certificate.hpp"
#include "rb10_rmpflow_rviz/joint_acceleration_limiter.hpp"
#include "rb10_rmpflow_rviz/paper_gds_collision.hpp"
#include "rb10_rmpflow_rviz/paper_gds_joint_limit.hpp"

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

enum class LeafAblationGroup : int
{
  CspaceTarget = 1,
  JointLimits = 2,
  JointVelocityCap = 3,
  Target = 4,
  Collision = 5,
  TangentEscape = 6,
  Damping = 7,
  BodyTarget = 8,
  AxisTargetX = 9,
  AxisTargetY = 10,
  AxisTargetZ = 11,
  WristAxisTarget = 12,
  External = 13,
  Other = 99
};

int leaf_ablation_group_id(
  const RmpNodeConfig & node,
  const std::string & leaf_type)
{
  if (leaf_type == "cspace_target") {
    return static_cast<int>(LeafAblationGroup::CspaceTarget);
  }
  if (leaf_type == "joint_limit") {
    return static_cast<int>(LeafAblationGroup::JointLimits);
  }
  if (leaf_type == "joint_velocity_cap") {
    return static_cast<int>(LeafAblationGroup::JointVelocityCap);
  }
  if (leaf_type == "target") {
    return static_cast<int>(
      node.name == "body_link4" ?
      LeafAblationGroup::BodyTarget : LeafAblationGroup::Target);
  }
  if (leaf_type == "collision") {
    return static_cast<int>(LeafAblationGroup::Collision);
  }
  if (leaf_type == "tangent_escape") {
    return static_cast<int>(LeafAblationGroup::TangentEscape);
  }
  if (leaf_type == "damping") {
    return static_cast<int>(LeafAblationGroup::Damping);
  }
  if (leaf_type == "axis_target") {
    if (node.axis == "x") {
      return static_cast<int>(LeafAblationGroup::AxisTargetX);
    }
    if (node.axis == "y") {
      return static_cast<int>(LeafAblationGroup::AxisTargetY);
    }
    return static_cast<int>(LeafAblationGroup::AxisTargetZ);
  }
  if (leaf_type == "wrist_axis_target") {
    return static_cast<int>(LeafAblationGroup::WristAxisTarget);
  }
  if (leaf_type == "external") {
    return static_cast<int>(LeafAblationGroup::External);
  }
  return static_cast<int>(LeafAblationGroup::Other);
}

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

bool all_finite(std::initializer_list<double> values)
{
  return std::all_of(
    values.begin(),
    values.end(),
    [](double value) {return std::isfinite(value);});
}

bool target_leaf_has_constant_gds_metric(const TargetRmpParams & params)
{
  return
    all_finite({
      params.accel_p_gain,
      params.accel_d_gain,
      params.accel_norm_eps,
      params.metric_alpha_length_scale,
      params.min_metric_alpha,
      params.max_metric_scalar,
      params.min_metric_scalar,
      params.proximity_metric_boost_scalar,
      params.proximity_metric_boost_length_scale}) &&
    params.min_metric_alpha == 1.0 &&
    params.proximity_metric_boost_scalar == 1.0 &&
    params.accel_norm_eps > 0.0 &&
    params.metric_alpha_length_scale > 0.0 &&
    params.proximity_metric_boost_length_scale > 0.0 &&
    params.max_metric_scalar > 0.0 &&
    params.min_metric_scalar >= 0.0 &&
    params.accel_p_gain >= 0.0 &&
    params.accel_d_gain >= 0.0;
}

bool axis_leaf_has_constant_gds_metric(const AxisTargetParams & params)
{
  return
    all_finite({
      params.accel_p_gain,
      params.accel_d_gain,
      params.metric_scalar,
      params.proximity_metric_boost_scalar,
      params.proximity_metric_boost_length_scale}) &&
    params.proximity_metric_boost_scalar == 1.0 &&
    params.proximity_metric_boost_length_scale > 0.0 &&
    params.metric_scalar >= 0.0 &&
    params.accel_p_gain >= 0.0 &&
    params.accel_d_gain >= 0.0;
}

bool base_graph_has_structured_gds_form(const EigenRmpConfig & config)
{
  if (config.solve_offset != 0.0) {
    return false;
  }
  for (const auto & node : config.graph_nodes) {
    if (!node.enabled) {
      continue;
    }
    // The proof certificate accepts only task maps that are smooth on the
    // configured operating domain.  In particular, generic abs/norm/
    // normalize and division nodes need additional nonzero-domain metadata
    // that the current graph schema does not provide.
    const auto task_map_supported = [](const std::string & task_map_type) {
        return
          task_map_type == "cspace_target" ||
          task_map_type == "identity" ||
          task_map_type == "joint_limit" ||
          task_map_type == "tcp_position" ||
          task_map_type == "link_position" ||
          task_map_type == "link_orientation_axis" ||
          task_map_type == "control_points" ||
          task_map_type == "collision_distance" ||
          task_map_type == "affine" ||
          task_map_type == "elem_multiply" ||
          task_map_type == "sin" ||
          task_map_type == "cos" ||
          task_map_type == "tanh" ||
          task_map_type == "square" ||
          task_map_type == "sum" ||
          task_map_type == "weighted_sum" ||
          task_map_type == "difference" ||
          task_map_type == "concat" ||
          task_map_type == "slice";
      };
    if (!task_map_supported(node.task_map_type)) {
      return false;
    }
    const auto leaf_attachment_supported =
      [&node](const std::string & leaf_type) {
        if (leaf_type == "cspace_target") {
          return
            node.task_map_type == "cspace_target" ||
            node.task_map_type == "identity";
        }
        if (leaf_type == "joint_limit") {
          return node.task_map_type == "joint_limit";
        }
        if (leaf_type == "collision") {
          return node.task_map_type == "collision_distance";
        }
        if (leaf_type == "axis_target") {
          return node.task_map_type == "link_orientation_axis";
        }
        return true;
      };
    if (
      !leaf_attachment_supported(node.leaf_rmp_type) ||
      !leaf_attachment_supported(node.handcrafted_leaf_rmp_type))
    {
      return false;
    }
    const auto leaf_supported = [&config](const std::string & leaf_type) {
        if (leaf_type.empty() || leaf_type == "none" || leaf_type == "tangent_escape") {
          return true;
        }
        if (leaf_type == "cspace_target") {
          return
            all_finite({
              config.cspace_target.metric_scalar,
              config.cspace_target.position_gain,
              config.cspace_target.damping_gain,
              config.cspace_target.robust_position_term_thresh,
              config.cspace_target.inertia}) &&
            config.cspace_target.metric_scalar + config.cspace_target.inertia > 0.0 &&
            config.cspace_target.position_gain >= 0.0 &&
            config.cspace_target.damping_gain >= 0.0 &&
            config.cspace_target.robust_position_term_thresh >= 0.0;
        }
        if (leaf_type == "joint_limit") {
          if (config.joint_limit.policy != "paper_gds") {
            return false;
          }
          for (std::size_t joint = 0; joint < 6; ++joint) {
            PaperGdsJointLimitConfig params;
            params.lower =
              config.joint_lower_limits[joint] +
              config.joint_limit_buffers[joint];
            params.upper =
              config.joint_upper_limits[joint] -
              config.joint_limit_buffers[joint];
            params.center_fraction =
              config.joint_limit.gds_center_fraction;
            params.task_metric =
              config.joint_limit.metric_scalar;
            params.potential_gain =
              config.joint_limit.metric_scalar *
              config.joint_limit.accel_potential_gain;
            params.damping =
              config.joint_limit.metric_scalar *
              config.joint_limit.accel_damper_gain;
            params.boundary_epsilon =
              config.joint_limit.gds_domain_epsilon;
            if (!valid_paper_gds_joint_limit_config(params)) {
              return false;
            }
          }
          return true;
        }
        if (leaf_type == "target") {
          return target_leaf_has_constant_gds_metric(config.target);
        }
        if (leaf_type == "axis_target") {
          return axis_leaf_has_constant_gds_metric(config.axis_target);
        }
        if (leaf_type == "wrist_axis_target") {
          // Its current implementation deliberately overwrites nonlinear
          // task-map curvature with zero, so it is outside this certificate.
          return false;
        }
        if (leaf_type == "collision") {
          if (config.collision.policy != "paper_gds") {
            return false;
          }
          paper_gds_collision::Params params;
          params.metric_scalar = config.collision.metric_scalar;
          params.metric_modulation_radius =
            config.collision.metric_modulation_radius;
          params.metric_exploder_std_dev =
            config.collision.metric_exploder_std_dev;
          params.metric_exploder_eps =
            config.collision.metric_exploder_eps;
          params.clearance_smoothing =
            config.collision.gds_clearance_smoothing;
          params.metric_velocity_floor =
            config.collision.gds_metric_velocity_floor;
          params.metric_velocity_scale =
            config.collision.gds_metric_velocity_scale;
          params.repulsion_gain = config.collision.repulsion_gain;
          params.repulsion_std_dev =
            config.collision.repulsion_std_dev;
          params.damping_gain = config.collision.damping_gain;
          params.damping_std_dev = config.collision.damping_std_dev;
          params.damping_robustness_eps =
            config.collision.damping_robustness_eps;
          params.damping_velocity_scale =
            config.collision.gds_damping_velocity_scale;
          return paper_gds_collision::parameters_are_valid(params);
        }
        if (leaf_type == "damping") {
          return
            all_finite({
              config.damping.metric_scalar,
              config.damping.inertia,
              config.damping.accel_d_gain}) &&
            config.damping.metric_scalar == 0.0 &&
            config.damping.inertia >= 0.0 &&
            config.damping.accel_d_gain >= 0.0;
        }
        // The current velocity-cap and externally supplied canonical leaves
        // do not expose the curvature/potential metadata required by the
        // structured-GDS theorem.
        return false;
      };
    if (
      !leaf_supported(node.leaf_rmp_type) ||
      !leaf_supported(node.handcrafted_leaf_rmp_type))
    {
      return false;
    }
  }
  return true;
}

bool same_environment(
  const std::unordered_map<std::string, Eigen::Vector3d> & previous_targets,
  const std::unordered_map<std::string, Eigen::Vector3d> & current_targets,
  const std::vector<ObstacleSphere> & previous_obstacles,
  const std::vector<ObstacleSphere> & current_obstacles)
{
  if (
    previous_targets.size() != current_targets.size() ||
    previous_obstacles.size() != current_obstacles.size())
  {
    return false;
  }
  for (const auto & target : current_targets) {
    const auto previous = previous_targets.find(target.first);
    if (
      previous == previous_targets.end() ||
      (previous->second - target.second).norm() > 1e-12)
    {
      return false;
    }
  }
  for (std::size_t index = 0; index < current_obstacles.size(); ++index) {
    const auto & previous = previous_obstacles[index];
    const auto & current = current_obstacles[index];
    if (
      previous.source_id != current.source_id ||
      previous.proximity_control_point_index !=
      current.proximity_control_point_index ||
      previous.radius != current.radius ||
      (previous.center - current.center).norm() > 1e-12)
    {
      return false;
    }
  }
  return true;
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
  if (
    config_.collision.policy != "repulsive" &&
    config_.collision.policy != "lula_canonical" &&
    config_.collision.policy != "paper_gds")
  {
    throw std::runtime_error(
            "collision_policy must be repulsive, lula_canonical, or paper_gds");
  }
  if (
    config_.joint_limit.policy != "lula_canonical" &&
    config_.joint_limit.policy != "paper_gds")
  {
    throw std::runtime_error(
            "joint_limit_policy must be lula_canonical or paper_gds");
  }
  const auto & tangent_escape = config_.tangent_escape;
  if (
    tangent_escape.acceleration_model != "canonical_velocity" &&
    tangent_escape.acceleration_model != "risk_damped")
  {
    throw std::runtime_error(
            "tangent_escape acceleration_model must be canonical_velocity or risk_damped");
  }
  if (tangent_escape.enabled && tangent_escape.acceleration_model == "risk_damped") {
    const bool finite =
      std::isfinite(tangent_escape.metric_scalar) &&
      std::isfinite(tangent_escape.clearance_margin) &&
      std::isfinite(tangent_escape.safe_distance) &&
      std::isfinite(tangent_escape.influence_distance) &&
      std::isfinite(tangent_escape.max_accel) &&
      std::isfinite(tangent_escape.max_speed) &&
      std::isfinite(tangent_escape.control_dt) &&
      std::isfinite(tangent_escape.handoff_duration) &&
      std::isfinite(tangent_escape.risk_distance_gain) &&
      std::isfinite(tangent_escape.risk_distance_scale) &&
      std::isfinite(tangent_escape.risk_approach_gain) &&
      std::isfinite(tangent_escape.risk_approach_distance_scale) &&
      std::isfinite(tangent_escape.risk_approach_epsilon) &&
      std::isfinite(tangent_escape.risk_velocity_gate_scale) &&
      std::isfinite(tangent_escape.risk_clearance_rate_filter_time_constant) &&
      std::isfinite(tangent_escape.risk_tangent_damping_gain);
    if (
      !finite ||
      tangent_escape.metric_scalar < 0.0 ||
      tangent_escape.clearance_margin < 0.0 ||
      tangent_escape.safe_distance < 0.0 ||
      tangent_escape.influence_distance <= tangent_escape.safe_distance ||
      tangent_escape.max_accel <= 0.0 ||
      tangent_escape.max_speed <= 0.0 ||
      tangent_escape.control_dt <= 0.0 ||
      tangent_escape.handoff_duration < 0.0 ||
      tangent_escape.risk_distance_gain < 0.0 ||
      tangent_escape.risk_distance_scale <= 0.0 ||
      tangent_escape.risk_approach_gain < 0.0 ||
      tangent_escape.risk_approach_distance_scale <= 0.0 ||
      tangent_escape.risk_approach_epsilon <= 0.0 ||
      tangent_escape.risk_velocity_gate_scale <= 0.0 ||
      tangent_escape.risk_clearance_rate_filter_time_constant < 0.0 ||
      tangent_escape.risk_tangent_damping_gain <= 0.0)
    {
      throw std::runtime_error("Invalid risk-damped Tangent Escape parameters");
    }
  }
  const auto & energy = config_.escape_energy_certificate;
  if (
    !std::isfinite(energy.tank_capacity) ||
    !std::isfinite(energy.initial_energy) ||
    !std::isfinite(energy.control_dt) ||
    !std::isfinite(energy.power_tolerance) ||
    energy.tank_capacity < 0.0 ||
    energy.initial_energy < 0.0 ||
    energy.initial_energy > energy.tank_capacity ||
    energy.control_dt <= 0.0 ||
    energy.scale_search_iterations < 1 ||
    energy.power_tolerance < 0.0)
  {
    throw std::runtime_error("Invalid Escape stability energy-certificate parameters");
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

  struct LeafRootContribution
  {
    Matrix6 metric{Matrix6::Zero()};
    JointVector force{JointVector::Zero()};
  };
  std::unordered_map<int, LeafRootContribution> leaf_root_contributions;

  const auto record_leaf_contribution =
    [&](const RmpNodeConfig & node,
      const std::string & leaf_type,
      const Matrix6 & metric_before,
      const JointVector & force_before) {
      if (
        !config_.enable_leaf_ablation_diagnostics ||
        leaf_type.empty() || leaf_type == "none")
      {
        return;
      }
      const int group_id = leaf_ablation_group_id(node, leaf_type);
      auto & contribution = leaf_root_contributions[group_id];
      contribution.metric += metric - metric_before;
      contribution.force += force - force_before;
    };

  const auto is_tangent_escape_leaf = [](const std::string & leaf_type) {
      return leaf_type == "tangent_escape";
    };

  // Build the complete non-Escape root RMP first. Candidate Escape directions
  // must be evaluated against the same M0/f0 that the final solve will use.
  for (const auto index : impl.topo_indices) {
    const auto & node = config_.graph_nodes[index];
    const auto & geometry = cache.at(node.name);
      const auto accumulate_base_leaf = [&](const std::string & leaf_type) {
        if (is_tangent_escape_leaf(leaf_type)) {
          return;
        }
        const Matrix6 metric_before = metric;
        const JointVector force_before = force;
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
        record_leaf_contribution(
          node,
          leaf_type,
          metric_before,
          force_before);
      };
    accumulate_base_leaf(node.leaf_rmp_type);
    accumulate_base_leaf(node.handcrafted_leaf_rmp_type);
  }

  const Matrix6 base_metric = metric;
  const JointVector base_force = force;

  std::size_t tangent_escape_leaf_count = 0;
  for (const auto index : impl.topo_indices) {
    const auto & node = config_.graph_nodes[index];
    const auto & geometry = cache.at(node.name);
    const auto accumulate_escape_leaf = [&](const std::string & leaf_type) {
        if (!is_tangent_escape_leaf(leaf_type)) {
          return;
        }
        ++tangent_escape_leaf_count;
        if (tangent_escape_leaf_count > 1) {
          throw std::runtime_error(
                  "Only one tangent_escape leaf may be enabled at a time");
        }
        const Matrix6 metric_before = metric;
        const JointVector force_before = force;
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
        record_leaf_contribution(
          node,
          leaf_type,
          metric_before,
          force_before);
      };
    accumulate_escape_leaf(node.leaf_rmp_type);
    accumulate_escape_leaf(node.handcrafted_leaf_rmp_type);
  }

  const Matrix6 requested_escape_metric = metric - base_metric;
  const JointVector requested_escape_force = force - base_force;

  struct ScaledEscapeSolve
  {
    Matrix6 metric{Matrix6::Zero()};
    JointVector force{JointVector::Zero()};
    JointVector raw_qdd{JointVector::Zero()};
    JointVector command_qdd{JointVector::Zero()};
    double escape_power{0.0};
    double solve_power{0.0};
    double clamp_power{0.0};
  };

  const auto solve_scaled_escape =
    [&](double scale) {
      ScaledEscapeSolve sample;
      const double bounded_scale = std::clamp(scale, 0.0, 1.0);
      const Matrix6 escape_metric =
        bounded_scale * requested_escape_metric;
      const JointVector escape_force =
        bounded_scale * requested_escape_force;
      sample.metric = base_metric + escape_metric;
      sample.force = base_force + escape_force;
      sample.raw_qdd = use_rmp2 ?
        resolve_root_rmp2(sample.metric, sample.force, config_.solve_offset) :
        resolve_root_direct(sample.metric, sample.force, config_.solve_offset);
      sample.command_qdd = limit_joint_acceleration(
        sample.raw_qdd,
        config_.max_joint_accel,
        config_.preserve_joint_accel_direction);
      sample.escape_power =
        qd.dot(escape_force - escape_metric * sample.command_qdd);
      sample.solve_power =
        qd.dot(sample.metric * sample.raw_qdd - sample.force);
      sample.clamp_power =
        qd.dot(sample.metric * (sample.command_qdd - sample.raw_qdd));
      return sample;
    };

  auto & energy_state = escape_energy_certificate_state_;
  const auto & energy_params = config_.escape_energy_certificate;
  if (!energy_state.initialized) {
    energy_state.initialized = true;
    energy_state.tank_energy =
      std::clamp(
      energy_params.initial_energy,
      0.0,
      energy_params.tank_capacity);
  }

  const bool environment_static =
    energy_state.previous_environment_valid &&
    external_rmps.empty() &&
    same_environment(
    energy_state.previous_vector_targets,
    vector_targets,
    energy_state.previous_obstacles,
    obstacles);
  energy_state.previous_vector_targets = vector_targets;
  energy_state.previous_obstacles = obstacles;
  energy_state.previous_environment_valid = true;

  const ScaledEscapeSolve requested_solve = solve_scaled_escape(1.0);
  double escape_scale = 1.0;
  ScaledEscapeSolve final_solve = requested_solve;

  if (
    energy_params.enabled &&
    requested_solve.escape_power > 0.0)
  {
    const double positive_power_budget =
      energy_state.tank_energy /
      std::max(energy_params.control_dt, 1e-12);
    if (
      requested_solve.escape_power >
      positive_power_budget)
    {
      double feasible_scale = 0.0;
      double infeasible_scale = 1.0;
      for (
        int iteration = 0;
        iteration < energy_params.scale_search_iterations;
        ++iteration)
      {
        const double trial_scale =
          0.5 * (feasible_scale + infeasible_scale);
        const auto trial = solve_scaled_escape(trial_scale);
        if (
          trial.escape_power <=
          positive_power_budget)
        {
          feasible_scale = trial_scale;
        } else {
          infeasible_scale = trial_scale;
        }
      }
      escape_scale = feasible_scale;
      final_solve = solve_scaled_escape(escape_scale);
      const double required_energy =
        energy_params.control_dt *
        std::max(final_solve.escape_power, 0.0);
      if (
        !std::isfinite(required_energy) ||
        required_energy > energy_state.tank_energy)
      {
        escape_scale = 0.0;
        final_solve = solve_scaled_escape(0.0);
      }
    }
  }

  metric = final_solve.metric;
  force = final_solve.force;
  const JointVector qdd = final_solve.raw_qdd;
  if (config_.enable_leaf_ablation_diagnostics) {
    const int escape_group_id = static_cast<int>(LeafAblationGroup::TangentEscape);
    const auto escape_contribution = leaf_root_contributions.find(escape_group_id);
    if (escape_contribution != leaf_root_contributions.end()) {
      escape_contribution->second.metric *= escape_scale;
      escape_contribution->second.force *= escape_scale;
    }
  }

  const double dt = energy_params.control_dt;
  const double positive_escape_energy =
    dt * std::max(final_solve.escape_power, 0.0);
  const double negative_escape_energy =
    dt * std::max(-final_solve.escape_power, 0.0);
  if (energy_params.enabled) {
    energy_state.tank_energy = std::clamp(
      energy_state.tank_energy - positive_escape_energy,
      0.0,
      energy_params.tank_capacity);
  }
  energy_state.positive_energy_integral += positive_escape_energy;
  energy_state.negative_energy_integral += negative_escape_energy;
  energy_state.net_energy_integral += dt * final_solve.escape_power;
  ++energy_state.sample_count;

  const ScaledEscapeSolve no_escape_solve = solve_scaled_escape(0.0);
  bool base_domain_valid =
    q.allFinite() &&
    qd.allFinite() &&
    base_metric.allFinite() &&
    base_force.allFinite();
  for (const auto & target : vector_targets) {
    base_domain_valid =
      base_domain_valid && target.second.allFinite();
  }
  for (const auto & obstacle : obstacles) {
    base_domain_valid =
      base_domain_valid &&
      obstacle.center.allFinite() &&
      std::isfinite(obstacle.radius) &&
      obstacle.radius >= 0.0;
  }
  for (const auto & entry : cache) {
    const auto & geometry = entry.second;
    base_domain_valid =
      base_domain_valid &&
      geometry.x.allFinite() &&
      geometry.jacobian.allFinite() &&
      geometry.velocity.allFinite() &&
      geometry.curvature.allFinite();
  }
  if (config_.collision.policy == "paper_gds") {
    for (const auto & point : context.control_points) {
      for (const auto & obstacle : obstacles) {
        if ((point.position - obstacle.center).norm() <= 1e-9) {
          base_domain_valid = false;
        }
      }
    }
  }
  for (const auto & node : config_.graph_nodes) {
    if (
      !node.enabled ||
      (
        node.leaf_rmp_type != "axis_target" &&
        node.handcrafted_leaf_rmp_type != "axis_target"))
    {
      continue;
    }
    const auto geometry = cache.find(node.name);
    const auto target = vector_targets.find(node.target_key);
    if (
      geometry == cache.end() ||
      geometry->second.x.norm() <= 1e-9 ||
      target == vector_targets.end() ||
      target->second.norm() <= 1e-9)
    {
      base_domain_valid = false;
    }
  }
  const Matrix6 symmetric_base_metric =
    0.5 * (base_metric + base_metric.transpose());
  Eigen::SelfAdjointEigenSolver<Matrix6> base_metric_eigensolver;
  if (symmetric_base_metric.allFinite()) {
    base_metric_eigensolver.compute(
      symmetric_base_metric,
      Eigen::EigenvaluesOnly);
  }
  const bool base_metric_spd =
    base_metric.allFinite() &&
    base_metric.isApprox(base_metric.transpose(), 1e-10) &&
    base_metric_eigensolver.info() == Eigen::Success &&
    base_metric_eigensolver.eigenvalues().minCoeff() >
    1e-12 * std::max(
    1.0,
    base_metric_eigensolver.eigenvalues().maxCoeff());
  const double base_solve_residual =
    (base_metric * no_escape_solve.raw_qdd - base_force).norm();
  const bool base_solve_valid =
    no_escape_solve.raw_qdd.allFinite() &&
    std::isfinite(base_solve_residual) &&
    base_solve_residual <= 1e-8 * (1.0 + base_force.norm());
  const bool base_config_profile_valid =
    base_graph_has_structured_gds_form(config_);
  const bool base_gds_structural =
    base_config_profile_valid &&
    base_domain_valid &&
    base_metric_spd &&
    base_solve_valid;
  const double numerical_power =
    final_solve.solve_power + final_solve.clamp_power;
  // When the tank is enabled, its update cancels positive Escape power in
  // the composite storage.  Without the tank, retain the uncovered positive
  // part explicitly instead of silently spending the numerical tolerance
  // twice.
  const double conditional_nonincrease_upper_bound =
    numerical_power +
    (
      energy_params.enabled ?
      0.0 :
      std::max(final_solve.escape_power, 0.0));
  const bool conditional_nonincrease_certificate =
    base_gds_structural &&
    environment_static &&
    external_rmps.empty() &&
    conditional_nonincrease_upper_bound <= energy_params.power_tolerance;
  const bool clamp_active =
    (final_solve.command_qdd - final_solve.raw_qdd).norm() > 1e-12;
  const double tank_spent =
    energy_params.initial_energy - energy_state.tank_energy;
  const double guarded_energy_residual =
    energy_params.enabled ?
    tank_spent - energy_state.positive_energy_integral :
    0.0;
  const double guarded_energy_violation =
    energy_params.enabled ?
    std::max(
      energy_state.positive_energy_integral -
      energy_params.initial_energy,
      0.0) :
    0.0;

  // Schema v1.  The accompanying proof document gives the exact field map.
  const std::vector<double> stability_certificate_data{
    1.0,
    base_gds_structural ? 1.0 : 0.0,
    environment_static ? 1.0 : 0.0,
    external_rmps.empty() ? 1.0 : 0.0,
    energy_params.enabled ? 1.0 : 0.0,
    conditional_nonincrease_certificate ? 1.0 : 0.0,
    escape_scale,
    energy_state.tank_energy,
    energy_params.tank_capacity,
    requested_solve.escape_power,
    final_solve.escape_power,
    final_solve.solve_power,
    final_solve.clamp_power,
    numerical_power,
    energy_state.positive_energy_integral,
    energy_state.negative_energy_integral,
    energy_state.net_energy_integral,
    static_cast<double>(energy_state.sample_count),
    requested_escape_metric.trace(),
    (escape_scale * requested_escape_metric).trace(),
    requested_escape_force.norm(),
    (escape_scale * requested_escape_force).norm(),
    final_solve.raw_qdd.norm(),
    final_solve.command_qdd.norm(),
    clamp_active ? 1.0 : 0.0,
    config_.solve_offset,
    requested_escape_metric.norm() > 1e-12 ? 1.0 : 0.0,
    guarded_energy_residual,
    guarded_energy_violation,
    conditional_nonincrease_upper_bound,
    energy_params.initial_energy,
    base_config_profile_valid ? 1.0 : 0.0,
    base_domain_valid ? 1.0 : 0.0,
    base_metric_spd ? 1.0 : 0.0,
    base_solve_valid ? 1.0 : 0.0
  };

  // Same-state counterfactual data.  This is computed from the already-built
  // base M0/f0, so it does not repeat FK, graph pushforward, or leaf
  // evaluation and it does not advance the hybrid Escape supervisor.
  std::vector<double> tangent_escape_dual_solve_data(43, 0.0);
  const bool escape_contribution_active =
    requested_escape_metric.norm() > 1e-12 ||
    requested_escape_force.norm() > 1e-12;
  tangent_escape_dual_solve_data[0] =
    escape_contribution_active ? 1.0 : 0.0;
  for (int index = 0; index < 6; ++index) {
    tangent_escape_dual_solve_data[1 + index] =
      final_solve.command_qdd[index];
    tangent_escape_dual_solve_data[7 + index] =
      no_escape_solve.command_qdd[index];
    tangent_escape_dual_solve_data[13 + index] =
      final_solve.command_qdd[index] - no_escape_solve.command_qdd[index];
  }

  const Eigen::Vector3d tcp_accel_with =
    context.tcp_jacobian * final_solve.command_qdd + context.tcp_curvature;
  const Eigen::Vector3d tcp_accel_without =
    context.tcp_jacobian * no_escape_solve.command_qdd + context.tcp_curvature;
  const Eigen::Vector3d delta_tcp_accel =
    tcp_accel_with - tcp_accel_without;
  for (int axis = 0; axis < 3; ++axis) {
    tangent_escape_dual_solve_data[19 + axis] = tcp_accel_with[axis];
    tangent_escape_dual_solve_data[22 + axis] = tcp_accel_without[axis];
    tangent_escape_dual_solve_data[25 + axis] = delta_tcp_accel[axis];
  }

  if (tangent_escape_debug_data.size() >= 23) {
    const int point_index = static_cast<int>(
      std::llround(tangent_escape_debug_data[1]));
    Eigen::Vector3d normal{
      tangent_escape_debug_data[17],
      tangent_escape_debug_data[18],
      tangent_escape_debug_data[19]};
    Eigen::Vector3d tangent{
      tangent_escape_debug_data[20],
      tangent_escape_debug_data[21],
      tangent_escape_debug_data[22]};
    if (normal.norm() > 1e-12) {
      normal.normalize();
    }
    if (tangent.norm() > 1e-12) {
      tangent.normalize();
    }
    tangent_escape_dual_solve_data[28] = delta_tcp_accel.dot(tangent);
    tangent_escape_dual_solve_data[29] = delta_tcp_accel.dot(normal);
    if (tangent_escape_debug_data.size() > 10) {
      tangent_escape_dual_solve_data[30] = tangent_escape_debug_data[6];
      tangent_escape_dual_solve_data[31] =
        escape_scale * tangent_escape_debug_data[10];
    }
    if (
      point_index >= 0 &&
      static_cast<std::size_t>(point_index) <
      context.control_point_jacobians.size() &&
      static_cast<std::size_t>(point_index) <
      context.control_point_curvatures.size())
    {
      const auto cp_index = static_cast<std::size_t>(point_index);
      const Eigen::Vector3d cp_accel_with =
        context.control_point_jacobians[cp_index] *
        final_solve.command_qdd +
        context.control_point_curvatures[cp_index];
      const Eigen::Vector3d cp_accel_without =
        context.control_point_jacobians[cp_index] *
        no_escape_solve.command_qdd +
        context.control_point_curvatures[cp_index];
      const Eigen::Vector3d delta_cp_accel =
        cp_accel_with - cp_accel_without;
      for (int axis = 0; axis < 3; ++axis) {
        tangent_escape_dual_solve_data[32 + axis] = cp_accel_with[axis];
        tangent_escape_dual_solve_data[35 + axis] = cp_accel_without[axis];
        tangent_escape_dual_solve_data[38 + axis] = delta_cp_accel[axis];
      }
      tangent_escape_dual_solve_data[41] = delta_cp_accel.dot(tangent);
      tangent_escape_dual_solve_data[42] = delta_cp_accel.dot(normal);
    }
  }

  if (config_.tangent_escape.enabled) {
    auto & escape_state =
      config_.tangent_escape.acceleration_model == "risk_damped" ?
      tangent_escape_risk_damped_state_ :
      tangent_escape_canonical_state_;
    escape_state.previous_qdd = qdd;
    escape_state.previous_qdd_valid = true;
  }

  std::vector<double> leaf_ablation_data{0.0};
  if (config_.enable_leaf_ablation_diagnostics) {
    // Schema v1 header (25 values): version, record count, acceleration
    // limit, raw/command norms, direction cosine, saturation count, then
    // raw qdd[6], command qdd[6], and command-minus-raw qdd[6].  Each
    // 17-value record contains group id, active flag, metric trace/norm,
    // force norm, delta raw qdd[6], and delta command qdd[6].
    constexpr double active_tolerance = 1e-12;
    constexpr std::size_t header_size = 25;
    constexpr std::size_t record_size = 17;
    std::vector<int> group_ids;
    group_ids.reserve(leaf_root_contributions.size());
    for (const auto & entry : leaf_root_contributions) {
      group_ids.push_back(entry.first);
    }
    std::sort(group_ids.begin(), group_ids.end());

    const JointVector & raw_qdd = final_solve.raw_qdd;
    const JointVector & command_qdd = final_solve.command_qdd;
    const double raw_norm = raw_qdd.norm();
    const double command_norm = command_qdd.norm();
    double direction_cosine = 1.0;
    if (raw_norm > active_tolerance && command_norm > active_tolerance) {
      direction_cosine = std::clamp(
        raw_qdd.dot(command_qdd) / (raw_norm * command_norm),
        -1.0,
        1.0);
    }
    int saturated_joint_count = 0;
    for (int joint = 0; joint < 6; ++joint) {
      if (std::abs(raw_qdd[joint]) > config_.max_joint_accel + active_tolerance) {
        ++saturated_joint_count;
      }
    }

    leaf_ablation_data.clear();
    leaf_ablation_data.reserve(
      header_size + record_size * group_ids.size());
    leaf_ablation_data.push_back(1.0);
    leaf_ablation_data.push_back(static_cast<double>(group_ids.size()));
    leaf_ablation_data.push_back(config_.max_joint_accel);
    leaf_ablation_data.push_back(raw_norm);
    leaf_ablation_data.push_back(command_norm);
    leaf_ablation_data.push_back(direction_cosine);
    leaf_ablation_data.push_back(static_cast<double>(saturated_joint_count));
    for (int joint = 0; joint < 6; ++joint) {
      leaf_ablation_data.push_back(raw_qdd[joint]);
    }
    for (int joint = 0; joint < 6; ++joint) {
      leaf_ablation_data.push_back(command_qdd[joint]);
    }
    for (int joint = 0; joint < 6; ++joint) {
      leaf_ablation_data.push_back(command_qdd[joint] - raw_qdd[joint]);
    }

    for (const int group_id : group_ids) {
      const auto & contribution = leaf_root_contributions.at(group_id);
      const Matrix6 without_metric = metric - contribution.metric;
      const JointVector without_force = force - contribution.force;
      const JointVector without_raw_qdd = use_rmp2 ?
        resolve_root_rmp2(without_metric, without_force, config_.solve_offset) :
        resolve_root_direct(without_metric, without_force, config_.solve_offset);
      const JointVector without_command_qdd = limit_joint_acceleration(
        without_raw_qdd,
        config_.max_joint_accel,
        config_.preserve_joint_accel_direction);
      const JointVector delta_raw_qdd = raw_qdd - without_raw_qdd;
      const JointVector delta_command_qdd = command_qdd - without_command_qdd;
      const bool active =
        contribution.metric.norm() > active_tolerance ||
        contribution.force.norm() > active_tolerance;

      leaf_ablation_data.push_back(static_cast<double>(group_id));
      leaf_ablation_data.push_back(active ? 1.0 : 0.0);
      leaf_ablation_data.push_back(contribution.metric.trace());
      leaf_ablation_data.push_back(contribution.metric.norm());
      leaf_ablation_data.push_back(contribution.force.norm());
      for (int joint = 0; joint < 6; ++joint) {
        leaf_ablation_data.push_back(delta_raw_qdd[joint]);
      }
      for (int joint = 0; joint < 6; ++joint) {
        leaf_ablation_data.push_back(delta_command_qdd[joint]);
      }
    }
  }

  RmpSolveResult result;
  result.qdd = qdd;
  result.metric = metric;
  result.force = force;
  result.tangent_escape_rmp_data = std::move(tangent_escape_debug_data);
  result.tangent_escape_dual_solve_data =
    std::move(tangent_escape_dual_solve_data);
  result.leaf_ablation_data = std::move(leaf_ablation_data);
  result.stability_certificate_data = stability_certificate_data;
  return result;
}

std::vector<RmpSolveResult> PinocchioDirectRmpSolver::solve_batch(
  const std::vector<RmpBatchInput> & batch_inputs) const
{
  const auto canonical_snapshot = tangent_escape_canonical_state_;
  const auto risk_damped_snapshot = tangent_escape_risk_damped_state_;
  const auto energy_snapshot = escape_energy_certificate_state_;
  const auto restore_state = [
      this,
      &canonical_snapshot,
      &risk_damped_snapshot,
      &energy_snapshot]() {
      tangent_escape_canonical_state_ = canonical_snapshot;
      tangent_escape_risk_damped_state_ = risk_damped_snapshot;
      escape_energy_certificate_state_ = energy_snapshot;
    };

  std::vector<RmpSolveResult> results;
  results.reserve(batch_inputs.size());
  try {
    for (const auto & input : batch_inputs) {
      restore_state();
      results.push_back(solve(
          input.q,
          input.qd,
          input.vector_targets,
          input.obstacles,
          input.external_rmps));
    }
  } catch (...) {
    restore_state();
    throw;
  }

  restore_state();
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
    const bool use_signed_collision_task =
      collision_params.policy == "paper_gds";
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
        const double x = use_signed_collision_task ?
          signed_distance :
          std::max(signed_distance, 0.0);
        const Eigen::RowVectorXd jacobian =
          (delta / center_distance).transpose() * point_jacobian;
        const double velocity = (delta / center_distance).dot(point_velocity);
        double curvature = 0.0;
        if (use_signed_collision_task || signed_distance > 0.0) {
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
          (use_signed_collision_task || signed_distance > 0.0) ? velocity : 0.0,
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
          const double x = use_signed_collision_task ?
            signed_distance :
            std::max(signed_distance, 0.0);
          const Eigen::RowVectorXd jacobian =
            (delta / center_distance).transpose() * point_jacobian;
          const double velocity = (delta / center_distance).dot(point_velocity);
          double curvature = 0.0;
          if (use_signed_collision_task || signed_distance > 0.0) {
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
            best_velocity =
              (use_signed_collision_task || signed_distance > 0.0) ? velocity : 0.0;
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
          const double x = use_signed_collision_task ?
            signed_distance :
            std::max(signed_distance, 0.0);
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
            best_velocity =
              (use_signed_collision_task || signed_distance > 0.0) ? velocity : 0.0;
            best_curvature =
              (use_signed_collision_task || signed_distance > 0.0) ? curvature : 0.0;
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

void PinocchioDirectRmpSolver::accumulate_scalar_natural_leaf(
  const RowVector6 & jacobian,
  double metric_scalar,
  double natural_force,
  double curvature,
  Matrix6 & metric,
  JointVector & force)
{
  metric.noalias() +=
    jacobian.transpose() * metric_scalar * jacobian;
  force.noalias() +=
    jacobian.transpose() *
    (natural_force - metric_scalar * curvature);
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
      const auto root_it = cache.find("root");
      if (root_it == cache.end() || root_it->second.x.size() != 6) {
        throw std::runtime_error("tangent_escape requires the six-dimensional root state");
      }
      const JointVector q = root_it->second.x.head<6>();
      accumulate_tangent_escape(
        q,
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
  const auto velocity = velocity_of(geometry, qd);
  if (config_.joint_limit.policy == "paper_gds") {
    if (geometry.x.size() != 12 || geometry.jacobian.rows() != 12) {
      throw std::runtime_error(
              "paper_gds joint-limit leaf expects six upper/lower clearance pairs");
    }
    for (int joint = 0; joint < 6; ++joint) {
      const Eigen::Index upper_row = 2 * joint;
      const Eigen::Index lower_row = upper_row + 1;
      const double lower =
        config_.joint_lower_limits[static_cast<std::size_t>(joint)] +
        config_.joint_limit_buffers[static_cast<std::size_t>(joint)];
      const double upper =
        config_.joint_upper_limits[static_cast<std::size_t>(joint)] -
        config_.joint_limit_buffers[static_cast<std::size_t>(joint)];
      PaperGdsJointLimitConfig params;
      params.lower = lower;
      params.upper = upper;
      params.center_fraction =
        config_.joint_limit.gds_center_fraction;
      params.task_metric =
        config_.joint_limit.metric_scalar;
      // Preserve the YAML gains as task-acceleration gains.  The structured
      // GDS helper takes potential and damping *force* coefficients.
      params.potential_gain =
        config_.joint_limit.metric_scalar *
        config_.joint_limit.accel_potential_gain;
      params.damping =
        config_.joint_limit.metric_scalar *
        config_.joint_limit.accel_damper_gain;
      params.boundary_epsilon =
        config_.joint_limit.gds_domain_epsilon;

      const double q_value = lower + geometry.x[lower_row];
      const double q_rate = velocity[lower_row];
      const auto leaf =
        evaluate_paper_gds_joint_limit(params, q_value, q_rate);
      if (!leaf.valid) {
        throw std::runtime_error(
                "paper_gds joint-limit task left its open numerical domain at joint " +
                std::to_string(joint));
      }

      const double lower_clearance = geometry.x[lower_row];
      const double upper_clearance = geometry.x[upper_row];
      const RowVector6 logit_jacobian =
        geometry.jacobian.row(lower_row) / lower_clearance -
        geometry.jacobian.row(upper_row) / upper_clearance;
      const double logit_curvature =
        geometry.curvature[lower_row] / lower_clearance -
        velocity[lower_row] * velocity[lower_row] /
        (lower_clearance * lower_clearance) -
        geometry.curvature[upper_row] / upper_clearance +
        velocity[upper_row] * velocity[upper_row] /
        (upper_clearance * upper_clearance);

      accumulate_scalar_natural_leaf(
        logit_jacobian,
        leaf.task_inertia,
        leaf.task_natural_force,
        logit_curvature,
        metric,
        force);
    }
    return;
  }

  const bool use_natural_rmp = uses_rmp2_solve() && uses_natural_rmp();
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
  const auto velocity = velocity_of(geometry, qd);
  if (config_.collision.policy == "paper_gds") {
    paper_gds_collision::Params params;
    params.metric_scalar = config_.collision.metric_scalar;
    params.metric_modulation_radius =
      config_.collision.metric_modulation_radius;
    params.metric_exploder_std_dev =
      config_.collision.metric_exploder_std_dev;
    params.metric_exploder_eps =
      config_.collision.metric_exploder_eps;
    params.clearance_smoothing =
      config_.collision.gds_clearance_smoothing;
    params.metric_velocity_floor =
      config_.collision.gds_metric_velocity_floor;
    params.metric_velocity_scale =
      config_.collision.gds_metric_velocity_scale;
    params.repulsion_gain = config_.collision.repulsion_gain;
    params.repulsion_std_dev =
      config_.collision.repulsion_std_dev;
    params.damping_gain = config_.collision.damping_gain;
    params.damping_std_dev = config_.collision.damping_std_dev;
    params.damping_robustness_eps =
      config_.collision.damping_robustness_eps;
    params.damping_velocity_scale =
      config_.collision.gds_damping_velocity_scale;

    for (Eigen::Index row = 0; row < geometry.x.size(); ++row) {
      const auto leaf = paper_gds_collision::evaluate(
        geometry.x[row], velocity[row], params);
      if (!leaf.valid) {
        throw std::runtime_error(
                "paper_gds collision leaf violated its GDS validity conditions");
      }
      accumulate_scalar_natural_leaf(
        geometry.jacobian.row(row),
        leaf.M,
        leaf.natural_force,
        geometry.curvature[row],
        metric,
        force);
    }
    return;
  }

  const bool use_natural_rmp = uses_rmp2_solve() && uses_natural_rmp();
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
  const JointVector & q,
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
  const Matrix6 base_metric = metric;
  const JointVector base_force = force;
  if (config_.tangent_escape.acceleration_model == "risk_damped") {
    accumulate_tangent_escape_risk_damped(
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
    return;
  }
  if (config_.tangent_escape.acceleration_model == "canonical_velocity") {
    accumulate_tangent_escape_canonical(
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
    return;
  }
  throw std::runtime_error(
          "Unsupported tangent Escape acceleration model: " +
          config_.tangent_escape.acceleration_model);
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
