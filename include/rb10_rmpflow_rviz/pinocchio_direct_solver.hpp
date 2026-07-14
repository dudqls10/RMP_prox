#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

#include "rb10_rmpflow_rviz/pinocchio_model.hpp"
#include "rb10_rmpflow_rviz/rmp_solver_interface.hpp"

namespace rb10_rmpflow_rviz
{

class PinocchioDirectRmpSolver : public RmpSolverInterface
{
public:
  using JointVector = RB10Model::JointVector;
  using Matrix6 = Eigen::Matrix<double, 6, 6>;
  using RowVector6 = Eigen::Matrix<double, 1, 6>;

  PinocchioDirectRmpSolver(EigenRmpConfig config, std::string urdf_path);

  RmpSolveResult solve(
    const JointVector & q,
    const JointVector & qd,
    const std::unordered_map<std::string, Eigen::Vector3d> & vector_targets,
    const std::vector<ObstacleSphere> & obstacles,
    const std::unordered_map<std::string, ExternalRmpFeature> & external_rmps = {}) const override;

  std::vector<RmpSolveResult> solve_batch(
    const std::vector<RmpBatchInput> & batch_inputs) const override;

public:
  struct NodeGeometry
  {
    Eigen::VectorXd x;
    Eigen::MatrixXd jacobian;
    Eigen::VectorXd velocity;
    Eigen::VectorXd curvature;
  };

private:
  static double sigmoid(double value);
  static double collision_scalar_acceleration(
    double clearance,
    double clearance_rate,
    const CollisionRmpParams & params);

  static void accumulate_scalar_leaf(
    bool use_natural_rmp,
    const RowVector6 & jacobian,
    double metric_scalar,
    double acceleration,
    double curvature,
    Matrix6 & metric,
    JointVector & force);

  static void accumulate_vector_leaf(
    bool use_natural_rmp,
    const Eigen::MatrixXd & jacobian,
    const Eigen::MatrixXd & leaf_metric,
    const Eigen::VectorXd & acceleration,
    const Eigen::VectorXd & curvature,
    Matrix6 & metric,
    JointVector & force);

  static Eigen::VectorXd make_default_q(const EigenRmpConfig & config);
  static Eigen::VectorXd velocity_of(const NodeGeometry & geometry, const JointVector & qd);
  static int control_point_count();
  static bool uses_casadi_task_map(const std::string & task_map_type);
  bool uses_rmp2_solve() const;
  bool uses_natural_rmp() const;
  static Eigen::Vector3d axis_unit_vector(const std::string & axis_name);
  static Eigen::VectorXd flatten_control_points(const KinematicsContext & context);
  static Eigen::MatrixXd stack_control_point_jacobians(const KinematicsContext & context);
  static constexpr std::size_t tangent_escape_softmax_candidate_count_ = 5;

  struct TangentEscapeGdsModeState
  {
    bool active{false};
    Eigen::Vector3d origin{Eigen::Vector3d::Zero()};
    Eigen::Vector3d tangent{Eigen::Vector3d::UnitX()};
    Eigen::Vector3d obstacle_direction{Eigen::Vector3d::UnitX()};
    double activation{0.0};
    double branch_weight{0.0};
    double goal_score{0.0};
    double continuity_score{0.0};
    double duplicate_risk{0.0};
    double adjacent_risk{0.0};
    double hold_bonus{0.0};
    double blocked_penalty{0.0};
    double base_score{0.0};
    double score{0.0};
    double metric_boost{1.0};
    double accel_boost{1.0};
    std::uint64_t generation{0};
    int supervisor_mode{0};
    bool hold_phase{false};
  };

  struct TangentEscapeSupervisorState
  {
    bool active{false};
    int mode{0};
    std::size_t control_point_index{0};
    std::size_t slot{0};
    Eigen::Vector3d tangent{Eigen::Vector3d::UnitX()};
    double branch_age_s{0.0};
    double hold_age_s{0.0};
    double stuck_timer_s{0.0};
    double recovery_timer_s{0.0};
    double start_scalar_s{0.0};
    double best_scalar_s{0.0};
    double start_clearance{0.0};
    double best_clearance{0.0};
  };

  static int infer_node_dim(
    const RmpNodeConfig & node,
    const std::unordered_map<std::string, int> & node_dims);
  bool node_enabled(const std::string & name) const;
  NodeGeometry evaluate_node(
    const RmpNodeConfig & node,
    const JointVector & q,
    const KinematicsContext & context,
    const std::vector<ObstacleSphere> & obstacles,
    std::unordered_map<std::string, NodeGeometry> & cache) const;
  NodeGeometry evaluate_native_node(
    const RmpNodeConfig & node,
    const JointVector & q,
    const KinematicsContext & context,
    const std::vector<ObstacleSphere> & obstacles,
    const std::vector<NodeGeometry> & parents) const;
  NodeGeometry evaluate_casadi_node(
    const RmpNodeConfig & node,
    const std::vector<NodeGeometry> & parents) const;

  void accumulate_cspace_target(
    const NodeGeometry & geometry,
    const JointVector & qd,
    Matrix6 & metric,
    JointVector & force) const;

  void accumulate_joint_limits(
    const NodeGeometry & geometry,
    const JointVector & qd,
    Matrix6 & metric,
    JointVector & force) const;

  void accumulate_joint_velocity_cap(
    const NodeGeometry & geometry,
    const JointVector & qd,
    Matrix6 & metric,
    JointVector & force) const;

  void accumulate_target(
    const NodeGeometry & geometry,
    const JointVector & qd,
    const Eigen::Vector3d & goal,
    Matrix6 & metric,
    JointVector & force) const;

  void accumulate_axis_target(
    const NodeGeometry & geometry,
    const JointVector & qd,
    const Eigen::Vector3d & goal,
    const Eigen::Vector3d * current_position,
    const Eigen::Vector3d * position_goal,
    Matrix6 & metric,
    JointVector & force) const;

  void accumulate_wrist_axis_target(
    const NodeGeometry & geometry,
    const JointVector & qd,
    const Eigen::Vector3d & goal,
    const Eigen::Vector3d * current_position,
    const Eigen::Vector3d * position_goal,
    Matrix6 & metric,
    JointVector & force) const;

  void accumulate_collision(
    const NodeGeometry & geometry,
    const JointVector & qd,
    Matrix6 & metric,
    JointVector & force) const;

  void accumulate_tangent_escape(
    const KinematicsContext & context,
    const NodeGeometry & geometry,
    const JointVector & qd,
    const Eigen::Vector3d & goal,
    const std::vector<ObstacleSphere> & obstacles,
    const JointVector & nominal_qdd,
    Matrix6 & metric,
    JointVector & force,
    std::vector<double> * debug_data) const;

  void accumulate_joint_damping(
    const NodeGeometry & geometry,
    const JointVector & qd,
    Matrix6 & metric,
    JointVector & force) const;
  void accumulate_external(
    const RmpNodeConfig & node,
    const NodeGeometry & geometry,
    const std::unordered_map<std::string, ExternalRmpFeature> & external_rmps,
    Matrix6 & metric,
    JointVector & force) const;
  void accumulate_leaf_type(
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
    std::vector<double> * tangent_escape_debug_data) const;
  JointVector compute_nominal_joint_acceleration(
    const JointVector & qd,
    const std::unordered_map<std::string, NodeGeometry> & cache,
    const KinematicsContext & context,
    const std::unordered_map<std::string, Eigen::Vector3d> & vector_targets,
    const std::unordered_map<std::string, ExternalRmpFeature> & external_rmps) const;

  EigenRmpConfig config_;
  std::shared_ptr<const PinocchioModel> model_;
  mutable std::array<
    TangentEscapeGdsModeState,
    RB10Model::sensor_control_points.size()> tangent_escape_gds_modes_{};
  mutable std::array<
    std::array<TangentEscapeGdsModeState, tangent_escape_softmax_candidate_count_>,
    RB10Model::sensor_control_points.size()> tangent_escape_softmax_gds_modes_{};
  mutable TangentEscapeSupervisorState tangent_escape_supervisor_{};
  mutable std::array<
    std::array<double, tangent_escape_softmax_candidate_count_>,
    RB10Model::sensor_control_points.size()> tangent_escape_blocked_memory_{};
  mutable std::uint64_t tangent_escape_mode_generation_{0};
  std::shared_ptr<const void> compiled_state_;
};

}  // namespace rb10_rmpflow_rviz
