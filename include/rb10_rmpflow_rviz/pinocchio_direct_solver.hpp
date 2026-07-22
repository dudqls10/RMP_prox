#pragma once

#include <array>
#include <cstdint>
#include <limits>
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

  static void accumulate_scalar_natural_leaf(
    const RowVector6 & jacobian,
    double metric_scalar,
    double natural_force,
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
  static constexpr std::size_t tangent_escape_failure_memory_count_ = 4;

  enum class TangentEscapeCanonicalPhase : int
  {
    Off = 0,
    Engage = 1,
    Drive = 2,
    ReleaseDriveDown = 3,
    ReleaseBrake = 4,
    ReleaseLambdaDown = 5,
    ReselectDriveDown = 6,
    ReselectBrake = 7,
    ReselectLambdaDown = 8
  };

  struct TangentEscapeFailureMemory
  {
    bool valid{false};
    Eigen::Vector3d tangent{Eigen::Vector3d::UnitX()};
    double strength{0.0};
  };

  struct TangentEscapeClearanceRateState
  {
    std::size_t control_point_index{0};
    std::int64_t obstacle_key{-1};
    double previous_clearance{0.0};
    double filtered_rate{0.0};
    bool previous_clearance_valid{false};
    bool seen_this_cycle{false};
  };

  struct TangentEscapeCanonicalState
  {
    TangentEscapeCanonicalPhase phase{TangentEscapeCanonicalPhase::Off};
    bool active_pair_valid{false};
    std::size_t control_point_index{0};
    std::int64_t obstacle_key{-1};
    bool pending_pair_valid{false};
    std::size_t pending_control_point_index{0};
    std::int64_t pending_obstacle_key{-1};
    bool tangent_valid{false};
    Eigen::Vector3d tangent{Eigen::Vector3d::UnitX()};
    Eigen::Vector3d obstacle_direction_at_selection{Eigen::Vector3d::UnitX()};
    bool pending_failure_memory{false};
    bool force_direction_change{false};
    int handoff_reason{0};
    double current_score{-std::numeric_limits<double>::infinity()};
    double z{0.0};
    double lambda{0.0};
    double drive_ramp{0.0};
    double release_brake{0.0};
    double desired_velocity{0.0};
    double phase_elapsed_s{0.0};
    double phase_start_lambda{0.0};
    double phase_start_drive_ramp{0.0};
    double phase_start_release_brake{0.0};
    double active_age_s{0.0};
    double previous_goal_error{0.0};
    bool previous_goal_error_valid{false};
    Eigen::Vector3d previous_goal{Eigen::Vector3d::Zero()};
    bool previous_goal_valid{false};
    double filtered_goal_progress{0.0};
    double last_alpha_stuck{0.0};
    double last_raw_activation{0.0};
    double last_blockage{0.0};
    double last_clearance{0.0};
    double last_beta{0.0};
    double command_distance{0.0};
    double actual_distance{0.0};
    double episode_start_sector_risk{0.0};
    double last_sector_risk{0.0};
    std::array<
      TangentEscapeFailureMemory,
      tangent_escape_failure_memory_count_> failure_memory{};
    std::size_t failure_memory_cursor{0};
    JointVector previous_qdd{JointVector::Zero()};
    bool previous_qdd_valid{false};
    std::uint64_t handoff_generation{0};
    std::vector<TangentEscapeClearanceRateState> clearance_rate_states{};
  };

  struct EscapeEnergyCertificateState
  {
    bool initialized{false};
    double tank_energy{0.0};
    double positive_energy_integral{0.0};
    double negative_energy_integral{0.0};
    double net_energy_integral{0.0};
    std::uint64_t sample_count{0};
    bool previous_environment_valid{false};
    std::unordered_map<std::string, Eigen::Vector3d> previous_vector_targets;
    std::vector<ObstacleSphere> previous_obstacles;
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
    const JointVector & q,
    const KinematicsContext & context,
    const NodeGeometry & geometry,
    const JointVector & qd,
    const Eigen::Vector3d & goal,
    const std::vector<ObstacleSphere> & obstacles,
    const JointVector & nominal_qdd,
    Matrix6 & metric,
    JointVector & force,
    std::vector<double> * debug_data) const;

  void accumulate_tangent_escape_canonical(
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
    std::vector<double> * debug_data) const;

  void accumulate_tangent_escape_risk_damped(
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
    std::vector<double> * debug_data) const;

  void accumulate_tangent_escape_impl(
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
  mutable TangentEscapeCanonicalState tangent_escape_canonical_state_{};
  mutable TangentEscapeCanonicalState tangent_escape_risk_damped_state_{};
  mutable EscapeEnergyCertificateState escape_energy_certificate_state_{};
  std::shared_ptr<const void> compiled_state_;
};

}  // namespace rb10_rmpflow_rviz
