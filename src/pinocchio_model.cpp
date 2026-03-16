#include "rb10_rmpflow_rviz/pinocchio_model.hpp"

#include <stdexcept>

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/parsers/urdf.hpp>

namespace rb10_rmpflow_rviz
{

namespace
{

Eigen::Matrix<double, 3, 6> linear_jacobian(
  const pinocchio::Model & model,
  pinocchio::Data & data,
  pinocchio::FrameIndex frame_id)
{
  pinocchio::Data::Matrix6x jacobian(6, model.nv);
  jacobian.setZero();
  pinocchio::getFrameJacobian(
    model,
    data,
    frame_id,
    pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
    jacobian);
  return jacobian.topRows<3>();
}

Eigen::Matrix<double, 3, 6> angular_jacobian(
  const pinocchio::Model & model,
  pinocchio::Data & data,
  pinocchio::FrameIndex frame_id)
{
  pinocchio::Data::Matrix6x jacobian(6, model.nv);
  jacobian.setZero();
  pinocchio::getFrameJacobian(
    model,
    data,
    frame_id,
    pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
    jacobian);
  return jacobian.bottomRows<3>();
}

Eigen::Vector3d frame_linear_velocity(
  const pinocchio::Model & model,
  pinocchio::Data & data,
  pinocchio::FrameIndex frame_id)
{
  return pinocchio::getFrameVelocity(
    model,
    data,
    frame_id,
    pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED).linear();
}

Eigen::Vector3d frame_linear_curvature(
  const pinocchio::Model & model,
  pinocchio::Data & data,
  pinocchio::FrameIndex frame_id)
{
  return pinocchio::getFrameClassicalAcceleration(
    model,
    data,
    frame_id,
    pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED).linear();
}

}  // namespace

PinocchioModel::PinocchioModel(const std::string & urdf_path)
{
  pinocchio::urdf::buildModel(urdf_path, model_);
  for (std::size_t index = 0; index < RB10Model::link_names.size(); ++index) {
    const auto frame_id = model_.getFrameId(RB10Model::link_names[index]);
    if (frame_id == static_cast<pinocchio::FrameIndex>(model_.frames.size())) {
      throw std::runtime_error(
              "Pinocchio frame not found in URDF: " + std::string(RB10Model::link_names[index]));
    }
    frame_ids_[index] = frame_id;
  }

  for (std::size_t index = 0; index < lower_limits_.size(); ++index) {
    lower_limits_[index] = model_.lowerPositionLimit[static_cast<Eigen::Index>(index)];
    upper_limits_[index] = model_.upperPositionLimit[static_cast<Eigen::Index>(index)];
  }
}

KinematicsContext PinocchioModel::forward_context(
  const RB10Model::JointVector & q,
  const RB10Model::JointVector & qd) const
{
  KinematicsContext context;
  pinocchio::Data data(model_);
  pinocchio::forwardKinematics(model_, data, q, qd, RB10Model::JointVector::Zero());
  pinocchio::computeJointJacobians(model_, data, q);
  pinocchio::updateFramePlacements(model_, data);

  for (std::size_t index = 0; index < RB10Model::LINK_COUNT; ++index) {
    const auto frame_id = frame_ids_[index];
    context.link_positions[index] = data.oMf[frame_id].translation();
    context.link_rotations[index] = data.oMf[frame_id].rotation();
    context.link_jacobians[index] = linear_jacobian(model_, data, frame_id);
    context.link_angular_jacobians[index] = angular_jacobian(model_, data, frame_id);
    context.link_velocities[index] = frame_linear_velocity(model_, data, frame_id);
    context.link_curvatures[index] = frame_linear_curvature(model_, data, frame_id);
    context.link_angular_velocities[index] = pinocchio::getFrameVelocity(
      model_, data, frame_id, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED).angular();
    context.link_angular_curvatures[index] = pinocchio::getFrameClassicalAcceleration(
      model_, data, frame_id, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED).angular();
  }

  context.tcp_position = context.link_positions[RB10Model::TCP];
  context.tcp_jacobian = context.link_jacobians[RB10Model::TCP];
  context.tcp_velocity = context.link_velocities[RB10Model::TCP];
  context.tcp_curvature = context.link_curvatures[RB10Model::TCP];

  context.control_points.reserve(33);
  context.control_point_jacobians.reserve(33);
  context.control_point_velocities.reserve(33);
  context.control_point_curvatures.reserve(33);
  for (const auto & spec : RB10Model::control_point_specs) {
    const auto & start = context.link_positions[static_cast<std::size_t>(spec.start_link)];
    const auto & end = context.link_positions[static_cast<std::size_t>(spec.end_link)];
    const auto & start_jacobian =
      context.link_jacobians[static_cast<std::size_t>(spec.start_link)];
    const auto & end_jacobian =
      context.link_jacobians[static_cast<std::size_t>(spec.end_link)];
    const auto & start_velocity =
      context.link_velocities[static_cast<std::size_t>(spec.start_link)];
    const auto & end_velocity =
      context.link_velocities[static_cast<std::size_t>(spec.end_link)];
    const auto & start_curvature =
      context.link_curvatures[static_cast<std::size_t>(spec.start_link)];
    const auto & end_curvature =
      context.link_curvatures[static_cast<std::size_t>(spec.end_link)];
    for (int index = 0; index < spec.interpolation_points; ++index) {
      const double alpha =
        static_cast<double>(index + 1) / static_cast<double>(spec.interpolation_points);
      context.control_points.push_back(ControlPoint{
        (1.0 - alpha) * start + alpha * end,
        spec.radius
      });
      context.control_point_jacobians.push_back(
        (1.0 - alpha) * start_jacobian + alpha * end_jacobian);
      context.control_point_velocities.push_back(
        (1.0 - alpha) * start_velocity + alpha * end_velocity);
      context.control_point_curvatures.push_back(
        (1.0 - alpha) * start_curvature + alpha * end_curvature);
    }
  }

  return context;
}

}  // namespace rb10_rmpflow_rviz
