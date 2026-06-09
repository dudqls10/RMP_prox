#pragma once

#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

#include "rb10_rmpflow_rviz/rmp_eigen_solver.hpp"

namespace rb10_rmpflow_rviz
{

struct RmpBatchInput
{
  RB10Model::JointVector q{RB10Model::JointVector::Zero()};
  RB10Model::JointVector qd{RB10Model::JointVector::Zero()};
  std::unordered_map<std::string, Eigen::Vector3d> vector_targets;
  std::vector<ObstacleSphere> obstacles;
  std::unordered_map<std::string, ExternalRmpFeature> external_rmps;
};

class RmpSolverInterface
{
public:
  using JointVector = RB10Model::JointVector;

  virtual ~RmpSolverInterface() = default;

  virtual RmpSolveResult solve(
    const JointVector & q,
    const JointVector & qd,
    const std::unordered_map<std::string, Eigen::Vector3d> & vector_targets,
    const std::vector<ObstacleSphere> & obstacles,
    const std::unordered_map<std::string, ExternalRmpFeature> & external_rmps = {}) const = 0;

  virtual std::vector<RmpSolveResult> solve_batch(
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
};

}  // namespace rb10_rmpflow_rviz
