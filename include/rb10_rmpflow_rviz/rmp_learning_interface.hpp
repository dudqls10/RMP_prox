#pragma once

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

#include "rb10_rmpflow_rviz/rmp_solver_interface.hpp"

namespace rb10_rmpflow_rviz
{

struct LearningObservation
{
  RB10Model::JointVector q{RB10Model::JointVector::Zero()};
  RB10Model::JointVector qd{RB10Model::JointVector::Zero()};
  std::unordered_map<std::string, Eigen::Vector3d> vector_targets;
  std::vector<ObstacleSphere> obstacles;
};

struct ExternalFeatureSpec
{
  std::string key;
  int dim{0};
  std::vector<std::string> node_names;
};

class ResidualPolicyInterface
{
public:
  virtual ~ResidualPolicyInterface() = default;

  virtual std::unordered_map<std::string, ExternalRmpFeature> infer(
    const LearningObservation & observation,
    const std::vector<ExternalFeatureSpec> & specs) const = 0;

  virtual std::vector<std::unordered_map<std::string, ExternalRmpFeature>> infer_batch(
    const std::vector<LearningObservation> & observations,
    const std::vector<ExternalFeatureSpec> & specs) const;
};

class NullResidualPolicy : public ResidualPolicyInterface
{
public:
  std::unordered_map<std::string, ExternalRmpFeature> infer(
    const LearningObservation & observation,
    const std::vector<ExternalFeatureSpec> & specs) const override;
};

class LearningReadyRmpAdapter
{
public:
  LearningReadyRmpAdapter(
    const RmpSolverInterface & solver,
    EigenRmpConfig config);

  const std::vector<ExternalFeatureSpec> & external_feature_specs() const
  {
    return external_feature_specs_;
  }

  RmpSolveResult rollout_single(
    const LearningObservation & observation,
    const std::unordered_map<std::string, ExternalRmpFeature> & residuals = {}) const;

  std::vector<RmpBatchInput> make_batch_inputs(
    const std::vector<LearningObservation> & observations,
    const std::vector<std::unordered_map<std::string, ExternalRmpFeature>> & residual_batches = {}) const;

  std::vector<RmpSolveResult> rollout_batch(
    const std::vector<LearningObservation> & observations,
    const ResidualPolicyInterface * policy = nullptr) const;

private:
  std::reference_wrapper<const RmpSolverInterface> solver_;
  EigenRmpConfig config_;
  std::vector<ExternalFeatureSpec> external_feature_specs_;
};

}  // namespace rb10_rmpflow_rviz
