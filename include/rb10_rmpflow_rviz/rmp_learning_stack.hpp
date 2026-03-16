#pragma once

#include <cstddef>
#include <functional>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

#include "rb10_rmpflow_rviz/rmp_learning_interface.hpp"

namespace rb10_rmpflow_rviz
{

struct ResidualSupervisionSample
{
  LearningObservation observation;
  std::unordered_map<std::string, ExternalRmpFeature> target_residuals;
};

struct ResidualStepResult
{
  LearningObservation observation;
  RmpSolveResult solve_result;
  double reward{0.0};
  bool done{false};
  std::size_t step{0};
};

struct ResidualEnvConfig
{
  double dt{0.01};
  std::size_t max_steps{512};
  double acceleration_penalty{1e-3};
  double velocity_penalty{1e-2};
  std::function<double(
    const LearningObservation & current,
    const RmpSolveResult & solve_result,
    const LearningObservation & next)> reward_function;
  std::function<bool(const LearningObservation & next, std::size_t step)> terminal_function;
};

struct SupervisedBatchStats
{
  double loss{0.0};
  Eigen::VectorXd gradient;
};

class TrainableResidualPolicy : public ResidualPolicyInterface
{
public:
  virtual Eigen::VectorXd parameters() const = 0;
  virtual void set_parameters(const Eigen::VectorXd & parameters) = 0;
};

class MlpResidualPolicy : public TrainableResidualPolicy
{
public:
  MlpResidualPolicy(
    std::vector<ExternalFeatureSpec> specs,
    std::vector<std::string> target_keys,
    int hidden_dim = 64,
    int max_obstacles = 4,
    unsigned int seed = 7u);

  int input_dim() const
  {
    return input_dim_;
  }

  int output_dim() const
  {
    return output_dim_;
  }

  const std::vector<std::string> & target_keys() const
  {
    return target_keys_;
  }

  std::unordered_map<std::string, ExternalRmpFeature> infer(
    const LearningObservation & observation,
    const std::vector<ExternalFeatureSpec> & specs) const override;

  Eigen::VectorXd parameters() const override;
  void set_parameters(const Eigen::VectorXd & parameters) override;

  Eigen::VectorXd encode_observation(const LearningObservation & observation) const;

  Eigen::VectorXd forward_raw(const Eigen::VectorXd & input) const;

  std::unordered_map<std::string, ExternalRmpFeature> decode_output(
    const Eigen::VectorXd & output,
    const std::vector<ExternalFeatureSpec> & specs) const;

  Eigen::VectorXd flatten_targets(
    const std::unordered_map<std::string, ExternalRmpFeature> & targets) const;

  SupervisedBatchStats supervised_batch_stats(
    const std::vector<ResidualSupervisionSample> & samples) const;

private:
  static double softplus(double x);
  static double sigmoid(double x);

  std::vector<ExternalFeatureSpec> specs_;
  std::vector<std::string> target_keys_;
  int hidden_dim_{64};
  int max_obstacles_{4};
  int input_dim_{0};
  int output_dim_{0};
  double min_metric_diag_{1e-3};

  Eigen::MatrixXd w1_;
  Eigen::VectorXd b1_;
  Eigen::MatrixXd w2_;
  Eigen::VectorXd b2_;
};

class SgdOptimizer
{
public:
  double learning_rate{1e-3};
  double weight_decay{0.0};
  double gradient_clip_norm{0.0};

  void step(TrainableResidualPolicy & policy, const Eigen::VectorXd & gradient) const;
};

class ResidualSupervisedTrainer
{
public:
  static double train_epoch(
    MlpResidualPolicy & policy,
    const std::vector<ResidualSupervisionSample> & samples,
    const SgdOptimizer & optimizer,
    std::size_t batch_size = 32);
};

class ResidualRmpEnv
{
public:
  ResidualRmpEnv(
    const LearningReadyRmpAdapter & adapter,
    ResidualEnvConfig config = {});

  void reset(const LearningObservation & initial_observation);

  const LearningObservation & observation() const
  {
    return observation_;
  }

  std::size_t step_count() const
  {
    return step_;
  }

  ResidualStepResult step(
    const std::unordered_map<std::string, ExternalRmpFeature> & residuals = {});

  ResidualStepResult step(const ResidualPolicyInterface & policy);

private:
  const LearningReadyRmpAdapter & adapter_;
  ResidualEnvConfig config_;
  LearningObservation observation_;
  std::size_t step_{0};
  bool initialized_{false};
};

}  // namespace rb10_rmpflow_rviz
