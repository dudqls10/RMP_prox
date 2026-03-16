#include "rb10_rmpflow_rviz/rmp_learning_stack.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace rb10_rmpflow_rviz
{

namespace
{

std::size_t obstacle_feature_offset(std::size_t base_offset, int index)
{
  return base_offset + static_cast<std::size_t>(4 * index);
}

}  // namespace

MlpResidualPolicy::MlpResidualPolicy(
  std::vector<ExternalFeatureSpec> specs,
  std::vector<std::string> target_keys,
  int hidden_dim,
  int max_obstacles,
  unsigned int seed)
: specs_(std::move(specs)),
  target_keys_(std::move(target_keys)),
  hidden_dim_(hidden_dim),
  max_obstacles_(max_obstacles)
{
  if (hidden_dim_ <= 0) {
    throw std::runtime_error("hidden_dim must be positive");
  }
  if (max_obstacles_ < 0) {
    throw std::runtime_error("max_obstacles must be non-negative");
  }

  input_dim_ = 12 + 3 * static_cast<int>(target_keys_.size()) + 4 * max_obstacles_;
  output_dim_ = 0;
  for (const auto & spec : specs_) {
    output_dim_ += 2 * spec.dim;
  }

  std::mt19937 rng(seed);
  std::normal_distribution<double> normal(0.0, 0.05);

  w1_.resize(hidden_dim_, input_dim_);
  b1_ = Eigen::VectorXd::Zero(hidden_dim_);
  w2_.resize(output_dim_, hidden_dim_);
  b2_ = Eigen::VectorXd::Zero(output_dim_);

  for (int row = 0; row < w1_.rows(); ++row) {
    for (int col = 0; col < w1_.cols(); ++col) {
      w1_(row, col) = normal(rng);
    }
  }
  for (int row = 0; row < w2_.rows(); ++row) {
    for (int col = 0; col < w2_.cols(); ++col) {
      w2_(row, col) = normal(rng);
    }
  }
}

std::unordered_map<std::string, ExternalRmpFeature> MlpResidualPolicy::infer(
  const LearningObservation & observation,
  const std::vector<ExternalFeatureSpec> & specs) const
{
  return decode_output(forward_raw(encode_observation(observation)), specs);
}

Eigen::VectorXd MlpResidualPolicy::parameters() const
{
  const Eigen::Index total_size =
    w1_.size() + b1_.size() + w2_.size() + b2_.size();
  Eigen::VectorXd out(total_size);
  Eigen::Index offset = 0;
  out.segment(offset, w1_.size()) = Eigen::Map<const Eigen::VectorXd>(w1_.data(), w1_.size());
  offset += w1_.size();
  out.segment(offset, b1_.size()) = b1_;
  offset += b1_.size();
  out.segment(offset, w2_.size()) = Eigen::Map<const Eigen::VectorXd>(w2_.data(), w2_.size());
  offset += w2_.size();
  out.segment(offset, b2_.size()) = b2_;
  return out;
}

void MlpResidualPolicy::set_parameters(const Eigen::VectorXd & parameters)
{
  const Eigen::Index expected_size =
    w1_.size() + b1_.size() + w2_.size() + b2_.size();
  if (parameters.size() != expected_size) {
    throw std::runtime_error("MlpResidualPolicy parameter size mismatch");
  }
  Eigen::Index offset = 0;
  Eigen::Map<const Eigen::MatrixXd> new_w1(parameters.data() + offset, w1_.rows(), w1_.cols());
  w1_ = new_w1;
  offset += w1_.size();
  b1_ = parameters.segment(offset, b1_.size());
  offset += b1_.size();
  Eigen::Map<const Eigen::MatrixXd> new_w2(parameters.data() + offset, w2_.rows(), w2_.cols());
  w2_ = new_w2;
  offset += w2_.size();
  b2_ = parameters.segment(offset, b2_.size());
}

Eigen::VectorXd MlpResidualPolicy::encode_observation(const LearningObservation & observation) const
{
  Eigen::VectorXd encoded = Eigen::VectorXd::Zero(input_dim_);
  encoded.segment<6>(0) = observation.q;
  encoded.segment<6>(6) = observation.qd;

  int offset = 12;
  for (const auto & key : target_keys_) {
    const auto it = observation.vector_targets.find(key);
    if (it != observation.vector_targets.end()) {
      encoded.segment<3>(offset) = it->second;
    }
    offset += 3;
  }

  const std::size_t obstacle_offset = static_cast<std::size_t>(offset);
  const int used_obstacles =
    std::min(static_cast<int>(observation.obstacles.size()), max_obstacles_);
  for (int index = 0; index < used_obstacles; ++index) {
    const auto & obstacle = observation.obstacles[static_cast<std::size_t>(index)];
    const auto base = static_cast<Eigen::Index>(obstacle_feature_offset(obstacle_offset, index));
    encoded.segment<3>(base) = obstacle.center;
    encoded[base + 3] = obstacle.radius;
  }

  return encoded;
}

Eigen::VectorXd MlpResidualPolicy::forward_raw(const Eigen::VectorXd & input) const
{
  const Eigen::VectorXd hidden = (w1_ * input + b1_).array().tanh().matrix();
  return w2_ * hidden + b2_;
}

std::unordered_map<std::string, ExternalRmpFeature> MlpResidualPolicy::decode_output(
  const Eigen::VectorXd & output,
  const std::vector<ExternalFeatureSpec> & specs) const
{
  std::unordered_map<std::string, ExternalRmpFeature> decoded;
  int offset = 0;
  for (const auto & spec : specs) {
    const Eigen::VectorXd acceleration = output.segment(offset, spec.dim);
    offset += spec.dim;
    Eigen::MatrixXd metric_sqrt = Eigen::MatrixXd::Zero(spec.dim, spec.dim);
    for (int diag = 0; diag < spec.dim; ++diag) {
      metric_sqrt(diag, diag) = softplus(output[offset + diag]) + min_metric_diag_;
    }
    offset += spec.dim;
    decoded.emplace(spec.key, ExternalRmpFeature{metric_sqrt, acceleration});
  }
  return decoded;
}

Eigen::VectorXd MlpResidualPolicy::flatten_targets(
  const std::unordered_map<std::string, ExternalRmpFeature> & targets) const
{
  Eigen::VectorXd flat = Eigen::VectorXd::Zero(output_dim_);
  int offset = 0;
  for (const auto & spec : specs_) {
    const auto it = targets.find(spec.key);
    if (it != targets.end()) {
      if (it->second.acceleration.size() != spec.dim) {
        throw std::runtime_error("Target residual acceleration size mismatch for key " + spec.key);
      }
      flat.segment(offset, spec.dim) = it->second.acceleration;
      offset += spec.dim;
      if (
        it->second.metric_sqrt.rows() != spec.dim ||
        it->second.metric_sqrt.cols() != spec.dim)
      {
        throw std::runtime_error("Target residual metric size mismatch for key " + spec.key);
      }
      for (int diag = 0; diag < spec.dim; ++diag) {
        flat[offset + diag] = it->second.metric_sqrt(diag, diag);
      }
      offset += spec.dim;
    } else {
      offset += 2 * spec.dim;
    }
  }
  return flat;
}

SupervisedBatchStats MlpResidualPolicy::supervised_batch_stats(
  const std::vector<ResidualSupervisionSample> & samples) const
{
  if (samples.empty()) {
    return SupervisedBatchStats{0.0, Eigen::VectorXd::Zero(parameters().size())};
  }

  Eigen::MatrixXd grad_w1 = Eigen::MatrixXd::Zero(w1_.rows(), w1_.cols());
  Eigen::VectorXd grad_b1 = Eigen::VectorXd::Zero(b1_.size());
  Eigen::MatrixXd grad_w2 = Eigen::MatrixXd::Zero(w2_.rows(), w2_.cols());
  Eigen::VectorXd grad_b2 = Eigen::VectorXd::Zero(b2_.size());
  double loss = 0.0;

  for (const auto & sample : samples) {
    const Eigen::VectorXd input = encode_observation(sample.observation);
    const Eigen::VectorXd z1 = w1_ * input + b1_;
    const Eigen::VectorXd hidden = z1.array().tanh().matrix();
    const Eigen::VectorXd raw_output = w2_ * hidden + b2_;
    const Eigen::VectorXd target = flatten_targets(sample.target_residuals);

    Eigen::VectorXd grad_output = Eigen::VectorXd::Zero(output_dim_);
    int offset = 0;
    for (const auto & spec : specs_) {
      const Eigen::VectorXd acceleration = raw_output.segment(offset, spec.dim);
      const Eigen::VectorXd target_accel = target.segment(offset, spec.dim);
      const Eigen::VectorXd accel_error = acceleration - target_accel;
      loss += accel_error.squaredNorm();
      grad_output.segment(offset, spec.dim) = accel_error;
      offset += spec.dim;

      for (int diag = 0; diag < spec.dim; ++diag) {
        const double raw_value = raw_output[offset + diag];
        const double predicted_diag = softplus(raw_value) + min_metric_diag_;
        const double target_diag = target[offset + diag];
        const double error = predicted_diag - target_diag;
        loss += error * error;
        grad_output[offset + diag] = error * sigmoid(raw_value);
      }
      offset += spec.dim;
    }

    grad_output *= (2.0 / static_cast<double>(output_dim_));
    grad_w2 += grad_output * hidden.transpose();
    grad_b2 += grad_output;

    const Eigen::VectorXd grad_hidden = w2_.transpose() * grad_output;
    const Eigen::VectorXd grad_z1 =
      grad_hidden.array() * (1.0 - hidden.array().square());
    grad_w1 += grad_z1 * input.transpose();
    grad_b1 += grad_z1;
  }

  const double batch_scale = 1.0 / static_cast<double>(samples.size());
  grad_w1 *= batch_scale;
  grad_b1 *= batch_scale;
  grad_w2 *= batch_scale;
  grad_b2 *= batch_scale;
  loss *= batch_scale / static_cast<double>(output_dim_);

  Eigen::VectorXd gradient(parameters().size());
  Eigen::Index flat_offset = 0;
  gradient.segment(flat_offset, grad_w1.size()) =
    Eigen::Map<const Eigen::VectorXd>(grad_w1.data(), grad_w1.size());
  flat_offset += grad_w1.size();
  gradient.segment(flat_offset, grad_b1.size()) = grad_b1;
  flat_offset += grad_b1.size();
  gradient.segment(flat_offset, grad_w2.size()) =
    Eigen::Map<const Eigen::VectorXd>(grad_w2.data(), grad_w2.size());
  flat_offset += grad_w2.size();
  gradient.segment(flat_offset, grad_b2.size()) = grad_b2;

  return SupervisedBatchStats{loss, gradient};
}

double MlpResidualPolicy::softplus(double x)
{
  if (x > 20.0) {
    return x;
  }
  return std::log1p(std::exp(x));
}

double MlpResidualPolicy::sigmoid(double x)
{
  return 1.0 / (1.0 + std::exp(-x));
}

void SgdOptimizer::step(TrainableResidualPolicy & policy, const Eigen::VectorXd & gradient) const
{
  Eigen::VectorXd adjusted = gradient;
  if (weight_decay != 0.0) {
    adjusted += weight_decay * policy.parameters();
  }
  if (gradient_clip_norm > 0.0) {
    const double norm = adjusted.norm();
    if (norm > gradient_clip_norm && norm > 1e-12) {
      adjusted *= gradient_clip_norm / norm;
    }
  }
  policy.set_parameters(policy.parameters() - learning_rate * adjusted);
}

double ResidualSupervisedTrainer::train_epoch(
  MlpResidualPolicy & policy,
  const std::vector<ResidualSupervisionSample> & samples,
  const SgdOptimizer & optimizer,
  std::size_t batch_size)
{
  if (samples.empty()) {
    return 0.0;
  }
  if (batch_size == 0) {
    throw std::runtime_error("batch_size must be positive");
  }

  double accumulated_loss = 0.0;
  std::size_t batch_count = 0;
  for (std::size_t offset = 0; offset < samples.size(); offset += batch_size) {
    const std::size_t end = std::min(samples.size(), offset + batch_size);
    std::vector<ResidualSupervisionSample> batch(
      samples.begin() + static_cast<std::ptrdiff_t>(offset),
      samples.begin() + static_cast<std::ptrdiff_t>(end));
    const auto stats = policy.supervised_batch_stats(batch);
    optimizer.step(policy, stats.gradient);
    accumulated_loss += stats.loss;
    ++batch_count;
  }
  return accumulated_loss / static_cast<double>(batch_count);
}

ResidualRmpEnv::ResidualRmpEnv(
  const LearningReadyRmpAdapter & adapter,
  ResidualEnvConfig config)
: adapter_(adapter),
  config_(std::move(config))
{}

void ResidualRmpEnv::reset(const LearningObservation & initial_observation)
{
  observation_ = initial_observation;
  step_ = 0;
  initialized_ = true;
}

ResidualStepResult ResidualRmpEnv::step(
  const std::unordered_map<std::string, ExternalRmpFeature> & residuals)
{
  if (!initialized_) {
    throw std::runtime_error("ResidualRmpEnv must be reset before step");
  }

  const auto solve_result = adapter_.rollout_single(observation_, residuals);
  LearningObservation next = observation_;
  next.qd += config_.dt * solve_result.qdd;
  next.q += config_.dt * next.qd;
  ++step_;

  double reward = 0.0;
  if (config_.reward_function) {
    reward = config_.reward_function(observation_, solve_result, next);
  } else {
    reward =
      -config_.acceleration_penalty * solve_result.qdd.squaredNorm() -
      config_.velocity_penalty * next.qd.squaredNorm();
  }

  bool done = step_ >= config_.max_steps;
  if (config_.terminal_function) {
    done = done || config_.terminal_function(next, step_);
  }

  observation_ = next;
  return ResidualStepResult{observation_, solve_result, reward, done, step_};
}

ResidualStepResult ResidualRmpEnv::step(const ResidualPolicyInterface & policy)
{
  return step(policy.infer(observation_, adapter_.external_feature_specs()));
}

}  // namespace rb10_rmpflow_rviz
