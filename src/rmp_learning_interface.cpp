#include "rb10_rmpflow_rviz/rmp_learning_interface.hpp"

#include <algorithm>
#include <stdexcept>

namespace rb10_rmpflow_rviz
{

namespace
{

int control_point_count()
{
  int count = 0;
  for (const auto & spec : RB10Model::control_point_specs) {
    count += spec.interpolation_points;
  }
  return count;
}

std::vector<std::size_t> build_topological_order(const std::vector<RmpNodeConfig> & nodes)
{
  std::unordered_map<std::string, std::size_t> enabled_nodes;
  for (std::size_t index = 0; index < nodes.size(); ++index) {
    if (!nodes[index].enabled) {
      continue;
    }
    const auto inserted = enabled_nodes.emplace(nodes[index].name, index);
    if (!inserted.second) {
      throw std::runtime_error("Duplicate graph node name: " + nodes[index].name);
    }
  }

  std::unordered_map<std::string, int> indegree;
  std::unordered_map<std::string, std::vector<std::string>> outgoing;
  for (const auto & entry : enabled_nodes) {
    indegree.emplace(entry.first, 0);
  }

  for (const auto & entry : enabled_nodes) {
    const auto & node = nodes[entry.second];
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
  for (const auto & entry : indegree) {
    if (entry.second == 0) {
      ready.push_back(entry.first);
    }
  }
  std::sort(ready.begin(), ready.end());

  std::vector<std::size_t> order;
  while (!ready.empty()) {
    const auto name = ready.front();
    ready.erase(ready.begin());
    order.push_back(enabled_nodes.at(name));
    for (const auto & child : outgoing[name]) {
      auto & child_indegree = indegree.at(child);
      --child_indegree;
      if (child_indegree == 0) {
        ready.push_back(child);
      }
    }
    std::sort(ready.begin(), ready.end());
  }

  if (order.size() != enabled_nodes.size()) {
    throw std::runtime_error("Cycle detected in graph configuration");
  }
  return order;
}

int infer_node_dim(
  const RmpNodeConfig & node,
  const std::unordered_map<std::string, int> & dims)
{
  if (
    node.task_map_type == "tcp_position" ||
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
        input_dim += dims.at(parent);
      }
      if (input_dim <= 0 || static_cast<int>(node.matrix.size()) % input_dim != 0) {
        throw std::runtime_error("Invalid affine matrix size for node " + node.name);
      }
      return static_cast<int>(node.matrix.size()) / input_dim;
    }
    return dims.at(node.parents.front());
  }
  if (node.task_map_type == "concat") {
    int dim = 0;
    for (const auto & parent : node.parents) {
      dim += dims.at(parent);
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
    return dims.at(node.parents.front());
  }
  if (node.task_map_type == "slice") {
    return node.slice_length > 0 ? node.slice_length :
           dims.at(node.parents.front()) - node.slice_start;
  }
  return dims.at(node.parents.front());
}

std::vector<ExternalFeatureSpec> build_external_feature_specs(const EigenRmpConfig & config)
{
  std::unordered_map<std::string, int> dims;
  dims.emplace("root", 6);
  for (const auto index : build_topological_order(config.graph_nodes)) {
    const auto & node = config.graph_nodes[index];
    dims.emplace(node.name, infer_node_dim(node, dims));
  }

  std::unordered_map<std::string, ExternalFeatureSpec> specs_by_key;
  for (const auto & node : config.graph_nodes) {
    if (!node.enabled) {
      continue;
    }
    const bool uses_external =
      node.leaf_rmp_type == "external" || node.handcrafted_leaf_rmp_type == "external";
    if (!uses_external) {
      continue;
    }
    const int dim = dims.at(node.name);
    auto [it, inserted] = specs_by_key.emplace(
      node.target_key,
      ExternalFeatureSpec{node.target_key, dim, {node.name}});
    if (!inserted) {
      if (it->second.dim != dim) {
        throw std::runtime_error(
                "External RMP key " + node.target_key + " has inconsistent dimensions");
      }
      it->second.node_names.push_back(node.name);
    }
  }

  std::vector<ExternalFeatureSpec> specs;
  specs.reserve(specs_by_key.size());
  for (auto & entry : specs_by_key) {
    std::sort(entry.second.node_names.begin(), entry.second.node_names.end());
    specs.push_back(std::move(entry.second));
  }
  std::sort(
    specs.begin(),
    specs.end(),
    [](const ExternalFeatureSpec & lhs, const ExternalFeatureSpec & rhs) {
      return lhs.key < rhs.key;
    });
  return specs;
}

}  // namespace

std::vector<std::unordered_map<std::string, ExternalRmpFeature>>
ResidualPolicyInterface::infer_batch(
  const std::vector<LearningObservation> & observations,
  const std::vector<ExternalFeatureSpec> & specs) const
{
  std::vector<std::unordered_map<std::string, ExternalRmpFeature>> outputs;
  outputs.reserve(observations.size());
  for (const auto & observation : observations) {
    outputs.push_back(infer(observation, specs));
  }
  return outputs;
}

std::unordered_map<std::string, ExternalRmpFeature> NullResidualPolicy::infer(
  const LearningObservation &,
  const std::vector<ExternalFeatureSpec> &) const
{
  return {};
}

LearningReadyRmpAdapter::LearningReadyRmpAdapter(
  const RmpSolverInterface & solver,
  EigenRmpConfig config)
: solver_(solver),
  config_(std::move(config)),
  external_feature_specs_(build_external_feature_specs(config_))
{}

RmpSolveResult LearningReadyRmpAdapter::rollout_single(
  const LearningObservation & observation,
  const std::unordered_map<std::string, ExternalRmpFeature> & residuals) const
{
  return solver_.get().solve(
    observation.q,
    observation.qd,
    observation.vector_targets,
    observation.obstacles,
    observation.sector_proximity,
    residuals);
}

std::vector<RmpBatchInput> LearningReadyRmpAdapter::make_batch_inputs(
  const std::vector<LearningObservation> & observations,
  const std::vector<std::unordered_map<std::string, ExternalRmpFeature>> & residual_batches) const
{
  if (!residual_batches.empty() && residual_batches.size() != observations.size()) {
    throw std::runtime_error("Residual batch size must match observation batch size");
  }

  std::vector<RmpBatchInput> batch_inputs;
  batch_inputs.reserve(observations.size());
  for (std::size_t index = 0; index < observations.size(); ++index) {
    const auto & observation = observations[index];
    batch_inputs.push_back(RmpBatchInput{
      observation.q,
      observation.qd,
      observation.vector_targets,
      observation.obstacles,
      observation.sector_proximity,
      residual_batches.empty() ? std::unordered_map<std::string, ExternalRmpFeature>{} :
      residual_batches[index]
    });
  }
  return batch_inputs;
}

std::vector<RmpSolveResult> LearningReadyRmpAdapter::rollout_batch(
  const std::vector<LearningObservation> & observations,
  const ResidualPolicyInterface * policy) const
{
  std::vector<std::unordered_map<std::string, ExternalRmpFeature>> residual_batches;
  if (policy != nullptr) {
    residual_batches = policy->infer_batch(observations, external_feature_specs_);
  }
  return solver_.get().solve_batch(make_batch_inputs(observations, residual_batches));
}

}  // namespace rb10_rmpflow_rviz
