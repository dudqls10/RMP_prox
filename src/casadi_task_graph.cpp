#include "rb10_rmpflow_rviz/casadi_task_graph.hpp"

#include <stdexcept>

namespace rb10_rmpflow_rviz
{

namespace
{

using casadi::dot;
using std::sqrt;

casadi::MX ensure_column(const casadi::MX & value)
{
  return value.is_column() ? value : casadi::MX::reshape(value, value.numel(), 1);
}

casadi::MX stacked_inputs(const std::vector<casadi::MX> & parents)
{
  std::vector<casadi::MX> cols;
  cols.reserve(parents.size());
  for (const auto & parent : parents) {
    cols.push_back(ensure_column(parent));
  }
  return casadi::MX::vertcat(cols);
}

}  // namespace

CompiledCasadiTaskMap CasadiTaskMapLibrary::compile(
  const RmpNodeConfig & node,
  const std::vector<int> & parent_dims) const
{
  std::vector<casadi::MX> parents;
  std::vector<casadi::MX> parent_velocities;
  std::vector<casadi::MX> inputs;
  parents.reserve(parent_dims.size());
  parent_velocities.reserve(parent_dims.size());
  inputs.reserve(parent_dims.size());
  for (std::size_t index = 0; index < parent_dims.size(); ++index) {
    auto parent = casadi::MX::sym(
      node.name + "_parent_" + std::to_string(index),
      parent_dims[index],
      1);
    auto parent_velocity = casadi::MX::sym(
      node.name + "_parent_velocity_" + std::to_string(index),
      parent_dims[index],
      1);
    parents.push_back(parent);
    parent_velocities.push_back(parent_velocity);
    inputs.push_back(parent);
    inputs.push_back(parent_velocity);
  }

  auto output = ensure_column(apply_task_map(node, parents));
  auto ydot = casadi::MX::zeros(output.size1(), 1);
  for (std::size_t index = 0; index < parents.size(); ++index) {
    ydot += casadi::MX::jacobian(output, parents[index]) * parent_velocities[index];
  }
  ydot = ensure_column(ydot);
  auto stacked_parents = casadi::MX::vertcat(parents);
  auto stacked_parent_velocities = casadi::MX::vertcat(parent_velocities);
  auto local_curvature = ensure_column(jtimes(ydot, stacked_parents, stacked_parent_velocities));

  std::vector<casadi::MX> outputs;
  outputs.push_back(output);
  outputs.push_back(ydot);
  outputs.push_back(local_curvature);
  for (const auto & parent : parents) {
    outputs.push_back(casadi::MX::jacobian(output, parent));
  }

  return CompiledCasadiTaskMap{
    casadi::Function(node.name + "_task_map", inputs, outputs),
    parent_dims,
    static_cast<int>(output.size1())
  };
}

casadi::MX CasadiTaskMapLibrary::apply_task_map(
  const RmpNodeConfig & node,
  const std::vector<casadi::MX> & parents)
{
  if (parents.empty()) {
    throw std::runtime_error("CasADi task map requires at least one parent input");
  }

  if (node.task_map_type == "identity" || node.task_map_type == "cspace_target") {
    return node.scale * ensure_column(parents.front());
  }

  if (node.task_map_type == "sum") {
    casadi::MX out = ensure_column(parents.front());
    for (std::size_t index = 1; index < parents.size(); ++index) {
      out += ensure_column(parents[index]);
    }
    return node.scale * out;
  }

  if (node.task_map_type == "weighted_sum") {
    casadi::MX out = casadi::MX::zeros(ensure_column(parents.front()).size1(), 1);
    for (std::size_t index = 0; index < parents.size(); ++index) {
      const double weight =
        index < node.parent_weights.size() ? node.parent_weights[index] : 1.0;
      out += weight * ensure_column(parents[index]);
    }
    if (!node.bias.empty()) {
      out += casadi::DM(node.bias);
    }
    return node.scale * out;
  }

  if (node.task_map_type == "affine") {
    const auto input = stacked_inputs(parents);
    casadi::MX out;
    if (!node.matrix.empty()) {
      const int input_dim = input.size1();
      if (input_dim <= 0 || static_cast<int>(node.matrix.size()) % input_dim != 0) {
        throw std::runtime_error("affine task map has invalid matrix size");
      }
      const int output_dim = static_cast<int>(node.matrix.size()) / input_dim;
      casadi::DM matrix = casadi::DM::zeros(output_dim, input_dim);
      for (int row = 0; row < output_dim; ++row) {
        for (int col = 0; col < input_dim; ++col) {
          matrix(row, col) = node.matrix[static_cast<std::size_t>(row * input_dim + col)];
        }
      }
      out = matrix * input;
    } else {
      out = input;
    }
    if (!node.bias.empty()) {
      out += casadi::DM(node.bias);
    }
    return node.scale * ensure_column(out);
  }

  if (node.task_map_type == "difference") {
    if (parents.size() != 2) {
      throw std::runtime_error("difference task map expects exactly two parents");
    }
    return node.scale * (ensure_column(parents[0]) - ensure_column(parents[1]));
  }

  if (node.task_map_type == "concat") {
    return node.scale * casadi::MX::vertcat(parents);
  }

  if (node.task_map_type == "elem_multiply") {
    if (parents.size() != 2) {
      throw std::runtime_error("elem_multiply task map expects exactly two parents");
    }
    const auto left = ensure_column(parents[0]);
    const auto right = ensure_column(parents[1]);
    if (left.size1() != right.size1()) {
      throw std::runtime_error("elem_multiply task map expects matching dimensions");
    }
    std::vector<casadi::MX> out;
    out.reserve(static_cast<std::size_t>(left.size1()));
    for (casadi_int index = 0; index < left.size1(); ++index) {
      out.push_back(left(index) * right(index));
    }
    return node.scale * casadi::MX::vertcat(out);
  }

  if (node.task_map_type == "elem_divide") {
    if (parents.size() != 2) {
      throw std::runtime_error("elem_divide task map expects exactly two parents");
    }
    const auto numerator = ensure_column(parents[0]);
    const auto denominator = ensure_column(parents[1]);
    if (numerator.size1() != denominator.size1()) {
      throw std::runtime_error("elem_divide task map expects matching dimensions");
    }
    std::vector<casadi::MX> out;
    out.reserve(static_cast<std::size_t>(numerator.size1()));
    for (casadi_int index = 0; index < numerator.size1(); ++index) {
      out.push_back(numerator(index) / (denominator(index) + node.epsilon));
    }
    return node.scale * casadi::MX::vertcat(out);
  }

  if (node.task_map_type == "slice") {
    const auto parent = ensure_column(parents.front());
    const int length = node.slice_length > 0 ? node.slice_length : parent.size1() - node.slice_start;
    return node.scale * parent(casadi::Slice(node.slice_start, node.slice_start + length));
  }

  if (node.task_map_type == "norm") {
    const auto parent = ensure_column(parents.front());
    const auto norm = sqrt(dot(parent, parent));
    std::vector<casadi::MX> outputs;
    outputs.push_back(norm);
    return node.scale * casadi::MX::vertcat(outputs);
  }

  if (node.task_map_type == "normalize") {
    const auto parent = ensure_column(parents.front());
    const auto norm = sqrt(dot(parent, parent));
    return node.scale * (parent / (norm + node.epsilon));
  }

  if (node.task_map_type == "sin") {
    return node.scale * sin(ensure_column(parents.front()));
  }

  if (node.task_map_type == "cos") {
    return node.scale * cos(ensure_column(parents.front()));
  }

  if (node.task_map_type == "tanh") {
    return node.scale * tanh(ensure_column(parents.front()));
  }

  if (node.task_map_type == "square") {
    const auto parent = ensure_column(parents.front());
    return node.scale * pow(parent, 2);
  }

  if (node.task_map_type == "abs") {
    return node.scale * fabs(ensure_column(parents.front()));
  }

  throw std::runtime_error("Unsupported CasADi task map type: " + node.task_map_type);
}

CasadiTaskGraph::CasadiTaskGraph(std::vector<RmpNodeConfig> nodes)
: nodes_(std::move(nodes))
{}

std::unordered_map<std::string, CompiledCasadiTaskMap> CasadiTaskGraph::compile(
  const std::unordered_map<std::string, int> & node_dims) const
{
  std::unordered_map<std::string, CompiledCasadiTaskMap> compiled;
  for (const auto & node : nodes_) {
    std::vector<int> parent_dims;
    parent_dims.reserve(node.parents.size());
    for (const auto & parent : node.parents) {
      const auto dim_it = node_dims.find(parent);
      if (dim_it == node_dims.end()) {
        throw std::runtime_error("Missing CasADi node dimension for parent: " + parent);
      }
      parent_dims.push_back(dim_it->second);
    }
    compiled.emplace(node.name, library_.compile(node, parent_dims));
  }
  return compiled;
}

}  // namespace rb10_rmpflow_rviz
