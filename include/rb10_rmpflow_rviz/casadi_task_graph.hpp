#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include <casadi/casadi.hpp>

#include "rb10_rmpflow_rviz/rmp_eigen_solver.hpp"

namespace rb10_rmpflow_rviz
{

struct CompiledCasadiTaskMap
{
  casadi::Function function;
  std::vector<int> parent_dims;
  int output_dim{0};
};

class CasadiTaskMapLibrary
{
public:
  CompiledCasadiTaskMap compile(
    const RmpNodeConfig & node,
    const std::vector<int> & parent_dims) const;

private:
  static casadi::MX apply_task_map(
    const RmpNodeConfig & node,
    const std::vector<casadi::MX> & parents);
};

class CasadiTaskGraph
{
public:
  explicit CasadiTaskGraph(std::vector<RmpNodeConfig> nodes);

  std::unordered_map<std::string, CompiledCasadiTaskMap> compile(
    const std::unordered_map<std::string, int> & node_dims) const;

private:
  std::vector<RmpNodeConfig> nodes_;
  CasadiTaskMapLibrary library_;
};

}  // namespace rb10_rmpflow_rviz
