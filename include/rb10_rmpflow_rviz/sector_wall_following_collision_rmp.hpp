#pragma once

#include <array>
#include <limits>

#include <Eigen/Dense>

#include "rb10_rmpflow_rviz/rmp_eigen_solver.hpp"

namespace rb10_rmpflow_rviz
{

class SectorWallFollowingCollisionRMP
{
public:
  using SectorArray = std::array<double, kWallFollowingSectorCount>;
  using BoolSectorArray = std::array<bool, kWallFollowingSectorCount>;
  using VectorSectorArray = std::array<Eigen::Vector3d, kWallFollowingSectorCount>;

  struct State
  {
    bool active{false};
    int active_sector{-1};
    int locked_follow_sector{-1};
    double lock_until_sec{0.0};
    bool derivative_initialized{false};
    int derivative_sector{-1};
    double last_distance{0.0};
    double last_time_sec{0.0};
    double filtered_distance_derivative{0.0};
  };

  struct Input
  {
    SectorArray distances{};
    SectorArray sigmas{};
    BoolSectorArray has_sigma{};
    BoolSectorArray valid{};
    VectorSectorArray normals_world{};
    Eigen::Vector3d xdot{Eigen::Vector3d::Zero()};
    Eigen::Vector3d v_goal{Eigen::Vector3d::Zero()};
    double time_sec{0.0};

    Input()
    {
      distances.fill(std::numeric_limits<double>::infinity());
      sigmas.fill(0.0);
      has_sigma.fill(false);
      valid.fill(false);
      normals_world.fill(Eigen::Vector3d::Zero());
    }
  };

  struct Result
  {
    Eigen::Vector3d acceleration{Eigen::Vector3d::Zero()};
    Eigen::Matrix3d metric{Eigen::Matrix3d::Zero()};
    Eigen::Vector3d v_ref{Eigen::Vector3d::Zero()};
    Eigen::Vector3d v_tangent{Eigen::Vector3d::Zero()};
    Eigen::Vector3d v_normal{Eigen::Vector3d::Zero()};
    Eigen::Vector3d safety_acceleration{Eigen::Vector3d::Zero()};
    double alpha{0.0};
    double effective_distance{std::numeric_limits<double>::infinity()};
    double h{std::numeric_limits<double>::infinity()};
    double h_dot{0.0};
    int active_sector{-1};
    int follow_sector{-1};
    bool active{false};
    bool recovery{false};
  };

  explicit SectorWallFollowingCollisionRMP(WallFollowingCollisionParams params);

  const WallFollowingCollisionParams & params() const
  {
    return params_;
  }

  Result evaluate(const Input & input, State & state) const;

private:
  static double clamp(double value, double lower, double upper);
  static Eigen::Vector3d clamp_norm(const Eigen::Vector3d & value, double max_norm);
  static double smooth_activation(double distance, double d_on, double d_off);
  static int opposite_sector(int sector);
  static std::array<int, 2> adjacent_sectors(int sector);
  static bool sector_index_valid(int sector);
  static Eigen::Vector3d normalized_or_zero(const Eigen::Vector3d & value);

  WallFollowingCollisionParams params_;
};

}  // namespace rb10_rmpflow_rviz
