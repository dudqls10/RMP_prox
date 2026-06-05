#include <cmath>
#include <iostream>
#include <stdexcept>

#include "rb10_rmpflow_rviz/sector_wall_following_collision_rmp.hpp"

namespace rb10_rmpflow_rviz
{

namespace
{

constexpr double kTol = 1e-6;

void require(bool condition, const char * message)
{
  if (!condition) {
    throw std::runtime_error(message);
  }
}

SectorWallFollowingCollisionRMP::Input base_input()
{
  SectorWallFollowingCollisionRMP::Input input;
  input.valid.fill(true);
  input.distances = {{0.5, 0.5, 0.5, 0.5}};
  input.normals_world = {{
    Eigen::Vector3d::UnitX(),
    -Eigen::Vector3d::UnitX(),
    Eigen::Vector3d::UnitY(),
    -Eigen::Vector3d::UnitY()
  }};
  return input;
}

WallFollowingCollisionParams test_params()
{
  WallFollowingCollisionParams params;
  params.d_safe = 0.10;
  params.d_ref = 0.15;
  params.d_on = 0.22;
  params.d_off = 0.30;
  params.k_dist = 1.0;
  params.k_vel = 4.0;
  params.k_safe_0 = 8.0;
  params.k_safe_1 = 4.0;
  params.v_t_max = 0.10;
  params.v_n_toward_max = 0.03;
  params.v_n_away_max = 0.10;
  params.a_safe_max = 0.50;
  params.direction_lock_time = 1.0;
  return params;
}

void test_no_obstacle()
{
  SectorWallFollowingCollisionRMP rmp(test_params());
  SectorWallFollowingCollisionRMP::State state;
  auto input = base_input();
  const auto result = rmp.evaluate(input, state);
  require(!result.active, "no obstacle should leave RMP inactive");
  require(result.alpha == 0.0, "no obstacle should have zero activation");
  require(result.acceleration.norm() < kTol, "no obstacle should have zero acceleration");
  require(result.metric.norm() < 1e-6, "no obstacle should have near-zero metric");
}

void test_left_wall_target_into_wall()
{
  SectorWallFollowingCollisionRMP rmp(test_params());
  SectorWallFollowingCollisionRMP::State state;
  auto input = base_input();
  input.distances[static_cast<std::size_t>(WallFollowingSector::West)] = 0.12;
  input.distances[static_cast<std::size_t>(WallFollowingSector::North)] = 0.50;
  input.distances[static_cast<std::size_t>(WallFollowingSector::South)] = 0.30;
  input.v_goal = -0.20 * Eigen::Vector3d::UnitX();
  const auto result = rmp.evaluate(input, state);
  const Eigen::Vector3d n_w = -Eigen::Vector3d::UnitX();
  require(result.active, "left wall should activate RMP");
  require(
    result.follow_sector == static_cast<int>(WallFollowingSector::North),
    "left wall should choose freer adjacent sector");
  require(std::abs(result.v_tangent.dot(n_w)) < kTol, "tangent velocity should remove normal");
  require(result.v_ref.dot(n_w) <= 0.0, "reference velocity should not move into left wall");
}

void test_too_close_safety()
{
  SectorWallFollowingCollisionRMP rmp(test_params());
  SectorWallFollowingCollisionRMP::State state;
  auto input = base_input();
  input.distances[static_cast<std::size_t>(WallFollowingSector::West)] = 0.05;
  input.xdot = -0.20 * Eigen::Vector3d::UnitX();
  const auto result = rmp.evaluate(input, state);
  const Eigen::Vector3d n_w = -Eigen::Vector3d::UnitX();
  require(result.active, "too-close wall should activate RMP");
  require(result.safety_acceleration.dot(-n_w) > 0.0, "safety acceleration should point away");
  require(
    result.safety_acceleration.norm() <= test_params().a_safe_max + kTol,
    "safety acceleration should be saturated");
}

void test_target_already_tangential()
{
  SectorWallFollowingCollisionRMP rmp(test_params());
  SectorWallFollowingCollisionRMP::State state;
  auto input = base_input();
  input.distances[static_cast<std::size_t>(WallFollowingSector::West)] = 0.12;
  input.v_goal = 0.07 * Eigen::Vector3d::UnitY();
  const auto result = rmp.evaluate(input, state);
  require(result.active, "tangential target should activate near wall");
  require(!result.recovery, "tangential target should not enter recovery");
  require(result.v_tangent.y() > 0.06, "tangential target velocity should be preserved");
}

void test_hysteresis()
{
  SectorWallFollowingCollisionRMP rmp(test_params());
  SectorWallFollowingCollisionRMP::State state;
  auto input = base_input();
  input.distances[static_cast<std::size_t>(WallFollowingSector::West)] = 0.25;
  auto result = rmp.evaluate(input, state);
  require(!result.active, "distance above d_on should not activate from inactive state");

  input.distances[static_cast<std::size_t>(WallFollowingSector::West)] = 0.21;
  result = rmp.evaluate(input, state);
  require(result.active, "distance below d_on should activate");

  input.distances[static_cast<std::size_t>(WallFollowingSector::West)] = 0.24;
  result = rmp.evaluate(input, state);
  require(result.active, "distance between thresholds should stay active");

  input.distances[static_cast<std::size_t>(WallFollowingSector::West)] = 0.31;
  result = rmp.evaluate(input, state);
  require(!result.active, "distance above d_off should deactivate");
}

void test_direction_lock()
{
  SectorWallFollowingCollisionRMP rmp(test_params());
  SectorWallFollowingCollisionRMP::State state;
  auto input = base_input();
  input.distances[static_cast<std::size_t>(WallFollowingSector::West)] = 0.12;
  input.distances[static_cast<std::size_t>(WallFollowingSector::North)] = 0.50;
  input.distances[static_cast<std::size_t>(WallFollowingSector::South)] = 0.40;
  input.time_sec = 0.0;
  auto result = rmp.evaluate(input, state);
  require(
    result.follow_sector == static_cast<int>(WallFollowingSector::North),
    "initial follow sector should be North");

  input.distances[static_cast<std::size_t>(WallFollowingSector::North)] = 0.30;
  input.distances[static_cast<std::size_t>(WallFollowingSector::South)] = 0.60;
  input.time_sec = 0.5;
  result = rmp.evaluate(input, state);
  require(
    result.follow_sector == static_cast<int>(WallFollowingSector::North),
    "locked follow sector should remain North");

  input.time_sec = 1.2;
  result = rmp.evaluate(input, state);
  require(
    result.follow_sector == static_cast<int>(WallFollowingSector::South),
    "follow sector should switch after lock expires");
}

void test_opposite_sectors_active()
{
  SectorWallFollowingCollisionRMP rmp(test_params());
  SectorWallFollowingCollisionRMP::State state;
  auto input = base_input();
  input.distances[static_cast<std::size_t>(WallFollowingSector::East)] = 0.11;
  input.distances[static_cast<std::size_t>(WallFollowingSector::West)] = 0.12;
  const auto result = rmp.evaluate(input, state);
  require(result.active, "opposite sectors should activate RMP");
  require(result.recovery, "opposite sectors should use recovery behavior");
  require(
    result.v_ref.dot(-Eigen::Vector3d::UnitX()) > 0.0,
    "opposite-sector recovery should move away from closest wall");
}

}  // namespace

}  // namespace rb10_rmpflow_rviz

int main()
{
  rb10_rmpflow_rviz::test_no_obstacle();
  rb10_rmpflow_rviz::test_left_wall_target_into_wall();
  rb10_rmpflow_rviz::test_too_close_safety();
  rb10_rmpflow_rviz::test_target_already_tangential();
  rb10_rmpflow_rviz::test_hysteresis();
  rb10_rmpflow_rviz::test_direction_lock();
  rb10_rmpflow_rviz::test_opposite_sectors_active();
  std::cout << "sector wall-following collision RMP tests passed\n";
  return 0;
}
