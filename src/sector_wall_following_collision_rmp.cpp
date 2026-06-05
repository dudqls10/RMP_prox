#include "rb10_rmpflow_rviz/sector_wall_following_collision_rmp.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace rb10_rmpflow_rviz
{

namespace
{

constexpr double kEps = 1e-9;

double sign_preserving_min(double value, double minimum_abs)
{
  if (std::abs(value) >= minimum_abs) {
    return value;
  }
  return value < 0.0 ? -minimum_abs : minimum_abs;
}

}  // namespace

SectorWallFollowingCollisionRMP::SectorWallFollowingCollisionRMP(
  WallFollowingCollisionParams params)
: params_(params)
{
  if (!(params_.d_ref > params_.d_safe)) {
    throw std::runtime_error("wall_following collision requires d_ref > d_safe");
  }
  if (!(params_.d_on < params_.d_off)) {
    throw std::runtime_error("wall_following collision requires d_on < d_off");
  }
  params_.v_t_max = std::max(0.0, params_.v_t_max);
  params_.v_n_toward_max = std::max(0.0, params_.v_n_toward_max);
  params_.v_n_away_max = std::max(0.0, params_.v_n_away_max);
  params_.a_safe_max = std::max(0.0, params_.a_safe_max);
  params_.m_t = std::clamp(params_.m_t, 0.0, params_.m_max);
  params_.m_n = std::clamp(std::max(params_.m_n, params_.m_t), 0.0, params_.m_max);
  params_.m_max = std::max(params_.m_max, 0.0);
  params_.direction_lock_time = std::max(0.0, params_.direction_lock_time);
  params_.derivative_filter_alpha = std::clamp(params_.derivative_filter_alpha, 0.0, 1.0);
  params_.near_zero_metric = std::max(0.0, params_.near_zero_metric);
}

double SectorWallFollowingCollisionRMP::clamp(double value, double lower, double upper)
{
  return std::min(std::max(value, lower), upper);
}

Eigen::Vector3d SectorWallFollowingCollisionRMP::clamp_norm(
  const Eigen::Vector3d & value,
  double max_norm)
{
  if (max_norm <= 0.0) {
    return Eigen::Vector3d::Zero();
  }
  const double norm = value.norm();
  if (norm <= max_norm || norm <= kEps) {
    return value;
  }
  return value * (max_norm / norm);
}

double SectorWallFollowingCollisionRMP::smooth_activation(
  double distance,
  double d_on,
  double d_off)
{
  if (distance <= d_on) {
    return 1.0;
  }
  if (distance >= d_off) {
    return 0.0;
  }
  const double t = clamp((d_off - distance) / sign_preserving_min(d_off - d_on, kEps), 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

bool SectorWallFollowingCollisionRMP::sector_index_valid(int sector)
{
  return sector >= 0 && sector < static_cast<int>(kWallFollowingSectorCount);
}

int SectorWallFollowingCollisionRMP::opposite_sector(int sector)
{
  switch (static_cast<WallFollowingSector>(sector)) {
    case WallFollowingSector::East:
      return static_cast<int>(WallFollowingSector::West);
    case WallFollowingSector::West:
      return static_cast<int>(WallFollowingSector::East);
    case WallFollowingSector::North:
      return static_cast<int>(WallFollowingSector::South);
    case WallFollowingSector::South:
      return static_cast<int>(WallFollowingSector::North);
  }
  return -1;
}

std::array<int, 2> SectorWallFollowingCollisionRMP::adjacent_sectors(int sector)
{
  switch (static_cast<WallFollowingSector>(sector)) {
    case WallFollowingSector::East:
    case WallFollowingSector::West:
      return {
        static_cast<int>(WallFollowingSector::North),
        static_cast<int>(WallFollowingSector::South)
      };
    case WallFollowingSector::North:
    case WallFollowingSector::South:
      return {
        static_cast<int>(WallFollowingSector::East),
        static_cast<int>(WallFollowingSector::West)
      };
  }
  return {{-1, -1}};
}

Eigen::Vector3d SectorWallFollowingCollisionRMP::normalized_or_zero(
  const Eigen::Vector3d & value)
{
  const double norm = value.norm();
  if (norm <= kEps) {
    return Eigen::Vector3d::Zero();
  }
  return value / norm;
}

SectorWallFollowingCollisionRMP::Result SectorWallFollowingCollisionRMP::evaluate(
  const Input & input,
  State & state) const
{
  Result result;

  SectorArray effective_distances{};
  effective_distances.fill(std::numeric_limits<double>::infinity());

  int active_sector_count = 0;
  int closest_sector = -1;
  double closest_distance = std::numeric_limits<double>::infinity();
  for (std::size_t index = 0; index < kWallFollowingSectorCount; ++index) {
    if (!input.valid[index] || !std::isfinite(input.distances[index])) {
      continue;
    }
    effective_distances[index] = input.distances[index];
    if (input.has_sigma[index] && std::isfinite(input.sigmas[index])) {
      effective_distances[index] -= params_.kappa_sigma * std::max(0.0, input.sigmas[index]);
    }
    if (effective_distances[index] < params_.d_on) {
      ++active_sector_count;
    }
    if (effective_distances[index] < closest_distance) {
      closest_distance = effective_distances[index];
      closest_sector = static_cast<int>(index);
    }
  }

  if (!sector_index_valid(closest_sector)) {
    state.active = false;
    result.metric = params_.near_zero_metric * Eigen::Matrix3d::Identity();
    return result;
  }

  if (!state.active && closest_distance < params_.d_on) {
    state.active = true;
  } else if (state.active && closest_distance > params_.d_off) {
    state.active = false;
  }
  state.active_sector = closest_sector;

  if (!state.active) {
    result.active_sector = closest_sector;
    result.effective_distance = closest_distance;
    result.metric = params_.near_zero_metric * Eigen::Matrix3d::Identity();
    return result;
  }

  Eigen::Vector3d n_w = normalized_or_zero(input.normals_world[closest_sector]);
  if (!params_.normal_points_toward_obstacle) {
    n_w = -n_w;
  }
  if (n_w.norm() <= kEps) {
    state.active = false;
    result.metric = params_.near_zero_metric * Eigen::Matrix3d::Identity();
    return result;
  }

  result.active = true;
  result.active_sector = closest_sector;
  result.effective_distance = closest_distance;
  result.h = closest_distance - params_.d_safe;
  result.alpha = smooth_activation(closest_distance, params_.d_on, params_.d_off);

  const Eigen::Matrix3d eye = Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d normal_projector = n_w * n_w.transpose();
  const Eigen::Matrix3d tangent_projector = eye - normal_projector;

  Eigen::Vector3d v_t = clamp_norm(tangent_projector * input.v_goal, params_.v_t_max);
  const bool opposite_pair_active =
    sector_index_valid(opposite_sector(closest_sector)) &&
    effective_distances[static_cast<std::size_t>(opposite_sector(closest_sector))] < params_.d_on;
  const bool crowded_module = active_sector_count >= 3 || opposite_pair_active;

  if (v_t.norm() < 1e-6) {
    if (crowded_module) {
      result.recovery = true;
    } else {
      const auto adjacent = adjacent_sectors(closest_sector);
      int follow_sector = -1;
      if (
        sector_index_valid(state.locked_follow_sector) &&
        input.time_sec < state.lock_until_sec)
      {
        follow_sector = state.locked_follow_sector;
      } else {
        const double d0 = sector_index_valid(adjacent[0]) ?
          effective_distances[static_cast<std::size_t>(adjacent[0])] :
          -std::numeric_limits<double>::infinity();
        const double d1 = sector_index_valid(adjacent[1]) ?
          effective_distances[static_cast<std::size_t>(adjacent[1])] :
          -std::numeric_limits<double>::infinity();
        follow_sector = d0 >= d1 ? adjacent[0] : adjacent[1];
        state.locked_follow_sector = follow_sector;
        state.lock_until_sec = input.time_sec + params_.direction_lock_time;
      }

      if (
        sector_index_valid(follow_sector) &&
        std::isfinite(effective_distances[static_cast<std::size_t>(follow_sector)]) &&
        effective_distances[static_cast<std::size_t>(follow_sector)] >= params_.d_on)
      {
        Eigen::Vector3d follow_normal =
          normalized_or_zero(input.normals_world[static_cast<std::size_t>(follow_sector)]);
        if (!params_.normal_points_toward_obstacle) {
          follow_normal = -follow_normal;
        }
        const Eigen::Vector3d tangent_dir = normalized_or_zero(tangent_projector * follow_normal);
        if (tangent_dir.norm() > kEps) {
          v_t = params_.v_t_max * tangent_dir;
          result.follow_sector = follow_sector;
        } else {
          result.recovery = true;
        }
      } else {
        result.recovery = true;
      }
    }
  }

  if (result.recovery) {
    v_t.setZero();
  }

  const double lower_away = -params_.v_n_away_max;
  const double upper_toward =
    std::min(params_.v_n_toward_max, params_.gamma_cbf * result.h);
  const double u_n0 = params_.k_dist * (closest_distance - params_.d_ref);
  const double u_n =
    upper_toward < lower_away ? lower_away : clamp(u_n0, lower_away, upper_toward);
  Eigen::Vector3d v_n = u_n * n_w;
  if (result.recovery) {
    v_n = -params_.v_n_away_max * n_w;
  }

  const double kinematic_h_dot = -n_w.dot(input.xdot);
  double h_dot = kinematic_h_dot;
  if (
    state.derivative_initialized &&
    state.derivative_sector == closest_sector &&
    input.time_sec > state.last_time_sec + kEps)
  {
    const double raw_derivative =
      (closest_distance - state.last_distance) / (input.time_sec - state.last_time_sec);
    state.filtered_distance_derivative =
      params_.derivative_filter_alpha * raw_derivative +
      (1.0 - params_.derivative_filter_alpha) * state.filtered_distance_derivative;
    h_dot = state.filtered_distance_derivative;
  } else {
    state.filtered_distance_derivative = kinematic_h_dot;
  }
  state.derivative_initialized = true;
  state.derivative_sector = closest_sector;
  state.last_distance = closest_distance;
  state.last_time_sec = input.time_sec;

  result.h_dot = h_dot;
  result.v_tangent = v_t;
  result.v_normal = v_n;
  result.v_ref = v_t + v_n;

  const Eigen::Vector3d a_wf = params_.k_vel * (result.v_ref - input.xdot);
  const double s_safe =
    std::max(0.0, -params_.k_safe_1 * h_dot - params_.k_safe_0 * result.h);
  result.safety_acceleration = clamp_norm(s_safe * (-n_w), params_.a_safe_max);
  result.acceleration = result.alpha * (a_wf + result.safety_acceleration);

  const double m_t = std::clamp(params_.m_t, 0.0, params_.m_max);
  const double m_n = std::clamp(std::max(params_.m_n, m_t), 0.0, params_.m_max);
  result.metric = result.alpha * (m_t * tangent_projector + m_n * normal_projector);

  return result;
}

}  // namespace rb10_rmpflow_rviz
