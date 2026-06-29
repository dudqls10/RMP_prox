#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Geometry>

#include "rcl_interfaces/msg/set_parameters_result.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rb10_rmpflow_rviz/rb10_model.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "sensor_msgs/msg/range.hpp"
#include "std_msgs/msg/u_int8.hpp"
#include "tf2/exceptions.h"
#include "tf2_ros/buffer.h"
#include "tf2_ros/create_timer_ros.h"
#include "tf2_ros/transform_listener.h"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

namespace rb10_rmpflow_rviz
{

namespace
{

constexpr int kPatchMarkerIdBase = 10000;
constexpr int kPatchMarkerIdStride = 100;
constexpr int kMaxPatchMarkerCount = 49;
constexpr int kFixedSurfaceCollisionMarkerIdBase = 200000;
constexpr double kPi = 3.14159265358979323846;
constexpr const char * kElbowIgnoreSensorFrame = "tof3_1_W";
constexpr std::size_t kElbowJointIndex = 2;

double degrees_to_radians(double degrees)
{
  return degrees * kPi / 180.0;
}

std::vector<std::string> default_range_topics()
{
  std::vector<std::string> topics;
  topics.reserve(RB10Model::sensor_control_points.size());
  for (std::size_t index = 0; index < RB10Model::sensor_control_points.size(); ++index) {
    topics.emplace_back("proximity_distance" + std::to_string(index + 1));
  }
  return topics;
}

std::vector<std::string> default_sensor_frames()
{
  std::vector<std::string> frames;
  frames.reserve(RB10Model::sensor_control_points.size());
  for (const auto & sensor : RB10Model::sensor_control_points) {
    frames.emplace_back(sensor.frame_name);
  }
  return frames;
}

class ProximityObstacleBridgeNode : public rclcpp::Node
{
public:
  ProximityObstacleBridgeNode()
  : Node("proximity_obstacle_bridge"),
    tf_buffer_(get_clock()),
    tf_listener_(tf_buffer_)
  {
    declare_parameter("fixed_frame", "base_link");
    declare_parameter("obstacle_topic", "obstacles");
    declare_parameter("visualization_obstacle_topic", "obstacle_markers");
    declare_parameter("publish_rate", 100.0);
    declare_parameter("visualization_publish_rate", 20.0);
    declare_parameter("publish_collision_obstacles", true);
    declare_parameter("obstacle_radius", 0.05);
    declare_parameter("obstacle_radii", std::vector<double>{});
    declare_parameter("surface_patch_enabled", false);
    declare_parameter("surface_patch_rows", 5);
    declare_parameter("surface_patch_cols", 5);
    declare_parameter("surface_patch_spacing", 0.03);
    declare_parameter("surface_patch_sphere_radius", 0.03);
    declare_parameter("surface_patch_marker_lifetime", 0.25);
    declare_parameter("surface_patch_fixed_visualization", false);
    declare_parameter("surface_patch_visualization_topic", "proximity_surface_patches");
    declare_parameter("surface_patch_memory_distance", 0.025);
    declare_parameter("surface_patch_memory_max_markers", 1200);
    declare_parameter("surface_patch_collision_memory_enabled", false);
    declare_parameter("valid_margin", 1e-3);
    declare_parameter("range_scale", 0.001);
    declare_parameter("minimum_hold_distance", 0.05);
    declare_parameter("trigger_distance", 0.15);
    declare_parameter("trigger_distances", std::vector<double>{});
    declare_parameter("rmp_flag_gate_enabled", false);
    declare_parameter("rmp_flag_topic", "/RMP_flag");
    declare_parameter("rmp_active_flag_value", 1);
    declare_parameter("enable_proximity_distance_1_4", true);
    declare_parameter("joint_state_topic", "joint_states");
    declare_parameter("elbow_tof3_1_w_ignore_enabled", false);
    declare_parameter("elbow_tof3_1_w_ignore_min_deg", -150.0);
    declare_parameter("elbow_tof3_1_w_ignore_max_deg", -140.0);
    declare_parameter("elbow_tof3_1_w_ignore_state_timeout", 0.5);
    declare_parameter("elbow_tof3_1_w_ignore_clear_memory", true);
    declare_parameter("sensor_enabled", std::vector<bool>{});
    declare_parameter("range_topics", default_range_topics());
    declare_parameter("sensor_frames", default_sensor_frames());

    fixed_frame_ = get_parameter("fixed_frame").as_string();
    publish_collision_obstacles_ = get_parameter("publish_collision_obstacles").as_bool();
    visualization_publish_rate_ =
      std::max(get_parameter("visualization_publish_rate").as_double(), 0.0);
    obstacle_radius_ = get_parameter("obstacle_radius").as_double();
    surface_patch_enabled_ = get_parameter("surface_patch_enabled").as_bool();
    surface_patch_rows_ = static_cast<int>(std::clamp(
        get_parameter("surface_patch_rows").as_int(), static_cast<int64_t>(1), static_cast<int64_t>(7)));
    surface_patch_cols_ = static_cast<int>(std::clamp(
        get_parameter("surface_patch_cols").as_int(), static_cast<int64_t>(1), static_cast<int64_t>(7)));
    surface_patch_spacing_ = std::max(get_parameter("surface_patch_spacing").as_double(), 0.0);
    surface_patch_sphere_radius_ =
      std::max(get_parameter("surface_patch_sphere_radius").as_double(), 0.0);
    surface_patch_marker_lifetime_ =
      std::max(get_parameter("surface_patch_marker_lifetime").as_double(), 0.0);
    surface_patch_fixed_visualization_ =
      get_parameter("surface_patch_fixed_visualization").as_bool();
    surface_patch_memory_distance_ =
      std::max(get_parameter("surface_patch_memory_distance").as_double(), 0.0);
    surface_patch_memory_max_markers_ = static_cast<std::size_t>(
      std::max<int64_t>(
        get_parameter("surface_patch_memory_max_markers").as_int(),
        0));
    fixed_surface_patches_.reserve(surface_patch_memory_max_markers_);
    surface_patch_collision_memory_enabled_ =
      get_parameter("surface_patch_collision_memory_enabled").as_bool();
    valid_margin_ = get_parameter("valid_margin").as_double();
    range_scale_ = get_parameter("range_scale").as_double();
    minimum_hold_distance_ = std::max(
      get_parameter("minimum_hold_distance").as_double(),
      0.0);
    trigger_distance_ = get_parameter("trigger_distance").as_double();
    rmp_flag_gate_enabled_ = get_parameter("rmp_flag_gate_enabled").as_bool();
    rmp_active_flag_value_ = static_cast<int>(get_parameter("rmp_active_flag_value").as_int());
    rmp_active_ = !rmp_flag_gate_enabled_;
    enable_proximity_distance_1_4_ =
      get_parameter("enable_proximity_distance_1_4").as_bool();
    elbow_tof3_1_w_ignore_enabled_ =
      get_parameter("elbow_tof3_1_w_ignore_enabled").as_bool();
    const double elbow_ignore_min_deg =
      get_parameter("elbow_tof3_1_w_ignore_min_deg").as_double();
    const double elbow_ignore_max_deg =
      get_parameter("elbow_tof3_1_w_ignore_max_deg").as_double();
    elbow_tof3_1_w_ignore_min_rad_ =
      degrees_to_radians(std::min(elbow_ignore_min_deg, elbow_ignore_max_deg));
    elbow_tof3_1_w_ignore_max_rad_ =
      degrees_to_radians(std::max(elbow_ignore_min_deg, elbow_ignore_max_deg));
    elbow_tof3_1_w_ignore_state_timeout_ = std::max(
      get_parameter("elbow_tof3_1_w_ignore_state_timeout").as_double(),
      0.0);
    elbow_tof3_1_w_ignore_clear_memory_ =
      get_parameter("elbow_tof3_1_w_ignore_clear_memory").as_bool();
    range_topics_ = get_parameter("range_topics").as_string_array();
    sensor_frames_ = get_parameter("sensor_frames").as_string_array();
    const auto obstacle_radii = get_parameter("obstacle_radii").as_double_array();
    const auto trigger_distances = get_parameter("trigger_distances").as_double_array();
    sensor_enabled_ = get_parameter("sensor_enabled").as_bool_array();

    if (range_topics_.size() != sensor_frames_.size()) {
      throw std::runtime_error("range_topics and sensor_frames must have the same size");
    }
    if (!obstacle_radii.empty() && obstacle_radii.size() != sensor_frames_.size()) {
      throw std::runtime_error("obstacle_radii must be empty or match sensor_frames size");
    }
    if (!trigger_distances.empty() && trigger_distances.size() != sensor_frames_.size()) {
      throw std::runtime_error("trigger_distances must be empty or match sensor_frames size");
    }
    if (!sensor_enabled_.empty() && sensor_enabled_.size() != sensor_frames_.size()) {
      throw std::runtime_error("sensor_enabled must be empty or match sensor_frames size");
    }
    if (sensor_enabled_.empty()) {
      sensor_enabled_.assign(sensor_frames_.size(), true);
    }

    latest_ranges_.resize(range_topics_.size());
    obstacle_radii_.assign(sensor_frames_.size(), obstacle_radius_);
    trigger_distances_.assign(sensor_frames_.size(), trigger_distance_);
    for (std::size_t index = 0; index < obstacle_radii.size(); ++index) {
      obstacle_radii_[index] = obstacle_radii[index];
    }
    for (std::size_t index = 0; index < trigger_distances.size(); ++index) {
      trigger_distances_[index] = trigger_distances[index];
    }

    tf_buffer_.setCreateTimerInterface(
      std::make_shared<tf2_ros::CreateTimerROS>(
        get_node_base_interface(), get_node_timers_interface()));

    if (publish_collision_obstacles_) {
      obstacle_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
        get_parameter("obstacle_topic").as_string(),
        10);
      if (visualization_publish_rate_ > 0.0) {
        visualization_obstacle_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
          get_parameter("visualization_obstacle_topic").as_string(),
          10);
      }
    }
    if (surface_patch_fixed_visualization_) {
      surface_patch_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
        get_parameter("surface_patch_visualization_topic").as_string(),
        10);
    }

    if (rmp_flag_gate_enabled_) {
      flag_sub_ = create_subscription<std_msgs::msg::UInt8>(
        get_parameter("rmp_flag_topic").as_string(),
        10,
        std::bind(&ProximityObstacleBridgeNode::on_rmp_flag, this, std::placeholders::_1));
    }
    if (elbow_tof3_1_w_ignore_enabled_) {
      joint_state_sub_ = create_subscription<sensor_msgs::msg::JointState>(
        get_parameter("joint_state_topic").as_string(),
        10,
        std::bind(&ProximityObstacleBridgeNode::on_joint_state, this, std::placeholders::_1));
    }

    range_subs_.resize(range_topics_.size());
    refresh_range_subscriptions();
    parameters_callback_handle_ = add_on_set_parameters_callback(
      std::bind(
        &ProximityObstacleBridgeNode::on_set_parameters,
        this,
        std::placeholders::_1));

    const auto period = std::chrono::duration<double>(
      1.0 / std::max(1.0, get_parameter("publish_rate").as_double()));
    timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::milliseconds>(period),
      std::bind(&ProximityObstacleBridgeNode::publish_obstacles, this));
  }

private:
  struct FixedSurfacePatch
  {
    Eigen::Vector3d center{Eigen::Vector3d::Zero()};
    double radius{0.0};
    std::string sensor_frame;
  };

  using SurfacePatchCenters = std::array<Eigen::Vector3d, kMaxPatchMarkerCount>;

  void publish_obstacles()
  {
    const bool publish_collision_this_cycle =
      publish_collision_obstacles_ && (!rmp_flag_gate_enabled_ || rmp_active_);
    if (rmp_flag_gate_enabled_ && !rmp_active_) {
      clear_obstacles_once();
    }

    visualization_msgs::msg::MarkerArray msg;
    const auto stamp = now();
    const bool ignore_tof3_1_w_this_cycle = should_ignore_tof3_1_w(stamp);
    if (
      ignore_tof3_1_w_this_cycle &&
      elbow_tof3_1_w_ignore_clear_memory_ &&
      erase_fixed_surface_patches_for_sensor(kElbowIgnoreSensorFrame) &&
      publish_collision_this_cycle)
    {
      append_delete_all_marker(msg, stamp);
    }

    for (std::size_t index = 0; index < latest_ranges_.size(); ++index) {
      visualization_msgs::msg::Marker marker;
      marker.header.frame_id = fixed_frame_;
      marker.header.stamp = stamp;
      marker.ns = "proximity_obstacles";
      marker.id = static_cast<int32_t>(index);
      marker.type = visualization_msgs::msg::Marker::SPHERE;
      marker.text = sensor_frames_[index];

      if (ignore_tof3_1_w_this_cycle && sensor_frames_[index] == kElbowIgnoreSensorFrame) {
        if (publish_collision_this_cycle) {
          append_delete_markers(msg, index, stamp);
        }
        continue;
      }

      if (!range_topic_enabled(index)) {
        if (publish_collision_this_cycle) {
          append_delete_markers(msg, index, stamp);
        }
        continue;
      }

      const auto & range_msg = latest_ranges_[index];
      if (!range_msg.has_value() || !range_is_usable(*range_msg)) {
        if (publish_collision_this_cycle) {
          append_delete_markers(msg, index, stamp);
        }
        continue;
      }

      const auto sensor_transform = lookup_sensor_transform(sensor_frames_[index]);
      if (!sensor_transform.has_value()) {
        continue;
      }

      const double range_m = effective_range_m(*range_msg);
      if (range_m > trigger_distances_[index]) {
        if (publish_collision_this_cycle) {
          append_delete_markers(msg, index, stamp);
        }
        continue;
      }

      if (surface_patch_fixed_visualization_ || surface_patch_collision_memory_enabled_) {
        SurfacePatchCenters surface_centers;
        const int surface_patch_count = make_surface_patch_centers(
          index,
          sensor_transform->first,
          sensor_transform->second,
          range_m,
          0.0,
          surface_centers);
        remember_fixed_surface_patches(index, surface_centers, surface_patch_count);
      }

      if (!publish_collision_this_cycle) {
        continue;
      }

      const double obstacle_radius = obstacle_radii_[index];
      const Eigen::Vector3d direction = sensor_transform->second * Eigen::Vector3d::UnitX();
      if (surface_patch_enabled_) {
        SurfacePatchCenters patch_centers;
        const int patch_count = make_surface_patch_centers(
          index,
          sensor_transform->first,
          sensor_transform->second,
          range_m,
          surface_patch_radius(index),
          patch_centers);
        append_surface_patch_markers(
          msg,
          index,
          stamp,
          patch_centers,
          patch_count);
      } else {
        const Eigen::Vector3d center =
          sensor_transform->first + direction * (range_m + obstacle_radius);

        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.position.x = center.x();
        marker.pose.position.y = center.y();
        marker.pose.position.z = center.z();
        marker.pose.orientation.w = 1.0;
        marker.scale.x = obstacle_radius * 2.0;
        marker.scale.y = obstacle_radius * 2.0;
        marker.scale.z = obstacle_radius * 2.0;
        marker.color.r = 1.0F;
        marker.color.g = 0.8F;
        marker.color.b = 0.1F;
        marker.color.a = 0.85F;
        set_marker_lifetime(marker);
        msg.markers.push_back(marker);
      }
    }

    if (publish_collision_this_cycle && surface_patch_collision_memory_enabled_) {
      append_fixed_surface_collision_markers(msg, stamp);
    }
    const bool publish_visualization_this_cycle = should_publish_visualization();
    if (publish_collision_this_cycle && obstacle_pub_) {
      obstacle_pub_->publish(msg);
    }
    if (publish_collision_this_cycle && visualization_obstacle_pub_ && publish_visualization_this_cycle) {
      visualization_obstacle_pub_->publish(msg);
    }
    if (publish_visualization_this_cycle) {
      publish_fixed_surface_patches(stamp);
    }
    if (publish_collision_this_cycle) {
      obstacles_cleared_for_inactive_ = false;
    }
  }

  void on_joint_state(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    std::optional<std::size_t> elbow_index;
    if (msg->name.size() == msg->position.size()) {
      const auto it = std::find(
        msg->name.begin(),
        msg->name.end(),
        std::string(RB10Model::joint_names[kElbowJointIndex]));
      if (it != msg->name.end()) {
        elbow_index = static_cast<std::size_t>(std::distance(msg->name.begin(), it));
      }
    }
    if (!elbow_index.has_value() && msg->position.size() > kElbowJointIndex) {
      elbow_index = kElbowJointIndex;
    }
    if (!elbow_index.has_value()) {
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        5000,
        "joint state for elbow ignore is missing elbow position");
      return;
    }

    const double elbow_position = msg->position[*elbow_index];
    if (!std::isfinite(elbow_position)) {
      return;
    }
    latest_elbow_position_rad_ = elbow_position;
    latest_elbow_state_time_ = now();
  }

  void on_rmp_flag(const std_msgs::msg::UInt8::SharedPtr msg)
  {
    const bool requested_active = static_cast<int>(msg->data) == rmp_active_flag_value_;
    rmp_active_ = requested_active;
    if (!requested_active) {
      obstacles_cleared_for_inactive_ = false;
    }
  }

  rcl_interfaces::msg::SetParametersResult on_set_parameters(
    const std::vector<rclcpp::Parameter> & parameters)
  {
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;

    for (const auto & parameter : parameters) {
      if (
        parameter.get_name() != "enable_proximity_distance_1_4" &&
        parameter.get_name() != "sensor_enabled")
      {
        continue;
      }

      if (parameter.get_name() == "enable_proximity_distance_1_4") {
        if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_BOOL) {
          result.successful = false;
          result.reason = "enable_proximity_distance_1_4 must be a bool";
          return result;
        }

        enable_proximity_distance_1_4_ = parameter.as_bool();
        refresh_range_subscriptions();
        RCLCPP_INFO(
          get_logger(),
          "proximity_distance1~4 input %s",
          enable_proximity_distance_1_4_ ? "enabled" : "disabled");
        continue;
      }

      if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_BOOL_ARRAY) {
        result.successful = false;
        result.reason = "sensor_enabled must be a bool array";
        return result;
      }

      const auto next_sensor_enabled = parameter.as_bool_array();
      if (next_sensor_enabled.size() != range_topics_.size()) {
        result.successful = false;
        result.reason = "sensor_enabled size must match range_topics size";
        return result;
      }

      sensor_enabled_ = next_sensor_enabled;
      refresh_range_subscriptions();
      RCLCPP_INFO(get_logger(), "updated per-sensor proximity input enable list");
    }

    return result;
  }

  void refresh_range_subscriptions()
  {
    for (std::size_t index = 0; index < range_topics_.size(); ++index) {
      if (!range_topic_enabled(index)) {
        latest_ranges_[index].reset();
        range_subs_[index].reset();
        continue;
      }

      if (range_subs_[index]) {
        continue;
      }

      range_subs_[index] = create_subscription<sensor_msgs::msg::Range>(
        range_topics_[index],
        10,
        [this, index](const sensor_msgs::msg::Range::SharedPtr msg) {
          latest_ranges_[index] = *msg;
        });
    }
  }

  static int patch_marker_id(std::size_t sensor_index, int patch_index)
  {
    return kPatchMarkerIdBase +
           static_cast<int>(sensor_index) * kPatchMarkerIdStride +
           patch_index;
  }

  void set_marker_lifetime(visualization_msgs::msg::Marker & marker) const
  {
    if (surface_patch_marker_lifetime_ <= 0.0) {
      return;
    }
    const auto sec = static_cast<int32_t>(std::floor(surface_patch_marker_lifetime_));
    const auto nanosec = static_cast<uint32_t>(
      std::round((surface_patch_marker_lifetime_ - static_cast<double>(sec)) * 1e9));
    marker.lifetime.sec = sec;
    marker.lifetime.nanosec = nanosec;
  }

  visualization_msgs::msg::Marker make_delete_marker(
    std::size_t sensor_index,
    int id,
    const rclcpp::Time & stamp) const
  {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = fixed_frame_;
    marker.header.stamp = stamp;
    marker.ns = "proximity_obstacles";
    marker.id = id;
    marker.type = visualization_msgs::msg::Marker::SPHERE;
    marker.text = sensor_index < sensor_frames_.size() ? sensor_frames_[sensor_index] : "";
    marker.action = visualization_msgs::msg::Marker::DELETE;
    return marker;
  }

  void append_delete_markers(
    visualization_msgs::msg::MarkerArray & msg,
    std::size_t sensor_index,
    const rclcpp::Time & stamp) const
  {
    msg.markers.push_back(make_delete_marker(
        sensor_index,
        static_cast<int>(sensor_index),
        stamp));
    for (int patch_index = 0; patch_index < kMaxPatchMarkerCount; ++patch_index) {
      msg.markers.push_back(make_delete_marker(
          sensor_index,
          patch_marker_id(sensor_index, patch_index),
          stamp));
    }
  }

  void append_delete_all_marker(
    visualization_msgs::msg::MarkerArray & msg,
    const rclcpp::Time & stamp) const
  {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = fixed_frame_;
    marker.header.stamp = stamp;
    marker.ns = "proximity_obstacles";
    marker.action = visualization_msgs::msg::Marker::DELETEALL;
    msg.markers.push_back(marker);
  }

  double surface_patch_radius(std::size_t sensor_index) const
  {
    return surface_patch_sphere_radius_ > 0.0 ?
           surface_patch_sphere_radius_ :
           obstacle_radii_[sensor_index];
  }

  int make_surface_patch_centers(
    std::size_t sensor_index,
    const Eigen::Vector3d & sensor_position,
    const Eigen::Matrix3d & sensor_rotation,
    double range_m,
    double normal_offset_m,
    SurfacePatchCenters & centers) const
  {
    (void)sensor_index;
    const int rows = std::max(surface_patch_rows_, 1);
    const int cols = std::max(surface_patch_cols_, 1);
    const Eigen::Vector3d normal = sensor_rotation * Eigen::Vector3d::UnitX();
    const Eigen::Vector3d tangent_y = sensor_rotation * Eigen::Vector3d::UnitY();
    const Eigen::Vector3d tangent_z = sensor_rotation * Eigen::Vector3d::UnitZ();
    const Eigen::Vector3d patch_center =
      sensor_position + normal * (range_m + normal_offset_m);
    const double row_center = 0.5 * static_cast<double>(rows - 1);
    const double col_center = 0.5 * static_cast<double>(cols - 1);

    int patch_index = 0;
    for (int row = 0; row < rows; ++row) {
      for (int col = 0; col < cols; ++col) {
        if (patch_index >= kMaxPatchMarkerCount) {
          return patch_index;
        }

        centers[patch_index] =
          patch_center +
          (static_cast<double>(row) - row_center) * surface_patch_spacing_ * tangent_y +
          (static_cast<double>(col) - col_center) * surface_patch_spacing_ * tangent_z;
        ++patch_index;
      }
    }

    return patch_index;
  }

  void append_surface_patch_markers(
    visualization_msgs::msg::MarkerArray & msg,
    std::size_t sensor_index,
    const rclcpp::Time & stamp,
    const SurfacePatchCenters & centers,
    int patch_count) const
  {
    msg.markers.push_back(make_delete_marker(
        sensor_index,
        static_cast<int>(sensor_index),
        stamp));

    const double sphere_radius = surface_patch_radius(sensor_index);

    int patch_index = 0;
    for (; patch_index < patch_count; ++patch_index) {
      const Eigen::Vector3d & center = centers[patch_index];
      visualization_msgs::msg::Marker marker;
      marker.header.frame_id = fixed_frame_;
      marker.header.stamp = stamp;
      marker.ns = "proximity_obstacles";
      marker.id = patch_marker_id(sensor_index, patch_index);
      marker.type = visualization_msgs::msg::Marker::SPHERE;
      marker.text = sensor_frames_[sensor_index];
      marker.action = visualization_msgs::msg::Marker::ADD;
      marker.pose.position.x = center.x();
      marker.pose.position.y = center.y();
      marker.pose.position.z = center.z();
      marker.pose.orientation.w = 1.0;
      marker.scale.x = sphere_radius * 2.0;
      marker.scale.y = sphere_radius * 2.0;
      marker.scale.z = sphere_radius * 2.0;
      marker.color.r = 1.0F;
      marker.color.g = 0.55F;
      marker.color.b = 0.05F;
      marker.color.a = 0.72F;
      set_marker_lifetime(marker);
      msg.markers.push_back(marker);
    }

    for (; patch_index < kMaxPatchMarkerCount; ++patch_index) {
      msg.markers.push_back(make_delete_marker(
          sensor_index,
          patch_marker_id(sensor_index, patch_index),
          stamp));
    }
  }

  bool fixed_surface_patch_is_new(const Eigen::Vector3d & center) const
  {
    if (surface_patch_memory_distance_ <= 0.0) {
      return true;
    }

    const double min_distance_sq =
      surface_patch_memory_distance_ * surface_patch_memory_distance_;
    for (const auto & patch : fixed_surface_patches_) {
      if ((patch.center - center).squaredNorm() < min_distance_sq) {
        return false;
      }
    }
    return true;
  }

  void remember_fixed_surface_patches(
    std::size_t sensor_index,
    const SurfacePatchCenters & centers,
    int patch_count)
  {
    if (
      !(surface_patch_fixed_visualization_ || surface_patch_collision_memory_enabled_) ||
      surface_patch_memory_max_markers_ == 0 ||
      sensor_index >= sensor_frames_.size())
    {
      return;
    }

    const double sphere_radius = surface_patch_radius(sensor_index);
    for (int patch_index = 0; patch_index < patch_count; ++patch_index) {
      const Eigen::Vector3d & center = centers[patch_index];
      if (!fixed_surface_patch_is_new(center)) {
        continue;
      }

      while (fixed_surface_patches_.size() >= surface_patch_memory_max_markers_) {
        fixed_surface_patches_.erase(fixed_surface_patches_.begin());
      }

      FixedSurfacePatch patch;
      patch.center = center;
      patch.radius = sphere_radius;
      patch.sensor_frame = sensor_frames_[sensor_index];
      fixed_surface_patches_.push_back(std::move(patch));
    }
  }

  bool erase_fixed_surface_patches_for_sensor(const std::string & sensor_frame)
  {
    const auto old_size = fixed_surface_patches_.size();
    fixed_surface_patches_.erase(
      std::remove_if(
        fixed_surface_patches_.begin(),
        fixed_surface_patches_.end(),
        [&sensor_frame](const FixedSurfacePatch & patch) {
          return patch.sensor_frame == sensor_frame;
        }),
      fixed_surface_patches_.end());
    if (fixed_surface_patches_.size() == old_size) {
      return false;
    }

    fixed_surface_markers_cleared_ = false;
    return true;
  }

  void publish_fixed_surface_patches(const rclcpp::Time & stamp)
  {
    if (!surface_patch_pub_) {
      return;
    }

    visualization_msgs::msg::MarkerArray msg;
    if (!fixed_surface_markers_cleared_) {
      visualization_msgs::msg::Marker clear_marker;
      clear_marker.header.frame_id = fixed_frame_;
      clear_marker.header.stamp = stamp;
      clear_marker.ns = "proximity_surface_patches";
      clear_marker.action = visualization_msgs::msg::Marker::DELETEALL;
      msg.markers.push_back(clear_marker);
      fixed_surface_markers_cleared_ = true;
    }

    for (std::size_t index = 0; index < fixed_surface_patches_.size(); ++index) {
      const auto & patch = fixed_surface_patches_[index];
      visualization_msgs::msg::Marker marker;
      marker.header.frame_id = fixed_frame_;
      marker.header.stamp = stamp;
      marker.ns = "proximity_surface_patches";
      marker.id = static_cast<int32_t>(index);
      marker.type = visualization_msgs::msg::Marker::SPHERE;
      marker.text = patch.sensor_frame;
      marker.action = visualization_msgs::msg::Marker::ADD;
      marker.pose.position.x = patch.center.x();
      marker.pose.position.y = patch.center.y();
      marker.pose.position.z = patch.center.z();
      marker.pose.orientation.w = 1.0;
      marker.scale.x = patch.radius * 2.0;
      marker.scale.y = patch.radius * 2.0;
      marker.scale.z = patch.radius * 2.0;
      marker.color.r = 0.05F;
      marker.color.g = 0.75F;
      marker.color.b = 1.0F;
      marker.color.a = 0.55F;
      msg.markers.push_back(marker);
    }

    surface_patch_pub_->publish(msg);
  }

  void append_fixed_surface_collision_markers(
    visualization_msgs::msg::MarkerArray & msg,
    const rclcpp::Time & stamp) const
  {
    for (std::size_t index = 0; index < fixed_surface_patches_.size(); ++index) {
      const auto & patch = fixed_surface_patches_[index];
      visualization_msgs::msg::Marker marker;
      marker.header.frame_id = fixed_frame_;
      marker.header.stamp = stamp;
      marker.ns = "proximity_obstacles";
      marker.id = kFixedSurfaceCollisionMarkerIdBase + static_cast<int32_t>(index);
      marker.type = visualization_msgs::msg::Marker::SPHERE;
      marker.text = patch.sensor_frame;
      marker.action = visualization_msgs::msg::Marker::ADD;
      marker.pose.position.x = patch.center.x();
      marker.pose.position.y = patch.center.y();
      marker.pose.position.z = patch.center.z();
      marker.pose.orientation.w = 1.0;
      marker.scale.x = patch.radius * 2.0;
      marker.scale.y = patch.radius * 2.0;
      marker.scale.z = patch.radius * 2.0;
      marker.color.r = 0.05F;
      marker.color.g = 0.75F;
      marker.color.b = 1.0F;
      marker.color.a = 0.55F;
      msg.markers.push_back(marker);
    }
  }

  bool range_topic_enabled(std::size_t index) const
  {
    if (index >= range_topics_.size()) {
      return false;
    }
    if (index >= sensor_enabled_.size() || !sensor_enabled_[index]) {
      return false;
    }
    if (!enable_proximity_distance_1_4_ && is_proximity_distance_1_4(range_topics_[index])) {
      return false;
    }
    return true;
  }

  static bool is_proximity_distance_1_4(const std::string & topic)
  {
    std::string normalized_topic = topic;
    if (!normalized_topic.empty() && normalized_topic.front() == '/') {
      normalized_topic.erase(normalized_topic.begin());
    }

    return
      normalized_topic == "proximity_distance1" ||
      normalized_topic == "proximity_distance2" ||
      normalized_topic == "proximity_distance3" ||
      normalized_topic == "proximity_distance4";
  }

  void clear_obstacles_once()
  {
    if (obstacles_cleared_for_inactive_) {
      return;
    }

    visualization_msgs::msg::Marker clear_marker;
    clear_marker.header.frame_id = fixed_frame_;
    clear_marker.header.stamp = now();
    clear_marker.action = visualization_msgs::msg::Marker::DELETEALL;

    visualization_msgs::msg::MarkerArray clear_array;
    clear_array.markers.push_back(clear_marker);
    bool published_clear = false;
    if (obstacle_pub_) {
      obstacle_pub_->publish(clear_array);
      published_clear = true;
    }
    if (visualization_obstacle_pub_) {
      visualization_obstacle_pub_->publish(clear_array);
      published_clear = true;
    }
    if (!published_clear) {
      return;
    }
    obstacles_cleared_for_inactive_ = true;
  }

  bool should_publish_visualization()
  {
    if (visualization_publish_rate_ <= 0.0) {
      return false;
    }

    const auto now = std::chrono::steady_clock::now();
    const auto period = std::chrono::duration<double>(1.0 / visualization_publish_rate_);
    if (
      last_visualization_publish_time_.time_since_epoch().count() != 0 &&
      now - last_visualization_publish_time_ < period)
    {
      return false;
    }

    last_visualization_publish_time_ = now;
    return true;
  }

  bool should_ignore_tof3_1_w(const rclcpp::Time & stamp) const
  {
    if (
      !elbow_tof3_1_w_ignore_enabled_ ||
      !latest_elbow_position_rad_.has_value() ||
      !latest_elbow_state_time_.has_value())
    {
      return false;
    }

    if (elbow_tof3_1_w_ignore_state_timeout_ > 0.0) {
      const double age = (stamp - *latest_elbow_state_time_).seconds();
      if (age < 0.0 || age > elbow_tof3_1_w_ignore_state_timeout_) {
        return false;
      }
    }

    return
      *latest_elbow_position_rad_ >= elbow_tof3_1_w_ignore_min_rad_ &&
      *latest_elbow_position_rad_ <= elbow_tof3_1_w_ignore_max_rad_;
  }

  bool range_is_usable(const sensor_msgs::msg::Range & msg) const
  {
    if (!std::isfinite(msg.range)) {
      return false;
    }
    if (msg.range < 0.0) {
      return false;
    }
    return msg.range < (msg.max_range - valid_margin_);
  }

  double effective_range_m(const sensor_msgs::msg::Range & msg) const
  {
    return std::max(msg.range * range_scale_, minimum_hold_distance_);
  }

  std::optional<std::pair<Eigen::Vector3d, Eigen::Matrix3d>> lookup_sensor_transform(
    const std::string & sensor_frame)
  {
    try {
      const auto tf = tf_buffer_.lookupTransform(
        fixed_frame_, sensor_frame, tf2::TimePointZero);
      const Eigen::Quaterniond quat(
        tf.transform.rotation.w,
        tf.transform.rotation.x,
        tf.transform.rotation.y,
        tf.transform.rotation.z);
      return std::make_pair(
        Eigen::Vector3d(
          tf.transform.translation.x,
          tf.transform.translation.y,
          tf.transform.translation.z),
        quat.normalized().toRotationMatrix());
    } catch (const tf2::TransformException & ex) {
      RCLCPP_DEBUG_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "TF lookup failed for %s: %s",
        sensor_frame.c_str(),
        ex.what());
      return std::nullopt;
    }
  }

  std::string fixed_frame_;
  bool publish_collision_obstacles_{true};
  double visualization_publish_rate_{20.0};
  double obstacle_radius_{0.05};
  bool surface_patch_enabled_{false};
  int surface_patch_rows_{5};
  int surface_patch_cols_{5};
  double surface_patch_spacing_{0.03};
  double surface_patch_sphere_radius_{0.03};
  double surface_patch_marker_lifetime_{0.25};
  bool surface_patch_fixed_visualization_{false};
  bool surface_patch_collision_memory_enabled_{false};
  bool fixed_surface_markers_cleared_{false};
  double surface_patch_memory_distance_{0.025};
  std::size_t surface_patch_memory_max_markers_{1200};
  double valid_margin_{1e-3};
  double range_scale_{0.001};
  double minimum_hold_distance_{0.05};
  double trigger_distance_{0.3};
  bool rmp_flag_gate_enabled_{false};
  bool rmp_active_{true};
  bool enable_proximity_distance_1_4_{true};
  bool elbow_tof3_1_w_ignore_enabled_{false};
  bool elbow_tof3_1_w_ignore_clear_memory_{true};
  double elbow_tof3_1_w_ignore_min_rad_{degrees_to_radians(-150.0)};
  double elbow_tof3_1_w_ignore_max_rad_{degrees_to_radians(-140.0)};
  double elbow_tof3_1_w_ignore_state_timeout_{0.5};
  bool obstacles_cleared_for_inactive_{false};
  int rmp_active_flag_value_{1};
  std::vector<std::string> range_topics_;
  std::vector<std::string> sensor_frames_;
  std::vector<bool> sensor_enabled_;
  std::vector<double> obstacle_radii_;
  std::vector<double> trigger_distances_;
  std::vector<std::optional<sensor_msgs::msg::Range>> latest_ranges_;
  std::vector<FixedSurfacePatch> fixed_surface_patches_;
  std::optional<double> latest_elbow_position_rad_;
  std::optional<rclcpp::Time> latest_elbow_state_time_;
  std::chrono::steady_clock::time_point last_visualization_publish_time_{};

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  rclcpp::Subscription<std_msgs::msg::UInt8>::SharedPtr flag_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
  std::vector<rclcpp::Subscription<sensor_msgs::msg::Range>::SharedPtr> range_subs_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr parameters_callback_handle_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr obstacle_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr visualization_obstacle_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr surface_patch_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace

}  // namespace rb10_rmpflow_rviz

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rb10_rmpflow_rviz::ProximityObstacleBridgeNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
