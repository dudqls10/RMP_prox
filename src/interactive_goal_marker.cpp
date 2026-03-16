#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "geometry_msgs/msg/point.hpp"
#include "interactive_markers/interactive_marker_server.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/color_rgba.hpp"
#include "visualization_msgs/msg/interactive_marker.hpp"
#include "visualization_msgs/msg/interactive_marker_control.hpp"
#include "visualization_msgs/msg/interactive_marker_feedback.hpp"
#include "visualization_msgs/msg/marker.hpp"

namespace rb10_rmpflow_rviz
{

class InteractiveGoalMarkerNode : public rclcpp::Node
{
public:
  InteractiveGoalMarkerNode()
  : Node("interactive_goal")
  {
    declare_parameter("initial_x", 0.6);
    declare_parameter("initial_y", -0.4);
    declare_parameter("initial_z", 0.6);

    goal_pub_ = create_publisher<geometry_msgs::msg::Point>("goal_position", 10);
    server_ = std::make_shared<interactive_markers::InteractiveMarkerServer>(
      "goal_marker_server",
      get_node_base_interface(),
      get_node_clock_interface(),
      get_node_logging_interface(),
      get_node_topics_interface(),
      get_node_services_interface());

    goal_.x = get_parameter("initial_x").as_double();
    goal_.y = get_parameter("initial_y").as_double();
    goal_.z = get_parameter("initial_z").as_double();

    create_marker();
    timer_ = create_wall_timer(
      std::chrono::milliseconds(100),
      std::bind(&InteractiveGoalMarkerNode::publish_goal, this));
  }

private:
  void create_marker()
  {
    visualization_msgs::msg::InteractiveMarker marker;
    marker.header.frame_id = "base_link";
    marker.name = "goal_position";
    marker.description = "Goal Position";
    marker.pose.position = goal_;
    marker.scale = 0.15;

    visualization_msgs::msg::Marker sphere;
    sphere.type = visualization_msgs::msg::Marker::SPHERE;
    sphere.scale.x = 0.1;
    sphere.scale.y = 0.1;
    sphere.scale.z = 0.1;
    sphere.color.r = 0.0F;
    sphere.color.g = 0.8F;
    sphere.color.b = 0.0F;
    sphere.color.a = 0.9F;

    visualization_msgs::msg::InteractiveMarkerControl visible;
    visible.always_visible = true;
    visible.markers.push_back(sphere);
    marker.controls.push_back(visible);

    marker.controls.push_back(axis_control("move_x", 1.0, 0.0, 0.0));
    marker.controls.push_back(axis_control("move_y", 0.0, 0.0, 1.0));
    marker.controls.push_back(axis_control("move_z", 0.0, 1.0, 0.0));

    visualization_msgs::msg::InteractiveMarkerControl move_3d;
    move_3d.orientation.w = 1.0;
    move_3d.name = "move_3d";
    move_3d.interaction_mode = visualization_msgs::msg::InteractiveMarkerControl::MOVE_3D;
    marker.controls.push_back(move_3d);

    server_->insert(
      marker,
      std::bind(&InteractiveGoalMarkerNode::feedback, this, std::placeholders::_1));
    server_->applyChanges();
  }

  visualization_msgs::msg::InteractiveMarkerControl axis_control(
    const std::string & name,
    double x,
    double y,
    double z) const
  {
    visualization_msgs::msg::InteractiveMarkerControl control;
    control.orientation.w = 1.0;
    control.orientation.x = x;
    control.orientation.y = y;
    control.orientation.z = z;
    control.name = name;
    control.interaction_mode = visualization_msgs::msg::InteractiveMarkerControl::MOVE_AXIS;
    return control;
  }

  void feedback(
    const visualization_msgs::msg::InteractiveMarkerFeedback::ConstSharedPtr & feedback_msg)
  {
    goal_ = feedback_msg->pose.position;
  }

  void publish_goal()
  {
    geometry_msgs::msg::Point msg;
    msg = goal_;
    goal_pub_->publish(msg);
  }

  geometry_msgs::msg::Point goal_;
  rclcpp::Publisher<geometry_msgs::msg::Point>::SharedPtr goal_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::shared_ptr<interactive_markers::InteractiveMarkerServer> server_;
};

}  // namespace rb10_rmpflow_rviz

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rb10_rmpflow_rviz::InteractiveGoalMarkerNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
