#!/usr/bin/env python3

from typing import Callable, List, Optional, Tuple

import rclpy
from geometry_msgs.msg import Point, Pose, PoseArray, PoseStamped
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import Marker, MarkerArray

try:
    from nav_msgs.msg import Path
except ImportError:  # pragma: no cover - nav_msgs is expected on ROS installs.
    Path = None


SupportedCallback = Callable[[object], None]


class RmpGoalSequenceVisualizer(Node):
    def __init__(self) -> None:
        super().__init__("rmp_goal_sequence_visualizer")
        self.declare_parameter("input_topic", "/RMP_goal_sequence")
        self.declare_parameter("input_type", "auto")
        self.declare_parameter("marker_topic", "/rmp_goal_sequence_marker")
        self.declare_parameter("frame_id", "base_link")
        self.declare_parameter("marker_scale", 0.03)
        self.declare_parameter("line_width", 0.008)
        self.declare_parameter("alpha", 0.25)
        self.declare_parameter("max_points", 100)
        self.declare_parameter("float_stride", 0)

        self.input_topic = self.get_parameter("input_topic").get_parameter_value().string_value
        self.input_type = self.get_parameter("input_type").get_parameter_value().string_value
        self.marker_topic = self.get_parameter("marker_topic").get_parameter_value().string_value
        self.default_frame_id = self.get_parameter("frame_id").get_parameter_value().string_value
        self.marker_scale = max(
            self.get_parameter("marker_scale").get_parameter_value().double_value,
            1e-4,
        )
        self.line_width = max(
            self.get_parameter("line_width").get_parameter_value().double_value,
            1e-4,
        )
        self.alpha = max(
            0.0,
            min(self.get_parameter("alpha").get_parameter_value().double_value, 1.0),
        )
        self.max_points = max(
            1,
            int(self.get_parameter("max_points").get_parameter_value().integer_value),
        )
        self.float_stride = int(self.get_parameter("float_stride").get_parameter_value().integer_value)

        self.marker_pub = self.create_publisher(MarkerArray, self.marker_topic, 10)
        self.subscription = None
        self.wait_logged = False

        if self.input_type != "auto":
            self.try_configure_subscription(self.input_type)
        else:
            self.create_timer(0.5, self.try_autodetect_subscription)

    def supported_types(self) -> dict:
        supported = {
            "geometry_msgs/msg/PoseArray": (PoseArray, self.on_pose_array),
            "geometry_msgs/msg/PoseStamped": (PoseStamped, self.on_pose_stamped),
            "geometry_msgs/msg/Pose": (Pose, self.on_pose),
            "std_msgs/msg/Float64MultiArray": (Float64MultiArray, self.on_float64_multi_array),
        }
        if Path is not None:
            supported["nav_msgs/msg/Path"] = (Path, self.on_path)
        return supported

    def normalize_type_name(self, value: str) -> str:
        aliases = {
            "pose_array": "geometry_msgs/msg/PoseArray",
            "posearray": "geometry_msgs/msg/PoseArray",
            "pose_stamped": "geometry_msgs/msg/PoseStamped",
            "posestamped": "geometry_msgs/msg/PoseStamped",
            "pose": "geometry_msgs/msg/Pose",
            "path": "nav_msgs/msg/Path",
            "float64_multi_array": "std_msgs/msg/Float64MultiArray",
            "float64multiarray": "std_msgs/msg/Float64MultiArray",
        }
        return aliases.get(value, aliases.get(value.lower(), value))

    def try_autodetect_subscription(self) -> None:
        if self.subscription is not None:
            return

        for topic_name, topic_types in self.get_topic_names_and_types():
            if topic_name != self.input_topic:
                continue
            for topic_type in topic_types:
                if self.try_configure_subscription(topic_type):
                    return

        if not self.wait_logged:
            self.get_logger().info(
                f"Waiting for supported goal sequence topic {self.input_topic} "
                "(PoseArray, Path, PoseStamped, Pose, or Float64MultiArray)."
            )
            self.wait_logged = True

    def try_configure_subscription(self, topic_type: str) -> bool:
        normalized_type = self.normalize_type_name(topic_type)
        entry = self.supported_types().get(normalized_type)
        if entry is None:
            self.get_logger().warn(f"Unsupported goal sequence type: {topic_type}")
            return False

        msg_type, callback = entry
        self.subscription = self.create_subscription(
            msg_type,
            self.input_topic,
            callback,
            10,
        )
        self.get_logger().info(
            f"Visualizing {self.input_topic} ({normalized_type}) on {self.marker_topic}"
        )
        return True

    def on_pose_array(self, msg: PoseArray) -> None:
        frame_id = msg.header.frame_id or self.default_frame_id
        points = [(pose.position.x, pose.position.y, pose.position.z) for pose in msg.poses]
        self.publish_markers(frame_id, points)

    def on_path(self, msg) -> None:
        frame_id = msg.header.frame_id or self.default_frame_id
        points = [
            (
                pose_stamped.pose.position.x,
                pose_stamped.pose.position.y,
                pose_stamped.pose.position.z,
            )
            for pose_stamped in msg.poses
        ]
        self.publish_markers(frame_id, points)

    def on_pose_stamped(self, msg: PoseStamped) -> None:
        frame_id = msg.header.frame_id or self.default_frame_id
        pose = msg.pose
        self.publish_markers(frame_id, [(pose.position.x, pose.position.y, pose.position.z)])

    def on_pose(self, msg: Pose) -> None:
        self.publish_markers(
            self.default_frame_id,
            [(msg.position.x, msg.position.y, msg.position.z)],
        )

    def on_float64_multi_array(self, msg: Float64MultiArray) -> None:
        data = list(msg.data)
        if not data:
            self.publish_markers(self.default_frame_id, [])
            return

        stride = self.float_stride
        if stride <= 0:
            if len(data) % 9 == 0:
                stride = 9
            elif len(data) % 8 == 0:
                stride = 8
            elif len(data) % 7 == 0:
                stride = 7
            else:
                stride = 3
        if stride < 3:
            self.get_logger().warn("float_stride must be 0, 3, or at least 3.")
            return

        points = []
        for start in range(0, len(data) - 2, stride):
            if stride == 9:
                # Layout: t, x, y, z, qx, qy, qz, qw, g. Visualize only x/y/z.
                points.append((data[start + 1], data[start + 2], data[start + 3]))
            elif stride == 8:
                # Legacy layout: t, x, y, z, qx, qy, qz, g. Visualize only x/y/z.
                points.append((data[start + 1], data[start + 2], data[start + 3]))
            elif stride == 7:
                # Layout: x, y, z, qx, qy, qz, qw. Visualize only x/y/z.
                points.append((data[start], data[start + 1], data[start + 2]))
            else:
                points.append((data[start], data[start + 1], data[start + 2]))
        self.publish_markers(self.default_frame_id, points)

    def make_delete_all_marker(self, frame_id: str) -> Marker:
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "rmp_goal_sequence"
        marker.action = Marker.DELETEALL
        return marker

    def publish_markers(
        self,
        frame_id: str,
        points_xyz: List[Tuple[float, float, float]],
    ) -> None:
        points_xyz = points_xyz[: self.max_points]

        markers = MarkerArray()
        markers.markers.append(self.make_delete_all_marker(frame_id))

        if not points_xyz:
            self.marker_pub.publish(markers)
            return

        stamp = self.get_clock().now().to_msg()

        line = Marker()
        line.header.frame_id = frame_id
        line.header.stamp = stamp
        line.ns = "rmp_goal_sequence_line"
        line.id = 0
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.pose.orientation.w = 1.0
        line.scale.x = self.line_width
        line.color.r = 0.0
        line.color.g = 1.0
        line.color.b = 0.0
        line.color.a = min(self.alpha, 0.35)
        for x, y, z in points_xyz:
            point = Point()
            point.x = float(x)
            point.y = float(y)
            point.z = float(z)
            line.points.append(point)
        markers.markers.append(line)

        for index, (x, y, z) in enumerate(points_xyz):
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = stamp
            marker.ns = "rmp_goal_sequence_points"
            marker.id = index
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(x)
            marker.pose.position.y = float(y)
            marker.pose.position.z = float(z)
            marker.pose.orientation.w = 1.0
            marker.scale.x = self.marker_scale
            marker.scale.y = self.marker_scale
            marker.scale.z = self.marker_scale
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = self.alpha
            markers.markers.append(marker)

        self.marker_pub.publish(markers)


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = RmpGoalSequenceVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
