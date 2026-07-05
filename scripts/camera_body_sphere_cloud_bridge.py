#!/usr/bin/env python3

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import rclpy
from builtin_interfaces.msg import Duration as DurationMsg
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _duration_msg(seconds: float) -> DurationMsg:
    seconds = max(float(seconds), 0.0)
    whole = int(math.floor(seconds))
    return DurationMsg(sec=whole, nanosec=int((seconds - whole) * 1e9))


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _rotate_vector(qx: float, qy: float, qz: float, qw: float, point: Sequence[float]) -> Tuple[float, float, float]:
    # Quaternion-vector multiply without requiring tf2_geometry_msgs.
    x, y, z = point
    tx = 2.0 * (qy * z - qz * y)
    ty = 2.0 * (qz * x - qx * z)
    tz = 2.0 * (qx * y - qy * x)
    return (
        x + qw * tx + (qy * tz - qz * ty),
        y + qw * ty + (qz * tx - qx * tz),
        z + qw * tz + (qx * ty - qy * tx),
    )


class CameraBodySphereCloudBridge(Node):
    def __init__(self) -> None:
        super().__init__("camera_body_sphere_cloud_bridge")

        self.declare_parameter("cloud_topic", "/rmp_camera/obstacle_body_sphere_cloud")
        self.declare_parameter("camera_marker_topic", "/rmp_camera/obstacle_body_spheres")
        self.declare_parameter("subscribe_camera_marker_topic", True)
        self.declare_parameter("relay_camera_markers_to_rviz", True)
        self.declare_parameter("additional_obstacle_topics", [""])
        self.declare_parameter("collision_obstacle_topic", "/obstacles")
        self.declare_parameter("rviz_marker_topic", "/obstacle_markers")
        self.declare_parameter("output_frame", "base_link")
        self.declare_parameter("default_radius_m", 0.08)
        self.declare_parameter("min_radius_m", 0.01)
        self.declare_parameter("max_radius_m", 0.50)
        self.declare_parameter("max_cloud_spheres", 128)
        self.declare_parameter("cloud_stride", 1)
        self.declare_parameter("marker_lifetime_sec", 0.35)
        self.declare_parameter("stale_timeout_sec", 0.50)
        self.declare_parameter("publish_rate", 30.0)
        self.declare_parameter("publish_empty_when_stale", True)
        self.declare_parameter("marker_namespace", "camera_obstacles")
        self.declare_parameter("log_rate_hz", 2.0)

        self.cloud_topic = str(self.get_parameter("cloud_topic").value)
        self.camera_marker_topic = str(self.get_parameter("camera_marker_topic").value)
        self.output_frame = str(self.get_parameter("output_frame").value).strip()
        self.default_radius_m = max(float(self.get_parameter("default_radius_m").value), 1e-4)
        self.min_radius_m = max(float(self.get_parameter("min_radius_m").value), 1e-4)
        self.max_radius_m = max(float(self.get_parameter("max_radius_m").value), self.min_radius_m)
        self.max_cloud_spheres = max(int(self.get_parameter("max_cloud_spheres").value), 1)
        self.cloud_stride = max(int(self.get_parameter("cloud_stride").value), 1)
        self.marker_lifetime_sec = max(float(self.get_parameter("marker_lifetime_sec").value), 0.0)
        self.stale_timeout_sec = max(float(self.get_parameter("stale_timeout_sec").value), 0.05)
        self.publish_empty_when_stale = _as_bool(self.get_parameter("publish_empty_when_stale").value)
        self.relay_camera_markers_to_rviz = _as_bool(
            self.get_parameter("relay_camera_markers_to_rviz").value
        )
        self.marker_namespace = str(self.get_parameter("marker_namespace").value)
        self.log_period_s = 1.0 / max(float(self.get_parameter("log_rate_hz").value), 0.1)
        self.last_log_time_s = 0.0

        self.additional_obstacle_topics = self._string_list_parameter("additional_obstacle_topics")
        self.latest_cloud_markers: Optional[MarkerArray] = None
        self.latest_cloud_time_s: Optional[float] = None
        self.latest_camera_rviz_markers: Optional[MarkerArray] = None
        self.latest_camera_rviz_time_s: Optional[float] = None
        self.latest_additional_markers: Dict[str, Tuple[MarkerArray, float]] = {}
        self.last_collision_had_markers = False
        self.last_rviz_had_markers = False

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        cloud_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        marker_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        self.create_subscription(PointCloud2, self.cloud_topic, self._on_cloud, cloud_qos)
        if _as_bool(self.get_parameter("subscribe_camera_marker_topic").value):
            self.create_subscription(
                MarkerArray,
                self.camera_marker_topic,
                self._on_camera_rviz_markers,
                marker_qos,
            )
        for topic in self.additional_obstacle_topics:
            self.create_subscription(
                MarkerArray,
                topic,
                lambda msg, topic=topic: self._on_additional_markers(topic, msg),
                marker_qos,
            )

        self.collision_pub = self.create_publisher(
            MarkerArray, str(self.get_parameter("collision_obstacle_topic").value), 10
        )
        self.rviz_pub = self.create_publisher(
            MarkerArray, str(self.get_parameter("rviz_marker_topic").value), 10
        )
        publish_rate = max(float(self.get_parameter("publish_rate").value), 1.0)
        self.create_timer(1.0 / publish_rate, self._publish_outputs)

        self.get_logger().info(
            "Camera body sphere cloud bridge started: "
            f"cloud={self.cloud_topic}, camera_markers={self.camera_marker_topic}, "
            f"additional={self.additional_obstacle_topics}, "
            f"collision_out={self.get_parameter('collision_obstacle_topic').value}, "
            f"rviz_out={self.get_parameter('rviz_marker_topic').value}"
        )

    def _string_list_parameter(self, name: str) -> List[str]:
        value = self.get_parameter(name).value
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return [str(item).strip() for item in value if str(item).strip()]

    def _now_s(self) -> float:
        return float(self.get_clock().now().nanoseconds) * 1e-9

    def _on_cloud(self, msg: PointCloud2) -> None:
        markers = self._cloud_to_markers(msg)
        if markers is None:
            return
        self.latest_cloud_markers = markers
        self.latest_cloud_time_s = self._now_s()

    def _on_camera_rviz_markers(self, msg: MarkerArray) -> None:
        self.latest_camera_rviz_markers = msg
        self.latest_camera_rviz_time_s = self._now_s()

    def _on_additional_markers(self, topic: str, msg: MarkerArray) -> None:
        self.latest_additional_markers[topic] = (msg, self._now_s())

    def _cloud_to_markers(self, msg: PointCloud2) -> Optional[MarkerArray]:
        if not msg.header.frame_id:
            self._log_throttled("Ignoring camera sphere cloud with empty frame_id.")
            return None

        output_frame = self.output_frame or msg.header.frame_id
        transform = None
        if output_frame != msg.header.frame_id:
            transform = self._lookup_transform(output_frame, msg.header.frame_id, msg)
            if transform is None:
                return None

        field_names = [field.name for field in msg.fields]
        if not {"x", "y", "z"}.issubset(set(field_names)):
            self._log_throttled(
                f"Ignoring {self.cloud_topic}: PointCloud2 needs x/y/z fields, got {field_names}."
            )
            return None

        radius_field = self._first_existing(field_names, ("radius", "sphere_radius", "r"))
        diameter_field = self._first_existing(field_names, ("diameter", "d", "scale", "scale_x"))
        read_fields = ["x", "y", "z"]
        if radius_field:
            read_fields.append(radius_field)
        elif diameter_field:
            read_fields.append(diameter_field)

        markers: List[Marker] = []
        for point_index, point in enumerate(
            point_cloud2.read_points(msg, field_names=read_fields, skip_nans=True)
        ):
            if point_index % self.cloud_stride != 0:
                continue
            if len(markers) >= self.max_cloud_spheres:
                break

            x = float(point[0])
            y = float(point[1])
            z = float(point[2])
            if not all(math.isfinite(value) for value in (x, y, z)):
                continue
            if radius_field:
                radius = float(point[3])
            elif diameter_field:
                radius = 0.5 * float(point[3])
            else:
                radius = self.default_radius_m
            if not math.isfinite(radius):
                radius = self.default_radius_m
            radius = _clamp(radius, self.min_radius_m, self.max_radius_m)

            if transform is not None:
                x, y, z = self._transform_point(transform, (x, y, z))

            marker = Marker()
            marker.header.stamp = msg.header.stamp
            marker.header.frame_id = output_frame
            marker.ns = self.marker_namespace
            marker.id = len(markers)
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = z
            marker.pose.orientation.w = 1.0
            marker.scale.x = radius * 2.0
            marker.scale.y = radius * 2.0
            marker.scale.z = radius * 2.0
            marker.color.r = 0.15
            marker.color.g = 0.75
            marker.color.b = 1.0
            marker.color.a = 0.45
            marker.lifetime = _duration_msg(self.marker_lifetime_sec)
            markers.append(marker)

        return MarkerArray(markers=markers)

    def _lookup_transform(self, target_frame: str, source_frame: str, msg: PointCloud2):
        try:
            return self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time.from_msg(msg.header.stamp),
                timeout=Duration(seconds=0.02),
            )
        except TransformException:
            try:
                return self.tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    Time(),
                    timeout=Duration(seconds=0.02),
                )
            except TransformException as exc:
                self._log_throttled(
                    f"Waiting for TF {target_frame} <- {source_frame} before publishing camera obstacles: {exc}"
                )
                return None

    @staticmethod
    def _transform_point(transform, point: Sequence[float]) -> Tuple[float, float, float]:
        q = transform.transform.rotation
        t = transform.transform.translation
        rx, ry, rz = _rotate_vector(q.x, q.y, q.z, q.w, point)
        return rx + t.x, ry + t.y, rz + t.z

    @staticmethod
    def _first_existing(field_names: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
        field_set = set(field_names)
        for candidate in candidates:
            if candidate in field_set:
                return candidate
        return None

    def _fresh(self, stamp_s: Optional[float], now_s: float) -> bool:
        return stamp_s is not None and (now_s - stamp_s) <= self.stale_timeout_sec

    def _publish_outputs(self) -> None:
        now_s = self._now_s()

        collision_markers: List[Marker] = []
        if self._fresh(self.latest_cloud_time_s, now_s) and self.latest_cloud_markers is not None:
            collision_markers.extend(self.latest_cloud_markers.markers)
        for topic, (msg, stamp_s) in list(self.latest_additional_markers.items()):
            if now_s - stamp_s <= self.stale_timeout_sec:
                collision_markers.extend(msg.markers)
            else:
                self.latest_additional_markers.pop(topic, None)

        if collision_markers or self.last_collision_had_markers or self.publish_empty_when_stale:
            self.collision_pub.publish(MarkerArray(markers=collision_markers))
            self.last_collision_had_markers = bool(collision_markers)

        rviz_markers: List[Marker] = []
        if (
            self.relay_camera_markers_to_rviz and
            self._fresh(self.latest_camera_rviz_time_s, now_s) and
            self.latest_camera_rviz_markers is not None
        ):
            rviz_markers.extend(self.latest_camera_rviz_markers.markers)
        elif self._fresh(self.latest_cloud_time_s, now_s) and self.latest_cloud_markers is not None:
            rviz_markers.extend(self.latest_cloud_markers.markers)

        if rviz_markers or self.last_rviz_had_markers:
            self.rviz_pub.publish(MarkerArray(markers=rviz_markers))
            self.last_rviz_had_markers = bool(rviz_markers)

    def _log_throttled(self, message: str) -> None:
        now_s = self._now_s()
        if now_s - self.last_log_time_s >= self.log_period_s:
            self.get_logger().info(message)
            self.last_log_time_s = now_s


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CameraBodySphereCloudBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
