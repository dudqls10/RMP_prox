#!/usr/bin/env python3

import math
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import TransformStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray


@dataclass
class ObstacleDetection:
    nearest_x: float
    nearest_y: float
    nearest_z: float
    nearest_distance: float
    nearest_u: int
    nearest_v: int
    body_center: Optional[Tuple[float, float, float]]
    body_radius: float
    body_height: float
    cluster_points: int


class CameraObstacleFeatureNode(Node):
    def __init__(self) -> None:
        super().__init__("camera_obstacle_feature_node")

        self.declare_parameter(
            "depth_topic", "/camera/camera/aligned_depth_to_color/image_raw"
        )
        self.declare_parameter(
            "camera_info_topic", "/camera/camera/aligned_depth_to_color/camera_info"
        )
        self.declare_parameter("point_topic", "/camera/nearest_obstacle_point")
        self.declare_parameter("distance_topic", "/camera/nearest_obstacle_distance")
        self.declare_parameter("marker_topic", "/camera/nearest_obstacle_marker")
        self.declare_parameter("body_marker_topic", "/camera/body_obstacle_markers")
        self.declare_parameter("body_marker_shape", "sphere")
        self.declare_parameter("collision_obstacle_topic", "/obstacles")
        self.declare_parameter("publish_collision_obstacles", False)
        self.declare_parameter("obstacle_output_frame", "base_link")
        self.declare_parameter("min_depth_m", 0.30)
        self.declare_parameter("max_depth_m", 1.50)
        self.declare_parameter("sample_step", 8)
        self.declare_parameter("cluster_depth_window_m", 0.35)
        self.declare_parameter("min_cluster_points", 20)
        self.declare_parameter("body_min_radius_m", 0.18)
        self.declare_parameter("body_max_radius_m", 0.45)
        self.declare_parameter("body_radius_padding_m", 0.08)
        self.declare_parameter("body_min_height_m", 0.50)
        self.declare_parameter("body_max_height_m", 1.80)
        self.declare_parameter("body_height_padding_m", 0.20)
        self.declare_parameter("roi_x_min", 0.05)
        self.declare_parameter("roi_x_max", 0.95)
        self.declare_parameter("roi_y_min", 0.05)
        self.declare_parameter("roi_y_max", 0.95)
        self.declare_parameter("marker_radius_m", 0.04)
        self.declare_parameter("log_rate_hz", 2.0)

        self.min_depth_m = float(self.get_parameter("min_depth_m").value)
        self.max_depth_m = float(self.get_parameter("max_depth_m").value)
        self.sample_step = max(int(self.get_parameter("sample_step").value), 1)
        self.cluster_depth_window_m = max(
            float(self.get_parameter("cluster_depth_window_m").value), 0.01
        )
        self.min_cluster_points = max(int(self.get_parameter("min_cluster_points").value), 1)
        self.body_min_radius_m = max(
            float(self.get_parameter("body_min_radius_m").value), 0.01
        )
        self.body_max_radius_m = max(
            float(self.get_parameter("body_max_radius_m").value), self.body_min_radius_m
        )
        self.body_radius_padding_m = max(
            float(self.get_parameter("body_radius_padding_m").value), 0.0
        )
        self.body_min_height_m = max(
            float(self.get_parameter("body_min_height_m").value), 0.01
        )
        self.body_max_height_m = max(
            float(self.get_parameter("body_max_height_m").value), self.body_min_height_m
        )
        self.body_height_padding_m = max(
            float(self.get_parameter("body_height_padding_m").value), 0.0
        )
        self.roi_x_min = self._clamp01(float(self.get_parameter("roi_x_min").value))
        self.roi_x_max = self._clamp01(float(self.get_parameter("roi_x_max").value))
        self.roi_y_min = self._clamp01(float(self.get_parameter("roi_y_min").value))
        self.roi_y_max = self._clamp01(float(self.get_parameter("roi_y_max").value))
        self.marker_radius_m = max(float(self.get_parameter("marker_radius_m").value), 0.001)
        self.body_marker_shape = str(self.get_parameter("body_marker_shape").value).lower()
        if self.body_marker_shape not in ("sphere", "cylinder"):
            self.get_logger().warn(
                f'Unsupported body_marker_shape="{self.body_marker_shape}". Using "sphere".'
            )
            self.body_marker_shape = "sphere"
        self.publish_collision_obstacles = self._bool_param("publish_collision_obstacles")
        self.obstacle_output_frame = str(self.get_parameter("obstacle_output_frame").value)
        log_rate_hz = max(float(self.get_parameter("log_rate_hz").value), 0.1)

        self.camera_info: Optional[CameraInfo] = None
        self.last_log_time_s = 0.0
        self.log_period_s = 1.0 / log_rate_hz
        self.last_detection = False
        self.last_body_detection = False
        self.last_collision_detection = False
        self.camera_body_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.create_subscription(
            CameraInfo,
            str(self.get_parameter("camera_info_topic").value),
            self._on_camera_info,
            qos,
        )
        self.create_subscription(
            Image,
            str(self.get_parameter("depth_topic").value),
            self._on_depth_image,
            qos,
        )

        self.point_pub = self.create_publisher(
            PointStamped, str(self.get_parameter("point_topic").value), 10
        )
        self.distance_pub = self.create_publisher(
            Float32, str(self.get_parameter("distance_topic").value), 10
        )
        self.marker_pub = self.create_publisher(
            Marker, str(self.get_parameter("marker_topic").value), 10
        )
        self.body_marker_pub = self.create_publisher(
            MarkerArray, str(self.get_parameter("body_marker_topic").value), 10
        )
        self.collision_obstacle_pub = None
        if self.publish_collision_obstacles:
            self.collision_obstacle_pub = self.create_publisher(
                MarkerArray, str(self.get_parameter("collision_obstacle_topic").value), 10
            )

        self.get_logger().info(
            "Depth-only obstacle feature node started: "
            f"depth={self.get_parameter('depth_topic').value}, "
            f"camera_info={self.get_parameter('camera_info_topic').value}, "
            f"range=[{self.min_depth_m:.2f}, {self.max_depth_m:.2f}] m, "
            f"sample_step={self.sample_step}"
        )
        if self.publish_collision_obstacles:
            self.get_logger().warn(
                "Publishing camera obstacles into the RMPflow collision topic. "
                f"Camera markers will be transformed into {self.obstacle_output_frame}; "
                "make sure the camera TF is calibrated before enabling robot motion."
            )

    def _on_camera_info(self, msg: CameraInfo) -> None:
        self.camera_info = msg

    def _on_depth_image(self, msg: Image) -> None:
        if self.camera_info is None:
            self._log_throttled("Waiting for camera_info before processing depth image.")
            return

        detection = self._extract_obstacle(msg, self.camera_info)
        if detection is None:
            self._publish_no_detection(msg)
            self._log_throttled("No valid obstacle point in configured depth range.")
            return

        point = PointStamped()
        point.header = msg.header
        point.point.x = detection.nearest_x
        point.point.y = detection.nearest_y
        point.point.z = detection.nearest_z
        self.point_pub.publish(point)

        distance_msg = Float32()
        distance_msg.data = float(detection.nearest_distance)
        self.distance_pub.publish(distance_msg)

        marker = self._make_marker(
            msg, detection.nearest_x, detection.nearest_y, detection.nearest_z
        )
        self.marker_pub.publish(marker)
        self._publish_body_obstacle(msg, detection)

        self._log_throttled(
            "Nearest obstacle: "
            f"pixel=({detection.nearest_u}, {detection.nearest_v}), "
            f"point=({detection.nearest_x:.3f}, {detection.nearest_y:.3f}, "
            f"{detection.nearest_z:.3f}) m, "
            f"distance={detection.nearest_distance:.3f} m, "
            f"body_points={detection.cluster_points}"
        )
        self.last_detection = True

    def _extract_obstacle(
        self, image: Image, camera_info: CameraInfo
    ) -> Optional[ObstacleDetection]:
        depth = self._depth_to_array(image)
        if depth is None:
            return None

        height, width = depth.shape
        x0, x1, y0, y1 = self._roi_bounds(width, height)
        if x1 <= x0 or y1 <= y0:
            return None

        roi = depth[y0:y1:self.sample_step, x0:x1:self.sample_step]
        if roi.size == 0:
            return None

        finite = np.isfinite(roi)
        valid = finite & (roi >= self.min_depth_m) & (roi <= self.max_depth_m)
        if not np.any(valid):
            return None

        masked = np.where(valid, roi, np.inf)
        flat_index = int(np.argmin(masked))
        row, col = np.unravel_index(flat_index, masked.shape)
        z = float(masked[row, col])
        if not math.isfinite(z):
            return None

        u = int(x0 + col * self.sample_step)
        v = int(y0 + row * self.sample_step)

        fx = float(camera_info.k[0])
        fy = float(camera_info.k[4])
        cx = float(camera_info.k[2])
        cy = float(camera_info.k[5])
        if fx == 0.0 or fy == 0.0:
            return None

        x = (float(u) - cx) * z / fx
        y = (float(v) - cy) * z / fy
        distance = math.sqrt(x * x + y * y + z * z)

        body_center = None
        body_radius = 0.0
        body_height = 0.0
        cluster_points = 0

        valid_depths = roi[valid]
        nearest_band_z = float(np.percentile(valid_depths, 5.0))
        cluster = valid & (roi <= nearest_band_z + self.cluster_depth_window_m)
        cluster_points = int(np.count_nonzero(cluster))
        if cluster_points >= self.min_cluster_points:
            rows, cols = np.nonzero(cluster)
            us = (x0 + cols * self.sample_step).astype(np.float64)
            vs = (y0 + rows * self.sample_step).astype(np.float64)
            zs = roi[cluster].astype(np.float64)
            xs = (us - cx) * zs / fx
            ys = (vs - cy) * zs / fy

            x_low, x_high = np.percentile(xs, [10.0, 90.0])
            y_low, y_high = np.percentile(ys, [10.0, 90.0])
            body_center = (
                float(np.median(xs)),
                float(0.5 * (y_low + y_high)),
                float(np.median(zs)),
            )
            body_radius = self._clamp(
                0.5 * float(x_high - x_low) + self.body_radius_padding_m,
                self.body_min_radius_m,
                self.body_max_radius_m,
            )
            body_height = self._clamp(
                float(y_high - y_low) + self.body_height_padding_m,
                self.body_min_height_m,
                self.body_max_height_m,
            )

        return ObstacleDetection(
            nearest_x=x,
            nearest_y=y,
            nearest_z=z,
            nearest_distance=distance,
            nearest_u=u,
            nearest_v=v,
            body_center=body_center,
            body_radius=body_radius,
            body_height=body_height,
            cluster_points=cluster_points,
        )

    def _depth_to_array(self, image: Image) -> Optional[np.ndarray]:
        encoding = image.encoding.upper()
        if encoding in ("16UC1", "MONO16"):
            dtype = np.dtype(np.uint16)
            data = np.frombuffer(image.data, dtype=dtype)
            if self._needs_byteswap(image):
                data = data.byteswap()
            row_elems = image.step // dtype.itemsize
            raw = data.reshape((image.height, row_elems))[:, : image.width]
            return raw.astype(np.float32) * 0.001

        if encoding == "32FC1":
            dtype = np.dtype(np.float32)
            data = np.frombuffer(image.data, dtype=dtype)
            if self._needs_byteswap(image):
                data = data.byteswap()
            row_elems = image.step // dtype.itemsize
            raw = data.reshape((image.height, row_elems))[:, : image.width]
            return raw.astype(np.float32, copy=False)

        self._log_throttled(f"Unsupported depth image encoding: {image.encoding}")
        return None

    def _roi_bounds(self, width: int, height: int) -> Tuple[int, int, int, int]:
        x_min = min(self.roi_x_min, self.roi_x_max)
        x_max = max(self.roi_x_min, self.roi_x_max)
        y_min = min(self.roi_y_min, self.roi_y_max)
        y_max = max(self.roi_y_min, self.roi_y_max)
        x0 = int(round(x_min * width))
        x1 = int(round(x_max * width))
        y0 = int(round(y_min * height))
        y1 = int(round(y_max * height))
        return max(x0, 0), min(x1, width), max(y0, 0), min(y1, height)

    def _make_marker(self, image: Image, x: float, y: float, z: float) -> Marker:
        marker = Marker()
        marker.header = image.header
        marker.ns = "camera_obstacle_feature"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z
        marker.pose.orientation.w = 1.0
        marker.scale.x = self.marker_radius_m
        marker.scale.y = self.marker_radius_m
        marker.scale.z = self.marker_radius_m
        marker.color.r = 1.0
        marker.color.g = 0.2
        marker.color.b = 0.0
        marker.color.a = 0.9
        marker.lifetime.sec = 0
        marker.lifetime.nanosec = 300_000_000
        return marker

    def _publish_body_obstacle(self, image: Image, detection: ObstacleDetection) -> None:
        if detection.body_center is None:
            self._publish_no_body_detection(image)
            return

        body_marker = self._make_body_marker(
            image.header,
            detection.body_center,
            detection.body_radius,
            detection.body_height,
            self.camera_body_axis,
        )
        self.body_marker_pub.publish(MarkerArray(markers=[body_marker]))
        self.last_body_detection = True

        if not self.publish_collision_obstacles or self.collision_obstacle_pub is None:
            return

        collision_marker = self._make_collision_marker(image, body_marker)
        if collision_marker is None:
            if self.last_collision_detection:
                self.collision_obstacle_pub.publish(MarkerArray())
                self.last_collision_detection = False
            return

        self.collision_obstacle_pub.publish(MarkerArray(markers=[collision_marker]))
        self.last_collision_detection = True

    def _make_body_marker(
        self,
        header,
        center: Tuple[float, float, float],
        radius: float,
        height: float,
        axis: np.ndarray,
    ) -> Marker:
        marker = Marker()
        marker.header = header
        marker.ns = "camera_body_obstacles"
        marker.id = 0
        marker.action = Marker.ADD
        marker.pose.position.x = center[0]
        marker.pose.position.y = center[1]
        marker.pose.position.z = center[2]
        if self.body_marker_shape == "cylinder":
            marker.type = Marker.CYLINDER
            qx, qy, qz, qw = self._quat_from_z_axis(axis)
            marker.pose.orientation.x = qx
            marker.pose.orientation.y = qy
            marker.pose.orientation.z = qz
            marker.pose.orientation.w = qw
            marker.scale.x = radius * 2.0
            marker.scale.y = radius * 2.0
            marker.scale.z = height
        else:
            marker.type = Marker.SPHERE
            marker.pose.orientation.w = 1.0
            sphere_radius = self._clamp(
                max(radius, 0.5 * height),
                self.body_min_radius_m,
                self.body_max_radius_m,
            )
            marker.scale.x = sphere_radius * 2.0
            marker.scale.y = sphere_radius * 2.0
            marker.scale.z = sphere_radius * 2.0
        marker.color.r = 1.0
        marker.color.g = 0.65
        marker.color.b = 0.0
        marker.color.a = 0.42
        marker.lifetime.sec = 0
        marker.lifetime.nanosec = 350_000_000
        return marker

    def _make_collision_marker(self, image: Image, body_marker: Marker) -> Optional[Marker]:
        source_frame = image.header.frame_id
        if not source_frame:
            self._log_throttled("Cannot publish collision obstacle: depth image has no frame_id.")
            return None

        center = np.array([
            body_marker.pose.position.x,
            body_marker.pose.position.y,
            body_marker.pose.position.z,
        ], dtype=np.float64)
        axis = self.camera_body_axis

        if self.obstacle_output_frame and self.obstacle_output_frame != source_frame:
            transform = self._lookup_transform(
                self.obstacle_output_frame,
                source_frame,
                image,
            )
            if transform is None:
                return None
            center = self._transform_point(transform, center)
            axis = self._transform_vector(transform, axis)
            output_frame = self.obstacle_output_frame
        else:
            output_frame = source_frame

        marker = Marker()
        marker.header.stamp = image.header.stamp
        marker.header.frame_id = output_frame
        marker.ns = "camera_body_obstacles"
        marker.id = 9000
        marker.type = body_marker.type
        marker.action = Marker.ADD
        marker.pose.position.x = float(center[0])
        marker.pose.position.y = float(center[1])
        marker.pose.position.z = float(center[2])
        if body_marker.type == Marker.CYLINDER:
            qx, qy, qz, qw = self._quat_from_z_axis(axis)
            marker.pose.orientation.x = qx
            marker.pose.orientation.y = qy
            marker.pose.orientation.z = qz
            marker.pose.orientation.w = qw
        else:
            marker.pose.orientation.w = 1.0
        marker.scale.x = body_marker.scale.x
        marker.scale.y = body_marker.scale.y
        marker.scale.z = body_marker.scale.z
        marker.color = body_marker.color
        marker.lifetime = body_marker.lifetime
        return marker

    def _lookup_transform(
        self,
        target_frame: str,
        source_frame: str,
        image: Image,
    ) -> Optional[TransformStamped]:
        try:
            return self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time.from_msg(image.header.stamp),
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
                    "Cannot publish collision obstacle until TF is available: "
                    f"{target_frame} <- {source_frame}: {exc}"
                )
                return None

    def _publish_no_body_detection(self, image: Image) -> None:
        if self.last_body_detection:
            marker = Marker()
            marker.header = image.header
            marker.ns = "camera_body_obstacles"
            marker.id = 0
            marker.action = Marker.DELETE
            self.body_marker_pub.publish(MarkerArray(markers=[marker]))
        self.last_body_detection = False

        if (
            self.publish_collision_obstacles and
            self.collision_obstacle_pub is not None and
            self.last_collision_detection
        ):
            self.collision_obstacle_pub.publish(MarkerArray())
            self.last_collision_detection = False

    def _publish_no_detection(self, image: Image) -> None:
        distance_msg = Float32()
        distance_msg.data = float("nan")
        self.distance_pub.publish(distance_msg)

        if self.last_detection:
            marker = Marker()
            marker.header = image.header
            marker.ns = "camera_obstacle_feature"
            marker.id = 0
            marker.action = Marker.DELETE
            self.marker_pub.publish(marker)
        self.last_detection = False
        self._publish_no_body_detection(image)

    def _log_throttled(self, message: str) -> None:
        now_s = float(self.get_clock().now().nanoseconds) * 1e-9
        if now_s - self.last_log_time_s >= self.log_period_s:
            self.get_logger().info(message)
            self.last_log_time_s = now_s

    @staticmethod
    def _needs_byteswap(image: Image) -> bool:
        machine_big_endian = sys.byteorder == "big"
        return bool(image.is_bigendian) != machine_big_endian

    @staticmethod
    def _clamp01(value: float) -> float:
        return min(max(value, 0.0), 1.0)

    @staticmethod
    def _clamp(value: float, min_value: float, max_value: float) -> float:
        return min(max(value, min_value), max_value)

    def _bool_param(self, name: str) -> bool:
        value = self.get_parameter(name).value
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "on")

    @staticmethod
    def _quat_from_z_axis(axis: np.ndarray) -> Tuple[float, float, float, float]:
        z_axis = np.asarray(axis, dtype=np.float64)
        norm = float(np.linalg.norm(z_axis))
        if norm < 1e-9:
            return 0.0, 0.0, 0.0, 1.0
        z_axis = z_axis / norm

        reference = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(reference, z_axis))) > 0.95:
            reference = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        x_axis = np.cross(reference, z_axis)
        x_axis = x_axis / max(float(np.linalg.norm(x_axis)), 1e-9)
        y_axis = np.cross(z_axis, x_axis)

        rotation = np.column_stack((x_axis, y_axis, z_axis))
        return CameraObstacleFeatureNode._quat_from_rotation_matrix(rotation)

    @staticmethod
    def _quat_from_rotation_matrix(rotation: np.ndarray) -> Tuple[float, float, float, float]:
        trace = float(rotation[0, 0] + rotation[1, 1] + rotation[2, 2])
        if trace > 0.0:
            scale = math.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * scale
            qx = (rotation[2, 1] - rotation[1, 2]) / scale
            qy = (rotation[0, 2] - rotation[2, 0]) / scale
            qz = (rotation[1, 0] - rotation[0, 1]) / scale
        elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
            scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            qw = (rotation[2, 1] - rotation[1, 2]) / scale
            qx = 0.25 * scale
            qy = (rotation[0, 1] + rotation[1, 0]) / scale
            qz = (rotation[0, 2] + rotation[2, 0]) / scale
        elif rotation[1, 1] > rotation[2, 2]:
            scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            qw = (rotation[0, 2] - rotation[2, 0]) / scale
            qx = (rotation[0, 1] + rotation[1, 0]) / scale
            qy = 0.25 * scale
            qz = (rotation[1, 2] + rotation[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            qw = (rotation[1, 0] - rotation[0, 1]) / scale
            qx = (rotation[0, 2] + rotation[2, 0]) / scale
            qy = (rotation[1, 2] + rotation[2, 1]) / scale
            qz = 0.25 * scale

        norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if norm < 1e-9:
            return 0.0, 0.0, 0.0, 1.0
        return qx / norm, qy / norm, qz / norm, qw / norm

    @staticmethod
    def _rotation_matrix_from_transform(transform: TransformStamped) -> np.ndarray:
        q = transform.transform.rotation
        x = q.x
        y = q.y
        z = q.z
        w = q.w
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if norm < 1e-9:
            return np.identity(3, dtype=np.float64)
        x /= norm
        y /= norm
        z /= norm
        w /= norm
        return np.array([
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ], dtype=np.float64)

    @classmethod
    def _transform_point(cls, transform: TransformStamped, point: np.ndarray) -> np.ndarray:
        translation = transform.transform.translation
        offset = np.array([translation.x, translation.y, translation.z], dtype=np.float64)
        return cls._rotation_matrix_from_transform(transform) @ point + offset

    @classmethod
    def _transform_vector(cls, transform: TransformStamped, vector: np.ndarray) -> np.ndarray:
        return cls._rotation_matrix_from_transform(transform) @ vector


def main() -> None:
    rclpy.init()
    node = CameraObstacleFeatureNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
