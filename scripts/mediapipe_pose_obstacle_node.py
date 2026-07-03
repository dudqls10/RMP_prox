#!/usr/bin/env python3

import math
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from visualization_msgs.msg import Marker, MarkerArray


class MediapipePoseObstacleNode(Node):
    def __init__(self) -> None:
        super().__init__("mediapipe_pose_obstacle_node")

        try:
            import mediapipe as mp  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise RuntimeError(
                "mediapipe is not installed. Install it with: "
                "python3 -m pip install --user mediapipe"
            ) from exc

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=int(self._declare("model_complexity", 1)),
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=float(self._declare("min_detection_confidence", 0.5)),
            min_tracking_confidence=float(self._declare("min_tracking_confidence", 0.5)),
        )

        self.declare_parameter("color_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter(
            "camera_info_topic", "/camera/camera/aligned_depth_to_color/camera_info"
        )
        self.declare_parameter("marker_topic", "/camera/human_pose_obstacles")
        self.declare_parameter("visibility_threshold", 0.65)
        self.declare_parameter("presence_threshold", 0.0)
        self.declare_parameter("landmark_set", "major")
        self.declare_parameter("min_valid_joints", 0)
        self.declare_parameter("min_core_joints", 1)
        self.declare_parameter("max_joint_depth_deviation_m", 0.35)
        self.declare_parameter("min_depth_m", 0.30)
        self.declare_parameter("max_depth_m", 2.50)
        self.declare_parameter("depth_patch_radius_px", 4)
        self.declare_parameter("joint_radius_m", 0.08)
        self.declare_parameter("limb_radius_m", 0.10)
        self.declare_parameter("limb_spacing_m", 0.16)
        self.declare_parameter("max_limb_spheres", 12)
        self.declare_parameter("publish_limbs", True)
        self.declare_parameter("publish_body_obstacle", True)
        self.declare_parameter("body_radius_m", 0.25)
        self.declare_parameter("body_min_joints", 1)
        self.declare_parameter("publish_segmentation_obstacles", True)
        self.declare_parameter("segmentation_model_selection", 1)
        self.declare_parameter("segmentation_mode", "compact")
        self.declare_parameter("segmentation_threshold", 0.65)
        self.declare_parameter("segmentation_stride_px", 24)
        self.declare_parameter("segmentation_compact_bands", 3)
        self.declare_parameter("segmentation_marker_radius_m", 0.18)
        self.declare_parameter("max_segmentation_markers", 80)
        self.declare_parameter("segmentation_min_markers", 1)
        self.declare_parameter("process_every_n", 1)
        self.declare_parameter("max_depth_age_s", 0.30)
        self.declare_parameter("marker_lifetime_s", 1.0)
        self.declare_parameter("log_rate_hz", 2.0)

        self.visibility_threshold = float(self.get_parameter("visibility_threshold").value)
        self.presence_threshold = float(self.get_parameter("presence_threshold").value)
        self.landmark_set = str(self.get_parameter("landmark_set").value).strip().lower()
        if self.landmark_set not in ("major", "full"):
            self.get_logger().warn(
                f'Unsupported landmark_set="{self.landmark_set}". Using "major".'
            )
            self.landmark_set = "major"
        self.min_valid_joints = max(int(self.get_parameter("min_valid_joints").value), 0)
        self.min_core_joints = max(int(self.get_parameter("min_core_joints").value), 0)
        self.max_joint_depth_deviation_m = max(
            float(self.get_parameter("max_joint_depth_deviation_m").value), 0.0
        )
        self.min_depth_m = float(self.get_parameter("min_depth_m").value)
        self.max_depth_m = float(self.get_parameter("max_depth_m").value)
        self.depth_patch_radius_px = max(
            int(self.get_parameter("depth_patch_radius_px").value), 0
        )
        self.joint_radius_m = max(float(self.get_parameter("joint_radius_m").value), 0.005)
        self.limb_radius_m = max(float(self.get_parameter("limb_radius_m").value), 0.005)
        self.limb_spacing_m = max(float(self.get_parameter("limb_spacing_m").value), 0.02)
        self.max_limb_spheres = max(int(self.get_parameter("max_limb_spheres").value), 2)
        self.publish_limbs = self._bool_param("publish_limbs")
        self.publish_body_obstacle = self._bool_param("publish_body_obstacle")
        self.body_radius_m = max(float(self.get_parameter("body_radius_m").value), 0.005)
        self.body_min_joints = max(int(self.get_parameter("body_min_joints").value), 1)
        self.publish_segmentation_obstacles = self._bool_param("publish_segmentation_obstacles")
        self.segmentation_model_selection = int(
            self.get_parameter("segmentation_model_selection").value
        )
        self.segmentation_mode = str(
            self.get_parameter("segmentation_mode").value
        ).strip().lower()
        if self.segmentation_mode not in ("compact", "dense"):
            self.get_logger().warn(
                f'Unsupported segmentation_mode="{self.segmentation_mode}". Using "compact".'
            )
            self.segmentation_mode = "compact"
        self.segmentation_threshold = float(self.get_parameter("segmentation_threshold").value)
        self.segmentation_stride_px = max(
            int(self.get_parameter("segmentation_stride_px").value), 4
        )
        self.segmentation_compact_bands = max(
            int(self.get_parameter("segmentation_compact_bands").value), 1
        )
        self.segmentation_marker_radius_m = max(
            float(self.get_parameter("segmentation_marker_radius_m").value), 0.005
        )
        self.max_segmentation_markers = max(
            int(self.get_parameter("max_segmentation_markers").value), 1
        )
        self.segmentation_min_markers = max(
            int(self.get_parameter("segmentation_min_markers").value), 1
        )
        self.process_every_n = max(int(self.get_parameter("process_every_n").value), 1)
        self.max_depth_age_s = max(float(self.get_parameter("max_depth_age_s").value), 0.0)
        self.marker_lifetime_s = max(float(self.get_parameter("marker_lifetime_s").value), 0.0)
        log_rate_hz = max(float(self.get_parameter("log_rate_hz").value), 0.1)
        self.log_period_s = 1.0 / log_rate_hz
        self.last_log_time_s = 0.0

        self.depth_image: Optional[np.ndarray] = None
        self.depth_stamp_ns: Optional[int] = None
        self.depth_frame_id = ""
        self.camera_info: Optional[CameraInfo] = None
        self.frame_count = 0
        self.had_markers = False
        self.last_reject_reason = ""
        self.segmenter = None
        if self.publish_segmentation_obstacles:
            self.segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(
                model_selection=self.segmentation_model_selection
            )

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
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
        self.create_subscription(
            Image,
            str(self.get_parameter("color_topic").value),
            self._on_color_image,
            qos,
        )
        self.marker_pub = self.create_publisher(
            MarkerArray, str(self.get_parameter("marker_topic").value), 10
        )

        self.landmark_names = [landmark.name.lower() for landmark in self.mp_pose.PoseLandmark]
        self.major_landmark_indices = self._major_landmark_indices()
        self.core_landmark_indices = self._core_landmark_indices()
        self.limb_connections = self._limb_connections()

        self.get_logger().info(
            "MediaPipe pose obstacle node started: "
            f"color={self.get_parameter('color_topic').value}, "
            f"depth={self.get_parameter('depth_topic').value}, "
            f"markers={self.get_parameter('marker_topic').value}"
        )

    def _declare(self, name: str, value):
        self.declare_parameter(name, value)
        return self.get_parameter(name).value

    def _on_camera_info(self, msg: CameraInfo) -> None:
        self.camera_info = msg

    def _on_depth_image(self, msg: Image) -> None:
        depth = self._depth_to_array(msg)
        if depth is None:
            return
        self.depth_image = depth
        self.depth_stamp_ns = self._stamp_ns(msg)
        self.depth_frame_id = msg.header.frame_id

    def _on_color_image(self, msg: Image) -> None:
        self.frame_count += 1
        if self.frame_count % self.process_every_n != 0:
            return
        if self.depth_image is None or self.camera_info is None:
            self._log_throttled("Waiting for aligned depth image and camera_info.")
            return
        if self.depth_stamp_ns is not None and self.max_depth_age_s > 0.0:
            age_s = abs(self._stamp_ns(msg) - self.depth_stamp_ns) * 1e-9
            if age_s > self.max_depth_age_s:
                self._log_throttled(
                    f"Skipping pose frame: depth is too old ({age_s:.3f} s)."
                )
                return

        rgb = self._color_to_rgb_array(msg)
        if rgb is None:
            return

        rgb.flags.writeable = False
        result = self.pose.process(rgb)
        markers: List[Marker] = []
        if result.pose_landmarks:
            points = self._landmarks_to_3d(result.pose_landmarks.landmark)
            markers.extend(self._make_markers(msg, points))
        else:
            self.last_reject_reason = "no human pose landmarks detected"

        if self.publish_segmentation_obstacles:
            markers.extend(self._make_segmentation_markers(msg, rgb))

        if not markers:
            self._clear_markers(msg)
            self._log_throttled(
                "No human obstacles published"
                f"{': ' + self.last_reject_reason if self.last_reject_reason else '.'}"
            )
            return

        self.marker_pub.publish(MarkerArray(markers=markers))
        self.had_markers = True
        body_count = sum(1 for marker in markers if marker.ns == "human_pose_body")
        joint_count = sum(1 for marker in markers if marker.ns == "human_pose_joints")
        limb_count = sum(1 for marker in markers if marker.ns == "human_pose_limbs")
        segmentation_count = sum(
            1 for marker in markers if marker.ns == "human_segmentation_obstacles"
        )
        self._log_throttled(
            "Human pose obstacles: "
            f"body={body_count}, joints={joint_count}, limb_spheres={limb_count}, "
            f"segmentation_spheres={segmentation_count}"
        )

    def _landmarks_to_3d(self, landmarks) -> Dict[int, np.ndarray]:
        points: Dict[int, np.ndarray] = {}
        if self.depth_image is None or self.camera_info is None:
            self.last_reject_reason = "missing depth image or camera_info"
            return points

        height, width = self.depth_image.shape
        fx = float(self.camera_info.k[0])
        fy = float(self.camera_info.k[4])
        cx = float(self.camera_info.k[2])
        cy = float(self.camera_info.k[5])
        if fx == 0.0 or fy == 0.0:
            self.last_reject_reason = "invalid camera intrinsics"
            return points

        allowed_count = 0
        visible_count = 0
        in_image_count = 0
        depth_count = 0
        for index, landmark in enumerate(landmarks):
            if not self._landmark_allowed(index):
                continue
            allowed_count += 1
            if not self._landmark_visible(landmark):
                continue
            visible_count += 1
            if landmark.x < 0.0 or landmark.x > 1.0 or landmark.y < 0.0 or landmark.y > 1.0:
                continue
            in_image_count += 1

            u = int(round(landmark.x * float(width - 1)))
            v = int(round(landmark.y * float(height - 1)))
            z = self._depth_at(u, v)
            if z is None:
                continue
            depth_count += 1

            x = (float(u) - cx) * z / fx
            y = (float(v) - cy) * z / fy
            points[index] = np.array([x, y, z], dtype=np.float64)

        before_outlier_count = len(points)
        points = self._filter_depth_outliers(points)
        if not self._pose_has_enough_points(points):
            core_count = sum(1 for index in self.core_landmark_indices if index in points)
            self.last_reject_reason = (
                f"allowed={allowed_count}, visible={visible_count}, "
                f"in_image={in_image_count}, valid_depth={depth_count}, "
                f"after_depth_filter={len(points)}/{before_outlier_count}, "
                f"core={core_count}, required_joints={self.min_valid_joints}, "
                f"required_core={self.min_core_joints}"
            )
            return {}
        self.last_reject_reason = ""
        return points

    def _landmark_visible(self, landmark) -> bool:
        visibility = float(getattr(landmark, "visibility", 1.0))
        presence = float(getattr(landmark, "presence", 1.0))
        return visibility >= self.visibility_threshold and presence >= self.presence_threshold

    def _depth_at(self, u: int, v: int) -> Optional[float]:
        if self.depth_image is None:
            return None
        height, width = self.depth_image.shape
        radius = self.depth_patch_radius_px
        x0 = max(u - radius, 0)
        x1 = min(u + radius + 1, width)
        y0 = max(v - radius, 0)
        y1 = min(v + radius + 1, height)
        patch = self.depth_image[y0:y1, x0:x1]
        if patch.size == 0:
            return None
        valid = np.isfinite(patch) & (patch >= self.min_depth_m) & (patch <= self.max_depth_m)
        if not np.any(valid):
            return None
        return float(np.median(patch[valid]))

    def _make_markers(self, image: Image, points: Dict[int, np.ndarray]) -> List[Marker]:
        markers: List[Marker] = []
        frame_id = self.depth_frame_id or image.header.frame_id
        stamp = image.header.stamp

        for index, point in sorted(points.items()):
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = stamp
            marker.ns = "human_pose_joints"
            marker.id = index
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(point[0])
            marker.pose.position.y = float(point[1])
            marker.pose.position.z = float(point[2])
            marker.pose.orientation.w = 1.0
            marker.scale.x = self.joint_radius_m * 2.0
            marker.scale.y = self.joint_radius_m * 2.0
            marker.scale.z = self.joint_radius_m * 2.0
            marker.color.r = 0.1
            marker.color.g = 0.85
            marker.color.b = 1.0
            marker.color.a = 0.78
            self._set_marker_lifetime(marker)
            marker.text = self.landmark_names[index] if index < len(self.landmark_names) else ""
            markers.append(marker)

        if self.publish_limbs:
            markers.extend(self._make_limb_markers(frame_id, stamp, points))

        if self.publish_body_obstacle:
            body_marker = self._make_body_marker(frame_id, stamp, points)
            if body_marker is not None:
                markers.append(body_marker)

        return markers

    def _make_limb_markers(self, frame_id: str, stamp, points: Dict[int, np.ndarray]) -> List[Marker]:
        markers: List[Marker] = []
        for connection_index, (start_index, end_index) in enumerate(self.limb_connections):
            if start_index not in points or end_index not in points:
                continue
            start = points[start_index]
            end = points[end_index]
            segment = end - start
            length = float(np.linalg.norm(segment))
            if length < 1e-6:
                continue
            sphere_count = max(int(math.ceil(length / self.limb_spacing_m)) + 1, 2)
            sphere_count = min(sphere_count, self.max_limb_spheres)
            for point_index in range(sphere_count):
                alpha = float(point_index) / float(sphere_count - 1)
                point = (1.0 - alpha) * start + alpha * end
                marker = Marker()
                marker.header.frame_id = frame_id
                marker.header.stamp = stamp
                marker.ns = "human_pose_limbs"
                marker.id = 1000 + connection_index * 32 + point_index
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position.x = float(point[0])
                marker.pose.position.y = float(point[1])
                marker.pose.position.z = float(point[2])
                marker.pose.orientation.w = 1.0
                marker.scale.x = self.limb_radius_m * 2.0
                marker.scale.y = self.limb_radius_m * 2.0
                marker.scale.z = self.limb_radius_m * 2.0
                marker.color.r = 1.0
                marker.color.g = 0.72
                marker.color.b = 0.05
                marker.color.a = 0.48
                self._set_marker_lifetime(marker)
                markers.append(marker)
        return markers

    def _make_body_marker(self, frame_id: str, stamp, points: Dict[int, np.ndarray]) -> Optional[Marker]:
        if len(points) < self.body_min_joints:
            return None

        core_points = [points[index] for index in self.core_landmark_indices if index in points]
        body_points = core_points if len(core_points) >= 2 else list(points.values())
        if len(body_points) < self.body_min_joints:
            return None

        center = np.median(np.stack(body_points, axis=0), axis=0)
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = "human_pose_body"
        marker.id = 9000
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(center[0])
        marker.pose.position.y = float(center[1])
        marker.pose.position.z = float(center[2])
        marker.pose.orientation.w = 1.0
        marker.scale.x = self.body_radius_m * 2.0
        marker.scale.y = self.body_radius_m * 2.0
        marker.scale.z = self.body_radius_m * 2.0
        marker.color.r = 1.0
        marker.color.g = 0.18
        marker.color.b = 0.08
        marker.color.a = 0.34
        self._set_marker_lifetime(marker)
        return marker

    def _make_segmentation_markers(self, image: Image, rgb: np.ndarray) -> List[Marker]:
        if self.segmenter is None or self.depth_image is None or self.camera_info is None:
            return []
        if self.depth_image.shape[:2] != rgb.shape[:2]:
            self.last_reject_reason = (
                "segmentation skipped because color and aligned depth sizes differ"
            )
            return []

        result = self.segmenter.process(rgb)
        if getattr(result, "segmentation_mask", None) is None:
            self.last_reject_reason = "segmentation produced no mask"
            return []
        mask = np.asarray(result.segmentation_mask)
        valid_mask = mask >= self.segmentation_threshold
        if not np.any(valid_mask):
            self.last_reject_reason = "segmentation mask had no valid person pixels"
            return []

        fx = float(self.camera_info.k[0])
        fy = float(self.camera_info.k[4])
        cx = float(self.camera_info.k[2])
        cy = float(self.camera_info.k[5])
        if fx == 0.0 or fy == 0.0:
            self.last_reject_reason = "invalid camera intrinsics"
            return []

        frame_id = self.depth_frame_id or image.header.frame_id
        stamp = image.header.stamp
        if self.segmentation_mode == "compact":
            points = self._compact_segmentation_points(valid_mask, fx, fy, cx, cy)
        else:
            pixels = self._sample_segmentation_pixels(valid_mask)
            points = self._pixels_to_3d(pixels, fx, fy, cx, cy)

        if len(points) < self.segmentation_min_markers:
            self.last_reject_reason = (
                f"segmentation had too few valid depth samples ({len(points)})"
            )
            return []

        markers: List[Marker] = []
        for marker_index, point in enumerate(points):
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = stamp
            marker.ns = "human_segmentation_obstacles"
            marker.id = 12000 + marker_index
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(point[0])
            marker.pose.position.y = float(point[1])
            marker.pose.position.z = float(point[2])
            marker.pose.orientation.w = 1.0
            marker.scale.x = self.segmentation_marker_radius_m * 2.0
            marker.scale.y = self.segmentation_marker_radius_m * 2.0
            marker.scale.z = self.segmentation_marker_radius_m * 2.0
            marker.color.r = 0.15
            marker.color.g = 1.0
            marker.color.b = 0.28
            marker.color.a = 0.42
            self._set_marker_lifetime(marker)
            markers.append(marker)
        return markers

    def _compact_segmentation_points(
        self,
        valid: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
    ) -> List[np.ndarray]:
        ys, xs = np.nonzero(valid)
        if len(xs) == 0:
            return []

        y_min = int(np.min(ys))
        y_max = int(np.max(ys)) + 1
        band_edges = np.linspace(y_min, y_max, self.segmentation_compact_bands + 1)
        points: List[np.ndarray] = []
        for band_index in range(self.segmentation_compact_bands):
            y0 = int(math.floor(band_edges[band_index]))
            y1 = int(math.ceil(band_edges[band_index + 1]))
            if y1 <= y0:
                y1 = y0 + 1
            in_band = (ys >= y0) & (ys < y1)
            if not np.any(in_band):
                continue

            band_xs = xs[in_band]
            band_ys = ys[in_band]
            sample_count = min(len(band_xs), 600)
            if len(band_xs) > sample_count:
                selected = np.linspace(0, len(band_xs) - 1, sample_count, dtype=np.int64)
                band_xs = band_xs[selected]
                band_ys = band_ys[selected]

            z = self.depth_image[band_ys, band_xs].astype(np.float64)
            depth_valid = np.isfinite(z) & (z >= self.min_depth_m) & (z <= self.max_depth_m)
            if not np.any(depth_valid):
                continue

            valid_xs = band_xs[depth_valid].astype(np.float64)
            valid_ys = band_ys[depth_valid].astype(np.float64)
            valid_z = z[depth_valid]
            point_x = (valid_xs - cx) * valid_z / fx
            point_y = (valid_ys - cy) * valid_z / fy
            points.append(np.array([
                float(np.median(point_x)),
                float(np.median(point_y)),
                float(np.median(valid_z)),
            ], dtype=np.float64))
        return points

    def _pixels_to_3d(
        self,
        pixels: List[Tuple[int, int]],
        fx: float,
        fy: float,
        cx: float,
        cy: float,
    ) -> List[np.ndarray]:
        points: List[np.ndarray] = []
        for u, v in pixels:
            z = self._depth_at(u, v)
            if z is None:
                continue
            x = (float(u) - cx) * z / fx
            y = (float(v) - cy) * z / fy
            points.append(np.array([x, y, z], dtype=np.float64))
        return points

    def _sample_segmentation_pixels(self, valid: np.ndarray) -> List[Tuple[int, int]]:
        if not np.any(valid):
            return []

        height, width = valid.shape
        offset = self.segmentation_stride_px // 2
        pixels: List[Tuple[int, int]] = []
        for v in range(offset, height, self.segmentation_stride_px):
            for u in range(offset, width, self.segmentation_stride_px):
                if bool(valid[v, u]):
                    pixels.append((u, v))

        if not pixels:
            ys, xs = np.nonzero(valid)
            if len(xs) == 0:
                return []
            sample_count = min(len(xs), self.max_segmentation_markers)
            selected = np.linspace(0, len(xs) - 1, sample_count, dtype=np.int64)
            return [(int(xs[index]), int(ys[index])) for index in selected]

        if len(pixels) > self.max_segmentation_markers:
            selected = np.linspace(0, len(pixels) - 1, self.max_segmentation_markers)
            pixels = [pixels[int(index)] for index in selected]
        return pixels

    def _set_marker_lifetime(self, marker: Marker) -> None:
        seconds = int(self.marker_lifetime_s)
        nanoseconds = int((self.marker_lifetime_s - float(seconds)) * 1_000_000_000)
        marker.lifetime.sec = seconds
        marker.lifetime.nanosec = nanoseconds

    def _clear_markers(self, image: Image) -> None:
        if not self.had_markers:
            return
        marker = Marker()
        marker.header.frame_id = self.depth_frame_id or image.header.frame_id
        marker.header.stamp = image.header.stamp
        marker.action = Marker.DELETEALL
        self.marker_pub.publish(MarkerArray(markers=[marker]))
        self.had_markers = False

    def _color_to_rgb_array(self, image: Image) -> Optional[np.ndarray]:
        encoding = image.encoding.lower()
        channels = {
            "rgb8": 3,
            "bgr8": 3,
            "rgba8": 4,
            "bgra8": 4,
        }.get(encoding)
        if channels is None:
            self._log_throttled(f"Unsupported color image encoding: {image.encoding}")
            return None

        data = np.frombuffer(image.data, dtype=np.uint8)
        row_elems = image.step
        raw = data.reshape((image.height, row_elems))[:, : image.width * channels]
        raw = raw.reshape((image.height, image.width, channels))
        if encoding == "rgb8":
            return np.ascontiguousarray(raw[:, :, :3])
        if encoding == "bgr8":
            return np.ascontiguousarray(raw[:, :, ::-1])
        if encoding == "rgba8":
            return np.ascontiguousarray(raw[:, :, :3])
        return np.ascontiguousarray(raw[:, :, [2, 1, 0]])

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

    def _limb_connections(self) -> List[Tuple[int, int]]:
        if self.landmark_set == "major":
            names = [
                ("left_shoulder", "right_shoulder"),
                ("left_hip", "right_hip"),
                ("left_shoulder", "left_elbow"),
                ("left_elbow", "left_wrist"),
                ("right_shoulder", "right_elbow"),
                ("right_elbow", "right_wrist"),
                ("left_shoulder", "left_hip"),
                ("right_shoulder", "right_hip"),
                ("left_hip", "left_knee"),
                ("left_knee", "left_ankle"),
                ("right_hip", "right_knee"),
                ("right_knee", "right_ankle"),
            ]
            return [(self._pose_index(start), self._pose_index(end)) for start, end in names]

        connections = []
        for start, end in self.mp_pose.POSE_CONNECTIONS:
            connections.append((self._landmark_index(start), self._landmark_index(end)))
        return sorted(set(connections))

    def _major_landmark_indices(self) -> set:
        names = [
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        ]
        return {self._pose_index(name) for name in names}

    def _core_landmark_indices(self) -> set:
        names = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
        return {self._pose_index(name) for name in names}

    def _pose_index(self, name: str) -> int:
        return int(getattr(self.mp_pose.PoseLandmark, name.upper()).value)

    def _landmark_allowed(self, index: int) -> bool:
        return self.landmark_set == "full" or index in self.major_landmark_indices

    def _filter_depth_outliers(self, points: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        if self.max_joint_depth_deviation_m <= 0.0 or len(points) < 2:
            return points
        median_z = float(np.median([point[2] for point in points.values()]))
        return {
            index: point for index, point in points.items()
            if abs(float(point[2]) - median_z) <= self.max_joint_depth_deviation_m
        }

    def _pose_has_enough_points(self, points: Dict[int, np.ndarray]) -> bool:
        if len(points) < self.min_valid_joints:
            return False
        core_count = sum(1 for index in self.core_landmark_indices if index in points)
        return core_count >= self.min_core_joints

    @staticmethod
    def _landmark_index(value) -> int:
        return int(value.value if hasattr(value, "value") else value)

    def _log_throttled(self, message: str) -> None:
        now_s = float(self.get_clock().now().nanoseconds) * 1e-9
        if now_s - self.last_log_time_s >= self.log_period_s:
            self.get_logger().info(message)
            self.last_log_time_s = now_s

    def _bool_param(self, name: str) -> bool:
        value = self.get_parameter(name).value
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "on")

    @staticmethod
    def _stamp_ns(image: Image) -> int:
        return int(image.header.stamp.sec) * 1_000_000_000 + int(image.header.stamp.nanosec)

    @staticmethod
    def _needs_byteswap(image: Image) -> bool:
        machine_big_endian = sys.byteorder == "big"
        return bool(image.is_bigendian) != machine_big_endian


def main() -> None:
    rclpy.init()
    node = MediapipePoseObstacleNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
