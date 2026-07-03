#!/usr/bin/env python3

import csv
import os
import threading
import time
from datetime import datetime
from typing import List

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from sensor_msgs.msg import JointState, Range
from std_srvs.srv import Trigger


JOINT_NAMES = ["base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"]


class RmpDataRecorder(Node):
    def __init__(self) -> None:
        super().__init__("rmp_data_recorder")

        self.declare_parameter("recording_rate", 100.0)
        self.declare_parameter(
            "output_directory",
            os.path.expanduser("~/ros2_ws/data/rmp_datasets"),
        )
        self.declare_parameter("output_prefix", "rmp_dataset")
        self.declare_parameter("mode", "simulation")
        self.declare_parameter("auto_start", True)
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter(
            "range_topics",
            [f"/raw_distance{index}" for index in range(1, 21)],
        )

        self.recording_rate = float(self.get_parameter("recording_rate").value)
        self.output_directory = str(self.get_parameter("output_directory").value)
        self.output_prefix = str(self.get_parameter("output_prefix").value)
        self.mode = str(self.get_parameter("mode").value)
        self.auto_start = self._as_bool(self.get_parameter("auto_start").value)
        self.joint_state_topic = str(self.get_parameter("joint_state_topic").value)
        self.range_topics = list(self.get_parameter("range_topics").value)

        os.makedirs(self.output_directory, exist_ok=True)

        self.cb_group = ReentrantCallbackGroup()
        self.data_lock = threading.Lock()
        self.file_lock = threading.Lock()

        self.latest_joint_positions = [float("nan")] * len(JOINT_NAMES)
        self.latest_joint_velocities = [float("nan")] * len(JOINT_NAMES)
        self.latest_ranges = [float("nan")] * len(self.range_topics)
        self.range_column_names = [
            self._topic_column_name(topic) for topic in self.range_topics
        ]

        self.prev_joint_positions = None
        self.prev_joint_time = None

        self.is_recording = False
        self.recording_path = None
        self.recording_handle = None
        self.recording_writer = None

        self.create_subscription(
            JointState,
            self.joint_state_topic,
            self.on_joint_state,
            10,
            callback_group=self.cb_group,
        )

        self.range_subs = []
        for index, topic in enumerate(self.range_topics):
            self.range_subs.append(
                self.create_subscription(
                    Range,
                    topic,
                    lambda msg, idx=index: self.on_range(msg, idx),
                    10,
                    callback_group=self.cb_group,
                )
            )

        self.start_service = self.create_service(
            Trigger,
            "~/start",
            self.on_start_recording,
            callback_group=self.cb_group,
        )
        self.stop_service = self.create_service(
            Trigger,
            "~/stop",
            self.on_stop_recording,
            callback_group=self.cb_group,
        )

        period = max(1.0 / max(self.recording_rate, 1e-3), 1e-3)
        self.recording_timer = self.create_timer(
            period,
            self.record_once,
            callback_group=self.cb_group,
        )

        self.get_logger().info(f"Recorder mode: {self.mode}")
        self.get_logger().info(f"Joint topic: {self.joint_state_topic}")
        self.get_logger().info(f"Range topics: {', '.join(self.range_topics)}")
        self.get_logger().info(f"Output directory: {self.output_directory}")

        if self.auto_start:
            self.start_recording()

    def on_joint_state(self, msg: JointState) -> None:
        positions = [float("nan")] * len(JOINT_NAMES)
        velocities = [float("nan")] * len(JOINT_NAMES)

        if len(msg.name) >= len(JOINT_NAMES):
            index_map = {name: idx for idx, name in enumerate(msg.name)}
            matched = all(name in index_map for name in JOINT_NAMES)
            if matched:
                for idx, joint_name in enumerate(JOINT_NAMES):
                    source_index = index_map[joint_name]
                    if source_index < len(msg.position):
                        positions[idx] = float(msg.position[source_index])
                    if source_index < len(msg.velocity):
                        velocities[idx] = float(msg.velocity[source_index])
            else:
                self._fill_joint_by_index(msg, positions, velocities)
        else:
            self._fill_joint_by_index(msg, positions, velocities)

        now_time = time.time()
        if any(not self._is_finite(value) for value in velocities):
            if self.prev_joint_positions is not None and self.prev_joint_time is not None:
                dt = now_time - self.prev_joint_time
                if 1e-4 < dt < 1.0:
                    velocities = [
                        (positions[idx] - self.prev_joint_positions[idx]) / dt
                        if self._is_finite(positions[idx]) and self._is_finite(self.prev_joint_positions[idx])
                        else float("nan")
                        for idx in range(len(JOINT_NAMES))
                    ]

        with self.data_lock:
            self.latest_joint_positions = positions
            self.latest_joint_velocities = velocities

        self.prev_joint_positions = positions
        self.prev_joint_time = now_time

    def on_range(self, msg: Range, index: int) -> None:
        with self.data_lock:
            if index < len(self.latest_ranges):
                self.latest_ranges[index] = float(msg.range)

    def on_start_recording(self, request, response):
        del request
        success, message = self.start_recording()
        response.success = success
        response.message = message
        return response

    def on_stop_recording(self, request, response):
        del request
        success, message = self.stop_recording()
        response.success = success
        response.message = message
        return response

    def start_recording(self):
        with self.file_lock:
            if self.is_recording:
                return True, f"Already recording: {self.recording_path}"

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.output_prefix}_{self.mode}_{timestamp}.csv"
            self.recording_path = os.path.join(self.output_directory, filename)

            self.recording_handle = open(self.recording_path, "w", newline="")
            self.recording_handle.write(f"# mode,{self.mode}\n")
            self.recording_handle.write(f"# started_at,{datetime.now().isoformat()}\n")
            self.recording_handle.write(f"# joint_state_topic,{self.joint_state_topic}\n")
            self.recording_handle.write(f"# range_topics,{';'.join(self.range_topics)}\n")
            self.recording_writer = csv.writer(self.recording_handle)
            self.recording_writer.writerow(self._header())
            self.recording_handle.flush()
            self.is_recording = True

        message = f"Recording started: {self.recording_path}"
        self.get_logger().info(message)
        return True, message

    def stop_recording(self):
        with self.file_lock:
            if not self.is_recording:
                return False, "Not recording"

            self.is_recording = False
            if self.recording_handle is not None:
                self.recording_handle.write(f"# stopped_at,{datetime.now().isoformat()}\n")
                self.recording_handle.flush()
                self.recording_handle.close()
            path = self.recording_path
            self.recording_handle = None
            self.recording_writer = None
            self.recording_path = None

        message = f"Recording stopped: {path}"
        self.get_logger().info(message)
        return True, message

    def record_once(self) -> None:
        if not self.is_recording:
            return

        with self.data_lock:
            row = [
                datetime.now().isoformat(timespec="milliseconds"),
                f"{time.time():.6f}",
                self.mode,
                *self.latest_joint_positions,
                *self.latest_joint_velocities,
                *self.latest_ranges,
            ]

        with self.file_lock:
            if not self.is_recording or self.recording_writer is None or self.recording_handle is None:
                return
            self.recording_writer.writerow(self._format_row(row))
            self.recording_handle.flush()

    def destroy_node(self):
        if self.is_recording:
            self.stop_recording()
        super().destroy_node()

    def _fill_joint_by_index(
        self,
        msg: JointState,
        positions: List[float],
        velocities: List[float],
    ) -> None:
        for idx in range(min(len(JOINT_NAMES), len(msg.position))):
            positions[idx] = float(msg.position[idx])
        for idx in range(min(len(JOINT_NAMES), len(msg.velocity))):
            velocities[idx] = float(msg.velocity[idx])

    def _header(self):
        header = [
            "timestamp_iso",
            "timestamp_unix",
            "mode",
        ]
        header.extend([f"q{i + 1}" for i in range(len(JOINT_NAMES))])
        header.extend([f"qd{i + 1}" for i in range(len(JOINT_NAMES))])
        header.extend(self.range_column_names)
        return header

    @staticmethod
    def _topic_column_name(topic: str) -> str:
        cleaned = topic.strip("/").replace("/", "_").replace("-", "_")
        return cleaned or "range"

    def _as_bool(self, value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def _format_row(self, row):
        formatted = []
        for value in row:
            if isinstance(value, str):
                formatted.append(value)
            elif self._is_finite(value):
                formatted.append(f"{float(value):.6f}")
            else:
                formatted.append("")
        return formatted

    def _is_finite(self, value) -> bool:
        try:
            return float(value) == float(value) and abs(float(value)) != float("inf")
        except (TypeError, ValueError):
            return False


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RmpDataRecorder()
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
