#!/usr/bin/env python3
import csv
import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import rclpy
from geometry_msgs.msg import Pose, PoseStamped
from rclpy.node import Node
from std_msgs.msg import UInt8


@dataclass
class StampedPose:
    receipt_time_s: float
    message_time_s: float
    pose: Pose


class RmpGoalLogger(Node):
    def __init__(self) -> None:
        super().__init__("rmp_goal_logger")

        self.declare_parameter("sample_rate_hz", 200.0)
        self.declare_parameter("output_directory", "~/ros2_ws/log/rmp_goal")
        self.declare_parameter("output_prefix", "rmp_goal")
        self.declare_parameter("external_goal_topic", "/RMP_goal")
        self.declare_parameter("controller_goal_topic", "/goal_pose")
        self.declare_parameter("rmp_flag_topic", "/RMP_flag")
        self.declare_parameter("active_flag_value", 1)
        self.declare_parameter("skip_until_first_goal", True)
        self.declare_parameter("flush_every", 1)

        self.active_flag_value = int(self.get_parameter("active_flag_value").value)
        self.skip_until_first_goal = self._as_bool(
            self.get_parameter("skip_until_first_goal").value
        )
        self.flush_every = max(int(self.get_parameter("flush_every").value), 1)

        self.latest_external_goal: Optional[StampedPose] = None
        self.latest_controller_goal: Optional[StampedPose] = None
        self.latest_flag: Optional[int] = None
        self.sample_index = 0
        self.rows_since_flush = 0

        output_dir = os.path.expanduser(str(self.get_parameter("output_directory").value))
        os.makedirs(output_dir, exist_ok=True)
        prefix = str(self.get_parameter("output_prefix").value)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(output_dir, f"{prefix}_{stamp}.csv")
        self.csv_file = open(self.csv_path, "w", newline="", buffering=1)
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self._header())
        self.csv_writer.writeheader()

        self.create_subscription(
            Pose,
            str(self.get_parameter("external_goal_topic").value),
            self._on_external_goal,
            10,
        )
        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("controller_goal_topic").value),
            self._on_controller_goal,
            10,
        )
        self.create_subscription(
            UInt8,
            str(self.get_parameter("rmp_flag_topic").value),
            self._on_flag,
            10,
        )

        sample_rate_hz = max(float(self.get_parameter("sample_rate_hz").value), 1.0)
        self.timer = self.create_timer(1.0 / sample_rate_hz, self._write_sample)

        self.get_logger().info(
            f"RMP goal logging at {sample_rate_hz:.1f} Hz to {self.csv_path}"
        )

    def destroy_node(self) -> bool:
        if self.csv_file:
            self.csv_file.flush()
            self.csv_file.close()
        return super().destroy_node()

    def _on_external_goal(self, msg: Pose) -> None:
        now_s = self._now_s()
        self.latest_external_goal = StampedPose(now_s, math.nan, msg)

    def _on_controller_goal(self, msg: PoseStamped) -> None:
        now_s = self._now_s()
        message_time_s = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        self.latest_controller_goal = StampedPose(now_s, message_time_s, msg.pose)

    def _on_flag(self, msg: UInt8) -> None:
        self.latest_flag = int(msg.data)

    def _write_sample(self) -> None:
        if self.skip_until_first_goal and self.latest_controller_goal is None:
            return

        now_s = self._now_s()
        wall_time = datetime.now().isoformat(timespec="microseconds")
        controller = self.latest_controller_goal
        external = self.latest_external_goal
        row = {
            "timestamp_iso": wall_time,
            "timestamp_ros_s": f"{now_s:.9f}",
            "sample_index": self.sample_index,
            "rmp_flag": "" if self.latest_flag is None else self.latest_flag,
            "rmp_active": "" if self.latest_flag is None else int(self.latest_flag == self.active_flag_value),
            "controller_goal_received": int(controller is not None),
            "controller_goal_age_s": self._age(now_s, controller),
            "controller_goal_msg_stamp_s": self._message_stamp(controller),
            **self._pose_fields("controller_goal", controller),
            "external_goal_received": int(external is not None),
            "external_goal_age_s": self._age(now_s, external),
            **self._pose_fields("external_goal", external),
        }
        self.csv_writer.writerow(row)
        self.sample_index += 1
        self.rows_since_flush += 1
        if self.rows_since_flush >= self.flush_every:
            self.csv_file.flush()
            self.rows_since_flush = 0

    def _age(self, now_s: float, stamped: Optional[StampedPose]) -> str:
        if stamped is None:
            return ""
        return f"{now_s - stamped.receipt_time_s:.9f}"

    def _message_stamp(self, stamped: Optional[StampedPose]) -> str:
        if stamped is None or not math.isfinite(stamped.message_time_s):
            return ""
        return f"{stamped.message_time_s:.9f}"

    def _pose_fields(self, prefix: str, stamped: Optional[StampedPose]) -> dict:
        if stamped is None:
            return {
                f"{prefix}_x": "",
                f"{prefix}_y": "",
                f"{prefix}_z": "",
                f"{prefix}_qx": "",
                f"{prefix}_qy": "",
                f"{prefix}_qz": "",
                f"{prefix}_qw": "",
            }
        pose = stamped.pose
        return {
            f"{prefix}_x": f"{pose.position.x:.9f}",
            f"{prefix}_y": f"{pose.position.y:.9f}",
            f"{prefix}_z": f"{pose.position.z:.9f}",
            f"{prefix}_qx": f"{pose.orientation.x:.9f}",
            f"{prefix}_qy": f"{pose.orientation.y:.9f}",
            f"{prefix}_qz": f"{pose.orientation.z:.9f}",
            f"{prefix}_qw": f"{pose.orientation.w:.9f}",
        }

    def _now_s(self) -> float:
        return float(self.get_clock().now().nanoseconds) * 1e-9

    @staticmethod
    def _as_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("1", "true", "yes", "on")
        return bool(value)

    @staticmethod
    def _header() -> list:
        return [
            "timestamp_iso",
            "timestamp_ros_s",
            "sample_index",
            "rmp_flag",
            "rmp_active",
            "controller_goal_received",
            "controller_goal_age_s",
            "controller_goal_msg_stamp_s",
            "controller_goal_x",
            "controller_goal_y",
            "controller_goal_z",
            "controller_goal_qx",
            "controller_goal_qy",
            "controller_goal_qz",
            "controller_goal_qw",
            "external_goal_received",
            "external_goal_age_s",
            "external_goal_x",
            "external_goal_y",
            "external_goal_z",
            "external_goal_qx",
            "external_goal_qy",
            "external_goal_qz",
            "external_goal_qw",
        ]


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RmpGoalLogger()
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
