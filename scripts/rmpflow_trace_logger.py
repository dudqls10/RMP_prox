#!/usr/bin/env python3
import csv
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import rclpy
from geometry_msgs.msg import Pose, PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState, Range
from std_msgs.msg import Float64MultiArray, UInt8
from visualization_msgs.msg import Marker, MarkerArray


SENSOR_NAMES = [
    "tof6_1_L",
    "tof6_1_F",
    "tof6_1_R",
    "tof6_1_U",
    "tof_S",
    "tof_E",
    "tof_N",
    "tof_W",
    "tof3_1_S",
    "tof3_1_W",
    "tof3_1_N",
    "tof3_1_E",
    "tof2_1_E",
    "tof2_1_S",
    "tof2_1_W",
    "tof2_1_N",
    "tof2_E",
    "tof2_S",
    "tof2_W",
    "tof2_N",
]

SENSOR_PARENT_LINKS = [
    "link5",
    "link5",
    "link5",
    "link5",
    "link3_5",
    "link3_5",
    "link3_5",
    "link3_5",
    "link3_5",
    "link3_5",
    "link3_5",
    "link3_5",
    "link2",
    "link2",
    "link2",
    "link2",
    "link2",
    "link2",
    "link2",
    "link2",
]

LEAF_ABLATION_GROUPS = {
    1: "cspace_target",
    2: "joint_limits",
    3: "joint_velocity_cap",
    4: "target",
    5: "collision",
    6: "tangent_escape",
    7: "damping",
    8: "body_target",
    9: "axis_target_x",
    10: "axis_target_y",
    11: "axis_target_z",
    12: "wrist_axis_target",
    13: "external",
    99: "other",
}


@dataclass
class StampedValue:
    stamp_s: float
    data: Any


class RmpflowTraceLogger(Node):
    DEBUG_FIELDS = [
        "goal_error_m",
        "tcp_z_m",
        "link6_z_m",
        "min_external_clearance_m",
        "min_body_clearance_m",
        "joint_velocity_norm",
        "min_z_safety_triggered",
        "min_link_z_m",
        "min_joint_z_m",
        "min_control_point_z_m",
        "min_body_obstacle_z_m",
    ]

    def __init__(self) -> None:
        super().__init__("rmpflow_trace_logger")

        self.declare_parameter("log_rate_hz", 1.0)
        self.declare_parameter("console_summary", True)
        self.declare_parameter("csv_enabled", True)
        self.declare_parameter("output_directory", "~/ros2_ws/log/rmpflow_trace")
        self.declare_parameter("output_prefix", "rmpflow_trace")
        self.declare_parameter("active_flag_value", 1)
        self.declare_parameter("range_scale", 0.001)
        self.declare_parameter("minimum_hold_distance", 0.05)
        self.declare_parameter("valid_margin", 0.001)
        self.declare_parameter("trigger_distance", 0.29)
        self.declare_parameter("rmp_flag_topic", "/RMP_flag")
        self.declare_parameter("external_goal_topic", "/RMP_goal")
        self.declare_parameter("controller_goal_topic", "/goal_pose")
        self.declare_parameter("joint_state_topic", "/rb10/joint_states")
        self.declare_parameter("command_topic", "/position_controllers/commands")
        self.declare_parameter("target_q_topic", "/target_q")
        self.declare_parameter("target_metric_topic", "/target_metric")
        self.declare_parameter("debug_state_topic", "/rmp_debug_state")
        self.declare_parameter("rmp_ee_pose_topic", "/rmp_ee_pose")
        self.declare_parameter("rmp_joint_accel_topic", "/rmp_joint_accel")
        self.declare_parameter("rmp_tcp_accel_topic", "/rmp_tcp_accel")
        self.declare_parameter("tangent_escape_rmp_data_topic", "/tangent_escape_rmp_data")
        self.declare_parameter("tangent_escape_dual_solve_topic", "/tangent_escape_dual_solve")
        self.declare_parameter("leaf_ablation_topic", "/rmp_leaf_ablation")
        self.declare_parameter("tangent_escape_geometry_data_topic", "/tangent_escape_geometry_data")
        self.declare_parameter(
            "stability_certificate_data_topic",
            "/rmp_stability_certificate",
        )
        self.declare_parameter("obstacle_marker_topic", "/obstacles")
        self.declare_parameter("repulsion_metric_marker_topic", "/repulsion_metric_markers")
        self.declare_parameter("tcp_accel_marker_topic", "/tcp_accel_marker")
        self.declare_parameter(
            "range_topics",
            [
                "/proximity_distance1",
                "/proximity_distance2",
                "/proximity_distance3",
                "/proximity_distance4",
                "/proximity_distance5",
                "/proximity_distance6",
                "/proximity_distance7",
                "/proximity_distance8",
                "/proximity_distance9",
                "/proximity_distance10",
                "/proximity_distance11",
                "/proximity_distance12",
            ],
        )

        self.console_summary = self._as_bool(self.get_parameter("console_summary").value)
        self.csv_enabled = self._as_bool(self.get_parameter("csv_enabled").value)
        self.active_flag_value = int(self.get_parameter("active_flag_value").value)
        self.range_scale = float(self.get_parameter("range_scale").value)
        self.minimum_hold_distance = float(self.get_parameter("minimum_hold_distance").value)
        self.valid_margin = float(self.get_parameter("valid_margin").value)
        self.trigger_distance = float(self.get_parameter("trigger_distance").value)
        self.range_topics = list(self.get_parameter("range_topics").value)
        self.range_labels = [self._topic_label(topic) for topic in self.range_topics]

        self.latest: Dict[str, StampedValue] = {}
        self.latest_intervals: Dict[str, float] = {}
        self.latest_ranges: Dict[str, StampedValue] = {}
        self.latest_range_intervals: Dict[str, float] = {}
        self.active_obstacle_markers: Dict[Tuple[str, int], Marker] = {}
        self.repulsion_metric_markers: Dict[Tuple[str, int], Marker] = {}
        self.tcp_accel_marker: Optional[StampedValue] = None
        self.csv_file = None
        self.csv_writer = None
        self.csv_path = ""
        self.csv_header = self._make_csv_header()
        if self.csv_enabled:
            output_dir = os.path.expanduser(str(self.get_parameter("output_directory").value))
            os.makedirs(output_dir, exist_ok=True)
            prefix = str(self.get_parameter("output_prefix").value)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.csv_path = os.path.join(output_dir, f"{prefix}_{stamp}.csv")
            self.csv_file = open(self.csv_path, "w", newline="", buffering=1)
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.csv_header)
            self.csv_writer.writeheader()

        self._create_subscriptions()

        period_s = 1.0 / max(float(self.get_parameter("log_rate_hz").value), 0.1)
        self.timer = self.create_timer(period_s, self._write_snapshot)

        if self.csv_enabled:
            self.get_logger().info(f"RMPFlow trace logging to {self.csv_path}")
        else:
            self.get_logger().info("RMPFlow trace CSV disabled; console summary only")

    def destroy_node(self) -> bool:
        if self.csv_file:
            self.csv_file.close()
        return super().destroy_node()

    def _create_subscriptions(self) -> None:
        self.create_subscription(
            UInt8,
            str(self.get_parameter("rmp_flag_topic").value),
            lambda msg: self._store("rmp_flag", int(msg.data)),
            10,
        )
        self.create_subscription(
            Pose,
            str(self.get_parameter("external_goal_topic").value),
            lambda msg: self._store("external_goal", msg),
            10,
        )
        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("controller_goal_topic").value),
            self._on_controller_goal,
            10,
        )
        self.create_subscription(
            JointState,
            str(self.get_parameter("joint_state_topic").value),
            lambda msg: self._store("joint_state", msg),
            10,
        )
        self.create_subscription(
            Float64MultiArray,
            str(self.get_parameter("command_topic").value),
            lambda msg: self._store("command_q", list(msg.data)),
            10,
        )
        self.create_subscription(
            Float64MultiArray,
            str(self.get_parameter("target_q_topic").value),
            lambda msg: self._store("target_q", list(msg.data)),
            10,
        )
        self.create_subscription(
            Float64MultiArray,
            str(self.get_parameter("target_metric_topic").value),
            lambda msg: self._store("target_metric", list(msg.data)),
            10,
        )
        self.create_subscription(
            Float64MultiArray,
            str(self.get_parameter("debug_state_topic").value),
            lambda msg: self._store("debug_state", list(msg.data)),
            10,
        )
        self.create_subscription(
            Pose,
            str(self.get_parameter("rmp_ee_pose_topic").value),
            lambda msg: self._store("rmp_ee_pose", msg),
            10,
        )
        self.create_subscription(
            Float64MultiArray,
            str(self.get_parameter("rmp_joint_accel_topic").value),
            lambda msg: self._store("rmp_joint_accel", list(msg.data)),
            10,
        )
        self.create_subscription(
            Float64MultiArray,
            str(self.get_parameter("rmp_tcp_accel_topic").value),
            lambda msg: self._store("rmp_tcp_accel", list(msg.data)),
            10,
        )
        self.create_subscription(
            Float64MultiArray,
            str(self.get_parameter("tangent_escape_rmp_data_topic").value),
            lambda msg: self._store("tangent_escape_rmp_data", list(msg.data)),
            10,
        )
        self.create_subscription(
            Float64MultiArray,
            str(self.get_parameter("tangent_escape_dual_solve_topic").value),
            lambda msg: self._store("tangent_escape_dual_solve", list(msg.data)),
            10,
        )
        self.create_subscription(
            Float64MultiArray,
            str(self.get_parameter("leaf_ablation_topic").value),
            lambda msg: self._store("leaf_ablation", list(msg.data)),
            10,
        )
        self.create_subscription(
            Float64MultiArray,
            str(self.get_parameter("tangent_escape_geometry_data_topic").value),
            lambda msg: self._store("tangent_escape_geometry_data", list(msg.data)),
            10,
        )
        self.create_subscription(
            Float64MultiArray,
            str(self.get_parameter("stability_certificate_data_topic").value),
            lambda msg: self._store("stability_certificate_data", list(msg.data)),
            10,
        )
        self.create_subscription(
            MarkerArray,
            str(self.get_parameter("obstacle_marker_topic").value),
            self._on_obstacles,
            10,
        )
        self.create_subscription(
            MarkerArray,
            str(self.get_parameter("repulsion_metric_marker_topic").value),
            self._on_repulsion_metrics,
            10,
        )
        self.create_subscription(
            Marker,
            str(self.get_parameter("tcp_accel_marker_topic").value),
            self._on_tcp_accel_marker,
            10,
        )
        for topic in self.range_topics:
            self.create_subscription(
                Range,
                topic,
                lambda msg, topic_name=topic: self._store_range(topic_name, msg),
                10,
            )

    def _store(self, key: str, data: Any) -> None:
        now_s = self._now_s()
        previous = self.latest.get(key)
        if previous is not None:
            self.latest_intervals[key] = now_s - previous.stamp_s
        self.latest[key] = StampedValue(now_s, data)

    def _on_controller_goal(self, msg: PoseStamped) -> None:
        self._store("controller_goal", msg.pose)
        stamp_s = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        self.latest["controller_goal_header_stamp_s"] = StampedValue(self._now_s(), stamp_s)

    def _store_range(self, topic: str, msg: Range) -> None:
        now_s = self._now_s()
        previous = self.latest_ranges.get(topic)
        if previous is not None:
            self.latest_range_intervals[topic] = now_s - previous.stamp_s
        self.latest_ranges[topic] = StampedValue(now_s, msg)

    def _on_obstacles(self, msg: MarkerArray) -> None:
        for marker in msg.markers:
            key = (marker.ns, marker.id)
            if marker.action == Marker.DELETEALL:
                self.active_obstacle_markers.clear()
            elif marker.action == Marker.DELETE:
                self.active_obstacle_markers.pop(key, None)
            elif marker.action == Marker.ADD:
                self.active_obstacle_markers[key] = marker
        self._store("obstacle_marker_stamp", True)

    def _on_repulsion_metrics(self, msg: MarkerArray) -> None:
        for marker in msg.markers:
            key = (marker.ns, marker.id)
            if marker.action == Marker.DELETEALL:
                self.repulsion_metric_markers.clear()
            elif marker.action == Marker.DELETE:
                self.repulsion_metric_markers.pop(key, None)
            elif marker.action == Marker.ADD:
                self.repulsion_metric_markers[key] = marker
        self._store("repulsion_metric_stamp", True)

    def _on_tcp_accel_marker(self, msg: Marker) -> None:
        if msg.action == Marker.DELETE or msg.action == Marker.DELETEALL:
            self.tcp_accel_marker = None
            return
        if msg.action == Marker.ADD:
            self.tcp_accel_marker = StampedValue(self._now_s(), msg)

    def _write_snapshot(self) -> None:
        row = self._snapshot_row()
        if self.csv_writer:
            self.csv_writer.writerow(row)
        if self.console_summary:
            self._log_console_summary(row)

    def _snapshot_row(self) -> Dict[str, Any]:
        row = {key: "" for key in self.csv_header}
        row["time_ros_s"] = f"{self._now_s():.6f}"
        row["time_wall_iso"] = datetime.now().isoformat(timespec="milliseconds")

        flag = self._latest_data("rmp_flag")
        if flag is not None:
            row["rmp_flag"] = flag
            row["rmp_active"] = int(flag == self.active_flag_value)
            row["rmp_flag_age_s"] = self._age_s("rmp_flag")

        self._fill_pose(row, "external_goal", "input_goal")
        self._fill_pose(row, "controller_goal", "controller_goal")
        self._fill_pose(row, "rmp_ee_pose", "rmp_ee")
        self._fill_joint_state(row)
        self._fill_vector(row, "target_q", "target_q", 6, "rad")
        self._fill_vector(row, "command_q", "command_q", 6, "rad")
        self._fill_vector(row, "rmp_joint_accel", "rmp_joint_accel", 6, "rad_s2")
        self._fill_cartesian_vector(row, "rmp_tcp_accel", "rmp_tcp_accel", "m_s2")
        self._fill_tangent_escape_rmp_data(row)
        self._fill_tangent_escape_dual_solve(row)
        self._fill_leaf_ablation(row)
        self._fill_tangent_escape_geometry_data(row)
        self._fill_stability_certificate_data(row)
        self._fill_target_metric(row)
        self._fill_debug_state(row)
        self._fill_marker_summaries(row)
        self._fill_range_values(row)
        return row

    def _fill_stability_certificate_data(self, row: Dict[str, Any]) -> None:
        values = self._latest_data("stability_certificate_data")
        if values is None:
            return
        row["stability_certificate_age_s"] = self._age_s(
            "stability_certificate_data"
        )
        row["stability_certificate_dt_s"] = self._interval_s(
            "stability_certificate_data"
        )
        if len(values) < 31 or int(round(float(values[0]))) != 1:
            return
        fields = [
            "stability_schema_version",
            "stability_base_gds_structural",
            "stability_environment_static",
            "stability_external_rmp_empty",
            "stability_guard_enabled",
            "stability_conditional_nonincrease",
            "stability_escape_scale",
            "stability_tank_energy",
            "stability_tank_capacity",
            "stability_escape_power_requested",
            "stability_escape_power_applied",
            "stability_solve_power",
            "stability_clamp_power",
            "stability_numerical_power",
            "stability_positive_escape_energy",
            "stability_negative_escape_energy",
            "stability_net_escape_energy",
            "stability_sample_count",
            "stability_escape_metric_trace_requested",
            "stability_escape_metric_trace_applied",
            "stability_escape_force_norm_requested",
            "stability_escape_force_norm_applied",
            "stability_raw_qdd_norm",
            "stability_command_qdd_norm",
            "stability_clamp_active",
            "stability_root_solve_offset",
            "stability_escape_active",
            "stability_tank_identity_residual",
            "stability_energy_bound_violation",
            "stability_nonincrease_upper_bound",
            "stability_initial_energy",
            "stability_base_config_profile_valid",
            "stability_base_domain_valid",
            "stability_base_metric_spd",
            "stability_base_solve_valid",
        ]
        for index, field in enumerate(fields):
            if index < len(values):
                row[field] = self._fmt(values[index])

    def _fill_pose(self, row: Dict[str, Any], key: str, prefix: str) -> None:
        pose = self._latest_data(key)
        if pose is None:
            return
        row[f"{prefix}_x"] = self._fmt(pose.position.x)
        row[f"{prefix}_y"] = self._fmt(pose.position.y)
        row[f"{prefix}_z"] = self._fmt(pose.position.z)
        row[f"{prefix}_qx"] = self._fmt(pose.orientation.x)
        row[f"{prefix}_qy"] = self._fmt(pose.orientation.y)
        row[f"{prefix}_qz"] = self._fmt(pose.orientation.z)
        row[f"{prefix}_qw"] = self._fmt(pose.orientation.w)
        row[f"{prefix}_age_s"] = self._age_s(key)
        row[f"{prefix}_dt_s"] = self._interval_s(key)
        if key == "controller_goal":
            stamp = self._latest_data("controller_goal_header_stamp_s")
            if stamp is not None:
                row["controller_goal_msg_stamp_s"] = self._fmt(stamp)

    def _fill_joint_state(self, row: Dict[str, Any]) -> None:
        joint_state = self._latest_data("joint_state")
        if joint_state is None:
            return
        row["joint_state_age_s"] = self._age_s("joint_state")
        row["joint_state_dt_s"] = self._interval_s("joint_state")
        for index in range(6):
            if index < len(joint_state.name):
                row[f"joint_{index + 1}_name"] = joint_state.name[index]
            if index < len(joint_state.position):
                row[f"joint_{index + 1}_pos_rad"] = self._fmt(joint_state.position[index])
            if index < len(joint_state.velocity):
                row[f"joint_{index + 1}_vel_rad_s"] = self._fmt(joint_state.velocity[index])
        if joint_state.velocity:
            velocity_norm = math.sqrt(sum(float(value) * float(value) for value in joint_state.velocity[:6]))
            row["joint_state_velocity_norm"] = self._fmt(velocity_norm)

    def _fill_vector(
        self,
        row: Dict[str, Any],
        key: str,
        prefix: str,
        count: int,
        unit_suffix: str,
    ) -> None:
        values = self._latest_data(key)
        if values is None:
            return
        row[f"{prefix}_age_s"] = self._age_s(key)
        row[f"{prefix}_dt_s"] = self._interval_s(key)
        norm_terms = []
        for index in range(count):
            if index < len(values):
                row[f"{prefix}_{index + 1}_{unit_suffix}"] = self._fmt(values[index])
                norm_terms.append(float(values[index]))
        if norm_terms:
            row[f"{prefix}_norm"] = self._fmt(
                math.sqrt(sum(value * value for value in norm_terms))
            )

    def _fill_cartesian_vector(
        self,
        row: Dict[str, Any],
        key: str,
        prefix: str,
        unit_suffix: str,
    ) -> None:
        values = self._latest_data(key)
        if values is None:
            return
        row[f"{prefix}_age_s"] = self._age_s(key)
        row[f"{prefix}_dt_s"] = self._interval_s(key)
        axes = ["x", "y", "z"]
        norm_terms = []
        for index, axis in enumerate(axes):
            if index < len(values):
                row[f"{prefix}_{axis}_{unit_suffix}"] = self._fmt(values[index])
                norm_terms.append(float(values[index]))
        if norm_terms:
            row[f"{prefix}_norm"] = self._fmt(
                math.sqrt(sum(value * value for value in norm_terms))
            )

    def _fill_xyz_from_array(
        self,
        row: Dict[str, Any],
        values: List[float],
        start: int,
        prefix: str,
        unit_suffix: str = "",
    ) -> None:
        axes = ["x", "y", "z"]
        norm_terms = []
        suffix = f"_{unit_suffix}" if unit_suffix else ""
        for offset, axis in enumerate(axes):
            index = start + offset
            if index < len(values):
                row[f"{prefix}_{axis}{suffix}"] = self._fmt(values[index])
                norm_terms.append(float(values[index]))
        if norm_terms:
            row[f"{prefix}_norm{suffix}"] = self._fmt(
                math.sqrt(sum(value * value for value in norm_terms))
            )

    def _fill_joint_array_from_values(
        self,
        row: Dict[str, Any],
        values: List[float],
        start: int,
        prefix: str,
    ) -> None:
        norm_terms = []
        for offset in range(6):
            index = start + offset
            if index < len(values):
                row[f"{prefix}_{offset + 1}_rad_s2"] = self._fmt(values[index])
                norm_terms.append(float(values[index]))
        if norm_terms:
            row[f"{prefix}_norm"] = self._fmt(
                math.sqrt(sum(value * value for value in norm_terms))
            )

    def _fill_tangent_escape_rmp_data(self, row: Dict[str, Any]) -> None:
        values = self._latest_data("tangent_escape_rmp_data")
        if values is None:
            return
        row["tangent_escape_rmp_age_s"] = self._age_s("tangent_escape_rmp_data")
        row["tangent_escape_rmp_dt_s"] = self._interval_s("tangent_escape_rmp_data")
        if not values:
            return
        row["tangent_escape_rmp_active"] = self._fmt(values[0])

        # An inactive tangent leaf publishes the one-value record [0].  The
        # preserved canonical-velocity mode uses schema 6/61 values; the active
        # risk-damped mode uses schema 7/68 values followed by the same
        # 17-value candidate records.
        if len(values) <= 26:
            return
        schema_id = float(values[26])
        if not math.isfinite(schema_id):
            return
        row["tangent_escape_rmp_schema_id"] = self._fmt(schema_id)
        schema = int(round(schema_id))
        prefix_size = 61 if schema == 6 else 68 if schema == 7 else 0
        if prefix_size == 0 or len(values) < prefix_size:
            return

        scalar_fields = [
            ("tangent_escape_rmp_control_point_index", 1),
            ("tangent_escape_rmp_clearance_m", 2),
            ("tangent_escape_rmp_beta", 3),
            ("tangent_escape_rmp_proximity_activation", 4),
            ("tangent_escape_rmp_blocking_activation", 5),
            ("tangent_escape_rmp_activation", 6),
            ("tangent_escape_rmp_score", 7),
            ("tangent_escape_rmp_tangent_velocity_m_s", 8),
            ("tangent_escape_rmp_desired_tangent_accel_m_s2", 9),
            ("tangent_escape_rmp_effective_metric_scalar", 10),
            ("tangent_escape_rmp_canonical_coordinate_m", 27),
            ("tangent_escape_rmp_velocity_reference_m_s", 28),
            ("tangent_escape_rmp_scalar_velocity_m_s", 29),
            ("tangent_escape_rmp_velocity_error_m_s", 30),
            ("tangent_escape_rmp_obstacle_episode_id", 31),
            ("tangent_escape_rmp_handoff_reason_id", 32),
            ("tangent_escape_rmp_handoff_phase_id", 33),
            ("tangent_escape_rmp_candidate_count", 37),
            ("tangent_escape_rmp_selected_candidate_index", 38),
            ("tangent_escape_rmp_selected_candidate_feasible", 39),
            ("tangent_escape_rmp_selected_candidate_score", 40),
            ("tangent_escape_rmp_selected_goal_score", 41),
            ("tangent_escape_rmp_selected_continuity_score", 42),
            ("tangent_escape_rmp_selected_sector_risk", 43),
            ("tangent_escape_rmp_alpha_stuck", 44),
            ("tangent_escape_rmp_lambda", 45),
            ("tangent_escape_rmp_release_velocity_gate", 46),
            ("tangent_escape_rmp_state_mode_id", 47),
            ("tangent_escape_rmp_drive_ramp", 48),
            ("tangent_escape_rmp_release_brake", 49),
            ("tangent_escape_rmp_raw_activation", 50),
            ("tangent_escape_rmp_goal_progress_m_s", 51),
            ("tangent_escape_rmp_stuck_active", 53),
            ("tangent_escape_rmp_z", 54),
            ("tangent_escape_rmp_selected_blocked_penalty", 56),
            ("tangent_escape_rmp_max_blocked_memory", 57),
            ("tangent_escape_rmp_command_distance_m", 58),
            ("tangent_escape_rmp_actual_distance_m", 59),
            ("tangent_escape_rmp_move_ratio", 60),
        ]
        for field, index in scalar_fields:
            row[field] = self._fmt(values[index])
        if schema == 7:
            risk_fields = [
                ("tangent_escape_rmp_clearance_rate_m_s", 61),
                ("tangent_escape_rmp_closing_speed_m_s", 62),
                ("tangent_escape_rmp_risk_distance_accel_m_s2", 63),
                ("tangent_escape_rmp_risk_approach_accel_m_s2", 64),
                ("tangent_escape_rmp_risk_accel_m_s2", 65),
                ("tangent_escape_rmp_tangent_damping_accel_m_s2", 66),
                ("tangent_escape_rmp_accel_input_m_s2", 67),
            ]
            for field, index in risk_fields:
                row[field] = self._fmt(values[index])
        self._fill_xyz_from_array(row, values, 11, "tangent_escape_rmp_control_point", "m")
        self._fill_xyz_from_array(row, values, 14, "tangent_escape_rmp_obstacle", "m")
        self._fill_xyz_from_array(row, values, 17, "tangent_escape_rmp_normal")
        self._fill_xyz_from_array(row, values, 20, "tangent_escape_rmp_tangent")
        self._fill_xyz_from_array(
            row,
            values,
            23,
            "tangent_escape_rmp_desired_accel",
            "m_s2",
        )
        self._fill_xyz_from_array(row, values, 34, "tangent_escape_rmp_mode_tangent")

        def vector3(start: int) -> Optional[List[float]]:
            if start + 2 >= len(values):
                return None
            result = [float(values[start + axis]) for axis in range(3)]
            if not all(math.isfinite(value) for value in result):
                return None
            return result

        def dot(lhs: Optional[List[float]], rhs: Optional[List[float]]) -> Optional[float]:
            if lhs is None or rhs is None:
                return None
            return sum(lhs[axis] * rhs[axis] for axis in range(3))

        normal = vector3(17)
        selected_tangent = vector3(20)
        desired_accel = vector3(23)
        normal_dot_tangent = dot(normal, selected_tangent)
        if normal_dot_tangent is not None:
            row["tangent_escape_rmp_normal_dot_tangent"] = self._fmt(normal_dot_tangent)
        accel_dot_normal = dot(desired_accel, normal)
        if accel_dot_normal is not None:
            row["tangent_escape_rmp_desired_accel_dot_normal_m_s2"] = self._fmt(
                accel_dot_normal
            )
        accel_dot_tangent = dot(desired_accel, selected_tangent)
        if accel_dot_tangent is not None:
            row["tangent_escape_rmp_desired_accel_dot_tangent_m_s2"] = self._fmt(
                accel_dot_tangent
            )

        candidate_count = (
            max(int(round(float(values[37]))), 0)
            if math.isfinite(float(values[37]))
            else 0
        )
        candidate_start = prefix_size
        candidate_stride = 17
        available_candidate_count = max(
            (len(values) - candidate_start) // candidate_stride,
            0,
        )
        decoded_candidate_count = min(candidate_count, available_candidate_count)
        candidates = []
        for candidate_index in range(decoded_candidate_count):
            base = candidate_start + candidate_index * candidate_stride
            direction = [
                values[base + 10],
                values[base + 11],
                values[base + 12],
            ]
            candidate_normal_dot = dot(normal, direction)
            candidates.append({
                "seed": values[base + 0],
                "feasible": values[base + 1],
                "score": values[base + 2],
                "goalScore": values[base + 3],
                "continuityScore": values[base + 4],
                "sectorRisk": values[base + 5],
                "accelerationJumpPenalty": values[base + 6],
                "blockedPenalty": values[base + 7],
                "alphaStuck": values[base + 8],
                "tangentDisplacement": values[base + 9],
                "direction": direction,
                "normalDotTangent": candidate_normal_dot,
                "metricScalar": values[base + 13],
                "scalarVelocity": values[base + 14],
                "scalarAcceleration": values[base + 15],
                "active": values[base + 16],
            })
        if candidates:
            selected_slot = None
            if math.isfinite(float(values[38])):
                selected_slot = int(round(float(values[38])))
            selected_candidate = None
            for candidate in candidates:
                if (
                    selected_slot is not None and
                    int(round(float(candidate["seed"]))) == selected_slot
                ):
                    selected_candidate = candidate
                    break
            if selected_candidate is not None:
                selected_score = float(selected_candidate["score"])
                other_scores = [
                    float(candidate["score"])
                    for candidate in candidates
                    if candidate is not selected_candidate and
                    math.isfinite(float(candidate["score"]))
                ]
                if math.isfinite(selected_score) and other_scores:
                    row["tangent_escape_rmp_selected_score_gap"] = self._fmt(
                        selected_score - max(other_scores)
                    )

            sector_risks = [
                float(candidate["sectorRisk"])
                for candidate in candidates
                if math.isfinite(float(candidate["sectorRisk"]))
            ]
            if sector_risks:
                row["tangent_escape_rmp_max_candidate_sector_risk"] = self._fmt(
                    max(sector_risks)
                )
            finite_candidate_normal_dots = [
                abs(float(candidate["normalDotTangent"]))
                for candidate in candidates
                if candidate["normalDotTangent"] is not None and
                math.isfinite(float(candidate["normalDotTangent"]))
            ]
            if finite_candidate_normal_dots:
                row["tangent_escape_rmp_max_abs_candidate_normal_dot_tangent"] = self._fmt(
                    max(finite_candidate_normal_dots)
                )
            row["tangent_escape_rmp_candidates_json"] = json.dumps(
                {
                    "count": candidate_count,
                    "decodedCount": decoded_candidate_count,
                    "candidates": candidates,
                },
                separators=(",", ":"),
            )

    def _fill_tangent_escape_dual_solve(self, row: Dict[str, Any]) -> None:
        values = self._latest_data("tangent_escape_dual_solve")
        if values is None:
            return
        row["tangent_escape_dual_solve_age_s"] = self._age_s(
            "tangent_escape_dual_solve"
        )
        row["tangent_escape_dual_solve_dt_s"] = self._interval_s(
            "tangent_escape_dual_solve"
        )
        if not values:
            return
        row["tangent_escape_dual_solve_active"] = self._fmt(values[0])
        self._fill_joint_array_from_values(row, values, 1, "tangent_escape_qdd_with")
        self._fill_joint_array_from_values(row, values, 7, "tangent_escape_qdd_without")
        self._fill_joint_array_from_values(row, values, 13, "tangent_escape_delta_qdd")
        self._fill_xyz_from_array(row, values, 19, "tangent_escape_tcp_accel_with", "m_s2")
        self._fill_xyz_from_array(row, values, 22, "tangent_escape_tcp_accel_without", "m_s2")
        self._fill_xyz_from_array(row, values, 25, "tangent_escape_delta_tcp_accel", "m_s2")
        self._fill_xyz_from_array(row, values, 32, "tangent_escape_cp_accel_with", "m_s2")
        self._fill_xyz_from_array(row, values, 35, "tangent_escape_cp_accel_without", "m_s2")
        self._fill_xyz_from_array(row, values, 38, "tangent_escape_delta_cp_accel", "m_s2")
        scalar_fields = [
            ("tangent_escape_delta_tcp_accel_dot_tangent_m_s2", 28),
            ("tangent_escape_delta_tcp_accel_dot_normal_m_s2", 29),
            ("tangent_escape_dual_activation", 30),
            ("tangent_escape_dual_effective_metric_scalar", 31),
            ("tangent_escape_delta_cp_accel_dot_tangent_m_s2", 41),
            ("tangent_escape_delta_cp_accel_dot_normal_m_s2", 42),
        ]
        for field, index in scalar_fields:
            if index < len(values):
                row[field] = self._fmt(values[index])

    def _fill_leaf_ablation(self, row: Dict[str, Any]) -> None:
        values = self._latest_data("leaf_ablation")
        if values is None:
            return
        row["leaf_ablation_age_s"] = self._age_s("leaf_ablation")
        row["leaf_ablation_dt_s"] = self._interval_s("leaf_ablation")
        if len(values) < 25 or int(round(float(values[0]))) != 1:
            return

        record_count = max(0, int(round(float(values[1]))))
        row["leaf_ablation_schema_version"] = self._fmt(values[0])
        row["leaf_ablation_record_count"] = record_count
        scalar_fields = [
            ("leaf_ablation_max_joint_accel_rad_s2", 2),
            ("leaf_ablation_raw_qdd_norm", 3),
            ("leaf_ablation_command_qdd_norm", 4),
            ("leaf_ablation_clip_direction_cosine", 5),
            ("leaf_ablation_saturated_joint_count", 6),
        ]
        for field, index in scalar_fields:
            row[field] = self._fmt(values[index])

        acceleration_limit = abs(float(values[2]))
        for joint in range(6):
            raw_value = float(values[7 + joint])
            command_value = float(values[13 + joint])
            row[f"leaf_ablation_raw_qdd_{joint + 1}_rad_s2"] = self._fmt(raw_value)
            row[f"leaf_ablation_command_qdd_{joint + 1}_rad_s2"] = self._fmt(
                command_value
            )
            row[f"leaf_ablation_clip_delta_qdd_{joint + 1}_rad_s2"] = self._fmt(
                values[19 + joint]
            )
            row[f"leaf_ablation_joint_{joint + 1}_saturated"] = int(
                abs(raw_value) > acceleration_limit + 1e-12
            )

        record_offset = 25
        record_size = 17
        available_record_count = min(
            record_count,
            max(0, (len(values) - record_offset) // record_size),
        )
        for record_index in range(available_record_count):
            offset = record_offset + record_index * record_size
            group_id = int(round(float(values[offset])))
            group_name = LEAF_ABLATION_GROUPS.get(group_id, "other")
            prefix = f"leaf_ablation_{group_name}"
            row[f"{prefix}_active"] = self._fmt(values[offset + 1])
            row[f"{prefix}_metric_trace"] = self._fmt(values[offset + 2])
            row[f"{prefix}_metric_frobenius"] = self._fmt(values[offset + 3])
            row[f"{prefix}_force_norm"] = self._fmt(values[offset + 4])
            for joint in range(6):
                row[f"{prefix}_delta_raw_qdd_{joint + 1}_rad_s2"] = self._fmt(
                    values[offset + 5 + joint]
                )
                row[f"{prefix}_delta_command_qdd_{joint + 1}_rad_s2"] = self._fmt(
                    values[offset + 11 + joint]
                )

    def _fill_tangent_escape_geometry_data(self, row: Dict[str, Any]) -> None:
        values = self._latest_data("tangent_escape_geometry_data")
        if values is None:
            return
        row["tangent_escape_geometry_age_s"] = self._age_s("tangent_escape_geometry_data")
        row["tangent_escape_geometry_dt_s"] = self._interval_s("tangent_escape_geometry_data")
        if not values:
            return
        row["tangent_escape_geometry_active"] = self._fmt(values[0])
        scalar_fields = [
            ("tangent_escape_geometry_sensor_index", 1),
            ("tangent_escape_geometry_clearance", 2),
            ("tangent_escape_geometry_center_distance", 3),
            ("tangent_escape_geometry_sensor_obstacle_dot", 25),
            ("tangent_escape_geometry_collision_obstacle_dot", 26),
            ("tangent_escape_geometry_bias_sensor_dot", 27),
            ("tangent_escape_geometry_bias_obstacle_dot", 28),
            ("tangent_escape_geometry_jacobian_frobenius_norm", 29),
            ("tangent_escape_geometry_sensor_normal_jacobian_norm", 30),
            ("tangent_escape_geometry_obstacle_direction_jacobian_norm", 31),
            ("tangent_escape_geometry_tangent_bias_jacobian_norm", 32),
            ("tangent_escape_geometry_velocity_obstacle_dot", 33),
            ("tangent_escape_geometry_velocity_tangent_dot", 34),
        ]
        for field, index in scalar_fields:
            if index < len(values):
                row[field] = self._fmt(values[index])
        if len(values) > 1:
            sensor_index = int(round(values[1]))
            if 0 <= sensor_index < len(SENSOR_NAMES):
                row["tangent_escape_geometry_sensor_index"] = sensor_index
                row["tangent_escape_geometry_sensor_name"] = SENSOR_NAMES[sensor_index]
                row["tangent_escape_geometry_sensor_parent_link"] = SENSOR_PARENT_LINKS[sensor_index]
        self._fill_xyz_from_array(row, values, 4, "tangent_escape_geometry_cp")
        self._fill_xyz_from_array(row, values, 7, "tangent_escape_geometry_obstacle")
        self._fill_xyz_from_array(row, values, 10, "tangent_escape_geometry_sensor_normal")
        self._fill_xyz_from_array(row, values, 13, "tangent_escape_geometry_obstacle_direction")
        self._fill_xyz_from_array(row, values, 16, "tangent_escape_geometry_collision_normal")
        self._fill_xyz_from_array(row, values, 19, "tangent_escape_geometry_tangent_bias")
        self._fill_xyz_from_array(row, values, 22, "tangent_escape_geometry_cp_velocity")

    def _fill_target_metric(self, row: Dict[str, Any]) -> None:
        metric = self._latest_data("target_metric")
        if metric is None:
            return
        row["target_metric_age_s"] = self._age_s("target_metric")
        row["target_metric_dt_s"] = self._interval_s("target_metric")
        for index in range(min(9, len(metric))):
            row[f"target_metric_m{index // 3}{index % 3}"] = self._fmt(metric[index])
        if len(metric) >= 9:
            row["target_metric_trace"] = self._fmt(metric[0] + metric[4] + metric[8])
            row["target_metric_frobenius"] = self._fmt(math.sqrt(sum(value * value for value in metric[:9])))

    def _fill_debug_state(self, row: Dict[str, Any]) -> None:
        debug = self._latest_data("debug_state")
        if debug is None:
            return
        row["debug_state_age_s"] = self._age_s("debug_state")
        row["debug_state_dt_s"] = self._interval_s("debug_state")
        for index, field in enumerate(self.DEBUG_FIELDS):
            if index < len(debug):
                row[field] = self._fmt(debug[index])

    def _fill_marker_summaries(self, row: Dict[str, Any]) -> None:
        row["obstacle_marker_count"] = len(self.active_obstacle_markers)
        row["obstacle_marker_update_age_s"] = self._age_s("obstacle_marker_stamp")
        row["obstacle_marker_update_dt_s"] = self._interval_s("obstacle_marker_stamp")
        obstacle_frames = sorted(
            marker.text for marker in self.active_obstacle_markers.values() if marker.text
        )
        row["obstacle_marker_frames"] = "|".join(obstacle_frames)
        row["obstacle_marker_centers"] = self._serialize_marker_centers(
            self.active_obstacle_markers
        )
        self._fill_closest_obstacle(row)

        row["repulsion_metric_dot_count"] = len(self.repulsion_metric_markers)
        row["repulsion_metric_update_age_s"] = self._age_s("repulsion_metric_stamp")
        row["repulsion_metric_update_dt_s"] = self._interval_s("repulsion_metric_stamp")
        if self.repulsion_metric_markers:
            row["repulsion_metric_max_red"] = self._fmt(
                max(marker.color.r for marker in self.repulsion_metric_markers.values())
            )
            row["repulsion_metric_max_alpha"] = self._fmt(
                max(marker.color.a for marker in self.repulsion_metric_markers.values())
            )
            row["repulsion_metric_centers"] = self._serialize_marker_centers(
                self.repulsion_metric_markers
            )

        if self.tcp_accel_marker is None:
            return
        marker = self.tcp_accel_marker.data
        row["tcp_accel_marker_age_s"] = self._fmt(self._now_s() - self.tcp_accel_marker.stamp_s)
        start, end = self._marker_start_end(marker)
        if start is None or end is None:
            return
        direction = [end[index] - start[index] for index in range(3)]
        length = math.sqrt(sum(value * value for value in direction))
        row["tcp_accel_marker_length_m"] = self._fmt(length)
        if length > 1e-12:
            row["tcp_accel_dir_x"] = self._fmt(direction[0] / length)
            row["tcp_accel_dir_y"] = self._fmt(direction[1] / length)
            row["tcp_accel_dir_z"] = self._fmt(direction[2] / length)
        row["tcp_accel_start_x"] = self._fmt(start[0])
        row["tcp_accel_start_y"] = self._fmt(start[1])
        row["tcp_accel_start_z"] = self._fmt(start[2])
        row["tcp_accel_end_x"] = self._fmt(end[0])
        row["tcp_accel_end_y"] = self._fmt(end[1])
        row["tcp_accel_end_z"] = self._fmt(end[2])

    def _fill_range_values(self, row: Dict[str, Any]) -> None:
        active_topics = []
        for topic, label in zip(self.range_topics, self.range_labels):
            stamped = self.latest_ranges.get(topic)
            if stamped is None:
                continue
            msg = stamped.data
            raw = float(msg.range)
            usable = self._range_is_usable(msg)
            effective_m = max(raw * self.range_scale, self.minimum_hold_distance)
            triggered = usable and effective_m <= self.trigger_distance
            row[f"{label}_raw"] = self._fmt(raw)
            row[f"{label}_effective_m"] = self._fmt(effective_m)
            row[f"{label}_usable"] = int(usable)
            row[f"{label}_triggered"] = int(triggered)
            row[f"{label}_age_s"] = self._fmt(self._now_s() - stamped.stamp_s)
            if topic in self.latest_range_intervals:
                row[f"{label}_dt_s"] = self._fmt(self.latest_range_intervals[topic])
            if triggered:
                active_topics.append(label)
        row["proximity_triggered_count"] = len(active_topics)
        row["proximity_triggered_topics"] = "|".join(active_topics)

    def _serialize_marker_centers(
        self,
        markers: Dict[Tuple[str, int], Marker],
        max_items: int = 80,
    ) -> str:
        entries = []
        for (namespace, marker_id), marker in sorted(markers.items(), key=lambda item: item[0]):
            position = marker.pose.position
            radius = max(float(marker.scale.x), float(marker.scale.y), float(marker.scale.z)) * 0.5
            entries.append(
                f"{namespace}:{marker_id}:"
                f"{position.x:.5f}:{position.y:.5f}:{position.z:.5f}:{radius:.5f}"
            )
            if len(entries) >= max_items:
                entries.append("truncated")
                break
        return "|".join(entries)

    def _fill_closest_obstacle(self, row: Dict[str, Any]) -> None:
        ee_pose = self._latest_data("rmp_ee_pose")
        if ee_pose is None or not self.active_obstacle_markers:
            return
        ee = ee_pose.position
        closest = None
        for (namespace, marker_id), marker in self.active_obstacle_markers.items():
            position = marker.pose.position
            radius = max(float(marker.scale.x), float(marker.scale.y), float(marker.scale.z)) * 0.5
            center_distance = math.sqrt(
                (position.x - ee.x) ** 2 +
                (position.y - ee.y) ** 2 +
                (position.z - ee.z) ** 2
            )
            surface_distance = center_distance - radius
            candidate = (surface_distance, center_distance, radius, namespace, marker_id, position)
            if closest is None or candidate[0] < closest[0]:
                closest = candidate
        if closest is None:
            return
        surface_distance, center_distance, radius, namespace, marker_id, position = closest
        row["closest_obstacle_ns"] = namespace
        row["closest_obstacle_id"] = marker_id
        row["closest_obstacle_x"] = self._fmt(position.x)
        row["closest_obstacle_y"] = self._fmt(position.y)
        row["closest_obstacle_z"] = self._fmt(position.z)
        row["closest_obstacle_radius_m"] = self._fmt(radius)
        row["closest_obstacle_center_dist_to_ee_m"] = self._fmt(center_distance)
        row["closest_obstacle_surface_dist_to_ee_m"] = self._fmt(surface_distance)

    def _range_is_usable(self, msg: Range) -> bool:
        if not math.isfinite(float(msg.range)):
            return False
        if msg.range < 0.0:
            return False
        return msg.range < (msg.max_range - self.valid_margin)

    def _marker_start_end(self, marker: Marker) -> Tuple[Optional[List[float]], Optional[List[float]]]:
        if len(marker.points) >= 2:
            start = [marker.points[0].x, marker.points[0].y, marker.points[0].z]
            end = [marker.points[1].x, marker.points[1].y, marker.points[1].z]
            return start, end
        position = marker.pose.position
        point = [position.x, position.y, position.z]
        return point, point

    def _make_csv_header(self) -> List[str]:
        header = [
            "time_ros_s",
            "time_wall_iso",
            "rmp_flag",
            "rmp_active",
            "rmp_flag_age_s",
        ]
        for prefix in ["input_goal", "controller_goal", "rmp_ee"]:
            header.extend(
                [
                    f"{prefix}_x",
                    f"{prefix}_y",
                    f"{prefix}_z",
                    f"{prefix}_qx",
                    f"{prefix}_qy",
                    f"{prefix}_qz",
                    f"{prefix}_qw",
                    f"{prefix}_age_s",
                    f"{prefix}_dt_s",
                ]
            )
        header.append("controller_goal_msg_stamp_s")
        header.append("joint_state_age_s")
        header.append("joint_state_dt_s")
        for index in range(6):
            header.extend(
                [
                    f"joint_{index + 1}_name",
                    f"joint_{index + 1}_pos_rad",
                    f"joint_{index + 1}_vel_rad_s",
                ]
            )
        header.append("joint_state_velocity_norm")
        for prefix in ["target_q", "command_q"]:
            header.append(f"{prefix}_age_s")
            header.append(f"{prefix}_dt_s")
            header.extend(f"{prefix}_{index + 1}_rad" for index in range(6))
            header.append(f"{prefix}_norm")
        header.append("rmp_joint_accel_age_s")
        header.append("rmp_joint_accel_dt_s")
        header.extend(f"rmp_joint_accel_{index + 1}_rad_s2" for index in range(6))
        header.append("rmp_joint_accel_norm")
        header.append("rmp_tcp_accel_age_s")
        header.append("rmp_tcp_accel_dt_s")
        header.extend(
            [
                "rmp_tcp_accel_x_m_s2",
                "rmp_tcp_accel_y_m_s2",
                "rmp_tcp_accel_z_m_s2",
                "rmp_tcp_accel_norm",
            ]
        )
        header.extend(
            [
                "tangent_escape_rmp_age_s",
                "tangent_escape_rmp_dt_s",
                "tangent_escape_rmp_active",
                "tangent_escape_rmp_control_point_index",
                "tangent_escape_rmp_clearance_m",
                "tangent_escape_rmp_beta",
                "tangent_escape_rmp_proximity_activation",
                "tangent_escape_rmp_blocking_activation",
                "tangent_escape_rmp_activation",
                "tangent_escape_rmp_score",
                "tangent_escape_rmp_tangent_velocity_m_s",
                "tangent_escape_rmp_desired_tangent_accel_m_s2",
                "tangent_escape_rmp_effective_metric_scalar",
                "tangent_escape_rmp_schema_id",
                "tangent_escape_rmp_clearance_rate_m_s",
                "tangent_escape_rmp_closing_speed_m_s",
                "tangent_escape_rmp_risk_distance_accel_m_s2",
                "tangent_escape_rmp_risk_approach_accel_m_s2",
                "tangent_escape_rmp_risk_accel_m_s2",
                "tangent_escape_rmp_tangent_damping_accel_m_s2",
                "tangent_escape_rmp_accel_input_m_s2",
                "tangent_escape_rmp_canonical_coordinate_m",
                "tangent_escape_rmp_velocity_reference_m_s",
                "tangent_escape_rmp_velocity_error_m_s",
                "tangent_escape_rmp_obstacle_episode_id",
                "tangent_escape_rmp_handoff_reason_id",
                "tangent_escape_rmp_handoff_phase_id",
                "tangent_escape_rmp_selected_candidate_feasible",
                "tangent_escape_rmp_selected_sector_risk",
                "tangent_escape_rmp_alpha_stuck",
                "tangent_escape_rmp_lambda",
                "tangent_escape_rmp_release_velocity_gate",
                "tangent_escape_rmp_state_mode_id",
                "tangent_escape_rmp_drive_ramp",
                "tangent_escape_rmp_release_brake",
                "tangent_escape_rmp_raw_activation",
                "tangent_escape_rmp_goal_progress_m_s",
                "tangent_escape_rmp_z",
                "tangent_escape_rmp_command_distance_m",
                "tangent_escape_rmp_actual_distance_m",
                "tangent_escape_rmp_move_ratio",
                "tangent_escape_rmp_scalar_velocity_m_s",
                "tangent_escape_rmp_candidate_count",
                "tangent_escape_rmp_selected_candidate_index",
                "tangent_escape_rmp_selected_candidate_score",
                "tangent_escape_rmp_selected_goal_score",
                "tangent_escape_rmp_selected_continuity_score",
                "tangent_escape_rmp_stuck_active",
                "tangent_escape_rmp_selected_blocked_penalty",
                "tangent_escape_rmp_max_blocked_memory",
                "tangent_escape_rmp_normal_dot_tangent",
                "tangent_escape_rmp_desired_accel_dot_normal_m_s2",
                "tangent_escape_rmp_desired_accel_dot_tangent_m_s2",
                "tangent_escape_rmp_selected_score_gap",
                "tangent_escape_rmp_max_candidate_sector_risk",
                "tangent_escape_rmp_max_abs_candidate_normal_dot_tangent",
                "tangent_escape_rmp_candidates_json",
            ]
        )
        for prefix, suffix in [
            ("tangent_escape_rmp_control_point", "m"),
            ("tangent_escape_rmp_obstacle", "m"),
            ("tangent_escape_rmp_normal", ""),
            ("tangent_escape_rmp_tangent", ""),
            ("tangent_escape_rmp_desired_accel", "m_s2"),
            ("tangent_escape_rmp_mode_tangent", ""),
        ]:
            axis_suffix = f"_{suffix}" if suffix else ""
            header.extend(f"{prefix}_{axis}{axis_suffix}" for axis in ["x", "y", "z"])
            header.append(f"{prefix}_norm{axis_suffix}")
        header.extend(
            [
                "stability_certificate_age_s",
                "stability_certificate_dt_s",
                "stability_schema_version",
                "stability_base_gds_structural",
                "stability_environment_static",
                "stability_external_rmp_empty",
                "stability_guard_enabled",
                "stability_conditional_nonincrease",
                "stability_escape_scale",
                "stability_tank_energy",
                "stability_tank_capacity",
                "stability_escape_power_requested",
                "stability_escape_power_applied",
                "stability_solve_power",
                "stability_clamp_power",
                "stability_numerical_power",
                "stability_positive_escape_energy",
                "stability_negative_escape_energy",
                "stability_net_escape_energy",
                "stability_sample_count",
                "stability_escape_metric_trace_requested",
                "stability_escape_metric_trace_applied",
                "stability_escape_force_norm_requested",
                "stability_escape_force_norm_applied",
                "stability_raw_qdd_norm",
                "stability_command_qdd_norm",
                "stability_clamp_active",
                "stability_root_solve_offset",
                "stability_escape_active",
                "stability_tank_identity_residual",
                "stability_energy_bound_violation",
                "stability_nonincrease_upper_bound",
                "stability_initial_energy",
                "stability_base_config_profile_valid",
                "stability_base_domain_valid",
                "stability_base_metric_spd",
                "stability_base_solve_valid",
            ]
        )
        header.extend(
            [
                "tangent_escape_dual_solve_age_s",
                "tangent_escape_dual_solve_dt_s",
                "tangent_escape_dual_solve_active",
            ]
        )
        for prefix in [
            "tangent_escape_qdd_with",
            "tangent_escape_qdd_without",
            "tangent_escape_delta_qdd",
        ]:
            header.extend(f"{prefix}_{index + 1}_rad_s2" for index in range(6))
            header.append(f"{prefix}_norm")
        for prefix in [
            "tangent_escape_tcp_accel_with",
            "tangent_escape_tcp_accel_without",
            "tangent_escape_delta_tcp_accel",
            "tangent_escape_cp_accel_with",
            "tangent_escape_cp_accel_without",
            "tangent_escape_delta_cp_accel",
        ]:
            header.extend(f"{prefix}_{axis}_m_s2" for axis in ["x", "y", "z"])
            header.append(f"{prefix}_norm_m_s2")
        header.extend(
            [
                "tangent_escape_delta_tcp_accel_dot_tangent_m_s2",
                "tangent_escape_delta_tcp_accel_dot_normal_m_s2",
                "tangent_escape_delta_cp_accel_dot_tangent_m_s2",
                "tangent_escape_delta_cp_accel_dot_normal_m_s2",
                "tangent_escape_dual_activation",
                "tangent_escape_dual_effective_metric_scalar",
            ]
        )
        header.extend(
            [
                "leaf_ablation_age_s",
                "leaf_ablation_dt_s",
                "leaf_ablation_schema_version",
                "leaf_ablation_record_count",
                "leaf_ablation_max_joint_accel_rad_s2",
                "leaf_ablation_raw_qdd_norm",
                "leaf_ablation_command_qdd_norm",
                "leaf_ablation_clip_direction_cosine",
                "leaf_ablation_saturated_joint_count",
            ]
        )
        for prefix in [
            "leaf_ablation_raw_qdd",
            "leaf_ablation_command_qdd",
            "leaf_ablation_clip_delta_qdd",
        ]:
            header.extend(f"{prefix}_{joint + 1}_rad_s2" for joint in range(6))
        header.extend(
            f"leaf_ablation_joint_{joint + 1}_saturated" for joint in range(6)
        )
        for group_name in LEAF_ABLATION_GROUPS.values():
            prefix = f"leaf_ablation_{group_name}"
            header.extend(
                [
                    f"{prefix}_active",
                    f"{prefix}_metric_trace",
                    f"{prefix}_metric_frobenius",
                    f"{prefix}_force_norm",
                ]
            )
            header.extend(
                f"{prefix}_delta_raw_qdd_{joint + 1}_rad_s2"
                for joint in range(6)
            )
            header.extend(
                f"{prefix}_delta_command_qdd_{joint + 1}_rad_s2"
                for joint in range(6)
            )
        header.extend(
            [
                "tangent_escape_geometry_age_s",
                "tangent_escape_geometry_dt_s",
                "tangent_escape_geometry_active",
                "tangent_escape_geometry_sensor_index",
                "tangent_escape_geometry_sensor_name",
                "tangent_escape_geometry_sensor_parent_link",
                "tangent_escape_geometry_clearance",
                "tangent_escape_geometry_center_distance",
            ]
        )
        for prefix in [
            "tangent_escape_geometry_cp",
            "tangent_escape_geometry_obstacle",
            "tangent_escape_geometry_sensor_normal",
            "tangent_escape_geometry_obstacle_direction",
            "tangent_escape_geometry_collision_normal",
            "tangent_escape_geometry_tangent_bias",
            "tangent_escape_geometry_cp_velocity",
        ]:
            header.extend(f"{prefix}_{axis}" for axis in ["x", "y", "z"])
            header.append(f"{prefix}_norm")
        header.extend(
            [
                "tangent_escape_geometry_sensor_obstacle_dot",
                "tangent_escape_geometry_collision_obstacle_dot",
                "tangent_escape_geometry_bias_sensor_dot",
                "tangent_escape_geometry_bias_obstacle_dot",
                "tangent_escape_geometry_jacobian_frobenius_norm",
                "tangent_escape_geometry_sensor_normal_jacobian_norm",
                "tangent_escape_geometry_obstacle_direction_jacobian_norm",
                "tangent_escape_geometry_tangent_bias_jacobian_norm",
                "tangent_escape_geometry_velocity_obstacle_dot",
                "tangent_escape_geometry_velocity_tangent_dot",
            ]
        )
        header.append("target_metric_age_s")
        header.append("target_metric_dt_s")
        header.extend(f"target_metric_m{row}{col}" for row in range(3) for col in range(3))
        header.extend(["target_metric_trace", "target_metric_frobenius", "debug_state_age_s"])
        header.append("debug_state_dt_s")
        header.extend(self.DEBUG_FIELDS)
        header.extend(
            [
                "obstacle_marker_count",
                "obstacle_marker_update_age_s",
                "obstacle_marker_update_dt_s",
                "obstacle_marker_frames",
                "obstacle_marker_centers",
                "closest_obstacle_ns",
                "closest_obstacle_id",
                "closest_obstacle_x",
                "closest_obstacle_y",
                "closest_obstacle_z",
                "closest_obstacle_radius_m",
                "closest_obstacle_center_dist_to_ee_m",
                "closest_obstacle_surface_dist_to_ee_m",
                "repulsion_metric_dot_count",
                "repulsion_metric_update_age_s",
                "repulsion_metric_update_dt_s",
                "repulsion_metric_max_red",
                "repulsion_metric_max_alpha",
                "repulsion_metric_centers",
                "tcp_accel_marker_age_s",
                "tcp_accel_marker_length_m",
                "tcp_accel_dir_x",
                "tcp_accel_dir_y",
                "tcp_accel_dir_z",
                "tcp_accel_start_x",
                "tcp_accel_start_y",
                "tcp_accel_start_z",
                "tcp_accel_end_x",
                "tcp_accel_end_y",
                "tcp_accel_end_z",
                "proximity_triggered_count",
                "proximity_triggered_topics",
            ]
        )
        for label in self.range_labels:
            header.extend(
                [
                    f"{label}_raw",
                    f"{label}_effective_m",
                    f"{label}_usable",
                    f"{label}_triggered",
                    f"{label}_age_s",
                    f"{label}_dt_s",
                ]
            )
        return header

    def _latest_data(self, key: str) -> Any:
        stamped = self.latest.get(key)
        if stamped is None:
            return None
        return stamped.data

    def _age_s(self, key: str) -> str:
        stamped = self.latest.get(key)
        if stamped is None:
            return ""
        return self._fmt(self._now_s() - stamped.stamp_s)

    def _interval_s(self, key: str) -> str:
        interval = self.latest_intervals.get(key)
        if interval is None:
            return ""
        return self._fmt(interval)

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    @staticmethod
    def _fmt(value: Any) -> str:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return ""
        if not math.isfinite(numeric):
            return ""
        return f"{numeric:.9g}"

    @staticmethod
    def _topic_label(topic: str) -> str:
        return topic.strip("/").replace("/", "_").replace("-", "_")

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("1", "true", "yes", "on")
        return bool(value)

    def _log_console_summary(self, row: Dict[str, Any]) -> None:
        target_q = self._compact_vector(row, "target_q", 6)
        command_q = self._compact_vector(row, "command_q", 6)
        triggered = row.get("proximity_triggered_topics", "")
        if not triggered:
            triggered = "none"
        self.get_logger().info(
            "trace "
            f"active={row.get('rmp_active', '')} "
            f"goal_err={row.get('goal_error_m', '')}m "
            f"min_ext_clear={row.get('min_external_clearance_m', '')}m "
            f"sensors={triggered} "
            f"obs={row.get('obstacle_marker_count', '')} "
            f"metric_dots={row.get('repulsion_metric_dot_count', '')} "
            f"qdd={row.get('rmp_joint_accel_norm', '')} "
            f"tcp_accel={row.get('rmp_tcp_accel_norm', '')} "
            f"tcp_accel_len={row.get('tcp_accel_marker_length_m', '')}m "
            f"target_q=[{target_q}] "
            f"cmd=[{command_q}]"
        )

    @staticmethod
    def _compact_vector(row: Dict[str, Any], prefix: str, count: int) -> str:
        values = []
        for index in range(count):
            value = row.get(f"{prefix}_{index + 1}_rad", "")
            if value == "":
                continue
            try:
                values.append(f"{float(value):.3f}")
            except ValueError:
                continue
        return ", ".join(values)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RmpflowTraceLogger()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, RuntimeError):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
