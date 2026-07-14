#!/usr/bin/env python3

import argparse
import csv
import html
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

from plot_tangent_escape_filter_debug import (
    active_rows,
    angle_degrees,
    distance_3d,
    dot,
    goal_points,
    normalize,
    parse_float,
    read_csv_rows,
    require_columns,
    resolve_input_path,
    vector,
    vector_norm,
)


DEFAULT_DATA_DIR = Path("~/ros2_ws/log/rmpflow_trace").expanduser()

JOINT_NAMES = ["base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"]

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

ROBOT_LINK_PATH = [
    "base_link",
    "link0",
    "link1",
    "link2",
    "link3",
    "link3_5",
    "link4",
    "link5",
    "link6",
    "tcp",
    "tcp_rmp",
]


def sensor_direction_suffix(name: str) -> str:
    separator = name.rfind("_")
    if separator < 0 or separator + 1 >= len(name):
        return name
    return name[separator + 1 :]


def default_urdf_path() -> Path:
    source_tree_path = Path(__file__).resolve().parents[1] / "urdf" / "rb10_1300e.urdf"
    if source_tree_path.exists():
        return source_tree_path
    try:
        from ament_index_python.packages import get_package_share_directory

        return Path(get_package_share_directory("rb10_rmpflow_rviz")) / "urdf" / "rb10_1300e.urdf"
    except Exception:
        return source_tree_path


def parse_vector_text(text: Optional[str], fallback: Sequence[float]) -> Tuple[float, float, float]:
    if not text:
        return (float(fallback[0]), float(fallback[1]), float(fallback[2]))
    parts = text.split()
    if len(parts) != 3:
        return (float(fallback[0]), float(fallback[1]), float(fallback[2]))
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def mat_identity() -> List[List[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def mat_mul(first: Sequence[Sequence[float]], second: Sequence[Sequence[float]]) -> List[List[float]]:
    out = [[0.0 for _ in range(4)] for _ in range(4)]
    for row in range(4):
        for col in range(4):
            out[row][col] = sum(first[row][idx] * second[idx][col] for idx in range(4))
    return out


def translation_matrix(xyz: Sequence[float]) -> List[List[float]]:
    matrix = mat_identity()
    matrix[0][3] = float(xyz[0])
    matrix[1][3] = float(xyz[1])
    matrix[2][3] = float(xyz[2])
    return matrix


def rotation_x(angle: float) -> List[List[float]]:
    c = math.cos(angle)
    s = math.sin(angle)
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, c, -s, 0.0],
        [0.0, s, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def rotation_y(angle: float) -> List[List[float]]:
    c = math.cos(angle)
    s = math.sin(angle)
    return [
        [c, 0.0, s, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-s, 0.0, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def rotation_z(angle: float) -> List[List[float]]:
    c = math.cos(angle)
    s = math.sin(angle)
    return [
        [c, -s, 0.0, 0.0],
        [s, c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def axis_rotation_matrix(axis: Sequence[float], angle: float) -> List[List[float]]:
    axis_unit = normalize(axis)
    if not all(math.isfinite(value) for value in axis_unit):
        return mat_identity()
    x, y, z = axis_unit
    c = math.cos(angle)
    s = math.sin(angle)
    one_c = 1.0 - c
    return [
        [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s, 0.0],
        [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s, 0.0],
        [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def origin_transform(xyz: Sequence[float], rpy: Sequence[float]) -> List[List[float]]:
    transform = translation_matrix(xyz)
    transform = mat_mul(transform, rotation_z(float(rpy[2])))
    transform = mat_mul(transform, rotation_y(float(rpy[1])))
    transform = mat_mul(transform, rotation_x(float(rpy[0])))
    return transform


def transform_point(transform: Sequence[Sequence[float]], point: Sequence[float]) -> Tuple[float, float, float]:
    return (
        transform[0][0] * point[0] + transform[0][1] * point[1] + transform[0][2] * point[2] + transform[0][3],
        transform[1][0] * point[0] + transform[1][1] * point[1] + transform[1][2] * point[2] + transform[1][3],
        transform[2][0] * point[0] + transform[2][1] * point[1] + transform[2][2] * point[2] + transform[2][3],
    )


def transform_direction(
    transform: Sequence[Sequence[float]], direction: Sequence[float]
) -> Tuple[float, float, float]:
    return (
        transform[0][0] * direction[0] + transform[0][1] * direction[1] + transform[0][2] * direction[2],
        transform[1][0] * direction[0] + transform[1][1] * direction[1] + transform[1][2] * direction[2],
        transform[2][0] * direction[0] + transform[2][1] * direction[1] + transform[2][2] * direction[2],
    )


class UrdfKinematics:
    def __init__(self, path: Path):
        self.path = path
        root = ET.parse(path).getroot()
        self.child_to_joint: Dict[str, Dict[str, object]] = {}
        self.children_by_parent: Dict[str, List[Dict[str, object]]] = {}
        for joint in root.findall("joint"):
            parent = joint.find("parent")
            child = joint.find("child")
            if parent is None or child is None:
                continue
            origin = joint.find("origin")
            axis = joint.find("axis")
            origin_xyz = parse_vector_text(
                origin.get("xyz") if origin is not None else None,
                (0.0, 0.0, 0.0),
            )
            origin_rpy = parse_vector_text(
                origin.get("rpy") if origin is not None else None,
                (0.0, 0.0, 0.0),
            )
            joint_data = {
                "name": joint.get("name", ""),
                "type": joint.get("type", "fixed"),
                "parent": parent.get("link", ""),
                "child": child.get("link", ""),
                "origin": origin_transform(origin_xyz, origin_rpy),
                "axis": parse_vector_text(axis.get("xyz") if axis is not None else None, (1.0, 0.0, 0.0)),
            }
            self.child_to_joint[str(joint_data["child"])] = joint_data
            self.children_by_parent.setdefault(str(joint_data["parent"]), []).append(joint_data)

    def forward(self, q: Sequence[float], root_link: str = "base_link") -> Dict[str, List[List[float]]]:
        q_by_name = {
            joint_name: float(q[index])
            for index, joint_name in enumerate(JOINT_NAMES)
            if index < len(q) and math.isfinite(float(q[index]))
        }
        transforms: Dict[str, List[List[float]]] = {root_link: mat_identity()}
        stack = [root_link]
        while stack:
            parent = stack.pop()
            parent_transform = transforms[parent]
            for joint in self.children_by_parent.get(parent, []):
                child = str(joint["child"])
                joint_transform = joint["origin"]
                if joint["type"] in ("revolute", "continuous"):
                    angle = q_by_name.get(str(joint["name"]), 0.0)
                    joint_transform = mat_mul(joint_transform, axis_rotation_matrix(joint["axis"], angle))
                transforms[child] = mat_mul(parent_transform, joint_transform)
                stack.append(child)
        return transforms

    def snapshot(
        self,
        q: Sequence[float],
        active_sensor_index: Optional[int],
    ) -> Dict[str, object]:
        transforms = self.forward(q)
        links = []
        for name in ROBOT_LINK_PATH:
            transform = transforms.get(name)
            if transform is None:
                continue
            links.append({"name": name, "position": vector_or_list(transform_point(transform, (0.0, 0.0, 0.0)))})

        segments = []
        for joint in self.child_to_joint.values():
            parent = str(joint.get("parent", ""))
            child = str(joint.get("child", ""))
            parent_transform = transforms.get(parent)
            child_transform = transforms.get(child)
            if parent_transform is None or child_transform is None:
                continue
            start = transform_point(parent_transform, (0.0, 0.0, 0.0))
            end = transform_point(child_transform, (0.0, 0.0, 0.0))
            if distance_3d(start, end) < 1e-6:
                continue
            segments.append({
                "parent": parent,
                "child": child,
                "joint": str(joint.get("name", "")),
                "start": vector_or_list(start),
                "end": vector_or_list(end),
            })

        sensors = []
        for index, name in enumerate(SENSOR_NAMES):
            transform = transforms.get(name)
            if transform is None:
                continue
            parent = str(self.child_to_joint.get(name, {}).get("parent", ""))
            sensors.append({
                "index": index,
                "name": name,
                "parent": parent,
                "position": vector_or_list(transform_point(transform, (0.0, 0.0, 0.0))),
                "direction": vector_or_list(normalize(transform_direction(transform, (1.0, 0.0, 0.0)))),
            })

        active_sensor = None
        if active_sensor_index is not None:
            for sensor in sensors:
                if sensor["index"] == active_sensor_index:
                    active_sensor = sensor
                    break

        duplicate_pair_sensor = None
        if (
            active_sensor is not None
            and active_sensor_index is not None
            and 0 <= active_sensor_index < len(SENSOR_PARENT_LINKS)
        ):
            active_position = active_sensor.get("position")
            active_parent = SENSOR_PARENT_LINKS[active_sensor_index]
            active_suffix = sensor_direction_suffix(SENSOR_NAMES[active_sensor_index])
            best_distance = float("inf")
            for sensor in sensors:
                index = int(sensor.get("index", -1))
                if (
                    index == active_sensor_index
                    or index < 0
                    or index >= len(SENSOR_PARENT_LINKS)
                    or SENSOR_PARENT_LINKS[index] != active_parent
                    or sensor_direction_suffix(str(sensor.get("name", ""))) != active_suffix
                ):
                    continue
                position = sensor.get("position")
                if not valid_vector(active_position) or not valid_vector(position):
                    continue
                distance = distance_3d(active_position, position)  # type: ignore[arg-type]
                if distance < 0.05 or distance >= best_distance:
                    continue
                duplicate_pair_sensor = sensor
                best_distance = distance

        joints = []
        for index, name in enumerate(JOINT_NAMES):
            value = float(q[index])
            joints.append({
                "name": name,
                "rad": finite_or_none(value),
                "deg": finite_or_none(math.degrees(value)),
            })

        return {
            "urdf": str(self.path),
            "links": links,
            "segments": segments,
            "sensors": sensors,
            "activeSensor": active_sensor,
            "duplicatePairSensor": duplicate_pair_sensor,
            "joints": joints,
        }


def accel_vector(row: Dict[str, str], prefix: str) -> Tuple[float, float, float]:
    return (
        parse_float(row, f"{prefix}_x_m_s2"),
        parse_float(row, f"{prefix}_y_m_s2"),
        parse_float(row, f"{prefix}_z_m_s2"),
    )


def row_goal(row: Dict[str, str]) -> Optional[Tuple[float, float, float]]:
    if all(key in row for key in ("controller_goal_x", "controller_goal_y", "controller_goal_z")):
        goal = vector(row, "controller_goal")
    elif all(key in row for key in ("goal_pose_x", "goal_pose_y", "goal_pose_z")):
        goal = vector(row, "goal_pose")
    elif all(key in row for key in ("input_goal_x", "input_goal_y", "input_goal_z")):
        goal = vector(row, "input_goal")
    else:
        goal = vector(row, "goal")
    if all(math.isfinite(value) for value in goal):
        return goal
    return None


def finite_or_none(value: float) -> Optional[float]:
    return float(value) if math.isfinite(value) else None


def vector_or_none(values: Sequence[float]) -> List[Optional[float]]:
    return [finite_or_none(value) for value in values]


def parse_joint_values(row: Dict[str, str]) -> Optional[List[float]]:
    column_groups = [
        [f"q{index}" for index in range(1, 7)],
        [f"joint_{index}_pos_rad" for index in range(1, 7)],
        [f"target_q_{index}_rad" for index in range(1, 7)],
        [f"command_q_{index}_rad" for index in range(1, 7)],
    ]
    for columns in column_groups:
        if not all(column in row for column in columns):
            continue
        values = [parse_float(row, column) for column in columns]
        if all(math.isfinite(value) for value in values):
            return values
    return None


def parse_active_sensor_index(row: Dict[str, str]) -> Optional[int]:
    for key in (
        "tangent_escape_cp_index",
        "tangent_escape_control_point_index",
        "tangent_escape_rmp_control_point_index",
    ):
        value = parse_float(row, key)
        if math.isfinite(value):
            index = int(round(value))
            if 0 <= index < len(SENSOR_NAMES):
                return index
    return None


def closest_sensor_index(robot: Dict[str, object], point: Sequence[float]) -> Optional[int]:
    if not all(math.isfinite(value) for value in point):
        return None
    sensors = robot.get("sensors")
    if not isinstance(sensors, list):
        return None
    best_index = None
    best_distance = float("inf")
    for sensor in sensors:
        if not isinstance(sensor, dict):
            continue
        position = sensor.get("position")
        if not valid_vector(position):
            continue
        distance = distance_3d(point, position)  # type: ignore[arg-type]
        if distance < best_distance:
            best_distance = distance
            best_index = int(sensor.get("index", -1))
    return best_index if best_index is not None and best_index >= 0 else None


def valid_vector(values: object) -> bool:
    return (
        isinstance(values, list)
        and len(values) == 3
        and all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values)
    )


def add_robot_context(
    row: Dict[str, str],
    sample: Dict[str, object],
    urdf_model: Optional[UrdfKinematics],
) -> None:
    if urdf_model is None:
        return
    q = parse_joint_values(row)
    if q is None:
        return
    cp = sample.get("cp")
    cp_vector = cp if valid_vector(cp) else None
    active_index = parse_active_sensor_index(row)
    robot = urdf_model.snapshot(q, active_index)
    if active_index is None and cp_vector is not None:
        active_index = closest_sensor_index(robot, cp_vector)  # type: ignore[arg-type]
        robot = urdf_model.snapshot(q, active_index)
    if cp_vector is not None and isinstance(robot.get("activeSensor"), dict):
        sensor_position = robot["activeSensor"].get("position")
        if valid_vector(sensor_position):
            robot["cpUrdfDistance"] = finite_or_none(
                distance_3d(cp_vector, sensor_position)  # type: ignore[arg-type]
            )
    sample["robot"] = robot
    sample["activeSensorIndex"] = active_index
    active_sensor = robot.get("activeSensor")
    if isinstance(active_sensor, dict):
        sample["activeSensorName"] = active_sensor.get("name")


def parse_candidate_payload(row: Dict[str, str]) -> Dict[str, object]:
    raw_payload = row.get("tangent_escape_candidates_json", "")
    if not raw_payload:
        return {
            "active": False,
            "candidate_count": 0,
            "selected_candidate_index": None,
            "weights": {},
            "candidates": [],
        }
    try:
        payload = json.loads(raw_payload)
    except json.JSONDecodeError:
        return {
            "active": False,
            "candidate_count": 0,
            "selected_candidate_index": None,
            "weights": {},
            "candidates": [],
        }
    if not isinstance(payload, dict):
        return {
            "active": False,
            "candidate_count": 0,
            "selected_candidate_index": None,
            "weights": {},
            "candidates": [],
        }
    return payload


def candidate_metric(payload: Dict[str, object], key: str) -> object:
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        return metrics.get(key)
    return None


def finite_row_or_metric(row: Dict[str, str], field: str, payload: Dict[str, object], metric: str) -> object:
    row_value = finite_or_none(parse_float(row, field))
    if row_value is not None:
        return row_value
    metric_value = candidate_metric(payload, metric)
    try:
        numeric = float(metric_value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if math.isfinite(numeric):
        return numeric
    return None


def active_row_time(rows: Sequence[Dict[str, str]], row: Dict[str, str], index: int) -> float:
    time_key = "time_ros_s" if row.get("time_ros_s") else "timestamp_unix"
    if time_key == "time_ros_s" and rows and not rows[0].get("time_ros_s"):
        time_key = "timestamp_unix"
    first_time = parse_float(rows[0], time_key) if rows else float("nan")
    stamp = parse_float(row, time_key)
    if math.isfinite(stamp) and math.isfinite(first_time):
        return stamp - first_time
    return float(index)


def make_sample(
    rows: Sequence[Dict[str, str]],
    row: Dict[str, str],
    index: int,
    urdf_model: Optional[UrdfKinematics],
) -> Dict[str, object]:
    normal = vector(row, "tangent_escape_normal")
    tangent = vector(row, "tangent_escape_tangent")
    raw = accel_vector(row, "tangent_escape_raw_cp_accel")
    filtered = accel_vector(row, "tangent_escape_filtered_cp_accel")
    filtered_tcp = accel_vector(row, "tangent_escape_filtered_tcp_accel")
    rmp_tcp = accel_vector(row, "rmp_tcp_accel")
    motion = filtered_tcp if all(math.isfinite(value) for value in filtered_tcp) else rmp_tcp
    normal_unit = normalize(normal)
    tangent_unit = normalize(tangent)
    raw_dot_normal = dot(raw, normal_unit)
    filtered_dot_normal = dot(filtered, normal_unit)
    raw_dot_tangent = dot(raw, tangent_unit)
    filtered_dot_tangent = dot(filtered, tangent_unit)
    active_sensor_index = parse_active_sensor_index(row)
    active_sensor_name = row.get("tangent_escape_sensor_name") or (
        SENSOR_NAMES[active_sensor_index]
        if active_sensor_index is not None and 0 <= active_sensor_index < len(SENSOR_NAMES)
        else ""
    )
    candidate_payload = parse_candidate_payload(row)
    sample: Dict[str, object] = {
        "activeIndex": index,
        "time": finite_or_none(active_row_time(rows, row, index)),
        "clearance": finite_or_none(parse_float(row, "tangent_escape_clearance")),
        "activation": finite_or_none(parse_float(row, "tangent_escape_activation")),
        "score": finite_or_none(parse_float(row, "tangent_escape_score")),
        "cpIndex": finite_or_none(parse_float(row, "tangent_escape_cp_index")),
        "activeSensorIndex": active_sensor_index,
        "activeSensorName": active_sensor_name,
        "activeSensorParent": row.get("tangent_escape_sensor_parent_link", ""),
        "cp": vector_or_none(vector(row, "tangent_escape_cp")),
        "obstacle": vector_or_none(vector(row, "tangent_escape_obstacle")),
        "goal": vector_or_none(row_goal(row) or (float("nan"), float("nan"), float("nan"))),
        "normal": vector_or_none(normal),
        "tangent": vector_or_none(tangent),
        "raw": vector_or_none(raw),
        "filtered": vector_or_none(filtered),
        "motion": vector_or_none(motion),
        "rawDotNormal": finite_or_none(raw_dot_normal),
        "filteredDotNormal": finite_or_none(filtered_dot_normal),
        "rawDotTangent": finite_or_none(raw_dot_tangent),
        "filteredDotTangent": finite_or_none(filtered_dot_tangent),
        "tangentDotNormal": finite_or_none(dot(tangent_unit, normal_unit)),
        "rawFilteredAngleDeg": finite_or_none(angle_degrees(raw, filtered)),
        "rawNorm": finite_or_none(vector_norm(raw)),
        "filteredNorm": finite_or_none(vector_norm(filtered)),
        "motionNorm": finite_or_none(vector_norm(motion)),
        "deltaQddNorm": finite_or_none(parse_float(row, "tangent_escape_delta_qdd_norm")),
        "deltaCpAccelNorm": finite_or_none(
            parse_float(row, "tangent_escape_delta_cp_accel_norm_m_s2")
        ),
        "deltaTcpAccelNorm": finite_or_none(
            parse_float(row, "tangent_escape_delta_tcp_accel_norm_m_s2")
        ),
        "scoreGap": finite_row_or_metric(
            row,
            "tangent_escape_selected_score_gap",
            candidate_payload,
            "selected_score_gap",
        ),
        "secondBestScore": finite_row_or_metric(
            row,
            "tangent_escape_selected_second_best_score",
            candidate_payload,
            "selected_second_best_score",
        ),
        "previousDirectionDot": finite_row_or_metric(
            row,
            "tangent_escape_previous_direction_dot",
            candidate_payload,
            "previous_direction_dot",
        ),
        "previousCpIndex": finite_row_or_metric(
            row,
            "tangent_escape_previous_cp_index",
            candidate_payload,
            "previous_cp_index",
        ),
        "activeCpChanged": finite_row_or_metric(
            row,
            "tangent_escape_active_cp_changed",
            candidate_payload,
            "active_cp_changed",
        ),
        "selectedDirectionChanged": finite_row_or_metric(
            row,
            "tangent_escape_selected_direction_changed",
            candidate_payload,
            "selected_direction_changed",
        ),
        "candidateData": candidate_payload,
    }
    add_robot_context(row, sample, urdf_model)
    return sample


def choose_initial_index(samples: Sequence[Dict[str, object]], args: argparse.Namespace) -> int:
    if args.sample_index is not None:
        return max(0, min(int(args.sample_index), len(samples) - 1))
    if args.time is not None:
        return min(
            range(len(samples)),
            key=lambda index: abs(float(samples[index].get("time") or 0.0) - args.time),
        )
    if args.select == "middle":
        return len(samples) // 2
    if args.select == "max-angle":
        return max(
            range(len(samples)),
            key=lambda index: float(samples[index].get("rawFilteredAngleDeg") or -1.0),
        )
    return min(
        range(len(samples)),
        key=lambda index: float(samples[index].get("clearance") or float("inf")),
    )


def html_template() -> str:
    return r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  :root {
    color-scheme: light;
    font-family: Inter, Segoe UI, Arial, sans-serif;
    --bg: #f5f6f8;
    --panel: #ffffff;
    --ink: #1e2329;
    --muted: #68707d;
    --grid: #d8dde5;
    --raw: #1f77b4;
    --filtered: #17becf;
    --motion: #111827;
    --normal: #d62728;
    --duplicate: #d62728;
    --tangent: #2ca02c;
    --candidate: #7d8794;
    --plane: rgba(44, 160, 44, 0.13);
    --obstacle: #ff8c00;
    --goal: #ffd400;
    --robot: #394150;
    --sensor: #8b5cf6;
    --sensor-ray: #6d28d9;
    --urdf-cp: #e11d48;
  }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--ink);
  }
  header {
    padding: 18px 22px 10px;
  }
  h1 {
    margin: 0 0 4px;
    font-size: 24px;
    font-weight: 700;
  }
  .sub {
    color: var(--muted);
    font-size: 13px;
  }
	  .controls {
	    display: grid;
	    grid-template-columns: auto 1fr auto auto auto;
	    gap: 10px;
	    align-items: center;
	    padding: 10px 22px 16px;
	  }
  .zoom-controls {
    display: grid;
    grid-template-columns: auto minmax(90px, 1fr) auto auto;
    gap: 8px;
    align-items: center;
    margin: 0 0 8px;
    color: var(--muted);
    font-size: 12px;
  }
	  button {
	    border: 1px solid #c9d0da;
	    background: #fff;
	    border-radius: 6px;
	    padding: 7px 10px;
	    font-size: 13px;
	    cursor: pointer;
	  }
  select {
    border: 1px solid #c9d0da;
    background: #fff;
    border-radius: 6px;
    padding: 5px 8px;
    font-size: 12px;
  }
  input[type="range"] {
    width: 100%;
  }
  .layout {
    display: grid;
    grid-template-columns: minmax(520px, 1.35fr) minmax(420px, 0.95fr);
    gap: 14px;
    padding: 0 22px 22px;
  }
  .panel {
    background: var(--panel);
    border: 1px solid #d8dde5;
    border-radius: 8px;
    padding: 12px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
  }
  .panel h2 {
    margin: 0 0 8px;
    font-size: 16px;
  }
  svg {
    width: 100%;
    height: auto;
    display: block;
  }
  #localSvg {
    cursor: grab;
    touch-action: none;
    user-select: none;
  }
  #localSvg.dragging {
    cursor: grabbing;
  }
  .legend {
    display: flex;
    flex-wrap: wrap;
    gap: 8px 14px;
    margin-top: 8px;
    color: var(--muted);
    font-size: 12px;
  }
  .legend span {
    display: inline-flex;
    align-items: center;
    gap: 5px;
  }
  .swatch {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    display: inline-block;
    border: 1px solid rgba(0,0,0,0.28);
  }
  .line-swatch {
    width: 18px;
    height: 3px;
    border-radius: 3px;
    display: inline-block;
  }
  .metrics {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 8px;
    margin-top: 10px;
  }
  .metric {
    border: 1px solid #e1e5eb;
    border-radius: 6px;
    padding: 8px;
  }
  .metric .label {
    color: var(--muted);
    font-size: 11px;
    margin-bottom: 3px;
  }
  .metric .value {
    font-variant-numeric: tabular-nums;
    font-size: 15px;
  }
  .bars {
    margin-top: 10px;
    display: grid;
    gap: 7px;
  }
  .bar-row {
    display: grid;
    grid-template-columns: 150px 1fr 68px;
    gap: 8px;
    align-items: center;
    font-size: 12px;
  }
  .bar-track {
    position: relative;
    height: 18px;
    background: #eef1f5;
    border-radius: 5px;
    overflow: hidden;
  }
  .bar-zero {
    position: absolute;
    left: 50%;
    top: 0;
    bottom: 0;
    width: 1px;
    background: #9aa3af;
  }
  .bar-fill {
    position: absolute;
    top: 2px;
    bottom: 2px;
    border-radius: 4px;
  }
  .note {
    margin-top: 10px;
    color: var(--muted);
    font-size: 12px;
    line-height: 1.45;
  }
  .candidate-section {
    margin-top: 12px;
    border-top: 1px solid #e1e5eb;
    padding-top: 10px;
  }
  .candidate-title {
    display: flex;
    justify-content: space-between;
    gap: 8px;
    align-items: baseline;
    margin-bottom: 7px;
  }
  .candidate-title h3 {
    margin: 0;
    font-size: 14px;
  }
  .candidate-summary {
    color: var(--muted);
    font-size: 12px;
    font-variant-numeric: tabular-nums;
  }
  .candidate-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 6px 10px;
    margin-bottom: 7px;
    color: var(--muted);
    font-size: 11px;
  }
  .candidate-legend span {
    display: inline-flex;
    align-items: center;
    gap: 4px;
  }
  .candidate-bars {
    display: grid;
    gap: 4px;
    max-height: 300px;
    overflow-y: auto;
    padding-right: 4px;
  }
  .candidate-row {
    display: grid;
    grid-template-columns: 32px 132px 1fr 52px;
    gap: 7px;
    align-items: center;
    font-size: 11px;
    font-variant-numeric: tabular-nums;
  }
  .candidate-row.selected {
    font-weight: 700;
    color: #1b6c2a;
  }
  .candidate-direction {
    color: var(--muted);
    overflow-wrap: anywhere;
  }
  .candidate-row.selected .candidate-direction {
    color: #1b6c2a;
  }
  .candidate-track {
    position: relative;
    height: 18px;
    background: #eef1f5;
    border-radius: 5px;
    overflow: hidden;
  }
  .candidate-track::after {
    content: "";
    position: absolute;
    top: 0;
    bottom: 0;
    left: 50%;
    width: 1px;
    background: #8f99a8;
  }
  .candidate-segment {
    position: absolute;
    top: 2px;
    bottom: 2px;
    min-width: 1px;
  }
  .candidate-selected-band {
    position: absolute;
    inset: 0;
    border: 2px solid rgba(44, 160, 44, 0.45);
    border-radius: 5px;
    pointer-events: none;
  }
  .robot-section {
    margin-top: 12px;
    border-top: 1px solid #e1e5eb;
    padding-top: 10px;
  }
  .robot-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 6px;
    margin-bottom: 8px;
    font-size: 12px;
  }
  .robot-cell {
    border: 1px solid #e1e5eb;
    border-radius: 6px;
    padding: 7px;
  }
  .robot-cell .label {
    color: var(--muted);
    font-size: 10px;
    margin-bottom: 3px;
  }
  .robot-cell .value {
    font-variant-numeric: tabular-nums;
    overflow-wrap: anywhere;
  }
  .joint-table {
    display: grid;
    grid-template-columns: 82px 1fr 1fr;
    gap: 1px;
    background: #e1e5eb;
    border: 1px solid #e1e5eb;
    border-radius: 6px;
    overflow: hidden;
    font-size: 11px;
    font-variant-numeric: tabular-nums;
  }
  .joint-table div {
    background: #fff;
    padding: 5px 6px;
  }
  .joint-table .head {
    background: #f0f3f7;
    color: var(--muted);
    font-weight: 650;
  }
</style>
</head>
<body>
<header>
  <h1>Tangent Escape Filter Report</h1>
  <div class="sub">__CSV_NAME__</div>
</header>
<div class="controls">
  <button id="prevBtn">Prev</button>
  <input id="sampleSlider" type="range" min="0" max="0" value="0" step="1">
  <button id="nextBtn">Next</button>
  <div id="sampleLabel" class="sub"></div>
  <div id="sampleTime" class="sub"></div>
</div>
<main class="layout">
	  <section class="panel">
	    <h2>Overall view</h2>
	    <svg id="overallSvg" viewBox="0 0 760 560" role="img" aria-label="Overall projected 3D view">
      <defs>
        <marker id="arrowRaw" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L10,4 L0,8 z" fill="var(--raw)"></path>
        </marker>
        <marker id="arrowFiltered" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L10,4 L0,8 z" fill="var(--filtered)"></path>
        </marker>
	        <marker id="arrowTangent" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto" markerUnits="strokeWidth">
	          <path d="M0,0 L10,4 L0,8 z" fill="var(--tangent)"></path>
	        </marker>
        <marker id="arrowMotion" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L10,4 L0,8 z" fill="var(--motion)"></path>
        </marker>
        <marker id="arrowCandidate" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L10,4 L0,8 z" fill="var(--candidate)"></path>
        </marker>
	      </defs>
    </svg>
	    <div class="legend">
	      <span><i class="line-swatch" style="background:#333"></i>active CP path</span>
	      <span><i class="swatch" style="background:var(--obstacle)"></i>obstacle</span>
      <span><i class="swatch" style="background:var(--goal)"></i>goal</span>
	      <span><i class="swatch" style="background:#fff"></i>selected CP</span>
      <span><i class="line-swatch" style="background:var(--tangent)"></i>selected escape</span>
      <span><i class="line-swatch" style="background:rgba(44,160,44,0.35)"></i>tangent plane</span>
      <span><i class="line-swatch" style="background:var(--raw)"></i>raw accel</span>
	      <span><i class="line-swatch" style="background:var(--filtered)"></i>filtered accel</span>
	      <span><i class="line-swatch" style="background:var(--motion)"></i>final motion</span>
	      <span><i class="line-swatch" style="background:var(--candidate)"></i>candidate tangent directions</span>
	      <span><i class="line-swatch" style="background:var(--robot)"></i>URDF links</span>
	      <span><i class="line-swatch" style="background:var(--sensor)"></i>URDF sensor branches</span>
	      <span><i class="line-swatch" style="background:var(--duplicate)"></i>duplicate pair</span>
	    </div>
    <div class="note">Click an active CP point/path sample in the overall view to select that moment.</div>
	  </section>
	  <section class="panel">
	    <h2>Selected moment: 3D zoom</h2>
    <div class="zoom-controls">
      <label for="zoomSlider">Zoom radius</label>
      <input id="zoomSlider" type="range" min="0.08" max="1.80" value="1.20" step="0.01">
      <span id="zoomValue"></span>
      <select id="viewSelect" aria-label="3D view direction">
        <option value="free">drag</option>
        <option value="iso">iso</option>
        <option value="xy">XY</option>
        <option value="xz">XZ</option>
        <option value="yz">YZ</option>
      </select>
      <button id="resetViewBtn" type="button">Reset view</button>
    </div>
	    <svg id="localSvg" viewBox="0 0 560 430" role="img" aria-label="Local 3D zoom view"></svg>
    <div class="metrics">
      <div class="metric"><div class="label">clearance</div><div class="value" id="clearanceValue"></div></div>
      <div class="metric"><div class="label">activation</div><div class="value" id="activationValue"></div></div>
      <div class="metric"><div class="label">raw-filtered angle</div><div class="value" id="angleValue"></div></div>
      <div class="metric"><div class="label">tangent dot normal</div><div class="value" id="orthogonalValue"></div></div>
      <div class="metric"><div class="label">delta qdd</div><div class="value" id="deltaQddValue"></div></div>
      <div class="metric"><div class="label">score gap</div><div class="value" id="scoreGapValue"></div></div>
      <div class="metric"><div class="label">prev dir dot</div><div class="value" id="previousDirectionValue"></div></div>
      <div class="metric"><div class="label">active CP changed</div><div class="value" id="activeCpChangedValue"></div></div>
    </div>
    <div class="robot-section">
      <div class="candidate-title">
        <h3>URDF robot context</h3>
        <div class="candidate-summary" id="sensorSummary"></div>
      </div>
      <div class="robot-grid" id="robotDetails"></div>
      <div class="joint-table" id="jointTable"></div>
    </div>
	    <div class="bars" id="componentBars"></div>
    <div class="candidate-section">
      <div class="candidate-title">
        <h3>Candidate score components</h3>
        <div class="candidate-summary" id="candidateSummary"></div>
      </div>
      <div class="candidate-legend">
        <span><i class="line-swatch" style="background:#4c78a8"></i>goal</span>
        <span><i class="line-swatch" style="background:#f58518"></i>nearest clearance</span>
        <span><i class="line-swatch" style="background:#54a24b"></i>aggregate clearance</span>
        <span><i class="line-swatch" style="background:#b279a2"></i>continuity</span>
        <span><i class="line-swatch" style="background:#d62728"></i>duplicate risk</span>
        <span><i class="line-swatch" style="background:#e45756"></i>adjacent block</span>
        <span><i class="line-swatch" style="background:#72b7b2"></i>branch hold</span>
        <span><i class="line-swatch" style="background:#111"></i>total shown at right</span>
      </div>
      <div class="candidate-bars" id="candidateBars"></div>
    </div>
	    <div class="note">
	      The zoom view keeps the original 3D coordinates and only changes the camera projection. Candidate bars update for the selected point and decompose why that tangent direction was selected.
	    </div>
  </section>
</main>
<script>
const reportData = __DATA_JSON__;
let selectedIndex = __INITIAL_INDEX__;
const overallSvg = document.getElementById("overallSvg");
const localSvg = document.getElementById("localSvg");
const slider = document.getElementById("sampleSlider");
const zoomSlider = document.getElementById("zoomSlider");
const zoomValue = document.getElementById("zoomValue");
const viewSelect = document.getElementById("viewSelect");
const resetViewBtn = document.getElementById("resetViewBtn");
slider.max = String(Math.max(reportData.samples.length - 1, 0));
slider.value = String(selectedIndex);
let freeYaw = -0.72;
let freePitch = -0.42;

function finite(value) {
  return typeof value === "number" && Number.isFinite(value);
}
function fmt(value, digits = 3) {
  return finite(value) ? value.toFixed(digits) : "n/a";
}
function flagText(value) {
  if (!finite(value)) {
    return "n/a";
  }
  return value >= 0.5 ? "yes" : "no";
}
function add(a, b) {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}
function sub(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}
function mul(a, scale) {
  return [a[0] * scale, a[1] * scale, a[2] * scale];
}
function norm(a) {
  return Math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
}
function normalizeJs(a) {
  const n = norm(a);
  if (n <= 1e-12) return [0, 0, 0];
  return [a[0] / n, a[1] / n, a[2] / n];
}
function dotJs(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
function crossJs(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}
function project(p) {
  return [p[0] - 0.52 * p[1], -p[2] + 0.34 * p[1]];
}
function clamp(value, minValue, maxValue) {
  return Math.max(minValue, Math.min(maxValue, value));
}
function projectFree(p) {
  const cy = Math.cos(freeYaw);
  const sy = Math.sin(freeYaw);
  const cp = Math.cos(freePitch);
  const sp = Math.sin(freePitch);
  const x1 = cy * p[0] - sy * p[1];
  const y1 = sy * p[0] + cy * p[1];
  const z1 = p[2];
  const y2 = cp * y1 - sp * z1;
  const z2 = sp * y1 + cp * z1;
  return [x1, -z2 + 0.12 * y2];
}
function projectView(p, view) {
  if (view === "free") return projectFree(p);
  if (view === "xy") return [p[0], -p[1]];
  if (view === "xz") return [p[0], -p[2]];
  if (view === "yz") return [p[1], -p[2]];
  return project(p);
}
function starPoints(cx, cy, outer, inner, count = 5) {
  const pts = [];
  for (let i = 0; i < count * 2; i += 1) {
    const angle = -Math.PI / 2 + i * Math.PI / count;
    const r = i % 2 === 0 ? outer : inner;
    pts.push(`${cx + Math.cos(angle) * r},${cy + Math.sin(angle) * r}`);
  }
  return pts.join(" ");
}
function makeSvg(tag, attrs = {}) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value)));
  return node;
}
function clearSvg(svg, keepDefs = false) {
  [...svg.children].forEach((child) => {
    if (keepDefs && child.tagName.toLowerCase() === "defs") return;
    child.remove();
  });
}
function validPoint(p) {
  return Array.isArray(p) && p.length === 3 && p.every(finite);
}
function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}
function robotLinks(sample) {
  const links = sample.robot && Array.isArray(sample.robot.links) ? sample.robot.links : [];
  return links.filter((link) => validPoint(link.position));
}
function robotSegments(sample) {
  const segments = sample.robot && Array.isArray(sample.robot.segments) ? sample.robot.segments : [];
  return segments.filter((segment) => validPoint(segment.start) && validPoint(segment.end));
}
function robotSensors(sample) {
  const sensors = sample.robot && Array.isArray(sample.robot.sensors) ? sample.robot.sensors : [];
  return sensors.filter((sensor) => validPoint(sensor.position));
}
function activeSensor(sample) {
  const sensor = sample.robot ? sample.robot.activeSensor : null;
  return sensor && validPoint(sensor.position) ? sensor : null;
}
function duplicatePairSensor(sample) {
  const sensor = sample.robot ? sample.robot.duplicatePairSensor : null;
  return sensor && validPoint(sensor.position) ? sensor : null;
}
function isSensorLinkName(name) {
  const text = String(name || "");
  return text.startsWith("tof") || text.startsWith("sensor_link") || text === "link6_1";
}
function isMainRobotSegment(segment) {
  return !isSensorLinkName(segment.parent) && !isSensorLinkName(segment.child);
}
function isActiveSensorSegment(segment, sample) {
  const sensor = activeSensor(sample);
  if (!sensor) return false;
  return segment.child === sensor.name || segment.parent === sensor.name;
}
function collectOverallPoints(sample) {
  const points = [];
  reportData.samples.forEach((item) => {
    if (validPoint(item.cp)) points.push(item.cp);
    if (validPoint(item.obstacle)) points.push(item.obstacle);
  });
  reportData.goals.forEach((goal) => {
    if (validPoint(goal)) points.push(goal);
  });
  ["tangent", "raw", "filtered", "motion"].forEach((key) => {
    const direction = normalizeJs(sample[key]);
    if (validPoint(sample.cp) && norm(direction) > 0) {
      points.push(add(sample.cp, mul(direction, 0.18)));
    }
  });
  robotLinks(sample).forEach((link) => points.push(link.position));
  robotSegments(sample).forEach((segment) => {
    points.push(segment.start);
    points.push(segment.end);
  });
  const sensor = activeSensor(sample);
  if (sensor) {
    points.push(sensor.position);
  }
  return points;
}
function overallScale(sample) {
  const projected = collectOverallPoints(sample).map(project);
  const xs = projected.map((p) => p[0]);
  const ys = projected.map((p) => p[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const width = 760, height = 560, pad = 46;
  const sx = (width - pad * 2) / Math.max(maxX - minX, 1e-6);
  const sy = (height - pad * 2) / Math.max(maxY - minY, 1e-6);
  const scale = Math.min(sx, sy);
  return (point) => {
    const p = project(point);
    return [
      pad + (p[0] - minX) * scale,
      height - pad - (p[1] - minY) * scale,
    ];
  };
}
function drawLine(svg, a, b, attrs = {}) {
  const line = makeSvg("line", {x1: a[0], y1: a[1], x2: b[0], y2: b[1], ...attrs});
  svg.appendChild(line);
  return line;
}
function drawRobotSegment(svg, a, b, attrs = {}) {
  drawLine(svg, a, b, {
    stroke: "#ffffff",
    "stroke-width": 8.0,
    opacity: 0.92,
    "stroke-linecap": "round",
  });
  drawLine(svg, a, b, attrs);
}
function drawUrdfSegment(svg, a, b, category) {
  if (category === "active") {
    drawRobotSegment(svg, a, b, {
      stroke: "var(--sensor)",
      "stroke-width": 4.2,
      opacity: 0.98,
      "stroke-linecap": "round",
    });
    return;
  }
  if (category === "sensor") {
    drawLine(svg, a, b, {
      stroke: "var(--sensor)",
      "stroke-width": 1.8,
      opacity: 0.55,
      "stroke-linecap": "round",
      "stroke-dasharray": "3 2",
    });
    return;
  }
  drawRobotSegment(svg, a, b, {
    stroke: "var(--robot)",
    "stroke-width": 4.0,
    opacity: 0.95,
    "stroke-linecap": "round",
  });
}
function drawArrow(svg, origin, direction, length, color, markerId, toScreen, width = 3) {
  const unit = normalizeJs(direction);
  if (norm(unit) <= 0) return;
  const end = add(origin, mul(unit, length));
  const a = toScreen(origin);
  const b = toScreen(end);
  drawLine(svg, a, b, {
    stroke: color,
    "stroke-width": width,
    "marker-end": `url(#${markerId})`,
    "stroke-linecap": "round",
  });
}
function drawRobotOverall(svg, sample, toScreen) {
  const links = robotLinks(sample);
  const segments = robotSegments(sample);
  if (segments.length > 0) {
    segments.forEach((segment) => {
      const category = isActiveSensorSegment(segment, sample)
        ? "active"
        : (isMainRobotSegment(segment) ? "main" : "sensor");
      drawUrdfSegment(svg, toScreen(segment.start), toScreen(segment.end), category);
    });
  } else {
    for (let i = 0; i < links.length - 1; i += 1) {
      drawUrdfSegment(svg, toScreen(links[i].position), toScreen(links[i + 1].position), "main");
    }
  }
  links.forEach((link) => {
    const p = toScreen(link.position);
    svg.appendChild(makeSvg("circle", {
      cx: p[0], cy: p[1], r: 4.0, fill: "#fff", stroke: "var(--robot)", "stroke-width": 1.4, opacity: 0.92,
    }));
  });
  ["base_link", "tcp"].forEach((name) => {
    const link = links.find((item) => item.name === name);
    if (!link) return;
    const p = toScreen(link.position);
    svg.appendChild(makeSvg("circle", {
      cx: p[0], cy: p[1], r: name === "tcp" ? 6 : 5, fill: "var(--robot)", stroke: "#fff", "stroke-width": 1.6,
    }));
    drawText(svg, [p[0] + 7, p[1] - 7], name === "base_link" ? "base" : "tcp", {
      fill: "var(--robot)",
      "font-weight": 700,
    });
  });

  const sensor = activeSensor(sample);
  if (!sensor) return;
  const sp = toScreen(sensor.position);
  svg.appendChild(makeSvg("circle", {
    cx: sp[0], cy: sp[1], r: 8, fill: "none", stroke: "var(--sensor)", "stroke-width": 2.2,
  }));
  const pair = duplicatePairSensor(sample);
  if (pair) {
    const pp = toScreen(pair.position);
    drawLine(svg, sp, pp, {
      stroke: "var(--duplicate)",
      "stroke-width": 1.7,
      "stroke-dasharray": "6 4",
      opacity: 0.8,
    });
    svg.appendChild(makeSvg("circle", {
      cx: pp[0], cy: pp[1], r: 7, fill: "none", stroke: "var(--duplicate)", "stroke-width": 2.0,
      "stroke-dasharray": "3 2",
    }));
    drawText(svg, [pp[0] + 8, pp[1] + 15], `dup pair ${pair.name || ""} #${pair.index ?? ""}`, {
      fill: "var(--duplicate)",
      "font-weight": 700,
    });
  }
  drawText(svg, [sp[0] + 9, sp[1] - 9], `${sensor.name || "sensor"} #${sensor.index ?? ""}`, {
    fill: "var(--sensor-ray)",
    "font-weight": 700,
  });
  if (validPoint(sample.cp) && norm(sub(sensor.position, sample.cp)) > 0.003) {
    drawLine(svg, toScreen(sample.cp), sp, {
      stroke: "var(--urdf-cp)",
      "stroke-width": 1.2,
      "stroke-dasharray": "4 3",
      opacity: 0.8,
    });
  }
}
function drawOverall(sample) {
  clearSvg(overallSvg, true);
  const toScreen = overallScale(sample);
  for (let i = 0; i < reportData.samples.length - 1; i += 1) {
    const a = reportData.samples[i].cp;
    const b = reportData.samples[i + 1].cp;
    if (validPoint(a) && validPoint(b)) {
      drawLine(overallSvg, toScreen(a), toScreen(b), {
        stroke: "#333",
        "stroke-width": 1.7,
        opacity: 0.55,
      });
    }
  }
  reportData.samples.forEach((item, index) => {
    if (!validPoint(item.cp)) return;
    const p = toScreen(item.cp);
    const dot = makeSvg("circle", {
      cx: p[0],
      cy: p[1],
      r: index === selectedIndex ? 6 : 4,
      fill: index === selectedIndex ? "#111" : "#ffffff",
      stroke: "#111",
      "stroke-width": 1.2,
      opacity: index === selectedIndex ? 0.95 : 0.48,
      style: "cursor:pointer",
    });
    dot.addEventListener("click", () => {
      selectedIndex = index;
      render();
    });
    overallSvg.appendChild(dot);
  });
  reportData.samples.forEach((item) => {
    if (!validPoint(item.obstacle)) return;
    const p = toScreen(item.obstacle);
    overallSvg.appendChild(makeSvg("circle", {
      cx: p[0], cy: p[1], r: 4.2, fill: "var(--obstacle)", stroke: "#c76c00", "stroke-width": 1, opacity: 0.72,
    }));
  });
  reportData.goals.forEach((goal) => {
    if (!validPoint(goal)) return;
    const p = toScreen(goal);
    overallSvg.appendChild(makeSvg("polygon", {
      points: starPoints(p[0], p[1], 13, 6),
      fill: "var(--goal)",
      stroke: "#111",
      "stroke-width": 1.5,
    }));
  });
  drawRobotOverall(overallSvg, sample, toScreen);
  if (validPoint(sample.obstacle) && validPoint(sample.cp)) {
    drawLine(overallSvg, toScreen(sample.obstacle), toScreen(sample.cp), {
      stroke: "#9aa3af",
      "stroke-width": 1.3,
      opacity: 0.55,
    });
  }
  const cp = toScreen(sample.cp);
  overallSvg.appendChild(makeSvg("circle", {
    cx: cp[0], cy: cp[1], r: 7, fill: "#fff", stroke: "#111", "stroke-width": 2,
  }));
  const obs = toScreen(sample.obstacle);
  overallSvg.appendChild(makeSvg("circle", {
    cx: obs[0], cy: obs[1], r: 7, fill: "var(--obstacle)", stroke: "#c76c00", "stroke-width": 1.5,
  }));
  drawArrow(overallSvg, sample.cp, sample.tangent, 0.16, "var(--tangent)", "arrowTangent", toScreen);
  drawArrow(overallSvg, sample.cp, sample.raw, 0.16, "var(--raw)", "arrowRaw", toScreen);
  drawArrow(overallSvg, sample.cp, sample.filtered, 0.16, "var(--filtered)", "arrowFiltered", toScreen);
  drawArrow(overallSvg, sample.cp, sample.motion, 0.16, "var(--motion)", "arrowMotion", toScreen);
  reportData.samples.forEach((item, index) => {
    if (!validPoint(item.cp)) return;
    const p = toScreen(item.cp);
    const hit = makeSvg("circle", {
      cx: p[0], cy: p[1], r: 10, fill: "transparent", "pointer-events": "all", style: "cursor:pointer",
    });
    hit.addEventListener("click", () => {
      selectedIndex = index;
      render();
    });
    overallSvg.appendChild(hit);
  });
}
function localToScreen(relativePoint, radius, view) {
  const center = [280, 215];
  const scale = 165 / Math.max(radius, 1e-6);
  const p = projectView(relativePoint, view);
  return [center[0] + p[0] * scale, center[1] + p[1] * scale];
}
function drawText(svg, point, text, attrs = {}) {
  const node = makeSvg("text", {x: point[0], y: point[1], "font-size": 12, fill: "#333", ...attrs});
  node.textContent = text;
  svg.appendChild(node);
  return node;
}
function drawLocalLine(svg, aRelative, bRelative, radius, view, attrs = {}) {
  drawLine(svg, localToScreen(aRelative, radius, view), localToScreen(bRelative, radius, view), attrs);
}
function drawLocalArrow3D(svg, originRelative, direction, length, color, label, markerId, radius, view, width = 3) {
  const unit = normalizeJs(direction);
  if (norm(unit) <= 0) return;
  const end = add(originRelative, mul(unit, length));
  const a = localToScreen(originRelative, radius, view);
  const b = localToScreen(end, radius, view);
  drawLine(svg, a, b, {
    stroke: color,
    "stroke-width": width,
    "marker-end": `url(#${markerId})`,
    "stroke-linecap": "round",
  });
  drawText(svg, [b[0] + 5, b[1] - 5], label, {
    fill: color,
    "font-weight": 650,
    stroke: "#ffffff",
    "stroke-width": 3,
    "paint-order": "stroke",
  });
}
function drawLocalCandidateArrow3D(svg, originRelative, direction, length, selected, radius, view, label) {
  const unit = normalizeJs(direction);
  if (norm(unit) <= 0) return;
  const end = add(originRelative, mul(unit, length));
  const a = localToScreen(originRelative, radius, view);
  const b = localToScreen(end, radius, view);
  drawLine(svg, a, b, {
    stroke: selected ? "var(--tangent)" : "var(--candidate)",
    "stroke-width": selected ? 3.0 : 1.5,
    "marker-end": selected ? "url(#arrowTangent)" : "url(#arrowCandidate)",
    "stroke-linecap": "round",
    opacity: selected ? 0.98 : 0.42,
  });
  if (label) {
    drawText(svg, [b[0] + 7, b[1] - 6], label, {
      fill: selected ? "var(--tangent)" : "var(--candidate)",
      "font-size": selected ? 12 : 11,
      "font-weight": selected ? 750 : 650,
      opacity: selected ? 1 : 0.72,
      stroke: "#ffffff",
      "stroke-width": 3,
      "paint-order": "stroke",
    });
  }
}
function drawRobotLocal(svg, sample, radius, view) {
  if (!validPoint(sample.cp)) return;
  const links = robotLinks(sample);
  const segments = robotSegments(sample);
  if (segments.length > 0) {
    segments.forEach((segment) => {
      const a = sub(segment.start, sample.cp);
      const b = sub(segment.end, sample.cp);
      if (norm(a) > radius * 2.0 && norm(b) > radius * 2.0) return;
      const category = isActiveSensorSegment(segment, sample)
        ? "active"
        : (isMainRobotSegment(segment) ? "main" : "sensor");
      drawUrdfSegment(svg, localToScreen(a, radius, view), localToScreen(b, radius, view), category);
    });
  } else {
    for (let i = 0; i < links.length - 1; i += 1) {
      const a = sub(links[i].position, sample.cp);
      const b = sub(links[i + 1].position, sample.cp);
      if (norm(a) > radius * 2.0 && norm(b) > radius * 2.0) continue;
      drawUrdfSegment(svg, localToScreen(a, radius, view), localToScreen(b, radius, view), "main");
    }
  }
  ["base_link", "tcp"].forEach((name) => {
    const link = links.find((item) => item.name === name);
    if (!link) return;
    const rel = sub(link.position, sample.cp);
    if (norm(rel) > radius * 1.55) return;
    const p = drawProjectedPoint(svg, rel, radius, view, {
      r: name === "tcp" ? 6.2 : 5.2,
      fill: "var(--robot)",
      stroke: "#ffffff",
      "stroke-width": 1.8,
      opacity: 0.98,
    });
    drawText(svg, [p[0] + 7, p[1] - 7], name === "base_link" ? "base" : "tcp", {
      fill: "var(--robot)",
      "font-weight": 700,
    });
  });
  const active = activeSensor(sample);
  robotSensors(sample).forEach((sensor) => {
    const rel = sub(sensor.position, sample.cp);
    if (norm(rel) > radius * 1.45) return;
    const isActive = active && sensor.index === active.index;
    drawProjectedPoint(svg, rel, radius, view, {
      r: isActive ? 7.5 : 4.2,
      fill: isActive ? "var(--sensor)" : "#ffffff",
      stroke: "var(--sensor)",
      "stroke-width": isActive ? 2.2 : 1.4,
      opacity: isActive ? 0.98 : 0.62,
    });
  });
  const sensor = active;
  if (!sensor) return;
  const relSensor = sub(sensor.position, sample.cp);
  const pair = duplicatePairSensor(sample);
  if (pair) {
    const relPair = sub(pair.position, sample.cp);
    if (norm(relPair) <= radius * 1.55) {
      const sp = localToScreen(relSensor, radius, view);
      const pp = localToScreen(relPair, radius, view);
      drawLine(svg, sp, pp, {
        stroke: "var(--duplicate)",
        "stroke-width": 1.7,
        "stroke-dasharray": "6 4",
        opacity: 0.82,
      });
      svg.appendChild(makeSvg("circle", {
        cx: pp[0], cy: pp[1], r: 7, fill: "none", stroke: "var(--duplicate)", "stroke-width": 2.0,
        "stroke-dasharray": "3 2",
      }));
      drawText(svg, [pp[0] + 8, pp[1] + 15], `dup pair ${pair.name || ""} #${pair.index ?? ""}`, {
        fill: "var(--duplicate)",
        "font-weight": 700,
        stroke: "#ffffff",
        "stroke-width": 3,
        "paint-order": "stroke",
      });
    }
  }
  if (norm(relSensor) > 0.003) {
    drawLocalLine(svg, [0, 0, 0], relSensor, radius, view, {
      stroke: "var(--urdf-cp)", "stroke-width": 1.2, "stroke-dasharray": "4 3", opacity: 0.8,
    });
  }
  const sp = localToScreen(relSensor, radius, view);
  drawText(svg, [sp[0] + 8, sp[1] + 16], `${sensor.name || "sensor"} #${sensor.index ?? ""}`, {
    fill: "var(--sensor-ray)",
    "font-weight": 700,
  });
}
function drawProjectedPoint(svg, relativePoint, radius, view, attrs = {}) {
  const p = localToScreen(relativePoint, radius, view);
  svg.appendChild(makeSvg("circle", {cx: p[0], cy: p[1], ...attrs}));
  return p;
}
function tangentPlaneBasis(normal) {
  const n = normalizeJs(normal);
  if (norm(n) <= 0) return null;
  const ref = Math.abs(n[2]) < 0.9 ? [0, 0, 1] : [0, 1, 0];
  let u = crossJs(ref, n);
  if (norm(u) <= 1e-9) {
    u = crossJs([1, 0, 0], n);
  }
  u = normalizeJs(u);
  const v = normalizeJs(crossJs(n, u));
  if (norm(u) <= 0 || norm(v) <= 0) return null;
  return [u, v];
}
function drawLocalTangentPlane(svg, sample, radius, view) {
  const basis = tangentPlaneBasis(sample.normal);
  if (!basis) return;
  const [u, v] = basis;
  const planeRadius = Math.min(radius * 0.48, 0.16);
  const points = [];
  for (let index = 0; index < 48; index += 1) {
    const angle = (2 * Math.PI * index) / 48;
    const relative = add(
      mul(u, Math.cos(angle) * planeRadius),
      mul(v, Math.sin(angle) * planeRadius),
    );
    const screen = localToScreen(relative, radius, view);
    points.push(`${screen[0]},${screen[1]}`);
  }
  svg.appendChild(makeSvg("polygon", {
    points: points.join(" "),
    fill: "var(--plane)",
    stroke: "rgba(44,160,44,0.55)",
    "stroke-width": 1.2,
    "stroke-dasharray": "4 3",
  }));
}
function activeCandidates(sample) {
  const payload = sample.candidateData || {};
  const candidates = Array.isArray(payload.candidates) ? payload.candidates : [];
  return candidates.filter((candidate) => validPoint(candidate.direction));
}
function selectedCandidate(sample) {
  return activeCandidates(sample).find((candidate) => candidate.selected) || null;
}
function selectedEscapeDirection(sample) {
  const selected = selectedCandidate(sample);
  if (selected && validPoint(selected.direction)) return selected.direction;
  return sample.tangent;
}
function selectedEscapeLabel(sample) {
  const selected = selectedCandidate(sample);
  return selected && selected.index !== undefined ? `selected cand #${selected.index}` : "selected";
}
function drawLocal(sample) {
  clearSvg(localSvg, false);
  const defs = overallSvg.querySelector("defs").cloneNode(true);
  localSvg.appendChild(defs);
  const radius = Number(zoomSlider.value);
  const view = viewSelect.value;
  const arrowLength = Math.min(radius * 0.55, 0.18);
  const gridRadius = radius;

  localSvg.appendChild(makeSvg("rect", {x: 0, y: 0, width: 560, height: 430, fill: "#fbfcfe"}));
  [-1, -0.5, 0, 0.5, 1].forEach((ratio) => {
    const value = ratio * gridRadius;
    drawLocalLine(localSvg, [-gridRadius, value, 0], [gridRadius, value, 0], radius, view, {
      stroke: "#d8dde5", "stroke-width": value === 0 ? 1.4 : 0.8,
    });
    drawLocalLine(localSvg, [value, -gridRadius, 0], [value, gridRadius, 0], radius, view, {
      stroke: "#d8dde5", "stroke-width": value === 0 ? 1.4 : 0.8,
    });
  });
  drawLocalTangentPlane(localSvg, sample, radius, view);

  for (let i = 0; i < reportData.samples.length - 1; i += 1) {
    const a = sub(reportData.samples[i].cp, sample.cp);
    const b = sub(reportData.samples[i + 1].cp, sample.cp);
    if (norm(a) <= radius * 1.35 || norm(b) <= radius * 1.35) {
      drawLocalLine(localSvg, a, b, radius, view, {
        stroke: "#333", "stroke-width": 1.4, opacity: 0.55,
      });
    }
  }

  reportData.samples.forEach((item, index) => {
    const relObstacle = sub(item.obstacle, sample.cp);
    if (norm(relObstacle) <= radius * 1.25) {
      drawProjectedPoint(localSvg, relObstacle, radius, view, {
        r: index === selectedIndex ? 7 : 4,
        fill: "var(--obstacle)",
        stroke: "#c76c00",
        "stroke-width": 1,
        opacity: index === selectedIndex ? 0.95 : 0.55,
      });
    }
  });

  if (validPoint(sample.goal)) {
    const relGoal = sub(sample.goal, sample.cp);
    if (norm(relGoal) <= radius * 1.45) {
      const gp = localToScreen(relGoal, radius, view);
      localSvg.appendChild(makeSvg("polygon", {
        points: starPoints(gp[0], gp[1], 13, 6), fill: "var(--goal)", stroke: "#111", "stroke-width": 1.4,
      }));
    }
  }

  const obstacle = sub(sample.obstacle, sample.cp);
  drawLocalLine(localSvg, [0, 0, 0], obstacle, radius, view, {
    stroke: "#9aa3af", "stroke-width": 1.2, opacity: 0.75,
  });

  drawRobotLocal(localSvg, sample, radius, view);
  const candidates = activeCandidates(sample);
  candidates.forEach((candidate) => {
    if (candidate.selected) return;
    drawLocalCandidateArrow3D(
      localSvg,
      [0, 0, 0],
      candidate.direction,
      arrowLength * 0.95,
      false,
      radius,
      view,
      `cand #${candidate.index ?? "?"}`,
    );
  });
  drawLocalArrow3D(localSvg, [0, 0, 0], selectedEscapeDirection(sample), arrowLength, "var(--tangent)", selectedEscapeLabel(sample), "arrowTangent", radius, view);
  drawLocalArrow3D(localSvg, [0, 0, 0], sample.raw, arrowLength, "var(--raw)", "raw", "arrowRaw", radius, view);
  drawLocalArrow3D(localSvg, [0, 0, 0], sample.filtered, arrowLength, "var(--filtered)", "filtered", "arrowFiltered", radius, view);
  drawLocalArrow3D(localSvg, [0, 0, 0], sample.motion, arrowLength, "var(--motion)", "motion", "arrowMotion", radius, view);

  const obs = drawProjectedPoint(localSvg, obstacle, radius, view, {
    r: 9, fill: "var(--obstacle)", stroke: "#c76c00", "stroke-width": 1.4,
  });
  const cp = drawProjectedPoint(localSvg, [0, 0, 0], radius, view, {
    r: 8, fill: "#fff", stroke: "#111", "stroke-width": 2,
  });
  drawText(localSvg, [cp[0] + 8, cp[1] + 18], sample.activeSensorName ? `active CP (${sample.activeSensorName})` : "active CP", {fill: "#111"});
  drawText(localSvg, [obs[0] + 8, obs[1] - 8], "obstacle", {fill: "var(--obstacle)", "font-weight": 650});
}
function signedBar(label, value, color, maxAbs) {
  const safe = finite(value) ? value : 0;
  const ratio = Math.min(Math.abs(safe) / Math.max(maxAbs, 1e-9), 1);
  const left = safe >= 0 ? 50 : 50 - ratio * 50;
  const width = ratio * 50;
  return `<div class="bar-row">
    <div>${label}</div>
    <div class="bar-track"><div class="bar-zero"></div><div class="bar-fill" style="left:${left}%;width:${width}%;background:${color}"></div></div>
    <div>${fmt(value, 2)}</div>
  </div>`;
}
function updateBars(sample) {
  const values = [
    sample.rawDotTangent,
    sample.filteredDotTangent,
    sample.rawDotNormal,
    sample.filteredDotNormal,
  ].filter(finite);
  const maxAbs = Math.max(...values.map(Math.abs), 1);
  document.getElementById("componentBars").innerHTML = [
    signedBar("raw tangent", sample.rawDotTangent, "var(--raw)", maxAbs),
    signedBar("filtered tangent", sample.filteredDotTangent, "var(--filtered)", maxAbs),
    signedBar("raw normal", sample.rawDotNormal, "var(--raw)", maxAbs),
    signedBar("filtered normal", sample.filteredDotNormal, "var(--filtered)", maxAbs),
  ].join("");
}
function candidateComponentValues(candidate) {
  return [
    ["goal", Number(candidate.weighted_goal || 0), "#4c78a8"],
    ["continuity", Number(candidate.weighted_continuity || 0), "#b279a2"],
    ["duplicate", Number(candidate.weighted_duplicate_risk || 0), "#d62728"],
    ["adjacent", Number(candidate.weighted_adjacent_block || 0), "#e45756"],
    ["branch", Number(candidate.weighted_branch_hold || 0), "#72b7b2"],
    ["up", Number(candidate.weighted_up || 0), "#9aa3af"],
  ];
}
function candidateScale(candidates) {
  const magnitudes = candidates.map((candidate) => {
    const values = candidateComponentValues(candidate).map(([, value]) => value);
    const positive = values.filter((value) => value > 0).reduce((a, b) => a + b, 0);
    const negative = Math.abs(values.filter((value) => value < 0).reduce((a, b) => a + b, 0));
    const total = Math.abs(Number(candidate.total_score || 0));
    return Math.max(positive, negative, total);
  });
  return Math.max(...magnitudes, 1e-6);
}
function candidateSegmentsHtml(candidate, maxMagnitude) {
  let positiveOffset = 0;
  let negativeOffset = 0;
  const scale = 48 / Math.max(maxMagnitude, 1e-9);
  const segments = [];
  candidateComponentValues(candidate).forEach(([, value, color]) => {
    if (!Number.isFinite(value) || Math.abs(value) < 1e-9) return;
    if (value >= 0) {
      const left = 50 + positiveOffset * scale;
      const width = Math.abs(value) * scale;
      segments.push(`<span class="candidate-segment" style="left:${left}%;width:${width}%;background:${color}"></span>`);
      positiveOffset += value;
    } else {
      const nextOffset = negativeOffset + value;
      const left = 50 + nextOffset * scale;
      const width = Math.abs(value) * scale;
      segments.push(`<span class="candidate-segment" style="left:${left}%;width:${width}%;background:${color}"></span>`);
      negativeOffset = nextOffset;
    }
  });
  if (candidate.selected) {
    segments.push(`<span class="candidate-selected-band"></span>`);
  }
  return segments.join("");
}
function updateCandidateScores(sample) {
  const payload = sample.candidateData || {};
  const candidates = activeCandidates(sample);
  const summary = document.getElementById("candidateSummary");
  const bars = document.getElementById("candidateBars");
  if (!payload.active || candidates.length === 0) {
    summary.textContent = "no active candidate data";
    bars.innerHTML = "";
    return;
  }
  const selected = candidates.find((candidate) => candidate.selected);
  const selectedIndex = selected ? selected.index : payload.selected_candidate_index;
  const selectedTotal = selected ? selected.total_score : null;
  const weights = payload.weights || {};
  const metrics = payload.metrics || {};
  summary.textContent =
    `selected ${selectedIndex ?? "n/a"}, total ${fmt(selectedTotal, 3)}, ` +
    `gap ${fmt(metrics.selected_score_gap, 3)}, prevDot ${fmt(metrics.previous_direction_dot, 3)}, ` +
    `N=${candidates.length}, weights g:${fmt(weights.goal, 1)} dup:${fmt(weights.duplicate_risk, 1)} adj:${fmt(weights.adjacent_block, 1)} cont:${fmt(weights.continuity, 1)} hold:${fmt(weights.branch_hold, 1)}`;
  const maxMagnitude = candidateScale(candidates);
  bars.innerHTML = candidates.map((candidate) => {
    const cls = candidate.selected ? "candidate-row selected" : "candidate-row";
    const index = candidate.index ?? "";
    const direction = vectorText(candidate.direction, 2);
    return `<div class="${cls}">
      <div>#${index}</div>
      <div class="candidate-direction">${escapeHtml(direction)}</div>
      <div class="candidate-track">${candidateSegmentsHtml(candidate, maxMagnitude)}</div>
      <div>${fmt(candidate.total_score, 2)}</div>
    </div>`;
  }).join("");
}
function vectorText(values, digits = 3) {
  return validPoint(values) ? `[${values.map((value) => fmt(value, digits)).join(", ")}]` : "n/a";
}
function updateRobotContext(sample) {
  const summary = document.getElementById("sensorSummary");
  const details = document.getElementById("robotDetails");
  const jointTable = document.getElementById("jointTable");
  const robot = sample.robot || null;
  const sensor = activeSensor(sample);
  const pair = duplicatePairSensor(sample);
  if (!robot || !sensor) {
    const name = sample.activeSensorName || "n/a";
    const index = sample.activeSensorIndex ?? sample.cpIndex ?? "n/a";
    const parent = sample.activeSensorParent || "n/a";
    summary.textContent = name !== "n/a" ? `${name} #${index}` : "no joint/URDF context";
    details.innerHTML = name !== "n/a" ? [
      ["sensor frame", `${name} (#${index})`],
      ["parent link", parent],
      ["URDF overlay", "not available because joint values or URDF context are missing"],
    ].map(([label, value]) => (
      `<div class="robot-cell"><div class="label">${escapeHtml(label)}</div><div class="value">${escapeHtml(String(value))}</div></div>`
    )).join("") : "";
    jointTable.innerHTML = "";
    return;
  }
  summary.textContent = `${sensor.name || "sensor"} #${sensor.index ?? "n/a"} on ${sensor.parent || "n/a"}`;
  const cpDelta = finite(robot.cpUrdfDistance) ? `${fmt(robot.cpUrdfDistance, 4)} m` : "n/a";
  details.innerHTML = [
    ["sensor frame", `${sensor.name || "n/a"} (#${sensor.index ?? "n/a"})`],
    ["parent link", sensor.parent || "n/a"],
    ["duplicate pair", pair ? `${pair.name || "n/a"} (#${pair.index ?? "n/a"})` : "n/a"],
    ["URDF sensor +X", vectorText(sensor.direction, 3)],
    ["debug CP to URDF sensor", cpDelta],
  ].map(([label, value]) => (
    `<div class="robot-cell"><div class="label">${escapeHtml(label)}</div><div class="value">${escapeHtml(value)}</div></div>`
  )).join("");
  const joints = Array.isArray(robot.joints) ? robot.joints : [];
  jointTable.innerHTML = [
    '<div class="head">joint</div><div class="head">rad</div><div class="head">deg</div>',
    ...joints.map((joint) => (
      `<div>${escapeHtml(joint.name)}</div><div>${fmt(joint.rad, 4)}</div><div>${fmt(joint.deg, 2)}</div>`
    )),
  ].join("");
}
function render() {
  selectedIndex = Math.max(0, Math.min(selectedIndex, reportData.samples.length - 1));
  slider.value = String(selectedIndex);
  zoomValue.textContent = `${Number(zoomSlider.value).toFixed(2)} m`;
  const sample = reportData.samples[selectedIndex];
  drawOverall(sample);
  drawLocal(sample);
  updateBars(sample);
  updateCandidateScores(sample);
  updateRobotContext(sample);
  document.getElementById("sampleLabel").textContent = `active sample ${selectedIndex + 1} / ${reportData.samples.length}`;
  document.getElementById("sampleTime").textContent = `t = ${fmt(sample.time, 2)} s`;
  document.getElementById("clearanceValue").textContent = `${fmt(sample.clearance, 4)} m`;
  document.getElementById("activationValue").textContent = fmt(sample.activation, 3);
  document.getElementById("angleValue").textContent = `${fmt(sample.rawFilteredAngleDeg, 1)} deg`;
  document.getElementById("orthogonalValue").textContent = fmt(sample.tangentDotNormal, 4);
  document.getElementById("deltaQddValue").textContent = fmt(sample.deltaQddNorm, 3);
  document.getElementById("scoreGapValue").textContent = fmt(sample.scoreGap, 3);
  document.getElementById("previousDirectionValue").textContent = fmt(sample.previousDirectionDot, 3);
  document.getElementById("activeCpChangedValue").textContent = flagText(sample.activeCpChanged);
}
slider.addEventListener("input", () => { selectedIndex = Number(slider.value); render(); });
zoomSlider.addEventListener("input", () => { render(); });
viewSelect.addEventListener("change", () => { render(); });
resetViewBtn.addEventListener("click", () => {
  freeYaw = -0.72;
  freePitch = -0.42;
  viewSelect.value = "free";
  render();
});
let dragState = null;
localSvg.addEventListener("pointerdown", (event) => {
  dragState = {x: event.clientX, y: event.clientY};
  localSvg.classList.add("dragging");
  localSvg.setPointerCapture(event.pointerId);
  viewSelect.value = "free";
});
localSvg.addEventListener("pointermove", (event) => {
  if (!dragState) return;
  const dx = event.clientX - dragState.x;
  const dy = event.clientY - dragState.y;
  dragState = {x: event.clientX, y: event.clientY};
  freeYaw += dx * 0.01;
  freePitch = clamp(freePitch + dy * 0.01, -1.45, 1.45);
  render();
});
["pointerup", "pointercancel", "pointerleave"].forEach((eventName) => {
  localSvg.addEventListener(eventName, (event) => {
    dragState = null;
    localSvg.classList.remove("dragging");
    if (event.pointerId !== undefined && localSvg.hasPointerCapture(event.pointerId)) {
      localSvg.releasePointerCapture(event.pointerId);
    }
  });
});
document.getElementById("prevBtn").addEventListener("click", () => { selectedIndex -= 1; render(); });
document.getElementById("nextBtn").addEventListener("click", () => { selectedIndex += 1; render(); });
render();
</script>
</body>
</html>
"""


def write_report(path: Path, samples: Sequence[Dict[str, object]], goals: Sequence[Sequence[float]], initial_index: int, output: Path) -> None:
    payload = {
        "samples": samples,
        "goals": [vector_or_list(goal) for goal in goals],
    }
    rendered = (
        html_template()
        .replace("__TITLE__", html.escape(f"Tangent Escape Filter Report - {path.name}"))
        .replace("__CSV_NAME__", html.escape(path.name))
        .replace("__DATA_JSON__", json.dumps(payload, separators=(",", ":")))
        .replace("__INITIAL_INDEX__", str(initial_index))
    )
    output.write_text(rendered, encoding="utf-8")


def read_csv_header(path: Path) -> List[str]:
    raw_lines: List[str] = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            raw_lines.append(line)
            break
    reader = csv.DictReader(raw_lines)
    return list(reader.fieldnames or [])


def find_latest_compatible_csv(directory: Path) -> Optional[Path]:
    if not directory.exists():
        return None
    for candidate in sorted(directory.glob("*.csv"), key=lambda path: path.stat().st_mtime, reverse=True):
        try:
            require_columns(read_csv_header(candidate))
        except Exception:
            continue
        return candidate
    return None


def vector_or_list(values: Sequence[float]) -> List[Optional[float]]:
    return [finite_or_none(value) for value in values]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create an interactive HTML report for tangent escape filter samples."
    )
    parser.add_argument("csv_path", nargs="?", help="Recorder CSV path. Defaults to latest CSV.")
    parser.add_argument(
        "--latest-dir",
        default=str(DEFAULT_DATA_DIR),
        help="Directory used when no CSV path is provided.",
    )
    parser.add_argument("--save", help="Output HTML path.")
    parser.add_argument(
        "--select",
        choices=("closest-clearance", "middle", "max-angle"),
        default="closest-clearance",
        help="Default selected sample when --sample-index and --time are not set.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        help="Initial active sample index, zero-based among active tangent escape rows.",
    )
    parser.add_argument(
        "--time",
        type=float,
        help="Initial active sample nearest this relative active-filter time in seconds.",
    )
    parser.add_argument(
        "--max-age",
        type=float,
        default=1.0,
        help="Ignore cached debug samples older than this many seconds.",
    )
    parser.add_argument(
        "--urdf",
        default=str(default_urdf_path()),
        help="URDF path used to recompute robot links and sensor frames from recorded joint values.",
    )
    parser.add_argument(
        "--no-robot",
        action="store_true",
        help="Disable URDF robot/sensor overlay even when joint columns are present.",
    )
    args = parser.parse_args()

    path = resolve_input_path(args.csv_path, args.latest_dir)
    try:
        fieldnames, rows = read_csv_rows(path)
        require_columns(fieldnames)
    except ValueError as error:
        if args.csv_path:
            raise
        latest_dir = Path(args.latest_dir).expanduser()
        compatible_path = find_latest_compatible_csv(latest_dir)
        if compatible_path is not None and compatible_path != path:
            path = compatible_path
            fieldnames, rows = read_csv_rows(path)
            require_columns(fieldnames)
        else:
            raise ValueError(
                f"No compatible tangent escape filter CSV found in {latest_dir}. "
                "The latest CSV exists, but it was recorded before filter debug columns were enabled. "
                "Run rb10_rmpflow_test.launch.py with use_rmpflow_trace_logger:=true, "
                "enable_tangent_escape_filter:=true, publish_tangent_escape_filter_data:=true, "
                "and publish_tangent_escape_filter_candidate_data:=true, then run this report again."
            ) from error
    selected_rows = active_rows(rows, args.max_age)
    if not selected_rows:
        stale_active_rows = active_rows(rows, float("inf"))
        if stale_active_rows:
            ages = [
                parse_float(row, "tangent_escape_age_s")
                for row in stale_active_rows
                if math.isfinite(parse_float(row, "tangent_escape_age_s"))
            ]
            age_hint = (
                f" Active debug ages are {min(ages):.3f}..{max(ages):.3f}s."
                if ages else ""
            )
            retry_age = max(1.0, max(ages) if ages else args.max_age)
            raise ValueError(
                f"No active tangent escape rows newer than --max-age {args.max_age:.3f}s found in: {path}."
                f"{age_hint} Re-run with --max-age {retry_age:.1f}."
            )
        raise ValueError(f"No active tangent escape rows found in: {path}")

    urdf_model = None
    if not args.no_robot:
        urdf_path = Path(args.urdf).expanduser()
        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")
        urdf_model = UrdfKinematics(urdf_path)

    samples = [
        make_sample(selected_rows, row, index, urdf_model)
        for index, row in enumerate(selected_rows)
    ]
    initial_index = choose_initial_index(samples, args)
    goals = goal_points(rows)
    output = (
        Path(args.save).expanduser()
        if args.save
        else path.with_name(f"{path.stem}_tangent_escape_report.html")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_report(path, samples, goals, initial_index, output)

    print(f"Input CSV: {path}")
    print(f"Active rows: {len(samples)}")
    print(f"Initial active sample index: {initial_index}")
    print(f"Saved report: {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
