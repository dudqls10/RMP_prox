#!/usr/bin/env python3
import argparse
import csv
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_LOG_DIR = Path("~/ros2_ws/log/rmpflow_trace").expanduser()
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

REQUIRED_COLUMNS = [
    "time_ros_s",
    "tangent_escape_rmp_active",
    "tangent_escape_rmp_control_point_index",
    "tangent_escape_rmp_schema_id",
    "tangent_escape_rmp_activation",
    "tangent_escape_rmp_raw_activation",
    "tangent_escape_rmp_clearance_m",
    "tangent_escape_rmp_beta",
    "tangent_escape_rmp_effective_metric_scalar",
    "tangent_escape_rmp_canonical_coordinate_m",
    "tangent_escape_rmp_velocity_reference_m_s",
    "tangent_escape_rmp_scalar_velocity_m_s",
    "tangent_escape_rmp_velocity_error_m_s",
    "tangent_escape_rmp_desired_tangent_accel_m_s2",
    "tangent_escape_rmp_alpha_stuck",
    "tangent_escape_rmp_z",
    "tangent_escape_rmp_lambda",
    "tangent_escape_rmp_drive_ramp",
    "tangent_escape_rmp_release_brake",
    "tangent_escape_rmp_state_mode_id",
    "tangent_escape_rmp_handoff_phase_id",
    "tangent_escape_rmp_handoff_reason_id",
    "tangent_escape_rmp_candidate_count",
    "tangent_escape_rmp_selected_candidate_feasible",
    "tangent_escape_rmp_selected_candidate_score",
    "tangent_escape_rmp_selected_sector_risk",
    "tangent_escape_rmp_selected_blocked_penalty",
    "tangent_escape_rmp_max_blocked_memory",
    "tangent_escape_rmp_command_distance_m",
    "tangent_escape_rmp_actual_distance_m",
    "tangent_escape_rmp_move_ratio",
    "tangent_escape_rmp_normal_x",
    "tangent_escape_rmp_normal_y",
    "tangent_escape_rmp_normal_z",
    "tangent_escape_rmp_tangent_x",
    "tangent_escape_rmp_tangent_y",
    "tangent_escape_rmp_tangent_z",
    "rmp_joint_accel_1_rad_s2",
    "rmp_joint_accel_2_rad_s2",
    "rmp_joint_accel_3_rad_s2",
    "rmp_joint_accel_4_rad_s2",
    "rmp_joint_accel_5_rad_s2",
    "rmp_joint_accel_6_rad_s2",
]


def parse_float(row: Dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return float("nan")


def finite_values(values: Iterable[float]) -> List[float]:
    return [value for value in values if math.isfinite(value)]


def norm(values: Sequence[float]) -> float:
    if any(not math.isfinite(value) for value in values):
        return float("nan")
    return math.sqrt(sum(value * value for value in values))


def vector_angle_degrees(
    lhs: Sequence[float],
    rhs: Sequence[float],
    minimum_norm: float,
) -> float:
    lhs_norm = norm(lhs)
    rhs_norm = norm(rhs)
    if (
        not math.isfinite(lhs_norm) or
        not math.isfinite(rhs_norm) or
        lhs_norm <= minimum_norm or
        rhs_norm <= minimum_norm
    ):
        return float("nan")
    cosine = sum(left * right for left, right in zip(lhs, rhs)) / (
        lhs_norm * rhs_norm
    )
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def percentile(values: Sequence[float], fraction: float) -> float:
    finite = sorted(finite_values(values))
    if not finite:
        return float("nan")
    position = (len(finite) - 1) * min(max(fraction, 0.0), 1.0)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return finite[lower]
    weight = position - lower
    return finite[lower] * (1.0 - weight) + finite[upper] * weight


def fmt(value: float, digits: int = 4) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}g}"


def fmt_range(values: Sequence[float], digits: int = 4) -> str:
    finite = finite_values(values)
    if not finite:
        return "n/a"
    return (
        f"min={fmt(min(finite), digits)} max={fmt(max(finite), digits)} "
        f"first={fmt(finite[0], digits)} last={fmt(finite[-1], digits)}"
    )


def latest_compatible_csv(log_dir: Path) -> Path:
    candidates = sorted(log_dir.glob("rmpflow_trace_*.csv"), key=lambda path: path.stat().st_mtime)
    for path in reversed(candidates):
        try:
            with path.open(newline="") as stream:
                reader = csv.DictReader(stream)
                fieldnames = set(reader.fieldnames or [])
        except OSError:
            continue
        if all(column in fieldnames for column in REQUIRED_COLUMNS):
            return path
    raise FileNotFoundError(
        f"No compatible rmpflow_trace CSV with tangent escape RMP columns found in {log_dir}"
    )


def load_rows(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
    if missing:
        raise ValueError("Missing closed-loop check columns: " + ", ".join(missing))
    return rows, fieldnames


def active_intervals(rows: Sequence[Dict[str, str]]) -> List[Tuple[int, int]]:
    intervals: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for index, row in enumerate(rows):
        active = parse_float(row, "tangent_escape_rmp_active") >= 0.5
        if active and start is None:
            start = index
        elif not active and start is not None:
            intervals.append((start, index - 1))
            start = None
    if start is not None:
        intervals.append((start, len(rows) - 1))
    return intervals


def interval_duration(rows: Sequence[Dict[str, str]], start: int, end: int) -> float:
    start_t = parse_float(rows[start], "time_ros_s")
    end_t = parse_float(rows[end], "time_ros_s")
    if math.isfinite(start_t) and math.isfinite(end_t):
        return max(end_t - start_t, 0.0)
    return float("nan")


def row_time(row: Dict[str, str]) -> float:
    return parse_float(row, "time_ros_s")


def qdd_vector(row: Dict[str, str]) -> List[float]:
    return [parse_float(row, f"rmp_joint_accel_{index}_rad_s2") for index in range(1, 7)]


def qdd_norm(row: Dict[str, str]) -> float:
    value = parse_float(row, "rmp_joint_accel_norm")
    if math.isfinite(value):
        return value
    return norm(qdd_vector(row))


def dual_qdd_vector(row: Dict[str, str], variant: str) -> List[float]:
    return [
        parse_float(row, f"tangent_escape_qdd_{variant}_{index}_rad_s2")
        for index in range(1, 7)
    ]


def dual_cp_accel_vector(row: Dict[str, str], variant: str) -> List[float]:
    return [
        parse_float(row, f"tangent_escape_cp_accel_{variant}_{axis}_m_s2")
        for axis in ("x", "y", "z")
    ]


def goal_error(row: Dict[str, str]) -> float:
    value = parse_float(row, "goal_error_m")
    if math.isfinite(value):
        return value
    goal = [parse_float(row, f"controller_goal_{axis}") for axis in ("x", "y", "z")]
    ee = [parse_float(row, f"rmp_ee_{axis}") for axis in ("x", "y", "z")]
    if all(math.isfinite(value) for value in goal + ee):
        return norm([ee[index] - goal[index] for index in range(3)])
    return float("nan")


def nearest_finite_goal_error(rows: Sequence[Dict[str, str]], index: int, direction: int) -> float:
    cursor = index
    while 0 <= cursor < len(rows):
        value = goal_error(rows[cursor])
        if math.isfinite(value):
            return value
        cursor += direction
    return float("nan")


def vector_dot(row: Dict[str, str], lhs_prefix: str, rhs_prefix: str) -> float:
    lhs = [parse_float(row, f"{lhs_prefix}_{axis}") for axis in ("x", "y", "z")]
    rhs = [parse_float(row, f"{rhs_prefix}_{axis}") for axis in ("x", "y", "z")]
    if not all(math.isfinite(value) for value in lhs + rhs):
        return float("nan")
    return sum(lhs[index] * rhs[index] for index in range(3))


def max_jerk(rows: Sequence[Dict[str, str]]) -> Tuple[float, Optional[int]]:
    best = float("nan")
    best_index: Optional[int] = None
    previous_qdd: Optional[List[float]] = None
    previous_time = float("nan")
    previous_index: Optional[int] = None

    for index, row in enumerate(rows):
        current_qdd = qdd_vector(row)
        current_time = row_time(row)
        if not all(math.isfinite(value) for value in current_qdd) or not math.isfinite(current_time):
            continue
        if previous_qdd is not None and math.isfinite(previous_time):
            dt = current_time - previous_time
            if dt > 1e-6:
                value = norm([
                    (current_qdd[joint] - previous_qdd[joint]) / dt
                    for joint in range(6)
                ])
                if math.isfinite(value) and (not math.isfinite(best) or value > best):
                    best = value
                    best_index = previous_index if previous_index is not None else index
        previous_qdd = current_qdd
        previous_time = current_time
        previous_index = index
    return best, best_index


def fake_range_summary(rows: Sequence[Dict[str, str]], fieldnames: Sequence[str]) -> List[str]:
    summaries: List[str] = []
    for index in range(1, 21):
        value_col = f"fake_raw_distance{index}_effective_m"
        trigger_col = f"fake_raw_distance{index}_triggered"
        if value_col not in fieldnames:
            continue
        values = finite_values(parse_float(row, value_col) for row in rows)
        triggered = sum(1 for row in rows if parse_float(row, trigger_col) >= 0.5)
        if not values or (min(values) >= 0.3 and triggered == 0):
            continue
        summaries.append(
            f"fake_raw_distance{index}: min={fmt(min(values))} max={fmt(max(values))} "
            f"triggered_rows={triggered}"
        )
    return summaries


def sensor_name(index: int) -> str:
    if 0 <= index < len(SENSOR_NAMES):
        return SENSOR_NAMES[index]
    return "unknown"


def make_report(
    rows: List[Dict[str, str]],
    fieldnames: List[str],
    path: Path,
    args: argparse.Namespace,
) -> int:
    if not rows:
        print(f"Input CSV: {path}")
        print("Error: CSV has no rows")
        return 1

    intervals = active_intervals(rows)
    active_indices = [
        index
        for start, end in intervals
        for index in range(start, end + 1)
    ]
    active_rows = [rows[index] for index in active_indices]
    start_time = row_time(rows[0])
    end_time = row_time(rows[-1])
    duration = (
        end_time - start_time
        if math.isfinite(start_time) and math.isfinite(end_time)
        else float("nan")
    )

    def rel_time(index: int) -> float:
        value = row_time(rows[index])
        if math.isfinite(value) and math.isfinite(start_time):
            return value - start_time
        return float("nan")

    def count_summary(values: Sequence[float]) -> str:
        counts = Counter(int(round(value)) for value in values)
        return ", ".join(
            f"{identifier}:{count}"
            for identifier, count in sorted(counts.items())
        ) or "none"

    print(f"Input CSV: {path}")
    print(f"Rows: {len(rows)}")
    print(f"Duration: {fmt(duration)} s")
    print("")

    if not active_rows:
        print("Result: NO_ESCAPE_ACTIVE")
        print("No tangent_escape_rmp_active rows were found.")
        return 1

    cp_counts: Counter[int] = Counter()
    mode_counts: Counter[int] = Counter()
    invalid_mode_rows: List[int] = []
    for source_index, row in zip(active_indices, active_rows):
        control_point = parse_float(row, "tangent_escape_rmp_control_point_index")
        mode = parse_float(row, "tangent_escape_rmp_schema_id")
        if math.isfinite(control_point):
            cp_counts[int(round(control_point))] += 1
        if math.isfinite(mode):
            mode_id = int(round(mode))
            mode_counts[mode_id] += 1
            if mode_id != 6:
                invalid_mode_rows.append(source_index)
        else:
            invalid_mode_rows.append(source_index)

    total_active_duration = sum(
        interval_duration(rows, start, end)
        for start, end in intervals
    )
    print("Canonical Tangent Escape RMP:")
    print(f"  active rows: {len(active_rows)}")
    print(f"  active intervals: {len(intervals)}")
    print(f"  active duration: {fmt(total_active_duration)} s")
    for interval_index, (start, end) in enumerate(intervals, start=1):
        print(
            f"  interval {interval_index}: rows {start}-{end}, "
            f"t_rel={fmt(rel_time(start))}-{fmt(rel_time(end))}, "
            f"duration={fmt(interval_duration(rows, start, end))} s"
        )
    print(
        "  control points: " +
        ", ".join(
            f"{index}({sensor_name(index)}):{count}"
            for index, count in sorted(cp_counts.items())
        )
    )
    print(
        "  wire schema ids: " +
        ", ".join(
            f"{mode}:{count}"
            for mode, count in sorted(mode_counts.items())
        )
    )
    print("")

    if invalid_mode_rows:
        preview = ", ".join(str(index) for index in invalid_mode_rows[:8])
        suffix = "..." if len(invalid_mode_rows) > 8 else ""
        print("Result:")
        print("  STRUCTURE_OK: no")
        print(
            "  ERROR: active records must use canonical wire schema ID 6; "
            f"invalid rows: {preview}{suffix}"
        )
        return 1

    print("Canonical Velocity Tracking:")
    for label, field in [
        ("activation", "tangent_escape_rmp_activation"),
        ("raw_activation", "tangent_escape_rmp_raw_activation"),
        ("clearance_m", "tangent_escape_rmp_clearance_m"),
        ("beta", "tangent_escape_rmp_beta"),
        ("metric_scalar", "tangent_escape_rmp_effective_metric_scalar"),
        ("canonical_coordinate_m", "tangent_escape_rmp_canonical_coordinate_m"),
        ("velocity_reference_m_s", "tangent_escape_rmp_velocity_reference_m_s"),
        ("scalar_velocity_m_s", "tangent_escape_rmp_scalar_velocity_m_s"),
        ("velocity_error_m_s", "tangent_escape_rmp_velocity_error_m_s"),
        (
            "desired_tangent_accel_m_s2",
            "tangent_escape_rmp_desired_tangent_accel_m_s2",
        ),
        ("alpha_stuck", "tangent_escape_rmp_alpha_stuck"),
        ("z", "tangent_escape_rmp_z"),
        ("lambda", "tangent_escape_rmp_lambda"),
        ("drive_ramp", "tangent_escape_rmp_drive_ramp"),
        ("release_brake", "tangent_escape_rmp_release_brake"),
        ("command_distance_m", "tangent_escape_rmp_command_distance_m"),
        ("actual_distance_m", "tangent_escape_rmp_actual_distance_m"),
        ("move_ratio", "tangent_escape_rmp_move_ratio"),
    ]:
        print(
            f"  {label}: " +
            fmt_range([parse_float(row, field) for row in active_rows])
        )

    normal_dot_tangent = finite_values(
        vector_dot(
            row,
            "tangent_escape_rmp_normal",
            "tangent_escape_rmp_tangent",
        )
        for row in active_rows
    )
    drive_normal_dot_tangent = finite_values(
        vector_dot(
            row,
            "tangent_escape_rmp_normal",
            "tangent_escape_rmp_tangent",
        )
        for row in active_rows
        if parse_float(row, "tangent_escape_rmp_handoff_phase_id") in (1.0, 2.0)
    )
    if normal_dot_tangent:
        print(
            "  max_abs_normal_dot_tangent_all_phases: " +
            fmt(max(abs(value) for value in normal_dot_tangent))
        )
    if drive_normal_dot_tangent:
        print(
            "  max_abs_normal_dot_tangent_engage_drive: " +
            fmt(max(abs(value) for value in drive_normal_dot_tangent))
        )
    logged_normal_dot = finite_values(
        parse_float(row, "tangent_escape_rmp_normal_dot_tangent")
        for row in active_rows
    )
    if logged_normal_dot:
        print("  logged_normal_dot_tangent: " + fmt_range(logged_normal_dot))
    print("")

    candidate_count = finite_values(
        parse_float(row, "tangent_escape_rmp_candidate_count")
        for row in active_rows
    )
    selected_feasible = finite_values(
        parse_float(row, "tangent_escape_rmp_selected_candidate_feasible")
        for row in active_rows
    )
    selected_score = finite_values(
        parse_float(row, "tangent_escape_rmp_selected_candidate_score")
        for row in active_rows
    )
    selected_risk = finite_values(
        parse_float(row, "tangent_escape_rmp_selected_sector_risk")
        for row in active_rows
    )
    score_gap = finite_values(
        parse_float(row, "tangent_escape_rmp_selected_score_gap")
        for row in active_rows
    )
    max_candidate_normal_dot = finite_values(
        parse_float(row, "tangent_escape_rmp_max_abs_candidate_normal_dot_tangent")
        for row in active_rows
    )
    print("Canonical Candidate Score:")
    print("  candidate_count: " + fmt_range(candidate_count))
    print("  selected_feasible: " + fmt_range(selected_feasible))
    print("  selected_score: " + fmt_range(selected_score))
    print("  selected_sector_risk: " + fmt_range(selected_risk))
    print("  selected_score_gap: " + fmt_range(score_gap))
    print(
        "  max_abs_candidate_normal_dot_tangent: " +
        fmt_range(max_candidate_normal_dot)
    )
    print(
        "  sector_risk_nonzero_rows: " +
        str(sum(1 for value in selected_risk if abs(value) > 1e-9))
    )
    print("")

    dual_column_present = "tangent_escape_dual_solve_active" in fieldnames
    dual_rows = [
        row
        for row in active_rows
        if parse_float(row, "tangent_escape_dual_solve_active") >= 0.5
    ]
    dual_drive_rows = [
        row
        for row in dual_rows
        if int(round(parse_float(row, "tangent_escape_rmp_handoff_phase_id"))) in (1, 2)
    ]
    dual_no_effect = False
    print("Same-State Collision-only vs Collision+Escape:")
    if not dual_column_present or not dual_rows:
        print("  data: unavailable")
        print(
            "  record with publish_tangent_escape_dual_solve_data:=true "
            "to measure the post-limit Escape contribution"
        )
    else:
        effect_rows = dual_drive_rows or dual_rows
        delta_qdd = finite_values(
            parse_float(row, "tangent_escape_delta_qdd_norm")
            for row in effect_rows
        )
        delta_cp = finite_values(
            parse_float(row, "tangent_escape_delta_cp_accel_norm_m_s2")
            for row in effect_rows
        )
        delta_cp_tangent = finite_values(
            parse_float(row, "tangent_escape_delta_cp_accel_dot_tangent_m_s2")
            for row in effect_rows
        )
        delta_cp_normal = finite_values(
            parse_float(row, "tangent_escape_delta_cp_accel_dot_normal_m_s2")
            for row in effect_rows
        )
        final_qdd_direction_change = finite_values(
            vector_angle_degrees(
                dual_qdd_vector(row, "with"),
                dual_qdd_vector(row, "without"),
                0.1,
            )
            for row in effect_rows
        )
        final_cp_direction_change = finite_values(
            vector_angle_degrees(
                dual_cp_accel_vector(row, "with"),
                dual_cp_accel_vector(row, "without"),
                0.01,
            )
            for row in effect_rows
        )
        relative_qdd = finite_values(
            parse_float(row, "tangent_escape_delta_qdd_norm") /
            max(parse_float(row, "tangent_escape_qdd_without_norm"), 1e-9)
            for row in effect_rows
            if math.isfinite(parse_float(row, "tangent_escape_delta_qdd_norm")) and
            math.isfinite(parse_float(row, "tangent_escape_qdd_without_norm"))
        )
        unchanged_rows = sum(
            value <= args.dual_effect_epsilon
            for value in delta_qdd
        )
        positive_tangent_rows = sum(value > 0.0 for value in delta_cp_tangent)
        print(f"  active rows: {len(dual_rows)} (drive={len(dual_drive_rows)})")
        print("  delta_qdd_norm_rad_s2: " + fmt_range(delta_qdd))
        print("  relative_qdd_change: " + fmt_range(relative_qdd))
        print("  delta_cp_accel_norm_m_s2: " + fmt_range(delta_cp))
        print("  delta_cp_accel_dot_tangent_m_s2: " + fmt_range(delta_cp_tangent))
        print("  delta_cp_accel_dot_normal_m_s2: " + fmt_range(delta_cp_normal))
        if final_qdd_direction_change:
            print(
                "  final_qdd_direction_change_deg: "
                f"median={fmt(statistics.median(final_qdd_direction_change))} "
                f"p95={fmt(percentile(final_qdd_direction_change, 0.95))} "
                f"max={fmt(max(final_qdd_direction_change))} "
                f"ge30={sum(value >= 30.0 for value in final_qdd_direction_change)}/"
                f"{len(final_qdd_direction_change)}"
            )
        if final_cp_direction_change:
            print(
                "  final_cp_accel_direction_change_deg: "
                f"median={fmt(statistics.median(final_cp_direction_change))} "
                f"p95={fmt(percentile(final_cp_direction_change, 0.95))} "
                f"max={fmt(max(final_cp_direction_change))} "
                f"ge30={sum(value >= 30.0 for value in final_cp_direction_change)}/"
                f"{len(final_cp_direction_change)} "
                f"ge90={sum(value >= 90.0 for value in final_cp_direction_change)}/"
                f"{len(final_cp_direction_change)}"
            )
        if delta_qdd:
            print(
                "  post-limit unchanged rows: "
                f"{unchanged_rows}/{len(delta_qdd)}"
            )
            print(
                "  median relative qdd change: "
                f"{fmt(statistics.median(relative_qdd))}"
                if relative_qdd else
                "  median relative qdd change: n/a"
            )
        if delta_cp_tangent:
            print(
                "  positive tangent contribution rows: "
                f"{positive_tangent_rows}/{len(delta_cp_tangent)}"
            )
        dual_no_effect = bool(delta_qdd) and unchanged_rows == len(delta_qdd)
    print("")

    state_modes = finite_values(
        parse_float(row, "tangent_escape_rmp_state_mode_id")
        for row in active_rows
    )
    handoff_phases = finite_values(
        parse_float(row, "tangent_escape_rmp_handoff_phase_id")
        for row in active_rows
    )
    handoff_reasons = finite_values(
        parse_float(row, "tangent_escape_rmp_handoff_reason_id")
        for row in active_rows
    )
    blocked_penalty = finite_values(
        parse_float(row, "tangent_escape_rmp_selected_blocked_penalty")
        for row in active_rows
    )
    blocked_memory = finite_values(
        parse_float(row, "tangent_escape_rmp_max_blocked_memory")
        for row in active_rows
    )
    print("Canonical State Machine:")
    print("  states: " + count_summary(state_modes))
    print("  handoff_phases: " + count_summary(handoff_phases))
    print("  handoff_reasons: " + count_summary(handoff_reasons))
    print("  selected_blocked_penalty: " + fmt_range(blocked_penalty))
    print("  max_blocked_memory: " + fmt_range(blocked_memory))

    handoff_changes = 0
    handoff_violations: List[int] = []
    handoff_unobservable: List[int] = []
    for previous_index, current_index in zip(active_indices, active_indices[1:]):
        if current_index != previous_index + 1:
            continue
        previous = rows[previous_index]
        current = rows[current_index]
        previous_cp = parse_float(
            previous,
            "tangent_escape_rmp_control_point_index",
        )
        current_cp = parse_float(
            current,
            "tangent_escape_rmp_control_point_index",
        )
        previous_tangent = [
            parse_float(previous, f"tangent_escape_rmp_tangent_{axis}")
            for axis in ("x", "y", "z")
        ]
        current_tangent = [
            parse_float(current, f"tangent_escape_rmp_tangent_{axis}")
            for axis in ("x", "y", "z")
        ]
        cp_changed = (
            math.isfinite(previous_cp) and
            math.isfinite(current_cp) and
            int(round(previous_cp)) != int(round(current_cp))
        )
        tangent_delta = norm([
            current_tangent[axis] - previous_tangent[axis]
            for axis in range(3)
        ])
        tangent_changed = (
            math.isfinite(tangent_delta) and
            tangent_delta > args.handoff_tangent_change
        )
        if not cp_changed and not tangent_changed:
            continue
        handoff_changes += 1
        previous_time = row_time(previous)
        current_time = row_time(current)
        if (
            not math.isfinite(previous_time) or
            not math.isfinite(current_time) or
            current_time - previous_time <= 0.0 or
            current_time - previous_time > args.handoff_fresh_dt
        ):
            handoff_unobservable.append(current_index)
            continue
        # The controller completes lambda-down, sets the old branch to zero,
        # swaps the CP/tangent, and evaluates the new branch in one solve
        # cycle.  Consequently, the row that first contains the new branch
        # must have zero effect; the immediately preceding old-branch row may
        # still have a small nonzero ramp value.
        current_lambda = parse_float(
            current,
            "tangent_escape_rmp_lambda",
        )
        current_metric = parse_float(
            current,
            "tangent_escape_rmp_effective_metric_scalar",
        )
        current_zero_effect = (
            math.isfinite(current_lambda) and
            math.isfinite(current_metric) and
            abs(current_lambda) <= args.handoff_lambda_tolerance and
            abs(current_metric) <= args.handoff_metric_tolerance
        )
        if current_zero_effect:
            continue

        # A logger running no faster than the controller can miss the exact
        # switch cycle and first observe the new branch after its ENGAGE ramp
        # has begun.  Without a controller-cycle sequence/switch flag this is
        # unobservable, not evidence that the controller switched at nonzero
        # effect.
        current_phase = parse_float(
            current,
            "tangent_escape_rmp_handoff_phase_id",
        )
        if math.isfinite(current_phase) and int(round(current_phase)) == 1:
            handoff_unobservable.append(current_index)
        else:
            handoff_violations.append(current_index)
    print(f"  detected CP/tangent changes: {handoff_changes}")
    print(f"  zero-effect handoff violations: {len(handoff_violations)}")
    print(
        "  zero-effect handoffs unobservable at logger rate: "
        f"{len(handoff_unobservable)}"
    )
    print("")

    qdd_norms = finite_values(qdd_norm(row) for row in active_rows)
    all_qdd_norms = finite_values(qdd_norm(row) for row in rows)
    saturation_threshold = args.qdd_limit * args.saturation_ratio
    saturation_rows: List[Tuple[int, float, List[int]]] = []
    for index, row in enumerate(rows):
        values = qdd_vector(row)
        if not all(math.isfinite(value) for value in values):
            continue
        joints = [
            joint + 1
            for joint, value in enumerate(values)
            if abs(value) >= saturation_threshold
        ]
        if joints:
            saturation_rows.append(
                (index, max(abs(value) for value in values), joints)
            )

    jerk, jerk_index = max_jerk(rows)
    active_index_set = set(active_indices)
    active_saturation = [
        item
        for item in saturation_rows
        if item[0] in active_index_set
    ]
    print("Closed Loop Motion:")
    print("  active qdd_norm: " + fmt_range(qdd_norms))
    print("  all qdd_norm: " + fmt_range(all_qdd_norms))
    print(
        f"  qdd saturation rows: {len(saturation_rows)} "
        f"(active={len(active_saturation)}, "
        f"threshold={fmt(saturation_threshold)} rad/s^2)"
    )
    if saturation_rows:
        first_index, max_abs, joints = saturation_rows[0]
        print(
            f"  first saturation: row={first_index} "
            f"t_rel={fmt(rel_time(first_index))} "
            f"max_abs={fmt(max_abs)} joints={joints}"
        )
    print(f"  max qdd jerk estimate: {fmt(jerk)} rad/s^3")
    if jerk_index is not None:
        print(f"  max jerk row pair starts at: {jerk_index}")

    first_start = intervals[0][0]
    last_end = intervals[-1][1]
    goal_before = nearest_finite_goal_error(
        rows,
        max(first_start - 1, 0),
        -1,
    )
    goal_start = nearest_finite_goal_error(rows, first_start, 1)
    goal_end = nearest_finite_goal_error(rows, last_end, -1)
    goal_after = nearest_finite_goal_error(
        rows,
        min(last_end + 1, len(rows) - 1),
        1,
    )
    goal_final = nearest_finite_goal_error(rows, len(rows) - 1, -1)
    print(f"  goal_error_before_active_m: {fmt(goal_before)}")
    print(f"  goal_error_start_m: {fmt(goal_start)}")
    print(f"  goal_error_end_m: {fmt(goal_end)}")
    print(f"  goal_error_after_active_m: {fmt(goal_after)}")
    print(f"  goal_error_final_m: {fmt(goal_final)}")
    released_before_end = last_end < len(rows) - 1
    print(f"  escape_released_before_log_end: {released_before_end}")
    print("")

    fake_summaries = fake_range_summary(rows, fieldnames)
    if fake_summaries:
        print("Fake Sensor Input:")
        for summary in fake_summaries:
            print("  " + summary)
        print("")

    warnings: List[str] = []
    if not dual_column_present or not dual_rows:
        warnings.append("same-state Collision-only versus Escape A/B data was not recorded")
    elif dual_no_effect:
        warnings.append("Escape produced no measurable post-limit joint-acceleration change")
    if (
        drive_normal_dot_tangent and
        max(abs(value) for value in drive_normal_dot_tangent) >
        args.normal_tolerance_warn
    ):
        warnings.append("tangent direction is not close to the current normal plane")
    velocity_errors = finite_values(
        parse_float(row, "tangent_escape_rmp_velocity_error_m_s")
        for row in active_rows
    )
    if (
        velocity_errors and
        max(abs(value) for value in velocity_errors) > args.velocity_error_warn
    ):
        warnings.append("large tangent velocity tracking error")
    if selected_feasible and any(value < 0.5 for value in selected_feasible):
        warnings.append("an active record reported an infeasible selected candidate")
    if handoff_violations:
        warnings.append("CP/tangent changed before the Escape metric reached zero")
    if active_saturation:
        warnings.append("joint acceleration saturation occurred while escape was active")
    if math.isfinite(jerk) and jerk > args.jerk_warn:
        warnings.append("large qdd jerk estimate")
    if not released_before_end:
        warnings.append("escape was still active at end of log")

    print("Result:")
    print("  STRUCTURE_OK: yes")
    if warnings:
        print("  CLOSED_LOOP_WARNINGS:")
        for warning in warnings:
            print("    - " + warning)
    else:
        print("  CLOSED_LOOP_WARNINGS: none")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check canonical tangent-escape closed-loop behavior from an "
            "rmpflow trace CSV."
        )
    )
    parser.add_argument(
        "csv",
        nargs="?",
        type=Path,
        help="Path to rmpflow_trace CSV. If omitted, the latest compatible CSV is used.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help="Directory used when no CSV path is provided.",
    )
    parser.add_argument(
        "--qdd-limit",
        type=float,
        default=10.0,
        help="Joint acceleration limit used for saturation reporting.",
    )
    parser.add_argument(
        "--saturation-ratio",
        type=float,
        default=0.95,
        help="Fraction of qdd-limit counted as saturation.",
    )
    parser.add_argument(
        "--jerk-warn",
        type=float,
        default=200.0,
        help="Warn when estimated qdd jerk norm exceeds this value.",
    )
    parser.add_argument(
        "--normal-tolerance-warn",
        type=float,
        default=0.20,
        help="Warn when |normal dot tangent| exceeds this value.",
    )
    parser.add_argument(
        "--velocity-error-warn",
        type=float,
        default=0.10,
        help="Warn when absolute tangent velocity error exceeds this value in m/s.",
    )
    parser.add_argument(
        "--dual-effect-epsilon",
        type=float,
        default=1e-6,
        help=(
            "Post-limit delta-qdd norm at or below this value is counted as "
            "unchanged in the same-state A/B check."
        ),
    )
    parser.add_argument(
        "--handoff-tangent-change",
        type=float,
        default=1e-3,
        help="Direction-vector change counted as a tangent handoff.",
    )
    parser.add_argument(
        "--handoff-lambda-tolerance",
        type=float,
        default=1e-3,
        help="Maximum lambda allowed across a CP/tangent replacement.",
    )
    parser.add_argument(
        "--handoff-metric-tolerance",
        type=float,
        default=1e-6,
        help="Maximum effective metric allowed across a CP/tangent replacement.",
    )
    parser.add_argument(
        "--handoff-fresh-dt",
        type=float,
        default=0.1,
        help=(
            "Maximum interval for judging a sampled handoff; slower samples are "
            "reported as unobservable (default: 0.1 s)."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = args.csv.expanduser() if args.csv else latest_compatible_csv(args.log_dir.expanduser())
    rows, fieldnames = load_rows(path)
    return make_report(rows, fieldnames, path, args)


if __name__ == "__main__":
    raise SystemExit(main())
