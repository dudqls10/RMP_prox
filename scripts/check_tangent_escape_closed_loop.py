#!/usr/bin/env python3
import argparse
import csv
import math
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
    "tangent_escape_rmp_leaf_mode_id",
    "tangent_escape_rmp_activation",
    "tangent_escape_rmp_clearance_m",
    "tangent_escape_rmp_scalar_s_m",
    "tangent_escape_rmp_scalar_target_m",
    "tangent_escape_rmp_scalar_error_m",
    "tangent_escape_rmp_desired_tangent_accel_m_s2",
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


def is_finite(value: float) -> bool:
    return math.isfinite(value)


def finite_values(values: Iterable[float]) -> List[float]:
    return [value for value in values if math.isfinite(value)]


def norm(values: Sequence[float]) -> float:
    if any(not math.isfinite(value) for value in values):
        return float("nan")
    return math.sqrt(sum(value * value for value in values))


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


def make_report(rows: List[Dict[str, str]], fieldnames: List[str], path: Path, args: argparse.Namespace) -> int:
    if not rows:
        print(f"Input CSV: {path}")
        print("Error: CSV has no rows")
        return 1

    intervals = active_intervals(rows)
    active_indices = [index for start, end in intervals for index in range(start, end + 1)]
    active_rows = [rows[index] for index in active_indices]
    start_time = row_time(rows[0])
    end_time = row_time(rows[-1])
    duration = end_time - start_time if math.isfinite(start_time) and math.isfinite(end_time) else float("nan")

    def rel_time(index: int) -> float:
        value = row_time(rows[index])
        if math.isfinite(value) and math.isfinite(start_time):
            return value - start_time
        return float("nan")

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
    for row in active_rows:
        cp = parse_float(row, "tangent_escape_rmp_control_point_index")
        mode = parse_float(row, "tangent_escape_rmp_leaf_mode_id")
        if math.isfinite(cp):
            cp_counts[int(round(cp))] += 1
        if math.isfinite(mode):
            mode_counts[int(round(mode))] += 1

    total_active_duration = sum(interval_duration(rows, start, end) for start, end in intervals)
    print("Tangent Escape RMP:")
    print(f"  active rows: {len(active_rows)}")
    print(f"  active intervals: {len(intervals)}")
    print(f"  active duration: {fmt(total_active_duration)} s")
    for idx, (start, end) in enumerate(intervals, start=1):
        print(
            f"  interval {idx}: rows {start}-{end}, "
            f"t_rel={fmt(rel_time(start))}-{fmt(rel_time(end))}, "
            f"duration={fmt(interval_duration(rows, start, end))} s"
        )
    print(
        "  control points: " +
        ", ".join(f"{index}({sensor_name(index)}):{count}" for index, count in sorted(cp_counts.items()))
    )
    print("  mode ids: " + ", ".join(f"{mode}:{count}" for mode, count in sorted(mode_counts.items())))
    print("")

    print("GDS Scalar:")
    print("  activation: " + fmt_range([parse_float(row, "tangent_escape_rmp_activation") for row in active_rows]))
    print("  clearance_m: " + fmt_range([parse_float(row, "tangent_escape_rmp_clearance_m") for row in active_rows]))
    print("  beta: " + fmt_range([parse_float(row, "tangent_escape_rmp_beta") for row in active_rows]))
    scalar_s = [parse_float(row, "tangent_escape_rmp_scalar_s_m") for row in active_rows]
    scalar_target = [parse_float(row, "tangent_escape_rmp_scalar_target_m") for row in active_rows]
    scalar_error = [parse_float(row, "tangent_escape_rmp_scalar_error_m") for row in active_rows]
    desired_accel = [
        parse_float(row, "tangent_escape_rmp_desired_tangent_accel_m_s2")
        for row in active_rows
    ]
    print("  scalar_s_m: " + fmt_range(scalar_s))
    print("  scalar_target_m: " + fmt_range(scalar_target))
    print("  scalar_error_m: " + fmt_range(scalar_error))
    print("  desired_tangent_accel_m_s2: " + fmt_range(desired_accel))
    target_values = finite_values(scalar_target)
    s_values = finite_values(scalar_s)
    target_reached = bool(target_values and s_values and max(s_values) >= 0.8 * target_values[0])
    target_crossed = bool(finite_values(scalar_error) and min(finite_values(scalar_error)) <= 0.0)
    print(f"  reached_80pct_target: {target_reached}")
    print(f"  crossed_target: {target_crossed}")

    normal_dot_tangent = [
        vector_dot(row, "tangent_escape_rmp_normal", "tangent_escape_rmp_tangent")
        for row in active_rows
    ]
    normal_dot_tangent = finite_values(normal_dot_tangent)
    if normal_dot_tangent:
        print(f"  max_abs_normal_dot_tangent: {fmt(max(abs(value) for value in normal_dot_tangent))}")
    logged_normal_dot = finite_values(
        parse_float(row, "tangent_escape_rmp_normal_dot_tangent") for row in active_rows
    )
    if logged_normal_dot:
        print(
            "  logged_normal_dot_tangent: " +
            fmt_range(logged_normal_dot)
        )
    print("")

    candidate_count = finite_values(
        parse_float(row, "tangent_escape_rmp_candidate_count") for row in active_rows
    )
    selected_weight = finite_values(
        parse_float(row, "tangent_escape_rmp_selected_candidate_weight") for row in active_rows
    )
    selected_score = finite_values(
        parse_float(row, "tangent_escape_rmp_selected_candidate_score") for row in active_rows
    )
    selected_adjacent_risk = finite_values(
        parse_float(row, "tangent_escape_rmp_selected_adjacent_risk") for row in active_rows
    )
    score_gap = finite_values(
        parse_float(row, "tangent_escape_rmp_selected_score_gap") for row in active_rows
    )
    weight_gap = finite_values(
        parse_float(row, "tangent_escape_rmp_selected_weight_gap") for row in active_rows
    )
    max_candidate_normal_dot = finite_values(
        parse_float(row, "tangent_escape_rmp_max_abs_candidate_normal_dot_tangent")
        for row in active_rows
    )
    if candidate_count:
        print("Stage-3 Score/Softmax:")
        print("  candidate_count: " + fmt_range(candidate_count))
        print("  selected_weight: " + fmt_range(selected_weight))
        print("  selected_score: " + fmt_range(selected_score))
        print("  selected_adjacent_risk: " + fmt_range(selected_adjacent_risk))
        if score_gap:
            print("  selected_score_gap: " + fmt_range(score_gap))
        if weight_gap:
            print("  selected_weight_gap: " + fmt_range(weight_gap))
        if max_candidate_normal_dot:
            print(
                "  max_abs_candidate_normal_dot_tangent: " +
                fmt_range(max_candidate_normal_dot)
            )
        print(
            "  adjacent_risk_nonzero_rows: " +
            str(sum(1 for value in selected_adjacent_risk if abs(value) > 1e-9))
        )
        print("")

    supervisor_modes = finite_values(
        parse_float(row, "tangent_escape_rmp_supervisor_mode_id") for row in active_rows
    )
    if supervisor_modes:
        supervisor_mode_counts = Counter(int(round(value)) for value in supervisor_modes)
        hold_active = finite_values(
            parse_float(row, "tangent_escape_rmp_hold_active") for row in active_rows
        )
        hold_bonus = finite_values(
            parse_float(row, "tangent_escape_rmp_selected_hold_bonus") for row in active_rows
        )
        stuck_active = finite_values(
            parse_float(row, "tangent_escape_rmp_stuck_active") for row in active_rows
        )
        stuck_timer = finite_values(
            parse_float(row, "tangent_escape_rmp_stuck_timer_s") for row in active_rows
        )
        metric_boost = finite_values(
            parse_float(row, "tangent_escape_rmp_metric_boost") for row in active_rows
        )
        accel_boost = finite_values(
            parse_float(row, "tangent_escape_rmp_accel_boost") for row in active_rows
        )
        blocked_penalty = finite_values(
            parse_float(row, "tangent_escape_rmp_selected_blocked_penalty")
            for row in active_rows
        )
        blocked_memory = finite_values(
            parse_float(row, "tangent_escape_rmp_max_blocked_memory") for row in active_rows
        )
        branch_age = finite_values(
            parse_float(row, "tangent_escape_rmp_branch_age_s") for row in active_rows
        )
        branch_progress = finite_values(
            parse_float(row, "tangent_escape_rmp_branch_progress_m") for row in active_rows
        )
        print("Stage-4 Supervisor:")
        print(
            "  modes: " +
            ", ".join(
                f"{mode}:{count}" for mode, count in sorted(supervisor_mode_counts.items())
            )
        )
        print("  hold_active_rows: " + str(sum(1 for value in hold_active if value >= 0.5)))
        print("  selected_hold_bonus: " + fmt_range(hold_bonus))
        print("  stuck_active_rows: " + str(sum(1 for value in stuck_active if value >= 0.5)))
        print("  stuck_timer_s: " + fmt_range(stuck_timer))
        print("  metric_boost: " + fmt_range(metric_boost))
        print("  accel_boost: " + fmt_range(accel_boost))
        print("  selected_blocked_penalty: " + fmt_range(blocked_penalty))
        print("  max_blocked_memory: " + fmt_range(blocked_memory))
        print("  branch_age_s: " + fmt_range(branch_age))
        print("  branch_progress_m: " + fmt_range(branch_progress))
        print("")

    qdd_norms = [qdd_norm(row) for row in active_rows]
    qdd_norms = finite_values(qdd_norms)
    all_qdd_norms = finite_values(qdd_norm(row) for row in rows)
    saturation_threshold = args.qdd_limit * args.saturation_ratio
    saturation_rows: List[Tuple[int, float, List[int]]] = []
    for index, row in enumerate(rows):
        values = qdd_vector(row)
        if not all(math.isfinite(value) for value in values):
            continue
        joints = [
            joint + 1 for joint, value in enumerate(values)
            if abs(value) >= saturation_threshold
        ]
        if joints:
            saturation_rows.append((index, max(abs(value) for value in values), joints))

    jerk, jerk_index = max_jerk(rows)
    active_saturation = [item for item in saturation_rows if item[0] in active_indices]
    print("Closed Loop Motion:")
    print("  active qdd_norm: " + fmt_range(qdd_norms))
    print("  all qdd_norm: " + fmt_range(all_qdd_norms))
    print(
        f"  qdd saturation rows: {len(saturation_rows)} "
        f"(active={len(active_saturation)}, threshold={fmt(saturation_threshold)} rad/s^2)"
    )
    if saturation_rows:
        first_index, max_abs, joints = saturation_rows[0]
        print(
            f"  first saturation: row={first_index} t_rel={fmt(rel_time(first_index))} "
            f"max_abs={fmt(max_abs)} joints={joints}"
        )
    print(f"  max qdd jerk estimate: {fmt(jerk)} rad/s^3")
    if jerk_index is not None:
        print(f"  max jerk row pair starts at: {jerk_index}")

    first_start, first_end = intervals[0]
    last_start, last_end = intervals[-1]
    goal_before = nearest_finite_goal_error(rows, max(first_start - 1, 0), -1)
    goal_start = nearest_finite_goal_error(rows, first_start, 1)
    goal_end = nearest_finite_goal_error(rows, last_end, -1)
    goal_after = nearest_finite_goal_error(rows, min(last_end + 1, len(rows) - 1), 1)
    goal_final = nearest_finite_goal_error(rows, len(rows) - 1, -1)
    print(f"  goal_error_before_active_m: {fmt(goal_before)}")
    print(f"  goal_error_start_m: {fmt(goal_start)}")
    print(f"  goal_error_end_m: {fmt(goal_end)}")
    print(f"  goal_error_after_active_m: {fmt(goal_after)}")
    print(f"  goal_error_final_m: {fmt(goal_final)}")
    released_before_end = intervals[-1][1] < len(rows) - 1
    print(f"  escape_released_before_log_end: {released_before_end}")
    print("")

    fake_summaries = fake_range_summary(rows, fieldnames)
    if fake_summaries:
        print("Fake Sensor Input:")
        for summary in fake_summaries:
            print("  " + summary)
        print("")

    warnings: List[str] = []
    if not all(mode in {2, 3} for mode in mode_counts):
        warnings.append("mode_id is not GDS(2) or softmax GDS(3)")
    if not target_reached:
        warnings.append("scalar_s did not reach 80% of target")
    if normal_dot_tangent and max(abs(value) for value in normal_dot_tangent) > 0.05:
        warnings.append("tangent direction is not close to the current normal plane")
    if supervisor_modes and all(int(round(value)) == 0 for value in supervisor_modes):
        warnings.append("Stage-4 supervisor did not enter an active mode")
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
        description="Check closed-loop behavior of the Stage-2 tangent escape RMP from a trace CSV."
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = args.csv.expanduser() if args.csv else latest_compatible_csv(args.log_dir.expanduser())
    rows, fieldnames = load_rows(path)
    return make_report(rows, fieldnames, path, args)


if __name__ == "__main__":
    raise SystemExit(main())
